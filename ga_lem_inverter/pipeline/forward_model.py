# model_runner.py
import xsimlab as xs
import numpy as np
from fastscape.models import basic_model
from fastscape.processes.boundary import BorderBoundary
from fastscape.processes.context import FastscapelibContext
from fastscape.processes.flow import FlowAccumulator
from fastscape.processes.grid import UniformRectilinearGrid2D
from fastscape.processes.main import SurfaceTopography
import logging
import warnings
from dataclasses import dataclass

from ga_lem_inverter.pipeline.rainfall import (
    RainfallConfig,
    decode_rainfall_params,
    evaluate_rainfall,
    get_registered_rainfall_function,
    register_rainfall_function,
    encode_rainfall_params,
    validate_rainfall_array,
)


VALID_BOUNDARY_STATUS = {"fixed_value", "core", "looped"}


@dataclass(frozen=True)
class FastscapeSeriesResult:
    """FastScape topography history plus the uplift field used at each output frame."""

    topography_series: np.ndarray
    uplift_series: np.ndarray
    output_times: np.ndarray
    stage_edges_years: np.ndarray
    stage_multipliers: np.ndarray


@xs.process
class InitialDEM:
    """Initialize FastScape from an explicit DEM instead of random noise."""

    initial_elevation = xs.variable(dims=("y", "x"), description="initial DEM")
    shape = xs.foreign(UniformRectilinearGrid2D, "shape")
    elevation = xs.foreign(SurfaceTopography, "elevation", intent="out")

    def initialize(self):
        dem = np.asarray(self.initial_elevation, dtype=float)
        if tuple(dem.shape) != tuple(self.shape):
            raise ValueError(f"initial_elevation shape {dem.shape} 与 FastScape 网格 {tuple(self.shape)} 不一致。")
        if not np.isfinite(dem).all():
            raise ValueError("initial_elevation 不能包含 NaN/Inf。")
        self.elevation = dem.copy()


@xs.process
class TimeScaledUplift:
    """Block uplift with a time-dependent multiplier applied to one spatial field."""

    rate = xs.variable(dims=[(), ("y", "x")], description="base uplift rate")
    stage_edges_years = xs.variable(dims="stage_edge", static=True, description="elapsed stage boundaries in years")
    stage_multipliers = xs.variable(dims="stage", static=True, description="uplift multiplier for each stage")

    shape = xs.foreign(UniformRectilinearGrid2D, "shape")
    status = xs.foreign(BorderBoundary, "border_status")
    fs_context = xs.foreign(FastscapelibContext, "context")

    uplift = xs.variable(
        dims=[(), ("y", "x")],
        intent="out",
        groups=["bedrock_forcing_upward", "surface_forcing_upward"],
        description="imposed vertical uplift",
    )

    def initialize(self):
        self._stage_edges = np.asarray(self.stage_edges_years, dtype=float).reshape(-1)
        self._stage_multipliers = np.asarray(self.stage_multipliers, dtype=float).reshape(-1)
        validate_stage_history(self._stage_edges, self._stage_multipliers)

        self._mask = np.ones(self.shape)
        _all = slice(None)
        slices = [(_all, 0), (_all, -1), (0, _all), (-1, _all)]
        for status, border in zip(self.status, slices):
            if status == "fixed_value":
                self._mask[border] = 0.0

    @xs.runtime(args=("step_start", "step_delta"))
    def run_step(self, current_time, dt):
        stage_index = stage_index_for_elapsed_time(np.asarray(current_time, dtype=float).item(), self._stage_edges)
        multiplier = self._stage_multipliers[stage_index]
        rate = np.broadcast_to(self.rate, self.shape) * self._mask * multiplier
        self.uplift = rate * dt


@xs.process
class RainfallFlowAccumulator(FlowAccumulator):
    """Flow accumulator that evaluates a user rainfall function each step."""

    runoff = xs.variable(dims=("y", "x"), intent="out", description="dynamic rainfall/runoff field")
    rainfall_function_key = xs.variable(default="", static=True, description="registered rainfall function key")
    rainfall_params_json = xs.variable(default="{}", static=True, description="rainfall function parameters as JSON")
    rainfall_factor = xs.variable(default=1.0, description="uniform rainfall/runoff fallback")
    rainfall_min = xs.variable(default=np.nan, static=True, description="optional minimum runoff clamp")
    rainfall_max = xs.variable(default=np.nan, static=True, description="optional maximum runoff clamp")
    total_time = xs.variable(default=10.0e6, description="total model time in years")

    x = xs.foreign(UniformRectilinearGrid2D, "x")
    y = xs.foreign(UniformRectilinearGrid2D, "y")
    elevation = xs.foreign(SurfaceTopography, "elevation")

    @xs.runtime(args=("step_start",))
    def run_step(self, current_time):
        x_grid, y_grid = np.meshgrid(np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float))
        rainfall_function = get_registered_rainfall_function(str(self.rainfall_function_key))
        rainfall = RainfallConfig(
            mode="python" if rainfall_function is not None else "uniform",
            factor=float(self.rainfall_factor),
            function=rainfall_function,
            params=decode_rainfall_params(str(self.rainfall_params_json)),
            min_value=float(self.rainfall_min) if np.isfinite(float(self.rainfall_min)) else None,
            max_value=float(self.rainfall_max) if np.isfinite(float(self.rainfall_max)) else None,
        )
        self.runoff = evaluate_rainfall(
            rainfall,
            x=x_grid,
            y=y_grid,
            z=np.asarray(self.elevation, dtype=float),
            elapsed_years=float(np.asarray(current_time, dtype=float).item()),
            total_time_years=float(self.total_time),
        )
        super().run_step()


def validate_stage_history(stage_edges_years, stage_multipliers):
    """Validate elapsed stage boundaries and multipliers."""
    edges = np.asarray(stage_edges_years, dtype=float).reshape(-1)
    multipliers = np.asarray(stage_multipliers, dtype=float).reshape(-1)
    if edges.size < 2:
        raise ValueError("stage_edges_years 至少需要两个边界。")
    if multipliers.size != edges.size - 1:
        raise ValueError(
            f"stage_multipliers 数量必须等于 stage_edges_years-1: "
            f"{multipliers.size} != {edges.size - 1}"
        )
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(multipliers)):
        raise ValueError("stage_edges_years/stage_multipliers 不能包含 NaN/Inf。")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("stage_edges_years 必须严格递增，单位为从模拟开始算起的年。")
    if np.any(multipliers <= 0):
        raise ValueError("stage_multipliers 必须为正数。")


def stage_edges_from_ma(stage_times_ma, *, time_total_years, tolerance=1e-6):
    """Convert geological stage times before present into elapsed model years.

    Example: ``10, 6, 3, 0`` Ma with ``time_total=10e6`` becomes
    ``0, 4e6, 7e6, 10e6`` elapsed years for the forward model.
    """
    times = np.asarray(stage_times_ma, dtype=float).reshape(-1)
    if times.size < 2:
        raise ValueError("stage_times_ma 至少需要两个时间点，例如 10,6,3,0。")
    if not np.all(np.isfinite(times)):
        raise ValueError("stage_times_ma 不能包含 NaN/Inf。")
    if not np.all(np.diff(times) < 0):
        raise ValueError("stage_times_ma 必须按从过去到现在递减填写，例如 10,6,3,0。")
    if abs(float(times[-1])) > tolerance:
        raise ValueError("stage_times_ma 最后一个值必须是 0，表示现今。")
    total_ma = float(time_total_years) / 1.0e6
    if abs(float(times[0]) - total_ma) > max(tolerance, total_ma * 1e-6):
        raise ValueError(
            f"stage_times_ma 第一个值必须和 [Model] time_total 对齐: "
            f"{times[0]} Ma != {total_ma} Ma"
        )
    edges = (times[0] - times) * 1.0e6
    edges[0] = 0.0
    edges[-1] = float(time_total_years)
    return edges.astype(float)


def normalize_stage_multipliers(stage_multipliers, stage_edges_years, *, enabled=True):
    """Normalize multipliers so their time-weighted mean equals one."""
    multipliers = np.asarray(stage_multipliers, dtype=float).reshape(-1)
    edges = np.asarray(stage_edges_years, dtype=float).reshape(-1)
    validate_stage_history(edges, multipliers)
    if not enabled:
        return multipliers.copy()
    durations = np.diff(edges)
    weighted_mean = float(np.sum(multipliers * durations) / np.sum(durations))
    if not np.isfinite(weighted_mean) or weighted_mean <= 0:
        raise ValueError("stage_multipliers 时间加权均值必须为正。")
    return multipliers / weighted_mean


def stage_index_for_elapsed_time(elapsed_years, stage_edges_years):
    """Return the stage index for an elapsed model time in years."""
    edges = np.asarray(stage_edges_years, dtype=float).reshape(-1)
    value = float(elapsed_years)
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(edges[1:], value, side="right"))


def stage_multipliers_for_times(output_times, stage_edges_years, stage_multipliers):
    """Map elapsed output times to stage multipliers."""
    times = np.asarray(output_times, dtype=float).reshape(-1)
    multipliers = np.asarray(stage_multipliers, dtype=float).reshape(-1)
    edges = np.asarray(stage_edges_years, dtype=float).reshape(-1)
    validate_stage_history(edges, multipliers)
    return np.asarray([multipliers[stage_index_for_elapsed_time(t, edges)] for t in times], dtype=float)


def normalize_boundary_status(status):
    """Return FastScape boundary status as scalar or [left, right, top, bottom]."""
    if status is None:
        return "fixed_value"
    if isinstance(status, str):
        text = status.strip()
        if not text:
            return "fixed_value"
        parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        if len(parts) == 1:
            value = parts[0]
            if value not in VALID_BOUNDARY_STATUS:
                raise ValueError(f"无效 boundary_status={value!r}，可选: {sorted(VALID_BOUNDARY_STATUS)}")
            return value
        status_list = parts
    else:
        status_list = list(status)

    if len(status_list) != 4:
        raise ValueError("boundary_status 四边写法必须是 left,right,top,bottom 四个值。")
    invalid = [value for value in status_list if value not in VALID_BOUNDARY_STATUS]
    if invalid:
        raise ValueError(f"无效边界状态 {invalid}，可选: {sorted(VALID_BOUNDARY_STATUS)}")
    if "fixed_value" not in status_list:
        raise ValueError("FastScape 至少需要一条 fixed_value 边界作为数值约束。")
    return status_list


def boundary_status_from_config(config, section="Model"):
    """Read scalar or per-edge FastScape boundary status from config."""
    edge_keys = ("boundary_left", "boundary_right", "boundary_top", "boundary_bottom")
    if all(config.has_option(section, key) for key in edge_keys):
        return normalize_boundary_status([config.get(section, key).strip() for key in edge_keys])
    return normalize_boundary_status(config.get(section, "boundary_status", fallback="fixed_value"))


def align_model_field(field, target_shape, *, label, order=1, fill_value=None):
    """Align a 2D model field to the DEM/FastScape grid shape."""
    array = np.asarray(field, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{label} 必须是二维数组，当前 ndim={array.ndim}")
    target_shape = tuple(int(value) for value in target_shape)
    if len(target_shape) != 2 or min(target_shape) < 1:
        raise ValueError(f"目标网格 shape 无效: {target_shape}")
    if array.shape == target_shape:
        return array.copy()

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        if fill_value is None:
            raise ValueError(f"{label} 没有有效数值，无法自动对齐。")
        array = np.full(array.shape, float(fill_value), dtype=float)
        finite = array.reshape(-1)
    else:
        median = float(np.nanmedian(finite))
        array = np.where(np.isfinite(array), array, median)

    from skimage.transform import resize

    aligned = resize(
        array,
        target_shape,
        order=order,
        mode="edge",
        anti_aliasing=order > 0,
        preserve_range=True,
    )
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    aligned = np.clip(aligned, min_value, max_value)
    logging.warning("%s shape %s 已自动对齐到 %s。", label, array.shape, target_shape)
    return aligned.astype(float)


def align_fastscape_inputs(k_sp, uplift, *, x_size, y_size):
    """Align Ksp and uplift fields to the FastScape grid."""
    target_shape = (int(y_size), int(x_size))
    k_sp_aligned = align_model_field(k_sp, target_shape, label="Ksp", order=1)
    uplift_aligned = align_model_field(uplift, target_shape, label="uplift", order=1)
    return k_sp_aligned, uplift_aligned


def _fastscape_times(time_total, output_steps):
    output_steps = max(2, int(output_steps))
    master_times = np.linspace(0, time_total, 101)
    output_steps = min(output_steps, len(master_times) - 1)
    output_indices = np.linspace(1, len(master_times) - 1, output_steps, dtype=int)
    out_times = master_times[output_indices]
    return master_times, out_times


def fastscape_output_times(time_total, output_steps):
    """Return elapsed FastScape output times in years for plot labels and QA."""
    return _fastscape_times(time_total, output_steps)[1]


def validate_rainfall_factor(rainfall_factor):
    """Return a positive scalar runoff/rainfall factor for FastScape."""
    return float(validate_rainfall_array(rainfall_factor, label="rainfall_factor"))


def _model_and_rainfall_input(rainfall, *, time_total):
    """Use FastScape's official runoff input or a rainfall-aware flow accumulator."""
    if isinstance(rainfall, RainfallConfig):
        if rainfall.mode == "python" and rainfall.function is not None:
            function_key = register_rainfall_function(rainfall.function)
            return basic_model.update_processes({"drainage": RainfallFlowAccumulator}), {
                "drainage__rainfall_function_key": function_key,
                "drainage__rainfall_params_json": encode_rainfall_params(rainfall.params),
                "drainage__rainfall_factor": rainfall.factor,
                "drainage__rainfall_min": np.nan if rainfall.min_value is None else float(rainfall.min_value),
                "drainage__rainfall_max": np.nan if rainfall.max_value is None else float(rainfall.max_value),
                "drainage__total_time": float(time_total),
            }
        value = validate_rainfall_factor(rainfall.factor)
    else:
        value = validate_rainfall_factor(rainfall)
    if np.isclose(value, 1.0):
        return basic_model, {}
    return basic_model.update_processes({"drainage": FlowAccumulator}), {"drainage__runoff": value}


def run_fastscape_series(
    k_sp,
    uplift,
    k_diff,
    x_size,
    y_size,
    spacing,
    boundary_status='fixed_value',
    area_exp=0.43,
    slope_exp=1,
    time_total=10e6,
    rainfall_factor=1.0,
    rainfall_model=None,
    initial_topography_seed=42,
    output_steps=21,
):
    """
    运行 FastScape 模型并返回输出时间序列。

    参数:
    - k_sp: 侵蚀系数。
    - uplift: 抬升速率，单位 mm/yr。
    - k_diff: 扩散系数。
    - x_size: x 方向的网格大小。
    - y_size: y 方向的网格大小。
    - spacing: 网格间距。
    - boundary_status: 边界状态。
    - area_exp: 面积指数。
    - slope_exp: 坡度指数。
    - time_total: 总模拟时间。
    - rainfall_factor: FastScape FlowAccumulator.runoff，单位面积地表径流/降雨系数。
    - rainfall_model: 可选 RainfallConfig，支持用户 Python 函数 p=f(x,y,z,t)。
    - initial_topography_seed: 初始随机地形种子。固定种子可让 GA 目标函数可复现。
    - output_steps: 输出地形序列帧数。

    返回:
    - elevation_series: 模拟地形时间序列，shape=(output_steps, y_size, x_size)。
    """
    try:
        logging.info(f"Fastscape input shapes:")
        logging.info(f"k_sp shape: {k_sp.shape}")
        logging.info(f"uplift shape: {uplift.shape}")
        logging.info(f"Requested grid size: {y_size} x {x_size}")

        k_sp, uplift = align_fastscape_inputs(k_sp, uplift, x_size=x_size, y_size=y_size)
        boundary_status = normalize_boundary_status(boundary_status)

        # 在运行模型前添加以下代码
        warnings.filterwarnings("ignore", category=FutureWarning,
                            message="variable .* with name matching its dimension")
        # Pecube cannot reliably consume FastScape's raw t=0 random seed
        # topography as the oldest surface. Emit only evolved snapshots while
        # keeping the output clock aligned with the master clock.
        master_times, out_times = _fastscape_times(time_total, output_steps)
        rainfall = rainfall_model if rainfall_model is not None else rainfall_factor
        model, rainfall_input = _model_and_rainfall_input(rainfall, time_total=time_total)
        input_vars = {
            'grid__shape': [y_size, x_size],
            'grid__length': [y_size * spacing, x_size * spacing],
            'boundary__status': boundary_status,
            'uplift__rate': uplift * 10**(-3),
            'init_topography__seed': initial_topography_seed,
            'spl__k_coef': k_sp,
            'spl__area_exp': area_exp,
            'spl__slope_exp': slope_exp,
            'diffusion__diffusivity': k_diff * 10**(-2),
            **rainfall_input,
        }
        ds_in = xs.create_setup(
            model=model,
            clocks={'time': master_times, 'out': out_times},
            master_clock='time',
            input_vars=input_vars,
            output_vars={
                'topography__elevation': 'out'}
        )
        out_ds = (ds_in.xsimlab.run(model=model))
        return out_ds.topography__elevation.values
    except Exception as e:
        logging.error(f"运行 fastscape 模型出错: {e}")
        raise RuntimeError(f"运行 fastscape 模型出错: {e}")


def run_fastscape_time_scaled_series(
    k_sp,
    uplift,
    k_diff,
    x_size,
    y_size,
    spacing,
    *,
    stage_edges_years,
    stage_multipliers,
    initial_dem=None,
    boundary_status='fixed_value',
    area_exp=0.43,
    slope_exp=1,
    time_total=10e6,
    rainfall_factor=1.0,
    rainfall_model=None,
    initial_topography_seed=42,
    output_steps=21,
):
    """Run FastScape with ``U(x,y,t) = U_base(x,y) * multiplier(stage)``.

    Uplift input and returned ``uplift_series`` are in mm/yr. FastScape itself
    receives m/yr, matching the static runner.
    """
    try:
        logging.info("Fastscape time-scaled input shapes:")
        logging.info(f"k_sp shape: {k_sp.shape}")
        logging.info(f"uplift shape: {uplift.shape}")
        logging.info(f"Requested grid size: {y_size} x {x_size}")

        k_sp, uplift = align_fastscape_inputs(k_sp, uplift, x_size=x_size, y_size=y_size)
        boundary_status = normalize_boundary_status(boundary_status)
        stage_edges = np.asarray(stage_edges_years, dtype=float).reshape(-1)
        multipliers = np.asarray(stage_multipliers, dtype=float).reshape(-1)
        validate_stage_history(stage_edges, multipliers)
        if abs(stage_edges[0]) > 1e-6 or abs(stage_edges[-1] - float(time_total)) > max(1e-6, float(time_total) * 1e-9):
            raise ValueError("stage_edges_years 必须从 0 开始，并以 time_total 结束。")

        warnings.filterwarnings("ignore", category=FutureWarning,
                            message="variable .* with name matching its dimension")
        master_times, out_times = _fastscape_times(time_total, output_steps)
        rainfall = rainfall_model if rainfall_model is not None else rainfall_factor
        model, rainfall_input = _model_and_rainfall_input(rainfall, time_total=time_total)
        model = model.update_processes({"uplift": TimeScaledUplift})
        input_vars = {
            'grid__shape': [y_size, x_size],
            'grid__length': [y_size * spacing, x_size * spacing],
            'boundary__status': boundary_status,
            'uplift__rate': uplift * 10**(-3),
            'uplift__stage_edges_years': stage_edges,
            'uplift__stage_multipliers': multipliers,
            'spl__k_coef': k_sp,
            'spl__area_exp': area_exp,
            'spl__slope_exp': slope_exp,
            'diffusion__diffusivity': k_diff * 10**(-2),
            **rainfall_input,
        }
        if initial_dem is None:
            input_vars['init_topography__seed'] = initial_topography_seed
        else:
            initial = align_model_field(initial_dem, (int(y_size), int(x_size)), label="initial_dem", order=1)
            model = model.update_processes({"init_topography": InitialDEM})
            input_vars['init_topography__initial_elevation'] = initial

        ds_in = xs.create_setup(
            model=model,
            clocks={'time': master_times, 'out': out_times},
            master_clock='time',
            input_vars=input_vars,
            output_vars={'topography__elevation': 'out'},
        )
        out_ds = ds_in.xsimlab.run(model=model)
        topographies = out_ds.topography__elevation.values
        output_multipliers = stage_multipliers_for_times(out_times, stage_edges, multipliers)
        uplift_series = np.asarray([uplift * multiplier for multiplier in output_multipliers], dtype=float)
        return FastscapeSeriesResult(
            topography_series=topographies,
            uplift_series=uplift_series,
            output_times=out_times,
            stage_edges_years=stage_edges,
            stage_multipliers=multipliers,
        )
    except Exception as e:
        logging.error(f"运行 time-scaled fastscape 模型出错: {e}")
        raise RuntimeError(f"运行 time-scaled fastscape 模型出错: {e}")


def run_fastscape_model(
    k_sp,
    uplift,
    k_diff,
    x_size,
    y_size,
    spacing,
    boundary_status='fixed_value',
    area_exp=0.43,
    slope_exp=1,
    time_total=10e6,
    rainfall_factor=1.0,
    rainfall_model=None,
    initial_topography_seed=42
):
    """运行 FastScape 模型并返回最终地形。"""
    series = run_fastscape_series(
        k_sp=k_sp,
        uplift=uplift,
        k_diff=k_diff,
        x_size=x_size,
        y_size=y_size,
        spacing=spacing,
        boundary_status=boundary_status,
        area_exp=area_exp,
        slope_exp=slope_exp,
        time_total=time_total,
        rainfall_factor=rainfall_factor,
        rainfall_model=rainfall_model,
        initial_topography_seed=initial_topography_seed,
        output_steps=21,
    )
    return series[-1]
