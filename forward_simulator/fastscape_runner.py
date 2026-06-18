"""FastScape 调用层。

策略：
- mode = none / stages → 直接复用 ``ga_lem_inverter.pipeline.forward_model`` 中
  已经稳定的 ``run_fastscape_series`` / ``run_fastscape_time_scaled_series``。
- mode = python / array → 这里实现一个独立的 ``TimeFunctionUplift`` process，
  让用户用 Python 函数或 ``(T,Y,X)`` 数组直接控制 U(x,y,t)。
  这条路径的 FastScape 设置代码独立写一份，因为现有反演工具不需要它。

无论哪条路径，都返回统一的 ``ForwardSeriesResult``：
``topography_series`` ``uplift_series`` ``output_times``。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xsimlab as xs
from fastscape.models import basic_model
from fastscape.processes.boundary import BorderBoundary
from fastscape.processes.context import FastscapelibContext
from fastscape.processes.flow import FlowAccumulator
from fastscape.processes.grid import UniformRectilinearGrid2D
from fastscape.processes.main import SurfaceTopography

# 复用反演工具里已稳定的 FastScape 调用与 RainfallConfig
from ga_lem_inverter.pipeline.forward_model import (
    InitialDEM,
    align_fastscape_inputs,
    align_model_field,
    fastscape_output_times,
    normalize_boundary_status,
    run_fastscape_series,
    run_fastscape_time_scaled_series,
    stage_edges_from_ma,
    stage_multipliers_for_times,
    validate_stage_history,
)
from ga_lem_inverter.pipeline.rainfall import (
    RainfallConfig as GaRainfallConfig,
    decode_rainfall_params,
    encode_rainfall_params,
    evaluate_rainfall,
    get_registered_rainfall_function,
    load_rainfall_function,
    register_rainfall_function,
    validate_rainfall_array,
)

from forward_simulator.config import ForwardConfig, RainfallConfig, UpliftTimeConfig


logger = logging.getLogger(__name__)


# ---------- 自定义 process：python / array uplift ----------------------------

UpliftTimeFunction = Callable[..., Any]
_UPLIFT_FUNCTION_REGISTRY: dict[str, UpliftTimeFunction] = {}
_UPLIFT_ARRAY_REGISTRY: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _register_uplift_function(func: UpliftTimeFunction) -> str:
    key = f"uplift_fn_{uuid.uuid4().hex}"
    _UPLIFT_FUNCTION_REGISTRY[key] = func
    return key


def _register_uplift_array(times_years: np.ndarray, frames_mm_per_yr: np.ndarray) -> str:
    key = f"uplift_arr_{uuid.uuid4().hex}"
    _UPLIFT_ARRAY_REGISTRY[key] = (np.asarray(times_years, dtype=float), np.asarray(frames_mm_per_yr, dtype=float))
    return key


def _load_user_module(module_path: str | Path, function_name: str) -> UpliftTimeFunction:
    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到 uplift 时间函数脚本: {path}")
    if not path.is_file():
        raise ValueError(f"uplift 时间函数路径不是文件: {path}")
    spec = importlib.util.spec_from_file_location(f"forward_user_uplift_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 uplift 时间函数脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name, None)
    if not callable(func):
        raise AttributeError(f"{path} 中找不到可调用函数 {function_name!r}")
    return func


def _evaluate_uplift_callable(
    func: UpliftTimeFunction,
    *,
    base_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    t_yr: float,
    params: dict[str, Any],
) -> np.ndarray:
    """统一处理用户函数的两种返回值：scalar 或 (Y,X) 数组。"""
    raw = func(t_yr=float(t_yr), x=x_grid, y=y_grid, z=z, params=dict(params or {}))
    array = np.asarray(raw, dtype=float)
    if array.ndim == 0:
        value = float(array)
        if not np.isfinite(value):
            raise ValueError(f"uplift_time 返回的 scalar 不是有限数: {value}")
        return base_field * value
    if array.shape != base_field.shape:
        raise ValueError(
            f"uplift_time 返回的二维场 shape={array.shape} 与 (Y,X)={base_field.shape} 不一致"
        )
    if not np.isfinite(array).all():
        raise ValueError("uplift_time 返回的二维场包含 NaN/Inf。")
    return array


def _interpolate_uplift_array(
    times_years: np.ndarray,
    frames: np.ndarray,
    t_yr: float,
) -> np.ndarray:
    """对 (T,Y,X) 数组在时间维上做线性插值。"""
    t = float(t_yr)
    if t <= times_years[0]:
        return frames[0].astype(float)
    if t >= times_years[-1]:
        return frames[-1].astype(float)
    idx = int(np.searchsorted(times_years, t))
    t0, t1 = float(times_years[idx - 1]), float(times_years[idx])
    w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return ((1.0 - w) * frames[idx - 1] + w * frames[idx]).astype(float)


@xs.process
class TimeFunctionUplift:
    """U(x,y,t) 由用户 Python 函数或 (T,Y,X) 数组逐步给出（mm/yr → m/yr 内部换算）。"""

    base_rate = xs.variable(dims=("y", "x"), description="base uplift rate in mm/yr")
    uplift_function_key = xs.variable(default="", static=True)
    uplift_array_key = xs.variable(default="", static=True)
    uplift_params_json = xs.variable(default="{}", static=True)
    total_time = xs.variable(default=10.0e6)

    shape = xs.foreign(UniformRectilinearGrid2D, "shape")
    x = xs.foreign(UniformRectilinearGrid2D, "x")
    y = xs.foreign(UniformRectilinearGrid2D, "y")
    elevation = xs.foreign(SurfaceTopography, "elevation")
    status = xs.foreign(BorderBoundary, "border_status")
    fs_context = xs.foreign(FastscapelibContext, "context")

    uplift = xs.variable(
        dims=[(), ("y", "x")],
        intent="out",
        groups=["bedrock_forcing_upward", "surface_forcing_upward"],
        description="imposed vertical uplift",
    )

    def initialize(self):
        self._mask = np.ones(self.shape)
        _all = slice(None)
        slices = [(_all, 0), (_all, -1), (0, _all), (-1, _all)]
        for status, border in zip(self.status, slices):
            if status == "fixed_value":
                self._mask[border] = 0.0
        self._function = (
            _UPLIFT_FUNCTION_REGISTRY.get(str(self.uplift_function_key).strip())
            if str(self.uplift_function_key).strip()
            else None
        )
        self._array = (
            _UPLIFT_ARRAY_REGISTRY.get(str(self.uplift_array_key).strip())
            if str(self.uplift_array_key).strip()
            else None
        )
        self._params = json.loads(str(self.uplift_params_json) or "{}")
        self._x_grid, self._y_grid = np.meshgrid(np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float))

    @xs.runtime(args=("step_start", "step_delta"))
    def run_step(self, current_time, dt):
        t_yr = float(np.asarray(current_time, dtype=float).item())
        base_field = np.broadcast_to(self.base_rate, self.shape).astype(float)
        if self._function is not None:
            field = _evaluate_uplift_callable(
                self._function,
                base_field=base_field,
                x_grid=self._x_grid,
                y_grid=self._y_grid,
                z=np.asarray(self.elevation, dtype=float),
                t_yr=t_yr,
                params=self._params,
            )
        elif self._array is not None:
            times_years, frames = self._array
            field = _interpolate_uplift_array(times_years, frames, t_yr)
        else:
            field = base_field
        if not np.isfinite(field).all():
            raise ValueError("uplift 场包含 NaN/Inf。")
        # FastScape 内部使用 m/yr，这里把 mm/yr 换算回去；同时乘 mask 和 dt
        rate = field * 1.0e-3 * self._mask
        self.uplift = rate * dt


# ---------- 顶层结果对象 -----------------------------------------------------

@dataclass(frozen=True)
class ForwardSeriesResult:
    topography_series: np.ndarray  # (T, Y, X) m
    uplift_series: np.ndarray      # (T, Y, X) mm/yr
    output_times: np.ndarray       # (T,) elapsed years
    multipliers: np.ndarray | None  # 仅 stages 模式给出，其余 None


# ---------- 顶层入口 ---------------------------------------------------------

def run_forward(
    cfg: ForwardConfig,
    *,
    initial_dem: np.ndarray,
    uplift_base_field: np.ndarray,
    ksp_field: np.ndarray | float,
) -> ForwardSeriesResult:
    """根据 cfg.uplift_time.mode 选择正演路径。"""
    y_size, x_size = initial_dem.shape
    rainfall_model = _build_rainfall_config(cfg.rainfall, cfg.config_path.parent)

    if cfg.uplift_time.mode == "none":
        topographies = run_fastscape_series(
            k_sp=np.asarray(ksp_field, dtype=float) if not np.isscalar(ksp_field) else float(ksp_field),
            uplift=np.asarray(uplift_base_field, dtype=float),
            k_diff=cfg.k_diff,
            x_size=x_size,
            y_size=y_size,
            spacing=cfg.spacing,
            boundary_status=cfg.boundary,
            area_exp=cfg.area_exp,
            slope_exp=cfg.slope_exp,
            time_total=cfg.time_total,
            rainfall_model=rainfall_model,
            initial_dem=initial_dem,
            output_steps=cfg.output_steps,
        )
        out_times = fastscape_output_times(cfg.time_total, cfg.output_steps)
        uplift_series = np.broadcast_to(
            np.asarray(uplift_base_field, dtype=float), (len(out_times), y_size, x_size)
        ).copy()
        return ForwardSeriesResult(
            topography_series=topographies,
            uplift_series=uplift_series,
            output_times=out_times,
            multipliers=None,
        )

    if cfg.uplift_time.mode == "stages":
        stage_edges = stage_edges_from_ma(
            cfg.uplift_time.stage_times_ma, time_total_years=cfg.time_total
        )
        multipliers = np.asarray(cfg.uplift_time.stage_multipliers, dtype=float)
        validate_stage_history(stage_edges, multipliers)
        result = run_fastscape_time_scaled_series(
            k_sp=np.asarray(ksp_field, dtype=float) if not np.isscalar(ksp_field) else float(ksp_field),
            uplift=np.asarray(uplift_base_field, dtype=float),
            k_diff=cfg.k_diff,
            x_size=x_size,
            y_size=y_size,
            spacing=cfg.spacing,
            stage_edges_years=stage_edges,
            stage_multipliers=multipliers,
            initial_dem=initial_dem,
            boundary_status=cfg.boundary,
            area_exp=cfg.area_exp,
            slope_exp=cfg.slope_exp,
            time_total=cfg.time_total,
            rainfall_model=rainfall_model,
            output_steps=cfg.output_steps,
        )
        out_multipliers = stage_multipliers_for_times(result.output_times, stage_edges, multipliers)
        return ForwardSeriesResult(
            topography_series=result.topography_series,
            uplift_series=result.uplift_series,
            output_times=result.output_times,
            multipliers=out_multipliers,
        )

    if cfg.uplift_time.mode in {"python", "array"}:
        return _run_fastscape_time_function(
            cfg=cfg,
            initial_dem=initial_dem,
            uplift_base_field=uplift_base_field,
            ksp_field=ksp_field,
            rainfall_model=rainfall_model,
        )

    raise ValueError(f"未知 uplift 时间模式: {cfg.uplift_time.mode!r}")


def _run_fastscape_time_function(
    *,
    cfg: ForwardConfig,
    initial_dem: np.ndarray,
    uplift_base_field: np.ndarray,
    ksp_field: np.ndarray | float,
    rainfall_model: GaRainfallConfig | None,
) -> ForwardSeriesResult:
    """python / array 模式：构造一个独立的 FastScape setup。"""
    y_size, x_size = initial_dem.shape
    boundary = normalize_boundary_status(cfg.boundary)
    if np.isscalar(ksp_field):
        ksp_array = np.full((y_size, x_size), float(ksp_field), dtype=float)
    else:
        ksp_array = np.asarray(ksp_field, dtype=float)
    ksp_aligned, uplift_aligned = align_fastscape_inputs(
        ksp_array, uplift_base_field, x_size=x_size, y_size=y_size
    )

    warnings.filterwarnings(
        "ignore", category=FutureWarning, message="variable .* with name matching its dimension"
    )

    if cfg.uplift_time.mode == "python":
        func = _load_user_module(cfg.uplift_time.module_path, cfg.uplift_time.function_name)
        function_key = _register_uplift_function(func)
        array_key = ""
    else:
        frames, times_years = _load_uplift_array(cfg.uplift_time, target_shape=(y_size, x_size))
        array_key = _register_uplift_array(times_years, frames)
        function_key = ""

    # 构造 master / output clocks（参考反演工具的 _fastscape_times）
    output_steps = max(2, int(cfg.output_steps))
    master_times = np.linspace(0.0, float(cfg.time_total), 101)
    output_indices = np.linspace(1, len(master_times) - 1, output_steps, dtype=int)
    out_times = master_times[output_indices]

    # rainfall 注入：和反演工具保持一致
    if rainfall_model is not None and rainfall_model.mode == "python":
        from ga_lem_inverter.pipeline.forward_model import RainfallFlowAccumulator

        rainfall_function_key = register_rainfall_function(rainfall_model.function)
        rainfall_input = {
            "drainage__rainfall_function_key": rainfall_function_key,
            "drainage__rainfall_params_json": encode_rainfall_params(rainfall_model.params),
            "drainage__rainfall_factor": float(rainfall_model.factor),
            "drainage__rainfall_min": np.nan if rainfall_model.min_value is None else float(rainfall_model.min_value),
            "drainage__rainfall_max": np.nan if rainfall_model.max_value is None else float(rainfall_model.max_value),
            "drainage__total_time": float(cfg.time_total),
        }
        model = basic_model.update_processes({"drainage": RainfallFlowAccumulator})
    else:
        runoff_value = float(rainfall_model.factor) if rainfall_model is not None else 1.0
        if not np.isclose(runoff_value, 1.0):
            model = basic_model.update_processes({"drainage": FlowAccumulator})
            rainfall_input = {"drainage__runoff": runoff_value}
        else:
            model = basic_model
            rainfall_input = {}

    model = model.update_processes({"uplift": TimeFunctionUplift, "init_topography": InitialDEM})

    input_vars = {
        "grid__shape": [y_size, x_size],
        "grid__length": [y_size * cfg.spacing, x_size * cfg.spacing],
        "boundary__status": boundary,
        "init_topography__initial_elevation": initial_dem,
        "uplift__base_rate": uplift_aligned,
        "uplift__uplift_function_key": function_key,
        "uplift__uplift_array_key": array_key,
        "uplift__uplift_params_json": json.dumps(cfg.uplift_time.params or {}, ensure_ascii=False, sort_keys=True),
        "uplift__total_time": float(cfg.time_total),
        "spl__k_coef": ksp_aligned,
        "spl__area_exp": cfg.area_exp,
        "spl__slope_exp": cfg.slope_exp,
        "diffusion__diffusivity": cfg.k_diff * 1.0e-2,
        **rainfall_input,
    }

    ds_in = xs.create_setup(
        model=model,
        clocks={"time": master_times, "out": out_times},
        master_clock="time",
        input_vars=input_vars,
        output_vars={"topography__elevation": "out"},
    )
    out_ds = ds_in.xsimlab.run(model=model)
    topographies = out_ds.topography__elevation.values

    # 计算每帧 uplift 场（与 FastScape 内部独立计算一次，用于剥蚀诊断）
    x_grid, y_grid = np.meshgrid(
        np.linspace(0.0, (x_size - 1) * cfg.spacing, x_size),
        np.linspace(0.0, (y_size - 1) * cfg.spacing, y_size),
    )
    uplift_frames = []
    for idx, t_yr in enumerate(out_times):
        z_frame = topographies[idx]
        if cfg.uplift_time.mode == "python":
            field = _evaluate_uplift_callable(
                _UPLIFT_FUNCTION_REGISTRY[function_key],
                base_field=uplift_aligned,
                x_grid=x_grid,
                y_grid=y_grid,
                z=z_frame,
                t_yr=float(t_yr),
                params=cfg.uplift_time.params or {},
            )
        else:
            times_years, frames = _UPLIFT_ARRAY_REGISTRY[array_key]
            field = _interpolate_uplift_array(times_years, frames, float(t_yr))
        uplift_frames.append(field)
    uplift_series = np.asarray(uplift_frames, dtype=float)

    return ForwardSeriesResult(
        topography_series=topographies,
        uplift_series=uplift_series,
        output_times=out_times,
        multipliers=None,
    )


def _load_uplift_array(cfg_section: UpliftTimeConfig, *, target_shape: tuple[int, int]):
    array_path = Path(cfg_section.array_path).expanduser().resolve()
    if not array_path.exists():
        raise FileNotFoundError(f"找不到 uplift 数组文件: {array_path}")
    array = np.load(array_path)
    array = np.asarray(array, dtype=float)
    if array.ndim != 3:
        raise ValueError(
            f"uplift 数组必须是 (T,Y,X) 三维，当前 shape={array.shape}"
        )
    times = np.asarray(cfg_section.array_times_years, dtype=float).reshape(-1)
    if times.size != array.shape[0]:
        raise ValueError(
            f"array_times_years 长度 {times.size} 与数组帧数 {array.shape[0]} 不一致"
        )
    if not np.all(np.diff(times) > 0):
        raise ValueError("array_times_years 必须严格递增。")
    if not np.all(np.isfinite(times)):
        raise ValueError("array_times_years 不能包含 NaN/Inf。")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("uplift 数组必须为非负有限数（mm/yr）。")
    if array.shape[1:] != tuple(target_shape):
        # 自动按双线性对齐到 DEM/FastScape 网格，复用反演工具的对齐逻辑
        aligned = np.stack(
            [align_model_field(frame, target_shape, label=f"uplift_array[{i}]", order=1) for i, frame in enumerate(array)],
            axis=0,
        )
        return aligned.astype(float), times
    return array, times


def _build_rainfall_config(rainfall: RainfallConfig, base_dir: Path) -> GaRainfallConfig | None:
    """把正演工具的 RainfallConfig 转成反演工具的 RainfallConfig（FastScape 输入格式一致）。"""
    if rainfall.mode == "uniform":
        validate_rainfall_array(rainfall.value, label="rainfall_factor")
        return GaRainfallConfig(mode="uniform", factor=float(rainfall.value))
    func = load_rainfall_function(rainfall.module_path, rainfall.function_name)
    return GaRainfallConfig(
        mode="python",
        factor=1.0,
        function=func,
        params=dict(rainfall.extra_params or {}),
        source_path=str(rainfall.module_path),
        function_name=rainfall.function_name,
        dynamic=True,
        min_value=rainfall.min_value,
        max_value=rainfall.max_value,
    )
