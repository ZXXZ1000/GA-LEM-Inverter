# model_runner.py
import xsimlab as xs
import numpy as np
from fastscape.models import basic_model
import logging
import warnings


VALID_BOUNDARY_STATUS = {"fixed_value", "core", "looped"}


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

        output_steps = max(2, int(output_steps))

        # 在运行模型前添加以下代码
        warnings.filterwarnings("ignore", category=FutureWarning,
                            message="variable .* with name matching its dimension")
        # Pecube cannot reliably consume FastScape's raw t=0 random seed
        # topography as the oldest surface. Emit only evolved snapshots while
        # keeping the output clock aligned with the master clock.
        master_times = np.linspace(0, time_total, 101)
        output_steps = min(output_steps, len(master_times) - 1)
        output_indices = np.linspace(1, len(master_times) - 1, output_steps, dtype=int)
        out_times = master_times[output_indices]
        ds_in = xs.create_setup(
            model=basic_model,
            clocks={'time': master_times, 'out': out_times},
            master_clock='time',
            input_vars={
                'grid__shape': [y_size, x_size],
                'grid__length': [y_size * spacing, x_size * spacing],
                'boundary__status': boundary_status,
                'uplift__rate': uplift * 10**(-3),
                'init_topography__seed': initial_topography_seed,
                'spl__k_coef': k_sp,
                'spl__area_exp': area_exp,
                'spl__slope_exp': slope_exp,
                'diffusion__diffusivity': k_diff * 10**(-2),
            },
            output_vars={
                'topography__elevation': 'out'}
        )
        out_ds = (ds_in.xsimlab.run(model=basic_model))
        return out_ds.topography__elevation.values
    except Exception as e:
        logging.error(f"运行 fastscape 模型出错: {e}")
        raise RuntimeError(f"运行 fastscape 模型出错: {e}")


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
        initial_topography_seed=initial_topography_seed,
        output_steps=21,
    )
    return series[-1]
