# Legacy main inversion workflow, now used through ga_lem_inverter.runner.
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from datetime import datetime
import sys
from scipy.stats import pearsonr, spearmanr
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
# 过滤所有警告
warnings.filterwarnings('ignore')
# 特定警告过滤
warnings.filterwarnings("ignore", category=FutureWarning, module='xsimlab')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", message=r"Setting up \[LPIPS\]")
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
np.seterr(all='ignore')
from ga_lem_inverter.config import UserConfigError
from ga_lem_inverter.integrations.pecube_fitness import PecubeFitnessEvaluator, pecube_spatial_adapter_from_dem_profile
from ga_lem_inverter.outputs import RunContext, write_metrics
from rasterio.warp import Resampling
from ga_lem_inverter.pipeline.data import (
    read_shapefile,
    load_dem_data,
    calculate_shp_rotation_angle,
    rotate_data,
    reproject_files_to_geographic,
    build_rotated_profile_from_study_area,
    build_rotated_profile_from_dem_footprint,
    reproject_array_to_profile,
)
from ga_lem_inverter.pipeline.preprocessing import interpolate_uplift_cv, unify_array_sizes
from ga_lem_inverter.pipeline.forward_model import (
    align_model_field,
    boundary_status_from_config,
    fastscape_output_times,
    normalize_stage_multipliers,
    run_fastscape_model,
    run_fastscape_series,
    run_fastscape_time_scaled_series,
    stage_edges_from_ma,
)
from ga_lem_inverter.pipeline.fitness import terrain_similarity
from ga_lem_inverter.pipeline.optimization import optimize_uplift_ga, split_decoded_candidate
from ga_lem_inverter.pipeline.rainfall import preview_rainfall_fields, rainfall_from_config, rainfall_metadata
from ga_lem_inverter.pipeline.erosion import create_erosion_field, display_erosion_field, verify_erosion_field
from ga_lem_inverter.pipeline.visualization import (
    plot_comparison,
    plot_uplift_distribution_x,
    plot_uplift_distribution_y,
    display_array_info,
    display_tiff_info,
    plot_3d_surface,
    plot_optimization_history,
    oriented_display_array,
)


def _config_int(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: int,
    aliases: tuple[tuple[str, str], ...] = (),
) -> int:
    if config.has_option(section, option):
        return config.getint(section, option)
    for alias_section, alias_option in aliases:
        if config.has_option(alias_section, alias_option):
            return config.getint(alias_section, alias_option)
    return fallback


def _config_float(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: float,
    aliases: tuple[tuple[str, str], ...] = (),
) -> float:
    if config.has_option(section, option):
        return config.getfloat(section, option)
    for alias_section, alias_option in aliases:
        if config.has_option(alias_section, alias_option):
            return config.getfloat(alias_section, alias_option)
    return fallback
from ga_lem_inverter.pipeline.path_validator import verify_config_paths, verify_file_path


def _config_bool(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: bool,
) -> bool:
    if config.has_option(section, option):
        return config.getboolean(section, option)
    return fallback


def _read_optimization_stages(config: configparser.ConfigParser) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for section in config.sections():
        if not section.lower().startswith("optimizationstage"):
            continue
        stage = {"name": config.get(section, "name", fallback=section)}
        mappings = {
            "population_size": int,
            "max_iterations": int,
            "mutation_probability": float,
            "cross_probability": float,
            "patience": int,
            "min_population_size": int,
            "diversity_threshold": int,
            "diversity_cooldown": int,
            "diversity_random_fraction": float,
            "diversity_best_fraction": float,
            "diversity_terrain_fraction": float,
            "mutation_max_multiplier": float,
            "mutation_stagnation_multiplier": float,
        }
        for key, caster in mappings.items():
            if config.has_option(section, key):
                stage[key] = caster(config.get(section, key))
        for key in ("mutation_schedule",):
            if config.has_option(section, key):
                stage[key] = config.get(section, key)
        if config.has_option(section, "mutation_stagnation_boost"):
            stage["mutation_stagnation_boost"] = config.getboolean(section, "mutation_stagnation_boost")
        stages.append((section, stage))
    stages.sort(key=lambda item: item[0].lower())
    return [stage for _, stage in stages]


def _config_float_list(config: configparser.ConfigParser, section: str, option: str, *, fallback: str) -> list[float]:
    raw = config.get(section, option, fallback=fallback)
    return [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]


def _read_stage_multiplier_bounds(
    config: configparser.ConfigParser,
    *,
    stage_count: int,
    search_mode: str,
) -> tuple[float | list[float], float | list[float]]:
    """Read scalar or per-stage multiplier bounds for uplift history."""
    if search_mode == "free":
        multiplier_min = config.getfloat("UpliftHistory", "multiplier_min", fallback=0.5)
        multiplier_max = config.getfloat("UpliftHistory", "multiplier_max", fallback=1.5)
        if multiplier_min <= 0 or multiplier_max < multiplier_min:
            raise UserConfigError("[UpliftHistory] multiplier_min/max 必须为正且 min <= max。")
        return multiplier_min, multiplier_max

    min_raw = config.get(
        "UpliftHistory",
        "stage_multiplier_min",
        fallback=config.get("UpliftHistory", "multiplier_min", fallback="0.5"),
    )
    max_raw = config.get(
        "UpliftHistory",
        "stage_multiplier_max",
        fallback=config.get("UpliftHistory", "multiplier_max", fallback="1.5"),
    )
    try:
        multiplier_min = [float(part.strip()) for part in min_raw.replace(";", ",").split(",") if part.strip()]
        multiplier_max = [float(part.strip()) for part in max_raw.replace(";", ",").split(",") if part.strip()]
    except ValueError as exc:
        raise UserConfigError(
            "[UpliftHistory] bounded 模式下 stage_multiplier_min/max 必须是逗号分隔数字列表。"
        ) from exc

    if len(multiplier_min) != stage_count or len(multiplier_max) != stage_count:
        raise UserConfigError(
            "[UpliftHistory] bounded 模式下 stage_multiplier_min/max 的数量必须等于阶段数。"
            f"当前阶段数为 {stage_count}，min 有 {len(multiplier_min)} 个，max 有 {len(multiplier_max)} 个。"
        )
    invalid = [
        index + 1
        for index, (low, high) in enumerate(zip(multiplier_min, multiplier_max))
        if low <= 0 or high < low
    ]
    if invalid:
        raise UserConfigError(
            "[UpliftHistory] stage_multiplier_min/max 每一阶段都必须为正且 min <= max。"
            f"配置错误阶段: {invalid}"
        )
    return multiplier_min, multiplier_max


def _read_uplift_history_config(config: configparser.ConfigParser, *, time_total_years: float) -> dict[str, Any]:
    enabled = _config_bool(config, "UpliftHistory", "enabled", fallback=False)
    if not enabled:
        return {"enabled": False}
    mode_raw = config.get("UpliftHistory", "mode", fallback="stage_multiplier").strip().lower()
    explicit_search_mode = config.get("UpliftHistory", "multiplier_search_mode", fallback="").strip().lower()
    if mode_raw in {"free", "bounded"}:
        mode = "stage_multiplier"
        search_mode = mode_raw
    else:
        mode = mode_raw
        search_mode = explicit_search_mode or "free"
    if mode != "stage_multiplier":
        raise UserConfigError("[UpliftHistory] mode 当前支持 stage_multiplier、free 或 bounded。")
    if search_mode not in {"free", "bounded"}:
        raise UserConfigError("[UpliftHistory] multiplier_search_mode 必须是 free 或 bounded。")
    stage_times = _config_float_list(config, "UpliftHistory", "stage_times_ma", fallback=f"{time_total_years / 1e6},0")
    try:
        stage_edges = stage_edges_from_ma(stage_times, time_total_years=time_total_years)
    except ValueError as exc:
        raise UserConfigError(f"[UpliftHistory] stage_times_ma 配置错误: {exc}") from exc
    stage_count = len(stage_times) - 1
    if stage_count < 1:
        raise UserConfigError("[UpliftHistory] stage_times_ma 至少需要形成一个阶段。")
    multiplier_precision = config.getfloat("UpliftHistory", "multiplier_precision", fallback=0.1)
    if multiplier_precision <= 0:
        raise UserConfigError("[UpliftHistory] multiplier_precision 必须大于 0。")
    multiplier_min, multiplier_max = _read_stage_multiplier_bounds(
        config,
        stage_count=stage_count,
        search_mode=search_mode,
    )
    return {
        "enabled": True,
        "mode": mode,
        "multiplier_search_mode": search_mode,
        "stage_times_ma": stage_times,
        "stage_edges_years": stage_edges,
        "stage_count": stage_count,
        "multiplier_min": multiplier_min,
        "multiplier_max": multiplier_max,
        "multiplier_precision": multiplier_precision,
        "normalize_time_weighted_mean": _config_bool(
            config,
            "UpliftHistory",
            "normalize_time_weighted_mean",
            fallback=True,
        ),
    }


def _normalized_stage_multipliers(stage_multipliers: np.ndarray | None, uplift_history: dict[str, Any]) -> np.ndarray | None:
    if not uplift_history.get("enabled"):
        return None
    if stage_multipliers is None:
        stage_multipliers = np.ones(int(uplift_history["stage_count"]), dtype=float)
    return normalize_stage_multipliers(
        stage_multipliers,
        uplift_history["stage_edges_years"],
        enabled=bool(uplift_history.get("normalize_time_weighted_mean", True)),
    )


def _run_candidate_fastscape_series(
    *,
    k_sp,
    uplift,
    k_diff,
    x_size,
    y_size,
    spacing,
    boundary_status,
    area_exp,
    slope_exp,
    rainfall_factor,
    rainfall_model,
    time_total,
    output_steps,
    uplift_history,
    stage_multipliers=None,
):
    output_times = fastscape_output_times(time_total, output_steps)
    multipliers = _normalized_stage_multipliers(stage_multipliers, uplift_history)
    if multipliers is None:
        topography_series = run_fastscape_series(
            k_sp=k_sp,
            uplift=uplift,
            k_diff=k_diff,
            x_size=x_size,
            y_size=y_size,
            spacing=spacing,
            boundary_status=boundary_status,
            area_exp=area_exp,
            slope_exp=slope_exp,
            rainfall_factor=rainfall_factor,
            rainfall_model=rainfall_model,
            time_total=time_total,
            output_steps=output_steps,
        )
        uplift_series = np.repeat(uplift[np.newaxis, :, :], len(topography_series), axis=0)
        return topography_series, uplift_series, None, output_times
    result = run_fastscape_time_scaled_series(
        k_sp=k_sp,
        uplift=uplift,
        k_diff=k_diff,
        x_size=x_size,
        y_size=y_size,
        spacing=spacing,
        boundary_status=boundary_status,
        area_exp=area_exp,
        slope_exp=slope_exp,
        rainfall_factor=rainfall_factor,
        rainfall_model=rainfall_model,
        time_total=time_total,
        output_steps=output_steps,
        stage_edges_years=uplift_history["stage_edges_years"],
        stage_multipliers=multipliers,
    )
    return result.topography_series, result.uplift_series, multipliers, result.output_times


def _plot_uplift_history_summary(
    *,
    stage_uplift: np.ndarray,
    cumulative_stage_uplift: np.ndarray,
    stage_times_ma: list[float],
    stage_multipliers: np.ndarray,
    output_path: Path,
    context: RunContext,
    display_rotated: bool = False,
) -> None:
    """Save a compact visual summary of the optimized stage uplift history."""
    n_stage = int(stage_uplift.shape[0])
    cumulative_stage_uplift_km = cumulative_stage_uplift / 1000.0
    fig, axes = plt.subplots(2, n_stage + 1, figsize=(4 * (n_stage + 1), 7), constrained_layout=True)
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    x = np.arange(n_stage)
    labels = [f"{stage_times_ma[i]:g}-{stage_times_ma[i + 1]:g} Ma" for i in range(n_stage)]
    axes[0, 0].plot(x, stage_multipliers, marker="o", color="#1f77b4", linewidth=2)
    axes[0, 0].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 0].set_ylabel("Multiplier")
    axes[0, 0].set_title("Stage multipliers")
    axes[0, 0].grid(True, alpha=0.25)

    total_cumulative = np.sum(cumulative_stage_uplift_km, axis=0)
    im = axes[1, 0].imshow(oriented_display_array(total_cumulative, rotated=display_rotated), cmap="magma", origin="upper")
    axes[1, 0].set_title("Cumulative uplift (km)")
    axes[1, 0].set_axis_off()
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    rate_vmin = float(np.nanmin(stage_uplift))
    rate_vmax = float(np.nanmax(stage_uplift))
    cum_vmin = float(np.nanmin(cumulative_stage_uplift_km))
    cum_vmax = float(np.nanmax(cumulative_stage_uplift_km))
    if rate_vmin == rate_vmax:
        rate_vmax = rate_vmin + 1.0
    if cum_vmin == cum_vmax:
        cum_vmax = cum_vmin + 1.0

    rate_im = None
    cum_im = None
    for idx in range(n_stage):
        title = labels[idx]
        rate_im = axes[0, idx + 1].imshow(
            oriented_display_array(stage_uplift[idx], rotated=display_rotated),
            cmap="RdBu_r",
            origin="upper",
            vmin=rate_vmin,
            vmax=rate_vmax,
        )
        axes[0, idx + 1].set_title(f"{title}\nrate mm/yr")
        axes[0, idx + 1].set_axis_off()

        cum_im = axes[1, idx + 1].imshow(
            oriented_display_array(cumulative_stage_uplift_km[idx], rotated=display_rotated),
            cmap="magma",
            origin="upper",
            vmin=cum_vmin,
            vmax=cum_vmax,
        )
        axes[1, idx + 1].set_title(f"{title}\ncumulative km")
        axes[1, idx + 1].set_axis_off()

    if rate_im is not None:
        fig.colorbar(rate_im, ax=axes[0, 1:].ravel().tolist(), shrink=0.8, label="Uplift rate (mm/yr)")
    if cum_im is not None:
        fig.colorbar(cum_im, ax=axes[1, 1:].ravel().tolist(), shrink=0.8, label="Cumulative uplift (km)")

    fig.suptitle("Optimized uplift history", fontsize=14)
    fig.savefig(output_path, dpi=200)
    context.add_artifact(output_path)
    plt.close(fig)


def _plot_topography_history_summary(
    *,
    topography_series: np.ndarray,
    output_path: Path,
    context: RunContext,
    output_times_years: np.ndarray | None = None,
    total_time_years: float | None = None,
    display_rotated: bool = False,
    max_frames: int = 6,
) -> None:
    """Save final-solution FastScape topography snapshots and elevation change."""
    series = np.asarray(topography_series, dtype=float)
    if series.ndim != 3 or series.shape[0] < 2:
        logging.warning("跳过地形演化图：topography_series 需要 shape=(time,y,x)，当前 %s", series.shape)
        return

    finite = series[np.isfinite(series)]
    if finite.size == 0:
        logging.warning("跳过地形演化图：topography_series 没有有效数值。")
        return

    n_frames = min(max(2, int(max_frames)), series.shape[0])
    frame_indices = np.unique(np.linspace(0, series.shape[0] - 1, n_frames, dtype=int))
    n_frames = len(frame_indices)
    output_times = None
    if output_times_years is not None:
        output_times = np.asarray(output_times_years, dtype=float).reshape(-1)
        if output_times.size != series.shape[0]:
            logging.warning(
                "地形演化时间标签数量与 topography_series 不一致: %s != %s，将退回 frame 标签。",
                output_times.size,
                series.shape[0],
            )
            output_times = None

    def _frame_label(frame_idx: int) -> str:
        if output_times is None or total_time_years is None:
            return "final" if frame_idx == series.shape[0] - 1 else f"frame {frame_idx + 1}"
        ma_before_present = max(0.0, (float(total_time_years) - float(output_times[frame_idx])) / 1.0e6)
        if frame_idx == series.shape[0] - 1 or ma_before_present < 1e-9:
            return "0 Ma"
        return f"{ma_before_present:g} Ma"

    vmin, vmax = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))

    delta_series = series - series[0]
    finite_delta = delta_series[np.isfinite(delta_series)]
    delta_limit = float(np.nanpercentile(np.abs(finite_delta), 98)) if finite_delta.size else 1.0
    if not np.isfinite(delta_limit) or delta_limit <= 0:
        delta_limit = 1.0

    fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 7), constrained_layout=True)
    if n_frames == 1:
        axes = axes.reshape(2, 1)

    terrain_im = None
    delta_im = None
    for col_idx, frame_idx in enumerate(frame_indices):
        label = _frame_label(frame_idx)
        terrain_im = axes[0, col_idx].imshow(
            oriented_display_array(series[frame_idx], rotated=display_rotated),
            cmap="terrain",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, col_idx].set_title(f"{label}\nelevation")
        axes[0, col_idx].set_axis_off()

        delta_im = axes[1, col_idx].imshow(
            oriented_display_array(delta_series[frame_idx], rotated=display_rotated),
            cmap="RdBu_r",
            origin="upper",
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        axes[1, col_idx].set_title(f"{label}\nchange from first")
        axes[1, col_idx].set_axis_off()

    if terrain_im is not None:
        fig.colorbar(terrain_im, ax=axes[0, :].ravel().tolist(), shrink=0.8, label="Elevation (m)")
    if delta_im is not None:
        fig.colorbar(delta_im, ax=axes[1, :].ravel().tolist(), shrink=0.8, label="Elevation change (m)")

    fig.suptitle("FastScape topography evolution for best solution", fontsize=14)
    fig.savefig(output_path, dpi=200)
    context.add_artifact(output_path)
    plt.close(fig)


def _plot_fastscape_forcing_summary(
    *,
    ksp: np.ndarray,
    rainfall_factor: float,
    uplift: np.ndarray,
    output_path: Path,
    context: RunContext,
    display_rotated: bool = False,
) -> None:
    """Save FastScape forcing fields with runoff separated from Ksp."""
    ksp_display = oriented_display_array(ksp, rotated=display_rotated)
    uplift_display = oriented_display_array(uplift, rotated=display_rotated)
    if np.ndim(rainfall_factor) == 0:
        rainfall_display = np.full_like(ksp_display, float(rainfall_factor), dtype=float)
        rainfall_title = f"FlowAccumulator.runoff = {float(rainfall_factor):g}"
    else:
        rainfall_display = oriented_display_array(np.asarray(rainfall_factor, dtype=float), rotated=display_rotated)
        rainfall_title = "FlowAccumulator.runoff"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    ksp_im = axes[0].imshow(ksp_display, cmap="viridis", origin="upper")
    axes[0].set_title("Ksp")
    axes[0].set_axis_off()
    fig.colorbar(ksp_im, ax=axes[0], fraction=0.045, pad=0.03, label="Ksp")

    rainfall_im = axes[1].imshow(rainfall_display, cmap="Blues", origin="upper", vmin=0.0)
    axes[1].set_title(rainfall_title)
    axes[1].set_axis_off()
    fig.colorbar(rainfall_im, ax=axes[1], fraction=0.045, pad=0.03, label="Runoff / rainfall factor")

    uplift_im = axes[2].imshow(uplift_display, cmap="RdBu_r", origin="upper")
    axes[2].set_title("Best uplift")
    axes[2].set_axis_off()
    fig.colorbar(uplift_im, ax=axes[2], fraction=0.045, pad=0.03, label="Uplift rate (mm/yr)")

    fig.suptitle("FastScape forcing inputs", fontsize=14)
    fig.savefig(output_path, dpi=200)
    context.add_artifact(output_path)
    plt.close(fig)


def _save_rainfall_preview_figures(
    *,
    rainfall_model,
    elevation: np.ndarray,
    shape: tuple[int, int],
    spacing: float,
    total_time_years: float,
    output_path: Path,
    context: RunContext,
    display_rotated: bool = False,
    times_ma: tuple[float, ...] | None = None,
) -> dict[float, np.ndarray]:
    """Save rainfall/runoff preview fields for selected times before present."""
    if times_ma is None:
        times_ma = (float(total_time_years) / 1.0e6, 0.0)
    fields = preview_rainfall_fields(
        rainfall_model,
        shape=shape,
        spacing=spacing,
        elevation=elevation,
        total_time_years=total_time_years,
        times_ma=times_ma,
    )
    count = len(fields)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), constrained_layout=True)
    if count == 1:
        axes = np.asarray([axes])
    values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields.values()])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + max(abs(vmin) * 0.05, 1.0)
    im = None
    for ax, (time_ma, field) in zip(axes, fields.items()):
        im = ax.imshow(oriented_display_array(field, rotated=display_rotated), cmap="Blues", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"{time_ma:g} Ma")
        ax.set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Runoff / rainfall factor")
    fig.suptitle("Rainfall/runoff preview", fontsize=14)
    fig.savefig(output_path, dpi=200)
    context.add_artifact(output_path)
    plt.close(fig)
    return fields


def _load_json_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("无法读取 GA metrics: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    import json
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("无法读取 JSON 列表输出: %s", path)
        return []
    return data if isinstance(data, list) else []


def _json_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        path.read_text(encoding="utf-8")
    except Exception:
        logging.warning("无法读取 JSON 输出: %s", path)
        return False
    return True


def validate_low_resolution_shape(shape: tuple[int, int], scale_factor: int) -> tuple[int, int]:
    """Validate the GA control grid implied by DEM shape and scale_factor."""
    rows, cols = (int(shape[0]), int(shape[1]))
    scale_factor = int(scale_factor)
    if scale_factor < 1:
        raise UserConfigError(f"scale_factor 必须 >= 1，当前为 {scale_factor}。")
    if rows < 2 or cols < 2:
        raise UserConfigError(f"DEM 尺寸太小，无法优化: shape={(rows, cols)}。")
    low_res_shape = (rows // scale_factor, cols // scale_factor)
    if min(low_res_shape) < 1:
        raise UserConfigError(
            f"scale_factor={scale_factor} 对 DEM shape={(rows, cols)} 过大，"
            "会导致低分辨率隆升控制网格为 0。请减小 scale_factor 或使用更大的 DEM。"
        )
    return low_res_shape


def validate_rotation_spatial_constraints(
    *,
    rotation_angle: float,
    pecube_enabled: bool,
    pecube_spatial_mode: str,
) -> dict[str, Any]:
    """Validate spatial interpretation when DEM arrays are rotated."""
    rotated = abs(float(rotation_angle)) > 1e-9
    spatial_mode = str(pecube_spatial_mode or "auto").strip().lower()
    if rotated and pecube_enabled and spatial_mode not in {"auto", "dem", "dem_profile"}:
        logging.warning(
            "已启用 DEM 旋转和 Pecube，但 Pecube spatial_grid=%s 不是 DEM 自动模式；"
            "请确认样品坐标与手动 Pecube 网格一致。",
            spatial_mode,
        )
    return {
        "dem_rotated": rotated,
        "rotation_angle_degrees": float(rotation_angle),
        "spatial_reference_mode": "rotated_georeferenced" if rotated else "dem_georeferenced",
    }


def _reproject_rotated_dem(
    dem_data: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
) -> np.ndarray:
    """Reproject DEM as a continuous elevation surface."""
    return reproject_array_to_profile(
        dem_data,
        src_profile,
        dst_profile,
        resampling=Resampling.bilinear,
    )


def _profile_affine(profile: dict) -> Affine | None:
    transform = profile.get("transform")
    if transform is None:
        return None
    return transform if isinstance(transform, Affine) else Affine(*tuple(transform)[:6])


def _thermo_observation_pixels_for_profile(
    observations: list[Any],
    profile: dict,
    shape: tuple[int, int],
    *,
    display_rotated: bool = False,
) -> list[dict[str, Any]]:
    """Project loaded thermochronology observations back onto a DEM pixel view."""
    if not observations:
        return []
    affine = _profile_affine(profile)
    crs = profile.get("crs")
    if affine is None or crs is None:
        return []

    try:
        crs_obj = CRS.from_user_input(crs)
        transformer = None
        if crs_obj.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
    except Exception as exc:
        logging.warning("无法为样品点建立 DEM 坐标转换，跳过样品点叠加: %s", exc)
        return []

    rows, cols = int(shape[0]), int(shape[1])
    inverse = ~affine
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for observation in observations:
        try:
            lon = float(observation.x)
            lat = float(observation.y)
            map_x, map_y = transformer.transform(lon, lat) if transformer is not None else (lon, lat)
            col, row = inverse * (map_x, map_y)
            pixel_x = float(col) - 0.5
            pixel_y = float(row) - 0.5
            if display_rotated:
                pixel_x = cols - 1 - pixel_x
                pixel_y = rows - 1 - pixel_y
        except Exception:
            continue
        if not (0.0 <= pixel_x < cols and 0.0 <= pixel_y < rows):
            continue
        key = (str(observation.sample_id), int(round(pixel_x * 1000)), int(round(pixel_y * 1000)))
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "sample_id": str(observation.sample_id),
                "x": pixel_x,
                "y": pixel_y,
                "observed_age": float(observation.observed_age),
                "system": str(observation.system),
            }
        )
    return points


def _overlay_sample_points(ax: plt.Axes, sample_points: list[dict[str, Any]] | None) -> None:
    if not sample_points:
        return
    xs = [float(point["x"]) for point in sample_points]
    ys = [float(point["y"]) for point in sample_points]
    ages = [float(point["observed_age"]) for point in sample_points]
    scatter = ax.scatter(
        xs,
        ys,
        c=ages,
        cmap="plasma",
        s=42,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    for index, point in enumerate(sample_points):
        dy = -8 if index % 2 == 0 else 8
        ax.annotate(
            f"{point['sample_id']}\n{float(point['observed_age']):g} Ma",
            xy=(float(point["x"]), float(point["y"])),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=7,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.82},
            arrowprops={"arrowstyle": "-", "color": "0.25", "linewidth": 0.6},
            zorder=6,
        )
    ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03, label="Observed age (Ma)")


def _plot_dem_with_sample_points(
    dem: np.ndarray,
    *,
    title: str,
    path: Path,
    context: RunContext,
    sample_points: list[dict[str, Any]] | None = None,
    display_rotated: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300, constrained_layout=True)
    image = ax.imshow(oriented_display_array(dem, rotated=display_rotated), cmap="terrain", origin="upper")
    ax.set_title(title, fontsize=16, weight="bold", fontstyle="italic")
    ax.set_xlabel("X", fontsize=14, weight="bold", fontstyle="italic")
    ax.set_ylabel("Y", fontsize=14, weight="bold", fontstyle="italic")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Elevation (m)")
    _overlay_sample_points(ax, sample_points)
    fig.savefig(path)
    context.add_artifact(path)
    plt.close(fig)


def _plot_dem_sample_comparison(
    original_dem: np.ndarray,
    rotated_dem: np.ndarray,
    *,
    path: Path,
    context: RunContext,
    original_points: list[dict[str, Any]] | None = None,
    rotated_points: list[dict[str, Any]] | None = None,
    display_rotated: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=220, constrained_layout=True)
    panels = (
        (axes[0], original_dem, "Original DEM + age samples", original_points, False),
        (axes[1], rotated_dem, "Rotated DEM + age samples", rotated_points, display_rotated),
    )
    for ax, dem, title, points, rotate_display in panels:
        image = ax.imshow(oriented_display_array(dem, rotated=rotate_display), cmap="terrain", origin="upper")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="Elevation (m)")
        _overlay_sample_points(ax, points)
    fig.savefig(path)
    context.add_artifact(path)
    plt.close(fig)


def _json_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


# 在文件开头添加这些日志配置函数
def setup_basic_logging():
    """设置基础日志配置"""
    # 清除现有的处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # 设置基本配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 使用stdout而不是stderr
        ]
    )

def setup_file_logging(output_path, filename="optimization.log"):
    """添加文件日志处理器"""
    # 创建文件处理器
    log_file = os.path.join(output_path, filename)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 设置格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # 添加到根日志记录器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logging.info(f"文件日志系统已初始化，日志文件: {log_file}")

# 将目标函数定义在全局作用域

def create_objective_function(resampled_dem, LOW_RES_SHAPE, ORIGINAL_SHAPE,
                           Ksp, D_DIFF, row, col, spacing, time_step_num,
                           total_simulation_time, terrain_resolution,
                           feature_smooth_radius, boundary_status='fixed_value',
                           area_exp=0.43, slope_exp=1, use_lpips=True,
                           rainfall_factor=1.0,
                           rainfall_model=None,
                           pecube_evaluator=None, pecube_time_steps=2,
                           uplift_history=None):
    """创建优化目标函数"""
    def objective_function(uplift_vector):
        try:
            uplift_vector, stage_multipliers = split_decoded_candidate(uplift_vector)
            # 重塑隆升率向量
            uplift_vector = np.array(uplift_vector).reshape(LOW_RES_SHAPE)

            # 插值到高分辨率
            full_res_uplift = interpolate_uplift_cv(uplift_vector, ORIGINAL_SHAPE)

            # 运行Fastscape模型
            topography_series = None
            uplift_series = None
            history_enabled = bool((uplift_history or {}).get("enabled"))
            if (pecube_evaluator is not None and pecube_evaluator.enabled) or history_enabled:
                topography_series, uplift_series, _, _ = _run_candidate_fastscape_series(
                    k_sp=Ksp,
                    uplift=full_res_uplift,
                    k_diff=D_DIFF,
                    x_size=col,
                    y_size=row,
                    spacing=spacing,
                    boundary_status=boundary_status,
                    area_exp=area_exp,
                    slope_exp=slope_exp,
                    rainfall_factor=rainfall_factor,
                    rainfall_model=rainfall_model,
                    time_total=total_simulation_time,
                    output_steps=max(2, pecube_time_steps if pecube_evaluator is not None and pecube_evaluator.enabled else 2),
                    uplift_history=uplift_history or {"enabled": False},
                    stage_multipliers=stage_multipliers,
                )
                generated_elevation = topography_series[-1]
            else:
                generated_elevation = run_fastscape_model(
                    k_sp=Ksp,
                    uplift=full_res_uplift,
                    k_diff=D_DIFF,
                    x_size=col,
                    y_size=row,
                    spacing=spacing,
                    boundary_status=boundary_status,
                    area_exp=area_exp,
                    slope_exp=slope_exp,
                    rainfall_factor=rainfall_factor,
                    rainfall_model=rainfall_model,
                    time_total=total_simulation_time
                )

            # 计算地形相似度
            similarity = terrain_similarity(
                matrix1=resampled_dem,
                matrix2=generated_elevation,
                resolution=terrain_resolution,
                smooth_radius=feature_smooth_radius,
                use_lpips=use_lpips
            )

            terrain_loss = 1 - similarity  # 最小化不相似度
            if pecube_evaluator is not None and pecube_evaluator.enabled:
                result = pecube_evaluator.evaluate(
                    terrain_loss=terrain_loss,
                    generated_dem=generated_elevation,
                    uplift=full_res_uplift,
                    topography_series=topography_series,
                    uplift_series=uplift_series,
                )
                return result.total_loss

            return terrain_loss

        except Exception as e:
            logging.error(f"目标函数计算失败: {e}")
            return np.inf  # 失败时交给 GA 失败率控制处理

    return objective_function

def load_config(config_path: str = './config.ini') -> configparser.ConfigParser:
    """加载配置文件"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        config = configparser.ConfigParser()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config.read_file(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config.read_file(f)

        if 'Paths' in config:
            for key in config['Paths']:
                value = config['Paths'][key].split(';')[0].strip()
                value = value.replace('\\\\', '\\').replace('\\', '/')
                config['Paths'][key] = value

        return config

    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        print(f"当前工作目录: {os.getcwd()}")
        raise

def verify_config(config: configparser.ConfigParser) -> bool:
    """验证配置文件的完整性"""
    required_sections = ['Paths', 'Model', 'GeneticAlgorithm', 'Preprocessing']
    required_params = {
        'Paths': ['terrain_path', 'output_path'],
        'Model': ['k_sp_value', 'ksp_fault', 'd_diff_value', 'boundary_status',
                 'area_exp', 'slope_exp', 'time_total'],
        'GeneticAlgorithm': ['ga_pop_size', 'ga_max_iter', 'ga_prob_cross',
                            'ga_prob_mut', 'lb', 'ub', 'n_jobs'],
        'Preprocessing': ['smooth_sigma', 'scale_factor',
                          'ratio']
    }

    for section in required_sections:
        if section not in config:
            logging.error(f"缺少配置节: {section}")
            return False
        for param in required_params[section]:
            if param not in config[section]:
                logging.error(f"缺少参数: {section}/{param}")
                return False

    if not verify_config_paths(config):
        return False

    return True

# 裁剪掉 NaN 值，保留有效区域
def trim_nan_edges(matrix):
    """裁剪矩阵边缘的 NaN 值"""
    # 找到非 NaN 值的边界
    rows = np.any(~np.isnan(matrix), axis=1)
    cols = np.any(~np.isnan(matrix), axis=0)

    # 获取有效区域的索引范围
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]

    # 裁剪矩阵
    return matrix[row_start:row_end+1, col_start:col_end+1]


def crop_output_border(matrix, border_width):
    """裁掉只用于边界保护的外圈像素；不参与 FastScape 计算本身。"""
    array = np.asarray(matrix)
    border_width = int(border_width or 0)
    if array.ndim != 2 or border_width <= 0:
        return array.copy()
    if array.shape[0] <= 2 * border_width or array.shape[1] <= 2 * border_width:
        return array.copy()
    return array[border_width:-border_width, border_width:-border_width].copy()


def setup_logging(output_path: str) -> None:
    """配置日志系统"""
    log_file = os.path.join(output_path, 'optimization.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_optimization_results(output_path: str | Path, results: dict, context: RunContext | None = None):
    """
    保存优化结果，处理MaskedArray类型。

    参数:
    - output_path: 输出目录路径
    - results: 包含结果数据的字典
    """
    try:
        for name, data in results.items():
            if data is None:
                logging.warning(f"跳过保存 {name}: 数据为None")
                continue

            save_path = Path(output_path) / f'{name}.npy'

            try:
                # 检查是否为MaskedArray类型
                if isinstance(data, np.ma.MaskedArray):
                    # 将masked值替换为NaN并转换为普通数组
                    data_filled = data.filled(np.nan)
                    np.save(save_path, data_filled)
                elif isinstance(data, (np.ndarray, list)):
                    np.save(save_path, np.array(data))
                else:
                    logging.warning(f"跳过保存 {name}: 不支持的数据类型 {type(data)}")
                    continue

                if context is not None:
                    context.add_artifact(save_path)
                logging.info(f"已保存 {name} 到 {save_path}")

            except Exception as e:
                logging.error(f"保存 {name} 时出错: {e}")
                continue

    except Exception as e:
        logging.error(f"保存结果时出错: {e}")
        raise

def fill_Nan(dem_array):
    """
    使用最邻近插值填充 NaN 值

    参数:
    - dem_array: 包含 NaN 的输入数组

    返回:
    - filled_array: 填充后的数组
    """
    from scipy.interpolate import NearestNDInterpolator

    # 创建数组副本
    filled_array = dem_array.copy()

    # 获取非 NaN 值的位置和值
    mask = ~np.isnan(dem_array)
    coords = np.array(np.where(mask)).T
    values = dem_array[mask]

    # 创建插值器
    interpolator = NearestNDInterpolator(coords, values)

    # 获取 NaN 值的位置
    nan_coords = np.array(np.where(~mask)).T

    # 填充 NaN 值
    if len(nan_coords) > 0:
        filled_array[~mask] = interpolator(nan_coords)

    return filled_array

def resolve_optional_file_path(path_value: str, file_type: str) -> Optional[str]:
    """
    读取可选输入文件路径。

    配置里把路径留空、写 none/null/skip/false 时，程序会跳过对应功能。
    这让 demo 可以只靠 DEM 跑通；正式实验需要断层或研究区约束时，再把
    对应 shapefile 路径填回 config.ini。
    """
    raw_path = (path_value or '').split(';')[0].strip()
    if raw_path.lower() in ('', 'none', 'null', 'skip', 'false', '0'):
        logging.info(f"{file_type} 未配置，跳过该输入")
        return None

    verified_path = verify_file_path(raw_path, file_type)
    if verified_path is None:
        logging.warning(f"{file_type} 无法读取，跳过该输入: {raw_path}")
        return None

    return verified_path

def create_uniform_erosion_field(shape, base_k_sp, border_width=2):
    """
    创建无断层约束时使用的均一侵蚀系数场。

    边界仍设为 0，以满足 Fastscape 固定边界和 verify_erosion_field 的检查。
    """
    row, col = shape
    ksp = np.ones((row, col), dtype=np.float64) * base_k_sp
    safe_border = min(border_width, max(row // 2, 0), max(col // 2, 0))
    if safe_border > 0:
        ksp[:safe_border, :] = 0
        ksp[-safe_border:, :] = 0
        ksp[:, :safe_border] = 0
        ksp[:, -safe_border:] = 0
    return ksp


def _apply_ksp_border(ksp: np.ndarray, border_width: int = 2) -> tuple[np.ndarray, int]:
    """Apply FastScape zero boundary rows/columns and return the effective width."""
    bordered = np.asarray(ksp, dtype=np.float64).copy()
    row, col = bordered.shape
    safe_border = min(border_width, max(row // 2, 0), max(col // 2, 0))
    if safe_border > 0:
        bordered[:safe_border, :] = 0
        bordered[-safe_border:, :] = 0
        bordered[:, :safe_border] = 0
        bordered[:, -safe_border:] = 0
    return bordered, safe_border


def _fill_ksp_nodata(ksp: np.ndarray, base_k_sp: float) -> np.ndarray:
    """Fill Ksp NoData with the configured background value, not neighbor interpolation."""
    return np.where(np.isfinite(ksp), ksp, float(base_k_sp)).astype(np.float64)


def create_model_grid_erosion_field(
    *,
    shape: tuple[int, int],
    base_k_sp: float,
    fault_k_sp: float,
    fault_shp_path: Optional[str],
    study_area_shp_path: Optional[str],
    dem_profile: dict,
    border_width: int = 2,
) -> np.ndarray:
    """Create Ksp directly on the final model grid instead of rotating/interpolating an old Ksp."""
    raster_transform = dem_profile.get("transform") if dem_profile is not None else None
    raster_crs = dem_profile.get("crs") if dem_profile is not None else None
    if fault_shp_path and study_area_shp_path and raster_transform is not None and raster_crs is not None:
        ksp = create_erosion_field(
            shape=shape,
            base_k_sp=base_k_sp,
            fault_k_sp=fault_k_sp,
            fault_shp_path=fault_shp_path,
            study_area_shp_path=study_area_shp_path,
            rotation_angle=0,
            border_width=border_width,
            raster_transform=raster_transform,
            raster_crs=raster_crs,
        )
        logging.info("已在最终 DEM 网格上重新栅格化断层 Ksp，未对 Ksp 做旋转插值。")
    else:
        ksp = create_uniform_erosion_field(shape=shape, base_k_sp=base_k_sp, border_width=border_width)
        logging.info("已在最终 DEM 网格上创建均一 Ksp，未对 Ksp 做旋转插值。")

    ksp = _fill_ksp_nodata(ksp, base_k_sp)
    ksp, _ = _apply_ksp_border(ksp, border_width=border_width)
    return ksp


def run_main_workflow(config: configparser.ConfigParser, context: RunContext) -> dict[str, Any]:
    """Run the real-DEM inversion workflow in a structured run directory."""
    try:
        # 1. 首先设置基础日志配置（在任何其他操作之前）
        setup_basic_logging()  # 这是新添加的基础日志配置
        logging.info("程序开始执行")

        logging.info("配置文件加载完成")

        random_seed = _config_int(
            config,
            'Optimization',
            'random_seed',
            fallback=42,
            aliases=(('GeneticAlgorithm', 'random_seed'),),
        )
        np.random.seed(random_seed)
        logging.info(f"随机种子已固定: {random_seed}")

        output_path = context.root
        figures_dir = context.figures_dir
        arrays_dir = context.arrays_dir
        metrics_dir = context.metrics_dir

        try:
            os.makedirs(output_path, exist_ok=True)
            logging.info(f"创建输出目录成功: {output_path}")

            # 4. 更新日志配置以包含文件输出
            setup_file_logging(context.logs_dir, "main.log")
            context.add_artifact(context.logs_dir / "main.log")

        except Exception as e:
            logging.error(f"创建输出目录失败: {e}", exc_info=True)
            return

        # 设置日志
        #setup_logging(output_path)
        logging.info("开始优化过程")

        # 验证输入文件路径。只有 DEM 是必需项；断层和研究区 shapefile 可选。
        terrain_path = verify_file_path(config['Paths']['terrain_path'], '地形栅格文件')
        fault_shp_path = resolve_optional_file_path(
            config['Paths'].get('fault_shp_path', ''),
            '断层 Shapefile'
        )
        study_area_shp_path = resolve_optional_file_path(
            config['Paths'].get('study_area_shp_path', ''),
            '研究区域 Shapefile'
        )

        if terrain_path is None:
            raise UserConfigError("地形栅格文件无效，无法继续。请检查 [Data] terrain_path 是否指向存在的 DEM 文件。")
        config['Paths']['terrain_path'] = terrain_path
        config['Paths']['fault_shp_path'] = fault_shp_path or ''
        config['Paths']['study_area_shp_path'] = study_area_shp_path or ''

        # 检查并统一投影坐标系 (保持这部分)
        logging.info("Step 2: 检查和统一投影坐标系")
        target_crs = config['Preprocessing']['target_crs'] # 从配置文件读取 target_crs
        config = reproject_files_to_geographic(config, target_crs=target_crs) # 传递 target_crs
        terrain_path = config['Paths']['terrain_path']
        fault_shp_path = config['Paths'].get('fault_shp_path', '').strip() or None
        study_area_shp_path = config['Paths'].get('study_area_shp_path', '').strip() or None

        # 1. 数据加载。研究区 shapefile 存在时裁剪 DEM；留空时使用 DEM 全域。
        logging.info("Step 1: 数据加载")
        ratio = config.getfloat('Preprocessing', 'ratio')
        dem_data, dem_profile = load_dem_data(
            file_path=terrain_path,
            study_area_shp_path=study_area_shp_path,
            ratio=ratio
        )
        original_dem_profile = dem_profile.copy()

        if study_area_shp_path:
            study_area = read_shapefile(study_area_shp_path)
            logging.info(f"研究区 Shapefile 已加载，要素数量: {len(study_area)}")
            rotation_angle = calculate_shp_rotation_angle(study_area_shp_path)
        else:
            study_area = None
            rotation_angle = 0.0
            logging.info("未配置研究区 Shapefile，将尝试从 DEM 有效 footprint 自动推断旋转角。")

        if fault_shp_path:
            fault_lines = read_shapefile(fault_shp_path)
            logging.info(f"断层 Shapefile 已加载，要素数量: {len(fault_lines)}")
        else:
            fault_lines = None
            logging.info("未配置断层 Shapefile，将使用均一侵蚀系数场")

        print(f"Calculated rotation angle: {rotation_angle:.2f}°")


        # 2. 创建侵蚀系数场 (在 *原始未旋转* 空间创建)
        logging.info("Step 2: 创建侵蚀系数场")
        global row, col, ORIGINAL_SHAPE, LOW_RES_SHAPE, matrix
        global Ksp, D_DIFF, time_step_num, total_simulation_time
        global terrain_resolution, feature_smooth_radius

        row, col = dem_data.shape # 使用原始 dem_data 的 shape
        ORIGINAL_SHAPE = (row, col)
        scale_factor = config.getint('Preprocessing', 'scale_factor')
        LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)
        logging.info(f"Original shape: {ORIGINAL_SHAPE}")
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        k_sp_value = config.getfloat('Model', 'k_sp_value')
        ksp_fault = config.getfloat('Model', 'ksp_fault')

        # 创建 erosion field：有断层和研究区时叠加断层弱抗性带，否则使用均一场。
        if fault_shp_path and study_area_shp_path:
            Ksp = create_erosion_field(
                shape=ORIGINAL_SHAPE,
                base_k_sp=k_sp_value,
                fault_k_sp=ksp_fault,
                fault_shp_path=fault_shp_path,
                study_area_shp_path=study_area_shp_path,
                rotation_angle=0,
                border_width=2,
                raster_transform=dem_profile.get("transform"),
                raster_crs=dem_profile.get("crs"),
            )
            logging.info("已根据断层 Shapefile 创建非均一侵蚀系数场")
        else:
            Ksp = create_uniform_erosion_field(
                shape=ORIGINAL_SHAPE,
                base_k_sp=k_sp_value,
                border_width=2
            )
            logging.info("已创建均一侵蚀系数场")

        if Ksp.shape != ORIGINAL_SHAPE:
            logging.warning(
                "Ksp shape=%s 与 DEM shape=%s 不一致，将在旋转前自动对齐。",
                Ksp.shape,
                ORIGINAL_SHAPE,
            )
            Ksp = align_model_field(Ksp, ORIGINAL_SHAPE, label="Ksp", order=1)
            safe_border = min(2, max(ORIGINAL_SHAPE[0] // 2, 0), max(ORIGINAL_SHAPE[1] // 2, 0))
            if safe_border > 0:
                Ksp[:safe_border, :] = 0
                Ksp[-safe_border:, :] = 0
                Ksp[:, :safe_border] = 0
                Ksp[:, -safe_border:] = 0

        if not verify_erosion_field(Ksp, shape=ORIGINAL_SHAPE):
            logging.error("侵蚀系数场验证失败")
            return

        # 3. 旋转 DEM，并在最终模型网格上创建 Ksp。
        logging.info("Step 3: 旋转 DEM 并在最终网格创建侵蚀系数场")
        spacing_x = abs(dem_profile['transform'][0])
        spacing_y = abs(dem_profile['transform'][4])
        spacing = (spacing_x + spacing_y) / 2
        display_requires_legacy_flip = False
        ksp_border_width = 2
        if study_area_shp_path and abs(rotation_angle) > 1e-9 and dem_profile.get("crs") is not None:
            rotated_profile = build_rotated_profile_from_study_area(dem_profile, study_area_shp_path, spacing=spacing)
        elif not study_area_shp_path and dem_profile.get("crs") is not None:
            rotated_profile, rotation_angle = build_rotated_profile_from_dem_footprint(
                dem_profile,
                dem_data,
                spacing=spacing,
            )
            logging.info("DEM footprint inferred rotation angle: %.4f°", rotation_angle)
        else:
            rotated_profile = None

        if rotated_profile is not None and abs(rotation_angle) > 1e-9:
            rotated_dem_data = _reproject_rotated_dem(
                dem_data,
                dem_profile,
                rotated_profile,
            )
            dem_profile = rotated_profile
            logging.info("已使用带 transform 的旋转投影网格重采样 DEM；Ksp 将在最终网格上重新生成。")
        elif rotated_profile is not None:
            rotated_dem_data = dem_data.copy()
            dem_profile = rotated_profile
            logging.info("DEM 不需要旋转，保留原始网格；Ksp 将在当前模型网格上重新生成。")
        elif abs(float(rotation_angle)) <= 1e-9:
            rotated_dem_data = dem_data.copy()
            logging.info("DEM 不需要旋转，保留原始数组网格。")
        else:
            rotated_dem_data = rotate_data(dem_data, rotation_angle)
            display_requires_legacy_flip = abs(float(rotation_angle)) > 1e-9

        rotated_dem_data = fill_Nan(rotated_dem_data)
        if dem_profile.get("transform") is not None and dem_profile.get("crs") is not None:
            rotated_Ksp = create_model_grid_erosion_field(
                shape=rotated_dem_data.shape,
                base_k_sp=k_sp_value,
                fault_k_sp=ksp_fault,
                fault_shp_path=fault_shp_path,
                study_area_shp_path=study_area_shp_path,
                dem_profile=dem_profile,
                border_width=ksp_border_width,
            )
        else:
            rotated_Ksp = rotate_data(Ksp, rotation_angle) if abs(float(rotation_angle)) > 1e-9 else Ksp.copy()
            rotated_Ksp = _fill_ksp_nodata(rotated_Ksp, k_sp_value)
            if rotated_Ksp.shape != rotated_dem_data.shape:
                logging.warning(
                    "Legacy Ksp shape=%s 与 DEM shape=%s 不一致，将用最近邻对齐。",
                    rotated_Ksp.shape,
                    rotated_dem_data.shape,
                )
                rotated_Ksp = align_model_field(rotated_Ksp, rotated_dem_data.shape, label="Ksp", order=0)
            rotated_Ksp, _ = _apply_ksp_border(rotated_Ksp, border_width=ksp_border_width)

        # 添加详细的尺寸日志
        logging.info(f"Shape comparison:")
        logging.info(f"Original DEM shape: {dem_data.shape}")
        logging.info(f"Rotated DEM shape: {rotated_dem_data.shape}")
        logging.info(f"Ksp shape: {Ksp.shape}")
        logging.info(f"rotated_Ksp shape: {rotated_Ksp.shape}")

        # 从日志可以看出，问题出在 Ksp 的创建过程中。
        # 虽然 Original DEM shape 是 (177, 189)，旋转后变成 (85, 176)，
        # 但是 Ksp 在创建时就是 (87, 176)，这说明在调用 create_erosion_field 时的形状计算有问题。
        # 添加数据验证
        if rotated_dem_data.shape != rotated_Ksp.shape:
            logging.error(f"Shape mismatch between rotated DEM and Ksp")
            logging.error(f"DEM: {rotated_dem_data.shape}, Ksp: {rotated_Ksp.shape}")
            rotated_Ksp = align_model_field(rotated_Ksp, rotated_dem_data.shape, label="Ksp", order=1)
            logging.info(f"After trimming:")
            logging.info(f"DEM shape: {rotated_dem_data.shape}")
            logging.info(f"Ksp shape: {rotated_Ksp.shape}")

        safe_border = min(
            ksp_border_width,
            max(rotated_Ksp.shape[0] // 2, 0),
            max(rotated_Ksp.shape[1] // 2, 0),
        )
        if safe_border > 0:
            rotated_Ksp[:safe_border, :] = 0
            rotated_Ksp[-safe_border:, :] = 0
            rotated_Ksp[:, :safe_border] = 0
            rotated_Ksp[:, -safe_border:] = 0
            logging.info(f"旋转/对齐后已重新补齐 Ksp 零边界，宽度: {safe_border}")

        # 保存旋转后的 Ksp
        try:
            rotated_ksp_save_path = arrays_dir / 'rotated_ksp.npy'
            np.save(rotated_ksp_save_path, rotated_Ksp)
            context.add_artifact(rotated_ksp_save_path)
            logging.info(f"已保存旋转后的侵蚀系数场到: {rotated_ksp_save_path}")
        except Exception as e:
            logging.error(f"保存旋转后的侵蚀系数场失败: {e}")
        # 更新全局变量以反映旋转后的尺寸
        resampled_dem = rotated_dem_data
        row, col = resampled_dem.shape  # 更新为旋转后的尺寸
        ORIGINAL_SHAPE = (row, col)     # 更新为旋转后的尺寸
        LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)

        # 更新 dem_profile
        dem_profile['height'] = row
        dem_profile['width'] = col

        # 检查并统一全局尺寸
        def validate_shapes():
            global row, col, ORIGINAL_SHAPE, LOW_RES_SHAPE
            if resampled_dem.shape != rotated_Ksp.shape:
                logging.error("Shape mismatch in global variables")
                return False
            row, col = resampled_dem.shape
            ORIGINAL_SHAPE = (row, col)
            LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)
            logging.info(f"Validated shapes: ORIGINAL_SHAPE={ORIGINAL_SHAPE}, LOW_RES_SHAPE={LOW_RES_SHAPE}")
            return True

        # 在更新全局变量后调用验证
        if not validate_shapes():
            raise ValueError("Shape validation failed")

        resampled_dem = rotated_dem_data #  重命名 rotated_dem_data 为 resampled_dem 以便后续代码兼容
        spacing_x = abs(dem_profile['transform'][0])
        spacing_y = abs(dem_profile['transform'][4])
        spacing = (spacing_x + spacing_y) / 2 # 计算 spacing (可能需要更精确的计算)


        display_array_info("Rotated DEM", rotated_dem_data, spacing)
        display_array_info("Rotated Ksp", rotated_Ksp, spacing)





        # In main.py, after DEM preprocessing (using rotated DEM):
        dem_profile_for_raster = {
            'transform': dem_profile['transform'], # 使用 dem_profile 的 transform (可能需要在重采样时更新 transform - 检查 preprocess_terrain_data)
            'shape': resampled_dem.shape # 使用 resampled DEM 的 shape
        }


        # 5. 设置模型参数 (使用 *旋转后* 的数据)
        logging.info("Step 5: 模型参数设置")


        LOW_RES_SHAPE = validate_low_resolution_shape((row, col), scale_factor) # 重新计算 LOW_RES_SHAPE
        logging.info(f"Resampled shape after rotation: {ORIGINAL_SHAPE}") #  注意这里 ORIGINAL_SHAPE 仍然是原始shape，应该输出 resampled_dem.shape 或 ORIGINAL_SHAPE = resampled_dem.shape
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        D_DIFF = config.getfloat('Model', 'd_diff_value')
        time_step_num = 101  # 可以添加到config文件中
        total_simulation_time = config.getfloat('Model', 'time_total')
        uplift_history = _read_uplift_history_config(config, time_total_years=total_simulation_time)
        terrain_resolution = spacing # 可以添加到config文件中
        feature_smooth_radius = 2  # 可以添加到config文件中
        use_lpips = config.getboolean('Fitness', 'use_lpips', fallback=True)
        pecube_time_steps = max(2, config.getint('Pecube', 'time_steps', fallback=2))

        ga_params = {
            'pop': _config_int(config, 'Optimization', 'population_size', fallback=6, aliases=(('GeneticAlgorithm', 'ga_pop_size'),)),
            'max_iter': _config_int(config, 'Optimization', 'max_iterations', fallback=5, aliases=(('GeneticAlgorithm', 'ga_max_iter'),)),
            'prob_cross': _config_float(config, 'Optimization', 'cross_probability', fallback=0.7, aliases=(('GeneticAlgorithm', 'ga_prob_cross'),)),
            'prob_mut': _config_float(config, 'Optimization', 'mutation_probability', fallback=0.05, aliases=(('GeneticAlgorithm', 'ga_prob_mut'),)),
            'lb': _config_float(config, 'Optimization', 'uplift_min', fallback=0.1, aliases=(('GeneticAlgorithm', 'lb'),)),
            'ub': _config_float(config, 'Optimization', 'uplift_max', fallback=1.0, aliases=(('GeneticAlgorithm', 'ub'),)),
            'uplift_precision': _config_float(config, 'Optimization', 'uplift_precision', fallback=0.1, aliases=(('GeneticAlgorithm', 'precision'),)),
            'decay_rate': _config_float(config, 'Optimization', 'decay_rate', fallback=1.0, aliases=(('GeneticAlgorithm', 'decay_rate'),)),
            'min_size_pop': _config_int(config, 'Optimization', 'min_population_size', fallback=4, aliases=(('GeneticAlgorithm', 'min_size_pop'),)),
            'patience': _config_int(config, 'Optimization', 'patience', fallback=3, aliases=(('GeneticAlgorithm', 'patience'),)),
            'random_seed': random_seed,
            'search_strategy': config.get('Optimization', 'search_strategy', fallback='staged').strip().lower(),
            'enable_fitness_cache': _config_bool(config, 'Optimization', 'enable_fitness_cache', fallback=True),
            'diversity_threshold': config.get('Optimization', 'diversity_threshold', fallback=None),
            'diversity_cooldown': _config_int(config, 'Optimization', 'diversity_cooldown', fallback=3),
            'diversity_random_fraction': _config_float(config, 'Optimization', 'diversity_random_fraction', fallback=0.3),
            'diversity_best_fraction': _config_float(config, 'Optimization', 'diversity_best_fraction', fallback=0.5),
            'diversity_terrain_fraction': _config_float(config, 'Optimization', 'diversity_terrain_fraction', fallback=0.2),
            'mutation_schedule': config.get('Optimization', 'mutation_schedule', fallback='adaptive').strip().lower(),
            'mutation_max_multiplier': _config_float(config, 'Optimization', 'mutation_max_multiplier', fallback=2.5),
            'mutation_stagnation_boost': _config_bool(config, 'Optimization', 'mutation_stagnation_boost', fallback=True),
            'mutation_stagnation_multiplier': _config_float(config, 'Optimization', 'mutation_stagnation_multiplier', fallback=1.5),
            'diagnostics_dir': str(context.root),
        }
        if uplift_history.get("enabled"):
            ga_params.update({
                "uplift_history_enabled": True,
                "uplift_history_stage_count": uplift_history["stage_count"],
                "uplift_history_multiplier_min": uplift_history["multiplier_min"],
                "uplift_history_multiplier_max": uplift_history["multiplier_max"],
                "uplift_history_multiplier_precision": uplift_history["multiplier_precision"],
            })
        stages = _read_optimization_stages(config)
        if stages:
            ga_params['stages'] = stages

        display_rotated = display_requires_legacy_flip
        model_params = {
            'Ksp': rotated_Ksp, # 使用 *旋转后* 的 Ksp
            'd_diff': config.getfloat('Model', 'd_diff_value'),
            'boundary_status': boundary_status_from_config(config),
            'area_exp': config.getfloat('Model', 'area_exp'),
            'slope_exp': config.getfloat('Model', 'slope_exp'),
            'rainfall_factor': config.getfloat('Model', 'rainfall_factor', fallback=1.0),
            'rainfall_model': rainfall_from_config(config, base_dir=context.config_path.parent),
            'time_total': config.getfloat('Model', 'time_total'),
            'spacing': spacing,
            'display_rotated': display_rotated,
        }

        pecube_evaluator = PecubeFitnessEvaluator.from_config(
            config=config,
            context=context,
            target_dem=resampled_dem,
            ksp=rotated_Ksp,
            model_params=model_params,
        )
        spatial_mode = config.get("Pecube", "spatial_grid", fallback="auto").strip().lower()
        spatial_metrics = validate_rotation_spatial_constraints(
            rotation_angle=rotation_angle,
            pecube_enabled=pecube_evaluator.enabled,
            pecube_spatial_mode=spatial_mode,
        )
        context.metrics.update(spatial_metrics)
        if pecube_evaluator.enabled:
            if spatial_mode in {"auto", "dem", "dem_profile"}:
                spatial_adapter = pecube_spatial_adapter_from_dem_profile(dem_profile, resampled_dem.shape)
                if spatial_adapter is not None:
                    pecube_evaluator.apply_spatial_adapter(spatial_adapter)
                    spatial_grid = spatial_adapter.grid
                    logging.info(
                        "Pecube 空间网格已由 DEM 自动推导: "
                        f"lon0={spatial_grid.lon0}, lat0={spatial_grid.lat0}, "
                        f"dlon={spatial_grid.dlon}, dlat={spatial_grid.dlat}, crs={spatial_grid.crs}, "
                        f"shape={spatial_adapter.target_shape}, resample={spatial_adapter.resample}"
                    )
                else:
                    logging.warning("DEM 缺少 CRS/transform，Pecube 使用 config.ini 中的 lon0/lat0/dlon/dlat。")
        original_sample_points = _thermo_observation_pixels_for_profile(
            pecube_evaluator.observations,
            original_dem_profile,
            dem_data.shape,
            display_rotated=False,
        )
        rotated_sample_points = _thermo_observation_pixels_for_profile(
            pecube_evaluator.observations,
            dem_profile,
            resampled_dem.shape,
            display_rotated=display_rotated,
        )
        if original_sample_points or rotated_sample_points:
            logging.info(
                "已准备热年代学样品点叠加: original=%s, rotated=%s",
                len(original_sample_points),
                len(rotated_sample_points),
            )

        # 创建objective function (使用 *旋转后* 的 resampled_dem 和 Ksp)
        obj_func = create_objective_function(
            resampled_dem=resampled_dem,
            LOW_RES_SHAPE=LOW_RES_SHAPE,
            ORIGINAL_SHAPE=ORIGINAL_SHAPE, #  这里 ORIGINAL_SHAPE 仍然是原始shape，需要考虑是否修改为旋转后的shape
            Ksp=rotated_Ksp,
            D_DIFF=D_DIFF,
            row=row,
            col=col,
            spacing=spacing,
            time_step_num=time_step_num,
            total_simulation_time=total_simulation_time,
            terrain_resolution=terrain_resolution,
            feature_smooth_radius=feature_smooth_radius,
            boundary_status=model_params["boundary_status"],
            area_exp=model_params["area_exp"],
            slope_exp=model_params["slope_exp"],
            rainfall_factor=model_params["rainfall_factor"],
            rainfall_model=model_params["rainfall_model"],
            use_lpips=use_lpips,
            pecube_evaluator=pecube_evaluator,
            pecube_time_steps=pecube_time_steps,
            uplift_history=uplift_history,
        )

        total_time_ma = total_simulation_time / 1_000_000.0

        # 显示原始DEM
        figure_path = context.figure_path('original_dem.png')
        _plot_dem_with_sample_points(
            dem_data,
            title="Original DEM",
            path=figure_path,
            context=context,
            sample_points=original_sample_points,
        )

        # 显示旋转后的DEM
        figure_path = context.figure_path('rotated_dem.png')
        _plot_dem_with_sample_points(
            rotated_dem_data,
            title="Rotated DEM",
            path=figure_path,
            context=context,
            sample_points=rotated_sample_points,
            display_rotated=display_rotated,
        )

        # 显示侵蚀系数场
        display_erosion_field(rotated_Ksp, shape=ORIGINAL_SHAPE, flip_display=display_rotated)
        figure_path = context.figure_path('erosion_field.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        #叠加显示DEM和侵蚀系数场
        plt.figure(figsize=(15, 10))
        plt.imshow(oriented_display_array(rotated_dem_data, rotated=display_rotated), cmap='terrain', origin='upper')
        plt.imshow(oriented_display_array(rotated_Ksp, rotated=display_rotated), cmap='RdBu_r', alpha=0.5, origin='upper')
        plt.title("Rotated DEM with Erosion Coefficient Field")
        figure_path = context.figure_path('dem_with_erosion_field.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()


        # 绘制DEM对比图
        plot_comparison(
            data1=dem_data, #  对比 *原始* DEM
            data2=oriented_display_array(rotated_dem_data, rotated=display_rotated), # 和 *旋转后* DEM
            title1='Original DEM',
            title2='Rotated DEM',
            value1='Elevation (m)',
            value2='Elevation (m)',
            cmap='terrain',
            figsize=(15, 10),
            origin='upper'
        )
        figure_path = context.figure_path('dem_rotation_comparison.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()
        if original_sample_points or rotated_sample_points:
            figure_path = context.figure_path('dem_rotation_samples_comparison.png')
            _plot_dem_sample_comparison(
                dem_data,
                rotated_dem_data,
                path=figure_path,
                context=context,
                original_points=original_sample_points,
                rotated_points=rotated_sample_points,
                display_rotated=display_rotated,
            )

        #绘制Ksp对比图
        plot_comparison(
            data1=Ksp, #  对比 *原始* Ksp
            data2=oriented_display_array(rotated_Ksp, rotated=display_rotated),
            title1='Original Ksp',
            title2='Rotated Ksp',
            value1='Erosion Coefficient',
            value2='Erosion Coefficient',
            cmap='RdBu_r',
            figsize=(15, 10),
            origin='upper'
        )
        figure_path = context.figure_path('ksp_rotation_comparison.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        # 5. 遗传算法优化
        logging.info("Step 4: 遗传算法优化")
        n_jobs = _config_int(config, 'Optimization', 'n_jobs', fallback=1, aliases=(('GeneticAlgorithm', 'n_jobs'),))

        logging.info("Genetic Algorithm Parameters:")
        for key, value in ga_params.items():
            logging.info(f"{key}: {value}")

        start_time = time.time()
        best_uplift, best_fitness, fitness_history = optimize_uplift_ga(
            obj_func=obj_func,
            resampled_dem=resampled_dem,
            LOW_RES_SHAPE=LOW_RES_SHAPE,
            ORIGINAL_SHAPE=ORIGINAL_SHAPE,
            ga_params=ga_params,
            model_params=model_params,
            n_jobs=n_jobs,
            run_mode='cached'
        )

        if best_uplift is not None:
            best_uplift_vector, best_stage_multipliers = split_decoded_candidate(best_uplift)
            best_stage_multipliers = _normalized_stage_multipliers(best_stage_multipliers, uplift_history)
            best_low_res_uplift = best_uplift_vector.reshape(LOW_RES_SHAPE)
            best_full_res_uplift = interpolate_uplift_cv(best_low_res_uplift, ORIGINAL_SHAPE)
            logging.info(f"Best fitness: {best_fitness}")
            if best_stage_multipliers is not None:
                logging.info("Best uplift-history multipliers: %s", best_stage_multipliers)

            # 6. 绘制优化历史
            if fitness_history is not None:
                fig_history = plot_optimization_history(fitness_history)
                figure_path = context.figure_path('optimization_history.png')
                fig_history.savefig(figure_path)
                context.add_artifact(figure_path)
                plt.close(fig_history)
                logging.info("优化历史曲线已保存")
        end_time = time.time()

        logging.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Best fitness: {best_fitness}")
        ga_metrics_path = context.metrics_dir / "ga_metrics.json"
        ga_history_path = context.tables_dir / "ga_history.csv"
        stage_metrics_path = context.metrics_dir / "stage_metrics.json"
        ga_metrics = _load_json_metrics(ga_metrics_path)
        stage_metrics_payload = _load_json_list(stage_metrics_path)
        if ga_metrics:
            context.add_artifact(ga_metrics_path)
            context.add_artifact(ga_history_path)
            for key, value in ga_metrics.items():
                context.metrics[f"ga_{key}"] = _json_metric_value(value)
        if stage_metrics_payload or _json_exists(stage_metrics_path):
            context.add_artifact(stage_metrics_path)
        if stage_metrics_payload:
            context.metrics["ga_stage_list"] = ", ".join(str(stage.get("stage", "")) for stage in stage_metrics_payload)
            context.metrics["ga_stage_best_fitness"] = {
                str(stage.get("stage", f"stage{idx + 1}")): _json_metric_value(stage.get("best_fitness"))
                for idx, stage in enumerate(stage_metrics_payload)
            }

        # 6. 结果处理和可视化
        logging.info("Step 5: 结果处理和可视化")
        if best_uplift is not None:
            best_uplift_vector, best_stage_multipliers = split_decoded_candidate(best_uplift)
            best_stage_multipliers = _normalized_stage_multipliers(best_stage_multipliers, uplift_history)
            best_low_res_uplift = best_uplift_vector.reshape(LOW_RES_SHAPE)
            best_full_res_uplift = interpolate_uplift_cv(best_low_res_uplift, ORIGINAL_SHAPE)
            display_low_res_uplift = oriented_display_array(best_low_res_uplift, rotated=display_rotated)
            display_full_res_uplift = oriented_display_array(best_full_res_uplift, rotated=display_rotated)

            display_array_info("Best Uplift Field", best_full_res_uplift, spacing)

            # 绘制隆升率对比图
            plot_comparison(
                data1=display_low_res_uplift,
                data2=display_full_res_uplift,
                title1='Best Low Resolution Uplift',
                title2='Best Full Resolution Uplift',
                value1='Uplift Rate (mm/yr)',
                value2='Uplift Rate (mm/yr)',
                cmap='RdBu_r'
            )
            figure_path = context.figure_path('uplift_comparison.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            true_uplift_path = os.path.join(
                os.path.dirname(config['Paths']['terrain_path']),
                'demo_true_uplift.npy'
            )
            demo_metrics = {}
            if os.path.exists(true_uplift_path):
                try:
                    true_uplift = np.load(true_uplift_path)
                    if true_uplift.shape == best_full_res_uplift.shape:
                        true_uplift_cropped = crop_output_border(true_uplift, safe_border)
                        best_full_res_uplift_for_metrics = crop_output_border(best_full_res_uplift, safe_border)
                        uplift_pearson = pearsonr(
                            true_uplift_cropped.ravel(),
                            best_full_res_uplift_for_metrics.ravel()
                        ).statistic
                        uplift_spearman = spearmanr(
                            true_uplift_cropped.ravel(),
                            best_full_res_uplift_for_metrics.ravel()
                        ).statistic
                        uplift_rmse = float(np.sqrt(np.mean((true_uplift_cropped - best_full_res_uplift_for_metrics) ** 2)))
                        demo_metrics.update({
                            'uplift_pearson': float(uplift_pearson),
                            'uplift_spearman': float(uplift_spearman),
                            'uplift_rmse': uplift_rmse
                        })
                        logging.info(
                            "Demo uplift metrics: "
                            f"Pearson={uplift_pearson:.4f}, "
                            f"Spearman={uplift_spearman:.4f}, RMSE={uplift_rmse:.4f}"
                        )
                        plot_comparison(
                            data1=oriented_display_array(true_uplift_cropped, rotated=display_rotated),
                            data2=oriented_display_array(best_full_res_uplift_for_metrics, rotated=display_rotated),
                            title1='Demo True Uplift',
                            title2='Inverted Uplift',
                            value1='Uplift Rate (mm/yr)',
                            value2='Uplift Rate (mm/yr)',
                            cmap='RdBu_r'
                        )
                        figure_path = context.figure_path('demo_true_vs_inverted_uplift.png')
                        plt.savefig(figure_path)
                        context.add_artifact(figure_path)
                        plt.close()
                    else:
                        logging.warning(
                            f"跳过 demo 真值 uplift 对比：形状不一致 "
                            f"{true_uplift.shape} vs {best_full_res_uplift.shape}"
                        )
                except Exception as e:
                    logging.warning(f"读取 demo 真值 uplift 失败，跳过对比: {e}")

            # 生成最终地形
            final_output_times_years = None
            if uplift_history.get("enabled"):
                final_series_for_output, best_uplift_series, best_stage_multipliers, final_output_times_years = _run_candidate_fastscape_series(
                    k_sp=rotated_Ksp,
                    uplift=best_full_res_uplift,
                    k_diff=D_DIFF,
                    x_size=col,
                    y_size=row,
                    spacing=spacing,
                    boundary_status=model_params['boundary_status'],
                    area_exp=config.getfloat('Model', 'area_exp'),
                    slope_exp=config.getfloat('Model', 'slope_exp'),
                    rainfall_factor=model_params['rainfall_factor'],
                    rainfall_model=model_params['rainfall_model'],
                    time_total=total_simulation_time,
                    output_steps=max(2, pecube_time_steps),
                    uplift_history=uplift_history,
                    stage_multipliers=best_stage_multipliers,
                )
                final_elevation = final_series_for_output[-1]
            else:
                best_uplift_series = None
                final_series_for_output = None
                final_elevation = run_fastscape_model(
                    k_sp=rotated_Ksp, # 使用旋转后的 Ksp
                    uplift=best_full_res_uplift,
                    k_diff=D_DIFF,
                    x_size=col,
                    y_size=row,
                    spacing=spacing,
                    boundary_status=model_params['boundary_status'],
                    area_exp=config.getfloat('Model', 'area_exp'),
                    slope_exp=config.getfloat('Model', 'slope_exp'),
                    rainfall_factor=model_params['rainfall_factor'],
                    rainfall_model=model_params['rainfall_model'],
                    time_total=total_simulation_time
                )
            final_elevation_cropped = crop_output_border(final_elevation, safe_border)
            target_dem_cropped = crop_output_border(resampled_dem, safe_border)
            best_full_res_uplift_cropped = crop_output_border(best_full_res_uplift, safe_border)
            topography_series_cropped = None
            if final_series_for_output is not None:
                topography_series_cropped = np.asarray([
                    crop_output_border(frame, safe_border)
                    for frame in final_series_for_output
                ])
            display_final_elevation = oriented_display_array(final_elevation_cropped, rotated=display_rotated)
            display_target_dem = oriented_display_array(target_dem_cropped, rotated=display_rotated)
            display_full_res_uplift_cropped = oriented_display_array(best_full_res_uplift_cropped, rotated=display_rotated)
            logging.info(
                "输出/可视化已裁掉 Ksp 零边界: "
                f"border={safe_border}, raw_shape={final_elevation.shape}, "
                f"cropped_shape={final_elevation_cropped.shape}"
            )
            rainfall_preview_fields = _save_rainfall_preview_figures(
                rainfall_model=model_params["rainfall_model"],
                elevation=resampled_dem,
                shape=ORIGINAL_SHAPE,
                spacing=spacing,
                total_time_years=total_simulation_time,
                output_path=context.figure_path("rainfall_preview.png"),
                context=context,
                display_rotated=display_rotated,
            )
            rainfall_present = crop_output_border(rainfall_preview_fields[min(rainfall_preview_fields)], safe_border)
            rotated_ksp_cropped = crop_output_border(rotated_Ksp, safe_border)
            _plot_fastscape_forcing_summary(
                ksp=rotated_ksp_cropped,
                rainfall_factor=rainfall_present,
                uplift=best_full_res_uplift_cropped,
                output_path=context.figure_path("fastscape_forcing_inputs.png"),
                context=context,
                display_rotated=display_rotated,
            )

            # 绘制地形对比图
            terrain_pearson = pearsonr(target_dem_cropped.ravel(), final_elevation_cropped.ravel()).statistic
            terrain_spearman = spearmanr(target_dem_cropped.ravel(), final_elevation_cropped.ravel()).statistic
            terrain_rmse = float(np.sqrt(np.mean((target_dem_cropped - final_elevation_cropped) ** 2)))
            rainfall_stats = rainfall_metadata(model_params["rainfall_model"])
            rainfall_stats.update({
                "rainfall_preview_min": float(np.nanmin(rainfall_present)),
                "rainfall_preview_max": float(np.nanmax(rainfall_present)),
                "rainfall_preview_mean": float(np.nanmean(rainfall_present)),
            })
            demo_metrics.update({
                **rainfall_stats,
                'terrain_pearson': float(terrain_pearson),
                'terrain_spearman': float(terrain_spearman),
                'terrain_rmse': terrain_rmse,
                'terrain_loss': float(best_fitness) if not pecube_evaluator.enabled else float(pecube_evaluator.best_result.terrain_loss if pecube_evaluator.best_result else best_fitness),
                'total_loss': float(best_fitness),
                'output_border_crop_pixels': int(safe_border),
                'raw_output_rows': int(final_elevation.shape[0]),
                'raw_output_cols': int(final_elevation.shape[1]),
                'cropped_output_rows': int(final_elevation_cropped.shape[0]),
                'cropped_output_cols': int(final_elevation_cropped.shape[1]),
            })
            if best_stage_multipliers is not None:
                demo_metrics["uplift_history_enabled"] = 1.0
                for idx, multiplier in enumerate(best_stage_multipliers, start=1):
                    demo_metrics[f"uplift_history_m{idx}"] = float(multiplier)
                for idx, duration in enumerate(np.diff(uplift_history["stage_edges_years"]) / 1e6, start=1):
                    demo_metrics[f"uplift_history_duration_ma_{idx}"] = float(duration)
                if final_output_times_years is not None:
                    for idx, time_years in enumerate(final_output_times_years, start=1):
                        demo_metrics[f"topography_output_time_ma_before_present_{idx}"] = float(
                            max(0.0, (total_simulation_time - time_years) / 1e6)
                        )
            else:
                demo_metrics["uplift_history_enabled"] = 0.0
            logging.info(
                "Terrain metrics: "
                f"Pearson={terrain_pearson:.4f}, "
                f"Spearman={terrain_spearman:.4f}, RMSE={terrain_rmse:.4f}"
            )

            metrics_txt_path = metrics_dir / 'demo_metrics.txt'
            with open(metrics_txt_path, 'w') as metrics_file:
                for key, value in demo_metrics.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        metrics_file.write(f"{key} = {float(value):.6f}\n")
                    else:
                        metrics_file.write(f"{key} = {value}\n")
            context.add_artifact(metrics_txt_path)
            if pecube_evaluator.enabled:
                if final_series_for_output is None:
                    final_series, final_uplift_series, _, final_output_times_years = _run_candidate_fastscape_series(
                        k_sp=rotated_Ksp,
                        uplift=best_full_res_uplift,
                        k_diff=D_DIFF,
                        x_size=col,
                        y_size=row,
                        spacing=spacing,
                        boundary_status=model_params['boundary_status'],
                        area_exp=config.getfloat('Model', 'area_exp'),
                        slope_exp=config.getfloat('Model', 'slope_exp'),
                        rainfall_factor=model_params['rainfall_factor'],
                        rainfall_model=model_params['rainfall_model'],
                        time_total=total_simulation_time,
                        output_steps=max(2, pecube_time_steps),
                        uplift_history=uplift_history,
                        stage_multipliers=best_stage_multipliers,
                    )
                else:
                    final_series = final_series_for_output
                    final_uplift_series = best_uplift_series
                final_terrain_loss = 1 - terrain_similarity(
                    matrix1=resampled_dem,
                    matrix2=final_series[-1],
                    resolution=terrain_resolution,
                    smooth_radius=feature_smooth_radius,
                    use_lpips=use_lpips,
                )
                final_pecube_result = pecube_evaluator.evaluate(
                    terrain_loss=final_terrain_loss,
                    generated_dem=final_series[-1],
                    uplift=best_full_res_uplift,
                    topography_series=final_series,
                    uplift_series=final_uplift_series,
                    force=True,
                    force_best=True,
                )
                demo_metrics.update(final_pecube_result.metrics())
            pecube_metrics = pecube_evaluator.save_best_outputs(
                generated_dem=final_elevation,
                uplift=best_full_res_uplift,
                output_border_crop=safe_border,
            )
            demo_metrics.update(pecube_metrics)
            write_metrics(context, "main_metrics.json", {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in demo_metrics.items()})

            plot_comparison(
                data1=display_final_elevation,
                data2=display_target_dem, #  注意这里对比的是 *旋转后且重采样* 的 DEM (resampled_dem)
                title1='Generated Terrain',
                title2='Target Landscape',
                value1='Elevation (m)',
                value2='Elevation (m)',
                cmap='terrain',
                shared_scale=False
            )
            figure_path = context.figure_path('terrain_comparison.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            plot_comparison(
                data1=display_final_elevation,
                data2=display_target_dem,
                title1='Generated Terrain',
                title2='Target Landscape',
                value1='Elevation (m)',
                value2='Elevation (m)',
                cmap='terrain',
                shared_scale=True
            )
            figure_path = context.figure_path('terrain_comparison_shared_scale.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            # 绘制隆升分布图
            plot_uplift_distribution_x(display_full_res_uplift_cropped, total_time_ma=total_time_ma)
            figure_path = context.figure_path('uplift_distribution_x.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            plot_uplift_distribution_y(display_full_res_uplift_cropped, total_time_ma=total_time_ma)
            figure_path = context.figure_path('uplift_distribution_y.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            stage_uplift_cropped = None
            cumulative_stage_uplift_cropped = None
            if best_stage_multipliers is not None:
                stage_uplift_cropped = np.asarray([
                    best_full_res_uplift_cropped * multiplier
                    for multiplier in best_stage_multipliers
                ])
                durations_ma = np.diff(uplift_history["stage_edges_years"]) / 1e6
                cumulative_stage_uplift_cropped = np.asarray([
                    stage_uplift_cropped[idx] * durations_ma[idx]
                    for idx in range(len(durations_ma))
                ])
                _plot_uplift_history_summary(
                    stage_uplift=stage_uplift_cropped,
                    cumulative_stage_uplift=cumulative_stage_uplift_cropped,
                    stage_times_ma=uplift_history["stage_times_ma"],
                    stage_multipliers=best_stage_multipliers,
                    output_path=context.figure_path("uplift_history_summary.png"),
                    context=context,
                    display_rotated=display_rotated,
                )
            if topography_series_cropped is not None:
                _plot_topography_history_summary(
                    topography_series=topography_series_cropped,
                    output_times_years=final_output_times_years,
                    total_time_years=total_simulation_time,
                    output_path=context.figure_path("topography_history_summary.png"),
                    context=context,
                    display_rotated=display_rotated,
                )

            # 绘制3D地形可视化
            fig_3d = plot_3d_surface(
                data=display_final_elevation,
                uplift=display_full_res_uplift_cropped,
                title="3D Terrain Surface"
            )
            figure_path = context.figure_path('3d_terrain.png')
            fig_3d.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close(fig_3d)
            logging.info("3D地形可视化已保存")

            # 保存结果
            results = {
                    'best_full_res_uplift': best_full_res_uplift_cropped,
                    'final_elevation': final_elevation_cropped,
                    'target_dem': target_dem_cropped,
                    'best_full_res_uplift_raw': best_full_res_uplift,
                    'final_elevation_raw': final_elevation,
                    'stage_uplift': stage_uplift_cropped,
                    'cumulative_stage_uplift': cumulative_stage_uplift_cropped,
                    'stage_multipliers': best_stage_multipliers,
                    'topography_series': topography_series_cropped,
                    'topography_output_times_years': final_output_times_years,
                    'fitness_history': np.array(fitness_history) if fitness_history is not None else None
                }
            save_optimization_results(arrays_dir, results, context)
            # 8. 保存参数配置
            config_file = metrics_dir / 'parameters.txt'
            with open(config_file, 'w') as f:
                f.write("Model Parameters:\n")
                for section in config.sections():
                    f.write(f"\n[{section}]\n")
                    for key, value in config[section].items():
                        f.write(f"{key} = {value}\n")

                f.write("\nOptimization Results:\n")
                f.write(f"Best Fitness: {best_fitness}\n")
                f.write(f"Optimization Time: {end_time - start_time:.2f} seconds\n")
            context.add_artifact(config_file)

            logging.info(f"所有结果已保存到: {output_path}")
            logging.info("优化过程成功完成")
            logging.info(f"Results saved to: {output_path}")
            return {
                "best_fitness": float(best_fitness),
                "optimization_time_seconds": float(end_time - start_time),
                "original_shape": list(ORIGINAL_SHAPE),
                "low_res_shape": list(LOW_RES_SHAPE),
                **{f"ga_{k}": _json_metric_value(v) for k, v in ga_metrics.items()},
                **{k: _json_metric_value(v) for k, v in demo_metrics.items()},
            }

        else:
            logging.warning("遗传算法未能找到有效解。")
            raise RuntimeError("遗传算法未能找到有效解。请检查 DEM、K、隆升率范围和 GA 参数。")

    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        logging.exception("Exception details:")
        raise

def main():
    from ga_lem_inverter.runner import main as runner_main
    runner_main(default_mode="main")


if __name__ == "__main__":
    main()
