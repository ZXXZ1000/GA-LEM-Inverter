"""正演主流程：读配置 → 载入数据 → 跑 FastScape → 算剥蚀 → 出图写盘。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from forward_simulator.config import ForwardConfig, load_forward_config
from forward_simulator.erosion_metrics import (
    ErosionFields,
    compute_erosion_fields,
    summarize_metrics,
)
from forward_simulator.fastscape_runner import ForwardSeriesResult, run_forward
from forward_simulator.outputs import (
    ForwardRunContext,
    create_run_context,
    finalize_run,
)
from forward_simulator import visualizer


logger = logging.getLogger(__name__)


def run_forward_simulation(config_path: str | Path) -> ForwardRunContext:
    """整个正演流程的唯一入口。"""
    cfg = load_forward_config(config_path)
    context = create_run_context(cfg.output_root, cfg.config_path)
    _attach_file_logger(context)

    logger.info("=== FastScape 正演开始 ===")
    logger.info("配置文件: %s", cfg.config_path)
    logger.info("输出目录: %s", context.root)
    logger.info("DEM 文件: %s", cfg.dem_path)
    logger.info("uplift_base: %s", cfg.uplift_base_path or f"标量 {cfg.uplift_value} mm/yr")
    logger.info(
        "时长 %.3g 年, spacing=%.1f m, output_steps=%d, uplift_time_mode=%s, rainfall_mode=%s",
        cfg.time_total,
        cfg.spacing,
        cfg.output_steps,
        cfg.uplift_time.mode,
        cfg.rainfall.mode,
    )

    info: dict[str, Any] = {"parameters": {}, "metrics": {}}
    try:
        initial_dem = _load_initial_dem(cfg)
        uplift_base = _load_uplift_base(cfg, target_shape=initial_dem.shape)
        ksp = _load_ksp(cfg, target_shape=initial_dem.shape)

        info["parameters"].update(_parameters_summary(cfg, initial_dem))

        wall_start = time.perf_counter()
        forward_result = run_forward(
            cfg,
            initial_dem=initial_dem,
            uplift_base_field=uplift_base,
            ksp_field=ksp,
        )
        wall_elapsed = time.perf_counter() - wall_start
        logger.info("FastScape 完成，耗时 %.1f 秒", wall_elapsed)

        topo_with_t0, uplift_with_t0, times_with_t0 = _prepend_initial_state(
            forward_result, initial_dem=initial_dem, uplift_base=uplift_base, cfg=cfg
        )

        fields = compute_erosion_fields(
            output_times_years=times_with_t0,
            topography_series=topo_with_t0,
            uplift_series_mm_per_yr=uplift_with_t0,
            initial_dem=initial_dem,
        )

        metrics = summarize_metrics(fields)
        metrics["wall_time_seconds"] = round(float(wall_elapsed), 2)
        info["metrics"] = metrics

        _save_arrays(context, cfg, fields, uplift_base=uplift_base, initial_dem=initial_dem)
        _save_figures(context, cfg, fields, uplift_base=uplift_base, forward_result=forward_result)

        finalize_run(context, status="success", info=info, message="正演完成")
        logger.info("=== 正演完成 ===")
        logger.info("查看 summary.md: %s", context.root / "summary.md")
    except Exception as exc:  # 让用户看见中文错误
        logger.exception("正演失败: %s", exc)
        info["metrics"]["error"] = str(exc)
        finalize_run(context, status="failed", info=info, message=str(exc))
        raise
    return context


# ---------- 数据载入 ---------------------------------------------------------

def _load_initial_dem(cfg: ForwardConfig) -> np.ndarray:
    path = cfg.dem_path
    if path.suffix.lower() == ".npy":
        dem = np.load(path)
    else:
        # tif 等栅格走 rasterio；这里复用反演工具的读取逻辑可以省事，但为了
        # 隔离不强依赖；用最直接的 rasterio 调用。
        import rasterio

        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float64)
            dem = np.flipud(dem)  # 与反演工具读取保持一致：让 origin=lower 对齐
            nodata = src.nodata if src.nodata is not None else -32768
            dem = np.where(dem == nodata, np.nan, dem)
    dem = np.asarray(dem, dtype=float)
    if dem.ndim != 2:
        raise ValueError(f"DEM 必须是二维，当前 shape={dem.shape}")
    if not np.isfinite(dem).any():
        raise ValueError("DEM 全为无效值。")
    if not np.isfinite(dem).all():
        median = float(np.nanmedian(dem))
        logger.warning("DEM 含 NaN，已用中位数 %.2f 填充。", median)
        dem = np.where(np.isfinite(dem), dem, median)
    logger.info("载入 DEM: shape=%s, range=[%.1f, %.1f] m", dem.shape, dem.min(), dem.max())
    return dem


def _load_uplift_base(cfg: ForwardConfig, *, target_shape: tuple[int, int]) -> np.ndarray:
    if cfg.uplift_base_path is None:
        return np.full(target_shape, cfg.uplift_value, dtype=float)
    path = cfg.uplift_base_path
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float64)
            array = np.flipud(array)
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"uplift_base 必须是二维，当前 shape={array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("uplift_base 必须是有限非负数，单位 mm/yr。")
    if array.shape != tuple(target_shape):
        from ga_lem_inverter.pipeline.forward_model import align_model_field

        array = align_model_field(array, target_shape, label="uplift_base", order=1)
    logger.info(
        "载入 uplift_base: shape=%s, range=[%.3f, %.3f] mm/yr",
        array.shape,
        float(array.min()),
        float(array.max()),
    )
    return array


def _load_ksp(cfg: ForwardConfig, *, target_shape: tuple[int, int]) -> np.ndarray | float:
    if cfg.ksp_path is None:
        return float(cfg.ksp_value)
    path = cfg.ksp_path
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float64)
            array = np.flipud(array)
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"ksp 数组必须是二维，当前 shape={array.shape}")
    if array.shape != tuple(target_shape):
        from ga_lem_inverter.pipeline.forward_model import align_model_field

        array = align_model_field(array, target_shape, label="ksp", order=1)
    logger.info("载入 ksp: shape=%s, range=[%.3g, %.3g]", array.shape, float(array.min()), float(array.max()))
    return array


# ---------- 序列后处理 -------------------------------------------------------

def _prepend_initial_state(
    result: ForwardSeriesResult,
    *,
    initial_dem: np.ndarray,
    uplift_base: np.ndarray,
    cfg: ForwardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 t=0 这一帧（现今 DEM）补到序列最前面，方便剥蚀计算。"""
    out_times = np.asarray(result.output_times, dtype=float).reshape(-1)
    if out_times.size > 0 and out_times[0] == 0.0:
        return result.topography_series, result.uplift_series, out_times
    times = np.concatenate([[0.0], out_times])
    topo = np.concatenate([initial_dem[np.newaxis, :, :], result.topography_series], axis=0)
    # t=0 的 uplift 沿用第一帧 uplift（最接近现今的瞬时场）
    if result.uplift_series.shape[0] >= 1:
        u0 = result.uplift_series[0]
    else:
        u0 = uplift_base
    uplifts = np.concatenate([u0[np.newaxis, :, :], result.uplift_series], axis=0)
    return topo, uplifts, times


# ---------- 写盘 -------------------------------------------------------------

def _save_arrays(
    context: ForwardRunContext,
    cfg: ForwardConfig,
    fields: ErosionFields,
    *,
    uplift_base: np.ndarray,
    initial_dem: np.ndarray,
) -> None:
    arrays_dir = context.arrays_dir
    np.save(arrays_dir / "initial_dem.npy", initial_dem)
    context.add_artifact(arrays_dir / "initial_dem.npy")
    np.save(arrays_dir / "uplift_base.npy", uplift_base)
    context.add_artifact(arrays_dir / "uplift_base.npy")
    np.save(arrays_dir / "output_times_years.npy", fields.output_times_years)
    context.add_artifact(arrays_dir / "output_times_years.npy")
    if cfg.save_topography_series:
        np.save(arrays_dir / "topography_series.npy", fields.topography_series)
        context.add_artifact(arrays_dir / "topography_series.npy")
    if cfg.save_uplift_series:
        np.save(arrays_dir / "uplift_series.npy", fields.uplift_series)
        context.add_artifact(arrays_dir / "uplift_series.npy")
    if cfg.save_cumulative_erosion:
        np.save(arrays_dir / "cumulative_erosion.npy", fields.cumulative_erosion)
        context.add_artifact(arrays_dir / "cumulative_erosion.npy")
    if cfg.save_mean_erosion_rate:
        np.save(arrays_dir / "mean_erosion_rate.npy", fields.mean_erosion_rate)
        context.add_artifact(arrays_dir / "mean_erosion_rate.npy")
    if cfg.save_net_uplift:
        np.save(arrays_dir / "net_uplift.npy", fields.net_uplift)
        context.add_artifact(arrays_dir / "net_uplift.npy")


def _save_figures(
    context: ForwardRunContext,
    cfg: ForwardConfig,
    fields: ErosionFields,
    *,
    uplift_base: np.ndarray,
    forward_result: ForwardSeriesResult,
) -> None:
    fig_path = visualizer.plot_dem(
        fields.topography_series[0],
        context.figure_path("initial_dem.png"),
        title="Initial DEM (t = 0)",
    )
    context.add_artifact(fig_path)

    fig_path = visualizer.plot_dem(
        fields.topography_series[-1],
        context.figure_path("final_dem.png"),
        title=f"Final DEM (t = {fields.output_times_years[-1]/1e6:.2f} Ma)",
    )
    context.add_artifact(fig_path)

    if cfg.plot_history_grid:
        fig_path = visualizer.plot_history_grid(
            fields,
            context.figure_path("topography_history.png"),
        )
        context.add_artifact(fig_path)

    fig_path = visualizer.plot_uplift_input(
        uplift_base,
        forward_result.output_times,
        forward_result.multipliers,
        context.figure_path("uplift_input.png"),
    )
    context.add_artifact(fig_path)

    if cfg.save_cumulative_erosion:
        fig_path = visualizer.plot_field_final(
            fields.cumulative_erosion[-1],
            context.figure_path("cumulative_erosion_final.png"),
            title=f"Cumulative erosion at t = {fields.output_times_years[-1]/1e6:.2f} Ma",
            label="erosion (m)",
            cmap="magma",
        )
        context.add_artifact(fig_path)

    if cfg.save_mean_erosion_rate:
        fig_path = visualizer.plot_field_final(
            fields.mean_erosion_rate[-1],
            context.figure_path("mean_erosion_rate_final.png"),
            title="Mean erosion rate over total time",
            label="rate (mm/yr)",
            cmap="plasma",
        )
        context.add_artifact(fig_path)

    if cfg.save_net_uplift:
        fig_path = visualizer.plot_field_final(
            fields.net_uplift[-1],
            context.figure_path("net_uplift_final.png"),
            title="Net surface uplift z(t) - z0",
            label="net uplift (m)",
            cmap="RdBu_r",
        )
        context.add_artifact(fig_path)

    if cfg.plot_erosion_history:
        fig_path = visualizer.plot_erosion_history(
            fields,
            context.figure_path("erosion_history.png"),
        )
        context.add_artifact(fig_path)

    if cfg.rainfall.mode == "python":
        # 只保存 t=0 的快照供检查
        try:
            from ga_lem_inverter.pipeline.rainfall import (
                RainfallConfig as GaRainfallConfig,
                load_rainfall_function,
                preview_rainfall_fields,
            )

            func = load_rainfall_function(cfg.rainfall.module_path, cfg.rainfall.function_name)
            rainfall_obj = GaRainfallConfig(
                mode="python",
                factor=1.0,
                function=func,
                params=dict(cfg.rainfall.extra_params or {}),
                source_path=str(cfg.rainfall.module_path),
                function_name=cfg.rainfall.function_name,
                min_value=cfg.rainfall.min_value,
                max_value=cfg.rainfall.max_value,
            )
            preview = preview_rainfall_fields(
                rainfall_obj,
                shape=fields.topography_series[0].shape,
                spacing=cfg.spacing,
                elevation=fields.topography_series[0],
                total_time_years=cfg.time_total,
                times_ma=(0.0,),
            )
            field0 = preview[0.0]
            fig_path = visualizer.plot_rainfall_preview(
                field0, context.figure_path("rainfall_preview.png")
            )
            context.add_artifact(fig_path)
        except Exception as exc:
            logger.warning("rainfall preview 出错（不影响主流程）: %s", exc)


# ---------- 摘要参数 ---------------------------------------------------------

def _parameters_summary(cfg: ForwardConfig, initial_dem: np.ndarray) -> dict[str, Any]:
    return {
        "dem_path": str(cfg.dem_path),
        "dem_shape": list(initial_dem.shape),
        "uplift_base_path": str(cfg.uplift_base_path) if cfg.uplift_base_path else f"uniform={cfg.uplift_value} mm/yr",
        "ksp": str(cfg.ksp_path) if cfg.ksp_path else f"uniform={cfg.ksp_value}",
        "time_total_years": cfg.time_total,
        "spacing_m": cfg.spacing,
        "output_steps": cfg.output_steps,
        "boundary_LRTB": ",".join(cfg.boundary),
        "k_diff": cfg.k_diff,
        "area_exp": cfg.area_exp,
        "slope_exp": cfg.slope_exp,
        "uplift_time_mode": cfg.uplift_time.mode,
        "rainfall_mode": cfg.rainfall.mode,
        "rainfall_value_or_module": (
            cfg.rainfall.value if cfg.rainfall.mode == "uniform" else cfg.rainfall.module_path
        ),
    }


def _attach_file_logger(context: ForwardRunContext) -> None:
    log_path = context.logs_dir / "forward.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    context.add_artifact(log_path)
