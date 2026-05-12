"""Shared workflow helpers for GA-LEM-Inverter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from ga_lem_inverter.config import get_bool, get_shape
from ga_lem_inverter.outputs import RunContext, write_metrics
from ga_lem_inverter.pipeline.fitness import terrain_similarity
from ga_lem_inverter.pipeline.forward_model import run_fastscape_model
from ga_lem_inverter.pipeline.optimization import optimize_uplift_ga
from ga_lem_inverter.pipeline.preprocessing import interpolate_uplift_cv
from ga_lem_inverter.pipeline.synthetic_erosion import create_synthetic_erosion_field
from ga_lem_inverter.pipeline.visualization import (
    plot_3d_surface,
    plot_comparison,
    plot_optimization_history,
    plot_uplift_distribution,
)


def config_float(config, section: str, key: str, default: float) -> float:
    return config.getfloat(section, key, fallback=default)


def config_int(config, section: str, key: str, default: int) -> int:
    return config.getint(section, key, fallback=default)


def ga_params_from_config(config, *, pop_default: int = 2, iter_default: int = 1) -> dict[str, Any]:
    pop = config_int(config, "Optimization", "population_size", pop_default)
    min_size = config_int(config, "Optimization", "min_population_size", min(pop, 2))
    min_size = min(min_size, pop)
    return {
        "pop": pop,
        "max_iter": config_int(config, "Optimization", "max_iterations", iter_default),
        "prob_cross": config_float(config, "Optimization", "cross_probability", 0.8),
        "prob_mut": config_float(config, "Optimization", "mutation_probability", 0.05),
        "lb": config_float(config, "Optimization", "uplift_min", 3.0),
        "ub": config_float(config, "Optimization", "uplift_max", 12.0),
        "decay_rate": config_float(config, "Optimization", "decay_rate", 1.0),
        "min_size_pop": min_size,
        "patience": config_int(config, "Optimization", "patience", 1),
        "random_seed": config_int(config, "Optimization", "random_seed", 42),
    }


def model_params_from_config(config, shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "k_sp_base": config_float(config, "Model", "k_sp_value", 6.92e-6),
        "k_sp_fault": config_float(config, "Model", "ksp_fault", 2.0e-5),
        "d_diff": config_float(config, "Model", "d_diff_value", 19.2),
        "boundary_status": config.get("Model", "boundary_status", fallback="fixed_value"),
        "area_exp": config_float(config, "Model", "area_exp", 0.43),
        "slope_exp": config_float(config, "Model", "slope_exp", 1.0),
        "time_total": config_float(config, "Model", "time_total", 1.0e4),
        "spacing": config_float(config, "Model", "spacing", 900.0),
        "shape": shape,
    }


def create_synthetic_uplift(shape: tuple[int, int], pattern: str = "simple", seed: int = 42) -> np.ndarray:
    """Create a known uplift field for quick validation experiments."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    if pattern == "simple":
        uplift = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
        uplift = 5 + 5 * uplift
    elif pattern == "medium":
        uplift1 = np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.1)
        uplift2 = np.exp(-((X - 0.7) ** 2 + (Y - 0.7) ** 2) / 0.1)
        uplift = 5 + 5 * (uplift1 + uplift2) / max((uplift1 + uplift2).max(), 1e-12)
    elif pattern == "complex":
        main_fault = np.exp(-((0.8 * X + 0.6 * Y - 0.8) ** 2) / 0.01) * 3.5
        conjugate_fault1 = np.exp(-((0.7 * X - 0.7 * Y - 0.2) ** 2) / 0.008) * 2.0
        conjugate_fault2 = np.exp(-((0.6 * X - 0.8 * Y + 0.3) ** 2) / 0.008) * 2.0
        regional_trend = 2.0 * (1 - Y)
        local_structure = gaussian_filter(rng.random((rows, cols)), sigma=6) * 0.5
        uplift = np.clip(4 + main_fault + conjugate_fault1 + conjugate_fault2 + regional_trend + local_structure, 5, 10)
    else:
        raise ValueError(f"未知 synthetic pattern: {pattern}")

    return uplift.astype(float)


def build_objective(target_dem: np.ndarray, low_res_shape: tuple[int, int], model_params: dict[str, Any],
                    ksp: np.ndarray, use_lpips: bool):
    shape = target_dem.shape

    def objective_function(uplift_vector):
        try:
            low_res_uplift = np.asarray(uplift_vector).reshape(low_res_shape)
            full_res_uplift = interpolate_uplift_cv(low_res_uplift, shape)
            generated = run_fastscape_model(
                k_sp=ksp,
                uplift=full_res_uplift,
                k_diff=model_params["d_diff"],
                x_size=shape[1],
                y_size=shape[0],
                spacing=model_params["spacing"],
                boundary_status=model_params["boundary_status"],
                area_exp=model_params["area_exp"],
                slope_exp=model_params["slope_exp"],
                time_total=model_params["time_total"],
            )
            similarity = terrain_similarity(
                matrix1=target_dem,
                matrix2=generated,
                resolution=model_params["spacing"],
                smooth_radius=2,
                use_lpips=use_lpips,
            )
            return 1 - similarity
        except Exception as exc:
            logging.error("目标函数计算失败: %s", exc)
            return 1.0

    return objective_function


def run_synthetic_case(
    *,
    config,
    context: RunContext,
    pattern: str,
    shape: tuple[int, int],
    scale_factor: int,
    output_prefix: str = "synthetic",
) -> dict[str, float]:
    seed = config_int(config, "Optimization", "random_seed", 42)
    np.random.seed(seed)
    low_res_shape = (shape[0] // scale_factor, shape[1] // scale_factor)
    if min(low_res_shape) < 1:
        raise ValueError(f"scale_factor={scale_factor} 对 shape={shape} 过大，低分辨率网格会变成 0。")

    true_uplift = create_synthetic_uplift(shape, pattern, seed)
    ksp = create_synthetic_erosion_field(shape=shape, base_k_sp=config_float(config, "Model", "k_sp_value", 6.92e-6))
    model_params = model_params_from_config(config, shape)
    model_params["Ksp"] = ksp

    target_dem = run_fastscape_model(
        k_sp=ksp,
        uplift=true_uplift,
        k_diff=model_params["d_diff"],
        x_size=shape[1],
        y_size=shape[0],
        spacing=model_params["spacing"],
        boundary_status=model_params["boundary_status"],
        area_exp=model_params["area_exp"],
        slope_exp=model_params["slope_exp"],
        time_total=model_params["time_total"],
    )

    use_lpips = get_bool(config, "Fitness", "use_lpips", True)
    objective = build_objective(target_dem, low_res_shape, model_params, ksp, use_lpips)
    ga_params = ga_params_from_config(config)
    n_jobs = config_int(config, "Optimization", "n_jobs", 1)
    best_uplift, best_fitness, fitness_history = optimize_uplift_ga(
        obj_func=objective,
        resampled_dem=target_dem,
        LOW_RES_SHAPE=low_res_shape,
        ORIGINAL_SHAPE=shape,
        ga_params=ga_params,
        model_params=model_params,
        n_jobs=n_jobs,
        run_mode=None,
    )
    if best_uplift is None:
        raise RuntimeError("GA 未能返回有效隆升场。")

    low_res_uplift = np.asarray(best_uplift).reshape(low_res_shape)
    inverted_uplift = interpolate_uplift_cv(low_res_uplift, shape)
    final_dem = run_fastscape_model(
        k_sp=ksp,
        uplift=inverted_uplift,
        k_diff=model_params["d_diff"],
        x_size=shape[1],
        y_size=shape[0],
        spacing=model_params["spacing"],
        boundary_status=model_params["boundary_status"],
        area_exp=model_params["area_exp"],
        slope_exp=model_params["slope_exp"],
        time_total=model_params["time_total"],
    )

    arrays = {
        f"{output_prefix}_true_uplift.npy": true_uplift,
        f"{output_prefix}_ksp.npy": ksp,
        f"{output_prefix}_target_dem.npy": target_dem,
        f"{output_prefix}_low_res_uplift.npy": low_res_uplift,
        f"{output_prefix}_inverted_uplift.npy": inverted_uplift,
        f"{output_prefix}_final_dem.npy": final_dem,
        f"{output_prefix}_fitness_history.npy": np.asarray(fitness_history, dtype=float),
    }
    for filename, array in arrays.items():
        path = context.arrays_dir / filename
        np.save(path, array)
        context.add_artifact(path)

    metrics = {
        f"{output_prefix}_best_fitness": float(best_fitness),
        f"{output_prefix}_uplift_pearson": float(pearsonr(true_uplift.ravel(), inverted_uplift.ravel()).statistic),
        f"{output_prefix}_uplift_spearman": float(spearmanr(true_uplift.ravel(), inverted_uplift.ravel()).statistic),
        f"{output_prefix}_uplift_rmse": float(np.sqrt(mean_squared_error(true_uplift, inverted_uplift))),
        f"{output_prefix}_terrain_pearson": float(pearsonr(target_dem.ravel(), final_dem.ravel()).statistic),
        f"{output_prefix}_terrain_spearman": float(spearmanr(target_dem.ravel(), final_dem.ravel()).statistic),
        f"{output_prefix}_terrain_rmse": float(np.sqrt(mean_squared_error(target_dem, final_dem))),
    }
    write_metrics(context, f"{output_prefix}_metrics.json", metrics)

    _save_case_figures(context, output_prefix, true_uplift, inverted_uplift, target_dem, final_dem, fitness_history)
    return metrics


def _save_case_figures(
    context: RunContext,
    prefix: str,
    true_uplift: np.ndarray,
    inverted_uplift: np.ndarray,
    target_dem: np.ndarray,
    final_dem: np.ndarray,
    fitness_history,
) -> None:
    plot_comparison(
        true_uplift,
        inverted_uplift,
        "True Uplift Rate",
        "Inverted Uplift Rate",
        "Uplift Rate (mm/yr)",
        "Uplift Rate (mm/yr)",
        cmap="RdBu_r",
    )
    _save_current_figure(context, f"{prefix}_uplift_comparison.png")

    plot_comparison(
        target_dem,
        final_dem,
        "Target DEM",
        "Simulated DEM",
        "Elevation (m)",
        "Elevation (m)",
        cmap="terrain",
    )
    _save_current_figure(context, f"{prefix}_dem_comparison.png")

    plot_uplift_distribution(inverted_uplift)
    _save_current_figure(context, f"{prefix}_uplift_distribution.png")

    fig_3d = plot_3d_surface(final_dem, inverted_uplift, "3D Terrain with Uplift Field")
    path = context.figure_path(f"{prefix}_3d_terrain.png")
    fig_3d.savefig(path)
    plt.close(fig_3d)
    context.add_artifact(path)

    if fitness_history is not None and len(fitness_history) > 0:
        fig_history = plot_optimization_history(fitness_history)
        path = context.figure_path(f"{prefix}_optimization_history.png")
        fig_history.savefig(path)
        plt.close(fig_history)
        context.add_artifact(path)


def _save_current_figure(context: RunContext, filename: str) -> None:
    path = context.figure_path(filename)
    plt.savefig(path)
    plt.close()
    context.add_artifact(path)


def default_synthetic_shape(config, section: str) -> tuple[int, int]:
    return get_shape(config, section, "shape", (64, 64))
