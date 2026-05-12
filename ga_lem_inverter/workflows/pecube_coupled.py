"""FastScape-to-Pecube coupling smoke workflow."""

from __future__ import annotations

import configparser
import json
from pathlib import Path

import numpy as np

from ga_lem_inverter.config import get_shape
from ga_lem_inverter.integrations.pecube import PecubeEngine
from ga_lem_inverter.integrations.pecube_fitness import (
    load_observations,
    normalized_rmse,
    pecube_grid_from_observation_bounds,
    plot_age_fit,
    plot_age_elevation,
    plot_age_surface_map,
    plot_pecube_dashboard,
    plot_pecube_loss_history,
    plot_residual_map,
    predictions_from_parsed,
    write_prediction_table,
)
from ga_lem_inverter.outputs import RunContext, write_metrics
from ga_lem_inverter.workflows.common import config_float, config_int, create_synthetic_uplift, model_params_from_config
from ga_lem_inverter.pipeline.forward_model import run_fastscape_model
from ga_lem_inverter.pipeline.synthetic_erosion import create_synthetic_erosion_field


def run_pecube_coupled_workflow(config: configparser.ConfigParser, context: RunContext) -> dict:
    """Generate a minimal FastScape-like sequence and run Pecube."""
    engine = PecubeEngine.from_config(config)
    shape = get_shape(config, "Pecube", "shape", get_shape(config, "Synthetic", "shape", (21, 21)))
    n_steps = config_int(config, "Pecube", "time_steps", 2)
    n_steps = max(n_steps, 2)
    if engine.config.sample_observations:
        spatial_grid = pecube_grid_from_observation_bounds(engine.config.sample_observations, shape)
        engine = engine.with_spatial_grid(
            lon0=spatial_grid.lon0,
            lat0=spatial_grid.lat0,
            dlon=spatial_grid.dlon,
            dlat=spatial_grid.dlat,
        )

    seed = config_int(config, "Optimization", "random_seed", 42)
    pattern = config.get("Pecube", "pattern", fallback=config.get("Synthetic", "pattern", fallback="simple"))
    true_uplift = create_synthetic_uplift(shape, pattern, seed)
    ksp = create_synthetic_erosion_field(shape=shape, base_k_sp=config_float(config, "Model", "k_sp_value", 6.92e-6))
    model_params = model_params_from_config(config, shape)

    topographies: list[np.ndarray] = []
    uplifts: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    for index in range(n_steps):
        factor = (index + 1) / n_steps
        uplift = true_uplift * factor
        topography = run_fastscape_model(
            k_sp=ksp,
            uplift=uplift,
            k_diff=model_params["d_diff"],
            x_size=shape[1],
            y_size=shape[0],
            spacing=model_params["spacing"],
            boundary_status=model_params["boundary_status"],
            area_exp=model_params["area_exp"],
            slope_exp=model_params["slope_exp"],
            time_total=model_params["time_total"] * factor,
        )
        topographies.append(np.asarray(topography, dtype=float))
        # Pecube expects uplift in a simple grid sequence. Use mm/yr values for
        # the data files and a separate uniform velocity in Pecube.in.
        uplifts.append(np.asarray(uplift, dtype=float))
        temperatures.append(np.zeros(shape, dtype=float))

    pecube_dir = context.root / "pecube"
    result = engine.run(
        topography_series=topographies,
        uplift_series=uplifts,
        temperature_series=temperatures,
        sample_observations=engine.config.sample_observations,
        output_dir=pecube_dir,
    )

    result_json = pecube_dir / "pecube_result.json"
    result_json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    context.add_artifact(result_json)
    context.add_artifact(result.project.input_file)
    context.add_artifact(result.project.dataset_dir)

    metrics = dict(result.metrics)
    metrics["pecube_project_dir"] = str(result.project.project_dir)
    if engine.config.sample_observations:
        coordinate_system = config.get("Pecube", "observation_coordinate_system", fallback="pecube_output")
        observations = load_observations(
            engine.config.sample_observations,
            shape,
            coordinate_system=coordinate_system,
            dlon=engine.config.dlon,
            dlat=engine.config.dlat,
            lon0=engine.config.lon0,
            lat0=engine.config.lat0,
            dem_crs="EPSG:4326",
            observation_crs=config.get("Pecube", "observation_crs", fallback="EPSG:4326"),
        )
        predictions = predictions_from_parsed(result.parsed, observations)
        thermo_loss = normalized_rmse([item.normalized_residual for item in predictions])
        prediction_path = context.tables_dir / "predicted_thermochronology.csv"
        write_prediction_table(prediction_path, predictions)
        context.add_artifact(prediction_path)
        age_fit_path = context.figure_path("pecube_observed_vs_predicted_ages.png")
        plot_age_fit(predictions, age_fit_path)
        context.add_artifact(age_fit_path)
        residual_path = context.figure_path("pecube_residual_spatial_map.png")
        plot_residual_map(predictions, residual_path)
        context.add_artifact(residual_path)
        age_elevation_path = context.figure_path("pecube_age_elevation.png")
        plot_age_elevation(predictions, age_elevation_path)
        context.add_artifact(age_elevation_path)
        age_surface_path = context.figure_path("pecube_age_surface_map.png")
        plot_age_surface_map(predictions, age_surface_path, terrain=topographies[-1], dlon=engine.config.dlon, dlat=engine.config.dlat)
        context.add_artifact(age_surface_path)
        loss_history = [{
            "evaluation": 1,
            "terrain_loss": 0.0,
            "thermo_loss": thermo_loss,
            "total_loss": thermo_loss,
        }]
        loss_history_path = context.figure_path("pecube_loss_history.png")
        plot_pecube_loss_history(loss_history, loss_history_path)
        context.add_artifact(loss_history_path)
        dashboard_path = context.figure_path("pecube_dashboard.png")
        plot_pecube_dashboard(
            predictions=predictions,
            history=loss_history,
            target_dem=topographies[0],
            generated_dem=topographies[-1],
            uplift=uplifts[-1],
            path=dashboard_path,
        )
        context.add_artifact(dashboard_path)
        metrics.update({
            "thermochronology_loss": thermo_loss,
            "thermo_n_observations": len(predictions),
        })
    local_metrics_path = pecube_dir / "pecube_metrics.json"
    local_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    context.add_artifact(local_metrics_path)
    metrics_path = write_metrics(context, "pecube_metrics.json", metrics)
    context.add_artifact(metrics_path)
    return metrics
