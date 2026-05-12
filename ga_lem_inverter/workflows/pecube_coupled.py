"""FastScape-to-Pecube coupling smoke workflow."""

from __future__ import annotations

import configparser
import json
from pathlib import Path

import numpy as np

from ga_lem_inverter.config import get_shape
from ga_lem_inverter.integrations.pecube import PecubeEngine
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
    local_metrics_path = pecube_dir / "pecube_metrics.json"
    local_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    context.add_artifact(local_metrics_path)
    metrics_path = write_metrics(context, "pecube_metrics.json", metrics)
    context.add_artifact(metrics_path)
    return metrics
