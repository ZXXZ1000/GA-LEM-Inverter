import configparser
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ga_lem_inverter.integrations.pecube_fitness import (
    PecubeFitnessEvaluator,
    ThermochronologyObservation,
)
from ga_lem_inverter.outputs import RunContext


class RecordingPecubeEngine:
    def __init__(self):
        self.config = SimpleNamespace(
            sample_observations=Path("samples.csv"),
            sea_level_temperature=15.0,
            lapse_rate=6.5,
        )
        self.calls = []

    def run(self, *, topography_series, uplift_series, temperature_series, sample_observations, output_dir):
        self.calls.append(
            {
                "topography_series": [np.asarray(item, dtype=float) for item in topography_series],
                "uplift_series": [np.asarray(item, dtype=float) for item in uplift_series],
                "temperature_series": [np.asarray(item, dtype=float) for item in temperature_series],
                "sample_observations": sample_observations,
                "output_dir": output_dir,
            }
        )
        return SimpleNamespace(
            parsed=SimpleNamespace(csv_files={}, files=[], output_dir=Path(output_dir)),
            project=SimpleNamespace(project_dir=Path(output_dir) / "PGB01"),
            metrics={},
        )


class PecubeFitnessAcceptanceTests(unittest.TestCase):
    def test_pecube_evaluator_passes_full_topography_series_to_engine(self):
        """产品验收：Pecube 约束应按现今到过去接收 FastScape 序列。"""
        config = configparser.ConfigParser()
        config["Pecube"] = {"enabled": "false"}
        config["Fitness"] = {"terrain_loss_weight": "1.0", "thermo_loss_weight": "0.2"}

        with tempfile.TemporaryDirectory() as tmpdir:
            context = RunContext(
                mode="main",
                root=Path(tmpdir),
                started_at="test",
                config_path=Path("config.ini"),
                config=config,
            )
            evaluator = PecubeFitnessEvaluator(
                config=config,
                context=context,
                target_dem=np.zeros((3, 3), dtype=float),
                ksp=np.ones((3, 3), dtype=float),
                model_params={},
            )
            engine = RecordingPecubeEngine()
            evaluator.engine = engine
            evaluator.enabled = True
            evaluator.observations = [
                ThermochronologyObservation("S1", 0.0, 0.0, 0.0, "AHe", 1.0, 0.1)
            ]

            evaluator.predictions_from_parsed = None
            topography_series = np.stack(
                [
                    np.zeros((3, 3), dtype=float),
                    np.ones((3, 3), dtype=float),
                    np.full((3, 3), 2.0, dtype=float),
                ]
            )
            uplift_series = np.stack(
                [
                    np.full((3, 3), 0.1, dtype=float),
                    np.full((3, 3), 0.3, dtype=float),
                    np.full((3, 3), 0.5, dtype=float),
                ]
            )
            temperature_series = np.stack(
                [
                    np.full((3, 3), 10.0, dtype=float),
                    np.full((3, 3), 20.0, dtype=float),
                    np.full((3, 3), 30.0, dtype=float),
                ]
            )

            with mock.patch(
                "ga_lem_inverter.integrations.pecube_fitness.predictions_from_parsed",
                return_value=[],
            ), mock.patch(
                "ga_lem_inverter.integrations.pecube_fitness.normalized_rmse",
                return_value=0.0,
            ):
                evaluator.evaluate(
                    terrain_loss=0.2,
                    generated_dem=np.full((3, 3), 2.0, dtype=float),
                    uplift=np.full((3, 3), 0.5, dtype=float),
                    topography_series=topography_series,
                    uplift_series=uplift_series,
                    temperature_series=temperature_series,
                )

        self.assertEqual(len(engine.calls), 1)
        call = engine.calls[0]
        self.assertEqual(len(call["topography_series"]), 3)
        self.assertEqual(len(call["uplift_series"]), 3)
        self.assertEqual(len(call["temperature_series"]), 3)
        self.assertTrue(np.array_equal(call["topography_series"][0], np.full((3, 3), 2.0)))
        self.assertTrue(np.array_equal(call["topography_series"][-1], np.zeros((3, 3))))
        self.assertTrue(np.array_equal(call["uplift_series"][0], np.full((3, 3), 0.5)))
        self.assertTrue(np.array_equal(call["uplift_series"][-1], np.full((3, 3), 0.1)))
        self.assertTrue(np.array_equal(call["temperature_series"][1], np.full((3, 3), 20.0)))

    def test_pecube_evaluator_rejects_mismatched_uplift_series_length(self):
        """产品验收：Pecube topo/uplift/temp 时间序列帧数不一致时必须直接报错。"""
        config = configparser.ConfigParser()
        config["Pecube"] = {"enabled": "false", "fail_strategy": "raise"}
        config["Fitness"] = {"terrain_loss_weight": "1.0", "thermo_loss_weight": "0.2"}

        with tempfile.TemporaryDirectory() as tmpdir:
            context = RunContext(
                mode="main",
                root=Path(tmpdir),
                started_at="test",
                config_path=Path("config.ini"),
                config=config,
            )
            evaluator = PecubeFitnessEvaluator(
                config=config,
                context=context,
                target_dem=np.zeros((3, 3), dtype=float),
                ksp=np.ones((3, 3), dtype=float),
                model_params={},
            )
            evaluator.engine = RecordingPecubeEngine()
            evaluator.enabled = True
            evaluator.observations = [
                ThermochronologyObservation("S1", 0.0, 0.0, 0.0, "AHe", 1.0, 0.1)
            ]

            with self.assertRaisesRegex(RuntimeError, "uplift_series 帧数必须等于 topography_series"):
                evaluator.evaluate(
                    terrain_loss=0.2,
                    generated_dem=np.full((3, 3), 2.0, dtype=float),
                    uplift=np.full((3, 3), 0.5, dtype=float),
                    topography_series=np.zeros((3, 3, 3), dtype=float),
                    uplift_series=np.zeros((2, 3, 3), dtype=float),
                    temperature_series=np.zeros((3, 3, 3), dtype=float),
                )

    def test_pecube_evaluator_autogenerates_temperature_series_from_topography(self):
        """产品验收：主优化不传 temp* 时，Pecube 温度序列必须自动由地形生成。"""
        config = configparser.ConfigParser()
        config["Pecube"] = {"enabled": "false"}
        config["Fitness"] = {"terrain_loss_weight": "1.0", "thermo_loss_weight": "0.2"}

        with tempfile.TemporaryDirectory() as tmpdir:
            context = RunContext(
                mode="main",
                root=Path(tmpdir),
                started_at="test",
                config_path=Path("config.ini"),
                config=config,
            )
            evaluator = PecubeFitnessEvaluator(
                config=config,
                context=context,
                target_dem=np.zeros((2, 2), dtype=float),
                ksp=np.ones((2, 2), dtype=float),
                model_params={},
            )
            engine = RecordingPecubeEngine()
            evaluator.engine = engine
            evaluator.enabled = True
            evaluator.observations = [
                ThermochronologyObservation("S1", 0.0, 0.0, 0.0, "AHe", 1.0, 0.1)
            ]

            topography_series = np.stack(
                [
                    np.array([[0.0, 1000.0], [2000.0, 3000.0]]),
                    np.array([[500.0, 1500.0], [2500.0, 3500.0]]),
                ]
            )
            uplift_series = np.ones((2, 2, 2), dtype=float)

            with mock.patch(
                "ga_lem_inverter.integrations.pecube_fitness.predictions_from_parsed",
                return_value=[],
            ), mock.patch(
                "ga_lem_inverter.integrations.pecube_fitness.normalized_rmse",
                return_value=0.0,
            ):
                evaluator.evaluate(
                    terrain_loss=0.2,
                    generated_dem=topography_series[-1],
                    uplift=np.ones((2, 2), dtype=float),
                    topography_series=topography_series,
                    uplift_series=uplift_series,
                )

        temperatures = engine.calls[0]["temperature_series"]
        self.assertTrue(np.allclose(engine.calls[0]["topography_series"][0], topography_series[-1]))
        self.assertTrue(np.allclose(engine.calls[0]["topography_series"][1], topography_series[0]))
        self.assertTrue(np.allclose(temperatures[0], [[11.75, 5.25], [-1.25, -7.75]]))
        self.assertTrue(np.allclose(temperatures[1], [[15.0, 8.5], [2.0, -4.5]]))


if __name__ == "__main__":
    unittest.main()
