import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ga_lem_inverter.integrations.pecube_fitness import (
    ThermochronologyObservation,
    predictions_from_parsed,
    scaled_unit_loss,
)
from ga_lem_inverter.integrations.pecube_parser import PecubeParsedOutput
from ga_lem_inverter.integrations.pecube_project import PecubeProjectBuilder
from ga_lem_inverter.integrations.pecube_project import PecubeProjectConfig, surface_temperature_from_topography


class PecubeSchemaAcceptanceTests(unittest.TestCase):
    def test_standard_sample_schema_converts_to_native_pecube_a_file(self):
        """产品验收：用户只维护一套 sample_observations CSV，系统自动转成 Pecube A-file。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "samples.csv"
            source.write_text(
                "\n".join(
                    [
                        "sample_id,lon,lat,elevation,system,observed_age,sigma",
                        "S1,103.5,31.2,1200,AHe,12.3,0.4",
                        "S1,103.5,31.2,1200,AFT,25.1,1.2",
                        "S2,103.6,31.3,2300,ZHe,45.0,2.0",
                    ]
                ),
                encoding="utf-8",
            )
            target = Path(tmpdir) / "observations.csv"
            PecubeProjectBuilder._write_native_observation_file(source, target)

            with target.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["SAMPLE"], "S1")
        self.assertEqual(rows[0]["AHE"], "12.3")
        self.assertEqual(rows[0]["DAHE"], "0.4")
        self.assertEqual(rows[0]["AFT"], "25.1")
        self.assertEqual(rows[0]["DAFT"], "1.2")
        self.assertEqual(rows[1]["SAMPLE"], "S2")
        self.assertEqual(rows[1]["ZHE"], "45.0")
        self.assertEqual(rows[1]["DZHE"], "2.0")

    def test_compareage_csv_is_accepted_for_prediction_matching(self):
        """产品验收：Pecube CompareAGE.csv 优先驱动 thermo loss，不被 Ages000 常数场覆盖。"""
        parsed = PecubeParsedOutput(
            output_dir=Path("."),
            csv_files={
                "Ages000.csv": [
                    {"Longitude": "103.5", "Latitude": "31.2", "Height": "1200", "HeApatite": "2.0", "HeZircon": "2.0"},
                    {"Longitude": "103.6", "Latitude": "31.3", "Height": "2300", "HeApatite": "2.0", "HeZircon": "2.0"},
                ],
                "CompareAGE.csv": [
                    {"LON": "103.5", "LAT": "31.2", "AHEPRED": "11.0", "AFTPRED": "23.0", "ZHEPRED": "41.0"},
                    {"LON": "103.6", "LAT": "31.3", "AHEPRED": "18.0", "AFTPRED": "29.0", "ZHEPRED": "46.5"},
                ]
            },
            files=["CompareAGE.csv"],
        )
        observations = [
            ThermochronologyObservation("S1", 103.5, 31.2, 1200.0, "AHe", 12.3, 0.4),
            ThermochronologyObservation("S2", 103.6, 31.3, 2300.0, "ZHe", 45.0, 2.0),
        ]

        predictions = predictions_from_parsed(parsed, observations)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0].pecube_column, "AHEPRED")
        self.assertAlmostEqual(predictions[0].predicted_age, 11.0)
        self.assertEqual(predictions[1].pecube_column, "ZHEPRED")
        self.assertAlmostEqual(predictions[1].predicted_age, 46.5)

    def test_effective_sigma_floor_limits_young_age_overweighting(self):
        """产品验收：年轻热年代学年龄不能因极小 sigma 在 loss 中获得失真权重。"""
        parsed = PecubeParsedOutput(
            output_dir=Path("."),
            csv_files={
                "CompareAGE.csv": [
                    {"LON": "103.5", "LAT": "31.2", "AHEPRED": "1.8"},
                ]
            },
            files=["CompareAGE.csv"],
        )
        observations = [
            ThermochronologyObservation("Y1", 103.5, 31.2, 1200.0, "AHe", 2.3, 0.1),
        ]

        predictions = predictions_from_parsed(
            parsed,
            observations,
            sigma_min=0.5,
            sigma_relative=0.1,
        )

        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0].sigma, 0.5)
        self.assertAlmostEqual(predictions[0].residual, -0.5)
        self.assertAlmostEqual(predictions[0].normalized_residual, -1.0)

    def test_thermo_loss_scaling_preserves_signal_above_scale(self):
        """产品验收：热年代学 raw loss 超过 scale 时仍保留候选差异，不能全部截断为 1。"""
        low = scaled_unit_loss(5.397, 5.0)
        high = scaled_unit_loss(5.538, 5.0)

        self.assertGreater(low, 0.0)
        self.assertLess(high, 1.0)
        self.assertGreater(high, low)

    def test_project_builder_writes_full_multistep_topography_history(self):
        """产品验收：FastScape 多步地形历史必须完整写入 Pecube project。"""
        topographies = [np.full((2, 3), value, dtype=float) for value in range(5)]
        uplifts = [np.full((2, 3), 0.5 + value * 0.1, dtype=float) for value in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = PecubeProjectBuilder(
                PecubeProjectConfig(
                    sea_level_temperature=15.0,
                    lapse_rate=6.5,
                    thickness=35.0,
                    basal_temperature=700.0,
                    thermal_diffusivity=25.0,
                )
            )
            project = builder.build(
                project_dir=Path(tmpdir) / "PGB01",
                topography_series=topographies,
                uplift_series=uplifts,
            )
            input_text = project.input_file.read_text(encoding="utf-8")
            topo_files = sorted(project.dataset_dir.glob("topo*"))
            uplift_files = sorted(project.dataset_dir.glob("uplift*"))
            temp_files = sorted(project.dataset_dir.glob("temp*"))
            temp4 = np.loadtxt(project.dataset_dir / "temp4")

        self.assertIn("ntime = 4", input_text)
        self.assertIn("time_topo4", input_text)
        self.assertIn("thickness = 35.0", input_text)
        self.assertIn("basal_temperature = 700.0", input_text)
        self.assertIn("thermal_diffusivity = 25.0", input_text)
        self.assertIn("sea_level_temperature = 15.0", input_text)
        self.assertIn("lapse_rate = 6.5", input_text)
        self.assertIn("nfault = 0", input_text)
        self.assertNotIn("npoint1 = -1", input_text)
        self.assertEqual([path.name for path in topo_files], ["topo0", "topo1", "topo2", "topo3", "topo4"])
        self.assertEqual(len(uplift_files), 5)
        self.assertEqual(len(temp_files), 5)
        self.assertTrue(np.allclose(temp4, 15.0 - 6.5 * 0.004))

    def test_project_builder_can_explicitly_write_uniform_velocity_field(self):
        """产品验收：只有显式开启时，才给 Pecube 写额外全域速度场。"""
        topographies = [np.zeros((2, 2), dtype=float), np.ones((2, 2), dtype=float)]
        uplifts = [np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)]

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = PecubeProjectBuilder(
                PecubeProjectConfig(include_uniform_velocity_field=True, velocity_km_per_myr=0.8)
            )
            project = builder.build(
                project_dir=Path(tmpdir) / "PGB01",
                topography_series=topographies,
                uplift_series=uplifts,
            )
            input_text = project.input_file.read_text(encoding="utf-8")

        self.assertIn("nfault = 1", input_text)
        self.assertIn("npoint1 = -1", input_text)
        self.assertIn("velo1_1 = 0.800000", input_text)

    def test_surface_temperature_uses_positive_lapse_rate_with_elevation(self):
        """产品验收：未显式提供 temp* 时，Pecube 地表温度必须随地形升高而降低。"""
        topography = np.array([[0.0, 1000.0], [2000.0, 3000.0]])
        config = PecubeProjectConfig(sea_level_temperature=15.0, lapse_rate=6.5)

        temperature = surface_temperature_from_topography(topography, config)

        self.assertTrue(np.allclose(temperature, [[15.0, 8.5], [2.0, -4.5]]))


if __name__ == "__main__":
    unittest.main()
