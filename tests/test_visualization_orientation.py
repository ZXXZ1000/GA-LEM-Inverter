import tempfile
import unittest
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from ga_lem_inverter.integrations.pecube_fitness import PecubeFitnessEvaluator, ThermochronologyPrediction
from ga_lem_inverter.outputs import create_run_context
from ga_lem_inverter.pipeline.visualization import flipped_display_array
import configparser


class VisualizationOrientationTests(unittest.TestCase):
    def test_flipped_display_array_matches_demo_alignment_rule(self):
        """产品验收：旋转后矩阵显示必须固定为上下+左右双翻。"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        expected = np.array([[6, 5, 4], [3, 2, 1]], dtype=float)
        np.testing.assert_array_equal(flipped_display_array(matrix), expected)

    def test_dem_and_uplift_keep_same_display_corner_after_flip(self):
        """产品验收：DEM、Ksp、uplift 和最终地形图必须使用同一显示方向。"""
        dem = np.zeros((4, 5), dtype=float)
        uplift = np.zeros_like(dem)
        dem[-1, -1] = 3000.0
        uplift[-1, -1] = 1.4

        dem_display = flipped_display_array(dem)
        uplift_display = flipped_display_array(uplift)

        self.assertEqual(np.unravel_index(np.argmax(dem_display), dem_display.shape), (0, 0))
        self.assertEqual(
            np.unravel_index(np.argmax(dem_display), dem_display.shape),
            np.unravel_index(np.argmax(uplift_display), uplift_display.shape),
        )

    def test_pecube_predictions_project_back_to_rotated_dem_pixels(self):
        """产品验收：Pecube 经纬度样品点必须能投回主 DEM 像素坐标视图。"""
        cfg = configparser.ConfigParser()
        cfg.read_dict(
            {
                "Pecube": {
                    "enabled": "false",
                    "spatial_grid": "auto",
                },
                "Fitness": {},
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.ini"
            config_path.write_text("[Data]\noutput_path = ./outputs\n", encoding="utf-8")
            context = create_run_context(config_path, cfg, "main")
            evaluator = PecubeFitnessEvaluator.from_config(
                config=cfg,
                context=context,
                target_dem=np.arange(100, dtype=float).reshape(10, 10),
                ksp=np.ones((10, 10), dtype=float),
                model_params={},
            )
            profile = {
                "crs": "EPSG:32648",
                "transform": from_origin(500000.0, 10000.0, 100.0, 100.0),
                "height": 10,
                "width": 10,
            }
            from ga_lem_inverter.integrations.pecube_fitness import pecube_spatial_adapter_from_dem_profile

            adapter = pecube_spatial_adapter_from_dem_profile(profile, (10, 10))
            self.assertIsNotNone(adapter)
            evaluator.apply_spatial_adapter(adapter)

            x_geo = adapter.grid.lon0 + 4 * adapter.grid.dlon
            y_geo = adapter.grid.lat0 + 6 * adapter.grid.dlat
            predictions = [
                ThermochronologyPrediction(
                    sample_id="S1",
                    x=x_geo,
                    y=y_geo,
                    elevation=100.0,
                    system="AHe",
                    observed_age=5.0,
                    predicted_age=5.5,
                    sigma=0.5,
                    residual=0.5,
                    normalized_residual=1.0,
                    pecube_column="AHEPRED",
                    source_file="CompareAGE.csv",
                )
            ]
            pixels = evaluator.observation_pixels(predictions)

        self.assertEqual(len(pixels), 1)
        self.assertGreaterEqual(pixels[0]["x"], 0.0)
        self.assertGreaterEqual(pixels[0]["y"], 0.0)
        self.assertLessEqual(pixels[0]["x"], 10.0)
        self.assertLessEqual(pixels[0]["y"], 10.0)


if __name__ == "__main__":
    unittest.main()
