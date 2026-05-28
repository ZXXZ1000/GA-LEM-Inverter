import tempfile
import unittest
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from ga_lem_inverter.integrations.pecube_fitness import PecubeFitnessEvaluator, ThermochronologyObservation, ThermochronologyPrediction
from ga_lem_inverter.outputs import create_run_context
from ga_lem_inverter.pipeline.visualization import flipped_display_array, oriented_display_array
from ga_lem_inverter.workflows.main_inversion import _thermo_observation_pixels_for_profile
import configparser


class VisualizationOrientationTests(unittest.TestCase):
    def test_flipped_display_array_matches_demo_alignment_rule(self):
        """产品验收：旧的像素旋转 demo 仍保持上下+左右双翻。"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        expected = np.array([[6, 5, 4], [3, 2, 1]], dtype=float)
        np.testing.assert_array_equal(flipped_display_array(matrix), expected)

    def test_dem_and_uplift_keep_same_display_corner_after_flip(self):
        """产品验收：旧 demo 的 DEM、Ksp、uplift 必须共用同一双翻方向。"""
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

    def test_unrotated_georeferenced_display_is_not_mirrored(self):
        """产品验收：未旋转 north-up DEM 显示不能被上下+左右镜像。"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        np.testing.assert_array_equal(oriented_display_array(matrix, rotated=False), matrix)

    def test_rotated_display_keeps_legacy_demo_alignment(self):
        """产品验收：只有旧的像素旋转视图仍沿用 demo 双翻规则。"""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        np.testing.assert_array_equal(
            oriented_display_array(matrix, rotated=True),
            flipped_display_array(matrix),
        )

    def test_pecube_pixel_background_uses_unrotated_orientation_by_default(self):
        """产品验收：Pecube 诊断底图默认不镜像未旋转 DEM。"""
        cfg = configparser.ConfigParser()
        cfg.read_dict({"Pecube": {"enabled": "false"}, "Fitness": {}})

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.ini"
            config_path.write_text("[Data]\noutput_path = ./outputs\n", encoding="utf-8")
            context = create_run_context(config_path, cfg, "main")
            target = np.arange(6, dtype=float).reshape(2, 3)
            evaluator = PecubeFitnessEvaluator.from_config(
                config=cfg,
                context=context,
                target_dem=target,
                ksp=np.ones_like(target),
                model_params={},
            )

        np.testing.assert_array_equal(evaluator.target_dem_pixel, target)

    def test_pecube_pixel_background_can_use_rotated_demo_orientation(self):
        """产品验收：显式要求 legacy 视图时，Pecube 底图仍可双翻。"""
        cfg = configparser.ConfigParser()
        cfg.read_dict({"Pecube": {"enabled": "false"}, "Fitness": {}})

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.ini"
            config_path.write_text("[Data]\noutput_path = ./outputs\n", encoding="utf-8")
            context = create_run_context(config_path, cfg, "main")
            target = np.arange(6, dtype=float).reshape(2, 3)
            evaluator = PecubeFitnessEvaluator.from_config(
                config=cfg,
                context=context,
                target_dem=target,
                ksp=np.ones_like(target),
                model_params={"display_rotated": True},
            )

        np.testing.assert_array_equal(evaluator.target_dem_pixel, flipped_display_array(target))

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
                model_params={"display_rotated": True},
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

    def test_loaded_observations_project_to_dem_pixel_overlay(self):
        """产品验收：original/rotated DEM 诊断图能把已读取样品点投回像素位置。"""
        observation = ThermochronologyObservation(
            sample_id="S1",
            x=87.0,
            y=43.0,
            elevation=100.0,
            system="AHe",
            observed_age=5.0,
            sigma=0.5,
        )
        profile = {
            "crs": "EPSG:4326",
            "transform": from_origin(80.0, 50.0, 1.0, 1.0),
        }

        pixels = _thermo_observation_pixels_for_profile([observation], profile, (20, 20))

        self.assertEqual(len(pixels), 1)
        self.assertAlmostEqual(pixels[0]["x"], 6.5)
        self.assertAlmostEqual(pixels[0]["y"], 6.5)
        self.assertEqual(pixels[0]["sample_id"], "S1")


if __name__ == "__main__":
    unittest.main()
