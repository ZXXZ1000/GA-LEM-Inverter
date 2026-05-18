import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ga_lem_inverter.config import UserConfigError
from ga_lem_inverter.workflows.main_inversion import (
    create_objective_function,
    validate_low_resolution_shape,
    validate_rotation_spatial_constraints,
)


class MainInversionValidationAcceptanceTests(unittest.TestCase):
    def test_scale_factor_returns_valid_control_grid(self):
        """产品验收：正常 DEM 和 scale_factor 应得到明确的低分辨率控制网格。"""
        self.assertEqual(validate_low_resolution_shape((64, 64), 8), (8, 8))

    def test_scale_factor_too_large_is_user_config_error(self):
        """产品验收：scale_factor 过大时必须给小白用户可读错误，不能进入 0 维 GA。"""
        with self.assertRaisesRegex(UserConfigError, "scale_factor=16"):
            validate_low_resolution_shape((8, 8), 16)

    def test_scale_factor_must_be_positive(self):
        """产品验收：scale_factor 必须是正整数。"""
        with self.assertRaisesRegex(UserConfigError, "必须 >= 1"):
            validate_low_resolution_shape((64, 64), 0)

    def test_rotated_dem_with_pecube_uses_rotated_georeferenced_mode(self):
        """产品验收：旋转 DEM 可以接 Pecube，但必须标记为旋转后的真实空间参考。"""
        metrics = validate_rotation_spatial_constraints(
            rotation_angle=12.5,
            pecube_enabled=True,
            pecube_spatial_mode="auto",
        )

        self.assertTrue(metrics["dem_rotated"])
        self.assertEqual(metrics["spatial_reference_mode"], "rotated_georeferenced")

    def test_rotated_terrain_only_records_rotated_spatial_mode(self):
        """产品验收：terrain-only 旋转优化也必须标记为旋转后的空间参考。"""
        metrics = validate_rotation_spatial_constraints(
            rotation_angle=12.5,
            pecube_enabled=False,
            pecube_spatial_mode="auto",
        )

        self.assertTrue(metrics["dem_rotated"])
        self.assertEqual(metrics["spatial_reference_mode"], "rotated_georeferenced")

    def test_unrotated_dem_remains_georeferenced(self):
        """产品验收：未旋转 DEM 保持真实 DEM transform 坐标语义。"""
        metrics = validate_rotation_spatial_constraints(
            rotation_angle=0.0,
            pecube_enabled=True,
            pecube_spatial_mode="auto",
        )

        self.assertFalse(metrics["dem_rotated"])
        self.assertEqual(metrics["spatial_reference_mode"], "dem_georeferenced")

    def test_pecube_objective_uses_fastscape_series_for_terrain_and_thermo(self):
        """产品验收：启用 Pecube 时 FastScape 序列最后一帧用于地形 loss，完整序列送入热史约束。"""
        class RecordingEvaluator:
            enabled = True

            def __init__(self):
                self.call = None

            def evaluate(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(total_loss=0.37)

        evaluator = RecordingEvaluator()
        series = np.stack(
            [
                np.zeros((2, 2), dtype=float),
                np.ones((2, 2), dtype=float),
                np.full((2, 2), 4.0, dtype=float),
            ]
        )

        with mock.patch(
            "ga_lem_inverter.workflows.main_inversion.interpolate_uplift_cv",
            return_value=np.full((2, 2), 0.5, dtype=float),
        ), mock.patch(
            "ga_lem_inverter.workflows.main_inversion.run_fastscape_series",
            return_value=series,
        ) as series_mock, mock.patch(
            "ga_lem_inverter.workflows.main_inversion.run_fastscape_model",
        ) as model_mock, mock.patch(
            "ga_lem_inverter.workflows.main_inversion.terrain_similarity",
            return_value=0.8,
        ) as similarity_mock:
            objective = create_objective_function(
                resampled_dem=np.ones((2, 2), dtype=float),
                LOW_RES_SHAPE=(1, 1),
                ORIGINAL_SHAPE=(2, 2),
                Ksp=np.ones((2, 2), dtype=float),
                D_DIFF=0.1,
                row=2,
                col=2,
                spacing=100.0,
                time_step_num=3,
                total_simulation_time=1000.0,
                terrain_resolution=100.0,
                feature_smooth_radius=1,
                pecube_evaluator=evaluator,
                pecube_time_steps=3,
            )
            loss = objective(np.array([0.5]))

        self.assertEqual(loss, 0.37)
        series_mock.assert_called_once()
        model_mock.assert_not_called()
        similarity_mock.assert_called_once()
        self.assertTrue(np.array_equal(similarity_mock.call_args.kwargs["matrix2"], series[-1]))
        self.assertTrue(np.array_equal(evaluator.call["topography_series"], series))
        self.assertEqual(evaluator.call["uplift_series"].shape, series.shape)
        self.assertNotIn("temperature_series", evaluator.call)

    def test_pecube_objective_passes_stage_specific_uplift_series_to_thermo_constraint(self):
        """产品验收：启用 uplift history 后，Pecube 收到的 uplift_series 不能是同一场重复。"""
        class RecordingEvaluator:
            enabled = True

            def __init__(self):
                self.call = None

            def evaluate(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(total_loss=0.25)

        evaluator = RecordingEvaluator()
        topography_series = np.stack([
            np.zeros((2, 2), dtype=float),
            np.ones((2, 2), dtype=float),
        ])
        uplift_series = np.stack([
            np.full((2, 2), 0.25, dtype=float),
            np.full((2, 2), 0.75, dtype=float),
        ])

        with mock.patch(
            "ga_lem_inverter.workflows.main_inversion.interpolate_uplift_cv",
            return_value=np.full((2, 2), 0.5, dtype=float),
        ), mock.patch(
            "ga_lem_inverter.workflows.main_inversion._run_candidate_fastscape_series",
            return_value=(topography_series, uplift_series, np.array([0.5, 1.5]), np.array([500.0, 1000.0])),
        ) as runner_mock, mock.patch(
            "ga_lem_inverter.workflows.main_inversion.terrain_similarity",
            return_value=0.9,
        ):
            objective = create_objective_function(
                resampled_dem=np.ones((2, 2), dtype=float),
                LOW_RES_SHAPE=(1, 1),
                ORIGINAL_SHAPE=(2, 2),
                Ksp=np.ones((2, 2), dtype=float),
                D_DIFF=0.1,
                row=2,
                col=2,
                spacing=100.0,
                time_step_num=3,
                total_simulation_time=1000.0,
                terrain_resolution=100.0,
                feature_smooth_radius=1,
                pecube_evaluator=evaluator,
                pecube_time_steps=2,
                uplift_history={
                    "enabled": True,
                    "stage_edges_years": np.array([0.0, 500.0, 1000.0]),
                    "stage_count": 2,
                    "normalize_time_weighted_mean": False,
                },
            )
            loss = objective({"uplift": np.array([0.5]), "stage_multipliers": np.array([0.5, 1.5])})

        self.assertEqual(loss, 0.25)
        runner_mock.assert_called_once()
        self.assertTrue(np.array_equal(evaluator.call["uplift_series"], uplift_series))
        self.assertFalse(np.array_equal(evaluator.call["uplift_series"][0], evaluator.call["uplift_series"][1]))

    def test_objective_passes_configured_fastscape_parameters(self):
        """产品验收：主反演 objective 必须使用 config.ini 中的 FastScape 边界和指数。"""
        with mock.patch(
            "ga_lem_inverter.workflows.main_inversion.interpolate_uplift_cv",
            return_value=np.full((2, 2), 0.5, dtype=float),
        ), mock.patch(
            "ga_lem_inverter.workflows.main_inversion.run_fastscape_model",
            return_value=np.ones((2, 2), dtype=float),
        ) as model_mock, mock.patch(
            "ga_lem_inverter.workflows.main_inversion.terrain_similarity",
            return_value=0.8,
        ):
            objective = create_objective_function(
                resampled_dem=np.ones((2, 2), dtype=float),
                LOW_RES_SHAPE=(1, 1),
                ORIGINAL_SHAPE=(2, 2),
                Ksp=np.ones((2, 2), dtype=float),
                D_DIFF=0.1,
                row=2,
                col=2,
                spacing=100.0,
                time_step_num=3,
                total_simulation_time=1000.0,
                terrain_resolution=100.0,
                feature_smooth_radius=1,
                boundary_status=["fixed_value", "fixed_value", "fixed_value", "core"],
                area_exp=0.42,
                slope_exp=1.3,
            )
            objective(np.array([0.5]))

        self.assertEqual(model_mock.call_args.kwargs["boundary_status"], ["fixed_value", "fixed_value", "fixed_value", "core"])
        self.assertEqual(model_mock.call_args.kwargs["area_exp"], 0.42)
        self.assertEqual(model_mock.call_args.kwargs["slope_exp"], 1.3)


if __name__ == "__main__":
    unittest.main()
