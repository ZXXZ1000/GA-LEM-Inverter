import unittest

from ga_lem_inverter.config import UserConfigError
from ga_lem_inverter.workflows.main_inversion import (
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

    def test_rotated_dem_blocks_pecube_auto_coordinates(self):
        """产品验收：旋转 DEM 不能和 Pecube 自动真实坐标混用，避免样品坐标被错误解释。"""
        with self.assertRaisesRegex(UserConfigError, "旋转后的 DEM 矩阵没有可靠的地理 transform"):
            validate_rotation_spatial_constraints(
                rotation_angle=12.5,
                pecube_enabled=True,
                pecube_spatial_mode="auto",
            )

    def test_rotated_terrain_only_records_matrix_spatial_mode(self):
        """产品验收：terrain-only 旋转优化可继续，但结果必须标记为矩阵坐标解释。"""
        metrics = validate_rotation_spatial_constraints(
            rotation_angle=12.5,
            pecube_enabled=False,
            pecube_spatial_mode="auto",
        )

        self.assertTrue(metrics["dem_rotated"])
        self.assertEqual(metrics["spatial_reference_mode"], "rotated_matrix")

    def test_unrotated_dem_remains_georeferenced(self):
        """产品验收：未旋转 DEM 保持真实 DEM transform 坐标语义。"""
        metrics = validate_rotation_spatial_constraints(
            rotation_angle=0.0,
            pecube_enabled=True,
            pecube_spatial_mode="auto",
        )

        self.assertFalse(metrics["dem_rotated"])
        self.assertEqual(metrics["spatial_reference_mode"], "dem_georeferenced")


if __name__ == "__main__":
    unittest.main()
