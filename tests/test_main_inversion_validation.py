import unittest

from ga_lem_inverter.config import UserConfigError
from ga_lem_inverter.workflows.main_inversion import validate_low_resolution_shape


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


if __name__ == "__main__":
    unittest.main()
