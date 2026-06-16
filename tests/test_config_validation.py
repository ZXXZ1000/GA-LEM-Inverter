import tempfile
import textwrap
import unittest
from pathlib import Path

from ga_lem_inverter.config import UserConfigError, load_app_config


class ConfigValidationAcceptanceTests(unittest.TestCase):
    def _write_config(self, text: str) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.ini"
        path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
        return path

    def test_pecube_time_window_must_match_fastscape_time_window(self):
        """产品验收：Pecube 热史时间窗不能和 FastScape 模拟时间静默错配。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6

            [Optimization]
            scale_factor = 8

            [Pecube]
            enabled = auto
            sample_observations = ./demo/data/demo1/demo_thermo_samples.csv
            total_time_myr = 1.0
            """
        )

        with self.assertRaisesRegex(UserConfigError, "total_time_myr.*time_total"):
            load_app_config(config_path)

    def test_aligned_pecube_time_window_is_accepted(self):
        """产品验收：Pecube total_time_myr 与 time_total/1e6 对齐时配置可直接加载。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6

            [Optimization]
            scale_factor = 8

            [Pecube]
            enabled = auto
            sample_observations = ./demo/data/demo1/demo_thermo_samples.csv
            total_time_myr = 2.0
            """
        )

        app_config = load_app_config(config_path)

        self.assertEqual(app_config.mode, "main")
        self.assertEqual(app_config.parser.getfloat("Pecube", "total_time_myr"), 2.0)

    def test_disabled_pecube_does_not_require_time_alignment(self):
        """产品验收：关闭 Pecube 后，旧 terrain-only 配置不会因为热史字段而失败。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6

            [Optimization]
            scale_factor = 8

            [Pecube]
            enabled = false
            sample_observations = none
            total_time_myr = 1.0
            """
        )

        app_config = load_app_config(config_path)

        self.assertEqual(app_config.mode, "main")

    def test_rainfall_factor_must_be_positive_when_configured(self):
        """产品验收：降雨量系数写进 config 后，非法值在启动阶段直接提示。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6
            rainfall_factor = 0

            [Optimization]
            scale_factor = 8
            """
        )

        with self.assertRaisesRegex(UserConfigError, "rainfall_factor.*正数"):
            load_app_config(config_path)

    def test_initial_topography_mode_must_be_explicit_supported_value(self):
        """产品验收：初始地形模式写错时，启动阶段直接给用户可读错误。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6
            initial_topography = zero

            [Optimization]
            scale_factor = 8
            """
        )

        with self.assertRaisesRegex(UserConfigError, "initial_topography.*random.*flat"):
            load_app_config(config_path)

    def test_flat_initial_topography_config_is_accepted(self):
        """产品验收：用户可以明确指定 flat 初始地形及海拔。"""
        config_path = self._write_config(
            """
            [Run]
            mode = main

            [Data]
            terrain_path = ./demo/data/demo1/demo_dem.tif
            output_path = ./demo/outputs

            [Model]
            time_total = 2e6
            initial_topography = flat
            initial_elevation = 0.0
            initial_topography_seed = 7

            [Optimization]
            scale_factor = 8
            """
        )

        app_config = load_app_config(config_path)

        self.assertEqual(app_config.parser.get("Model", "initial_topography"), "flat")
        self.assertEqual(app_config.parser.getfloat("Model", "initial_elevation"), 0.0)
        self.assertEqual(app_config.parser.getint("Model", "initial_topography_seed"), 7)


if __name__ == "__main__":
    unittest.main()
