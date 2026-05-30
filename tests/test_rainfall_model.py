import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from ga_lem_inverter.pipeline.rainfall import (
    RainfallConfig,
    evaluate_rainfall,
    load_rainfall_function,
    preview_rainfall_fields,
    rainfall_from_config,
)


class RainfallModelAcceptanceTests(unittest.TestCase):
    def test_user_python_rainfall_function_returns_matrix(self):
        """产品验收：用户脚本函数 p=f(x,y,z,t) 输出的矩阵可作为 runoff 场。"""
        def rainfall(x, y, z, t_ma, params):
            return 1.0 + x / np.nanmax(x) + 0.1 * t_ma + 0.001 * (z - np.nanmean(z))

        shape = (4, 5)
        x, y = np.meshgrid(np.linspace(0, 1000, shape[1]), np.linspace(0, 900, shape[0]))
        z = np.linspace(100, 500, np.prod(shape)).reshape(shape)

        field = evaluate_rainfall(
            RainfallConfig(mode="python", function=rainfall, params={}),
            x=x,
            y=y,
            z=z,
            elapsed_years=5.0e6,
            total_time_years=10.0e6,
        )

        self.assertEqual(field.shape, shape)
        self.assertTrue(np.isfinite(field).all())
        self.assertGreater(field.max(), field.min())

    def test_invalid_rainfall_output_is_rejected(self):
        """产品验收：用户脚本返回非正或 NaN 降雨时必须直接报错。"""
        def rainfall(x, y, z, t_ma, params):
            return np.zeros_like(z)

        shape = (3, 3)
        with self.assertRaisesRegex(ValueError, "必须全部为正"):
            evaluate_rainfall(
                RainfallConfig(mode="python", function=rainfall, params={}),
                x=np.zeros(shape),
                y=np.zeros(shape),
                z=np.ones(shape),
                elapsed_years=0.0,
                total_time_years=1.0e6,
            )

    def test_rainfall_output_shape_must_match_dem_grid(self):
        """产品验收：用户脚本返回矩阵必须和 DEM/Ksp 运行网格 shape 一致。"""
        def rainfall(x, y, z, t_ma, params):
            return np.ones((z.shape[0] + 1, z.shape[1]), dtype=float)

        shape = (3, 4)
        with self.assertRaisesRegex(ValueError, "shape .*目标网格"):
            evaluate_rainfall(
                RainfallConfig(mode="python", function=rainfall, params={}),
                x=np.zeros(shape),
                y=np.zeros(shape),
                z=np.ones(shape),
                elapsed_years=0.0,
                total_time_years=1.0e6,
            )

    def test_config_loads_user_script_relative_to_config_dir(self):
        """产品验收：[Rainfall] module_path 可相对配置文件目录加载用户脚本。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rainfall_model.py").write_text(
                textwrap.dedent(
                    """
                    import numpy as np

                    def rainfall(x, y, z, t_ma, params):
                        return np.ones_like(z) * float(params.get("base", 1.2))
                    """
                ),
                encoding="utf-8",
            )
            config_text = textwrap.dedent(
                """
                [Model]
                rainfall_factor = 1.0

                [Rainfall]
                mode = python
                module_path = ./rainfall_model.py
                function = rainfall
                base = 1.7
                """
            )
            import configparser

            parser = configparser.ConfigParser()
            parser.read_string(config_text)
            rainfall = rainfall_from_config(parser, base_dir=root)

            field = preview_rainfall_fields(
                rainfall,
                shape=(2, 2),
                spacing=100.0,
                elevation=np.ones((2, 2)),
                total_time_years=1.0e6,
                times_ma=(0.0,),
            )[0.0]

        self.assertTrue(np.allclose(field, 1.7))
        self.assertEqual(rainfall.mode, "python")

    def test_rainfall_preview_requires_elevation_shape_to_match_grid(self):
        """产品验收：预览图使用的 elevation 必须和 FastScape 运行网格一致。"""
        with self.assertRaisesRegex(ValueError, "elevation shape .*目标网格"):
            preview_rainfall_fields(
                RainfallConfig(mode="uniform", factor=1.0),
                shape=(2, 3),
                spacing=100.0,
                elevation=np.ones((3, 2)),
                total_time_years=1.0e6,
                times_ma=(0.0,),
            )

    def test_load_rainfall_function_reports_missing_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rainfall_model.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(AttributeError, "找不到可调用函数"):
                load_rainfall_function(path)

    def test_python_rainfall_requires_dynamic_true_for_now(self):
        import configparser

        parser = configparser.ConfigParser()
        parser.read_string(
            textwrap.dedent(
                """
                [Model]
                rainfall_factor = 1.0

                [Rainfall]
                mode = python
                module_path = ./rainfall_model.py
                dynamic = false
                """
            )
        )

        with self.assertRaisesRegex(ValueError, "dynamic=true"):
            rainfall_from_config(parser, base_dir=Path.cwd())


if __name__ == "__main__":
    unittest.main()
