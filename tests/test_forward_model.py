import unittest
from unittest.mock import patch

import numpy as np

from ga_lem_inverter.pipeline.forward_model import (
    align_fastscape_inputs,
    fastscape_output_times,
    align_model_field,
    normalize_stage_multipliers,
    run_fastscape_series,
    run_fastscape_time_scaled_series,
    validate_rainfall_factor,
    stage_edges_from_ma,
    stage_index_for_elapsed_time,
)
from ga_lem_inverter.pipeline.rainfall import RainfallConfig


class ForwardModelAcceptanceTests(unittest.TestCase):
    def test_fastscape_inputs_auto_align_ksp_and_uplift_to_dem_grid(self):
        """产品验收：Ksp/uplift 和 DEM 网格轻微错配时，系统应自动对齐到运行网格。"""
        ksp = np.arange(12, dtype=float).reshape(3, 4)
        uplift = np.linspace(0.1, 1.0, 20, dtype=float).reshape(4, 5)

        aligned_ksp, aligned_uplift = align_fastscape_inputs(ksp, uplift, x_size=5, y_size=4)

        self.assertEqual(aligned_ksp.shape, (4, 5))
        self.assertEqual(aligned_uplift.shape, (4, 5))
        self.assertGreaterEqual(aligned_ksp.min(), ksp.min())
        self.assertLessEqual(aligned_ksp.max(), ksp.max())
        self.assertGreaterEqual(aligned_uplift.min(), uplift.min())
        self.assertLessEqual(aligned_uplift.max(), uplift.max())

    def test_field_alignment_rejects_non_2d_manual_input(self):
        """产品验收：用户手动输入的非法 Ksp 场必须给出明确错误，而不是生成不可解释结果。"""
        with self.assertRaisesRegex(ValueError, "二维数组"):
            align_model_field(np.ones((2, 2, 2), dtype=float), (4, 4), label="Ksp")

    def test_fastscape_series_returns_requested_time_frames(self):
        """产品验收：Pecube 耦合需要真实 FastScape 地形时间序列，而不是只有最终地形。"""
        shape = (5, 5)
        series = run_fastscape_series(
            k_sp=np.ones(shape, dtype=float) * 1.0e-6,
            uplift=np.ones(shape, dtype=float) * 0.1,
            k_diff=0.1,
            x_size=shape[1],
            y_size=shape[0],
            spacing=1000.0,
            time_total=1000.0,
            output_steps=3,
        )

        self.assertEqual(series.shape, (3, *shape))
        self.assertTrue(np.isfinite(series).all())

    def test_fastscape_series_accepts_official_runoff_rainfall_factor(self):
        """产品验收：降雨量系数必须走 FastScape FlowAccumulator.runoff，而不是改写 Ksp。"""
        shape = (5, 5)
        series = run_fastscape_series(
            k_sp=np.ones(shape, dtype=float) * 1.0e-6,
            uplift=np.ones(shape, dtype=float) * 0.1,
            k_diff=0.1,
            x_size=shape[1],
            y_size=shape[0],
            spacing=1000.0,
            time_total=1000.0,
            output_steps=3,
            rainfall_factor=2.0,
        )

        self.assertEqual(series.shape, (3, *shape))
        self.assertTrue(np.isfinite(series).all())

    def test_rainfall_factor_is_passed_as_runoff_without_rewriting_ksp(self):
        """产品验收：降雨系数只能进入 FastScape runoff，不能折进 Ksp。"""
        shape = (5, 5)
        ksp = np.ones(shape, dtype=float) * 1.0e-6
        captured = {}

        class DummyXsimlab:
            def run(self, *, model):
                class DummyElevation:
                    values = np.zeros((3, *shape), dtype=float)

                class DummyDataset:
                    topography__elevation = DummyElevation()

                return DummyDataset()

        class DummySetup:
            xsimlab = DummyXsimlab()

        def fake_create_setup(**kwargs):
            captured.update(kwargs)
            return DummySetup()

        with patch("ga_lem_inverter.pipeline.forward_model.xs.create_setup", side_effect=fake_create_setup):
            series = run_fastscape_series(
                k_sp=ksp,
                uplift=np.ones(shape, dtype=float) * 0.1,
                k_diff=0.1,
                x_size=shape[1],
                y_size=shape[0],
                spacing=1000.0,
                time_total=1000.0,
                output_steps=3,
                rainfall_factor=2.0,
            )

        input_vars = captured["input_vars"]
        self.assertEqual(series.shape, (3, *shape))
        self.assertEqual(input_vars["drainage__runoff"], 2.0)
        self.assertTrue(np.array_equal(input_vars["spl__k_coef"], ksp))

    def test_python_rainfall_model_runs_inside_fastscape(self):
        """产品验收：用户 Python 降雨函数应在 FastScape step 内生成 runoff 场。"""
        shape = (5, 5)

        def rainfall(x, y, z, t_ma, params):
            return 1.0 + 0.5 * x / max(float(np.nanmax(x)), 1.0)

        series = run_fastscape_series(
            k_sp=np.ones(shape, dtype=float) * 1.0e-6,
            uplift=np.ones(shape, dtype=float) * 0.1,
            k_diff=0.1,
            x_size=shape[1],
            y_size=shape[0],
            spacing=1000.0,
            time_total=1000.0,
            output_steps=3,
            rainfall_model=RainfallConfig(mode="python", function=rainfall, params={}),
        )

        self.assertEqual(series.shape, (3, *shape))
        self.assertTrue(np.isfinite(series).all())

    def test_rainfall_factor_must_be_positive(self):
        """产品验收：非法降雨/径流系数必须给明确错误。"""
        self.assertEqual(validate_rainfall_factor(1.5), 1.5)
        with self.assertRaisesRegex(ValueError, "rainfall_factor 必须为正数"):
            validate_rainfall_factor(0.0)

    def test_fastscape_output_times_match_series_frames(self):
        """产品验收：地形演化图必须使用真实输出时间标签，而不是 frame 1/2/3。"""
        times = fastscape_output_times(10.0e6, 6)

        self.assertEqual(times.shape, (6,))
        self.assertGreater(times[0], 0.0)
        self.assertAlmostEqual(float(times[-1]), 10.0e6)
        ma_before_present = (10.0e6 - times) / 1.0e6
        self.assertAlmostEqual(float(ma_before_present[-1]), 0.0)
        self.assertGreater(float(ma_before_present[0]), 0.0)

    def test_stage_times_convert_from_geologic_ma_to_elapsed_fastscape_years(self):
        """产品验收：10,6,3,0 Ma 必须转换成 FastScape 从 0 开始的连续时间边界。"""
        edges = stage_edges_from_ma([10, 6, 3, 0], time_total_years=10e6)

        self.assertTrue(np.allclose(edges, [0.0, 4e6, 7e6, 10e6]))
        self.assertEqual(stage_index_for_elapsed_time(0.0, edges), 0)
        self.assertEqual(stage_index_for_elapsed_time(4e6, edges), 1)
        self.assertEqual(stage_index_for_elapsed_time(9.9e6, edges), 2)

    def test_stage_multipliers_normalize_to_time_weighted_mean_one(self):
        """产品验收：倍率归一化后，U_base 仍代表整个时间窗的平均 uplift。"""
        edges = np.array([0.0, 4e6, 10e6])
        normalized = normalize_stage_multipliers([0.5, 1.5], edges, enabled=True)
        durations = np.diff(edges)

        self.assertAlmostEqual(float(np.sum(normalized * durations) / durations.sum()), 1.0)
        self.assertLess(normalized[0], normalized[1])

    def test_time_scaled_fastscape_returns_stage_specific_uplift_series(self):
        """产品验收：FastScape 输出给 Pecube 的 uplift_series 必须随阶段倍率变化。"""
        shape = (5, 5)
        result = run_fastscape_time_scaled_series(
            k_sp=np.ones(shape, dtype=float) * 1.0e-6,
            uplift=np.ones(shape, dtype=float) * 0.1,
            k_diff=0.1,
            x_size=shape[1],
            y_size=shape[0],
            spacing=1000.0,
            time_total=1000.0,
            output_steps=4,
            stage_edges_years=[0.0, 500.0, 1000.0],
            stage_multipliers=[0.5, 1.5],
        )

        self.assertEqual(result.topography_series.shape, (4, *shape))
        self.assertEqual(result.uplift_series.shape, (4, *shape))
        self.assertTrue(np.isfinite(result.topography_series).all())
        self.assertTrue(np.allclose(result.uplift_series[0], 0.05))
        self.assertTrue(np.allclose(result.uplift_series[-1], 0.15))


if __name__ == "__main__":
    unittest.main()
