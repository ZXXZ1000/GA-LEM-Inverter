import unittest

import numpy as np

from ga_lem_inverter.pipeline.forward_model import align_fastscape_inputs, align_model_field, run_fastscape_series


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


if __name__ == "__main__":
    unittest.main()
