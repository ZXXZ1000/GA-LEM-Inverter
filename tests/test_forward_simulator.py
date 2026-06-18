"""Forward simulator smoke tests.

These tests exercise the config / erosion-metrics layers WITHOUT actually
running FastScape (which is heavy and would slow down CI). The FastScape path
is covered by demo/forward/* and the existing test_forward_model.py suite for
the underlying ga_lem_inverter functions we reuse.
"""

from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from forward_simulator.config import load_forward_config
from forward_simulator.erosion_metrics import compute_erosion_fields, summarize_metrics


class ConfigParsingTests(unittest.TestCase):
    def test_uniform_uplift_and_uniform_rainfall_round_trip(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dem = tmp_path / "fake_dem.npy"
            np.save(dem, np.zeros((4, 5), dtype=float))
            config_path = tmp_path / "forward.ini"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [Run]
                    output_root = {tmp_path / 'outputs'}

                    [Data]
                    dem_path = {dem}
                    uplift_base_path = none
                    uplift_value = 1.5
                    ksp_value = 1e-5
                    ksp_path = none

                    [Model]
                    time_total = 1e6
                    spacing = 100
                    output_steps = 5
                    boundary_left = fixed_value
                    boundary_right = fixed_value
                    boundary_top = fixed_value
                    boundary_bottom = core
                    area_exp = 0.43
                    slope_exp = 1
                    k_diff = 1.0

                    [UpliftTime]
                    mode = none

                    [Rainfall]
                    mode = uniform
                    value = 1.0

                    [Output]
                    save_cumulative_erosion = true
                    save_mean_erosion_rate = true
                    save_net_uplift = true
                    save_topography_series = true
                    save_uplift_series = true
                    plot_history_grid = true
                    plot_erosion_history = true
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_forward_config(config_path)

            self.assertEqual(cfg.uplift_base_path, None)
            self.assertEqual(cfg.uplift_value, 1.5)
            self.assertEqual(cfg.boundary, ["fixed_value", "fixed_value", "fixed_value", "core"])
            self.assertEqual(cfg.uplift_time.mode, "none")
            self.assertEqual(cfg.rainfall.mode, "uniform")
            self.assertEqual(cfg.rainfall.value, 1.0)

    def test_python_uplift_time_collects_extra_params(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dem = tmp_path / "fake_dem.npy"
            np.save(dem, np.zeros((4, 5), dtype=float))
            module_path = tmp_path / "uplift_fn.py"
            module_path.write_text("def uplift_time(t_yr, x, y, z, params):\n    return 1.0\n", encoding="utf-8")
            config_path = tmp_path / "forward.ini"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [Run]
                    output_root = {tmp_path / 'outputs'}

                    [Data]
                    dem_path = {dem}
                    uplift_value = 1.0
                    ksp_value = 1e-5

                    [Model]
                    time_total = 1e6
                    spacing = 100
                    output_steps = 3
                    boundary_left = fixed_value
                    boundary_right = fixed_value
                    boundary_top = fixed_value
                    boundary_bottom = core

                    [UpliftTime]
                    mode = python
                    module_path = {module_path}
                    function = uplift_time
                    amplitude = 0.4
                    period_ma = 1.0

                    [Rainfall]
                    mode = uniform
                    value = 1.0

                    [Output]
                    save_cumulative_erosion = true
                    save_mean_erosion_rate = true
                    save_net_uplift = true
                    save_topography_series = true
                    save_uplift_series = true
                    plot_history_grid = false
                    plot_erosion_history = false
                    """
                ).strip(),
                encoding="utf-8",
            )
            cfg = load_forward_config(config_path)
            self.assertEqual(cfg.uplift_time.mode, "python")
            self.assertEqual(cfg.uplift_time.params, {"amplitude": "0.4", "period_ma": "1.0"})
            self.assertTrue(Path(cfg.uplift_time.module_path).exists())

    def test_invalid_boundary_raises(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dem = tmp_path / "fake_dem.npy"
            np.save(dem, np.zeros((4, 5), dtype=float))
            config_path = tmp_path / "forward.ini"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    [Run]
                    output_root = {tmp_path / 'outputs'}

                    [Data]
                    dem_path = {dem}
                    uplift_value = 1.0
                    ksp_value = 1e-5

                    [Model]
                    time_total = 1e6
                    spacing = 100
                    output_steps = 3
                    boundary_left = looped
                    boundary_right = looped
                    boundary_top = looped
                    boundary_bottom = looped
                    """
                ).strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fixed_value"):
                load_forward_config(config_path)


class ErosionMetricsTests(unittest.TestCase):
    def test_uniform_uplift_no_change_yields_known_erosion(self):
        # 构造：uplift = 1 mm/yr，地形从 0 → 0（完全被剥蚀），time = 1e6 yr。
        # 累积抬升 = 1mm/yr * 1e-3 * 1e6yr = 1000 m。z(t)-z0 = 0。剥蚀 = 1000 m。
        times = np.array([0.0, 5.0e5, 1.0e6])
        z0 = np.zeros((4, 5))
        topo = np.stack([z0, z0, z0], axis=0)
        uplift = np.ones_like(topo)  # 1 mm/yr

        fields = compute_erosion_fields(
            output_times_years=times,
            topography_series=topo,
            uplift_series_mm_per_yr=uplift,
            initial_dem=z0,
        )

        np.testing.assert_allclose(fields.cumulative_uplift[0], 0.0)
        np.testing.assert_allclose(fields.cumulative_uplift[-1], 1000.0)
        np.testing.assert_allclose(fields.cumulative_erosion[-1], 1000.0)
        np.testing.assert_allclose(fields.mean_erosion_rate[-1], 1.0)  # 1 mm/yr
        np.testing.assert_allclose(fields.net_uplift[-1], 0.0)

    def test_pure_uplift_no_erosion(self):
        # 抬升全部沉积到地形上：z(t) = z0 + 1 mm/yr * t；剥蚀应该 = 0。
        times = np.array([0.0, 5.0e5, 1.0e6])
        z0 = np.full((3, 3), 100.0)
        u_m = 1.0e-3  # mm/yr → m/yr
        topo = np.stack([z0 + u_m * t for t in times], axis=0)
        uplift = np.ones_like(topo)

        fields = compute_erosion_fields(
            output_times_years=times,
            topography_series=topo,
            uplift_series_mm_per_yr=uplift,
            initial_dem=z0,
        )

        np.testing.assert_allclose(fields.cumulative_erosion, 0.0, atol=1e-9)
        np.testing.assert_allclose(fields.mean_erosion_rate, 0.0, atol=1e-9)

    def test_summarize_metrics_shape(self):
        times = np.array([0.0, 1.0e6])
        z0 = np.zeros((2, 2))
        topo = np.stack([z0, z0 + 0.5], axis=0)
        uplift = np.ones_like(topo)

        fields = compute_erosion_fields(
            output_times_years=times,
            topography_series=topo,
            uplift_series_mm_per_yr=uplift,
            initial_dem=z0,
        )
        metrics = summarize_metrics(fields)
        self.assertEqual(metrics["frames"], 2)
        self.assertGreater(metrics["cumulative_erosion_final_max_m"], 0)


if __name__ == "__main__":
    unittest.main()
