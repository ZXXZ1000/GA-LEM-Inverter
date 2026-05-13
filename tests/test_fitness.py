import unittest
from unittest import mock

import numpy as np

from ga_lem_inverter.pipeline import fitness


class FitnessTests(unittest.TestCase):
    def setUp(self):
        fitness._FEATURE_CACHE.clear()

    def test_terrain_similarity_caches_target_features(self):
        target = np.arange(64 * 64, dtype=float).reshape(64, 64)
        generated1 = target.copy()
        generated1[0, 0] += 100.0
        generated2 = target.copy()
        generated2[-1, -1] -= 100.0
        original_extract = fitness.extract_terrain_features
        calls = {"total": 0}

        def wrapped_extract(dem, resolution=347.4, smooth_radius=3):
            calls["total"] += 1
            return original_extract(dem, resolution, smooth_radius)

        with mock.patch.object(fitness, "extract_terrain_features", side_effect=wrapped_extract):
            fitness.terrain_similarity(target, generated1, use_lpips=False)
            fitness.terrain_similarity(target, generated2, use_lpips=False)

        self.assertEqual(calls["total"], 3)

    def test_terrain_similarity_is_always_unit_interval(self):
        """产品验收：地形相似度必须稳定落在 0-1，不能把超界值传给 GA 造成假失败。"""
        target = np.arange(64 * 64, dtype=float).reshape(64, 64)
        generated = np.flipud(target) * 100.0

        similarity = fitness.terrain_similarity(target, generated, use_lpips=False)

        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_terrain_similarity_handles_flat_fields_without_nan(self):
        """产品验收：平坦 DEM 或退化候选地形不能产生 NaN/Inf。"""
        target = np.ones((64, 64), dtype=float)
        generated = np.zeros((64, 64), dtype=float)

        similarity = fitness.terrain_similarity(target, generated, use_lpips=False)

        self.assertTrue(np.isfinite(similarity))
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


if __name__ == "__main__":
    unittest.main()
