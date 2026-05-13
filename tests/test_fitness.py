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


if __name__ == "__main__":
    unittest.main()
