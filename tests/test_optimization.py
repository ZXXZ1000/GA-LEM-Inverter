import unittest

import numpy as np

from ga_lem_inverter.pipeline.optimization import MyGA, optimize_uplift_ga


class MyGATests(unittest.TestCase):
    def test_best_x_matches_sorted_best_fitness_before_population_mutation(self):
        """最优分数必须和排序后的最优个体对应，不能从旧 self.X 里错取。"""
        target = np.array([2, 2, 2])

        def objective(x):
            return float(np.sum((x - target) ** 2))

        ga = MyGA(
            func=objective,
            n_dim=3,
            size_pop=3,
            max_iter=1,
            prob_mut=0,
            lb=0,
            ub=9,
            n_jobs=1,
        )
        ga.Chrom = np.array(
            [
                [9, 9, 9],
                [2, 2, 2],
                [5, 5, 5],
            ],
            dtype=int,
        )
        ga.low_res_shape = (1, 3)

        # 本测试只验证“评估/排序/记录最优”的对应关系，关闭后续遗传算子避免引入无关随机性。
        ga.selection = lambda: None
        ga.crossover = lambda: None
        ga.mutation = lambda: None
        ga.reduce_population_size = lambda: None

        best_x, best_y, history = ga.run(max_iter=1, patience=10)

        self.assertTrue(np.array_equal(best_x, target))
        self.assertEqual(best_y, 0.0)
        self.assertEqual(history, [0.0])

        ga.Chrom[0, 0] = 99
        self.assertEqual(best_x[0], 2)

    def test_optimize_uplift_ga_decodes_integer_chromosomes_to_real_uplift(self):
        """GA 内部用整数编码搜索，但 objective 和返回值必须使用真实 mm/yr。"""
        seen_values = []

        def objective(uplift_vector):
            uplift_vector = np.asarray(uplift_vector, dtype=float)
            seen_values.append(uplift_vector.copy())
            return float(np.sum((uplift_vector - 0.2) ** 2))

        best_x, best_y, history = optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(9, dtype=float).reshape(3, 3),
            LOW_RES_SHAPE=(3, 3),
            ORIGINAL_SHAPE=(3, 3),
            ga_params={
                "pop": 6,
                "max_iter": 1,
                "prob_cross": 0.0,
                "prob_mut": 0.0,
                "lb": 0.0,
                "ub": 0.3,
                "uplift_precision": 0.1,
                "decay_rate": 1.0,
                "min_size_pop": 6,
                "patience": 10,
                "random_seed": 7,
            },
            model_params={},
            n_jobs=1,
            run_mode=None,
        )

        self.assertIsNotNone(best_x)
        self.assertTrue(all(np.allclose(values % 0.1, 0.0, atol=1e-9) for values in seen_values))
        self.assertTrue(np.all(best_x >= -1e-12))
        self.assertTrue(np.all(best_x <= 0.3 + 1e-12))
        self.assertFalse(np.all(np.equal(best_x, best_x.astype(int))))
        self.assertAlmostEqual(best_y, history[-1])

    def test_spatial_crossover_handles_small_low_resolution_shapes(self):
        """低分辨率矩阵很小时，block_size 不能变成 0。"""
        shapes = [(1, 8), (2, 2), (8, 8)]

        for shape in shapes:
            with self.subTest(shape=shape):
                n_dim = shape[0] * shape[1]
                ga = MyGA(
                    func=lambda x: 0.0,
                    n_dim=n_dim,
                    size_pop=4,
                    max_iter=5,
                    prob_mut=0,
                    lb=0,
                    ub=100,
                    n_jobs=1,
                )
                ga.low_res_shape = shape
                parent1 = np.arange(n_dim, dtype=int).reshape(shape)
                parent2 = np.arange(n_dim, 2 * n_dim, dtype=int).reshape(shape)

                np.random.seed(123)
                for _ in range(20):
                    child1, child2 = ga.spatial_crossover(parent1, parent2)
                    self.assertEqual(child1.shape, shape)
                    self.assertEqual(child2.shape, shape)
                    self.assertTrue(np.all(child1 >= 0))
                    self.assertTrue(np.all(child1 <= 100))
                    self.assertTrue(np.all(child2 >= 0))
                    self.assertTrue(np.all(child2 <= 100))

    def test_terrain_initialization_handles_flat_and_nonfinite_dem(self):
        """平坦、NaN、Inf DEM 不应让 terrain prior 初始化产生 NaN 或越界。"""
        matrices = [
            np.ones((6, 6), dtype=float),
            np.array(
                [
                    [np.nan, np.inf, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, np.nan, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, np.inf, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, np.inf],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype=float,
            ),
        ]

        for matrix in matrices:
            with self.subTest(matrix=matrix):
                ga = MyGA(
                    func=lambda x: 0.0,
                    n_dim=9,
                    size_pop=10,
                    max_iter=1,
                    prob_mut=0,
                    lb=1,
                    ub=10,
                    n_jobs=1,
                )
                ga.init_population_based_on_terrain(
                    matrix=matrix,
                    lb=1,
                    ub=10,
                    low_res_shape=(3, 3),
                    noise_level=0,
                    random_fraction=0.2,
                )

                self.assertEqual(ga.Chrom.shape, (10, 9))
                self.assertTrue(np.isfinite(ga.Chrom).all())
                self.assertTrue(np.all(ga.Chrom >= 1))
                self.assertTrue(np.all(ga.Chrom <= 10))
                self.assertGreater(len(np.unique(ga.Chrom, axis=0)), 1)

    def test_diversity_injection_handles_flat_dem(self):
        """停滞注入时的地形 prior 也要能处理平坦 DEM。"""
        ga = MyGA(
            func=lambda x: 0.0,
            n_dim=9,
            size_pop=10,
            max_iter=1,
            prob_mut=0,
            lb=1,
            ub=10,
            n_jobs=1,
        )
        ga.low_res_shape = (3, 3)
        ga.Chrom = np.ones((10, 9), dtype=int)

        ga.inject_diversity(best_x=np.full(9, 5, dtype=int), resampled_dem=np.ones((6, 6), dtype=float))

        self.assertEqual(ga.Chrom.shape, (10, 9))
        self.assertTrue(np.isfinite(ga.Chrom).all())
        self.assertTrue(np.all(ga.Chrom >= 1))
        self.assertTrue(np.all(ga.Chrom <= 10))


if __name__ == "__main__":
    unittest.main()
