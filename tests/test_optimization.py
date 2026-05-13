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


if __name__ == "__main__":
    unittest.main()
