import unittest

import numpy as np

from ga_lem_inverter.pipeline.optimization import MyGA


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


if __name__ == "__main__":
    unittest.main()
