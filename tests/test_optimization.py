import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from ga_lem_inverter.pipeline.optimization import MyGA, optimize_uplift_ga, _allocate_counts, split_decoded_candidate


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
                "search_strategy": "single",
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

    def test_uplift_encoding_respects_unaligned_real_bounds(self):
        """产品验收：非步长整数倍的 uplift 边界不能越界搜索。"""
        seen_values = []

        def objective(uplift_vector):
            uplift_vector = np.asarray(uplift_vector, dtype=float)
            seen_values.append(uplift_vector.copy())
            return float(np.mean(uplift_vector))

        optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(4, dtype=float).reshape(2, 2),
            LOW_RES_SHAPE=(2, 2),
            ORIGINAL_SHAPE=(2, 2),
            ga_params={
                "pop": 4,
                "max_iter": 1,
                "search_strategy": "single",
                "prob_cross": 0.0,
                "prob_mut": 0.0,
                "lb": 0.15,
                "ub": 0.26,
                "uplift_precision": 0.1,
                "decay_rate": 1.0,
                "min_size_pop": 4,
                "patience": 10,
                "random_seed": 7,
            },
            model_params={},
            n_jobs=1,
            run_mode=None,
        )

        all_values = np.concatenate(seen_values)
        self.assertTrue(np.all(all_values >= 0.2 - 1e-12))
        self.assertTrue(np.all(all_values <= 0.2 + 1e-12))

    def test_optimize_uplift_ga_decodes_stage_multipliers_as_low_dimensional_tail(self):
        """产品验收：时间变化 uplift history 只增加 m_stage 低维参数，不复制完整空间场。"""
        seen = []

        def objective(candidate):
            uplift_vector, multipliers = split_decoded_candidate(candidate)
            seen.append((uplift_vector.copy(), multipliers.copy()))
            return float(np.mean((uplift_vector - 0.2) ** 2) + np.mean((multipliers - np.array([0.8, 1.2])) ** 2))

        best_x, best_y, history = optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(4, dtype=float).reshape(2, 2),
            LOW_RES_SHAPE=(2, 2),
            ORIGINAL_SHAPE=(2, 2),
            ga_params={
                "pop": 6,
                "max_iter": 1,
                "search_strategy": "single",
                "prob_cross": 0.0,
                "prob_mut": 0.0,
                "lb": 0.0,
                "ub": 0.4,
                "uplift_precision": 0.1,
                "uplift_history_enabled": True,
                "uplift_history_stage_count": 2,
                "uplift_history_multiplier_min": 0.5,
                "uplift_history_multiplier_max": 1.5,
                "uplift_history_multiplier_precision": 0.1,
                "decay_rate": 1.0,
                "min_size_pop": 6,
                "patience": 10,
                "random_seed": 9,
            },
            model_params={},
            n_jobs=1,
            run_mode=None,
        )

        uplift, multipliers = split_decoded_candidate(best_x)
        self.assertEqual(uplift.shape, (4,))
        self.assertEqual(multipliers.shape, (2,))
        self.assertTrue(all(item[0].shape == (4,) and item[1].shape == (2,) for item in seen))
        seen_multipliers = np.asarray([item[1] for item in seen], dtype=float)
        unique_multipliers = np.unique(np.round(seen_multipliers, decimals=6), axis=0)
        self.assertGreater(unique_multipliers.shape[0], 1)
        self.assertTrue(np.all(multipliers >= 0.5 - 1e-12))
        self.assertTrue(np.all(multipliers <= 1.5 + 1e-12))
        self.assertAlmostEqual(best_y, history[-1])

    def test_optimize_uplift_ga_respects_per_stage_multiplier_bounds(self):
        """产品验收：bounded uplift history 中每个阶段的 multiplier 使用自己的搜索范围。"""
        seen = []

        def objective(candidate):
            _, multipliers = split_decoded_candidate(candidate)
            seen.append(multipliers.copy())
            return float(np.mean((multipliers - np.array([0.5, 1.1, 1.7])) ** 2))

        best_x, best_y, history = optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(4, dtype=float).reshape(2, 2),
            LOW_RES_SHAPE=(2, 2),
            ORIGINAL_SHAPE=(2, 2),
            ga_params={
                "pop": 8,
                "max_iter": 2,
                "search_strategy": "single",
                "prob_cross": 0.4,
                "prob_mut": 0.5,
                "lb": 0.0,
                "ub": 0.4,
                "uplift_precision": 0.1,
                "uplift_history_enabled": True,
                "uplift_history_stage_count": 3,
                "uplift_history_multiplier_min": [0.4, 0.9, 1.4],
                "uplift_history_multiplier_max": [0.7, 1.2, 1.8],
                "uplift_history_multiplier_precision": 0.1,
                "decay_rate": 1.0,
                "min_size_pop": 8,
                "patience": 10,
                "random_seed": 17,
            },
            model_params={},
            n_jobs=1,
            run_mode=None,
        )

        _, multipliers = split_decoded_candidate(best_x)
        all_seen = np.asarray(seen, dtype=float)
        self.assertEqual(multipliers.shape, (3,))
        self.assertGreater(all_seen.shape[0], 0)
        self.assertTrue(np.all(all_seen[:, 0] >= 0.4 - 1e-12))
        self.assertTrue(np.all(all_seen[:, 0] <= 0.7 + 1e-12))
        self.assertTrue(np.all(all_seen[:, 1] >= 0.9 - 1e-12))
        self.assertTrue(np.all(all_seen[:, 1] <= 1.2 + 1e-12))
        self.assertTrue(np.all(all_seen[:, 2] >= 1.4 - 1e-12))
        self.assertTrue(np.all(all_seen[:, 2] <= 1.8 + 1e-12))
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

    def test_diversity_injection_preserves_best_and_never_leaves_zero_slots(self):
        """停滞注入必须显式保留 best，并且所有槽位都在合法边界内。"""
        ga = MyGA(
            func=lambda x: 0.0,
            n_dim=4,
            size_pop=6,
            max_iter=1,
            prob_mut=0,
            lb=2,
            ub=5,
            n_jobs=1,
            diversity_random_fraction=0.0,
            diversity_best_fraction=1.0,
            diversity_terrain_fraction=0.0,
        )
        ga.low_res_shape = (2, 2)
        ga.Chrom = np.full((6, 4), 3, dtype=int)
        best = np.full(4, 5, dtype=int)

        ga.inject_diversity(best_x=best, resampled_dem=None)

        self.assertTrue(np.array_equal(ga.Chrom[0], best))
        self.assertTrue(np.all(ga.Chrom >= 2))
        self.assertTrue(np.all(ga.Chrom <= 5))

    def test_ga_aborts_when_objective_systematically_returns_invalid_values(self):
        """整代 objective 系统性失败时，不应继续假装优化成功。"""
        ga = MyGA(
            func=lambda x: np.nan,
            n_dim=4,
            size_pop=5,
            max_iter=1,
            prob_mut=0,
            lb=0,
            ub=10,
            n_jobs=1,
        )
        ga.Chrom = np.ones((5, 4), dtype=int)
        ga.low_res_shape = (2, 2)

        with self.assertRaisesRegex(RuntimeError, "failed for 5/5 candidates"):
            ga.run(max_iter=1, patience=10)

    def test_ga_aborts_when_objective_systematically_raises(self):
        """objective 直接抛异常时，也要计入失败率并中止。"""
        def objective(_x):
            raise ValueError("boom")

        ga = MyGA(
            func=objective,
            n_dim=4,
            size_pop=5,
            max_iter=1,
            prob_mut=0,
            lb=0,
            ub=10,
            n_jobs=1,
        )
        ga.Chrom = np.ones((5, 4), dtype=int)
        ga.low_res_shape = (2, 2)

        with self.assertRaisesRegex(RuntimeError, "failed for 5/5 candidates"):
            ga.run(max_iter=1, patience=10)

    def test_ga_penalizes_sparse_invalid_fitness_values(self):
        """少量非法 fitness 会被惩罚为 1.0，但不触发系统性失败。"""
        calls = {"count": 0}

        def objective(x):
            calls["count"] += 1
            if calls["count"] == 1:
                return np.nan
            return 0.2

        ga = MyGA(
            func=objective,
            n_dim=4,
            size_pop=5,
            max_iter=1,
            prob_mut=0,
            lb=0,
            ub=10,
            n_jobs=1,
        )
        ga.Chrom = np.ones((5, 4), dtype=int)
        ga.low_res_shape = (2, 2)
        ga.selection = lambda: None
        ga.crossover = lambda: None
        ga.mutation = lambda: None
        ga.reduce_population_size = lambda: None

        best_x, best_y, history = ga.run(max_iter=1, patience=10)

        self.assertIsNotNone(best_x)
        self.assertEqual(best_y, 0.2)
        self.assertEqual(history, [0.2])
        self.assertTrue(np.all(np.isfinite(ga.Y)))
        self.assertIn(1.0, ga.Y)

    def test_fitness_cache_reuses_duplicate_chromosomes(self):
        """重复染色体应只评价一次，其余从 fitness cache 读取。"""
        calls = {"count": 0}

        def objective(x):
            calls["count"] += 1
            return float(np.sum(x) / 100.0)

        ga = MyGA(
            func=objective,
            n_dim=4,
            size_pop=4,
            max_iter=1,
            prob_mut=0,
            lb=0,
            ub=10,
            n_jobs=1,
            enable_fitness_cache=True,
        )
        ga.Chrom = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=int,
        )
        ga.low_res_shape = (2, 2)
        ga.selection = lambda: None
        ga.crossover = lambda: None
        ga.mutation = lambda: None
        ga.reduce_population_size = lambda: None

        ga.run(max_iter=1, patience=10)

        self.assertEqual(calls["count"], 2)
        self.assertEqual(ga.cache_hits, 2)
        self.assertEqual(ga.cache_misses, 2)
        self.assertEqual(len(ga.fitness_cache), 2)

    def test_fitness_cache_does_not_change_fixed_seed_result(self):
        """cache 只能减少重复评价，不能改变固定 seed 下的最优结果。"""
        params = {
            "pop": 6,
            "max_iter": 2,
            "search_strategy": "single",
            "prob_cross": 0.0,
            "prob_mut": 0.0,
            "lb": 0.0,
            "ub": 0.5,
            "uplift_precision": 0.1,
            "decay_rate": 1.0,
            "min_size_pop": 6,
            "patience": 10,
            "random_seed": 11,
        }

        def objective(uplift_vector):
            uplift_vector = np.asarray(uplift_vector, dtype=float)
            return float(np.mean((uplift_vector - 0.3) ** 2))

        result_without_cache = optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(9, dtype=float).reshape(3, 3),
            LOW_RES_SHAPE=(3, 3),
            ORIGINAL_SHAPE=(3, 3),
            ga_params={**params, "enable_fitness_cache": False},
            model_params={},
            n_jobs=1,
            run_mode=None,
        )
        result_with_cache = optimize_uplift_ga(
            obj_func=objective,
            resampled_dem=np.arange(9, dtype=float).reshape(3, 3),
            LOW_RES_SHAPE=(3, 3),
            ORIGINAL_SHAPE=(3, 3),
            ga_params={**params, "enable_fitness_cache": True},
            model_params={},
            n_jobs=1,
            run_mode=None,
        )

        self.assertTrue(np.array_equal(result_without_cache[0], result_with_cache[0]))
        self.assertEqual(result_without_cache[1], result_with_cache[1])

    def test_diversity_fraction_allocation_exact_total(self):
        """小种群下 diversity fraction 分配不能超过或少于可注入个体数。"""
        counts = _allocate_counts(5, (0.3, 0.5, 0.2))

        self.assertEqual(sum(counts), 5)
        self.assertTrue(all(count >= 0 for count in counts))

    def test_diversity_threshold_and_cooldown_trigger_injection(self):
        """停滞 objective 应按 threshold/cooldown 触发 diversity injection。"""
        np.random.seed(3)
        ga = MyGA(
            func=lambda x: 0.5,
            n_dim=4,
            size_pop=6,
            max_iter=3,
            prob_mut=0.0,
            lb=0,
            ub=10,
            n_jobs=1,
            enable_fitness_cache=False,
            diversity_threshold=1,
            diversity_cooldown=1,
        )
        ga.Chrom = np.ones((6, 4), dtype=int)
        ga.low_res_shape = (2, 2)
        ga.run(max_iter=3, patience=10, resampled_dem=np.ones((4, 4), dtype=float))

        self.assertGreaterEqual(ga.diversity_injections, 1)
        self.assertTrue(any(row["diversity_injected"] for row in ga.history_rows))

    def test_mutation_schedule_respects_upper_bound_and_stagnation_boost(self):
        """停滞增强可以提高 mutation，但不能超过 max multiplier。"""
        ga = MyGA(
            func=lambda x: 0.0,
            n_dim=4,
            size_pop=4,
            max_iter=10,
            prob_mut=0.1,
            lb=0,
            ub=10,
            n_jobs=1,
            mutation_schedule="adaptive",
            mutation_max_multiplier=1.5,
            mutation_stagnation_boost=True,
            mutation_stagnation_multiplier=10.0,
        )
        ga.current_gen = 10
        ga._stagnation_count = 3

        self.assertAlmostEqual(ga.get_adaptive_mutation_prob(), 0.15)

    def test_seed_population_from_previous_best_preserves_best_without_counting_injection(self):
        """后续 stage 初始种群必须包含前一阶段 best，但不应计为停滞 diversity injection。"""
        ga = MyGA(
            func=lambda x: 0.0,
            n_dim=4,
            size_pop=6,
            max_iter=1,
            prob_mut=0.0,
            lb=0,
            ub=10,
            n_jobs=1,
        )
        ga.low_res_shape = (2, 2)
        ga.Chrom = np.zeros((6, 4), dtype=int)
        best = np.array([5, 6, 7, 8], dtype=int)

        ga.seed_population_from_best(best, resampled_dem=np.ones((4, 4), dtype=float))

        self.assertTrue(np.array_equal(ga.Chrom[0], best))
        self.assertEqual(ga.diversity_injections, 0)

    def test_staged_search_runs_ordered_stages_and_writes_diagnostics(self):
        """staged 模式应顺序执行、继承 best，并输出 GA 诊断文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calls = []

            def objective(uplift_vector):
                calls.append(np.asarray(uplift_vector, dtype=float).copy())
                return float(np.mean((uplift_vector - 0.2) ** 2))

            best_x, best_y, history = optimize_uplift_ga(
                obj_func=objective,
                resampled_dem=np.arange(9, dtype=float).reshape(3, 3),
                LOW_RES_SHAPE=(3, 3),
                ORIGINAL_SHAPE=(3, 3),
                ga_params={
                    "search_strategy": "staged",
                    "enable_fitness_cache": True,
                    "stages": [
                        {
                            "name": "coarse",
                            "population_size": 4,
                            "max_iterations": 1,
                            "mutation_probability": 0.0,
                            "cross_probability": 0.0,
                            "patience": 2,
                            "min_population_size": 4,
                        },
                        {
                            "name": "refine",
                            "population_size": 4,
                            "max_iterations": 1,
                            "mutation_probability": 0.0,
                            "cross_probability": 0.0,
                            "patience": 2,
                            "min_population_size": 4,
                        },
                    ],
                    "pop": 4,
                    "max_iter": 1,
                    "prob_cross": 0.0,
                    "prob_mut": 0.0,
                    "lb": 0.0,
                    "ub": 0.4,
                    "uplift_precision": 0.1,
                    "decay_rate": 1.0,
                    "min_size_pop": 4,
                    "patience": 2,
                    "random_seed": 5,
                    "diagnostics_dir": tmpdir,
                },
                model_params={},
                n_jobs=1,
                run_mode=None,
            )

            metrics_dir = Path(tmpdir) / "metrics"
            tables_dir = Path(tmpdir) / "tables"
            ga_metrics = json.loads((metrics_dir / "ga_metrics.json").read_text(encoding="utf-8"))
            stage_metrics = json.loads((metrics_dir / "stage_metrics.json").read_text(encoding="utf-8"))
            with (tables_dir / "ga_history.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertIsNotNone(best_x)
            self.assertLessEqual(best_y, stage_metrics[0]["best_fitness"])
            self.assertEqual([stage["stage"] for stage in stage_metrics], ["coarse", "refine"])
            self.assertFalse(stage_metrics[0]["inherited_previous_best"])
            self.assertTrue(stage_metrics[1]["inherited_previous_best"])
            self.assertEqual(ga_metrics["search_strategy"], "staged")
            self.assertEqual(ga_metrics["stage_count"], 2)
            self.assertEqual(len(history), 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["stage"] for row in rows], ["coarse", "refine"])
            self.assertGreater(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
