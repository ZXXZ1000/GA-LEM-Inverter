# genetic_algorithm.py
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import psutil
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import csv
import json

def _evaluate_fitness(obj_func, x):
    try:
        return obj_func(x)
    except Exception as exc:
        logging.error("目标函数评价失败，使用 inf 作为失败标记: %s", exc)
        return np.inf


def parallel_fitness(obj_func, X, n_jobs=-1):
    """并行计算适应度函数"""
    try:
        fitness_values = Parallel(n_jobs=n_jobs)(delayed(_evaluate_fitness)(obj_func, x) for x in X)
        return np.array(fitness_values)
    except Exception as e:
        logging.error(f"并行计算失败,切换为串行: {e}")
        return np.array([_evaluate_fitness(obj_func, x) for x in X])


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _chromosome_key(x):
    return tuple(int(v) for v in np.asarray(x, dtype=int).reshape(-1))


def _normalize_fractions(random_fraction, best_fraction, terrain_fraction):
    values = np.asarray([random_fraction, best_fraction, terrain_fraction], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("diversity fractions 必须是非负有限数。")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("diversity fractions 不能全部为 0。")
    values = values / total
    return tuple(float(v) for v in values)


def _allocate_counts(total, fractions):
    """Allocate integer counts with exact total and no negative values."""
    total = int(total)
    if total <= 0:
        return [0 for _ in fractions]
    raw = np.asarray(fractions, dtype=float) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return [int(v) for v in counts]


def _sanitize_fitness_values(values, penalty=1.0):
    """把目标函数返回值限制为有限的 0-1 loss，并返回失败掩码。"""
    raw_values = np.asarray(values, dtype=float)
    failure_mask = ~np.isfinite(raw_values) | (raw_values < 0.0) | (raw_values > 1.0)
    sanitized = np.clip(np.nan_to_num(raw_values, nan=penalty, posinf=penalty, neginf=penalty), 0.0, 1.0)
    sanitized[failure_mask] = penalty
    return sanitized, failure_mask


def _get_uplift_precision(ga_params):
    """读取隆升率搜索步长，单位 mm/yr。"""
    precision = float(ga_params.get('uplift_precision', ga_params.get('precision', 1.0)))
    if precision <= 0:
        raise ValueError(f"uplift_precision 必须大于 0，当前为 {precision}")
    return precision


def _encode_uplift_value(value, precision):
    """把真实隆升率 mm/yr 转成 GA 内部整数编码。"""
    return int(round(float(value) / precision))


def _encode_uplift_bounds(lb, ub, precision):
    encoded_lb = int(math.ceil(float(lb) / precision - 1e-12))
    encoded_ub = int(math.floor(float(ub) / precision + 1e-12))
    if encoded_lb > encoded_ub:
        raise ValueError(
            f"uplift_min/uplift_max 与 uplift_precision 不兼容: "
            f"{lb}..{ub} mm/yr, precision={precision}"
        )
    return encoded_lb, encoded_ub


def _decode_uplift_vector(encoded_vector, precision):
    """把 GA 内部整数编码解码成真实隆升率 mm/yr。"""
    return np.asarray(encoded_vector, dtype=float) * precision


def _get_history_settings(ga_params):
    """Resolve optional low-dimensional uplift-history encoding settings."""
    enabled = _as_bool(ga_params.get("uplift_history_enabled"), False)
    count = int(ga_params.get("uplift_history_stage_count", 0) or 0)
    if not enabled:
        return {
            "enabled": False,
            "stage_count": 0,
            "precision": 0.1,
            "encoded_lb": np.empty(0, dtype=int),
            "encoded_ub": np.empty(0, dtype=int),
        }
    if count < 1:
        raise ValueError("启用 uplift history 时，uplift_history_stage_count 必须 >= 1。")
    precision = float(ga_params.get("uplift_history_multiplier_precision", 0.1))
    if precision <= 0:
        raise ValueError(f"uplift_history_multiplier_precision 必须大于 0，当前为 {precision}")
    multiplier_min = ga_params.get("uplift_history_multiplier_min", 0.5)
    multiplier_max = ga_params.get("uplift_history_multiplier_max", 1.5)
    lb = np.asarray(multiplier_min if isinstance(multiplier_min, (list, tuple, np.ndarray)) else [multiplier_min] * count, dtype=float)
    ub = np.asarray(multiplier_max if isinstance(multiplier_max, (list, tuple, np.ndarray)) else [multiplier_max] * count, dtype=float)
    if lb.size != count or ub.size != count:
        raise ValueError("uplift history multiplier_min/max 数量必须等于 stage_count。")
    encoded_lb = np.asarray([_encode_uplift_bounds(l, u, precision)[0] for l, u in zip(lb, ub)], dtype=int)
    encoded_ub = np.asarray([_encode_uplift_bounds(l, u, precision)[1] for l, u in zip(lb, ub)], dtype=int)
    return {
        "enabled": True,
        "stage_count": count,
        "precision": precision,
        "encoded_lb": encoded_lb,
        "encoded_ub": encoded_ub,
    }


def _decode_candidate_vector(encoded_vector, *, uplift_dim, uplift_precision, history_settings):
    """Decode a mixed chromosome into real uplift plus optional multipliers."""
    encoded = np.asarray(encoded_vector, dtype=float).reshape(-1)
    uplift = _decode_uplift_vector(encoded[:uplift_dim], uplift_precision)
    if not history_settings["enabled"]:
        return uplift
    stage_count = history_settings["stage_count"]
    multipliers = encoded[uplift_dim:uplift_dim + stage_count] * history_settings["precision"]
    return {
        "uplift": uplift,
        "stage_multipliers": np.asarray(multipliers, dtype=float),
    }


def split_decoded_candidate(candidate):
    """Return ``(uplift_vector, stage_multipliers_or_none)`` from GA output."""
    if isinstance(candidate, dict):
        return np.asarray(candidate["uplift"], dtype=float), np.asarray(candidate["stage_multipliers"], dtype=float)
    return np.asarray(candidate, dtype=float), None


def _safe_int_bounds(lb, ub):
    lb_array = np.asarray(lb).reshape(-1)
    ub_array = np.asarray(ub).reshape(-1)
    lb_val = int(round(float(lb_array[0])))
    ub_val = int(round(float(ub_array[0])))
    if lb_val > ub_val:
        raise ValueError(f"Invalid GA bounds: lb={lb_val}, ub={ub_val}")
    return lb_val, ub_val


def _terrain_prior_vector(matrix, low_res_shape, lb, ub, n_dim=None):
    """把 DEM 安全映射成 GA 整数编码 prior；平坦或异常 DEM 退化为中值场。"""
    lb_array = np.asarray(lb).reshape(-1)
    ub_array = np.asarray(ub).reshape(-1)
    spatial_dim = int(np.prod(low_res_shape))
    if n_dim is None:
        n_dim = spatial_dim
    if lb_array.size == 1:
        lb_val = int(round(float(lb_array[0])))
    else:
        lb_val = int(round(float(lb_array[:spatial_dim][0])))
    if ub_array.size == 1:
        ub_val = int(round(float(ub_array[0])))
    else:
        ub_val = int(round(float(ub_array[:spatial_dim][0])))
    midpoint = int(round((lb_val + ub_val) / 2))

    terrain = np.asarray(matrix, dtype=float)
    if terrain.size == 0 or not np.isfinite(terrain).any():
        spatial = np.full(spatial_dim, midpoint, dtype=int)
        return _append_default_tail(spatial, lb, ub, n_dim)

    finite_values = terrain[np.isfinite(terrain)]
    fill_value = float(np.nanmedian(finite_values))
    terrain = np.where(np.isfinite(terrain), terrain, fill_value)
    smoothed_matrix = gaussian_filter(terrain, sigma=5)

    if smoothed_matrix.shape != low_res_shape:
        from skimage.transform import resize
        smoothed_matrix = resize(
            smoothed_matrix,
            low_res_shape,
            mode='edge',
            anti_aliasing=True,
            preserve_range=True
        )

    min_val = float(np.nanmin(smoothed_matrix))
    max_val = float(np.nanmax(smoothed_matrix))
    value_range = max_val - min_val
    if not np.isfinite(value_range) or value_range <= 1e-12:
        spatial = np.full(spatial_dim, midpoint, dtype=int)
        return _append_default_tail(spatial, lb, ub, n_dim)

    scaled_matrix = (smoothed_matrix - min_val) / value_range
    scaled_matrix = lb_val + (ub_val - lb_val) * scaled_matrix
    spatial = np.clip(np.rint(scaled_matrix), lb_val, ub_val).astype(int).flatten()
    return _append_default_tail(spatial, lb, ub, n_dim)


def _append_default_tail(spatial_vector, lb, ub, n_dim):
    spatial_vector = np.asarray(spatial_vector, dtype=int).reshape(-1)
    n_dim = int(n_dim)
    if spatial_vector.size >= n_dim:
        return spatial_vector[:n_dim]
    lb_array = np.asarray(lb).reshape(-1)
    ub_array = np.asarray(ub).reshape(-1)
    if lb_array.size == 1:
        lb_tail = np.full(n_dim - spatial_vector.size, int(round(float(lb_array[0]))), dtype=int)
    else:
        lb_tail = np.rint(lb_array[spatial_vector.size:n_dim]).astype(int)
    if ub_array.size == 1:
        ub_tail = np.full(n_dim - spatial_vector.size, int(round(float(ub_array[0]))), dtype=int)
    else:
        ub_tail = np.rint(ub_array[spatial_vector.size:n_dim]).astype(int)
    tail = np.rint((lb_tail + ub_tail) / 2).astype(int)
    return np.concatenate([spatial_vector, tail])


def _stratified_tail_population(count, lb, ub, *, spatial_dim, n_dim):
    """Generate low-dimensional tail values with explicit coverage of stage trends."""
    count = int(count)
    spatial_dim = int(spatial_dim)
    n_dim = int(n_dim)
    tail_dim = n_dim - spatial_dim
    if count <= 0 or tail_dim <= 0:
        return np.empty((max(0, count), 0), dtype=int)

    lb_array = np.asarray(lb).reshape(-1)
    ub_array = np.asarray(ub).reshape(-1)
    if lb_array.size == 1:
        tail_lb = np.full(tail_dim, int(round(float(lb_array[0]))), dtype=int)
    else:
        tail_lb = np.rint(lb_array[spatial_dim:n_dim]).astype(int)
    if ub_array.size == 1:
        tail_ub = np.full(tail_dim, int(round(float(ub_array[0]))), dtype=int)
    else:
        tail_ub = np.rint(ub_array[spatial_dim:n_dim]).astype(int)

    midpoint = np.rint((tail_lb + tail_ub) / 2).astype(int)
    if tail_dim == 1:
        profiles = [midpoint, tail_lb, tail_ub]
    else:
        trend = np.linspace(0.0, 1.0, tail_dim)
        increasing = np.rint(tail_lb + trend * (tail_ub - tail_lb)).astype(int)
        decreasing = np.rint(tail_lb + trend[::-1] * (tail_ub - tail_lb)).astype(int)
        profiles = [midpoint, increasing, decreasing, tail_lb, tail_ub]

    tails = np.empty((count, tail_dim), dtype=int)
    for idx in range(count):
        if idx < len(profiles):
            tails[idx] = profiles[idx]
        else:
            tails[idx] = np.random.randint(tail_lb, tail_ub + 1)
    return np.clip(tails, tail_lb, tail_ub).astype(int)


def _smooth_random_population(count, n_dim, low_res_shape, lb, ub):
    """生成平滑随机场个体，作为非 DEM prior 的空间结构初始化。"""
    spatial_dim = int(np.prod(low_res_shape))
    lb_array = np.asarray(lb).reshape(-1)
    ub_array = np.asarray(ub).reshape(-1)
    lb_val = int(round(float(lb_array[0])))
    ub_val = int(round(float(ub_array[0])))
    if count <= 0:
        return np.empty((0, n_dim), dtype=int)

    individuals = np.zeros((count, n_dim), dtype=int)
    for i in range(count):
        field = gaussian_filter(np.random.rand(*low_res_shape), sigma=1)
        min_val = float(field.min())
        max_val = float(field.max())
        if max_val - min_val <= 1e-12:
            scaled = np.full(low_res_shape, (lb_val + ub_val) / 2)
        else:
            scaled = (field - min_val) / (max_val - min_val)
            scaled = lb_val + (ub_val - lb_val) * scaled
        spatial = np.clip(np.rint(scaled), lb_val, ub_val).astype(int).flatten()
        individuals[i, :] = _append_default_tail(spatial, lb, ub, n_dim)
    return individuals

class MyGA:
    """自定义遗传算法类，专门用于隆升率场优化"""
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.01,
                 lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1, decay_rate=0.95, min_size_pop=10, patience=40,
                 n_jobs=-1, enable_fitness_cache=False,
                 diversity_threshold=None, diversity_cooldown=10,
                 diversity_random_fraction=0.2, diversity_best_fraction=0.5,
                 diversity_terrain_fraction=0.3,
                 mutation_schedule="adaptive", mutation_max_multiplier=2.0,
                 mutation_stagnation_boost=False, mutation_stagnation_multiplier=1.5):
        """
        初始化遗传算法类

        参数:
        - func: 目标函数
        - n_dim: 问题维度
        - size_pop: 种群大小
        - max_iter: 最大迭代次数
        - prob_mut: 变异概率
        - lb, ub: 参数取值范围的上下界
        - decay_rate: 种群衰减率
        - min_size_pop: 最小种群大小
        - patience: 早停耐心值
        - n_jobs: 并行计算的进程数
        """
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut
        self.prob_cross = 0.7  # 交叉概率
        self.current_gen = 0

        # 设置参数上下界
        self.lb = np.array(lb) if isinstance(lb, np.ndarray) else np.full(n_dim, lb)
        self.ub = np.array(ub) if isinstance(ub, np.ndarray) else np.full(n_dim, ub)

        # 其他参数
        self.precision = precision
        self.decay_rate = decay_rate
        self.min_size_pop = min_size_pop
        self.patience = patience
        self.n_jobs = n_jobs
        self.failure_threshold = 0.8
        self.max_consecutive_full_failures = 2
        self.enable_fitness_cache = bool(enable_fitness_cache)
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.diversity_threshold = diversity_threshold
        self.diversity_cooldown = max(0, int(diversity_cooldown))
        (
            self.diversity_random_fraction,
            self.diversity_best_fraction,
            self.diversity_terrain_fraction,
        ) = _normalize_fractions(diversity_random_fraction, diversity_best_fraction, diversity_terrain_fraction)
        self.diversity_injections = 0
        self.mutation_schedule = str(mutation_schedule or "adaptive").strip().lower()
        self.mutation_max_multiplier = max(1.0, float(mutation_max_multiplier))
        self.mutation_stagnation_boost = bool(mutation_stagnation_boost)
        self.mutation_stagnation_multiplier = max(1.0, float(mutation_stagnation_multiplier))
        self._stagnation_count = 0
        self._current_mutation_prob = self.prob_mut
        self.history_rows = []
        self.stage_name = "single"

        # 初始化种群和适应度
        self.Chrom = None
        self.X = None  # 种群的表现型
        self.Y = None  # 种群的适应度
        self.low_res_shape = None  # 低分辨率形状
        self.spatial_dim = n_dim

        # 记录最优解
        self.best_x = None
        self.best_y = float('inf')

    def evaluate_population(self):
        """Evaluate population, using optional chromosome-level cache."""
        if not self.enable_fitness_cache:
            return parallel_fitness(self.func, self.X, n_jobs=self.n_jobs)

        results = [None] * len(self.X)
        missing_by_key = {}
        missing_vectors = []
        for idx, x in enumerate(self.X):
            key = _chromosome_key(x)
            if key in self.fitness_cache:
                self.cache_hits += 1
                results[idx] = self.fitness_cache[key]
            elif key in missing_by_key:
                self.cache_hits += 1
                missing_by_key[key].append(idx)
            else:
                self.cache_misses += 1
                missing_by_key[key] = [idx]
                missing_vectors.append(x)

        if missing_vectors:
            evaluated = parallel_fitness(self.func, np.asarray(missing_vectors), n_jobs=self.n_jobs)
            for key, value in zip(missing_by_key.keys(), evaluated):
                self.fitness_cache[key] = float(value)
                for idx in missing_by_key[key]:
                    results[idx] = float(value)

        return np.asarray(results, dtype=float)

    def init_population_based_on_terrain(self, matrix, lb, ub, low_res_shape, noise_level=10, random_fraction=0.2):
        """
        基于地形矩阵初始化种群

        参数:
        - matrix: 地形矩阵
        - lb, ub: 参数下界和上界
        - low_res_shape: 低分辨率形状
        - noise_level: 添加随机噪声的级别
        - random_fraction: 随机初始化的比例
        """
        spatial_dim = int(np.prod(low_res_shape)) if low_res_shape is not None else 0
        if low_res_shape is None or spatial_dim > self.n_dim or spatial_dim < 1:
            raise ValueError(f"Invalid low_res_shape: {low_res_shape}. Expected product <= {self.n_dim}")

        self.low_res_shape = low_res_shape
        self.spatial_dim = spatial_dim
        random_pop_size = int(self.size_pop * random_fraction)
        smooth_random_pop_size = max(1, int(self.size_pop * 0.2)) if self.size_pop >= 3 else 0
        terrain_pop_size = self.size_pop - random_pop_size - smooth_random_pop_size
        if terrain_pop_size < 0:
            terrain_pop_size = 0
            smooth_random_pop_size = self.size_pop - random_pop_size
        initial_population = np.zeros((self.size_pop, self.n_dim), dtype=int)
        lb_array = np.asarray(lb).reshape(-1)
        ub_array = np.asarray(ub).reshape(-1)
        lb_val = int(round(float(lb_array[0])))
        ub_val = int(round(float(ub_array[0])))

        if terrain_pop_size > 0:
            initial_vector = _terrain_prior_vector(matrix, low_res_shape, lb, ub, n_dim=self.n_dim)
            terrain_individuals = np.zeros((terrain_pop_size, self.n_dim), dtype=int)

            for i in range(terrain_pop_size):
                noise = np.zeros(self.n_dim, dtype=int)
                noise[:spatial_dim] = np.random.randint(-noise_level, noise_level + 1, size=spatial_dim)
                individual = initial_vector + noise
                individual = np.clip(individual, self.lb, self.ub)
                terrain_individuals[i, :] = individual

            initial_population[:terrain_pop_size, :] = terrain_individuals

        smooth_start = terrain_pop_size
        smooth_end = smooth_start + smooth_random_pop_size
        if smooth_random_pop_size > 0:
            initial_population[smooth_start:smooth_end, :] = _smooth_random_population(
                smooth_random_pop_size,
                self.n_dim,
                low_res_shape,
                lb,
                ub
            )

        if random_pop_size > 0:
            random_individuals = np.random.randint(
                self.lb,
                self.ub + 1,
                size=(random_pop_size, self.n_dim)
            )
            initial_population[smooth_end:, :] = random_individuals

        if self.n_dim > spatial_dim:
            initial_population[:, spatial_dim:] = _stratified_tail_population(
                self.size_pop,
                self.lb,
                self.ub,
                spatial_dim=spatial_dim,
                n_dim=self.n_dim,
            )

        self.Chrom = initial_population


    def get_adaptive_block_size(self):
        """
        获取自适应块大小，根据问题复杂度和当前代数动态调整
        """
        # 基础块大小
        base_size = 1

        # 根据问题维度调整块大小
        if self.low_res_shape[0] > 20 or self.low_res_shape[1] > 20:
            base_size = 2

        # 根据迭代进度调整块大小
        progress = min(1.0, self.current_gen / self.max_iter)
        if progress > 0.7:
            # 后期使用更小的块以进行精细调整
            return max(1, base_size - 1)
        elif progress < 0.3:
            # 早期使用更大的块以进行粗略探索
            return base_size + 1
        else:
            return base_size

    def spatial_crossover(self, parent1, parent2):
        """
        增强的空间感知交叉操作：动态块大小和多种交叉策略

        参数:
        - parent1, parent2: 父代个体(低分辨率矩阵)

        返回:
        - child1, child2: 子代个体
        """
        rows, cols = parent1.shape
        if rows < 1 or cols < 1:
            raise ValueError(f"Invalid crossover matrix shape: {parent1.shape}")

        # 动态调整块大小
        block_size = self.get_adaptive_block_size()
        # 小矩阵例如 1xN、2x2 时 rows//3 或 cols//3 会变成 0。
        # 这时退化成 1x1 空间块交换，仍保留局部交叉含义。
        max_block_size = max(1, min(rows, cols))
        third_size = max(1, min(rows // 3, cols // 3))
        block_size = max(1, min(block_size, max_block_size, third_size))

        total_blocks = (rows * cols) // (block_size * block_size)
        max_row = rows - block_size
        max_col = cols - block_size

        # 选择交叉策略
        strategy_probs = {
            'block_exchange': 0.5,  # 块交换策略
            'gradient_blend': 0.3,  # 渐变混合策略
            'feature_based': 0.2    # 基于特征的交叉策略
        }

        strategy = np.random.choice(
            list(strategy_probs.keys()),
            p=list(strategy_probs.values())
        )

        if strategy == 'block_exchange':
            # 策略1：增强的块交换
            # 随机选择交换块的数量(根据总块数动态调整)
            exchange_ratio = 0.01 + 0.04 * (1 - min(1.0, self.current_gen / (self.max_iter * 0.8)))
            n_blocks = max(1, int(exchange_ratio * total_blocks))

            logging.debug(f"Exchanging {n_blocks} blocks out of {total_blocks} total blocks")

            # 创建子代
            child1 = parent1.copy()
            child2 = parent2.copy()

            # 记录已交换的位置
            exchanged_positions = set()

            # 执行多块交换
            for _ in range(n_blocks):
                # 尝试找到未交换的位置
                for attempt in range(10):  # 最多尝试10次
                    if max_row >= 0 and max_col >= 0:
                        start_row = np.random.randint(0, max_row + 1)
                        start_col = np.random.randint(0, max_col + 1)

                        pos = (start_row, start_col)
                        if pos not in exchanged_positions:
                            # 执行块交换
                            temp = child1[start_row:start_row+block_size,
                                        start_col:start_col+block_size].copy()

                            child1[start_row:start_row+block_size,
                                  start_col:start_col+block_size] = \
                                child2[start_row:start_row+block_size,
                                      start_col:start_col+block_size]

                            child2[start_row:start_row+block_size,
                                  start_col:start_col+block_size] = temp

                            exchanged_positions.add(pos)
                            break

                    if attempt == 9:
                        logging.debug("Failed to find non-exchanged position after 10 attempts")

        elif strategy == 'gradient_blend':
            # 策略2：改进的渐变混合
            # 自适应sigma参数，随着迭代进度减小，使混合更加局部化
            progress = min(1.0, self.current_gen / self.max_iter)
            sigma = 2.0 * (1.0 - progress * 0.5)  # 从2.0逐渐减小到1.0

            # 生成混合权重矩阵
            alpha = np.random.rand(rows, cols)
            alpha = gaussian_filter(alpha, sigma=sigma)
            alpha = np.clip(alpha, 0, 1)

            # 局部特征保留：随机选择一些区域保持原始特征
            if np.random.rand() < 0.3:
                feature_mask = np.random.rand(rows, cols) > 0.85
                alpha[feature_mask] = 1.0  # 这些区域完全保留parent1的特征

            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1

        else:  # feature_based
            # 策略3：基于特征的交叉
            # 计算每个父代的局部梯度作为特征指标
            from scipy.ndimage import sobel

            # 计算梯度幅度
            parent1_grad_x = sobel(parent1, axis=0)
            parent1_grad_y = sobel(parent1, axis=1)
            parent1_grad = np.sqrt(parent1_grad_x**2 + parent1_grad_y**2)

            parent2_grad_x = sobel(parent2, axis=0)
            parent2_grad_y = sobel(parent2, axis=1)
            parent2_grad = np.sqrt(parent2_grad_x**2 + parent2_grad_y**2)

            # 创建特征掩码：选择梯度更大的父代
            feature_mask = parent1_grad > parent2_grad

            # 基于特征掩码创建子代
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)

            # 子代1从父代1继承高梯度区域，从父代2继承低梯度区域
            child1[feature_mask] = parent1[feature_mask]
            child1[~feature_mask] = parent2[~feature_mask]

            # 子代2从父代2继承高梯度区域，从父代1继承低梯度区域
            child2[feature_mask] = parent2[feature_mask]
            child2[~feature_mask] = parent1[~feature_mask]

            # 添加一些随机性，避免过度继承
            random_mask = np.random.rand(rows, cols) < 0.1
            child1[random_mask] = parent2[random_mask]
            child2[random_mask] = parent1[random_mask]

        # 确保值在边界内
        child1 = np.clip(child1, self.lb[0], self.ub[0]).astype(int)
        child2 = np.clip(child2, self.lb[0], self.ub[0]).astype(int)

        return child1, child2

    def selection(self):
        """
        选择操作：使用锦标赛选择和精英保留策略
        """
        sorted_indices = np.argsort(self.Y)
        elite_size = max(1, int(0.05 * self.size_pop))  # 保留5%的精英
        elites = self.Chrom[sorted_indices[:elite_size]]

        tournament_size = min(3, self.size_pop)
        selected = []
        for _ in range(self.size_pop - elite_size):
            participants = np.random.choice(self.size_pop, tournament_size, replace=False)
            winner = participants[np.argmin(self.Y[participants])]
            selected.append(self.Chrom[winner])

        self.Chrom = np.vstack((elites, np.array(selected)))

    def crossover(self):
        """交叉操作：空间感知的交叉"""
        np.random.shuffle(self.Chrom)
        for i in range(0, self.size_pop - 1, 2):
            if np.random.rand() < self.prob_cross:
                spatial_dim = int(np.prod(self.low_res_shape))
                parent1_tail = self.Chrom[i, spatial_dim:].copy()
                parent2_tail = self.Chrom[i + 1, spatial_dim:].copy()
                parent1 = self.Chrom[i, :spatial_dim].reshape(self.low_res_shape)
                parent2 = self.Chrom[i + 1, :spatial_dim].reshape(self.low_res_shape)

                # 生成新后代
                child1, child2 = self.spatial_crossover(parent1, parent2)
                if parent1_tail.size:
                    tail_mask = np.random.rand(parent1_tail.size) < 0.5
                    child1_tail = np.where(tail_mask, parent1_tail, parent2_tail)
                    child2_tail = np.where(tail_mask, parent2_tail, parent1_tail)
                else:
                    child1_tail = parent1_tail
                    child2_tail = parent2_tail

                # 将后代展平并存回种群
                self.Chrom[i] = np.concatenate([child1.flatten(), child1_tail])
                self.Chrom[i + 1] = np.concatenate([child2.flatten(), child2_tail])

    def get_adaptive_mutation_prob(self):
        """
        获取自适应变异概率，随着迭代进度增加变异概率
        """
        base_prob = self.prob_mut
        if self.mutation_schedule in {"none", "fixed", "constant"}:
            adaptive_prob = base_prob
        else:
            progress = min(1.0, self.current_gen / max(1.0, (self.max_iter * 0.7)))
            adaptive_prob = base_prob * (1 + progress)
        if self.mutation_stagnation_boost and self._stagnation_count > 0:
            adaptive_prob *= self.mutation_stagnation_multiplier
        max_prob = base_prob * self.mutation_max_multiplier
        adaptive_prob = min(adaptive_prob, max_prob, 1.0)
        self._current_mutation_prob = adaptive_prob
        return adaptive_prob

    def mutation(self):
        """
        增强的变异操作：实现多种变异策略和自适应变异概率
        """
        # 获取自适应变异概率
        adaptive_prob = self.get_adaptive_mutation_prob()

        # 确定各种变异策略的概率
        small_step_prob = 0.4  # 小步长变异概率
        large_step_prob = 0.2  # 大步长变异概率
        gaussian_prob = 0.3    # 高斯变异概率
        reset_prob = 0.1       # 重置变异概率

        for i in range(self.size_pop):
            for j in range(self.n_dim):
                if np.random.rand() < adaptive_prob:
                    # 选择变异策略
                    strategy = np.random.choice(['small', 'large', 'gaussian', 'reset'],
                                              p=[small_step_prob, large_step_prob, gaussian_prob, reset_prob])

                    lb_val = self.lb[j] if isinstance(self.lb, np.ndarray) else self.lb
                    ub_val = self.ub[j] if isinstance(self.ub, np.ndarray) else self.ub
                    range_val = ub_val - lb_val

                    if strategy == 'small':
                        # 小步长变异：±1
                        if np.random.rand() < 0.5:
                            self.Chrom[i, j] += 1
                        else:
                            self.Chrom[i, j] -= 1

                    elif strategy == 'large':
                        # 大步长变异：±(2~5)
                        step = np.random.randint(2, 6)
                        if np.random.rand() < 0.5:
                            self.Chrom[i, j] += step
                        else:
                            self.Chrom[i, j] -= step

                    elif strategy == 'gaussian':
                        # 高斯变异：适合地形问题的平滑变异
                        sigma = range_val * 0.05  # 标准差为参数范围的5%
                        delta = int(np.random.normal(0, sigma))
                        self.Chrom[i, j] += delta

                    else:  # reset
                        # 重置变异：完全随机重置该位置的值
                        self.Chrom[i, j] = np.random.randint(lb_val, ub_val + 1)

                    # 确保值在边界内
                    self.Chrom[i, j] = np.clip(self.Chrom[i, j], lb_val, ub_val)

    def ranking(self):
        """种群排序"""
        self.Y = np.array(self.Y)
        sorted_indices = np.argsort(self.Y)
        self.Chrom = self.Chrom[sorted_indices]
        self.Y = self.Y[sorted_indices]

    def reduce_population_size(self):
        """动态减少种群大小"""
        new_size_pop = int(self.size_pop * self.decay_rate)
        if new_size_pop < self.min_size_pop:
            new_size_pop = self.min_size_pop

        if new_size_pop < self.size_pop:
            logging.info(f"Reducing population size from {self.size_pop} to {new_size_pop}")
            self.size_pop = new_size_pop
            self.Chrom = self.Chrom[:self.size_pop]

    def inject_diversity(self, best_x=None, resampled_dem=None):
        """
        多样性注入机制：当算法停滞时触发，帮助跳出局部最优

        参数:
        - best_x: 当前最优解
        - resampled_dem: 重采样后的DEM数据（用于基于地形的注入）
        """
        logging.info("Injecting diversity to escape local optima")

        # 确定保留的精英个体数量
        elite_size = max(1, int(0.1 * self.size_pop))  # 保留10%的精英
        elites = self.Chrom[:elite_size].copy()  # 假设种群已经排序

        # 确定注入策略的比例
        random_fraction = self.diversity_random_fraction
        best_based_fraction = self.diversity_best_fraction
        terrain_fraction = self.diversity_terrain_fraction

        # 计算各策略的个体数量
        random_size, best_based_size, terrain_size = _allocate_counts(
            self.size_pop - elite_size,
            (random_fraction, best_based_fraction, terrain_fraction),
        )

        lb_val, ub_val = _safe_int_bounds(self.lb, self.ub)

        # 创建新种群。先填充合法随机个体，避免不可用策略留下越界 0 值。
        new_population = np.random.randint(self.lb, self.ub + 1, size=(self.size_pop, self.n_dim))
        new_population[:elite_size] = elites
        if best_x is not None:
            new_population[0] = np.clip(np.asarray(best_x, dtype=int).reshape(-1), self.lb, self.ub)

        # 1. 随机注入
        if random_size > 0:
            random_individuals = np.random.randint(
                self.lb, self.ub + 1,
                size=(random_size, self.n_dim)
            )
            new_population[elite_size:elite_size+random_size] = random_individuals

        # 2. 基于最优解的变异注入
        if best_based_size > 0 and best_x is not None:
            best_based_start = elite_size + random_size
            best_based_end = best_based_start + best_based_size

            for i in range(best_based_start, best_based_end):
                # 复制最优解并添加较大的变异
                individual = best_x.copy()

                # 大幅度变异，变异比例为20%-40%
                mutation_ratio = 0.2 + 0.2 * np.random.rand()
                mutation_indices = np.random.choice(
                    self.n_dim,
                    size=int(mutation_ratio * self.n_dim),
                    replace=False
                )

                # 对选中的位置进行变异
                for idx in mutation_indices:
                    lb_val = self.lb[idx] if isinstance(self.lb, np.ndarray) else self.lb
                    ub_val = self.ub[idx] if isinstance(self.ub, np.ndarray) else self.ub
                    range_val = ub_val - lb_val

                    # 大幅度变异
                    if np.random.rand() < 0.5:
                        # 高斯变异
                        sigma = range_val * 0.2  # 较大的标准差
                        delta = int(np.random.normal(0, sigma))
                        individual[idx] += delta
                    else:
                        # 完全随机重置
                        individual[idx] = np.random.randint(lb_val, ub_val + 1)

                    # 确保在边界内
                    individual[idx] = np.clip(individual[idx], lb_val, ub_val)

                new_population[i] = individual

        # 3. 基于地形的注入
        if terrain_size > 0 and resampled_dem is not None and self.low_res_shape is not None:
            terrain_start = elite_size + random_size + best_based_size
            spatial_dim = int(np.prod(self.low_res_shape))
            initial_vector = _terrain_prior_vector(resampled_dem, self.low_res_shape, self.lb, self.ub, n_dim=self.n_dim)

            for i in range(terrain_start, self.size_pop):
                # 添加较大的随机噪声
                noise_level = int((ub_val - lb_val) * 0.2)  # 噪声级别为参数范围的20%
                noise = np.zeros(self.n_dim, dtype=int)
                noise[:spatial_dim] = np.random.randint(-noise_level, noise_level + 1, size=spatial_dim)
                individual = initial_vector + noise
                individual = np.clip(individual, self.lb, self.ub)
                new_population[i] = individual

        if self.n_dim > self.spatial_dim and elite_size < self.size_pop:
            new_population[elite_size:, self.spatial_dim:] = _stratified_tail_population(
                self.size_pop - elite_size,
                self.lb,
                self.ub,
                spatial_dim=self.spatial_dim,
                n_dim=self.n_dim,
            )

        # 更新种群
        self.Chrom = new_population
        self.diversity_injections += 1
        logging.info(f"Diversity injected: {elite_size} elites, {random_size} random, "
                    f"{best_based_size} best-based, {terrain_size} terrain-based")

    def seed_population_from_best(self, best_x, resampled_dem=None):
        """Initialize this stage around a previous best while preserving diversity."""
        if best_x is None:
            return
        if self.Chrom is None:
            self.Chrom = np.random.randint(self.lb, self.ub + 1, size=(self.size_pop, self.n_dim))
        best_x = np.asarray(best_x, dtype=int).reshape(-1)
        self.Chrom[0] = np.clip(best_x, self.lb, self.ub)
        if self.size_pop > 1:
            old = self.Chrom.copy()
            injections_before = self.diversity_injections
            self.inject_diversity(best_x=best_x, resampled_dem=resampled_dem)
            self.diversity_injections = injections_before
            self.Chrom[0] = np.clip(best_x, self.lb, self.ub)
            if self.size_pop > 2:
                self.Chrom[1] = old[0] if old.shape[0] else self.Chrom[1]

    def run(self, max_iter=None, patience=None, resampled_dem=None):
        """运行遗传算法优化"""
        self.max_iter = max_iter or self.max_iter
        patience_value = patience if patience is not None else self.patience
        best = None
        no_improve_count = 0
        consecutive_full_failures = 0
        fitness_history = []

        # 多样性注入参数
        diversity_threshold = self.diversity_threshold
        if diversity_threshold is None:
            diversity_threshold = max(1, min(25, patience_value // 2))
        else:
            diversity_threshold = max(1, int(diversity_threshold))
        diversity_cooldown = self.diversity_cooldown
        last_injection_gen = -diversity_cooldown  # 上次注入的代数

        try:
            # 重置代数计数器
            self.current_gen = 0

            for gen in tqdm(range(self.max_iter), desc="Generations", ncols=100):
                self.current_gen = gen  # 更新当前代数

                # 评估当前种群
                self.X = self.Chrom.copy()
                gen_cache_hits_before = self.cache_hits
                gen_cache_misses_before = self.cache_misses
                raw_y = self.evaluate_population()
                self.Y, failure_mask = _sanitize_fitness_values(raw_y)
                failure_count = int(np.count_nonzero(failure_mask))
                failure_rate = failure_count / max(1, len(self.Y))
                if failure_count:
                    logging.warning(
                        "Generation %s: %s/%s fitness values invalid and replaced with penalty %.3f",
                        gen,
                        failure_count,
                        len(self.Y),
                        1.0,
                    )
                if failure_rate >= 1.0:
                    consecutive_full_failures += 1
                else:
                    consecutive_full_failures = 0
                if failure_rate >= self.failure_threshold or consecutive_full_failures >= self.max_consecutive_full_failures:
                    raise RuntimeError(
                        f"GA objective failed for {failure_count}/{len(self.Y)} candidates "
                        f"(failure_rate={failure_rate:.0%}). 请检查 FastScape/LPIPS/Pecube 或输入数据。"
                    )

                # 排序后立即记录本代最优个体。此时 self.Chrom 和 self.Y 是对应关系；
                # 后续 selection/crossover/mutation 会改写种群，必须先 copy 固定下来。
                self.ranking()
                current_best = (self.Chrom[0].copy(), float(self.Y[0]))

                if best is None or current_best[1] < best[1]:
                    best = current_best
                    no_improve_count = 0
                    fitness_history.append(best[1])
                else:
                    no_improve_count += 1
                    fitness_history.append(best[1])
                self._stagnation_count = no_improve_count

                generation_best = float(np.min(self.Y))
                generation_mean = float(np.mean(self.Y))
                generation_std = float(np.std(self.Y))
                unique_chromosomes = int(len({_chromosome_key(row) for row in self.Chrom}))

                # 选择、交叉和变异会生成下一代，不能再用这之后的种群记录当前代最优。
                self.selection()
                self.crossover()
                self.mutation()

                # 检查是否需要注入多样性
                if (no_improve_count >= diversity_threshold and
                    gen - last_injection_gen >= diversity_cooldown):
                    logging.info(f"No improvement for {no_improve_count} generations, injecting diversity")
                    self.inject_diversity(best_x=best[0], resampled_dem=resampled_dem)
                    last_injection_gen = gen
                    no_improve_count = 0  # 重置计数器
                    injected = True
                else:
                    injected = False

                # 动态调整种群大小
                self.reduce_population_size()

                self.history_rows.append({
                    "stage": self.stage_name,
                    "generation": int(gen),
                    "best_fitness": float(best[1]),
                    "generation_best_fitness": generation_best,
                    "mean_fitness": generation_mean,
                    "std_fitness": generation_std,
                    "unique_chromosomes": unique_chromosomes,
                    "cache_hits": int(self.cache_hits - gen_cache_hits_before),
                    "cache_misses": int(self.cache_misses - gen_cache_misses_before),
                    "cache_hits_total": int(self.cache_hits),
                    "cache_misses_total": int(self.cache_misses),
                    "cache_size": int(len(self.fitness_cache)),
                    "diversity_injected": bool(injected),
                    "diversity_injections_total": int(self.diversity_injections),
                    "mutation_probability": float(self._current_mutation_prob),
                    "population_size": int(self.size_pop),
                    "failure_count": failure_count,
                    "failure_rate": float(failure_rate),
                })

                logging.info(f"Generation {gen}: Best fitness = {best[1]}, Population size = {self.size_pop}")
                # 检查早停条件
                if no_improve_count >= patience_value:
                    logging.info(f"Early stopping after {patience_value} generations without improvement")
                    break

        except Exception as e:
            logging.error(f"Error in GA run: {e}")
            logging.exception("Exception details:")
            raise

        self.best_x, self.best_y = best if best is not None else (None, float('inf'))
        return self.best_x, self.best_y, fitness_history

    def metrics(self):
        total_cache = self.cache_hits + self.cache_misses
        return {
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_size": int(len(self.fitness_cache)),
            "cache_hit_rate": float(self.cache_hits / total_cache) if total_cache else 0.0,
            "diversity_injections": int(self.diversity_injections),
            "history_rows": len(self.history_rows),
            "best_y": float(self.best_y),
        }

def optimize_uplift_ga(obj_func, resampled_dem, LOW_RES_SHAPE, ORIGINAL_SHAPE,
                      ga_params, model_params, n_jobs=-1, run_mode='cached'):
    """
    优化隆升率场

    参数:
    - obj_func: 目标函数
    - resampled_dem: 重采样后的DEM数据
    - LOW_RES_SHAPE: 低分辨率形状
    - ORIGINAL_SHAPE: 原始形状
    - ga_params: 遗传算法参数字典
    - model_params: 模型参数字典
    - n_jobs: 并行任务数
    - run_mode: 运行模式

    返回:
    - best_x: 最优解（隆升率场）
    - best_y: 最优适应度值
    - fitness_history: 适应度历史
    """
    try:
        if 'random_seed' in ga_params and ga_params['random_seed'] is not None:
            np.random.seed(int(ga_params['random_seed']))

        uplift_dim = LOW_RES_SHAPE[0] * LOW_RES_SHAPE[1]
        uplift_precision = _get_uplift_precision(ga_params)
        encoded_lb, encoded_ub = _encode_uplift_bounds(ga_params['lb'], ga_params['ub'], uplift_precision)
        history_settings = _get_history_settings(ga_params)
        n_dim = uplift_dim + int(history_settings["stage_count"])
        lb_array = np.concatenate([np.full(uplift_dim, encoded_lb, dtype=int), history_settings["encoded_lb"]])
        ub_array = np.concatenate([np.full(uplift_dim, encoded_ub, dtype=int), history_settings["encoded_ub"]])

        def decoded_obj_func(encoded_vector):
            return obj_func(
                _decode_candidate_vector(
                    encoded_vector,
                    uplift_dim=uplift_dim,
                    uplift_precision=uplift_precision,
                    history_settings=history_settings,
                )
            )

        logging.info(
            "Uplift encoding: real range %.6g..%.6g mm/yr, precision %.6g mm/yr, "
            "integer range %s..%s, uplift_dim=%s, history_stages=%s",
            ga_params['lb'],
            ga_params['ub'],
            uplift_precision,
            encoded_lb,
            encoded_ub,
            uplift_dim,
            history_settings["stage_count"],
        )

        # scikit-opt 的 set_run_mode 只是可选优化；部分环境导入 sko.tools
        # 会重复设置 multiprocessing start method，失败时不应中断自定义 GA。
        if run_mode:
            try:
                from sko.tools import set_run_mode
                set_run_mode(decoded_obj_func, run_mode)
            except Exception as e:
                logging.warning(f"Skipping scikit-opt run mode '{run_mode}': {e}")

        search_strategy = str(ga_params.get("search_strategy", "staged")).strip().lower()
        if search_strategy not in {"single", "staged"}:
            raise ValueError(f"search_strategy 必须是 single 或 staged，当前为 {search_strategy!r}")

        if search_strategy == "staged":
            stages = ga_params.get("stages") or _default_stages_from_params(ga_params)
        else:
            stages = [_single_stage_from_params(ga_params)]

        logging.info("Starting genetic algorithm optimization (%s, %s stage(s))...", search_strategy, len(stages))
        shared_cache = {} if _as_bool(ga_params.get("enable_fitness_cache"), True) else None
        all_history_rows = []
        stage_metrics = []
        overall_best_x = None
        overall_best_y = float("inf")
        overall_history = []
        previous_best_x = None

        for stage_index, stage in enumerate(stages, start=1):
            stage_name = str(stage.get("name", f"stage{stage_index}"))
            stage_pop = int(stage.get("population_size", ga_params.get("pop", 10)))
            stage_iter = int(stage.get("max_iterations", ga_params.get("max_iter", 10)))
            stage_patience = int(stage.get("patience", ga_params.get("patience", stage_iter)))
            stage_prob_mut = float(stage.get("mutation_probability", ga_params.get("prob_mut", 0.05)))
            stage_prob_cross = float(stage.get("cross_probability", ga_params.get("prob_cross", 0.7)))

            ga = MyGA(
                func=decoded_obj_func,
                n_dim=n_dim,
                size_pop=stage_pop,
                max_iter=stage_iter,
                prob_mut=stage_prob_mut,
                lb=lb_array,
                ub=ub_array,
                precision=uplift_precision,
                decay_rate=float(stage.get("decay_rate", ga_params.get("decay_rate", 1.0))),
                min_size_pop=int(stage.get("min_population_size", stage.get("min_size_pop", ga_params.get("min_size_pop", stage_pop)))),
                patience=stage_patience,
                n_jobs=n_jobs,
                enable_fitness_cache=shared_cache is not None,
                diversity_threshold=stage.get("diversity_threshold", ga_params.get("diversity_threshold")),
                diversity_cooldown=stage.get("diversity_cooldown", ga_params.get("diversity_cooldown", 10)),
                diversity_random_fraction=stage.get("diversity_random_fraction", ga_params.get("diversity_random_fraction", 0.2)),
                diversity_best_fraction=stage.get("diversity_best_fraction", ga_params.get("diversity_best_fraction", 0.5)),
                diversity_terrain_fraction=stage.get("diversity_terrain_fraction", ga_params.get("diversity_terrain_fraction", 0.3)),
                mutation_schedule=stage.get("mutation_schedule", ga_params.get("mutation_schedule", "adaptive")),
                mutation_max_multiplier=stage.get("mutation_max_multiplier", ga_params.get("mutation_max_multiplier", 2.0)),
                mutation_stagnation_boost=_as_bool(stage.get("mutation_stagnation_boost", ga_params.get("mutation_stagnation_boost")), False),
                mutation_stagnation_multiplier=stage.get("mutation_stagnation_multiplier", ga_params.get("mutation_stagnation_multiplier", 1.5)),
            )
            ga.stage_name = stage_name
            if shared_cache is not None:
                ga.fitness_cache = shared_cache
            ga.init_population_based_on_terrain(
                matrix=resampled_dem,
                lb=lb_array,
                ub=ub_array,
                low_res_shape=LOW_RES_SHAPE,
                noise_level=int(stage.get("terrain_noise_level", ga_params.get("terrain_noise_level", 3))),
                random_fraction=float(stage.get("random_fraction", ga_params.get("random_fraction", 0.2))),
            )
            ga.prob_cross = stage_prob_cross
            if previous_best_x is not None:
                ga.seed_population_from_best(previous_best_x, resampled_dem=resampled_dem)

            best_x, best_y, fitness_history = ga.run(
                max_iter=stage_iter,
                patience=stage_patience,
                resampled_dem=resampled_dem
            )
            previous_best_x = best_x.copy() if best_x is not None else previous_best_x
            if best_y < overall_best_y:
                overall_best_x = best_x.copy() if best_x is not None else None
                overall_best_y = float(best_y)
            overall_history.extend([float(v) for v in fitness_history])
            all_history_rows.extend(ga.history_rows)
            metrics = ga.metrics()
            metrics.update({
                "stage": stage_name,
                "stage_index": stage_index,
                "inherited_previous_best": bool(stage_index > 1),
                "population_size": stage_pop,
                "max_iterations": stage_iter,
                "mutation_probability": stage_prob_mut,
                "cross_probability": stage_prob_cross,
                "patience": stage_patience,
                "best_fitness": float(best_y),
                "is_global_best_stage": False,
            })
            stage_metrics.append(metrics)

        best_stage_name = None
        for metrics in stage_metrics:
            if abs(metrics["best_fitness"] - overall_best_y) <= 1e-15:
                metrics["is_global_best_stage"] = True
                best_stage_name = metrics["stage"]
                break

        diagnostics_dir = ga_params.get("diagnostics_dir")
        if diagnostics_dir:
            _write_ga_diagnostics(
                diagnostics_dir,
                history_rows=all_history_rows,
                stage_metrics=stage_metrics,
                ga_metrics={
                    "search_strategy": search_strategy,
                    "stage_count": len(stages),
                    "best_fitness": float(overall_best_y),
                    "best_stage": best_stage_name,
                    "cache_hits": int(sum(row.get("cache_hits", 0) for row in all_history_rows)),
                    "cache_misses": int(sum(row.get("cache_misses", 0) for row in all_history_rows)),
                    "cache_size": int(len(shared_cache) if shared_cache is not None else 0),
                    "diversity_injections": int(sum(m.get("diversity_injections", 0) for m in stage_metrics)),
                },
            )

        decoded_best_x = None
        if overall_best_x is not None:
            decoded_best_x = _decode_candidate_vector(
                overall_best_x,
                uplift_dim=uplift_dim,
                uplift_precision=uplift_precision,
                history_settings=history_settings,
            )
        return decoded_best_x, overall_best_y, overall_history

    except Exception as e:
        logging.error(f"Error in optimize_uplift_ga: {e}")
        logging.exception("Exception details:")
        raise


def _single_stage_from_params(ga_params):
    return {
        "name": "single",
        "population_size": ga_params.get("pop", 10),
        "max_iterations": ga_params.get("max_iter", 10),
        "mutation_probability": ga_params.get("prob_mut", 0.05),
        "cross_probability": ga_params.get("prob_cross", 0.7),
        "patience": ga_params.get("patience", ga_params.get("max_iter", 10)),
        "min_population_size": ga_params.get("min_size_pop", ga_params.get("pop", 10)),
    }


def _default_stages_from_params(ga_params):
    pop = int(ga_params.get("pop", 8))
    max_iter = int(ga_params.get("max_iter", 8))
    prob_mut = float(ga_params.get("prob_mut", 0.08))
    prob_cross = float(ga_params.get("prob_cross", 0.7))
    patience = int(ga_params.get("patience", max_iter))
    return [
        {
            "name": "coarse",
            "population_size": pop,
            "max_iterations": max_iter,
            "mutation_probability": max(prob_mut, 0.12),
            "cross_probability": max(prob_cross, 0.75),
            "patience": patience,
            "min_population_size": pop,
        },
        {
            "name": "refine",
            "population_size": pop,
            "max_iterations": max_iter,
            "mutation_probability": prob_mut,
            "cross_probability": prob_cross,
            "patience": patience,
            "min_population_size": pop,
        },
        {
            "name": "verify",
            "population_size": max(pop, int(round(pop * 1.25))),
            "max_iterations": max(1, int(round(max_iter * 0.75))),
            "mutation_probability": max(prob_mut * 0.6, 0.01),
            "cross_probability": min(prob_cross, 0.65),
            "patience": max(1, int(round(patience * 0.75))),
            "min_population_size": max(pop, int(round(pop * 1.25))),
        },
    ]


def _write_ga_diagnostics(diagnostics_dir, *, history_rows, stage_metrics, ga_metrics):
    diagnostics_dir = os.fspath(diagnostics_dir)
    tables_dir = os.path.join(diagnostics_dir, "tables")
    metrics_dir = os.path.join(diagnostics_dir, "metrics")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    history_path = os.path.join(tables_dir, "ga_history.csv")
    fieldnames = [
        "stage", "generation", "best_fitness", "generation_best_fitness", "mean_fitness",
        "std_fitness", "unique_chromosomes", "cache_hits", "cache_misses",
        "cache_hits_total", "cache_misses_total", "cache_size", "diversity_injected",
        "diversity_injections_total", "mutation_probability", "population_size",
        "failure_count", "failure_rate",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    total = ga_metrics["cache_hits"] + ga_metrics["cache_misses"]
    ga_metrics = dict(ga_metrics)
    ga_metrics["cache_hit_rate"] = float(ga_metrics["cache_hits"] / total) if total else 0.0
    ga_metrics["history_rows"] = len(history_rows)
    with open(os.path.join(metrics_dir, "ga_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ga_metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(metrics_dir, "stage_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(stage_metrics, f, indent=2, ensure_ascii=False)
