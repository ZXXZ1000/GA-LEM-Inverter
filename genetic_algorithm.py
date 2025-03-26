# genetic_algorithm.py
import numpy as np
from scipy.ndimage import gaussian_filter
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import psutil
from tqdm.notebook import tqdm
import time
from joblib import Parallel, delayed

def parallel_fitness(obj_func, X, n_jobs=-1):
    """并行计算适应度函数"""
    try:
        fitness_values = Parallel(n_jobs=n_jobs)(delayed(obj_func)(x) for x in X)
        return np.array(fitness_values)
    except Exception as e:
        logging.error(f"并行计算失败,切换为串行: {e}")
        return np.array([obj_func(x) for x in X])

class MyGA:
    """自定义遗传算法类，专门用于隆升率场优化"""
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, prob_mut=0.01,
                 lb=-1, ub=1, constraint_eq=tuple(), constraint_ueq=tuple(), 
                 precision=1, decay_rate=0.95, min_size_pop=10, patience=40,
                 n_jobs=-1):
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
        
        # 初始化种群和适应度
        self.Chrom = None
        self.X = None  # 种群的表现型
        self.Y = None  # 种群的适应度
        self.low_res_shape = None  # 低分辨率形状
        
        # 记录最优解
        self.best_x = None
        self.best_y = float('inf')

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
        if low_res_shape is None or np.prod(low_res_shape) != self.n_dim:
            raise ValueError(f"Invalid low_res_shape: {low_res_shape}. Expected product to be {self.n_dim}")

        self.low_res_shape = low_res_shape
        terrain_pop_size = int(self.size_pop * (1 - random_fraction))
        random_pop_size = self.size_pop - terrain_pop_size
        initial_population = np.zeros((self.size_pop, self.n_dim), dtype=int)

        if terrain_pop_size > 0:
            # 对地形进行平滑处理
            smoothed_matrix = gaussian_filter(matrix, sigma=5)
            
            # 调整矩阵大小
            if smoothed_matrix.shape != low_res_shape:
                from skimage.transform import resize
                smoothed_matrix = resize(smoothed_matrix, 
                                      low_res_shape,
                                      mode='edge',
                                      anti_aliasing=True)
            
            # 映射到参数范围
            min_val = np.nanmin(smoothed_matrix)
            max_val = np.nanmax(smoothed_matrix)
            lb_val = lb if isinstance(lb, (int, float)) else lb[0]
            ub_val = ub if isinstance(ub, (int, float)) else ub[0]
            
            scaled_matrix = (smoothed_matrix - min_val) / (max_val - min_val)
            scaled_matrix = lb_val + (ub_val - lb_val) * scaled_matrix
            scaled_matrix = np.clip(scaled_matrix, lb_val, ub_val).astype(int)
            
            initial_vector = scaled_matrix.flatten()
            terrain_individuals = np.zeros((terrain_pop_size, self.n_dim), dtype=int)
            
            for i in range(terrain_pop_size):
                noise = np.random.randint(-noise_level, noise_level + 1, size=self.n_dim)
                individual = initial_vector + noise
                individual = np.clip(individual, lb_val, ub_val)
                terrain_individuals[i, :] = individual
            
            initial_population[:terrain_pop_size, :] = terrain_individuals  

        if random_pop_size > 0:
            lb_val = lb if isinstance(lb, (int, float)) else lb[0]
            ub_val = ub if isinstance(ub, (int, float)) else ub[0]
            random_individuals = np.random.randint(
                lb_val,
                ub_val + 1,
                size=(random_pop_size, self.n_dim)
            )
            initial_population[terrain_pop_size:, :] = random_individuals

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
        
        # 动态调整块大小
        block_size = self.get_adaptive_block_size()
        block_size = min(block_size, min(rows // 3, cols // 3))  # 确保块大小不超过矩阵尺寸的1/3
        
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
                    if max_row > 0 and max_col > 0:
                        start_row = np.random.randint(0, max_row)
                        start_col = np.random.randint(0, max_col)
                        
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
        
        tournament_size = 3
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
                # 将染色体重塑为2D矩阵 
                parent1 = self.Chrom[i].reshape(self.low_res_shape)
                parent2 = self.Chrom[i + 1].reshape(self.low_res_shape)
                
                # 生成新后代
                child1, child2 = self.spatial_crossover(parent1, parent2)
                
                # 将后代展平并存回种群
                self.Chrom[i] = child1.flatten()
                self.Chrom[i + 1] = child2.flatten()

    def get_adaptive_mutation_prob(self):
        """
        获取自适应变异概率，随着迭代进度增加变异概率
        """
        base_prob = self.prob_mut
        progress = min(1.0, self.current_gen / (self.max_iter * 0.7))
        # 随着迭代进度增加变异概率，最大增加到原始概率的2倍
        adaptive_prob = base_prob * (1 + progress)
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
        random_fraction = 0.2       # 完全随机注入
        best_based_fraction = 0.5   # 基于最优解的变异注入
        terrain_fraction = 0.3      # 基于地形的注入
        
        # 计算各策略的个体数量
        random_size = int(random_fraction * (self.size_pop - elite_size))
        best_based_size = int(best_based_fraction * (self.size_pop - elite_size))
        terrain_size = self.size_pop - elite_size - random_size - best_based_size
        
        # 创建新种群
        new_population = np.zeros((self.size_pop, self.n_dim), dtype=int)
        new_population[:elite_size] = elites
        
        # 1. 随机注入
        if random_size > 0:
            lb_val = self.lb[0] if isinstance(self.lb, np.ndarray) else self.lb
            ub_val = self.ub[0] if isinstance(self.ub, np.ndarray) else self.ub
            random_individuals = np.random.randint(
                lb_val, ub_val + 1, 
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
            
            # 对地形进行平滑处理
            smoothed_matrix = gaussian_filter(resampled_dem, sigma=5)
            
            # 调整矩阵大小
            if smoothed_matrix.shape != self.low_res_shape:
                from skimage.transform import resize
                smoothed_matrix = resize(smoothed_matrix, 
                                      self.low_res_shape,
                                      mode='edge',
                                      anti_aliasing=True)
            
            # 映射到参数范围
            min_val = np.nanmin(smoothed_matrix)
            max_val = np.nanmax(smoothed_matrix)
            lb_val = self.lb[0] if isinstance(self.lb, np.ndarray) else self.lb
            ub_val = self.ub[0] if isinstance(self.ub, np.ndarray) else self.ub
            
            scaled_matrix = (smoothed_matrix - min_val) / (max_val - min_val)
            scaled_matrix = lb_val + (ub_val - lb_val) * scaled_matrix
            scaled_matrix = np.clip(scaled_matrix, lb_val, ub_val).astype(int)
            
            initial_vector = scaled_matrix.flatten()
            
            for i in range(terrain_start, self.size_pop):
                # 添加较大的随机噪声
                noise_level = int((ub_val - lb_val) * 0.2)  # 噪声级别为参数范围的20%
                noise = np.random.randint(-noise_level, noise_level + 1, size=self.n_dim)
                individual = initial_vector + noise
                individual = np.clip(individual, lb_val, ub_val)
                new_population[i] = individual
        
        # 更新种群
        self.Chrom = new_population
        logging.info(f"Diversity injected: {elite_size} elites, {random_size} random, "
                    f"{best_based_size} best-based, {terrain_size} terrain-based")
    
    def run(self, max_iter=None, patience=None, resampled_dem=None):
        """运行遗传算法优化"""
        self.max_iter = max_iter or self.max_iter
        patience_value = patience if patience is not None else self.patience
        best = None
        no_improve_count = 0
        fitness_history = []
        
        # 多样性注入参数
        diversity_threshold = min(25, patience_value // 2)  # 多样性注入触发阈值
        diversity_cooldown = 10  # 多样性注入冷却期
        last_injection_gen = -diversity_cooldown  # 上次注入的代数
        
        try:
            # 重置代数计数器
            self.current_gen = 0
            
            for gen in tqdm(range(self.max_iter), desc="Generations", ncols=100):
                self.current_gen = gen  # 更新当前代数
                
                # 评估当前种群
                self.X = self.Chrom
                self.Y = parallel_fitness(self.func, self.X, n_jobs=self.n_jobs)

                # 排序和选择
                self.ranking()
                self.selection()
                
                # 交叉和变异
                self.crossover()
                self.mutation()

                # 更新最优解
                gen_best_index = self.Y.argmin()
                current_best = (self.X[gen_best_index], self.Y[gen_best_index])
                
                if best is None or current_best[1] < best[1]:
                    best = current_best
                    no_improve_count = 0
                    fitness_history.append(best[1])
                else:
                    no_improve_count += 1
                    fitness_history.append(best[1])
                
                # 检查是否需要注入多样性
                if (no_improve_count >= diversity_threshold and 
                    gen - last_injection_gen >= diversity_cooldown):
                    logging.info(f"No improvement for {no_improve_count} generations, injecting diversity")
                    self.inject_diversity(best_x=best[0], resampled_dem=resampled_dem)
                    last_injection_gen = gen
                    no_improve_count = 0  # 重置计数器
                
                # 动态调整种群大小
                self.reduce_population_size()

                logging.info(f"Generation {gen}: Best fitness = {best[1]}, Population size = {self.size_pop}")
                # 检查早停条件
                if no_improve_count >= patience_value:
                    logging.info(f"Early stopping after {patience_value} generations without improvement")
                    break

        except Exception as e:
            logging.error(f"Error in GA run: {e}")
            logging.exception("Exception details:")

        self.best_x, self.best_y = best if best is not None else (None, float('inf'))
        return self.best_x, self.best_y, fitness_history

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
        # 设置遗传算法维度
        n_dim = LOW_RES_SHAPE[0] * LOW_RES_SHAPE[1]
        lb_array = np.full(n_dim, ga_params['lb'])
        ub_array = np.full(n_dim, ga_params['ub'])

        # 设置运行模式
        from sko.tools import set_run_mode
        set_run_mode(obj_func, run_mode)

        # 创建遗传算法实例
        ga = MyGA(
            func=obj_func,
            n_dim=n_dim,
            size_pop=ga_params['pop'],
            max_iter=ga_params['max_iter'],
            prob_mut=ga_params['prob_mut'],
            lb=lb_array,
            ub=ub_array,
            precision=1,
            decay_rate=ga_params.get('decay_rate', 0.95),
            min_size_pop=ga_params.get('min_size_pop', 10),
            patience=ga_params.get('patience', 40),
            n_jobs=n_jobs
        )
        
        # 基于地形初始化种群
        ga.init_population_based_on_terrain(
            matrix=resampled_dem,
            lb=ga_params['lb'],
            ub=ga_params['ub'],
            low_res_shape=LOW_RES_SHAPE,
            noise_level=3,
            random_fraction=0.2
        )
        
        ga.prob_cross = ga_params['prob_cross']

        # 运行优化
        logging.info("Starting genetic algorithm optimization...")
        best_x, best_y, fitness_history = ga.run(
            max_iter=ga_params['max_iter'],
            patience=ga_params.get('patience', 40),
            resampled_dem=resampled_dem
        )

        return best_x, best_y, fitness_history

    except Exception as e:
        logging.error(f"Error in optimize_uplift_ga: {e}")
        logging.exception("Exception details:")
        return None, float('inf'), None