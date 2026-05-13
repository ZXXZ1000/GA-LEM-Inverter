# run_synthetic_experiment.py

import numpy as np
import matplotlib
# 设置matplotlib后端为非交互式
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import logging
import os
import time
import json
import warnings
from datetime import datetime
from typing import Tuple, Dict, Optional, Any

warnings.filterwarnings("ignore", message="dropping variables using `drop` is deprecated")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

# Import your existing modules
from ga_lem_inverter.pipeline.forward_model import run_fastscape_model
from ga_lem_inverter.pipeline.optimization import optimize_uplift_ga
from ga_lem_inverter.pipeline.visualization import plot_comparison, plot_uplift_distribution, plot_3d_surface
from ga_lem_inverter.pipeline.preprocessing import interpolate_uplift_cv, interpolate_uplift_cv1
from ga_lem_inverter.pipeline.synthetic_erosion import create_synthetic_erosion_field
from ga_lem_inverter.pipeline.fitness import terrain_similarity
from ga_lem_inverter.pipeline.array_io import safe_save_array, safe_load_array
from ga_lem_inverter.pipeline.analysis import ResultAnalyzer
from ga_lem_inverter.outputs import RunContext
from ga_lem_inverter.workflows.common import default_synthetic_shape, run_synthetic_case


def run_synthetic_workflow(config, context: RunContext) -> dict[str, Any]:
    """Run the beginner-facing synthetic validation experiment."""
    pattern = config.get("Synthetic", "pattern", fallback="simple").strip() or "simple"
    shape = default_synthetic_shape(config, "Synthetic")
    scale_factor = config.getint("Optimization", "scale_factor", fallback=8)
    logging.info(
        "运行 synthetic 模式: pattern=%s, shape=%s, scale_factor=%s",
        pattern,
        shape,
        scale_factor,
    )
    return run_synthetic_case(
        config=config,
        context=context,
        pattern=pattern,
        shape=shape,
        scale_factor=scale_factor,
        output_prefix=f"synthetic_{pattern}",
    )

class SyntheticExperiment:
    def __init__(self, config_path: str = None):
        warnings.warn(
            "SyntheticExperiment 是旧兼容类。普通用户和新代码请通过 config.ini 设置 mode=synthetic，"
            "然后运行 python runner.py。",
            DeprecationWarning,
            stacklevel=2,
        )
        """
        初始化实验设置

        参数:
        - config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 默认配置用于内部兼容路径。普通用户请通过 config.ini 设置
        # mode = synthetic，然后运行 python runner.py。
        #
        # 如果要做正式实验，通常只需要改下面几类参数：
        # 1. experiment.shape：把 (64, 64) 改成 (100, 100) 或更高分辨率。
        # 2. experiment.patterns：把 ['simple'] 改成 ['simple', 'medium', 'complex']。
        # 3. ga_params.pop / max_iter：把 2 / 1 提高到 50-100 / 100-200。
        # 4. ga_params.n_jobs：按 CPU 核心数调整，-1 表示使用所有核心。
        # 5. fitness.use_lpips：默认 True，使用 LPIPS 深度感知相似度；只做基础诊断时可设为 False。
        self.config = {
            'experiment': {
                # 固定随机种子，保证 demo 在不同电脑上首次运行结果相近。
                'random_seed': 42,
                # 合成 DEM 的网格大小。默认 64x64 是为了满足地形特征提取的最小尺寸。
                'shape': (64, 64),
                # 默认只跑 simple，确保首次运行不会过慢。
                'patterns': ['simple'],
                # 降维因子；64/8=8，因此 GA 只需要优化 8x8 个变量。
                'scale_factor': 8,
                # 输出目录。每次运行会在该目录下生成带时间戳的子目录。
                'output_base_dir': 'outputs'
            },
            'ga_params': {
                # 种群大小。demo 用 2；正式实验建议 50-100。
                'pop': 2,
                # 最大迭代次数。demo 用 1；正式实验建议 100-200。
                'max_iter': 1,
                # 交叉概率，控制父代组合生成新个体的比例。
                'prob_cross': 0.8,
                # 变异概率，控制随机扰动强度。
                'prob_mut': 0.05,
                # 隆升率搜索下界和上界，单位 mm/yr。
                'lb': 0.1,
                'ub': 1.0,
                # 隆升率搜索步长，单位 mm/yr。GA 内部用整数编码，进入 FastScape 前自动解码。
                'uplift_precision': 0.1,
                # 种群衰减率。demo 不衰减；正式实验可用 0.95-0.98。
                'decay_rate': 1.0,
                # 最小种群数，不能大于 pop。
                'min_size_pop': 2,
                # 早停耐心值：连续多少代没有改进后停止。
                'patience': 1,
                # 并行任务数。demo 固定 1，避免 Windows 首次运行多进程开销。
                'n_jobs': 1
            },
            'model_params': {
                # Fastscape 模型参数。demo 中 total time 较小，保证运行快速。
                'k_sp_base': 6.92e-6,
                'k_sp_fault': 2e-5,
                'd_diff': 19.2,
                'boundary_status': 'fixed_value',
                'area_exp': 0.43,
                'slope_exp': 1,
                'time_total': 1e5,
                'spacing': 900
            },
            'fitness': {
                # True 表示启用 LPIPS 深度感知相似度。安装脚本会提前初始化模型。
                # 如果只想做极轻量基础诊断，可临时改为 False。
                'use_lpips': True
            }
        }

        # 如果提供了配置文件，则加载它
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self._deep_update(self.config, loaded_config)

        np.random.seed(int(self.config['experiment'].get('random_seed', 42)))
        self.config['ga_params']['random_seed'] = int(self.config['experiment'].get('random_seed', 42))

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_output_dir = os.path.join(
            self.config['experiment']['output_base_dir'],
            f'experiment_{timestamp}'
        )
        os.makedirs(self.base_output_dir, exist_ok=True)

        # 配置日志
        self.setup_logging()

        # 保存配置
        with open(os.path.join(self.base_output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    def _deep_update(self, base: Dict, updates: Dict) -> Dict:
        """递归更新配置，避免外部 JSON 只改一项时覆盖整个配置段。"""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def setup_logging(self):
        """配置日志系统"""
        log_file = os.path.join(self.base_output_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def create_synthetic_uplift(self, shape: Tuple[int, int], pattern: str = 'simple') -> np.ndarray:
        """创建合成的uplift rate场"""
        logging.info(f"Creating synthetic uplift field with pattern: {pattern}")
        rows, cols = shape
        x = np.linspace(0, 1, cols)
        y = np.linspace(0, 1, rows)
        X, Y = np.meshgrid(x, y)

        if pattern == 'simple':
            # 简单的高斯分布
            uplift = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
            uplift = 0.5 + 0.5 * uplift  # 范围在0.5-1.0 mm/yr

        elif pattern == 'medium':
            # 两个高斯分布的组合
            uplift1 = np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / 0.1)
            uplift2 = np.exp(-((X - 0.7)**2 + (Y - 0.7)**2) / 0.1)
            uplift = uplift1 + uplift2
            uplift = 0.5 + 0.5 * uplift / uplift.max()  # 归一化到0.5-1.0 mm/yr

        elif pattern == 'complex':
            # 模拟多断层系统
            # 主断层（走向斜切，更陡的梯度）
            main_fault = np.exp(-((0.8*X + 0.6*Y - 0.8)**2) / 0.01) * 0.35  # 减小宽度，降低幅度
            # 共轭断层系统（更清晰的断层带）
            conjugate_fault1 = np.exp(-((0.7*X - 0.7*Y - 0.2)**2) / 0.008) * 0.2
            conjugate_fault2 = np.exp(-((0.6*X - 0.8*Y + 0.3)**2) / 0.008) * 0.2
            # 添加渐变的区域性抬升
            regional_trend = 0.2 * (1 - Y)  # 南北向的渐变趋势
            # 组合所有构造特征
            uplift = 0.4 + main_fault + conjugate_fault1 + conjugate_fault2 + regional_trend
            # 添加小尺度构造起伏（更细致的局部变化）
            local_structure = gaussian_filter(np.random.rand(rows, cols), sigma=6) * 0.05
            uplift += local_structure
            # 确保uplift在合理范围内
            uplift = np.clip(uplift, 0.5, 1.0)

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        logging.info(f"Synthetic uplift field created with shape {uplift.shape}")
        return uplift

    def create_objective_function(self, target_dem: np.ndarray, shape: Tuple[int, int],
                                model_params: Dict) -> callable:
        """创建目标函数"""
        def objective_function(uplift_vector):
            try:
                # 获取低分辨率形状
                scale_factor = self.config['experiment']['scale_factor']
                low_res_shape = (shape[0]//scale_factor, shape[1]//scale_factor)

                # 重塑uplift向量
                uplift_vector = np.array(uplift_vector).reshape(low_res_shape)

                # 使用新的插值函数
                full_res_uplift = interpolate_uplift_cv(uplift_vector, shape)

                # 运行FastScape模型
                generated_elevation = run_fastscape_model(
                    k_sp=model_params['Ksp'],
                    uplift=full_res_uplift,
                    k_diff=model_params['d_diff'],
                    x_size=shape[1],
                    y_size=shape[0],
                    spacing=model_params['spacing'],
                    boundary_status=model_params['boundary_status'],
                    area_exp=model_params['area_exp'],
                    slope_exp=model_params['slope_exp'],
                    time_total=model_params['time_total']
                )

                # 添加新的地形特征评估
                similarity = terrain_similarity(
                    matrix1=target_dem,
                    matrix2=generated_elevation,
                    resolution=model_params['spacing'],
                    smooth_radius=2,  # 调整平滑半径
                    use_lpips=self.config.get('fitness', {}).get('use_lpips', True)
                )

                return 1 - similarity

            except Exception as e:
                logging.error(f"Error in objective function: {e}")
                return np.inf

        return objective_function

    def evaluate_results(self, inverted_uplift: np.ndarray,
                        true_uplift: np.ndarray,
                        simulated_dem: np.ndarray,
                        target_dem: np.ndarray) -> Dict:
        """评估反演结果的质量"""
        logging.info("Evaluating inversion results...")

        # Uplift rate场评估
        corr, _ = pearsonr(inverted_uplift.flatten(), true_uplift.flatten())
        rmse = np.sqrt(mean_squared_error(true_uplift, inverted_uplift))
        rel_error = np.abs(inverted_uplift - true_uplift) / true_uplift
        mean_rel_error = np.mean(rel_error)
        max_error = np.max(np.abs(inverted_uplift - true_uplift))
        min_error = np.min(np.abs(inverted_uplift - true_uplift))

        # 地形评估
        dem_corr, _ = pearsonr(simulated_dem.flatten(), target_dem.flatten())
        dem_rmse = np.sqrt(mean_squared_error(target_dem, simulated_dem))

        metrics = {
            'uplift_correlation': corr,
            'uplift_rmse': rmse,
            'uplift_mean_relative_error': mean_rel_error,
            'uplift_max_error': max_error,
            'uplift_min_error': min_error,
            'dem_correlation': dem_corr,
            'dem_rmse': dem_rmse
        }

        logging.info("Evaluation completed.")
        return metrics

    def save_visualizations(self, output_dir: str, true_uplift: np.ndarray,
                          inverted_uplift: np.ndarray, target_dem: np.ndarray,
                          simulated_dem: np.ndarray):
        """保存所有可视化结果"""
        logging.info("Generating and saving visualizations...")

        # Uplift rate场对比图
        plot_comparison(
            true_uplift, inverted_uplift,
            'True Uplift Rate', 'Inverted Uplift Rate',
            'Uplift Rate (mm/yr)', 'Uplift Rate (mm/yr)',
            cmap='RdBu_r'
        )
        plt.savefig(os.path.join(output_dir, 'uplift_comparison.png'))
        plt.close()

        # 地形对比图
        plot_comparison(
            target_dem, simulated_dem,
            'Target DEM', 'Simulated DEM',
            'Elevation (m)', 'Elevation (m)',
            cmap='terrain'
        )
        plt.savefig(os.path.join(output_dir, 'dem_comparison.png'))
        plt.close()

        # Uplift分布图
        plot_uplift_distribution(inverted_uplift)
        plt.savefig(os.path.join(output_dir, 'uplift_distribution.png'))
        plt.close()

        # 3D地形可视化
        fig_3d = plot_3d_surface(
            data=simulated_dem,
            uplift=inverted_uplift,
            title="3D Terrain with Uplift Field"
        )
        fig_3d.savefig(os.path.join(output_dir, '3d_terrain.png'))
        plt.close(fig_3d)

        logging.info("All visualizations saved.")

    # run_experiment.py 修复版本
    def run_experiment(self, pattern: str) -> Optional[Dict]:
        """运行单个合成实验"""
        logging.info(f"\nStarting experiment with {pattern} uplift pattern...")

        # 创建实验输出目录
        output_dir = os.path.join(self.base_output_dir, f'pattern_{pattern}')
        os.makedirs(output_dir, exist_ok=True)

        shape = self.config['experiment']['shape']

        try:
            # 1. 创建合成的uplift rate场
            logging.info("Creating synthetic uplift field...")
            true_uplift = self.create_synthetic_uplift(shape, pattern)
            np.save(os.path.join(output_dir, 'true_uplift.npy'), true_uplift)

            # 2. 创建合成erosion coefficient场
            logging.info("Creating synthetic erosion coefficient field...")
            Ksp = create_synthetic_erosion_field(
                shape=shape,
                base_k_sp=self.config['model_params']['k_sp_base']
            )

            # 更新模型参数
            model_params = self.config['model_params'].copy()
            model_params['Ksp'] = Ksp

            # 3. 运行前向模型生成合成地形
            logging.info("Running forward model to generate synthetic topography...")
            synthetic_dem = run_fastscape_model(
                k_sp=Ksp,
                uplift=true_uplift,
                k_diff=model_params['d_diff'],
                x_size=shape[1],
                y_size=shape[0],
                spacing=model_params['spacing'],
                boundary_status=model_params['boundary_status'],
                area_exp=model_params['area_exp'],
                slope_exp=model_params['slope_exp'],
                time_total=model_params['time_total']
            )
            np.save(os.path.join(output_dir, 'synthetic_dem.npy'), synthetic_dem)

            # 4. 创建目标函数
            logging.info("Creating objective function...")
            obj_func = self.create_objective_function(
                target_dem=synthetic_dem,
                shape=shape,
                model_params=model_params
            )

            # 5. GA反演
            logging.info("Starting GA inversion...")
            start_time = time.time()
            best_uplift, best_fitness, fitness_history = optimize_uplift_ga(
                obj_func=obj_func,
                resampled_dem=synthetic_dem,
                LOW_RES_SHAPE=(shape[0]//self.config['experiment']['scale_factor'],
                            shape[1]//self.config['experiment']['scale_factor']),
                ORIGINAL_SHAPE=shape,
                ga_params=self.config['ga_params'],
                model_params=model_params,
                n_jobs=self.config['ga_params'].get('n_jobs', 1),
                run_mode=None
            )
            inversion_time = time.time() - start_time
            logging.info(f"GA inversion completed in {inversion_time:.2f} seconds")

            if best_uplift is not None:
                # 6. 处理反演结果
                best_low_res_uplift = best_uplift.reshape(
                    (shape[0]//self.config['experiment']['scale_factor'],
                    shape[1]//self.config['experiment']['scale_factor'])
                )
                inverted_uplift = interpolate_uplift_cv(best_low_res_uplift, shape)
                # 使用安全的保存函数
                if not safe_save_array(inverted_uplift,
                                    os.path.join(output_dir, 'inverted_uplift.npy')):
                    raise RuntimeError("Failed to save inverted uplift array")

                # 7. 运行最终模拟
                logging.info("Running final simulation with inverted uplift...")
                final_dem = run_fastscape_model(
                    k_sp=Ksp,
                    uplift=inverted_uplift,
                    k_diff=model_params['d_diff'],
                    x_size=shape[1],
                    y_size=shape[0],
                    spacing=model_params['spacing'],
                    boundary_status=model_params['boundary_status'],
                    area_exp=model_params['area_exp'],
                    slope_exp=model_params['slope_exp'],
                    time_total=model_params['time_total']
                )
                # 保存最终DEM
                if not safe_save_array(final_dem,
                                    os.path.join(output_dir, 'final_dem.npy')):
                    raise RuntimeError("Failed to save final DEM array")

                # 8. 计算评估指标
                metrics = self.evaluate_results(
                    inverted_uplift=inverted_uplift,
                    true_uplift=true_uplift,
                    simulated_dem=final_dem,
                    target_dem=synthetic_dem
                )
                metrics['inversion_time'] = inversion_time
                metrics['final_fitness'] = best_fitness

                # 保存评估指标
                metrics_path = os.path.join(output_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)

                # 保存适应度历史
                if fitness_history is not None:
                    if not safe_save_array(np.array(fitness_history),
                                     os.path.join(output_dir, 'fitness_history.npy')):
                        logging.warning("Failed to save fitness history")

                logging.info(f"Experiment with {pattern} pattern completed successfully")
                return metrics
            else:
                logging.error("GA inversion failed to produce valid results")
                return None

        except Exception as e:
            logging.error(f"Error during experiment with {pattern} pattern: {e}")
            return None



    def run_all_experiments(self):
        """运行所有实验模式"""
        logging.info("Starting synthetic experiments suite")
        results = {}

        for pattern in self.config['experiment']['patterns']:
            try:
                results[pattern] = self.run_experiment(pattern)
            except Exception as e:
                logging.error(f"Failed to run experiment with {pattern} pattern: {e}")
                results[pattern] = None

        # 保存总结果
        summary_path = os.path.join(self.base_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 打印结果总结
        self.print_summary(results)

        # 创建分析器实例并生成分析图
        try:
            analyzer = ResultAnalyzer(self.base_output_dir)
            analyzer.patterns = list(self.config['experiment']['patterns'])
            # 加载所有模式的数据
            all_data = analyzer.load_all_patterns_data()
            # 创建综合图
            analyzer.create_composite_figures()
            # 生成比较分析图
            if len(analyzer.patterns) > 1:
                analyzer.plot_comparative_analysis(all_data)
            logging.info("Analysis figures generated successfully")
        except Exception as e:
            logging.error(f"Error generating analysis figures: {e}")

        return results

    def print_summary(self, results: Dict):
        """打印实验结果总结"""
        logging.info("\nExperiment Results Summary:")
        print("\n" + "="*50)
        print("SYNTHETIC EXPERIMENTS SUMMARY")
        print("="*50)

        for pattern, metrics in results.items():
            print(f"\n{pattern.upper()} Pattern:")
            print("-"*30)
            if metrics is not None:
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{metric:25}: {value:.4f}")
                    else:
                        print(f"{metric:25}: {value}")
            else:
                print("Experiment failed")
        print("\n" + "="*50)

def main():
    """主函数"""
    try:
        print("开始运行轻量合成地形 demo。正式实验参数请查看 run_synthetic_experiment.py 顶部默认配置注释。")
        # 创建实验实例
        experiment = SyntheticExperiment()

        # 运行所有实验
        results = experiment.run_all_experiments()

        logging.info("All experiments completed successfully")
        return results

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.exception("Exception details:")
        return None

if __name__ == "__main__":
    main()
