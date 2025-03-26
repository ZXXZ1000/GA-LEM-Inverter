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
from datetime import datetime
from typing import Tuple, Dict, Optional

# Import your existing modules
from model_runner import run_fastscape_model
from genetic_algorithm import optimize_uplift_ga
from visualization_utils import plot_comparison, plot_uplift_distribution, plot_3d_surface
from data_preprocessing import interpolate_uplift_cv, interpolate_uplift_cv1
from synthetic_erosion_field import create_synthetic_erosion_field  # 使用新的合成erosion field函数
from fitness_evaluator import terrain_similarity
from array_save_utils import safe_save_array, safe_load_array
from analyze_results import ResultAnalyzer

class SyntheticExperiment:
    def __init__(self, config_path: str = None):
        """
        初始化实验设置
        
        参数:
        - config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 设置默认配置
        self.config = {
            'experiment': {
                'shape': (100, 100),
                'patterns': ['simple', 'medium', 'complex'],
                'scale_factor': 5,
                'output_base_dir': 'synthetic_experiments'
            },
            'ga_params': {
                'pop': 100,  # 增大种群数
                'max_iter': 200,
                'prob_cross': 0.8,  # 调整交叉概率
                'prob_mut': 0.05,
                'lb': 3,
                'ub': 12,
                'decay_rate': 0.97,  # 调整衰减率
                'min_size_pop': 30,  # 调整最小种群数
                'patience': 60  # 调整早停步数
            },
            'model_params': {
                'k_sp_base': 6.92e-6,
                'k_sp_fault': 2e-5,
                'd_diff': 19.2,
                'boundary_status': 'fixed_value',
                'area_exp': 0.43,
                'slope_exp': 1,
                'time_total': 10e6,
                'spacing': 900
            }
        }
        
        # 如果提供了配置文件，则加载它
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)

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
            uplift = 5 + 5 * uplift  # 范围在5-10 mm/yr
            
        elif pattern == 'medium':
            # 两个高斯分布的组合
            uplift1 = np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / 0.1)
            uplift2 = np.exp(-((X - 0.7)**2 + (Y - 0.7)**2) / 0.1)
            uplift = uplift1 + uplift2
            uplift = 5 + 5 * uplift / uplift.max()  # 归一化到5-10 mm/yr
            
        elif pattern == 'complex':
            # 模拟多断层系统
            # 主断层（走向斜切，更陡的梯度）
            main_fault = np.exp(-((0.8*X + 0.6*Y - 0.8)**2) / 0.01) * 3.5  # 减小宽度，降低幅度
            # 共轭断层系统（更清晰的断层带）
            conjugate_fault1 = np.exp(-((0.7*X - 0.7*Y - 0.2)**2) / 0.008) * 2.0
            conjugate_fault2 = np.exp(-((0.6*X - 0.8*Y + 0.3)**2) / 0.008) * 2.0
            # 添加渐变的区域性抬升
            regional_trend = 2.0 * (1 - Y)  # 南北向的渐变趋势
            # 组合所有构造特征
            uplift = 4 + main_fault + conjugate_fault1 + conjugate_fault2 + regional_trend
            # 添加小尺度构造起伏（更细致的局部变化）
            local_structure = gaussian_filter(np.random.rand(rows, cols), sigma=6) * 0.5
            uplift += local_structure
            # 确保uplift在合理范围内
            uplift = np.clip(uplift, 5, 10)
        
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
                    smooth_radius=2  # 调整平滑半径
                )
                
                return 1 - similarity
                
            except Exception as e:
                logging.error(f"Error in objective function: {e}")
                return 1.0
                
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
                n_jobs=12,  # 按核心数或线程数（最大线程数填-1）调整
                run_mode='cached'
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
            # 加载所有模式的数据
            all_data = analyzer.load_all_patterns_data()
            # 创建综合图
            analyzer.create_composite_figures()
            # 生成比较分析图
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