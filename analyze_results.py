# analyze_results.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from scipy.stats import entropy
from scipy.spatial import distance
from scipy.stats.stats import pearsonr
from libpysal.weights import lat2W
from esda.moran import Moran
from esda.geary import Geary
import seaborn as sns

class ResultAnalyzer:
    """结果分析和可视化类"""
    
    def __init__(self, result_dir: str):
        """
        初始化结果分析器
        
        参数:
        - result_dir: 结果目录路径
        """
        self.result_dir = result_dir
        self.patterns = ['simple', 'medium', 'complex']
        self.setup_logging()
        
        # 设置绘图风格
        plt.style.use('default')  # 使用默认样式
        plt.rcParams.update({
            'figure.dpi': 300,
            'font.size': 12,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'axes.axisbelow': True,  # 网格线置于数据下方
            'figure.autolayout': True,  # 自动调整布局
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2
        })
        
    def setup_logging(self):
        """配置日志系统"""
        log_file = os.path.join(self.result_dir, 'analysis.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_data(self, pattern: str) -> Dict:
        """加载特定模式的所有数据"""
        pattern_dir = os.path.join(self.result_dir, f'pattern_{pattern}')
        if not os.path.exists(pattern_dir):
            logging.error(f"Directory not found: {pattern_dir}")
            return None

        data = {}
        try:
            # 加载numpy数组
            arrays = {
                'true_uplift': 'true_uplift.npy',
                'inverted_uplift': 'inverted_uplift.npy',
                'synthetic_dem': 'synthetic_dem.npy',
                'final_dem': 'final_dem.npy',
                'fitness_history': 'fitness_history.npy'
            }
            
            for key, filename in arrays.items():
                filepath = os.path.join(pattern_dir, filename)
                if os.path.exists(filepath):
                    try:
                        data[key] = np.load(filepath)
                        logging.info(f"Successfully loaded {filename}")
                    except Exception as e:
                        logging.error(f"Error loading {filename}: {e}")
                        data[key] = None
                else:
                    logging.warning(f"File not found: {filepath}")
                    data[key] = None
            
            # 加载评估指标
            metrics_path = os.path.join(pattern_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data['metrics'] = json.load(f)
                    logging.info("Successfully loaded metrics.json")
            else:
                logging.warning(f"Metrics file not found: {metrics_path}")
                data['metrics'] = None
                
            return data
            
        except Exception as e:
            logging.error(f"Error loading data for pattern {pattern}: {e}")
            return None

    def load_all_patterns_data(self) -> Dict:
        """加载所有模式的数据"""
        all_data = {}
        for pattern in self.patterns:
            data = self.load_data(pattern)
            if data is not None:
                all_data[pattern] = data
            else:
                logging.error(f"Failed to load data for {pattern} pattern")
        return all_data

    def plot_comparative_analysis(self, all_data: Dict):
        """绘制综合对比分析图"""
        # 计算每个模式的统计指标
        metrics = {}
        for pattern in self.patterns:
            if pattern in all_data:
                metrics[pattern] = self.calculate_advanced_statistics(all_data[pattern])

        if not metrics:
            logging.error("No valid metrics calculated for any pattern")
            return

        # 创建图形
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 基础性能指标对比 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_basic_metrics(ax1, metrics)
        
        # 2. 分布相似性趋势 (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_distribution_similarity(ax2, metrics)
        
        # 3. 空间自相关对比 (左下)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_spatial_autocorrelation(ax3, metrics)
        
        # 4. 综合性能评估 (右下)
        ax4 = fig.add_subplot(gs[1, 1], projection='polar')
        self._plot_performance_summary(ax4, metrics)
        
        # 添加图形总标题
        fig.suptitle('Comparative Analysis of Inversion Results\nAcross Pattern Complexity',
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 添加注释说明
        fig.text(0.02, 0.98, 
                'MAE/RMSE: Lower is better\n' +
                'R²/Pattern Similarity: Higher is better\n' +
                'Spatial Metrics: Closer to true values is better',
                fontsize=10, va='top', ha='left')
        
        # 调整布局并保存
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'comparative_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        self._save_metrics_summary(metrics)
        logging.info(f"Saved comparative analysis plot to {save_path}")

    def _plot_performance_summary(self, ax, metrics: Dict):
        """绘制综合性能评估雷达图"""
        # 定义评估维度
        dimensions = [
            ('Accuracy', lambda m: 1 - m['mae']),
            ('Correlation', lambda m: m['r2']),
            ('Distribution', lambda m: 1 - m['js_divergence']),
            ('Spatial Pattern', lambda m: m['spatial_pattern_similarity']),
            ('Convergence', lambda m: 1 - m['mean_rel_error'])
        ]
        dim_names = [d[0] for d in dimensions]
        
        # 计算各维度得分
        scores = {
            pattern: [dim[1](metric) for dim in dimensions]
            for pattern, metric in metrics.items()
        }
        
        # 设置雷达图角度
        angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # 绘制各模式的雷达图
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (pattern, pattern_scores) in enumerate(scores.items()):
            values = np.concatenate((pattern_scores, [pattern_scores[0]]))
            ax.plot(angles, values, 'o-', label=pattern, color=colors[i], linewidth=2)
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # 设置刻度和标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_names)
        ax.set_ylim(0, 1)
        
        # 添加网格和图例
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    def _save_metrics_summary(self, metrics: Dict):
        """保存数值结果摘要"""
        summary = {
            'pattern_comparison': {
                'mae': {p: m['mae'] for p, m in metrics.items()},
                'rmse': {p: m['rmse'] for p, m in metrics.items()},
                'r2': {p: m['r2'] for p, m in metrics.items()},
                'spatial_similarity': {p: m['spatial_pattern_similarity'] 
                                    for p, m in metrics.items()},
                'distribution_similarity': {p: 1-m['js_divergence'] 
                                        for p, m in metrics.items()}
            },
            'improvement_relative_to_simple': {
                metric: {
                    p: (metrics[p][metric] - metrics['simple'][metric]) 
                    / abs(metrics['simple'][metric]) * 100
                    for p in ['medium', 'complex']
                }
                for metric in ['mae', 'rmse', 'r2']
            }
        }
        
        save_path = os.path.join(self.result_dir, 'comparative_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)


    def _plot_basic_metrics(self, ax, metrics: Dict):
        """绘制基础性能指标对比"""
        basic_metrics = ['mae', 'rmse', 'r2']
        x = np.arange(len(self.patterns))
        width = 0.25
        
        for i, metric in enumerate(basic_metrics):
            values = [metrics[p][metric] for p in self.patterns]
            ax.bar(x + i*width, values, width, label=metric.upper())
        
        ax.set_title('Basic Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.patterns)
        ax.legend()
        ax.grid(True)

    def _plot_distribution_similarity(self, ax, metrics: Dict):
        """绘制分布相似性趋势"""
        dist_metrics = ['kl_divergence', 'js_divergence']
        for metric in dist_metrics:
            values = [metrics[p][metric] for p in self.patterns]
            ax.plot(self.patterns, values, 'o-', 
                   label=metric.replace('_', ' ').title())
        
        ax.set_title('Distribution Similarity Trends')
        ax.legend()
        ax.grid(True)

    def _plot_spatial_autocorrelation(self, ax, metrics: Dict):
        """绘制空间自相关对比"""
        spatial_metrics = {
            'Moran\'s I True': 'moran_i_true',
            'Moran\'s I Inv': 'moran_i_inv',
            'Geary\'s C True': 'geary_c_true',
            'Geary\'s C Inv': 'geary_c_inv'
        }
        
        for label, metric in spatial_metrics.items():
            values = [metrics[p][metric] for p in self.patterns]
            ax.plot(self.patterns, values, 'o-', label=label)
        
        ax.set_title('Spatial Autocorrelation Comparison')
        ax.legend()
        ax.grid(True)

    def _plot_radar_chart(self, ax, metrics: Dict):
        """绘制综合性能雷达图"""
        summary_metrics = {
            'R²': [metrics[p]['r2'] for p in self.patterns],
            'Inv Accuracy': [1-metrics[p]['mae'] for p in self.patterns],
            'Dist Sim': [1-metrics[p]['js_divergence'] for p in self.patterns],
            'Spat Struct': [1-abs(metrics[p]['moran_i_true']-metrics[p]['moran_i_inv']) 
                           for p in self.patterns]
        }
        
        angles = np.linspace(0, 2*np.pi, len(summary_metrics), endpoint=False)
        
        for i, pattern in enumerate(self.patterns):
            values = [summary_metrics[m][i] for m in summary_metrics]
            values += values[:1]
            ang = np.concatenate((angles, [angles[0]]))
            ax.plot(ang, values, 'o-', label=pattern)
            ax.fill(ang, values, alpha=0.25)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(summary_metrics.keys())
        ax.set_title('Overall Performance')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    def plot_uplift_comparison(self, data: Dict, pattern: str):
        """绘制真实和反演uplift rate场的对比图"""
        if data['true_uplift'] is None or data['inverted_uplift'] is None:
            logging.error("Missing uplift data for comparison")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 计算共同的颜色范围
        vmin = min(np.min(data['true_uplift']), np.min(data['inverted_uplift']))
        vmax = max(np.max(data['true_uplift']), np.max(data['inverted_uplift']))
        
        # 绘制真实uplift
        im1 = ax1.imshow(data['true_uplift'], cmap='RdBu_r', origin='lower',
                        vmin=vmin, vmax=vmax)
        ax1.set_title('True Uplift Rate')
        plt.colorbar(im1, ax=ax1, label='Uplift Rate (mm/yr)')
        
        # 绘制反演uplift
        im2 = ax2.imshow(data['inverted_uplift'], cmap='RdBu_r', origin='lower',
                        vmin=vmin, vmax=vmax)
        ax2.set_title('Inverted Uplift Rate')
        plt.colorbar(im2, ax=ax2, label='Uplift Rate (mm/yr)')
        # 绘制等高线
        #ax1.contour(data['true_uplift'], levels=10, colors='k', linewidths=0.5)
        #ax2.contour(data['inverted_uplift'], levels=10, colors='k', linewidths=0.5)
        
        # 添加标题和其他信息
        fig.suptitle(f'Uplift Rate Comparison - {pattern.upper()} Pattern', 
                    fontsize=14, fontweight='bold')
        
        # 保存图像
        save_path = os.path.join(self.result_dir, f'uplift_comparison_{pattern}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved uplift comparison plot to {save_path}")

    def plot_dem_comparison(self, data: Dict, pattern: str):
        """绘制DEM对比图"""
        if data['synthetic_dem'] is None or data['final_dem'] is None:
            logging.error("Missing DEM data for comparison")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 计算共同的颜色范围
        vmin = min(np.min(data['synthetic_dem']), np.min(data['final_dem']))
        vmax = max(np.max(data['synthetic_dem']), np.max(data['final_dem']))
        
        # 绘制合成DEM
        im1 = ax1.imshow(data['synthetic_dem'], cmap='terrain', origin='lower',
                        vmin=vmin, vmax=vmax)
        ax1.set_title('Target DEM')
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        
        # 绘制模拟DEM
        im2 = ax2.imshow(data['final_dem'], cmap='terrain', origin='lower',
                        vmin=vmin, vmax=vmax)
        ax2.set_title('Simulated DEM')
        plt.colorbar(im2, ax=ax2, label='Elevation (m)')
        # 绘制等高线
        ax1.contour(data['synthetic_dem'], levels=10, colors='k', linewidths=0.5)
        ax2.contour(data['final_dem'], levels=10, colors='k', linewidths=0.5)
        
        # 添加标题和其他信息
        fig.suptitle(f'DEM Comparison - {pattern.upper()} Pattern', 
                    fontsize=14, fontweight='bold')
        
        # 保存图像
        save_path = os.path.join(self.result_dir, f'dem_comparison_{pattern}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved DEM comparison plot to {save_path}")

    def plot_error_distribution(self, data: Dict, pattern: str):
        """绘制误差分布图"""
        if data['true_uplift'] is None or data['inverted_uplift'] is None:
            logging.error("Missing data for error distribution")
            return
            
        # 计算相对误差
        rel_error = 100 * (data['inverted_uplift'] - data['true_uplift']) / data['true_uplift']
        
        plt.figure(figsize=(10, 6))
        plt.hist(rel_error.flatten(), bins=50, density=True, alpha=0.7)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Density')
        plt.title(f'Error Distribution - {pattern.upper()} Pattern')
        
        # 添加统计信息
        mean_error = np.mean(rel_error)
        std_error = np.std(rel_error)
        plt.axvline(mean_error, color='r', linestyle='--', 
                   label=f'Mean: {mean_error:.2f}%')
        plt.axvline(mean_error + std_error, color='g', linestyle=':', 
                   label=f'Std: {std_error:.2f}%')
        plt.axvline(mean_error - std_error, color='g', linestyle=':')
        plt.legend()
        
        # 保存图像
        save_path = os.path.join(self.result_dir, f'error_distribution_{pattern}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved error distribution plot to {save_path}")

    def plot_fitness_history(self, data: Dict, pattern: str):
        """绘制适应度历史曲线"""
        if data['fitness_history'] is None:
            logging.error("Missing fitness history data")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(data['fitness_history'], linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title(f'Fitness History - {pattern.upper()} Pattern')
        
        # 添加最佳适应度标记
        best_gen = np.argmin(data['fitness_history'])
        best_fitness = np.min(data['fitness_history'])
        plt.plot(best_gen, best_fitness, 'ro', 
                label=f'Best: {best_fitness:.4f} at gen {best_gen}')
        plt.legend()
        
        # 保存图像
        save_path = os.path.join(self.result_dir, f'fitness_history_{pattern}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved fitness history plot to {save_path}")

    def calculate_statistics(self, data: Dict) -> Dict:
        """计算各种统计指标"""
        stats_dict = {}
        
        try:
            if data['true_uplift'] is not None and data['inverted_uplift'] is not None:
                # 计算相关系数
                corr, p_value = stats.pearsonr(data['true_uplift'].flatten(), 
                                             data['inverted_uplift'].flatten())
                stats_dict['uplift_correlation'] = corr
                stats_dict['correlation_p_value'] = p_value
                
                # 计算RMSE
                rmse = np.sqrt(np.mean((data['true_uplift'] - data['inverted_uplift'])**2))
                stats_dict['uplift_rmse'] = rmse
                
                # 计算相对误差
                rel_error = np.abs(data['true_uplift'] - data['inverted_uplift']) / data['true_uplift']
                stats_dict['mean_relative_error'] = np.mean(rel_error)
                stats_dict['median_relative_error'] = np.median(rel_error)
                stats_dict['max_relative_error'] = np.max(rel_error)
                stats_dict['std_relative_error'] = np.std(rel_error)
                
                # 计算R²
                r2 = 1 - (np.sum((data['true_uplift'] - data['inverted_uplift'])**2) / 
                         np.sum((data['true_uplift'] - np.mean(data['true_uplift']))**2))
                stats_dict['r_squared'] = r2
            
            if data['fitness_history'] is not None:
                stats_dict['final_fitness'] = float(data['fitness_history'][-1])
                stats_dict['best_fitness'] = float(np.min(data['fitness_history']))
                stats_dict['convergence_generation'] = int(np.argmin(data['fitness_history']))
                
                # 计算收敛速度
                threshold = np.min(data['fitness_history']) * 1.05  # 5%阈值
                convergence_gens = np.where(data['fitness_history'] <= threshold)[0]
                if len(convergence_gens) > 0:
                    stats_dict['generations_to_converge'] = int(convergence_gens[0])
                
            return stats_dict
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {e}")
            return stats_dict
        
    @staticmethod
    def calculate_r2(y_true, y_pred):
        """
        计算修正的 R² 值（静态方法）
        
        参数:
        - y_true: 真实值数组
        - y_pred: 预测值数组
        
        返回:
        - float: R² 值，范围 [0, 1]
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 计算均值
        y_true_mean = np.mean(y_true)
        
        # 计算总平方和
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        
        # 计算残差平方和
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # 如果总平方和为0，返回0
        if ss_tot == 0:
            return 0
        
        # 确保 R² 不小于0
        r2 = max(0, 1 - (ss_res / ss_tot))
        
        return r2

    def calculate_spatial_autocorrelation(self, data: np.ndarray) -> Tuple[float, float]:
        """改进的空间自相关计算"""
        def create_weight_matrix(shape: Tuple[int, int]) -> np.ndarray:
            """创建改进的空间权重矩阵"""
            rows, cols = shape
            n = rows * cols
            W = np.zeros((n, n))
            
            # 考虑8个方向的邻居（包括对角线）
            directions = [(-1,-1), (-1,0), (-1,1), 
                        (0,-1),         (0,1),
                        (1,-1),  (1,0),  (1,1)]
                        
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            nidx = ni * cols + nj
                            # 使用距离加权
                            W[idx, nidx] = 1/np.sqrt(di*di + dj*dj)
            
            # 行标准化
            row_sums = W.sum(axis=1)
            W = W / row_sums[:, np.newaxis]
            return W

        try:
            # 标准化数据
            x = (data - np.mean(data)) / np.std(data)
            x = x.flatten()
            n = len(x)
            
            # 创建权重矩阵
            W = create_weight_matrix(data.shape)
            
            # 计算 Moran's I
            z_lag = np.dot(W, x)
            numerator = n * np.sum(x * z_lag)
            denominator = np.sum(x * x) * np.sum(np.sum(W))
            moran_i = numerator / denominator
            
            # 计算 Geary's C
            diff_squared = np.array([[(x[i] - x[j])**2 for j in range(n)] 
                                    for i in range(n)])
            numerator = (n-1) * np.sum(np.sum(W * diff_squared))
            denominator = 2 * np.sum(np.sum(W)) * np.sum(x * x)
            geary_c = numerator / denominator
            
            return moran_i, geary_c
            
        except Exception as e:
            logging.error(f"Error in spatial autocorrelation calculation: {e}")
            return np.nan, np.nan

    def calculate_advanced_statistics(self, data: Dict) -> Dict:
        """计算高级统计指标"""
        stats_dict = {}
        
        try:
            if data['true_uplift'] is None or data['inverted_uplift'] is None:
                return stats_dict
                
            # 标准化数据
            def normalize_data(arr):
                mean = np.mean(arr)
                std = np.std(arr)
                return (arr - mean) / std
                
            true_uplift = data['true_uplift']
            inv_uplift = data['inverted_uplift']
            
            # 基础统计指标 (使用归一化数据)
            true_norm = normalize_data(true_uplift)
            inv_norm = normalize_data(inv_uplift)
            
            stats_dict['mae'] = np.mean(np.abs(true_norm - inv_norm))
            stats_dict['rmse'] = np.sqrt(np.mean((true_norm - inv_norm)**2))
            
            # R²计算改进 (使用原始数据)
            stats_dict['r2'] = self.calculate_r2(true_uplift, inv_uplift)
            
            # 分布相似性指标
            # 使用更合理的bin范围
            min_val = min(np.min(true_uplift), np.min(inv_uplift))
            max_val = max(np.max(true_uplift), np.max(inv_uplift))
            bins = np.linspace(min_val, max_val, 51)  # 50个区间
            
            hist_true, _ = np.histogram(true_uplift, bins=bins, density=True)
            hist_inv, _ = np.histogram(inv_uplift, bins=bins, density=True)
            
            # 平滑处理
            def smooth_hist(hist):
                eps = 1e-10
                hist_smooth = hist + eps
                return hist_smooth / np.sum(hist_smooth)  # 重新归一化
                
            hist_true = smooth_hist(hist_true)
            hist_inv = smooth_hist(hist_inv)
            
            # KL散度和JS散度
            stats_dict['kl_divergence'] = entropy(hist_true, hist_inv)
            stats_dict['js_divergence'] = distance.jensenshannon(hist_true, hist_inv)
            
            # 空间自相关指标
            # 使用改进的空间自相关计算
            true_moran, true_geary = self.calculate_spatial_autocorrelation(true_uplift)
            inv_moran, inv_geary = self.calculate_spatial_autocorrelation(inv_uplift)
            
            stats_dict['moran_i_true'] = true_moran
            stats_dict['moran_i_inv'] = inv_moran
            stats_dict['geary_c_true'] = true_geary
            stats_dict['geary_c_inv'] = inv_geary
            
            # 添加相对误差统计
            rel_error = (inv_uplift - true_uplift) / true_uplift
            stats_dict['mean_rel_error'] = np.mean(rel_error)
            stats_dict['std_rel_error'] = np.std(rel_error)
            stats_dict['median_rel_error'] = np.median(rel_error)
            
            # 添加空间模式相似度
            pattern_similarity = 1 - np.abs(true_moran - inv_moran) / max(abs(true_moran), abs(inv_moran))
            stats_dict['spatial_pattern_similarity'] = pattern_similarity
            
            return stats_dict
            
        except Exception as e:
            logging.error(f"Error calculating advanced statistics: {e}")
            logging.exception("Exception details:")
            return stats_dict

    def _plot_spatial_autocorrelation(self, ax, metrics: Dict):
        """改进的空间自相关可视化"""
        patterns = self.patterns
        
        # 提取数据
        moran_true = [metrics[p]['moran_i_true'] for p in patterns]
        moran_inv = [metrics[p]['moran_i_inv'] for p in patterns]
        geary_true = [metrics[p]['geary_c_true'] for p in patterns]
        geary_inv = [metrics[p]['geary_c_inv'] for p in patterns]
        
        # 使用双y轴以更好地显示两个指标
        ax1 = ax
        ax2 = ax1.twinx()
        
        # 绘制Moran's I
        l1 = ax1.plot(patterns, moran_true, 'b-o', label="Moran's I (True)")
        l2 = ax1.plot(patterns, moran_inv, 'b--o', label="Moran's I (Inv)")
        ax1.set_ylabel("Moran's I")
        
        # 绘制Geary's C
        l3 = ax2.plot(patterns, geary_true, 'r-o', label="Geary's C (True)")
        l4 = ax2.plot(patterns, geary_inv, 'r--o', label="Geary's C (Inv)")
        ax2.set_ylabel("Geary's C")
        
        # 合并图例
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        
        ax1.set_title('Spatial Autocorrelation Comparison')
        ax1.grid(True)

    def create_summary_report(self):
        """创建综合分析报告"""
        report = []
        report.append("="*50)
        report.append("INVERSION RESULTS ANALYSIS REPORT")
        report.append("="*50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Results Directory: {self.result_dir}")
        report.append("-"*50)
        
        all_stats = {}
        for pattern in self.patterns:
            data = self.load_data(pattern)
            if data is not None:
                stats = self.calculate_statistics(data)
                all_stats[pattern] = stats
                
                report.append(f"\nPattern: {pattern.upper()}")
                report.append("-"*30)
                
                # 组织指标显示顺序
                metric_groups = {
                    'Correlation Metrics': ['uplift_correlation', 'correlation_p_value', 'r_squared'],
                    'Error Metrics': ['uplift_rmse', 'mean_relative_error', 'median_relative_error',
                                    'max_relative_error', 'std_relative_error'],
                    'Convergence Metrics': ['final_fitness', 'best_fitness', 'convergence_generation',
                                          'generations_to_converge']
                }
                
                for group_name, metrics in metric_groups.items():
                    report.append(f"\n{group_name}:")
                    for metric in metrics:
                        if metric in stats:
                            value = stats[metric]
                            if isinstance(value, float):
                                report.append(f"{metric:25}: {value:.4f}")
                            else:
                                report.append(f"{metric:25}: {value}")
        
        # 保存报告
        report_path = os.path.join(self.result_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        logging.info(f"Analysis report saved to {report_path}")
            
        return all_stats

    def plot_statistics_comparison(self, data: Dict, pattern: str):
        """绘制统计指标对比图"""
        if data['true_uplift'] is None or data['inverted_uplift'] is None:
            return
            
        # 计算统计指标
        stats = self.calculate_advanced_statistics(data)
        
        # 创建多子图布局
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. 基础指标条形图
        ax1 = fig.add_subplot(gs[0, 0])
        basic_metrics = ['mae', 'rmse', 'r2']
        values = [stats[m] for m in basic_metrics]
        sns.barplot(x=basic_metrics, y=values, ax=ax1)
        ax1.set_title('Basic Metrics')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # 2. 分布相似性指标
        ax2 = fig.add_subplot(gs[0, 1])
        dist_metrics = ['kl_divergence', 'js_divergence']
        values = [stats[m] for m in dist_metrics]
        sns.barplot(x=dist_metrics, y=values, ax=ax2)
        ax2.set_title('Distribution Similarity')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # 3. 空间自相关对比
        ax3 = fig.add_subplot(gs[1, 0])
        width = 0.35
        x = np.arange(2)
        ax3.bar(x - width/2, [stats['moran_i_true'], stats['geary_c_true']], 
                width, label='True')
        ax3.bar(x + width/2, [stats['moran_i_inv'], stats['geary_c_inv']], 
                width, label='Inverted')
        ax3.set_xticks(x)
        ax3.set_xticklabels(["Moran's I", "Geary's C"])
        ax3.legend()
        ax3.set_title('Spatial Autocorrelation')
        
        # 4. 相关性散点图
        ax4 = fig.add_subplot(gs[1, 1])
        true_flat = data['true_uplift'].flatten()
        inv_flat = data['inverted_uplift'].flatten()
        ax4.scatter(true_flat, inv_flat, alpha=0.5, s=1)
        ax4.plot([true_flat.min(), true_flat.max()], 
                [true_flat.min(), true_flat.max()], 'r--')
        ax4.set_xlabel('True Uplift')
        ax4.set_ylabel('Inverted Uplift')
        ax4.set_title(f'Correlation (R² = {stats["r2"]:.3f})')
        
        # 整体标题
        fig.suptitle(f'Statistical Analysis - {pattern.upper()} Pattern', 
                    fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.result_dir, f'statistics_comparison_{pattern}.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved statistics comparison plot to {save_path}")

    def analyze_all_patterns(self):
        """分析所有模式的结果"""
        try:
            # 只加载一次数据
            all_data = self.load_all_patterns_data()
            all_results = {}
            
            # 生成单独的分析图
            for pattern in self.patterns:
                if pattern in all_data:
                    data = all_data[pattern]
                    try:
                        # 删除单独的统计对比图绘制
                        self.plot_uplift_comparison(data, pattern)
                        self.plot_dem_comparison(data, pattern)
                        self.plot_error_distribution(data, pattern)
                        self.plot_fitness_history(data, pattern)
                        # 移除 self.plot_statistics_comparison(data, pattern)
                        
                        # 计算统计指标
                        stats = self.calculate_statistics(data)
                        all_results[pattern] = {
                            'stats': stats,
                            'data_loaded': True,
                            'visualizations_generated': True
                        }
                        logging.info(f"Successfully analyzed {pattern} pattern")
                    except Exception as e:
                        logging.error(f"Error analyzing {pattern} pattern: {e}")
                        all_results[pattern] = {
                            'stats': None,
                            'data_loaded': True,
                            'visualizations_generated': False,
                            'error': str(e)
                        }
                else:
                    logging.error(f"Failed to load data for {pattern} pattern")
                    all_results[pattern] = {
                        'stats': None,
                        'data_loaded': False,
                        'visualizations_generated': False,
                        'error': 'Data loading failed'
                    }
            
            # 生成综合对比分析图
            self.plot_comparative_analysis(all_data)
            
            # 创建综合报告
            try:
                stats = self.create_summary_report()
                all_results['summary_report_generated'] = True
            except Exception as e:
                logging.error(f"Error generating summary report: {e}")
                all_results['summary_report_generated'] = False
                all_results['summary_error'] = str(e)
                
            return all_results
            
        except Exception as e:
            logging.error(f"Error in analyze_all_patterns: {e}")
            return {'error': str(e)}

    def create_composite_figures(self):
        """
        为每种模式创建综合图，按照以下布局：
        - (a)-(d) 为两行两列排版：
        (a) Target DEM, (b) Simulated DEM
        (c) True uplift rate, (d) Inverted uplift rate
        - (e) Fitness history 在最下方，宽度与上方对齐
        """
        logging.info("Creating composite figures for all patterns...")
        all_data = self.load_all_patterns_data()
        
        for pattern in self.patterns:
            if pattern not in all_data:
                logging.error(f"Missing data for pattern: {pattern}")
                continue
                
            data = all_data[pattern]
            if (data['true_uplift'] is None or data['inverted_uplift'] is None or 
                data['synthetic_dem'] is None or data['final_dem'] is None or
                data['fitness_history'] is None):
                logging.error(f"Incomplete data for pattern: {pattern}")
                continue
            
            try:
                # 创建新的布局：使用更精确的GridSpec控制
                fig = plt.figure(figsize=(16, 16))
                # 调整高度比例并添加水平和垂直间距
                gs = GridSpec(8, 2, figure=fig, 
                            height_ratios=[1, 1, 0.2, 1, 1, 0.2, 0.8, 0.1],  # 添加更多间距
                            hspace=0.0, wspace=0.3)  # 增加水平间距
                
                # (a) Target DEM - 左上
                ax1 = fig.add_subplot(gs[0:2, 0])
                vmin_dem = min(np.min(data['synthetic_dem']), np.min(data['final_dem']))
                vmax_dem = max(np.max(data['synthetic_dem']), np.max(data['final_dem']))
                im1 = ax1.imshow(data['synthetic_dem'], cmap='terrain', origin='lower',
                            vmin=vmin_dem, vmax=vmax_dem)
                ax1.set_title('Target DEM')
                plt.colorbar(im1, ax=ax1, label='Elevation (m)')
                ax1.contour(data['synthetic_dem'], levels=10, colors='k', linewidths=0.5)
                
                # (b) Simulated DEM - 右上
                ax2 = fig.add_subplot(gs[0:2, 1])
                im2 = ax2.imshow(data['final_dem'], cmap='terrain', origin='lower',
                            vmin=vmin_dem, vmax=vmax_dem)
                ax2.set_title('Simulated DEM')
                plt.colorbar(im2, ax=ax2, label='Elevation (m)')
                ax2.contour(data['final_dem'], levels=10, colors='k', linewidths=0.5)
                
                # (c) True uplift rate - 左下
                ax3 = fig.add_subplot(gs[3:5, 0])
                vmin_uplift = min(np.min(data['true_uplift']), np.min(data['inverted_uplift']))
                vmax_uplift = max(np.max(data['true_uplift']), np.max(data['inverted_uplift']))
                im3 = ax3.imshow(data['true_uplift'], cmap='RdBu_r', origin='lower',
                            vmin=vmin_uplift, vmax=vmax_uplift)
                ax3.set_title('True Uplift Rate')
                plt.colorbar(im3, ax=ax3, label='Uplift Rate (mm/yr)')
                
                # (d) Inverted uplift rate - 右下
                ax4 = fig.add_subplot(gs[3:5, 1])
                im4 = ax4.imshow(data['inverted_uplift'], cmap='RdBu_r', origin='lower',
                            vmin=vmin_uplift, vmax=vmax_uplift)
                ax4.set_title('Inverted Uplift Rate')
                plt.colorbar(im4, ax=ax4, label='Uplift Rate (mm/yr)')
                
                # (e) Fitness history - 底部跨两列
                # Fitness History 部分调整
                ax5 = fig.add_subplot(gs[6, :])
                ax5.plot(data['fitness_history'], color='#1E88E5', linewidth=2)
                best_gen = np.argmin(data['fitness_history'])
                best_fitness = np.min(data['fitness_history'])
                ax5.plot(best_gen, best_fitness, 'ro', 
                        label=f'Best: {best_fitness:.4f} at gen {best_gen}')
                ax5.set_xlabel('Generation')
                ax5.set_ylabel('Fitness Value')
                ax5.set_title('Fitness History')
                ax5.legend(loc='upper right')  # 调整图例位置
                ax5.grid(True, alpha=0.3)  # 降低网格线透明度
                
                # 调整整体布局
                plt.suptitle(f'Comprehensive Analysis - {pattern.upper()} Pattern',
                            fontsize=16, fontweight='bold', y=0.98)
                
                # 使用更精确的边距控制
                plt.subplots_adjust(
                    left=0.08,    # 减小左边距，使图形整体向左移动
                    right=0.92,   # 增大右边距，确保colorbar有足够空间且对齐
                    bottom=0.08,  # 略微增加下边距
                    top=0.95,     # 保持上边距
                    hspace=0.3,   # 稍微减小垂直间距
                    wspace=0.2    # 保持水平间距以容纳colorbar
                )
                
                # 保存图像
                save_path = os.path.join(self.result_dir, f'composite_figure_{pattern}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved composite figure to {save_path}")
                
            except Exception as e:
                logging.error(f"Error creating composite figure for {pattern}: {e}")
                logging.exception("Exception details:")

    def main():
        """主函数"""
        import argparse
        
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description='Analyze inversion results')
        parser.add_argument('--result-dir', type=str, required=True,
                        help='Directory containing the results to analyze')
        parser.add_argument('--patterns', type=str, nargs='+',
                        default=['simple', 'medium', 'complex'],
                        help='Patterns to analyze')
        parser.add_argument('--composite', action='store_true',
                        help='Create composite figures for each pattern')
        parser.add_argument('--minimal', action='store_true',
                        help='Only generate composite figures and comparative analysis')
        
        args = parser.parse_args()
        
        # 验证结果目录
        if not os.path.exists(args.result_dir):
            print(f"Error: Results directory not found: {args.result_dir}")
            return
        
        try:
            # 创建分析器实例
            analyzer = ResultAnalyzer(args.result_dir)
            analyzer.patterns = args.patterns
            
            # 根据参数决定分析方式
            if args.minimal:
                # 只加载数据
                all_data = analyzer.load_all_patterns_data()
                # 只生成综合图和比较分析
                analyzer.create_composite_figures()
                analyzer.plot_comparative_analysis(all_data)
                analyzer.create_summary_report()
                print("✓ Generated minimal analysis (composite figures and comparative analysis)")
            else:
                # 运行完整分析
                results = analyzer.analyze_all_patterns()
                
                # 如果指定了composite参数，创建综合图
                if args.composite:
                    analyzer.create_composite_figures()
            
            # 打印分析总结
            print("\nAnalysis Summary:")
            print("="*50)
            for pattern in args.patterns:
                print(f"\n{pattern.upper()} Pattern:")
                if pattern in results:
                    pattern_results = results[pattern]
                    if pattern_results['data_loaded']:
                        if pattern_results['visualizations_generated']:
                            print("✓ Analysis completed successfully")
                            if pattern_results['stats']:
                                print("\nKey Statistics:")
                                stats = pattern_results['stats']
                                for key in ['uplift_correlation', 'uplift_rmse', 'best_fitness']:
                                    if key in stats:
                                        print(f"{key:20}: {stats[key]:.4f}")
                        else:
                            print("✗ Error generating visualizations:")
                            print(f"  {pattern_results.get('error', 'Unknown error')}")
                    else:
                        print("✗ Failed to load data")
                else:
                    print("✗ Pattern not processed")
            
            print("\nOutput files:")
            print(f"- Analysis report: {os.path.join(args.result_dir, 'analysis_report.txt')}")
            print(f"- Analysis results: {os.path.join(args.result_dir, 'analysis_results.json')}")
            print(f"- Analysis log: {os.path.join(args.result_dir, 'analysis.log')}")
            print(f"- Visualization plots: {args.result_dir}/*.png")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            logging.exception("Analysis failed with error:")
            return

    if __name__ == "__main__":
        main()