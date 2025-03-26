# k_sensitivity_experiment.py

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import time
import os
import json
import logging
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from array_save_utils import safe_save_array

# Import required modules from your codebase
from run_synthetic_experiment import SyntheticExperiment
from genetic_algorithm import optimize_uplift_ga
from model_runner import run_fastscape_model
from data_preprocessing import interpolate_uplift_cv
from visualization_utils import plot_comparison
from fitness_evaluator import terrain_similarity


class KSensitivityExperiment:
    """Experiment to evaluate sensitivity to scaling factor K"""
    
    def __init__(self, base_output_dir='sensitivity_experiments'):
        """
        Initialize the experiment
        
        Parameters:
        - base_output_dir: Base directory for saving results
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            base_output_dir,
            f'k_sensitivity_{timestamp}'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Basic configuration
        self.config = {
            'experiment': {
                'shape': (100, 100),
                'pattern': 'medium',    # Use medium complexity as the test case
                'k_values': [3, 5, 7, 10, 15],  # K values to test
                'repetitions': 1        # Run each K value multiple times for robust results
            },
            'ga_params': {
                'pop': 100,
                'max_iter': 200,
                'prob_cross': 0.7,
                'prob_mut': 0.05,
                'lb': 3,
                'ub': 12,
                'decay_rate': 0.97,
                'min_size_pop': 20,
                'patience': 60
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
        
        # Save configuration
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_logging(self):
        """Configure logging system"""
        log_file = os.path.join(self.output_dir, 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def create_objective_function(self, target_dem, shape, model_params, k):
        """Create objective function with specific K value"""
        
        def objective_function(uplift_vector):
            try:
                # Get low-res shape with specific K
                low_res_shape = (shape[0]//k, shape[1]//k)
                
                # Reshape uplift vector
                uplift_vector = np.array(uplift_vector).reshape(low_res_shape)
                
                # Interpolate to full resolution
                full_res_uplift = interpolate_uplift_cv(uplift_vector, shape)
                
                # Run FastScape model
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
                
                # Calculate terrain similarity
                similarity = terrain_similarity(
                    matrix1=target_dem,
                    matrix2=generated_elevation,
                    resolution=model_params['spacing'],
                    smooth_radius=2
                )
                
                return 1 - similarity
                
            except Exception as e:
                logging.error(f"Error in objective function: {e}")
                return 1.0
                
        return objective_function
    
    def evaluate_performance(self, inverted_uplift, true_uplift, simulated_dem, target_dem):
        """Evaluate inversion performance metrics"""
        
        # Calculate correlation
        corr, _ = pearsonr(inverted_uplift.flatten(), true_uplift.flatten())
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_uplift, inverted_uplift))
        
        # Calculate relative error
        rel_error = np.abs(inverted_uplift - true_uplift) / true_uplift
        mean_rel_error = np.nanmean(rel_error)  # Use nanmean to handle division by zero
        
        # Calculate R²
        ss_total = np.sum((true_uplift - np.mean(true_uplift))**2)
        ss_residual = np.sum((true_uplift - inverted_uplift)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Calculate DEM metrics
        dem_corr, _ = pearsonr(simulated_dem.flatten(), target_dem.flatten())
        dem_rmse = np.sqrt(mean_squared_error(target_dem, simulated_dem))
        
        return {
            'correlation': corr,
            'rmse': rmse,
            'mean_relative_error': mean_rel_error,
            'r_squared': r_squared,
            'dem_correlation': dem_corr,
            'dem_rmse': dem_rmse
        }
    
    def run_experiment(self):
        """Run the K sensitivity experiment"""
        
        logging.info("Starting K sensitivity experiment")
        
        # Create synthetic experiment to leverage existing functionality
        syn_exp = SyntheticExperiment()
        
        # Parameters
        shape = self.config['experiment']['shape']
        pattern = self.config['experiment']['pattern']
        k_values = self.config['experiment']['k_values']
        repetitions = self.config['experiment']['repetitions']
        
        # Create synthetic uplift field (ground truth)
        true_uplift = syn_exp.create_synthetic_uplift(shape, pattern)
        
        # Create synthetic erosion coefficient field
        from synthetic_erosion_field import create_synthetic_erosion_field
        Ksp = create_synthetic_erosion_field(
            shape=shape,
            base_k_sp=self.config['model_params']['k_sp_base']
        )
        
        # Update model parameters
        model_params = self.config['model_params'].copy()
        model_params['Ksp'] = Ksp
        
        # Generate target DEM by forward modeling
        target_dem = run_fastscape_model(
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
        
        # Save ground truth
        safe_save_array(true_uplift, os.path.join(self.output_dir, 'true_uplift.npy'))
        safe_save_array(target_dem, os.path.join(self.output_dir, 'target_dem.npy'))
        
        # Results container
        results = {
            'k_values': k_values,
            'metrics': {k: [] for k in k_values},
            'computation_time': {k: [] for k in k_values},
            'parameter_count': {k: (shape[0] * shape[1]) // (k*k) for k in k_values}
        }
        
        # Test each K value
        for k in k_values:
            logging.info(f"Testing K = {k}")
            
            # Run multiple repetitions
            for rep in range(repetitions):
                logging.info(f"Repetition {rep+1}/{repetitions}")
                
                # Track parameters
                parameter_count = (shape[0] * shape[1]) // (k*k)
                logging.info(f"Parameter count: {parameter_count}")
                
                # Create objective function
                obj_func = self.create_objective_function(
                    target_dem=target_dem,
                    shape=shape,
                    model_params=model_params,
                    k=k
                )
                
                # Time the optimization process
                start_time = time.time()
                
                # Run optimization
                best_uplift, best_fitness, fitness_history = optimize_uplift_ga(
                    obj_func=obj_func,
                    resampled_dem=target_dem,
                    LOW_RES_SHAPE=(shape[0]//k, shape[1]//k),
                    ORIGINAL_SHAPE=shape,
                    ga_params=self.config['ga_params'],
                    model_params=model_params,
                    n_jobs=12,
                    run_mode='cached'
                )
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                if best_uplift is not None:
                    # Reshape to low resolution
                    best_low_res_uplift = best_uplift.reshape((shape[0]//k, shape[1]//k))
                    
                    # Interpolate to full resolution
                    inverted_uplift = interpolate_uplift_cv(best_low_res_uplift, shape)
                    
                    # Run final simulation
                    simulated_dem = run_fastscape_model(
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
                    
                    # Evaluate performance
                    metrics = self.evaluate_performance(
                        inverted_uplift=inverted_uplift,
                        true_uplift=true_uplift,
                        simulated_dem=simulated_dem,
                        target_dem=target_dem
                    )
                    
                    # Add fitness and time metrics
                    metrics['best_fitness'] = best_fitness
                    metrics['computation_time'] = elapsed_time
                    
                    # Save this run's results
                    results['metrics'][k].append(metrics)
                    results['computation_time'][k].append(elapsed_time)
                    
                    # Save uplift field for this run
                    safe_save_array(
                        inverted_uplift,
                        os.path.join(self.output_dir, f'inverted_uplift_k{k}_rep{rep}.npy')
                    )
                    
                    # Save fitness history
                    safe_save_array(
                        np.array(fitness_history),
                        os.path.join(self.output_dir, f'fitness_history_k{k}_rep{rep}.npy')
                    )
                    
                    logging.info(f"K={k}, Rep={rep+1}: RMSE={metrics['rmse']:.4f}, R²={metrics['r_squared']:.4f}, Time={elapsed_time:.2f}s")
                else:
                    logging.error(f"Optimization failed for K={k}, Rep={rep+1}")
                    # Add failed run info
                    results['metrics'][k].append({
                        'failed': True,
                        'computation_time': elapsed_time
                    })
                    results['computation_time'][k].append(elapsed_time)
        
        # Calculate average metrics across repetitions
        avg_results = {
            'k_values': k_values,
            'parameter_count': results['parameter_count'],
            'avg_metrics': {},
            'std_metrics': {}
        }
        
        for k in k_values:
            # Filter out failed runs
            valid_runs = [m for m in results['metrics'][k] if 'failed' not in m]
            
            if valid_runs:
                # Initialize containers for this k
                avg_results['avg_metrics'][k] = {}
                avg_results['std_metrics'][k] = {}
                
                # Calculate average and std for each metric
                for metric in valid_runs[0].keys():
                    metric_values = [run[metric] for run in valid_runs]
                    avg_results['avg_metrics'][k][metric] = np.mean(metric_values)
                    avg_results['std_metrics'][k][metric] = np.std(metric_values)
            else:
                avg_results['avg_metrics'][k] = {'all_runs_failed': True}
                avg_results['std_metrics'][k] = {'all_runs_failed': True}
        
        # Save all results
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        with open(os.path.join(self.output_dir, 'avg_results.json'), 'w') as f:
            json.dump(avg_results, f, indent=4)
        
        # Visualize results
        self.visualize_results(avg_results)
        
        return avg_results
    
    def visualize_results(self, results):
        """Create visualizations for the experiment results"""
        
        k_values = results['k_values']
        
        # 设置科学绘图风格
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.dpi'] = 300
        
        # 创建K值对比可视化
        self.plot_k_comparison()
        
        # 定义美观的配色方案 - 使用科学绘图常用的配色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # 1. Accuracy vs. Parameter Count
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        
        # Plot RMSE
        rmse_values = [results['avg_metrics'][k]['rmse'] for k in k_values]
        param_counts = [results['parameter_count'][k] for k in k_values]
        
        ax1.plot(param_counts, rmse_values, 'o-', color='#1E88E5', label='RMSE', linewidth=2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Parameter Count (log scale)', fontweight='bold')
        ax1.set_ylabel('RMSE', color='#1E88E5', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#1E88E5')
        
        # Add second y-axis for R²
        ax2 = ax1.twinx()
        r2_values = [results['avg_metrics'][k]['r_squared'] for k in k_values]
        ax2.plot(param_counts, r2_values, 'o-', color='#B32626', label='R²', linewidth=2)
        ax2.set_ylabel('R²', color='#B32626', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#B32626')
        
        # Add K labels with improved style
        for i, k in enumerate(k_values):
            plt.annotate(f'K={k}', 
                         (param_counts[i], rmse_values[i]),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.title('Accuracy vs. Parameter Count', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add combined legend with improved style
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_vs_parameters.png'), dpi=300)
        plt.close()
        
        # 2. Computation Time vs. K Value
        plt.figure(figsize=(10, 6))
        
        time_values = [results['avg_metrics'][k]['computation_time'] for k in k_values]
        time_std = [results['std_metrics'][k]['computation_time'] for k in k_values]
        
        plt.errorbar(k_values, time_values, yerr=time_std, fmt='o-', capsize=5, 
                    color='#1E88E5', ecolor='#B32626', label='Computation Time', linewidth=2)
        
        plt.xlabel('Scaling Factor (K)', fontweight='bold')
        plt.ylabel('Computation Time (seconds)', fontweight='bold')
        plt.title('Computation Time vs. Scaling Factor', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(k_values)
        
        # 添加95%置信区间阴影
        plt.fill_between(k_values, 
                         [time_values[i] - time_std[i] for i in range(len(k_values))],
                         [time_values[i] + time_std[i] for i in range(len(k_values))],
                         color='gray', alpha=0.2, label='95% Confidence Interval')
        
        plt.legend(frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_vs_k.png'), dpi=300)
        plt.close()
        
        # 3. Multi-metric plot
        plt.figure(figsize=(12, 10))
        metrics = ['rmse', 'r_squared', 'correlation', 'dem_correlation']
        metric_colors = {'rmse': '#1E88E5', 'r_squared': '#B32626', 
                         'correlation': '#2CA02C', 'dem_correlation': '#9467BD'}
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            
            values = [results['avg_metrics'][k][metric] for k in k_values]
            std_values = [results['std_metrics'][k][metric] for k in k_values]
            
            # 使用统一的颜色方案
            plt.errorbar(k_values, values, yerr=std_values, fmt='o-', capsize=5,
                        color=metric_colors[metric], ecolor='gray', linewidth=2)
            
            # 添加95%置信区间阴影
            plt.fill_between(k_values, 
                            [values[i] - std_values[i] for i in range(len(k_values))],
                            [values[i] + std_values[i] for i in range(len(k_values))],
                            color='gray', alpha=0.2)
            
            # 为每个K值添加标签
            for j, k in enumerate(k_values):
                plt.annotate(f'K={k}', 
                            (k_values[j], values[j]),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            plt.xlabel('Scaling Factor (K)', fontweight='bold')
            plt.ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            plt.title(f'{metric.replace("_", " ").title()} vs. K', fontweight='bold', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xticks(k_values)
            
            # 添加R²值
            if len(values) > 1:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(k_values, values)
                r_squared = r_value**2
                plt.annotate(f'R² = {r_squared:.4f}', 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multi_metric_plot.png'), dpi=300)
        plt.close()
        
        # 4. Trade-off visualization
        plt.figure(figsize=(10, 6))
        
        # Normalize metrics to 0-1 range for comparison
        def normalize(values):
            min_val = min(values)
            max_val = max(values)
            return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
        
        norm_time = normalize(time_values)
        norm_rmse = normalize(rmse_values)
        norm_r2 = [1 - v for v in normalize(r2_values)]  # Invert R² so lower is better
        
        # Combined score (lower is better)
        combined_score = [0.4*t + 0.3*r + 0.3*r2 for t, r, r2 in zip(norm_time, norm_rmse, norm_r2)]
        
        # 使用统一的配色方案
        plt.plot(k_values, norm_time, 'o-', color='#1E88E5', label='Normalized Time', linewidth=2)
        plt.plot(k_values, norm_rmse, 's-', color='#B32626', label='Normalized RMSE', linewidth=2)
        plt.plot(k_values, norm_r2, '^-', color='#2CA02C', label='Normalized (1-R²)', linewidth=2)
        plt.plot(k_values, combined_score, 'D-', color='#9467BD', label='Combined Score', linewidth=2.5)
        
        plt.xlabel('Scaling Factor (K)', fontweight='bold')
        plt.ylabel('Normalized Score (lower is better)', fontweight='bold')
        plt.title('Trade-off Analysis', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(k_values)
        
        # 添加K值标签
        for i, k in enumerate(k_values):
            plt.annotate(f'K={k}', 
                        (k_values[i], combined_score[i]),
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 改进图例样式
        plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
        
        # Highlight the best K value based on combined score
        best_k_idx = np.argmin(combined_score)
        best_k = k_values[best_k_idx]
        plt.axvline(x=best_k, color='#8C564B', linestyle='--', alpha=0.7, linewidth=1.5)
        plt.annotate(f'Best K = {best_k}', 
                    xy=(best_k, combined_score[best_k_idx]),
                    xytext=(0, 20),
                    textcoords="offset points",
                    ha='center',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#8C564B', boxstyle='round,pad=0.3'),
                    arrowprops=dict(arrowstyle="->", color='#8C564B', linewidth=1.5))
        
        # 添加R²值标注
        for i, (metric_name, values) in enumerate([
            ('Time', norm_time), 
            ('RMSE', norm_rmse), 
            ('1-R²', norm_r2), 
            ('Combined', combined_score)
        ]):
            if len(values) > 1:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(k_values, values)
                r_squared = r_value**2
                colors = ['#1E88E5', '#B32626', '#2CA02C', '#9467BD']
                plt.annotate(f'{metric_name}: R² = {r_squared:.2f}', 
                            xy=(0.02, 0.95 - i*0.05), xycoords='axes fraction',
                            fontsize=9, fontweight='bold', color=colors[i],
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tradeoff_analysis.png'), dpi=300)
        plt.close()
        
        # 5. Create summary plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Parameter Count vs. Accuracy
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(param_counts, rmse_values, 'o-', color='#1E88E5', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Parameter Count (log scale)', fontweight='bold')
        plt.ylabel('RMSE', fontweight='bold')
        plt.title('Parameter Count vs. RMSE', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加K值标签
        for i, k in enumerate(k_values):
            plt.annotate(f'K={k}', 
                        (param_counts[i], rmse_values[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 添加R²值
        if len(param_counts) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(np.log10(param_counts), rmse_values)
            r_squared = r_value**2
            plt.annotate(f'R² = {r_squared:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Plot 2: K vs. Computation Time
        ax2 = plt.subplot(2, 2, 2)
        plt.errorbar(k_values, time_values, yerr=time_std, fmt='o-', capsize=5,
                    color='#B32626', ecolor='gray', linewidth=2)
        plt.fill_between(k_values, 
                        [time_values[i] - time_std[i] for i in range(len(k_values))],
                        [time_values[i] + time_std[i] for i in range(len(k_values))],
                        color='gray', alpha=0.2)
        plt.xlabel('Scaling Factor (K)', fontweight='bold')
        plt.ylabel('Computation Time (seconds)', fontweight='bold')
        plt.title('K vs. Computation Time', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(k_values)
        
        # 添加R²值
        if len(k_values) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(k_values, time_values)
            r_squared = r_value**2
            plt.annotate(f'R² = {r_squared:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Plot 3: K vs. R²
        ax3 = plt.subplot(2, 2, 3)
        plt.errorbar(k_values, r2_values, 
                    yerr=[results['std_metrics'][k]['r_squared'] for k in k_values], 
                    fmt='o-', capsize=5, color='#2CA02C', ecolor='gray', linewidth=2)
        plt.fill_between(k_values, 
                        [r2_values[i] - results['std_metrics'][k_values[i]]['r_squared'] for i in range(len(k_values))],
                        [r2_values[i] + results['std_metrics'][k_values[i]]['r_squared'] for i in range(len(k_values))],
                        color='gray', alpha=0.2)
        plt.xlabel('Scaling Factor (K)', fontweight='bold')
        plt.ylabel('R²', fontweight='bold')
        plt.title('K vs. R²', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(k_values)
        
        # 添加R²值
        if len(k_values) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(k_values, r2_values)
            r_squared = r_value**2
            plt.annotate(f'R² = {r_squared:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Plot 4: Trade-off visualization
        ax4 = plt.subplot(2, 2, 4)
        plt.plot(k_values, combined_score, 'o-', color='#9467BD', linewidth=2)
        plt.xlabel('Scaling Factor (K)', fontweight='bold')
        plt.ylabel('Combined Score (lower is better)', fontweight='bold')
        plt.title('K vs. Combined Score', fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(k_values)
        
        # 高亮最佳K值
        best_k_idx = np.argmin(combined_score)
        best_k = k_values[best_k_idx]
        plt.axvline(x=best_k, color='#8C564B', linestyle='--', alpha=0.7)
        plt.annotate(f'Best K = {best_k}', 
                    xy=(best_k, combined_score[best_k_idx]),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha='center',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='#8C564B', boxstyle='round,pad=0.2'),
                    arrowprops=dict(arrowstyle="->", color='#8C564B'))
        
        # 添加R²值
        if len(k_values) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(k_values, combined_score)
            r_squared = r_value**2
            plt.annotate(f'R² = {r_squared:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.suptitle('K Sensitivity Experiment - Summary', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'summary.png'), dpi=300)
        plt.close()


    def plot_k_comparison(self):
        """Create a comparison visualization of DEMs and uplift fields for different K values"""
        
        # 加载真实隆升场和目标DEM
        true_uplift = np.load(os.path.join(self.output_dir, 'true_uplift.npy'))
        target_dem = np.load(os.path.join(self.output_dir, 'target_dem.npy'))
        
        # 获取K值列表
        k_values = self.config['experiment']['k_values']
        
        # 选择要展示的K值（如果K值太多，可以选择其中的几个）
        if len(k_values) > 3:
            # 选择最小、中间和最大的K值
            display_k_values = [k_values[0], k_values[len(k_values)//2], k_values[-1]]
        else:
            display_k_values = k_values
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10), dpi=300)
        
        # 设置标题
        plt.suptitle('Comparison of DEMs and Uplift Fields for Different K Values', 
                    fontsize=16, fontweight='bold')
        
        # 确定子图数量和布局
        n_k_values = len(display_k_values)
        n_rows = 2  # 第一行是DEM，第二行是隆升场
        n_cols = n_k_values + 1  # 每个K值一列，加上真实值一列
        
        # 统一的颜色映射
        dem_cmap = 'terrain'
        uplift_cmap = 'viridis'
        
        # 获取DEM和隆升场的值范围，用于统一颜色映射
        dem_vmin = np.min(target_dem)
        dem_vmax = np.max(target_dem)
        uplift_vmin = np.min(true_uplift)
        uplift_vmax = np.max(true_uplift)
        
        # 绘制真实DEM和隆升场（第一列）
        # 真实DEM
        ax_true_dem = plt.subplot(n_rows, n_cols, 1)
        im_true_dem = ax_true_dem.imshow(target_dem, cmap=dem_cmap, 
                                        vmin=dem_vmin, vmax=dem_vmax)
        ax_true_dem.set_title('Target DEM', fontweight='bold')
        ax_true_dem.set_xticks([])
        ax_true_dem.set_yticks([])
        
        # 真实隆升场
        ax_true_uplift = plt.subplot(n_rows, n_cols, n_cols + 1)
        im_true_uplift = ax_true_uplift.imshow(true_uplift, cmap=uplift_cmap, 
                                            vmin=uplift_vmin, vmax=uplift_vmax)
        ax_true_uplift.set_title('True Uplift', fontweight='bold')
        ax_true_uplift.set_xticks([])
        ax_true_uplift.set_yticks([])
        
        # 为每个K值绘制反演结果
        for i, k in enumerate(display_k_values):
            col_idx = i + 1  # 列索引（从0开始）
            
            # 加载该K值的反演隆升场（使用第一次重复实验的结果）
            inverted_uplift_path = os.path.join(self.output_dir, f'inverted_uplift_k{k}_rep0.npy')
            
            if os.path.exists(inverted_uplift_path):
                inverted_uplift = np.load(inverted_uplift_path)
                
                # 使用反演的隆升场运行前向模型生成DEM
                model_params = self.config['model_params'].copy()
                
                # 创建合成侵蚀系数场
                from synthetic_erosion_field import create_synthetic_erosion_field
                Ksp = create_synthetic_erosion_field(
                    shape=inverted_uplift.shape,
                    base_k_sp=model_params['k_sp_base']
                )
                model_params['Ksp'] = Ksp
                
                # 运行前向模型
                simulated_dem = run_fastscape_model(
                    k_sp=Ksp,
                    uplift=inverted_uplift,
                    k_diff=model_params['d_diff'],
                    x_size=inverted_uplift.shape[1],
                    y_size=inverted_uplift.shape[0],
                    spacing=model_params['spacing'],
                    boundary_status=model_params['boundary_status'],
                    area_exp=model_params['area_exp'],
                    slope_exp=model_params['slope_exp'],
                    time_total=model_params['time_total']
                )
                
                # 绘制模拟DEM
                ax_dem = plt.subplot(n_rows, n_cols, col_idx + 1)
                im_dem = ax_dem.imshow(simulated_dem, cmap=dem_cmap, 
                                      vmin=dem_vmin, vmax=dem_vmax)
                ax_dem.set_title(f'DEM (K={k})', fontweight='bold')
                ax_dem.set_xticks([])
                ax_dem.set_yticks([])
                
                # 绘制反演隆升场
                ax_uplift = plt.subplot(n_rows, n_cols, n_cols + col_idx + 1)
                im_uplift = ax_uplift.imshow(inverted_uplift, cmap=uplift_cmap, 
                                           vmin=uplift_vmin, vmax=uplift_vmax)
                ax_uplift.set_title(f'Inverted Uplift (K={k})', fontweight='bold')
                ax_uplift.set_xticks([])
                ax_uplift.set_yticks([])
                
                # 计算相似度指标
                from scipy.stats import pearsonr
                from sklearn.metrics import mean_squared_error
                
                # DEM相似度
                dem_corr, _ = pearsonr(target_dem.flatten(), simulated_dem.flatten())
                dem_rmse = np.sqrt(mean_squared_error(target_dem, simulated_dem))
                
                # 隆升场相似度
                uplift_corr, _ = pearsonr(true_uplift.flatten(), inverted_uplift.flatten())
                uplift_rmse = np.sqrt(mean_squared_error(true_uplift, inverted_uplift))
                
                # 在图上添加相似度指标
                ax_dem.text(0.5, -0.1, f'Corr: {dem_corr:.2f}, RMSE: {dem_rmse:.2f}',
                          transform=ax_dem.transAxes, ha='center', fontsize=9)
                
                ax_uplift.text(0.5, -0.1, f'Corr: {uplift_corr:.2f}, RMSE: {uplift_rmse:.2f}',
                             transform=ax_uplift.transAxes, ha='center', fontsize=9)
            else:
                # 如果文件不存在，显示空白子图
                ax_dem = plt.subplot(n_rows, n_cols, col_idx + 1)
                ax_dem.text(0.5, 0.5, f'No data for K={k}', ha='center', va='center')
                ax_dem.set_xticks([])
                ax_dem.set_yticks([])
                
                ax_uplift = plt.subplot(n_rows, n_cols, n_cols + col_idx + 1)
                ax_uplift.text(0.5, 0.5, f'No data for K={k}', ha='center', va='center')
                ax_uplift.set_xticks([])
                ax_uplift.set_yticks([])
        
        # 添加颜色条
        # DEM颜色条
        cbar_dem_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
        cbar_dem = fig.colorbar(im_true_dem, cax=cbar_dem_ax)
        cbar_dem.set_label('Elevation (m)', fontweight='bold')
        
        # 隆升场颜色条
        cbar_uplift_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
        cbar_uplift = fig.colorbar(im_true_uplift, cax=cbar_uplift_ax)
        cbar_uplift.set_label('Uplift Rate (mm/yr)', fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'k_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved K comparison visualization to {os.path.join(self.output_dir, 'k_comparison.png')}")

if __name__ == "__main__":
    # Run the experiment
    experiment = KSensitivityExperiment()
    results = experiment.run_experiment()
    
    # Print summary of results
    print("\nExperiment Results Summary:")
    print("="*50)
    print(f"Experiment output directory: {experiment.output_dir}")
    print("\nAccuracy vs. K Value:")
    for k in results['k_values']:
        metrics = results['avg_metrics'][k]
        if 'all_runs_failed' not in metrics:
            print(f"K={k}: RMSE={metrics['rmse']:.4f}, R²={metrics['r_squared']:.4f}, "
                 f"Time={metrics['computation_time']:.2f}s, "
                 f"Params={results['parameter_count'][k]}")
    
    # Determine best K value based on combined score
    k_values = results['k_values']
    time_values = [results['avg_metrics'][k]['computation_time'] for k in k_values]
    rmse_values = [results['avg_metrics'][k]['rmse'] for k in k_values]
    r2_values = [results['avg_metrics'][k]['r_squared'] for k in k_values]
    
    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
    
    norm_time = normalize(time_values)
    norm_rmse = normalize(rmse_values)
    norm_r2 = [1 - v for v in normalize(r2_values)]
    
    combined_score = [0.4*t + 0.3*r + 0.3*r2 for t, r, r2 in zip(norm_time, norm_rmse, norm_r2)]
    best_k_idx = np.argmin(combined_score)
    best_k = k_values[best_k_idx]
    
    print("\nOptimal Scaling Factor:")
    print(f"Best K value: {best_k} (based on combined performance score)")
    print("="*50)