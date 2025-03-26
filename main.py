# main.py
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
from typing import Dict, Any, Optional
import warnings
from datetime import datetime
import sys
# 过滤所有警告
warnings.filterwarnings('ignore')
# 特定警告过滤
warnings.filterwarnings("ignore", category=FutureWarning, module='xsimlab')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", message="Setting up \[LPIPS\]")
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
np.seterr(all='ignore')
# 导入自定义模块
from data_loader import read_terrain_data, read_shapefile,load_dem_data,calculate_shp_rotation_angle, rotate_data, reproject_files_to_geographic
from data_preprocessing import preprocess_terrain_data, interpolate_uplift_cv,unify_array_sizes
from model_runner import run_fastscape_model
from fitness_evaluator import terrain_similarity
from genetic_algorithm import optimize_uplift_ga  
from erosion_field import create_erosion_field, display_erosion_field, verify_erosion_field
from visualization_utils import (
    plot_comparison,
    plot_uplift_distribution_x,
    plot_uplift_distribution_y,
    plot_single_data,
    display_array_info,
    display_tiff_info,
    plot_3d_surface,
    plot_optimization_history
)
from path_validator import verify_config_paths, verify_file_path, verify_directory_path
from scipy.ndimage import generic_filter

# 在文件开头添加这些日志配置函数
def setup_basic_logging():
    """设置基础日志配置"""
    # 清除现有的处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # 设置基本配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 使用stdout而不是stderr
        ]
    )

def setup_file_logging(output_path):
    """添加文件日志处理器"""
    # 创建文件处理器
    log_file = os.path.join(output_path, 'optimization.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # 添加到根日志记录器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logging.info(f"文件日志系统已初始化，日志文件: {log_file}")

# 将目标函数定义在全局作用域

def create_objective_function(resampled_dem, LOW_RES_SHAPE, ORIGINAL_SHAPE, 
                           Ksp, D_DIFF, row, col, spacing, time_step_num, 
                           total_simulation_time, terrain_resolution, 
                           feature_smooth_radius, boundary_status='fixed_value',
                           area_exp=0.43, slope_exp=1):
    """创建优化目标函数"""
    def objective_function(uplift_vector):
        try:
            # 重塑隆升率向量
            uplift_vector = np.array(uplift_vector).reshape(LOW_RES_SHAPE)
            
            # 插值到高分辨率
            full_res_uplift = interpolate_uplift_cv(uplift_vector, ORIGINAL_SHAPE)
            
            # 运行Fastscape模型
            generated_elevation = run_fastscape_model(
                k_sp=Ksp,
                uplift=full_res_uplift,
                k_diff=D_DIFF,
                x_size=col,
                y_size=row,
                spacing=spacing,
                boundary_status=boundary_status,
                area_exp=area_exp,
                slope_exp=slope_exp,
                time_total=total_simulation_time
            )
            
            # 计算地形相似度
            similarity = terrain_similarity(
                matrix1=resampled_dem,
                matrix2=generated_elevation,
                resolution=terrain_resolution,
                smooth_radius=feature_smooth_radius
            )
            
            return 1 - similarity  # 最小化不相似度
            
        except Exception as e:
            logging.error(f"目标函数计算失败: {e}")
            return 1.0  # 失败时返回最大不相似度
            
    return objective_function

def load_config(config_path: str = './config.ini') -> configparser.ConfigParser:
    """加载配置文件"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        config = configparser.ConfigParser()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config.read_file(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config.read_file(f)
                
        if 'Paths' in config:
            for key in config['Paths']:
                value = config['Paths'][key].split(';')[0].strip()
                value = value.replace('\\\\', '\\').replace('\\', '/')
                config['Paths'][key] = value
        
        return config
        
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        print(f"当前工作目录: {os.getcwd()}")
        raise

def verify_config(config: configparser.ConfigParser) -> bool:
    """验证配置文件的完整性"""
    required_sections = ['Paths', 'Model', 'GeneticAlgorithm', 'Preprocessing']
    required_params = {
        'Paths': ['terrain_path', 'fault_shp_path', 'study_area_shp_path', 'output_path'],
        'Model': ['k_sp_value', 'ksp_fault', 'd_diff_value', 'boundary_status', 
                 'area_exp', 'slope_exp', 'time_total'],
        'GeneticAlgorithm': ['ga_pop_size', 'ga_max_iter', 'ga_prob_cross', 
                            'ga_prob_mut', 'lb', 'ub', 'n_jobs'],
        'Preprocessing': ['smooth_sigma', 'scale_factor', 
                          'ratio']
    }
    
    for section in required_sections:
        if section not in config:
            logging.error(f"缺少配置节: {section}")
            return False
        for param in required_params[section]:
            if param not in config[section]:
                logging.error(f"缺少参数: {section}/{param}")
                return False

    if not verify_config_paths(config):
        return False

    return True

# 裁剪掉 NaN 值，保留有效区域
def trim_nan_edges(matrix):
    """裁剪矩阵边缘的 NaN 值"""
    # 找到非 NaN 值的边界
    rows = np.any(~np.isnan(matrix), axis=1)
    cols = np.any(~np.isnan(matrix), axis=0)
    
    # 获取有效区域的索引范围
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]
    
    # 裁剪矩阵
    return matrix[row_start:row_end+1, col_start:col_end+1]

def setup_logging(output_path: str) -> None:
    """配置日志系统"""
    log_file = os.path.join(output_path, 'optimization.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_optimization_results(output_path: str, results: dict):
    """
    保存优化结果，处理MaskedArray类型。

    参数:
    - output_path: 输出目录路径
    - results: 包含结果数据的字典
    """
    try:
        for name, data in results.items():
            if data is None:
                logging.warning(f"跳过保存 {name}: 数据为None")
                continue

            save_path = os.path.join(output_path, f'{name}.npy')
            
            try:
                # 检查是否为MaskedArray类型
                if isinstance(data, np.ma.MaskedArray):
                    # 将masked值替换为NaN并转换为普通数组
                    data_filled = data.filled(np.nan)
                    np.save(save_path, data_filled)
                elif isinstance(data, (np.ndarray, list)):
                    np.save(save_path, np.array(data))
                else:
                    logging.warning(f"跳过保存 {name}: 不支持的数据类型 {type(data)}")
                    continue
                    
                logging.info(f"已保存 {name} 到 {save_path}")
                
            except Exception as e:
                logging.error(f"保存 {name} 时出错: {e}")
                continue
                
    except Exception as e:
        logging.error(f"保存结果时出错: {e}")
        raise

def fill_Nan(dem_array):
    """
    使用最邻近插值填充 NaN 值
    
    参数:
    - dem_array: 包含 NaN 的输入数组
    
    返回:
    - filled_array: 填充后的数组
    """
    from scipy.interpolate import NearestNDInterpolator
    
    # 创建数组副本
    filled_array = dem_array.copy()
    
    # 获取非 NaN 值的位置和值
    mask = ~np.isnan(dem_array)
    coords = np.array(np.where(mask)).T
    values = dem_array[mask]
    
    # 创建插值器
    interpolator = NearestNDInterpolator(coords, values)
    
    # 获取 NaN 值的位置
    nan_coords = np.array(np.where(~mask)).T
    
    # 填充 NaN 值
    if len(nan_coords) > 0:
        filled_array[~mask] = interpolator(nan_coords)
    
    return filled_array

def main():
    """主程序入口"""
    try:
        # 1. 首先设置基础日志配置（在任何其他操作之前）
        setup_basic_logging()  # 这是新添加的基础日志配置
        logging.info("程序开始执行")
        
        # 2. 加载配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        
        if not os.path.exists(config_path):
            logging.error(f"配置文件未找到: {config_path}")
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        config = load_config(config_path)
        logging.info("配置文件加载完成")

        # 3. 创建输出目录
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = os.path.join(
            config['Paths']['output_path'],
            f'Expt_{timestamp}'
        )
        
        try:
            os.makedirs(output_path, exist_ok=True)
            logging.info(f"创建输出目录成功: {output_path}")
            
            # 4. 更新日志配置以包含文件输出
            setup_file_logging(output_path)  # 添加文件日志处理器
            
        except Exception as e:
            logging.error(f"创建输出目录失败: {e}", exc_info=True)
            return

        # 设置日志
        #setup_logging(output_path)
        logging.info("开始优化过程")

        # 验证输入文件路径
        terrain_path = verify_file_path(config['Paths']['terrain_path'], '地形栅格文件')
        fault_shp_path = verify_file_path(config['Paths']['fault_shp_path'], '断层 Shapefile')
        study_area_shp_path = verify_file_path(config['Paths']['study_area_shp_path'], '研究区域 Shapefile')

        if None in (terrain_path, fault_shp_path, study_area_shp_path):
            logging.error("一个或多个输入文件路径无效")
            return

        # 检查并统一投影坐标系 (保持这部分)
        logging.info("Step 2: 检查和统一投影坐标系")
        target_crs = config['Preprocessing']['target_crs'] # 从配置文件读取 target_crs
        config = reproject_files_to_geographic(config, target_crs=target_crs) # 传递 target_crs
        terrain_path = config['Paths']['terrain_path']
        fault_shp_path = config['Paths']['fault_shp_path']
        study_area_shp_path = config['Paths']['study_area_shp_path']

        # 1. 数据加载 (加载 *未旋转的* DEM 和 Shapefiles)
        logging.info("Step 1: 数据加载")
        ratio = config.getfloat('Preprocessing', 'ratio')
        # 加载DEM数据，不进行旋转
        dem_data, dem_profile = load_dem_data(
            file_path=terrain_path,
            study_area_shp_path=study_area_shp_path,
            ratio=ratio
        )

        # 读取 shapefiles
        study_area = read_shapefile(study_area_shp_path)
        fault_lines = read_shapefile(fault_shp_path)

        # 计算旋转角度 (基于 study_area shapefile)
        rotation_angle = calculate_shp_rotation_angle(study_area_shp_path)
        print(f"Calculated rotation angle: {rotation_angle:.2f}°")


        # 2. 创建侵蚀系数场 (在 *原始未旋转* 空间创建)
        logging.info("Step 2: 创建侵蚀系数场")
        global row, col, ORIGINAL_SHAPE, LOW_RES_SHAPE, matrix
        global Ksp, D_DIFF, time_step_num, total_simulation_time
        global terrain_resolution, feature_smooth_radius

        row, col = dem_data.shape # 使用原始 dem_data 的 shape
        ORIGINAL_SHAPE = (row, col)
        scale_factor = config.getint('Preprocessing', 'scale_factor')
        LOW_RES_SHAPE = (row // scale_factor, col // scale_factor)
        logging.info(f"Original shape: {ORIGINAL_SHAPE}")
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        k_sp_value = config.getfloat('Model', 'k_sp_value')
        ksp_fault = config.getfloat('Model', 'ksp_fault')

        #  创建 erosion field *在原始 DEM 空间* (rotation_angle=0)
        Ksp = create_erosion_field(
            shape=ORIGINAL_SHAPE,
            base_k_sp=k_sp_value,
            fault_k_sp=ksp_fault,
            fault_shp_path=fault_shp_path,
            study_area_shp_path=study_area_shp_path,
            rotation_angle = rotation_angle,
            border_width=2
        )

        if not verify_erosion_field(Ksp, shape=ORIGINAL_SHAPE):
            logging.error("侵蚀系数场验证失败")
            return

        # 3. 旋转 DEM 和 Ksp Field (一起旋转)
        logging.info("Step 3: 旋转 DEM 和侵蚀系数场")
        rotated_dem_data = rotate_data(dem_data, rotation_angle)
        rotated_dem_data = fill_Nan(rotated_dem_data)
        rotated_Ksp = Ksp 
        rotated_Ksp = fill_Nan(rotated_Ksp)
        
        # 添加详细的尺寸日志
        logging.info(f"Shape comparison:")
        logging.info(f"Original DEM shape: {dem_data.shape}")
        logging.info(f"Rotated DEM shape: {rotated_dem_data.shape}")
        logging.info(f"Ksp shape: {Ksp.shape}")
        logging.info(f"rotated_Ksp shape: {rotated_Ksp.shape}")

        # 从日志可以看出，问题出在 Ksp 的创建过程中。
        # 虽然 Original DEM shape 是 (177, 189)，旋转后变成 (85, 176)，
        # 但是 Ksp 在创建时就是 (87, 176)，这说明在调用 create_erosion_field 时的形状计算有问题。
        # 添加数据验证
        if rotated_dem_data.shape != rotated_Ksp.shape:
            logging.error(f"Shape mismatch between rotated DEM and Ksp")
            logging.error(f"DEM: {rotated_dem_data.shape}, Ksp: {rotated_Ksp.shape}")
            # 统一尺寸
            rotated_dem_data, rotated_Ksp = unify_array_sizes(rotated_dem_data, rotated_Ksp)
            logging.info(f"After trimming:")
            logging.info(f"DEM shape: {rotated_dem_data.shape}")
            logging.info(f"Ksp shape: {rotated_Ksp.shape}")

        # 保存旋转后的 Ksp
        try:
            rotated_ksp_save_path = os.path.join(output_path, 'rotated_ksp.npy')
            np.save(rotated_ksp_save_path, rotated_Ksp)
            logging.info(f"已保存旋转后的侵蚀系数场到: {rotated_ksp_save_path}")
        except Exception as e:
            logging.error(f"保存旋转后的侵蚀系数场失败: {e}")
        # 更新全局变量以反映旋转后的尺寸
        resampled_dem = rotated_dem_data
        row, col = resampled_dem.shape  # 更新为旋转后的尺寸
        ORIGINAL_SHAPE = (row, col)     # 更新为旋转后的尺寸
        LOW_RES_SHAPE = (row // scale_factor, col // scale_factor)

        # 更新 dem_profile
        dem_profile['height'] = row
        dem_profile['width'] = col

        # 检查并统一全局尺寸
        def validate_shapes():
            global row, col, ORIGINAL_SHAPE, LOW_RES_SHAPE
            if resampled_dem.shape != rotated_Ksp.shape:
                logging.error("Shape mismatch in global variables")
                return False
            row, col = resampled_dem.shape
            ORIGINAL_SHAPE = (row, col)
            LOW_RES_SHAPE = (row // scale_factor, col // scale_factor)
            logging.info(f"Validated shapes: ORIGINAL_SHAPE={ORIGINAL_SHAPE}, LOW_RES_SHAPE={LOW_RES_SHAPE}")
            return True

        # 在更新全局变量后调用验证
        if not validate_shapes():
            raise ValueError("Shape validation failed")

        resampled_dem = rotated_dem_data #  重命名 rotated_dem_data 为 resampled_dem 以便后续代码兼容
        spacing_x = dem_profile['transform'][0]
        spacing_y = abs(dem_profile['transform'][4])
        spacing = (spacing_x + spacing_y) / 2 # 计算 spacing (可能需要更精确的计算)


        display_array_info("Rotated DEM", rotated_dem_data, spacing)
        display_array_info("Rotated Ksp", rotated_Ksp, spacing)





        # In main.py, after DEM preprocessing (using rotated DEM):
        dem_profile_for_raster = {
            'transform': dem_profile['transform'], # 使用 dem_profile 的 transform (可能需要在重采样时更新 transform - 检查 preprocess_terrain_data)
            'shape': resampled_dem.shape # 使用 resampled DEM 的 shape
        }


        # 5. 设置模型参数 (使用 *旋转后* 的数据)
        logging.info("Step 5: 模型参数设置")


        LOW_RES_SHAPE = (row // scale_factor, col // scale_factor) # 重新计算 LOW_RES_SHAPE
        logging.info(f"Resampled shape after rotation: {ORIGINAL_SHAPE}") #  注意这里 ORIGINAL_SHAPE 仍然是原始shape，应该输出 resampled_dem.shape 或 ORIGINAL_SHAPE = resampled_dem.shape
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        D_DIFF = config.getfloat('Model', 'd_diff_value')
        time_step_num = 101  # 可以添加到config文件中
        total_simulation_time = config.getfloat('Model', 'time_total')
        terrain_resolution = spacing # 可以添加到config文件中
        feature_smooth_radius = 2  # 可以添加到config文件中

        ga_params = {
            'pop': config.getint('GeneticAlgorithm', 'ga_pop_size'),
            'max_iter': config.getint('GeneticAlgorithm', 'ga_max_iter'),
            'prob_cross': config.getfloat('GeneticAlgorithm', 'ga_prob_cross'),
            'prob_mut': config.getfloat('GeneticAlgorithm', 'ga_prob_mut'),
            'lb': config.getfloat('GeneticAlgorithm', 'lb'),
            'ub': config.getfloat('GeneticAlgorithm', 'ub'),
            'decay_rate': config.getfloat('GeneticAlgorithm', 'decay_rate'),
            'min_size_pop': config.getint('GeneticAlgorithm', 'min_size_pop'),
            'patience': config.getint('GeneticAlgorithm', 'patience')
        }

        model_params = {
            'Ksp': rotated_Ksp, # 使用 *旋转后* 的 Ksp
            'd_diff': config.getfloat('Model', 'd_diff_value'),
            'boundary_status': config['Model']['boundary_status'],
            'area_exp': config.getfloat('Model', 'area_exp'),
            'slope_exp': config.getfloat('Model', 'slope_exp'),
            'time_total': config.getfloat('Model', 'time_total'),
            'spacing': spacing
        }

        # 创建objective function (使用 *旋转后* 的 resampled_dem 和 Ksp)
        obj_func = create_objective_function(
            resampled_dem=resampled_dem,
            LOW_RES_SHAPE=LOW_RES_SHAPE,
            ORIGINAL_SHAPE=ORIGINAL_SHAPE, #  这里 ORIGINAL_SHAPE 仍然是原始shape，需要考虑是否修改为旋转后的shape
            Ksp=rotated_Ksp,
            D_DIFF=D_DIFF,
            row=row,
            col=col,
            spacing=spacing,
            time_step_num=time_step_num,
            total_simulation_time=total_simulation_time,
            terrain_resolution=terrain_resolution,
            feature_smooth_radius=feature_smooth_radius
        )

        # 显示原始DEM
        plt.figure(figsize=(15, 10))
        plot_single_data(dem_data, "Original DEM", cmap='terrain', origin='upper') # 显示 *原始* DEM
        plt.savefig(os.path.join(output_path, 'original_dem.png'))
        plt.close()

        # 显示旋转后的DEM
        plt.figure(figsize=(15, 10))
        plot_single_data(rotated_dem_data, "Rotated DEM", cmap='terrain', origin='upper') # 显示 *旋转后* DEM
        plt.savefig(os.path.join(output_path, 'rotated_dem.png'))
        plt.close()

        # 显示侵蚀系数场
        display_erosion_field(rotated_Ksp, shape=ORIGINAL_SHAPE) #  显示 *旋转后* 的 Ksp
        plt.savefig(os.path.join(output_path, 'erosion_field.png'))
        plt.close()

        #叠加显示DEM和侵蚀系数场
        plt.figure(figsize=(15, 10))
        plt.imshow(rotated_dem_data, cmap='terrain', origin='upper')
        plt.imshow(rotated_Ksp, cmap='RdBu_r', alpha=0.5, origin='upper')
        plt.title("Rotated DEM with Erosion Coefficient Field")
        plt.savefig(os.path.join(output_path, 'dem_with_erosion_field.png'))
        plt.close()
        

        # 绘制DEM对比图
        plot_comparison(
            data1=dem_data, #  对比 *原始* DEM
            data2=rotated_dem_data, # 和 *旋转后* DEM
            title1='Original DEM',
            title2='Rotated DEM',
            value1='Elevation (m)',
            value2='Elevation (m)',
            cmap='terrain',
            figsize=(15, 10)
        )
        plt.savefig(os.path.join(output_path, 'dem_rotation_comparison.png'))
        plt.close()

        #绘制Ksp对比图
        plot_comparison(
            data1=Ksp, #  对比 *原始* Ksp
            data2=rotated_Ksp, # 和 *旋转后* Ksp
            title1='Original Ksp',
            title2='Rotated Ksp',
            value1='Erosion Coefficient',
            value2='Erosion Coefficient',
            cmap='RdBu_r',
            figsize=(15, 10)
        )
        plt.savefig(os.path.join(output_path, 'ksp_rotation_comparison.png'))
        plt.close()

        # 5. 遗传算法优化
        logging.info("Step 4: 遗传算法优化")
        n_jobs = config.getint('GeneticAlgorithm', 'n_jobs')

        logging.info("Genetic Algorithm Parameters:")
        for key, value in ga_params.items():
            logging.info(f"{key}: {value}")

        start_time = time.time()
        best_uplift, best_fitness, fitness_history = optimize_uplift_ga(
            obj_func=obj_func,
            resampled_dem=resampled_dem,
            LOW_RES_SHAPE=LOW_RES_SHAPE,
            ORIGINAL_SHAPE=ORIGINAL_SHAPE,
            ga_params=ga_params,
            model_params=model_params,
            n_jobs=n_jobs,
            run_mode='cached'
        )

        if best_uplift is not None:
            best_low_res_uplift = best_uplift.reshape(LOW_RES_SHAPE)
            best_full_res_uplift = interpolate_uplift_cv(best_low_res_uplift, ORIGINAL_SHAPE)
            logging.info(f"Best fitness: {best_fitness}")

            # 6. 绘制优化历史
            if fitness_history is not None:
                fig_history = plot_optimization_history(fitness_history)
                fig_history.savefig(os.path.join(output_path, 'optimization_history.png'))
                plt.close(fig_history)
                logging.info("优化历史曲线已保存")
        end_time = time.time()

        logging.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Best fitness: {best_fitness}")

        # 6. 结果处理和可视化
        logging.info("Step 5: 结果处理和可视化")
        if best_uplift is not None:
            best_low_res_uplift = best_uplift.reshape(LOW_RES_SHAPE)
            best_full_res_uplift = interpolate_uplift_cv(best_low_res_uplift, ORIGINAL_SHAPE)

            display_array_info("Best Uplift Field", best_full_res_uplift, spacing)

            # 绘制隆升率对比图
            plot_comparison(
                data1=best_low_res_uplift/10,
                data2=best_full_res_uplift/10,
                title1='Best Low Resolution Uplift',
                title2='Best Full Resolution Uplift',
                value1='Uplift Rate (mm/y)',
                value2='Uplift Rate (mm/y)',
                cmap='RdBu_r'
            )
            plt.savefig(os.path.join(output_path, 'uplift_comparison.png'))
            plt.close()

            # 生成最终地形
            final_elevation = run_fastscape_model(
                    k_sp=rotated_Ksp, # 使用旋转后的 Ksp
                    uplift=best_full_res_uplift,
                    k_diff=D_DIFF,
                    x_size=col,
                    y_size=row,
                    spacing=spacing,
                    boundary_status=config['Model']['boundary_status'],
                    area_exp=config.getfloat('Model', 'area_exp'),
                    slope_exp=config.getfloat('Model', 'slope_exp'),
                    time_total=total_simulation_time
            )

            # 绘制地形对比图
            plot_comparison(
                data1=final_elevation,
                data2=resampled_dem, #  注意这里对比的是 *旋转后且重采样* 的 DEM (resampled_dem)
                title1='Generated Terrain',
                title2='Target Landscape',
                value1='Elevation (m)',
                value2='Elevation (m)',
                cmap='terrain'
            )
            plt.savefig(os.path.join(output_path, 'terrain_comparison.png'))
            plt.close()

            # 绘制隆升分布图
            plot_uplift_distribution_x(best_full_res_uplift)
            plt.savefig(os.path.join(output_path, 'uplift_distribution.png'))
            plt.close()

            plot_uplift_distribution_y(best_full_res_uplift)
            plt.savefig(os.path.join(output_path, 'uplift_distribution.png'))
            plt.close()

            # 绘制3D地形可视化
            fig_3d = plot_3d_surface(
                data=final_elevation,
                uplift=best_full_res_uplift,
                title="3D Terrain with Uplift Field"
            )
            fig_3d.savefig(os.path.join(output_path, '3d_terrain.png'))
            plt.close(fig_3d)
            logging.info("3D地形可视化已保存")

            # 保存结果
            results = {
                    'best_full_res_uplift': best_full_res_uplift,
                    'final_elevation': final_elevation,
                    'fitness_history': np.array(fitness_history) if fitness_history is not None else None
                }
            save_optimization_results(output_path, results)
            # 8. 保存参数配置
            config_file = os.path.join(output_path, 'parameters.txt')
            with open(config_file, 'w') as f:
                f.write("Model Parameters:\n")
                for section in config.sections():
                    f.write(f"\n[{section}]\n")
                    for key, value in config[section].items():
                        f.write(f"{key} = {value}\n")

                f.write("\nOptimization Results:\n")
                f.write(f"Best Fitness: {best_fitness}\n")
                f.write(f"Optimization Time: {end_time - start_time:.2f} seconds\n")

            logging.info(f"所有结果已保存到: {output_path}")
            logging.info("优化过程成功完成")
            logging.info(f"Results saved to: {output_path}")

        else:
            logging.warning("遗传算法未能找到有效解。")

    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        logging.exception("Exception details:")

if __name__ == "__main__":
    main()  