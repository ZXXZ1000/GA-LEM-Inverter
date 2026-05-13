# Legacy main inversion workflow, now used through ga_lem_inverter.runner.
import logging
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from datetime import datetime
import sys
from scipy.stats import pearsonr, spearmanr
# 过滤所有警告
warnings.filterwarnings('ignore')
# 特定警告过滤
warnings.filterwarnings("ignore", category=FutureWarning, module='xsimlab')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", message=r"Setting up \[LPIPS\]")
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
np.seterr(all='ignore')
from ga_lem_inverter.config import UserConfigError
from ga_lem_inverter.integrations.pecube_fitness import PecubeFitnessEvaluator, pecube_spatial_adapter_from_dem_profile
from ga_lem_inverter.outputs import RunContext, write_metrics
from ga_lem_inverter.pipeline.data import read_shapefile, load_dem_data, calculate_shp_rotation_angle, rotate_data, reproject_files_to_geographic
from ga_lem_inverter.pipeline.preprocessing import interpolate_uplift_cv, unify_array_sizes
from ga_lem_inverter.pipeline.forward_model import align_model_field, run_fastscape_model, run_fastscape_series
from ga_lem_inverter.pipeline.fitness import terrain_similarity
from ga_lem_inverter.pipeline.optimization import optimize_uplift_ga
from ga_lem_inverter.pipeline.erosion import create_erosion_field, display_erosion_field, verify_erosion_field
from ga_lem_inverter.pipeline.visualization import (
    plot_comparison,
    plot_uplift_distribution_x,
    plot_uplift_distribution_y,
    plot_single_data,
    display_array_info,
    display_tiff_info,
    plot_3d_surface,
    plot_optimization_history
)


def _config_int(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: int,
    aliases: tuple[tuple[str, str], ...] = (),
) -> int:
    if config.has_option(section, option):
        return config.getint(section, option)
    for alias_section, alias_option in aliases:
        if config.has_option(alias_section, alias_option):
            return config.getint(alias_section, alias_option)
    return fallback


def _config_float(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: float,
    aliases: tuple[tuple[str, str], ...] = (),
) -> float:
    if config.has_option(section, option):
        return config.getfloat(section, option)
    for alias_section, alias_option in aliases:
        if config.has_option(alias_section, alias_option):
            return config.getfloat(alias_section, alias_option)
    return fallback
from ga_lem_inverter.pipeline.path_validator import verify_config_paths, verify_file_path


def validate_low_resolution_shape(shape: tuple[int, int], scale_factor: int) -> tuple[int, int]:
    """Validate the GA control grid implied by DEM shape and scale_factor."""
    rows, cols = (int(shape[0]), int(shape[1]))
    scale_factor = int(scale_factor)
    if scale_factor < 1:
        raise UserConfigError(f"scale_factor 必须 >= 1，当前为 {scale_factor}。")
    if rows < 2 or cols < 2:
        raise UserConfigError(f"DEM 尺寸太小，无法优化: shape={(rows, cols)}。")
    low_res_shape = (rows // scale_factor, cols // scale_factor)
    if min(low_res_shape) < 1:
        raise UserConfigError(
            f"scale_factor={scale_factor} 对 DEM shape={(rows, cols)} 过大，"
            "会导致低分辨率隆升控制网格为 0。请减小 scale_factor 或使用更大的 DEM。"
        )
    return low_res_shape


def validate_rotation_spatial_constraints(
    *,
    rotation_angle: float,
    pecube_enabled: bool,
    pecube_spatial_mode: str,
) -> dict[str, Any]:
    """Validate spatial interpretation when DEM arrays are rotated."""
    rotated = abs(float(rotation_angle)) > 1e-9
    spatial_mode = str(pecube_spatial_mode or "auto").strip().lower()
    if rotated and pecube_enabled and spatial_mode in {"auto", "dem", "dem_profile"}:
        raise UserConfigError(
            "当前配置同时启用了研究区旋转和 Pecube 自动 DEM 坐标。"
            "旋转后的 DEM 矩阵没有可靠的地理 transform，不能直接用于真实样品坐标。"
            "请将 study_area_shp_path 设为 none，或先在 GIS 中预处理 DEM 后再运行。"
        )
    return {
        "dem_rotated": rotated,
        "rotation_angle_degrees": float(rotation_angle),
        "spatial_reference_mode": "rotated_matrix" if rotated else "dem_georeferenced",
    }


def _json_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


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

def setup_file_logging(output_path, filename="optimization.log"):
    """添加文件日志处理器"""
    # 创建文件处理器
    log_file = os.path.join(output_path, filename)
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
                           area_exp=0.43, slope_exp=1, use_lpips=True,
                           pecube_evaluator=None, pecube_time_steps=2):
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
                smooth_radius=feature_smooth_radius,
                use_lpips=use_lpips
            )

            terrain_loss = 1 - similarity  # 最小化不相似度
            if pecube_evaluator is not None and pecube_evaluator.enabled:
                topography_series = None
                if pecube_time_steps > 2:
                    topography_series = run_fastscape_series(
                        k_sp=Ksp,
                        uplift=full_res_uplift,
                        k_diff=D_DIFF,
                        x_size=col,
                        y_size=row,
                        spacing=spacing,
                        boundary_status=boundary_status,
                        area_exp=area_exp,
                        slope_exp=slope_exp,
                        time_total=total_simulation_time,
                        output_steps=pecube_time_steps,
                    )
                result = pecube_evaluator.evaluate(
                    terrain_loss=terrain_loss,
                    generated_dem=generated_elevation,
                    uplift=full_res_uplift,
                    topography_series=topography_series,
                )
                return result.total_loss

            return terrain_loss

        except Exception as e:
            logging.error(f"目标函数计算失败: {e}")
            return np.inf  # 失败时交给 GA 失败率控制处理

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
        'Paths': ['terrain_path', 'output_path'],
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

def save_optimization_results(output_path: str | Path, results: dict, context: RunContext | None = None):
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

            save_path = Path(output_path) / f'{name}.npy'

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

                if context is not None:
                    context.add_artifact(save_path)
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

def resolve_optional_file_path(path_value: str, file_type: str) -> Optional[str]:
    """
    读取可选输入文件路径。

    配置里把路径留空、写 none/null/skip/false 时，程序会跳过对应功能。
    这让 demo 可以只靠 DEM 跑通；正式实验需要断层或研究区约束时，再把
    对应 shapefile 路径填回 config.ini。
    """
    raw_path = (path_value or '').split(';')[0].strip()
    if raw_path.lower() in ('', 'none', 'null', 'skip', 'false', '0'):
        logging.info(f"{file_type} 未配置，跳过该输入")
        return None

    verified_path = verify_file_path(raw_path, file_type)
    if verified_path is None:
        logging.warning(f"{file_type} 无法读取，跳过该输入: {raw_path}")
        return None

    return verified_path

def create_uniform_erosion_field(shape, base_k_sp, border_width=2):
    """
    创建无断层约束时使用的均一侵蚀系数场。

    边界仍设为 0，以满足 Fastscape 固定边界和 verify_erosion_field 的检查。
    """
    row, col = shape
    ksp = np.ones((row, col), dtype=np.float64) * base_k_sp
    safe_border = min(border_width, max(row // 2, 0), max(col // 2, 0))
    if safe_border > 0:
        ksp[:safe_border, :] = 0
        ksp[-safe_border:, :] = 0
        ksp[:, :safe_border] = 0
        ksp[:, -safe_border:] = 0
    return ksp

def run_main_workflow(config: configparser.ConfigParser, context: RunContext) -> dict[str, Any]:
    """Run the real-DEM inversion workflow in a structured run directory."""
    try:
        # 1. 首先设置基础日志配置（在任何其他操作之前）
        setup_basic_logging()  # 这是新添加的基础日志配置
        logging.info("程序开始执行")

        logging.info("配置文件加载完成")

        random_seed = _config_int(
            config,
            'Optimization',
            'random_seed',
            fallback=42,
            aliases=(('GeneticAlgorithm', 'random_seed'),),
        )
        np.random.seed(random_seed)
        logging.info(f"随机种子已固定: {random_seed}")

        output_path = context.root
        figures_dir = context.figures_dir
        arrays_dir = context.arrays_dir
        metrics_dir = context.metrics_dir

        try:
            os.makedirs(output_path, exist_ok=True)
            logging.info(f"创建输出目录成功: {output_path}")

            # 4. 更新日志配置以包含文件输出
            setup_file_logging(context.logs_dir, "main.log")
            context.add_artifact(context.logs_dir / "main.log")

        except Exception as e:
            logging.error(f"创建输出目录失败: {e}", exc_info=True)
            return

        # 设置日志
        #setup_logging(output_path)
        logging.info("开始优化过程")

        # 验证输入文件路径。只有 DEM 是必需项；断层和研究区 shapefile 可选。
        terrain_path = verify_file_path(config['Paths']['terrain_path'], '地形栅格文件')
        fault_shp_path = resolve_optional_file_path(
            config['Paths'].get('fault_shp_path', ''),
            '断层 Shapefile'
        )
        study_area_shp_path = resolve_optional_file_path(
            config['Paths'].get('study_area_shp_path', ''),
            '研究区域 Shapefile'
        )

        if terrain_path is None:
            raise UserConfigError("地形栅格文件无效，无法继续。请检查 [Data] terrain_path 是否指向存在的 DEM 文件。")
        config['Paths']['terrain_path'] = terrain_path
        config['Paths']['fault_shp_path'] = fault_shp_path or ''
        config['Paths']['study_area_shp_path'] = study_area_shp_path or ''

        # 检查并统一投影坐标系 (保持这部分)
        logging.info("Step 2: 检查和统一投影坐标系")
        target_crs = config['Preprocessing']['target_crs'] # 从配置文件读取 target_crs
        config = reproject_files_to_geographic(config, target_crs=target_crs) # 传递 target_crs
        terrain_path = config['Paths']['terrain_path']
        fault_shp_path = config['Paths'].get('fault_shp_path', '').strip() or None
        study_area_shp_path = config['Paths'].get('study_area_shp_path', '').strip() or None

        # 1. 数据加载。研究区 shapefile 存在时裁剪 DEM；留空时使用 DEM 全域。
        logging.info("Step 1: 数据加载")
        ratio = config.getfloat('Preprocessing', 'ratio')
        dem_data, dem_profile = load_dem_data(
            file_path=terrain_path,
            study_area_shp_path=study_area_shp_path,
            ratio=ratio
        )

        if study_area_shp_path:
            study_area = read_shapefile(study_area_shp_path)
            logging.info(f"研究区 Shapefile 已加载，要素数量: {len(study_area)}")
            rotation_angle = calculate_shp_rotation_angle(study_area_shp_path)
        else:
            study_area = None
            rotation_angle = 0.0
            logging.info("未配置研究区 Shapefile，使用 DEM 全域且不旋转")

        if fault_shp_path:
            fault_lines = read_shapefile(fault_shp_path)
            logging.info(f"断层 Shapefile 已加载，要素数量: {len(fault_lines)}")
        else:
            fault_lines = None
            logging.info("未配置断层 Shapefile，将使用均一侵蚀系数场")

        print(f"Calculated rotation angle: {rotation_angle:.2f}°")


        # 2. 创建侵蚀系数场 (在 *原始未旋转* 空间创建)
        logging.info("Step 2: 创建侵蚀系数场")
        global row, col, ORIGINAL_SHAPE, LOW_RES_SHAPE, matrix
        global Ksp, D_DIFF, time_step_num, total_simulation_time
        global terrain_resolution, feature_smooth_radius

        row, col = dem_data.shape # 使用原始 dem_data 的 shape
        ORIGINAL_SHAPE = (row, col)
        scale_factor = config.getint('Preprocessing', 'scale_factor')
        LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)
        logging.info(f"Original shape: {ORIGINAL_SHAPE}")
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        k_sp_value = config.getfloat('Model', 'k_sp_value')
        ksp_fault = config.getfloat('Model', 'ksp_fault')

        # 创建 erosion field：有断层和研究区时叠加断层弱抗性带，否则使用均一场。
        if fault_shp_path and study_area_shp_path:
            Ksp = create_erosion_field(
                shape=ORIGINAL_SHAPE,
                base_k_sp=k_sp_value,
                fault_k_sp=ksp_fault,
                fault_shp_path=fault_shp_path,
                study_area_shp_path=study_area_shp_path,
                rotation_angle=rotation_angle,
                border_width=2
            )
            logging.info("已根据断层 Shapefile 创建非均一侵蚀系数场")
        else:
            Ksp = create_uniform_erosion_field(
                shape=ORIGINAL_SHAPE,
                base_k_sp=k_sp_value,
                border_width=2
            )
            logging.info("已创建均一侵蚀系数场")

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
            rotated_Ksp = align_model_field(rotated_Ksp, rotated_dem_data.shape, label="Ksp", order=1)
            logging.info(f"After trimming:")
            logging.info(f"DEM shape: {rotated_dem_data.shape}")
            logging.info(f"Ksp shape: {rotated_Ksp.shape}")

        # 保存旋转后的 Ksp
        try:
            rotated_ksp_save_path = arrays_dir / 'rotated_ksp.npy'
            np.save(rotated_ksp_save_path, rotated_Ksp)
            context.add_artifact(rotated_ksp_save_path)
            logging.info(f"已保存旋转后的侵蚀系数场到: {rotated_ksp_save_path}")
        except Exception as e:
            logging.error(f"保存旋转后的侵蚀系数场失败: {e}")
        # 更新全局变量以反映旋转后的尺寸
        resampled_dem = rotated_dem_data
        row, col = resampled_dem.shape  # 更新为旋转后的尺寸
        ORIGINAL_SHAPE = (row, col)     # 更新为旋转后的尺寸
        LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)

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
            LOW_RES_SHAPE = validate_low_resolution_shape(ORIGINAL_SHAPE, scale_factor)
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


        LOW_RES_SHAPE = validate_low_resolution_shape((row, col), scale_factor) # 重新计算 LOW_RES_SHAPE
        logging.info(f"Resampled shape after rotation: {ORIGINAL_SHAPE}") #  注意这里 ORIGINAL_SHAPE 仍然是原始shape，应该输出 resampled_dem.shape 或 ORIGINAL_SHAPE = resampled_dem.shape
        logging.info(f"Low resolution shape: {LOW_RES_SHAPE}")


        D_DIFF = config.getfloat('Model', 'd_diff_value')
        time_step_num = 101  # 可以添加到config文件中
        total_simulation_time = config.getfloat('Model', 'time_total')
        terrain_resolution = spacing # 可以添加到config文件中
        feature_smooth_radius = 2  # 可以添加到config文件中
        use_lpips = config.getboolean('Fitness', 'use_lpips', fallback=True)
        pecube_time_steps = max(2, config.getint('Pecube', 'time_steps', fallback=2))

        ga_params = {
            'pop': _config_int(config, 'Optimization', 'population_size', fallback=6, aliases=(('GeneticAlgorithm', 'ga_pop_size'),)),
            'max_iter': _config_int(config, 'Optimization', 'max_iterations', fallback=5, aliases=(('GeneticAlgorithm', 'ga_max_iter'),)),
            'prob_cross': _config_float(config, 'Optimization', 'cross_probability', fallback=0.7, aliases=(('GeneticAlgorithm', 'ga_prob_cross'),)),
            'prob_mut': _config_float(config, 'Optimization', 'mutation_probability', fallback=0.05, aliases=(('GeneticAlgorithm', 'ga_prob_mut'),)),
            'lb': _config_float(config, 'Optimization', 'uplift_min', fallback=0.1, aliases=(('GeneticAlgorithm', 'lb'),)),
            'ub': _config_float(config, 'Optimization', 'uplift_max', fallback=1.0, aliases=(('GeneticAlgorithm', 'ub'),)),
            'uplift_precision': _config_float(config, 'Optimization', 'uplift_precision', fallback=0.1, aliases=(('GeneticAlgorithm', 'precision'),)),
            'decay_rate': _config_float(config, 'Optimization', 'decay_rate', fallback=1.0, aliases=(('GeneticAlgorithm', 'decay_rate'),)),
            'min_size_pop': _config_int(config, 'Optimization', 'min_population_size', fallback=4, aliases=(('GeneticAlgorithm', 'min_size_pop'),)),
            'patience': _config_int(config, 'Optimization', 'patience', fallback=3, aliases=(('GeneticAlgorithm', 'patience'),)),
            'random_seed': random_seed
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

        pecube_evaluator = PecubeFitnessEvaluator.from_config(
            config=config,
            context=context,
            target_dem=resampled_dem,
            ksp=rotated_Ksp,
            model_params=model_params,
        )
        spatial_mode = config.get("Pecube", "spatial_grid", fallback="auto").strip().lower()
        spatial_metrics = validate_rotation_spatial_constraints(
            rotation_angle=rotation_angle,
            pecube_enabled=pecube_evaluator.enabled,
            pecube_spatial_mode=spatial_mode,
        )
        context.metrics.update(spatial_metrics)
        if pecube_evaluator.enabled:
            if spatial_mode in {"auto", "dem", "dem_profile"}:
                spatial_adapter = pecube_spatial_adapter_from_dem_profile(dem_profile, resampled_dem.shape)
                if spatial_adapter is not None:
                    pecube_evaluator.apply_spatial_adapter(spatial_adapter)
                    spatial_grid = spatial_adapter.grid
                    logging.info(
                        "Pecube 空间网格已由 DEM 自动推导: "
                        f"lon0={spatial_grid.lon0}, lat0={spatial_grid.lat0}, "
                        f"dlon={spatial_grid.dlon}, dlat={spatial_grid.dlat}, crs={spatial_grid.crs}, "
                        f"shape={spatial_adapter.target_shape}, resample={spatial_adapter.resample}"
                    )
                else:
                    logging.warning("DEM 缺少 CRS/transform，Pecube 使用 config.ini 中的 lon0/lat0/dlon/dlat。")

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
            feature_smooth_radius=feature_smooth_radius,
            use_lpips=use_lpips,
            pecube_evaluator=pecube_evaluator,
            pecube_time_steps=pecube_time_steps,
        )

        # 显示原始DEM
        plt.figure(figsize=(15, 10))
        plot_single_data(dem_data, "Original DEM", cmap='terrain', origin='upper') # 显示 *原始* DEM
        figure_path = context.figure_path('original_dem.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        # 显示旋转后的DEM
        plt.figure(figsize=(15, 10))
        plot_single_data(rotated_dem_data, "Rotated DEM", cmap='terrain', origin='upper') # 显示 *旋转后* DEM
        figure_path = context.figure_path('rotated_dem.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        # 显示侵蚀系数场
        display_erosion_field(rotated_Ksp, shape=ORIGINAL_SHAPE) #  显示 *旋转后* 的 Ksp
        figure_path = context.figure_path('erosion_field.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        #叠加显示DEM和侵蚀系数场
        plt.figure(figsize=(15, 10))
        plt.imshow(rotated_dem_data, cmap='terrain', origin='upper')
        plt.imshow(rotated_Ksp, cmap='RdBu_r', alpha=0.5, origin='upper')
        plt.title("Rotated DEM with Erosion Coefficient Field")
        figure_path = context.figure_path('dem_with_erosion_field.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
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
        figure_path = context.figure_path('dem_rotation_comparison.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
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
        figure_path = context.figure_path('ksp_rotation_comparison.png')
        plt.savefig(figure_path)
        context.add_artifact(figure_path)
        plt.close()

        # 5. 遗传算法优化
        logging.info("Step 4: 遗传算法优化")
        n_jobs = _config_int(config, 'Optimization', 'n_jobs', fallback=1, aliases=(('GeneticAlgorithm', 'n_jobs'),))

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
                figure_path = context.figure_path('optimization_history.png')
                fig_history.savefig(figure_path)
                context.add_artifact(figure_path)
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
                data1=best_low_res_uplift,
                data2=best_full_res_uplift,
                title1='Best Low Resolution Uplift',
                title2='Best Full Resolution Uplift',
                value1='Uplift Rate (mm/yr)',
                value2='Uplift Rate (mm/yr)',
                cmap='RdBu_r'
            )
            figure_path = context.figure_path('uplift_comparison.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            true_uplift_path = os.path.join(
                os.path.dirname(config['Paths']['terrain_path']),
                'demo_true_uplift.npy'
            )
            demo_metrics = {}
            if os.path.exists(true_uplift_path):
                try:
                    true_uplift = np.load(true_uplift_path)
                    if true_uplift.shape == best_full_res_uplift.shape:
                        uplift_pearson = pearsonr(
                            true_uplift.ravel(),
                            best_full_res_uplift.ravel()
                        ).statistic
                        uplift_spearman = spearmanr(
                            true_uplift.ravel(),
                            best_full_res_uplift.ravel()
                        ).statistic
                        uplift_rmse = float(np.sqrt(np.mean((true_uplift - best_full_res_uplift) ** 2)))
                        demo_metrics.update({
                            'uplift_pearson': float(uplift_pearson),
                            'uplift_spearman': float(uplift_spearman),
                            'uplift_rmse': uplift_rmse
                        })
                        logging.info(
                            "Demo uplift metrics: "
                            f"Pearson={uplift_pearson:.4f}, "
                            f"Spearman={uplift_spearman:.4f}, RMSE={uplift_rmse:.4f}"
                        )
                        plot_comparison(
                            data1=true_uplift,
                            data2=best_full_res_uplift,
                            title1='Demo True Uplift',
                            title2='Inverted Uplift',
                            value1='Uplift Rate (mm/yr)',
                            value2='Uplift Rate (mm/yr)',
                            cmap='RdBu_r'
                        )
                        figure_path = context.figure_path('demo_true_vs_inverted_uplift.png')
                        plt.savefig(figure_path)
                        context.add_artifact(figure_path)
                        plt.close()
                    else:
                        logging.warning(
                            f"跳过 demo 真值 uplift 对比：形状不一致 "
                            f"{true_uplift.shape} vs {best_full_res_uplift.shape}"
                        )
                except Exception as e:
                    logging.warning(f"读取 demo 真值 uplift 失败，跳过对比: {e}")

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
            terrain_pearson = pearsonr(resampled_dem.ravel(), final_elevation.ravel()).statistic
            terrain_spearman = spearmanr(resampled_dem.ravel(), final_elevation.ravel()).statistic
            terrain_rmse = float(np.sqrt(np.mean((resampled_dem - final_elevation) ** 2)))
            demo_metrics.update({
                'terrain_pearson': float(terrain_pearson),
                'terrain_spearman': float(terrain_spearman),
                'terrain_rmse': terrain_rmse,
                'terrain_loss': float(best_fitness) if not pecube_evaluator.enabled else float(pecube_evaluator.best_result.terrain_loss if pecube_evaluator.best_result else best_fitness),
                'total_loss': float(best_fitness)
            })
            logging.info(
                "Terrain metrics: "
                f"Pearson={terrain_pearson:.4f}, "
                f"Spearman={terrain_spearman:.4f}, RMSE={terrain_rmse:.4f}"
            )

            metrics_txt_path = metrics_dir / 'demo_metrics.txt'
            with open(metrics_txt_path, 'w') as metrics_file:
                for key, value in demo_metrics.items():
                    metrics_file.write(f"{key} = {value:.6f}\n")
            context.add_artifact(metrics_txt_path)
            pecube_metrics = pecube_evaluator.save_best_outputs(
                generated_dem=final_elevation,
                uplift=best_full_res_uplift,
            )
            demo_metrics.update(pecube_metrics)
            write_metrics(context, "main_metrics.json", {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in demo_metrics.items()})

            plot_comparison(
                data1=final_elevation,
                data2=resampled_dem, #  注意这里对比的是 *旋转后且重采样* 的 DEM (resampled_dem)
                title1='Generated Terrain',
                title2='Target Landscape',
                value1='Elevation (m)',
                value2='Elevation (m)',
                cmap='terrain',
                shared_scale=False
            )
            figure_path = context.figure_path('terrain_comparison.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            plot_comparison(
                data1=final_elevation,
                data2=resampled_dem,
                title1='Generated Terrain',
                title2='Target Landscape',
                value1='Elevation (m)',
                value2='Elevation (m)',
                cmap='terrain',
                shared_scale=True
            )
            figure_path = context.figure_path('terrain_comparison_shared_scale.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            # 绘制隆升分布图
            plot_uplift_distribution_x(best_full_res_uplift)
            figure_path = context.figure_path('uplift_distribution_x.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            plot_uplift_distribution_y(best_full_res_uplift)
            figure_path = context.figure_path('uplift_distribution_y.png')
            plt.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close()

            # 绘制3D地形可视化
            fig_3d = plot_3d_surface(
                data=final_elevation,
                uplift=best_full_res_uplift,
                title="3D Terrain with Uplift Field"
            )
            figure_path = context.figure_path('3d_terrain.png')
            fig_3d.savefig(figure_path)
            context.add_artifact(figure_path)
            plt.close(fig_3d)
            logging.info("3D地形可视化已保存")

            # 保存结果
            results = {
                    'best_full_res_uplift': best_full_res_uplift,
                    'final_elevation': final_elevation,
                    'fitness_history': np.array(fitness_history) if fitness_history is not None else None
                }
            save_optimization_results(arrays_dir, results, context)
            # 8. 保存参数配置
            config_file = metrics_dir / 'parameters.txt'
            with open(config_file, 'w') as f:
                f.write("Model Parameters:\n")
                for section in config.sections():
                    f.write(f"\n[{section}]\n")
                    for key, value in config[section].items():
                        f.write(f"{key} = {value}\n")

                f.write("\nOptimization Results:\n")
                f.write(f"Best Fitness: {best_fitness}\n")
                f.write(f"Optimization Time: {end_time - start_time:.2f} seconds\n")
            context.add_artifact(config_file)

            logging.info(f"所有结果已保存到: {output_path}")
            logging.info("优化过程成功完成")
            logging.info(f"Results saved to: {output_path}")
            return {
                "best_fitness": float(best_fitness),
                "optimization_time_seconds": float(end_time - start_time),
                "original_shape": list(ORIGINAL_SHAPE),
                "low_res_shape": list(LOW_RES_SHAPE),
                **{k: _json_metric_value(v) for k, v in demo_metrics.items()},
            }

        else:
            logging.warning("遗传算法未能找到有效解。")
            raise RuntimeError("遗传算法未能找到有效解。请检查 DEM、K、隆升率范围和 GA 参数。")

    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        logging.exception("Exception details:")
        raise

def main():
    from ga_lem_inverter.runner import main as runner_main
    runner_main(default_mode="main")


if __name__ == "__main__":
    main()
