# data_preprocessing.py
import numpy as np
from skimage.transform import resize, rotate
from scipy.ndimage import gaussian_filter, median_filter
import cv2
import logging
from scipy.interpolate import griddata
from skimage.transform import AffineTransform, warp


def preprocess_terrain_data(dem_array, dem_profile, resample_ratio=None, resample_method='bilinear', fill_nodata=True, smooth_sigma=2, rotation_angle=0):
    """
    预处理地形数据，包括平滑、重采样、旋转和处理 NoData 值。

    参数:
    - dem_array: 地形高程数据的 numpy 数组。
    - dem_profile: rasterio 的 profile 对象，包含栅格文件的元数据。
    - resample_ratio: 缩放比例 (0-1 之间)。如果为 None，则不进行基于比例的重采样。
    - resample_method: 重采样方法，可选 'nearest'、'bilinear'、'cubic'， 默认为 'bilinear'
    - fill_nodata: 是否填充 NoData 值。
    - smooth_sigma: 高斯平滑的 sigma 值。如果为 None，则不进行平滑。
    - rotation_angle: 旋转角度，单位为度。

    返回:
    - matrix: 预处理后的地形高程数据。
    - spacing: 重采样后的像素间距 (米/像素)。
    """
    if dem_array is None:
        raise ValueError("输入矩阵不能为空。")
    if not isinstance(dem_array, np.ndarray):
        raise TypeError("输入必须是 NumPy 数组。")
    if dem_array.ndim != 2:
        raise ValueError(f"输入矩阵必须是二维的，但得到的是 {dem_array.ndim} 维。")

    # 数据类型转换，确保为浮点型
    matrix = dem_array.astype(np.float32)

    # 填充 NaN 值，使用有效值的平均值
    nan_mask = np.isnan(matrix)
    if np.any(nan_mask):
        valid_values = matrix[~nan_mask]
        if valid_values.size == 0:
            raise ValueError("矩阵中没有有效值可用于填充 NaN。")
        fill_value = np.mean(valid_values)
        matrix[nan_mask] = fill_value

    # 高斯平滑
    if smooth_sigma:
        matrix = gaussian_filter(matrix, sigma=smooth_sigma)



    # 基于比例的重采样
    if resample_ratio:
        if not 0 < resample_ratio <= 1:
            raise ValueError("缩放比例必须在 0 到 1 之间。")

        # 计算目标尺寸
        input_shape = matrix.shape
        output_shape = (int(input_shape[0] * resample_ratio), int(input_shape[1] * resample_ratio))

        # 选择重采样顺序
        order = {
            'nearest': 0,
            'bilinear': 1,
            'cubic': 3
        }.get(resample_method, 1)

        # 执行重采样
        resampled_matrix = resize(
            matrix,
            output_shape,
            order=order,
            mode='edge',
            anti_aliasing=True,
            preserve_range=True
        )
        matrix = resampled_matrix
        print(f"原始形状: {input_shape}")
        print(f"重采样后形状: {matrix.shape}")

        # 计算重采样后的像素间距
        xres, yres = dem_profile['transform'][0], dem_profile['transform'][4]
        width = abs(xres) * dem_profile['width']
        height = abs(yres) * dem_profile['height']
        spacing_x = width / output_shape[1]
        spacing_y = height / output_shape[0]
        spacing = (spacing_x + spacing_y) / 2  # 取平均值
    else:
        # 如果没有重采样，则使用原始像素间距
        xres, yres = dem_profile['transform'][0], dem_profile['transform'][4]
        width = abs(xres) * dem_profile['width']
        height = abs(yres) * dem_profile['height']
        spacing_x = width / dem_profile['width']
        spacing_y = height / dem_profile['height']
        spacing = (spacing_x + spacing_y) / 2  # 取平均值

    return matrix, spacing

def interpolate_uplift_cv(low_res_uplift, target_shape):
    """
    使用Kriging方法将低分辨率uplift矩阵插值到高分辨率。

    参数:
    - low_res_uplift: 低分辨率的uplift矩阵
    - target_shape: 目标高分辨率矩阵的形状 (height, width)

    返回:
    - high_res_uplift: 高分辨率的uplift矩阵
    """
    try:
        from pykrige.ok import OrdinaryKriging
        logging.info(f"开始Kriging插值: {low_res_uplift.shape} -> {target_shape}")

        # 创建网格坐标
        low_res_shape = low_res_uplift.shape
        x = np.linspace(0, 1, low_res_shape[1])
        y = np.linspace(0, 1, low_res_shape[0])
        X, Y = np.meshgrid(x, y)
        z = low_res_uplift.flatten()
        x_flat, y_flat = X.flatten(), Y.flatten()

        # 创建目标网格
        new_x = np.linspace(0, 1, target_shape[1])
        new_y = np.linspace(0, 1, target_shape[0])

        # 执行Kriging插值
        ok = OrdinaryKriging(
            x_flat, y_flat, z,
            variogram_model='spherical',  # 球面模型，适用于中等范围的空间相关性
            enable_plotting=False,
            nlags=10,
            weight=True
        )
        
        z_interpolated, _ = ok.execute("grid", new_x, new_y)
        
        # 应用中值滤波平滑结果（可选）
        # z_interpolated = median_filter(z_interpolated, size=3)
        
        logging.info("Kriging插值完成")
        return z_interpolated

    except Exception as e:
        logging.error(f"Kriging插值过程出错: {e}")
        logging.warning("回退到使用OpenCV双三次插值...")
        return interpolate_uplift_cv(low_res_uplift, target_shape)
    
def interpolate_uplift_cv1(input_data, target_shape):
    """
    使用OpenCV的插值方法处理uplift数据。支持文件路径或数组输入。

    参数:
    - input_data: 输入数据，可以是文件路径(str)或numpy数组
    - target_shape: 目标形状 (height, width)

    返回:
    - high_res_uplift: 插值后的numpy数组
    """
    try:
        # 处理输入数据
        if isinstance(input_data, str):
            try:
                # 尝试从文件加载数据
                if input_data.endswith('.npy'):
                    low_res_uplift = np.load(input_data)
                else:
                    low_res_uplift = np.genfromtxt(input_data, delimiter=',')
            except Exception as e:
                logging.error(f"读取文件失败: {e}")
                return None
        else:
            low_res_uplift = input_data

        # 确保数据是numpy数组且类型正确
        low_res_uplift = np.array(low_res_uplift, dtype=np.float32)
        
        # 验证输入数据
        if low_res_uplift is None or low_res_uplift.size == 0:
            raise ValueError("无效的输入数据")
            
        # 尝试使用双三次插值
        try:
            high_res_uplift = cv2.resize(
                low_res_uplift,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
            return high_res_uplift
            
        except Exception as e:
            logging.warning(f"双三次插值失败，尝试双线性插值: {e}")
            # 回退到双线性插值
            return cv2.resize(
                low_res_uplift,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
    except Exception as e:
        logging.error(f"插值过程出错: {e}")
        raise RuntimeError(f"插值过程出错: {e}")  
    
def unify_array_sizes(array1: np.ndarray, array2: np.ndarray) :
    """
    将两个数组统一到相同尺寸，使用最邻近插值填充新增的行
    
    参数:
    - array1: 第一个输入数组
    - array2: 第二个输入数组
    
    返回:
    - Tuple[np.ndarray, np.ndarray]: 统一尺寸后的两个数组
    """
    from scipy.interpolate import NearestNDInterpolator
    
    # 获取目标尺寸
    target_rows = max(array1.shape[0], array2.shape[0])
    target_cols = max(array1.shape[1], array2.shape[1])
    
    result1, result2 = array1.copy(), array2.copy()
    
    # 调整第一个数组
    if array1.shape[0] < target_rows or array1.shape[1] < target_cols:
        # 获取原始数组的网格点和值
        y, x = np.mgrid[0:array1.shape[0], 0:array1.shape[1]]
        points = np.column_stack((y.flat, x.flat))
        values = array1.flat
        
        # 创建插值器
        interpolator = NearestNDInterpolator(points, values)
        
        # 创建目标网格
        y_new, x_new = np.mgrid[0:target_rows, 0:target_cols]
        new_points = np.column_stack((y_new.flat, x_new.flat))
        
        # 插值
        result1 = interpolator(new_points).reshape((target_rows, target_cols))
    
    # 调整第二个数组
    if array2.shape[0] < target_rows or array2.shape[1] < target_cols:
        # 获取原始数组的网格点和值
        y, x = np.mgrid[0:array2.shape[0], 0:array2.shape[1]]
        points = np.column_stack((y.flat, x.flat))
        values = array2.flat
        
        # 创建插值器
        interpolator = NearestNDInterpolator(points, values)
        
        # 创建目标网格
        y_new, x_new = np.mgrid[0:target_rows, 0:target_cols]
        new_points = np.column_stack((y_new.flat, x_new.flat))
        
        # 插值
        result2 = interpolator(new_points).reshape((target_rows, target_cols))
    
    return result1, result2