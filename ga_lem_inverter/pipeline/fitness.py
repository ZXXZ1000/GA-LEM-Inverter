# fitness_evaluator.py
import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from skimage.feature import hog
from scipy import ndimage
import torch
import lpips
import logging

_FEATURE_CACHE = {}
_LPIPS_LOSS_FN = None

def terrain_mae(matrix1, matrix2):
    """计算两个地形矩阵之间的平均绝对误差（Mean Absolute Error, MAE）"""
    return np.mean(np.abs(matrix1 - matrix2))

def terrain_ssim(matrix1, matrix2):
    """计算两个地形矩阵之间的结构相似性指数（Structural Similarity Index, SSIM）"""
    data_range = max(float(matrix1.max() - matrix1.min()), 1e-12)
    ssim_index = ssim(matrix1, matrix2, data_range=data_range)
    return _unit_interval(ssim_index)

def terrain_correlation(matrix1, matrix2):
    """计算两个地形矩阵之间的相关系数"""
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()
    if np.var(flat1) <= 1e-12 or np.var(flat2) <= 1e-12:
        return 1.0 if np.allclose(flat1, flat2) else 0.0
    correlation, _ = stats.pearsonr(flat1, flat2)
    if not np.isfinite(correlation):
        return 0.0
    return _unit_interval((correlation + 1.0) / 2.0)

def hypscurve(dem, bins=60):
    """
    计算DEM的高程曲线（Hypsometric curve）

    参数:
    dem : numpy array
        数字高程模型
    bins : int, 可选
        直方图的箱数

    返回:
    tuple
        相对频率和对应的高程值
    """
    # 移除NaN值并展平数组
    dem = dem[~np.isnan(dem)].flatten()

    if bins is not None:
        # 如果指定了箱数，使用直方图方法
        n, elev = np.histogram(dem, bins=bins)
        n = np.flipud(n)
        elev = np.flipud((elev[:-1] + elev[1:]) / 2)  # 使用箱的中心值
        n = np.cumsum(n)
    else:
        # 如果没有指定箱数，使用排序方法
        elev = np.sort(dem)[::-1]
        n = np.arange(1, len(elev) + 1)

    # 计算相对频率
    n = n / n[-1] * 100
    return n, elev

def localtopography(dem, radius, cellsize, type='range'):
    """
    计算局部地形指标

    参数:
    dem : numpy array
        数字高程模型
    radius : float
        局部计算的半径（地图单位）
    cellsize : float
        DEM的单元格大小
    type : str
        局部地形指标的类型

    返回:
    numpy array
        局部地形指标
    """
    # 将半径转换为像素数
    radius_px = int(np.ceil(radius / cellsize))
    # 创建结构元素
    se = ndimage.generate_binary_structure(2, 1)
    se = ndimage.iterate_structure(se, radius_px)

    if type == 'range':
        # 计算局部高程范围
        local_max = ndimage.maximum_filter(dem, footprint=se)
        local_min = ndimage.minimum_filter(dem, footprint=se)
        return local_max - local_min
    elif type == 'max':
        # 计算局部最大值
        return ndimage.maximum_filter(dem, footprint=se)
    elif type == 'min':
        # 计算局部最小值
        return ndimage.minimum_filter(dem, footprint=se)
    elif type == 'mean':
        # 计算局部平均值
        return ndimage.uniform_filter(dem, size=2*radius_px+1)
    elif type == 'median':
        # 计算局部中值
        return ndimage.median_filter(dem, size=2*radius_px+1)
    elif type == 'std':
        # 计算局部标准差
        return ndimage.generic_filter(dem, np.std, size=2*radius_px+1)

def roughness(dem, type='srf', kernel_size=(3, 3)):
    """
    计算各种粗糙度指标

    参数:
    dem : numpy array
        数字高程模型
    type : str
        粗糙度指标的类型
    kernel_size : tuple
        计算用的核大小

    返回:
    numpy array
        粗糙度指标
    """
    dem = dem.astype(float)

    if type == 'tpi':
        # 地形位置指数 (Topographic Position Index)
        kernel = np.ones(kernel_size)
        kernel[kernel_size[0]//2, kernel_size[1]//2] = 0
        num_cells = np.sum(kernel)
        mean = ndimage.convolve(dem, kernel) / num_cells
        return dem - mean
    elif type == 'tri':
        # 地形粗糙度指数 (Terrain Ruggedness Index)
        kernel = np.ones(kernel_size)
        kernel[kernel_size[0]//2, kernel_size[1]//2] = 0
        num_cells = np.sum(kernel)
        mean = ndimage.convolve(dem, kernel) / num_cells
        return np.abs(dem - mean)
    elif type in ['roughness', 'ruggedness']:
        # 粗糙度（使用局部标准差）
        return ndimage.generic_filter(dem, np.std, size=kernel_size)
    elif type == 'srf':
        # 表面粗糙度因子 (Surface Roughness Factor)
        gy, gx = np.gradient(dem)
        normal_x = -gx
        normal_y = -gy
        normal_z = np.ones_like(dem)
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm
        kernel = np.ones(kernel_size)
        nx = ndimage.convolve(normal_x, kernel)**2
        ny = ndimage.convolve(normal_y, kernel)**2
        nz = ndimage.convolve(normal_z, kernel)**2
        return np.sqrt(nx + ny + nz) / np.prod(kernel_size)

def extract_terrain_features(dem, resolution=347.4, smooth_radius=3):
    """
    提取地形特征，包括高程曲线、局部地形和粗糙度

    参数:
    dem : numpy array
        数字高程模型
    resolution : float
        DEM分辨率
    smooth_radius : int
        平滑半径

    返回:
    dict
        地形特征字典
    """
    # 对DEM进行平滑处理
    smoothed_dem = gaussian_filter(dem, sigma=smooth_radius)

    # 计算高程曲线
    hyps_n, hyps_elev = hypscurve(smoothed_dem, bins=60)

    # 计算局部地形
    local_topo = localtopography(smoothed_dem, radius=smooth_radius*resolution, cellsize=resolution)

    # 计算粗糙度
    rough = roughness(smoothed_dem, type='srf')

    # 计算HOG特征
    hog_features = hog(smoothed_dem, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=False)

    return {
        'elevation': smoothed_dem,
        'hypsometric': hyps_n,
        'local_topography': local_topo,
        'roughness': rough,
        'hog': hog_features
    }

def _array_cache_key(array, resolution, smooth_radius):
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return (
        arr.shape,
        str(arr.dtype),
        float(resolution),
        int(smooth_radius),
        hash(arr.tobytes()),
    )


def get_cached_terrain_features(dem, resolution=347.4, smooth_radius=3):
    """进程级缓存固定 DEM 特征，避免 GA 每个候选解重复提取目标 DEM 特征。"""
    key = _array_cache_key(dem, resolution, smooth_radius)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = extract_terrain_features(dem, resolution, smooth_radius)
    return _FEATURE_CACHE[key]


def get_lpips_loss_fn():
    """进程级缓存 LPIPS 模型；多进程时每个 worker 只加载一次。"""
    global _LPIPS_LOSS_FN
    if _LPIPS_LOSS_FN is None:
        _LPIPS_LOSS_FN = lpips.LPIPS(net='alex', verbose=False)
    return _LPIPS_LOSS_FN


def compare_features(feature1, feature2):
    """
    比较两个地形特征

    参数:
    feature1, feature2 : numpy arrays 或 tuples
        要比较的特征

    返回:
    float
        相似度分数
    """
    if isinstance(feature1, tuple):  # 对于高程曲线
        variance_sum = np.var(feature1[0]) + np.var(feature2[0])
        if variance_sum <= 1e-12:
            return 1.0 if np.allclose(feature1[0], feature2[0]) else 0.0
        return _unit_interval(1 - mean_squared_error(feature1[0], feature2[0]) / variance_sum)
    variance_sum = np.var(feature1) + np.var(feature2)
    if variance_sum <= 1e-12:
        return 1.0 if np.allclose(feature1, feature2) else 0.0
    return _unit_interval(1 - mean_squared_error(feature1, feature2) / variance_sum)


def _unit_interval(value):
    """Return a finite metric value in the 0-1 interval."""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))

def terrain_hash(dem, bits=256):
    """
    计算地形矩阵的高精度哈希值

    参数:
    dem : numpy array
        数字高程模型
    bits : int
        哈希值的位数，默认为256位以保持高精度

    返回:
    np.array
        地形的二进制哈希值
    """
    # 计算平均高程
    avg_height = np.mean(dem)

    # 生成二进制哈希
    hash_bits = dem.flatten() > avg_height

    # 如果需要，调整哈希长度到指定的位数
    if len(hash_bits) > bits:
        # 如果原始矩阵太大，进行降采样
        indices = np.linspace(0, len(hash_bits) - 1, bits, dtype=int)
        hash_bits = hash_bits[indices]
    elif len(hash_bits) < bits:
        # 如果原始矩阵太小，进行插值
        hash_bits = np.interp(np.linspace(0, len(hash_bits) - 1, bits),
                              np.arange(len(hash_bits)), hash_bits)
        hash_bits = hash_bits > 0.5

    return hash_bits

def hamming_distance(hash1, hash2):
    """
    计算两个哈希值之间的汉明距离

    参数:
    hash1, hash2 : np.array
        要比较的哈希值

    返回:
    int
        汉明距离
    """
    return np.sum(hash1 != hash2)

def terrain_hash_similarity(matrix1, matrix2, bits=256):
    """
    基于高精度哈希计算两个地形矩阵的相似度

    参数:
    matrix1, matrix2 : numpy arrays
        要比较的地形矩阵
    bits : int
        哈希值的位数

    返回:
    float
        相似度分数（0到1之间，1表示完全相同）
    """
    hash1 = terrain_hash(matrix1, bits)
    hash2 = terrain_hash(matrix2, bits)
    max_distance = bits
    similarity = 1 - hamming_distance(hash1, hash2) / max_distance
    return similarity

def terrain_similarity(matrix1, matrix2, resolution=347.4, smooth_radius=3, use_lpips=True):
    """
    基于多个特征和指标计算地形相似度。

    参数:
    - matrix1: 实际地形高程数据。
    - matrix2: 模型生成的地形高程数据。
    - resolution: DEM 分辨率。
    - smooth_radius: 平滑半径。

    返回:
    - total_sim: 综合相似度评分。
    """
    logging.debug(f"terrain_similarity: matrix1 type={type(matrix1)}, shape={matrix1.shape if hasattr(matrix1, 'shape') else 'None'}, matrix2 type={type(matrix2)}, shape={matrix2.shape if hasattr(matrix2, 'shape') else 'None'}")
    try:
        matrix1 = np.asarray(matrix1, dtype=np.float64)
        matrix2 = np.asarray(matrix2, dtype=np.float64)

        def normalize_relief(matrix):
            """去掉绝对海拔基准，比较相对地形形态。"""
            finite = matrix[np.isfinite(matrix)]
            if finite.size == 0:
                raise ValueError("地形矩阵没有有效值")
            relief = matrix - np.nanmin(finite)
            relief_range = np.nanmax(relief) - np.nanmin(relief)
            if relief_range <= 1e-12:
                return np.zeros_like(relief)
            return relief / relief_range

        norm_matrix1 = normalize_relief(matrix1)
        norm_matrix2 = normalize_relief(matrix2)

        # 提取两个 DEM 的相对地形特征，避免不同海拔基准或固定边界把适应度带偏。
        features1 = get_cached_terrain_features(norm_matrix1, resolution, smooth_radius)
        features2 = extract_terrain_features(norm_matrix2, resolution, smooth_radius)

        # 计算地形特征的相似度
        elevation_sim = compare_features(features1['elevation'], features2['elevation'])
        hypsometric_sim = compare_features(features1['hypsometric'], features2['hypsometric'])
        local_topo_sim = compare_features(features1['local_topography'], features2['local_topography'])
        roughness_sim = compare_features(features1['roughness'], features2['roughness'])
        terrien_sim = (elevation_sim + hypsometric_sim + local_topo_sim ) / 3


        # 计算其他相似性指标
        hog_sim = compare_features(features1['hog'], features2['hog'])
        mae = terrain_mae(norm_matrix1, norm_matrix2)
        ssim_score = terrain_ssim(norm_matrix1, norm_matrix2)
        correlation = terrain_correlation(norm_matrix1, norm_matrix2)
        hash_sim = terrain_hash_similarity(norm_matrix1, norm_matrix2)

        # 归一化MAE
        max_possible_mae = max(np.max(norm_matrix1) - np.min(norm_matrix1), 1e-12)
        normalized_mae = _unit_interval(1 - (mae / max_possible_mae))

        if use_lpips:
            def terrain_perceptual_distance(matrix1, matrix2):
                """添加警告过滤"""
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    # 归一化到[-1, 1]范围
                    range1 = matrix1.max() - matrix1.min()
                    range2 = matrix2.max() - matrix2.min()
                    if range1 <= 1e-12 or range2 <= 1e-12:
                        return 0.0 if np.allclose(matrix1, matrix2) else 1.0
                    matrix1 = (matrix1 - matrix1.min()) / range1 * 2 - 1
                    matrix2 = (matrix2 - matrix2.min()) / range2 * 2 - 1

                    # 转换为PyTorch张量并添加通道维度
                    tensor1 = torch.from_numpy(matrix1).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                    tensor2 = torch.from_numpy(matrix2).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

                    # 使用缓存的LPIPS模型
                    with torch.no_grad():
                        distance = get_lpips_loss_fn()(tensor1, tensor2)

                    return distance.item()

            terrain_similarity_DL = _unit_interval(1 - terrain_perceptual_distance(norm_matrix1, norm_matrix2))
        else:
            terrain_similarity_DL = terrien_sim

        #总相似度
        total_sim = (
            0.25 * terrien_sim +
            0.25 * terrain_similarity_DL +
            0.20 * normalized_mae +
            0.20 * correlation +
            0.10 * ssim_score
        )
        #total_sim = terrien_sim * terrain_similarity_DL
        return _unit_interval(total_sim)
    except Exception as e:
        logging.error(f"计算地形相似度出错: {e}")
        raise RuntimeError(f"计算地形相似度出错: {e}")
