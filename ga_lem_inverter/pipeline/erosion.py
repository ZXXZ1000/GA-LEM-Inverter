# erosion_field.py
import numpy as np
import geopandas as gpd
import logging
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import shapely.affinity  # 用于几何旋转
from ga_lem_inverter.pipeline.data import read_shapefile
from scipy.ndimage import rotate
from affine import Affine
from ga_lem_inverter.pipeline.data import rotate_data

def clip_and_rasterize(
    line_shp_path: str,
    polygon_shp_path: str,
    matrix_shape: Tuple[int, int],
    rotation_angle: float = 0,
    raster_transform: Optional[Affine] = None,
    raster_crs: Optional[Union[str, object]] = None,
):
    """
    裁切和栅格化断层线
    """
    try:
        # 1. 读取要素
        polygon_gdf = read_shapefile(polygon_shp_path)
        line_gdf = read_shapefile(line_shp_path)

        if polygon_gdf.empty or line_gdf.empty:
            logging.warning("输入shapefile为空")
            return np.zeros(matrix_shape, dtype=np.uint8), None

        # 2. 统一坐标系。优先对齐到 DEM 栅格 CRS，这样断层/研究区/DEM 共用同一个 transform。
        target_crs = raster_crs or polygon_gdf.crs or line_gdf.crs
        if target_crs is not None:
            if polygon_gdf.crs != target_crs:
                polygon_gdf = polygon_gdf.to_crs(target_crs)
            if line_gdf.crs != target_crs:
                line_gdf = line_gdf.to_crs(target_crs)
            logging.info(f"研究区和断层已投影至栅格坐标系: {target_crs}")
        elif polygon_gdf.crs != line_gdf.crs:
            polygon_gdf = polygon_gdf.to_crs(line_gdf.crs)
            logging.info(f"研究区已投影至断层坐标系: {line_gdf.crs}")

        # 3. 裁切断层线
        clipped_lines = gpd.clip(line_gdf, polygon_gdf)
        if clipped_lines.empty:
            logging.warning("断层线与研究区无交集")
            return np.zeros(matrix_shape, dtype=np.uint8), None

        # 4. 使用 DEM transform 栅格化；没有传入时才退回旧的研究区 bounds transform。
        if raster_transform is not None:
            transform = raster_transform
        else:
            bounds = polygon_gdf.total_bounds
            transform = from_origin(
                bounds[0],  # minx
                bounds[3],  # maxy
                (bounds[2] - bounds[0]) / matrix_shape[1],  # pixel width
                (bounds[3] - bounds[1]) / matrix_shape[0]   # pixel height
            )

        # 5. 栅格化断层线
        fault_raster = rasterize(
            [(geom, 1) for geom in clipped_lines.geometry],
            out_shape=matrix_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        # 6. 栅格化研究区范围作为掩膜
        mask = rasterize(
            [(geom, 1) for geom in polygon_gdf.geometry],
            out_shape=matrix_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        # 7. 应用掩膜并处理研究区外的值。
        # 如果使用 DEM transform 栅格化，输出必须保留 DEM 原始 shape，
        # 不能再按研究区 NaN 边缘裁剪，否则 Ksp 与 DEM 像素坐标会错位。
        final_raster = fault_raster if raster_transform is not None else np.where(mask == 1, fault_raster, np.nan)

        return final_raster, transform

    except Exception as e:
        logging.error(f"栅格化过程出错: {e}")
        return np.zeros(matrix_shape, dtype=np.uint8), None


def create_erosion_field(shape: Tuple[int, int],
                        base_k_sp: float,
                        fault_k_sp: float,
                        fault_shp_path: str,
                        study_area_shp_path: str,
                        rotation_angle: float = 0,
                        border_width: int = 2,
                        raster_transform: Optional[Affine] = None,
                        raster_crs: Optional[Union[str, object]] = None) -> np.ndarray:
    """
    4. 创建基础侵蚀系数场并添加断层弱抗性带
    """
    try:
        row, col = shape

        # 创建基础侵蚀系数场
        Ksp = np.ones((row, col)) * base_k_sp

        # 添加已旋转和栅格化的断层弱抗性带
        line_raster, transform = clip_and_rasterize(
            fault_shp_path,
            study_area_shp_path,
            matrix_shape=(row, col),
            rotation_angle=rotation_angle,
            raster_transform=raster_transform,
            raster_crs=raster_crs,
        )

        Ksp += line_raster * fault_k_sp
        # 旋转
        if rotation_angle != 0:
            Ksp = rotate_data(Ksp, rotation_angle)
        # 旧路径没有 DEM transform，只能按研究区有效范围裁边；DEM transform 路径
        # 必须保留原始 DEM shape，后续和 DEM 一起旋转/重采样。
        if raster_transform is None:
            Ksp = trim_nan_edges(Ksp)
        # 设置边界条件
        Ksp[:border_width, :] = 0  # 下边界
        Ksp[-border_width:, :] = 0  # 上边界
        Ksp[:, :border_width] = 0  # 左边界
        Ksp[:, -border_width:] = 0  # 右边界

        return Ksp

    except Exception as e:
        logging.error(f"创建侵蚀系数场失败: {e}")
        raise RuntimeError(f"创建侵蚀系数场失败: {e}")

def display_erosion_field(Ksp: np.ndarray,
                         shape: Optional[Tuple[int, int]] = None,
                         title: str = "Erosion Coefficient Field",
                         cmap: str = 'RdBu_r') -> None:
    """
    显示侵蚀系数场。

    参数:
    - Ksp: 侵蚀系数场矩阵（可以是展平的一维数组）
    - shape: 如果Ksp是展平的，提供原始形状 (rows, cols)
    - title: 图的标题
    - cmap: 颜色图配置
    """
    try:
        if shape is not None and Ksp.ndim == 1:
            Ksp = Ksp.reshape(shape)

        plt.figure(figsize=(10, 8), dpi=300)
        im = plt.imshow(Ksp, cmap=cmap, origin='upper')
        plt.colorbar(im, label='Erosion Coefficient')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    except Exception as e:
        logging.error(f"显示侵蚀系数场失败: {e}")
        raise RuntimeError(f"显示侵蚀系数场失败: {e}")

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

def verify_erosion_field(Ksp: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    验证侵蚀系数场的有效性。

    参数:
    - Ksp: 侵蚀系数场矩阵
    - shape: 如果Ksp是展平的，提供原始形状

    返回:
    - bool: 是否有效
    """
    try:
        if shape is not None and Ksp.ndim == 1:
            Ksp = Ksp.reshape(shape)

        # 验证没有负值
        if np.any(Ksp < 0):
            logging.warning("侵蚀系数场包含负值")
            return False

        # 验证边界条件
        if not np.all(Ksp[0, :] == 0) or not np.all(Ksp[-1, :] == 0) or \
           not np.all(Ksp[:, 0] == 0) or not np.all(Ksp[:, -1] == 0):
            logging.warning("边界条件不满足要求")
            return False

        return True

    except Exception as e:
        logging.error(f"验证侵蚀系数场失败: {e}")
        return False
