# data_loader.py
import rasterio
import numpy as np
import logging
from rasterio.errors import RasterioIOError
from scipy.ndimage import rotate
import geopandas as gpd
import os
import rasterio
import geopandas as gpd
from shapely.geometry import box
import os
import logging
import rasterio
import rasterio.mask
import geopandas as gpd
from typing import Tuple, Optional
import logging
import os
import numpy as np
from scipy.ndimage import rotate
from spatial_processor import SpatialProcessor
import configparser
from rasterio import warp



def clip_dem_with_shapefile(dem_data: np.ndarray, dem_profile: dict, 
                          shp_path: str) -> Tuple[np.ndarray, dict]:
    """
    使用 shapefile 裁切 DEM 数据，保持输入分辨率。

    参数:
    - dem_data: 原始DEM数组
    - dem_profile: DEM元数据
    - shp_path: Shapefile路径

    返回:
    - clipped_dem: 裁切后的DEM数组
    - clipped_profile: 裁切后的元数据
    """
    try:
        # 确保数据类型为浮点型
        dem_data = dem_data.astype(np.float32)
        
        # 读取shapefile并转换为相同的坐标系统
        with rasterio.open(dem_profile['path']) as src:
            mask = gpd.read_file(shp_path)
            if mask.crs != src.crs:
                mask = mask.to_crs(src.crs)

            # 计算新的变换矩阵，保持像素大小
            out_transform = rasterio.transform.from_origin(
                src.bounds.left,
                src.bounds.top,
                src.res[0] * (src.width / dem_data.shape[1]),  # 保持与输入数据相同的分辨率
                src.res[1] * (src.height / dem_data.shape[0])
            )

            # 创建临时rasterio数据集
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=dem_data.shape[0],
                    width=dem_data.shape[1],
                    count=1,
                    dtype=dem_data.dtype,
                    crs=src.crs,
                    transform=out_transform,
                ) as temp_dataset:
                    temp_dataset.write(dem_data, 1)
                    
                    # 执行裁切
                    out_image, out_transform = rasterio.mask.mask(
                        temp_dataset, 
                        mask.geometry, 
                        crop=True,
                        nodata=np.nan
                    )
                    out_image = out_image[0]  # 获取第一个波段

            # 更新profile
            clipped_profile = src.profile.copy()
            clipped_profile.update({
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform,
                "dtype": 'float32'
            })

            logging.info(f"Clipped DEM shape: {out_image.shape}")
            logging.info(f"Pixel size maintained at: {out_transform[0]}, {abs(out_transform[4])}")

            return out_image, clipped_profile

    except Exception as e:
        logging.error(f"Clipping DEM with shapefile failed: {e}")
        return None, None

def load_dem_data(file_path: str, 
                 study_area_shp_path: Optional[str] = None,
                 ratio: float = None) -> Tuple[np.ndarray, dict]:
    """
    加载DEM数据，先进行缩放，再进行裁切。
    
    参数:
    - file_path: DEM文件路径（.tif或.npy）
    - study_area_shp_path: 研究区shapefile路径（可选）
    - ratio: 缩放比例(0-1之间)，None表示不缩放

    返回:
    - dem_array: 地形高程数据的numpy数组
    - profile: 栅格文件的元数据
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 1. 加载数据
        if file_ext == '.npy':
            logging.info(f"Loading NPY file: {file_path}")
            dem_array = np.load(file_path)
            dem_array = dem_array.astype(np.float32)
            profile = {'path': file_path}
            
        elif file_ext in ['.tif', '.tiff']:
            logging.info(f"Loading TIFF file: {file_path}")
            with rasterio.open(file_path) as src:
                dem_array = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                profile.update({'path': file_path})
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # 2. 重采样缩放
        if ratio is not None:
            if not 0 < ratio <= 1:
                raise ValueError(f"缩放比例必须在0-1之间，当前值: {ratio}")
                
            logging.info(f"Resizing DEM with ratio: {ratio}")
            original_shape = dem_array.shape
            new_shape = (int(original_shape[0] * ratio), 
                        int(original_shape[1] * ratio))
            
            from skimage.transform import resize
            dem_array = resize(dem_array, 
                             new_shape,
                             order=1,  # 双线性插值
                             mode='edge',
                             anti_aliasing=True,
                             preserve_range=True)
            
            # 更新profile中的尺寸信息
            if 'transform' in profile:
                from rasterio.transform import from_origin
                old_transform = profile['transform']
                new_transform = from_origin(
                    old_transform.c,  # 原点x
                    old_transform.f,  # 原点y
                    old_transform.a / ratio,  # 新的像素宽度
                    old_transform.e / ratio   # 新的像素高度
                )
                profile.update({
                    'height': new_shape[0],
                    'width': new_shape[1],
                    'transform': new_transform
                })
            
            logging.info(f"DEM resized from {original_shape} to {new_shape}")

        # 3. 研究区裁切
        if study_area_shp_path and os.path.exists(study_area_shp_path):
            logging.info(f"Clipping DEM with shapefile: {study_area_shp_path}")
            dem_array, profile = clip_dem_with_shapefile(dem_array, profile, 
                                                       study_area_shp_path)
            if dem_array is None:
                raise ValueError("DEM clipping failed")
            


        return dem_array, profile

    except Exception as e:
        logging.error(f"Error loading DEM data: {e}")
        raise


def rotate_data(data: np.ndarray, angle: float, 
               target_bounds: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
    """
    旋转数据数组，对连续值和离散值使用不同的插值方法
    """
    try:
        # 垂直翻转
        data = np.flipud(data)
        
        # 检查数据是否为离散值（比如断层线栅格）
        unique_values = np.unique(data[~np.isnan(data)])
        is_discrete = len(unique_values) <= 2  # 假设二值图像为离散数据
        
        # 选择插值阶数
        # order=0: 最近邻插值，适合离散值
        # order=1: 双线性插值，适合连续值
        interp_order = 0 if is_discrete else 1
        
        # 旋转数组
        rotated_data = rotate(data, angle, reshape=True, 
                            order=interp_order,  # 根据数据类型选择插值方法
                            mode='constant', 
                            cval=np.nan,
                            prefilter=False)  # 对离散值禁用预滤波
        
        # 裁剪处理...（保持不变）
        if target_bounds is None:
            # 创建掩码
            mask = ~np.isnan(rotated_data)
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0:
                row_slice = slice(rows.min(), rows.max() + 1)
                col_slice = slice(cols.min(), cols.max() + 1)
                rotated_data = rotated_data[row_slice, col_slice]
        else:
            rotated_data = rotated_data[target_bounds[0], target_bounds[1]]
        
        # 垂直翻转回来
        rotated_data = np.flipud(rotated_data)
        return rotated_data
        
    except Exception as e:
        logging.error(f"Error rotating data: {e}")
        raise

    
def read_terrain_data(tiff_path, rotation_angle=0):
    """
    读取地形栅格数据，并进行旋转。

    参数:
    - tiff_path: 地形栅格文件的路径。
    - rotation_angle: 旋转角度，正值为逆时针旋转 (可选，默认为 0)。

    返回:
    - dem_array: 旋转后的地形高程数据的 numpy 数组。
    - profile: rasterio 的 profile 对象，包含栅格文件的元数据。
    """
    try:
        with rasterio.open(tiff_path) as src:
            # 读取第一个波段数据并转换为浮点型
            dem_array = src.read(1).astype(np.float32)
            # 垂直反转（根据需要保留或移除）
            dem_array = np.flipud(dem_array)

            # 将 nodata 值替换为 NaN
            nodata_value = src.nodata if src.nodata is not None else -32768
            dem_array[dem_array == nodata_value] = np.nan

            # 创建有效数据的掩码（True 表示有效数据）
            mask = ~np.isnan(dem_array)

            if not np.any(mask):
                logging.warning("未找到有效数据。")
                return None, None

            # 旋转数组
            if rotation_angle != 0:
                rotated_dem = rotate(dem_array, rotation_angle, reshape=True, order=1, mode='constant', cval=np.nan)
                rotated_mask = rotate(mask.astype(np.float32), rotation_angle, reshape=True, order=0, mode='constant', cval=0) > 0.5

                # 找到旋转后有效数据的边界
                rotated_rows, rotated_cols = np.where(rotated_mask)
                if len(rotated_rows) == 0 or len(rotated_cols) == 0:
                    logging.warning("旋转后未找到有效数据。")
                    return None, None

                min_r, max_r = rotated_rows.min(), rotated_rows.max()
                min_c, max_c = rotated_cols.min(), rotated_cols.max()

                # 裁剪到旋转后有效数据的最小边界矩形
                dem_array = rotated_dem[min_r:max_r+1, min_c:max_c+1]
            
            profile = src.profile
            return dem_array, profile

    except RasterioIOError as e:
        logging.error(f"无法读取文件 {tiff_path}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        return None, None



def read_shapefile(shp_path, encoding='utf-8', tried_encodings=None):
    """读取 Shapefile 数据，尝试不同的编码"""
    if tried_encodings is None:
        tried_encodings = ['utf-8', 'gbk', 'gb18030', 'latin1'] # 扩展编码列表
    
    current_encoding = tried_encodings[0]
    remaining_encodings = tried_encodings[1:]

    try:
        gdf = gpd.read_file(shp_path, encoding=current_encoding)
        logging.info(f"Shapefile successfully read with encoding: {current_encoding}")
        return gdf
    except UnicodeDecodeError as e:
        logging.warning(f"使用 {current_encoding} 编码读取 Shapefile 失败: {e}")
        if remaining_encodings:
            logging.info(f"尝试使用编码: {remaining_encodings[0]}")
            return read_shapefile(shp_path, encoding=remaining_encodings[0], tried_encodings=remaining_encodings) # 递归调用，尝试下一个编码
        else:
            logging.error(f"尝试所有编码 (utf-8, gbk, gb18030, latin1) 读取 Shapefile 均失败: {e}")
            raise ValueError(f"无法读取 Shapefile 数据，编码错误: {e}")
    except Exception as e:
        logging.error(f"无法读取 Shapefile 数据: {e}")
        raise ValueError(f"无法读取 Shapefile 数据: {e}")


    
def calculate_shp_rotation_angle(shp_path: str) -> float:
    """
    使用最小外接矩形(MBR)计算研究区的主方向角度。
    
    角度规则：
    - 当倾角<90度时进行顺时针旋转（返回正值）
    - 当倾角>90度时进行逆时针旋转
    
    参数:
    - shp_path: 研究区shapefile路径
    
    返回:
    - angle: 需要旋转的角度（度数）
    """
    try:
        # 读取shapefile
        gdf = gpd.read_file(shp_path)
        
        if len(gdf) > 0:
            # 获取第一个多边形要素
            polygon = gdf.geometry.iloc[0]
            
            # 获取最小外接矩形
            mbr = polygon.minimum_rotated_rectangle
            coords = np.array(mbr.exterior.coords)[:-1]
            
            # 计算各边的长度和方向
            edges = np.diff(coords, axis=0)
            lengths = np.sqrt(np.sum(edges**2, axis=1))
            
            # 找出最长边
            longest_edge = edges[np.argmax(lengths)]
            
            # 计算最长边与正东方向的夹角（0-180度）
            angle = np.arctan2(longest_edge[1], longest_edge[0])
            angle_deg = np.degrees(angle)
            
            # 确保角度为正值（0-180度）
            if angle_deg < 0:
                angle_deg += 180
                
            # 根据角度确定旋转方向和大小
            if angle_deg <= 90:
                # 顺时针旋转到水平
                rotation_angle = angle_deg
            else:
                # 逆时针旋转到水平
                rotation_angle = -(180 - angle_deg)
            
            # 输出详细信息
            print(f"\nStudy Area Rotation Analysis:")
            print(f"MBR long edge angle: {angle_deg:.2f}°")
            print(f"Required rotation: {rotation_angle:.2f}°")
            print(f"Rotation direction: {'Clockwise' if rotation_angle > 0 else 'Counter-clockwise'}")
            
            logging.info(f"Study area rotation analysis:")
            logging.info(f"MBR long edge angle: {angle_deg:.2f}°")
            logging.info(f"Required rotation: {rotation_angle:.2f}°")
            logging.info(f"Rotation direction: {'Clockwise' if rotation_angle > 0 else 'Counter-clockwise'}")
            
            return rotation_angle
            
        logging.warning("No valid polygon found in shapefile")
        return 0.0
        
    except Exception as e:
        logging.error(f"Error calculating shapefile rotation angle: {e}")
        logging.exception("Exception details:")
        return 0.0
    

def rotate_and_crop_raster(raster_data: np.ndarray, rotation_angle: float, 
                          fill_value: Optional[float] = np.nan) -> np.ndarray:
    """
    旋转栅格数据并裁剪到有效区域。
    
    参数:
    - raster_data: 输入栅格数据
    - rotation_angle: 旋转角度（度数）
    - fill_value: 填充值，默认为NaN

    返回:
    - rotated_data: 旋转并裁剪后的栅格数据
    """
    try:
        # 将栅格数据转换为浮点型
        raster_data = raster_data.astype(np.float32)
        
        # 旋转数据
        rotated_data = rotate(raster_data, rotation_angle, 
                            reshape=True, order=1, 
                            mode='constant', cval=fill_value)
        
        # 创建有效数据掩码
        if np.isnan(fill_value):
            mask = ~np.isnan(rotated_data)
        else:
            mask = (rotated_data != fill_value)
            
        # 裁剪到有效区域
        rows, cols = np.where(mask)
        if len(rows) > 0 and len(cols) > 0:
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            rotated_data = rotated_data[min_r:max_r+1, min_c:max_c+1]
        
        return rotated_data
        
    except Exception as e:
        logging.error(f"Error rotating raster data: {e}")
        return raster_data
    


def load_and_process_data(dem_path: str, 
                         study_area_path: str,
                         fault_path: str,
                         target_crs: str = 'EPSG:32648'):
    """
    加载并处理所有输入数据，确保坐标系统一致
    """
    try:
        # 初始化空间处理器
        processor = SpatialProcessor(target_crs=target_crs)
        
        # 处理DEM
        dem_data, profile = processor.process_dem(dem_path)
        
        # 处理研究区边界
        study_area = processor.process_vector(study_area_path)
        
        # 处理断层数据
        faults = processor.process_vector(fault_path)
        
        # 将断层对齐到DEM栅格
        fault_raster = processor.align_vector_to_raster(
            faults, 
            matrix_shape=dem_data.shape
        )
        
        return dem_data, fault_raster, profile
        
    except Exception as e:
        logging.error(f"Error in data loading and processing: {e}")
        raise


def reproject_files_to_geographic(config: configparser.ConfigParser, target_crs: str) -> configparser.ConfigParser:
    """
    检查并重新投影输入文件到指定的**目标坐标系**，如果需要的话。

    参数:
    - config: 配置对象
    - target_crs: 目标坐标系 EPSG 代码字符串 (例如 'EPSG:32648', 'EPSG:4326')，从配置文件中读取.

    返回:
    - config: 更新后的配置对象，如果文件被重新投影
    """
    logging.info(f"开始检查和重新投影文件到 {target_crs}")
    paths_section = config['Paths']
    files_to_reproject = {
        'terrain_path': paths_section['terrain_path'],
        'fault_shp_path': paths_section['fault_shp_path'],
        'study_area_shp_path': paths_section['study_area_shp_path'],
    }

    for config_key, file_path in files_to_reproject.items():
        if not file_path or not os.path.exists(file_path):
            logging.warning(f"文件路径无效，跳过 {config_key}: {file_path}")
            continue

        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in ['.tif', '.tiff']:
                with rasterio.open(file_path) as src:
                    current_crs = src.crs
                    if current_crs and current_crs != target_crs:
                        logging.info(f"重新投影栅格文件 {file_path} 从 {current_crs.to_string()} 到 {target_crs}")
                        output_path = os.path.splitext(file_path)[0] + "_reprojected.tif"

                        # 计算目标坐标系的 transform, width, height
                        dst_crs = target_crs
                        dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                            src.crs, dst_crs, src.width, src.height, *src.bounds)

                        # 更新 profile
                        profile = src.profile.copy()
                        profile.update({
                            'crs': dst_crs,
                            'transform': dst_transform,
                            'width': dst_width,
                            'height': dst_height
                        })

                        with rasterio.open(output_path, 'w', **profile) as dst:
                            for i in range(1, src.count + 1):
                                warp.reproject(
                                    source=rasterio.band(src, i),
                                    destination=rasterio.band(dst, i),
                                    src_transform=src.transform,
                                    src_crs=current_crs,
                                    dst_transform=dst_transform, # 使用计算出的 dst_transform
                                    dst_crs=dst_crs,
                                    resampling=warp.Resampling.bilinear)
                        config['Paths'][config_key] = output_path # 更新配置文件中的路径
                        logging.info(f"已保存重新投影的文件到 {output_path}")
                    else:
                        logging.info(f"栅格文件 {file_path} 已经是目标坐标系或无需重新投影。")

            elif file_ext == '.shp':
                gdf = gpd.read_file(file_path)
                current_crs = gdf.crs
                if current_crs and current_crs != target_crs:
                    logging.info(f"重新投影Shapefile {file_path} 从 {current_crs} 到 {target_crs}")
                    output_path = os.path.splitext(file_path)[0] + "_reprojected.shp"
                    gdf_reprojected = gdf.to_crs(target_crs)
                    gdf_reprojected.to_file(output_path, encoding='utf-8') # 明确指定UTF-8编码
                    config['Paths'][config_key] = output_path # 更新配置文件路径
                    logging.info(f"已保存重新投影的Shapefile到 {output_path}")
                else:
                    logging.info(f"Shapefile {file_path} 已经是目标坐标系或无需重新投影。")

            elif file_ext == '.npy':
                logging.info(f"跳过 NPY 文件 {file_path} 的投影检查。") # NPY文件不包含投影信息
                continue

            else:
                logging.warning(f"不支持的文件格式，跳过投影检查 {file_path}")

        except Exception as e:
            logging.error(f"处理文件 {file_path} 投影时出错: {e}")
            logging.exception("异常详情:")

    logging.info("文件投影检查和重新投影完成。")
    return config

def get_valid_bounds(data: np.ndarray) -> Tuple[slice, slice]:
    """
    获取数组中非NaN值的有效边界
    
    参数:
    - data: 输入数组
    
    返回:
    - Tuple[slice, slice]: 行和列的切片范围
    """
    mask = ~np.isnan(data)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return slice(None), slice(None)
        
    return (slice(row_indices[0], row_indices[-1] + 1),
            slice(col_indices[0], col_indices[-1] + 1))