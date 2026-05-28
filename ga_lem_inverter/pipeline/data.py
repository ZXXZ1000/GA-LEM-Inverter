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
import os
import logging
import rasterio
import rasterio.mask
import geopandas as gpd
from rasterio.features import shapes as rasterio_shapes
from math import ceil
from typing import Tuple, Optional
import logging
import os
import numpy as np
from scipy.ndimage import rotate
from ga_lem_inverter.pipeline.spatial import SpatialProcessor
import configparser
from pathlib import Path
from rasterio import warp
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def _profile_x_axis_unit(profile: dict) -> np.ndarray:
    """Return the positive source-raster x-axis direction in map coordinates."""
    transform = profile["transform"]
    if not isinstance(transform, Affine):
        transform = Affine(*tuple(transform)[:6])
    x_axis = np.array([transform.a, transform.d], dtype=float)
    norm = float(np.linalg.norm(x_axis))
    if norm <= 0:
        return np.array([1.0, 0.0], dtype=float)
    x_axis = x_axis / norm
    if x_axis[0] < 0 or (abs(x_axis[0]) < 1e-12 and x_axis[1] < 0):
        x_axis = -x_axis
    return x_axis



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
                dem_array = src.read(1, masked=True).astype(np.float32).filled(np.nan)
                profile = src.profile.copy()
                profile.update({'path': file_path, 'dtype': 'float32', 'nodata': np.nan})
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
                    abs(old_transform.e) / ratio   # 新的像素高度，from_origin 会自动设置负 y 方向
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


def _rotation_angle_from_axis(x_unit: np.ndarray) -> float:
    """返回把 x_unit 旋到正东方向所需的角度，正值表示顺时针。"""
    angle_deg = float(np.degrees(np.arctan2(x_unit[1], x_unit[0])))
    if angle_deg < 0:
        angle_deg += 180.0
    if angle_deg <= 90.0:
        return angle_deg
    return -(180.0 - angle_deg)


def _rotated_profile_from_geometry(
    profile: dict,
    geometry,
    spacing: float,
    *,
    label: str,
    near_square_ratio: float = 1.05,
    x_axis_strategy: str = "longest",
) -> tuple[dict, float]:
    """根据几何 footprint 的最小旋转矩形生成带真实 transform 的旋转栅格 profile。"""
    if "transform" not in profile or profile.get("crs") is None:
        raise ValueError("DEM profile 缺少 transform 或 CRS，无法建立旋转后的空间参考。")
    if spacing <= 0:
        raise ValueError(f"旋转栅格 spacing 必须大于 0，当前为 {spacing}。")
    if geometry.is_empty:
        raise ValueError(f"{label} 为空，无法建立旋转栅格。")

    crs = profile.get("crs")
    rectangle = geometry.minimum_rotated_rectangle
    coords = np.asarray(rectangle.exterior.coords[:-1], dtype=float)
    if coords.shape[0] < 4:
        raise ValueError(f"{label} 最小外接矩形无效，无法建立旋转栅格。")

    edges = np.roll(coords, -1, axis=0) - coords
    lengths = np.linalg.norm(edges, axis=1)
    valid_edges = lengths > 0
    if not np.any(valid_edges):
        raise ValueError(f"{label} 最小外接矩形边长无效，无法建立旋转栅格。")
    min_length = float(np.min(lengths[valid_edges]))
    max_length = float(np.max(lengths[valid_edges]))
    if min_length > 0 and max_length / min_length < near_square_ratio:
        logging.info("%s footprint 近似正方形，跳过主方向旋转。", label)
        return profile.copy(), 0.0

    if x_axis_strategy == "source_x":
        source_x_unit = _profile_x_axis_unit(profile)
        candidate_indices = np.where(valid_edges)[0]
        candidate_units = np.asarray([edges[idx] / lengths[idx] for idx in candidate_indices])
        scores = np.abs(candidate_units @ source_x_unit)
        selected_index = int(candidate_indices[int(np.argmax(scores))])
        x_unit = edges[selected_index] / lengths[selected_index]
        if float(np.dot(x_unit, source_x_unit)) < 0:
            x_unit = -x_unit
    elif x_axis_strategy == "longest":
        longest_index = int(np.argmax(lengths))
        x_unit = edges[longest_index] / lengths[longest_index]
        if x_unit[0] < 0 or (abs(x_unit[0]) < 1e-12 and x_unit[1] < 0):
            x_unit = -x_unit
    else:
        raise ValueError(f"未知旋转轴选择策略: {x_axis_strategy}")
    y_unit = np.array([-x_unit[1], x_unit[0]], dtype=float)

    x_projection = coords @ x_unit
    y_projection = coords @ y_unit
    min_x, max_x = float(np.min(x_projection)), float(np.max(x_projection))
    min_y, max_y = float(np.min(y_projection)), float(np.max(y_projection))
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} 旋转栅格宽高无效，无法继续。")

    cols = max(2, int(ceil(width / spacing)))
    rows = max(2, int(ceil(height / spacing)))
    top_left = x_unit * min_x + y_unit * max_y

    rotated_transform = Affine(
        x_unit[0] * spacing,
        -y_unit[0] * spacing,
        top_left[0],
        x_unit[1] * spacing,
        -y_unit[1] * spacing,
        top_left[1],
    )

    rotated_profile = profile.copy()
    rotated_profile.update(
        {
            "height": rows,
            "width": cols,
            "transform": rotated_transform,
            "crs": crs,
            "dtype": "float32",
            "nodata": np.nan,
        }
    )
    return rotated_profile, _rotation_angle_from_axis(x_unit)


def build_rotated_profile_from_study_area(
    profile: dict,
    study_area_shp_path: str,
    spacing: float,
) -> dict:
    """
    根据研究区最小外接旋转矩形建立一个带真实 transform 的旋转栅格 profile。

    这个函数用于替代单纯的 ndarray 旋转：输出网格仍在 DEM 的 CRS 中，只是像素
    x/y 轴沿研究区主方向排列。后续 Pecube 从该 profile 推导样品坐标时，样品点
    和旋转后的 DEM/Ksp 会处在同一套空间参考下。
    """
    if "transform" not in profile or profile.get("crs") is None:
        raise ValueError("DEM profile 缺少 transform 或 CRS，无法建立旋转后的空间参考。")

    crs = profile.get("crs")
    study_area = read_shapefile(study_area_shp_path)
    if study_area.crs is not None and study_area.crs != crs:
        study_area = study_area.to_crs(crs)

    geometry = study_area.geometry.union_all() if hasattr(study_area.geometry, "union_all") else study_area.geometry.unary_union
    rotated_profile, _ = _rotated_profile_from_geometry(profile, geometry, spacing, label="研究区 Shapefile")
    return rotated_profile


def build_rotated_profile_from_dem_footprint(
    profile: dict,
    dem_array: np.ndarray,
    spacing: float,
    *,
    angle_threshold_degrees: float = 0.25,
) -> tuple[dict, float]:
    """
    没有研究区 Shapefile 时，根据 DEM 自身有效数据 footprint 推断旋转网格。

    对已经裁切好的 DEM，用户通常不会再提供 study_area_shp_path。此时如果 DEM 是
    经纬度矩形投影到 UTM，目标 raster 的外接网格会保持正北，但有效 footprint 会
    带一个小角度；这里用有效像元范围恢复这个角度，避免后续把斜 footprint 当作
    已经转正的模型域。

    和 study-area shapefile 不同，这里不会强制把最长边转成 X 轴。用户直接给一个
    已裁切 DEM 时，Y 方向长于 X 方向是合法的模型域形状；强制最长边横放会造成
    不必要的 90 度旋转。
    """
    if "transform" not in profile or profile.get("crs") is None:
        return profile.copy(), 0.0
    if spacing <= 0:
        raise ValueError(f"旋转栅格 spacing 必须大于 0，当前为 {spacing}。")

    transform = profile["transform"]
    if not isinstance(transform, Affine):
        transform = Affine(*tuple(transform)[:6])

    finite_mask = np.isfinite(dem_array)
    if np.any(finite_mask):
        mask_image = finite_mask.astype(np.uint8)
        geometries = [
            shape(geometry)
            for geometry, value in rasterio_shapes(mask_image, mask=finite_mask, transform=transform)
            if int(value) == 1
        ]
        geometry = unary_union(geometries) if geometries else None
    else:
        geometry = None

    if geometry is None or geometry.is_empty:
        height = int(profile.get("height", dem_array.shape[0]))
        width = int(profile.get("width", dem_array.shape[1]))
        corners = [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
        ]
        geometry = Polygon(corners)

    rotated_profile, rotation_angle = _rotated_profile_from_geometry(
        profile,
        geometry,
        spacing,
        label="DEM",
        near_square_ratio=1.001,
        x_axis_strategy="source_x",
    )
    if abs(rotation_angle) < angle_threshold_degrees:
        logging.info(
            "DEM footprint inferred rotation %.4f° below threshold %.4f°; skip rotation.",
            rotation_angle,
            angle_threshold_degrees,
        )
        return profile.copy(), 0.0
    return rotated_profile, rotation_angle


def reproject_array_to_profile(
    array: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """把数组从 src_profile 的空间参考重采样到 dst_profile。"""
    if "transform" not in src_profile or src_profile.get("crs") is None:
        raise ValueError("源 profile 缺少 transform 或 CRS，无法重采样。")
    if "transform" not in dst_profile or dst_profile.get("crs") is None:
        raise ValueError("目标 profile 缺少 transform 或 CRS，无法重采样。")

    src_transform = src_profile["transform"]
    dst_transform = dst_profile["transform"]
    if not isinstance(src_transform, Affine):
        src_transform = Affine(*tuple(src_transform)[:6])
    if not isinstance(dst_transform, Affine):
        dst_transform = Affine(*tuple(dst_transform)[:6])

    destination = np.full(
        (int(dst_profile["height"]), int(dst_profile["width"])),
        np.nan,
        dtype=np.float32,
    )
    reproject(
        source=np.asarray(array, dtype=np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_profile["crs"],
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


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

            # 正方形或近正方形研究区没有稳定的“长轴”方向。
            # demo_study_area 就是这种情况，强行取某一条边会得到任意的 90° 旋转。
            # 这里直接返回 0°，避免默认 demo 做无意义旋转。
            min_length = np.min(lengths)
            max_length = np.max(lengths)
            if min_length > 0 and max_length / min_length < 1.05:
                logging.info("研究区近似正方形，跳过主方向旋转")
                return 0.0

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


def _reprojection_output_path(paths_section, file_path: str, config_key: str) -> str:
    """把重投影中间文件写到配置输出目录，避免依赖输入文件所在目录可写。"""
    output_root = str(paths_section.get("output_path", "")).strip()
    if output_root.lower() in {"", "none", "null", "skip", "false", "0"}:
        output_root = str(Path.cwd() / "outputs")

    output_dir = Path(output_root).expanduser() / "reprojected_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(file_path)
    return str(output_dir / f"{source.stem}_{config_key}_reprojected{source.suffix}")


def _normalize_reprojected_raster_profile(profile: dict) -> dict:
    """清理源 GeoTIFF profile 中与新写入布局不兼容的 GDAL 参数。"""
    normalized = profile.copy()
    tiled = str(normalized.get("tiled", "")).strip().lower() in {"yes", "true", "1"}
    if not tiled:
        normalized.pop("blockxsize", None)
        normalized.pop("blockysize", None)
    normalized.pop("compress", None)
    normalized.pop("interleave", None)
    return normalized


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
                        output_path = _reprojection_output_path(paths_section, file_path, config_key)

                        # 计算目标坐标系的 transform, width, height
                        dst_crs = target_crs
                        dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                            src.crs, dst_crs, src.width, src.height, *src.bounds)

                        # 更新 profile
                        profile = _normalize_reprojected_raster_profile(src.profile)
                        profile.update({
                            'crs': dst_crs,
                            'transform': dst_transform,
                            'width': dst_width,
                            'height': dst_height,
                            'dtype': 'float32',
                            'nodata': np.nan,
                        })

                        with rasterio.open(output_path, 'w', **profile) as dst:
                            for i in range(1, src.count + 1):
                                source = src.read(i, masked=True).astype(np.float32).filled(np.nan)
                                destination = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                                warp.reproject(
                                    source=source,
                                    destination=destination,
                                    src_transform=src.transform,
                                    src_crs=current_crs,
                                    src_nodata=np.nan,
                                    dst_transform=dst_transform, # 使用计算出的 dst_transform
                                    dst_crs=dst_crs,
                                    dst_nodata=np.nan,
                                    init_dest_nodata=True,
                                    resampling=warp.Resampling.bilinear)
                                dst.write(destination, i)
                        config['Paths'][config_key] = output_path # 更新配置文件中的路径
                        logging.info(f"已保存重新投影的文件到 {output_path}")
                    else:
                        logging.info(f"栅格文件 {file_path} 已经是目标坐标系或无需重新投影。")

            elif file_ext == '.shp':
                gdf = gpd.read_file(file_path)
                current_crs = gdf.crs
                if current_crs and current_crs != target_crs:
                    logging.info(f"重新投影Shapefile {file_path} 从 {current_crs} 到 {target_crs}")
                    output_path = _reprojection_output_path(paths_section, file_path, config_key)
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
