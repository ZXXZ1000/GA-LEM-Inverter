# spatial_processor.py
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import from_origin
import logging
from typing import Tuple, Optional, Dict, Any

class SpatialProcessor:
    """空间数据预处理器，处理坐标系统统一和空间转换"""
    
    def __init__(self, target_crs: str = 'EPSG:4326'):  # WGS 84
        """
        初始化空间处理器
        
        参数:
        - target_crs: 目标坐标系统，默认为 WGS 84
        """
        self.target_crs = target_crs
        self.transform = None
        self.bounds = None
        self.resolution = None
    
    def process_dem(self, dem_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        处理DEM数据：读取、重投影(如需)并获取空间参考信息
        """
        try:
            with rasterio.open(dem_path) as src:
                # 检查源数据CRS
                src_crs = src.crs
                
                if src_crs != self.target_crs:
                    logging.info(f"Reprojecting DEM from {src_crs} to {self.target_crs}")
                    
                    # 计算重投影变换参数
                    transform, width, height = calculate_default_transform(
                        src_crs, self.target_crs,
                        src.width, src.height,
                        *src.bounds
                    )
                    
                    # 准备重投影后的数据
                    dem_data = np.zeros((height, width), dtype=src.dtypes[0])
                    
                    # 执行重投影
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dem_data,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=self.target_crs,
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                else:
                    dem_data = src.read(1)
                    transform = src.transform
                
                # 存储空间参考信息
                self.transform = transform
                self.bounds = src.bounds
                self.resolution = (transform[0], -transform[4])
                
                # 创建元数据字典
                profile = {
                    'crs': self.target_crs,
                    'transform': transform,
                    'bounds': self.bounds,
                    'resolution': self.resolution,
                    'width': dem_data.shape[1],
                    'height': dem_data.shape[0]
                }
                
                return dem_data.astype(np.float32), profile
                
        except Exception as e:
            logging.error(f"Error processing DEM: {e}")
            raise
    
    def process_vector(self, shp_path: str) -> gpd.GeoDataFrame:
        """
        处理矢量数据：读取并投影转换
        """
        try:
            # 读取shapefile
            gdf = gpd.read_file(shp_path)
            
            # 检查并转换坐标系
            if gdf.crs != self.target_crs:
                logging.info(f"Reprojecting vector from {gdf.crs} to {self.target_crs}")
                gdf = gdf.to_crs(self.target_crs)
            
            return gdf
            
        except Exception as e:
            logging.error(f"Error processing vector data: {e}")
            raise
    
    def align_vector_to_raster(self, gdf: gpd.GeoDataFrame, 
                             matrix_shape: Tuple[int, int]) -> np.ndarray:
        """
        将矢量数据对齐到栅格空间
        """
        try:
            if self.transform is None:
                raise ValueError("No raster reference available. Process DEM first.")
                
            # 栅格化
            from rasterio.features import rasterize
            
            shapes = [(geom, 1) for geom in gdf.geometry]
            raster = rasterize(
                shapes,
                out_shape=matrix_shape,
                transform=self.transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            
            return raster
            
        except Exception as e:
            logging.error(f"Error aligning vector to raster: {e}")
            raise