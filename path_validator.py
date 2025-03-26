# path_validator.py
import os
import logging
from typing import Dict, List, Optional

def clean_path(path: str) -> str:
    """清理路径字符串，移除多余的空格和注释"""
    # 移除注释（分号后的内容）
    path = path.split(';')[0]
    # 移除首尾空格
    return path.strip()

def verify_file_path(path: str, file_type: str) -> Optional[str]:
    """
    验证文件路径的有效性。

    参数:
    - path: 文件路径
    - file_type: 文件类型描述（用于日志）

    返回:
    - str: 清理后的有效路径，如果无效则返回None
    """
    try:
        cleaned_path = clean_path(path)
        if not cleaned_path:
            logging.error(f"{file_type} 路径为空")
            return None
            
        if not os.path.exists(cleaned_path):
            logging.error(f"{file_type} 文件不存在: {cleaned_path}")
            return None
            
        # 对于地形数据，检查文件扩展名
        if file_type == '地形栅格文件':
            ext = os.path.splitext(cleaned_path)[1].lower()
            if ext not in ['.tif', '.tiff', '.npy']:
                logging.error(f"不支持的地形数据格式: {ext}")
                return None
                
        return cleaned_path
        
    except Exception as e:
        logging.error(f"验证 {file_type} 路径时出错: {e}")
        return None

def verify_directory_path(path: str, create: bool = True) -> Optional[str]:
    """
    验证目录路径的有效性，可选择自动创建目录。

    参数:
    - path: 目录路径
    - create: 是否在目录不存在时创建它

    返回:
    - str: 清理后的有效路径，如果无效则返回None
    """
    try:
        cleaned_path = clean_path(path)
        if not cleaned_path:
            logging.error("目录路径为空")
            return None
            
        if not os.path.exists(cleaned_path):
            if create:
                os.makedirs(cleaned_path, exist_ok=True)
                logging.info(f"创建目录: {cleaned_path}")
            else:
                logging.error(f"目录不存在: {cleaned_path}")
                return None
                
        return cleaned_path
        
    except Exception as e:
        logging.error(f"验证目录路径时出错: {e}")
        return None

def verify_config_paths(config: Dict[str, Dict[str, str]]) -> bool:
    """验证配置文件中的所有路径"""
    if 'Paths' not in config:
        logging.error("配置中缺少 'Paths' 节")
        return False

    paths = config['Paths']
    required_files = {
        'terrain_path': '地形栅格文件'
    }
    
    # 可选文件（仅当使用shapefile功能时需要）
    optional_files = {
        'fault_shp_path': '断层 Shapefile'
    }

    # 验证必需的输入文件
    for path_key, file_type in required_files.items():
        if path_key not in paths:
            logging.error(f"配置中缺少 {path_key}")
            return False
        
        clean_path = verify_file_path(paths[path_key], file_type)
        if clean_path is None:
            return False
        paths[path_key] = clean_path

    # 验证可选文件（如果提供）
    for path_key, file_type in optional_files.items():
        if path_key in paths and paths[path_key].strip():
            clean_path = verify_file_path(paths[path_key], file_type)
            if clean_path is None:
                logging.warning(f"可选文件 {file_type} 验证失败，将不使用该功能")
                paths[path_key] = None
            else:
                paths[path_key] = clean_path
        else:
            paths[path_key] = None

    # 验证输出目录
    if 'output_path' not in paths:
        logging.error("配置中缺少 output_path")
        return False
        
    output_path = verify_directory_path(paths['output_path'], create=True)
    if output_path is None:
        return False
    paths['output_path'] = output_path

    return True