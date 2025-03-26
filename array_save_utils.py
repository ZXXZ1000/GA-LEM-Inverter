# array_save_utils.py

import numpy as np
import numpy.ma as ma
import logging
import os
from typing import Union, Any

def safe_save_array(array: Union[np.ndarray, ma.MaskedArray], 
                   filepath: str, 
                   fill_value: Any = np.nan) -> bool:
    """
    安全地保存数组到文件，正确处理MaskedArray
    
    参数:
    - array: 要保存的数组（可以是普通ndarray或MaskedArray）
    - filepath: 保存路径
    - fill_value: 用于填充掩码值的值，默认为np.nan
    
    返回:
    - bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 检查是否为MaskedArray
        if isinstance(array, ma.MaskedArray):
            # 将MaskedArray转换为普通数组，用fill_value填充掩码值
            array_to_save = array.filled(fill_value)
        else:
            array_to_save = array
            
        # 保存数组
        np.save(filepath, array_to_save)
        logging.info(f"Successfully saved array to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving array to {filepath}: {e}")
        return False

def safe_load_array(filepath: str) -> Union[np.ndarray, None]:
    """
    安全地加载数组
    
    参数:
    - filepath: 文件路径
    
    返回:
    - np.ndarray 或 None（如果加载失败）
    """
    try:
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return None
            
        array = np.load(filepath)
        return array
        
    except Exception as e:
        logging.error(f"Error loading array from {filepath}: {e}")
        return None