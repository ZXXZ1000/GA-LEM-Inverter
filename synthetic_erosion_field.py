# synthetic_erosion_field.py

import numpy as np
import logging
from typing import Tuple

def create_synthetic_erosion_field(shape: Tuple[int, int], 
                                 base_k_sp: float,
                                 border_width: int = 2) -> np.ndarray:
    """
    为合成实验创建简化版本的侵蚀系数场
    
    参数:
    - shape: 输出矩阵的形状 (rows, cols)
    - base_k_sp: 基础侵蚀系数
    - border_width: 边界宽度（像素）
    
    返回:
    - Ksp: 侵蚀系数场矩阵
    """
    try:
        row, col = shape
        
        # 创建基础侵蚀系数场
        Ksp = np.ones((row, col)) * base_k_sp

        # 设置边界条件
        Ksp[:border_width, :] = 0  # 下边界
        Ksp[-border_width:, :] = 0  # 上边界
        Ksp[:, :border_width] = 0  # 左边界
        Ksp[:, -border_width:] = 0  # 右边界

        # 添加一些随机变化（可选）
        random_variation = np.random.normal(0, 0.1 * base_k_sp, Ksp.shape)
        Ksp += random_variation
        Ksp = np.clip(Ksp, 0, base_k_sp * 1.2)  # 确保值在合理范围内
        
        return Ksp

    except Exception as e:
        logging.error(f"创建合成侵蚀系数场失败: {e}")
        raise RuntimeError(f"创建合成侵蚀系数场失败: {e}")