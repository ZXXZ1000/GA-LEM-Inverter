# model_runner.py
import xsimlab as xs
import numpy as np
from fastscape.models import basic_model
import logging
import warnings

def run_fastscape_model(k_sp, uplift, k_diff, x_size, y_size, spacing, boundary_status='fixed_value', area_exp=0.43, slope_exp=1, time_total=10e6):
    """
    运行 fastscape 模型。

    参数:
    - k_sp: 侵蚀系数。
    - uplift: 抬升速率。
    - k_diff: 扩散系数。
    - x_size: x 方向的网格大小。
    - y_size: y 方向的网格大小。
    - spacing: 网格间距。
    - boundary_status: 边界状态。
    - area_exp: 面积指数。
    - slope_exp: 坡度指数。
    - time_total: 总模拟时间。

    返回:
    - elevation: 模拟后的地形高程数据。
    """
    try:
        logging.info(f"Fastscape input shapes:")
        logging.info(f"k_sp shape: {k_sp.shape}")
        logging.info(f"uplift shape: {uplift.shape}")
        logging.info(f"Requested grid size: {y_size} x {x_size}")
        
        # 确保尺寸匹配
        if k_sp.shape != uplift.shape:
            logging.error(f"Shape mismatch: k_sp {k_sp.shape} vs uplift {uplift.shape}")
            # 调整到相同尺寸
            min_rows = min(k_sp.shape[0], uplift.shape[0])
            min_cols = min(k_sp.shape[1], uplift.shape[1])
            k_sp = k_sp[:min_rows, :min_cols]
            uplift = uplift[:min_rows, :min_cols]
            logging.info(f"Adjusted shapes to: {k_sp.shape}")
            
        # 在运行模型前添加以下代码
        warnings.filterwarnings("ignore", category=FutureWarning, 
                            message="variable .* with name matching its dimension")
        ds_in = xs.create_setup(
            model=basic_model,
            clocks={'time': np.linspace(0, time_total, 101),
                    'out': np.linspace(0, time_total, 21)},
            master_clock='time',
            input_vars={
                'grid__shape': [y_size, x_size],
                'grid__length': [y_size * spacing, x_size * spacing],
                'boundary__status': boundary_status,
                'uplift__rate': uplift * 10**(-4),
                'init_topography__seed': None,
                'spl__k_coef': k_sp,
                'spl__area_exp': area_exp,
                'spl__slope_exp': slope_exp,
                'diffusion__diffusivity': k_diff * 10**(-2),
            },
            output_vars={
                'topography__elevation': 'out'}
        )
        out_ds = (ds_in.xsimlab.run(model=basic_model))
        elevation = out_ds.topography__elevation.isel(out=-1).values
        return elevation
    except Exception as e:
        logging.error(f"运行 fastscape 模型出错: {e}")
        raise RuntimeError(f"运行 fastscape 模型出错: {e}")