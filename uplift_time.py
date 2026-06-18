"""自定义 uplift 时间函数（仅当 forward_config.ini 中 [UpliftTime] mode = python 时被加载）。

返回值约定：
- 返回 scalar：所有像素共用同一倍率，最终 U(x,y,t) = uplift_base(x,y) * scalar
- 返回 (Y,X) 数组：直接作为该时刻的 uplift 场（mm/yr），uplift_base 被忽略

约束：
- 必须返回有限数；scalar 时不限正负，但负值意味着该像素正在沉降，建议确认你的物理意图。
- 矩阵 shape 必须和当前 DEM/FastScape 网格一致。
"""

from __future__ import annotations

import numpy as np


def uplift_time(t_yr, x, y, z, params):
    """U(x,y,t) 的时间函数。

    Parameters
    ----------
    t_yr : float
        从 t=0 起算的已用模型年数。t=0 = 现今 DEM；t = time_total = 演化结束时刻。
    x, y : np.ndarray
        网格坐标 (Y, X)，单位 m。
    z : np.ndarray
        当前地形高程 (Y, X)，单位 m。
    params : dict[str, str]
        来自 [UpliftTime] 段的额外键值（按字符串传入）。

    Returns
    -------
    scalar 或 (Y, X) np.ndarray
        见模块顶部说明。
    """
    amplitude = float(params.get("amplitude", 0.3))
    period_ma = float(params.get("period_ma", 3.0))
    t_ma = t_yr / 1.0e6
    # 围绕 1.0 振荡的倍率：默认参数下幅度 ±0.3，周期 3 Ma。
    return 1.0 + amplitude * np.sin(2.0 * np.pi * t_ma / period_ma)
