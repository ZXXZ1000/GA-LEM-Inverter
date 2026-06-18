"""Demo uplift 时间函数：周期化的全域倍率。"""

from __future__ import annotations

import numpy as np


def uplift_time(t_yr, x, y, z, params):
    amplitude = float(params.get("amplitude", 0.4))
    period_ma = float(params.get("period_ma", 0.5))
    t_ma = t_yr / 1.0e6
    return 1.0 + amplitude * np.sin(2.0 * np.pi * t_ma / period_ma)
