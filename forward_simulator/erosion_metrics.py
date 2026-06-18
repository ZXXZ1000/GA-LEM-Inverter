"""从 FastScape 输出序列计算剥蚀字段。

约定：
- ``output_times_years``: 形状 (T,)，从 t=0 起算的累计年数；t=0 = 现今 DEM。
- ``topography_series``:  形状 (T, Y, X)，单位 m。
- ``uplift_series``:       形状 (T, Y, X)，单位 mm/yr，对应每个输出时刻的 U(x,y,t)。

返回字段：
- ``cumulative_erosion`` (T,Y,X)，单位 m，定义 = 累积抬升 − 净抬升 = ∫U dt − (z(t)−z₀)。
- ``mean_erosion_rate``  (T,Y,X)，单位 mm/yr，= cumulative_erosion / t * 1e3。t=0 帧设为 0。
- ``net_uplift``         (T,Y,X)，单位 m，= z(t) − z₀。
- ``cumulative_uplift``  (T,Y,X)，单位 m，∫U dt（用梯形规则）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErosionFields:
    output_times_years: np.ndarray
    topography_series: np.ndarray
    uplift_series: np.ndarray
    cumulative_uplift: np.ndarray
    net_uplift: np.ndarray
    cumulative_erosion: np.ndarray
    mean_erosion_rate: np.ndarray


def compute_erosion_fields(
    output_times_years: np.ndarray,
    topography_series: np.ndarray,
    uplift_series_mm_per_yr: np.ndarray,
    initial_dem: np.ndarray,
) -> ErosionFields:
    times = np.asarray(output_times_years, dtype=float).reshape(-1)
    z = np.asarray(topography_series, dtype=float)
    u = np.asarray(uplift_series_mm_per_yr, dtype=float)
    z0 = np.asarray(initial_dem, dtype=float)

    if z.ndim != 3:
        raise ValueError(f"topography_series 必须是 (T,Y,X)，当前 shape={z.shape}")
    if u.shape != z.shape:
        raise ValueError(
            f"uplift_series shape={u.shape} 与 topography_series shape={z.shape} 不一致"
        )
    if z.shape[0] != times.shape[0]:
        raise ValueError(
            f"output_times_years 长度 {times.shape[0]} 与帧数 {z.shape[0]} 不一致"
        )
    if z0.shape != z.shape[1:]:
        raise ValueError(
            f"initial_dem shape={z0.shape} 与 (Y,X)={z.shape[1:]} 不一致"
        )
    if not np.all(np.diff(times) > 0):
        raise ValueError("output_times_years 必须严格递增。")

    # 累积抬升：U 单位 mm/yr → m/yr 后对时间做梯形积分
    u_m_per_yr = u * 1.0e-3
    cumulative_uplift = _cumulative_trapz(u_m_per_yr, times)
    net_uplift = z - z0[np.newaxis, :, :]
    cumulative_erosion = cumulative_uplift - net_uplift

    mean_erosion_rate = np.zeros_like(cumulative_erosion)
    nonzero = times > 0
    if np.any(nonzero):
        denom = times[nonzero][:, np.newaxis, np.newaxis]
        mean_erosion_rate[nonzero] = cumulative_erosion[nonzero] / denom * 1.0e3

    return ErosionFields(
        output_times_years=times,
        topography_series=z,
        uplift_series=u,
        cumulative_uplift=cumulative_uplift,
        net_uplift=net_uplift,
        cumulative_erosion=cumulative_erosion,
        mean_erosion_rate=mean_erosion_rate,
    )


def _cumulative_trapz(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """对第 0 维做累积梯形积分，第 0 帧填 0。"""
    out = np.zeros_like(values)
    if values.shape[0] < 2:
        return out
    dt = np.diff(times)
    midpoint = 0.5 * (values[1:] + values[:-1]) * dt[:, np.newaxis, np.newaxis]
    out[1:] = np.cumsum(midpoint, axis=0)
    return out


def summarize_metrics(fields: ErosionFields) -> dict[str, float]:
    """挑几个标量指标写进 summary.md。"""
    z_final = fields.topography_series[-1]
    z_initial = fields.topography_series[0]
    erosion_final = fields.cumulative_erosion[-1]
    rate_final = fields.mean_erosion_rate[-1]
    metrics = {
        "time_total_years": float(fields.output_times_years[-1]),
        "frames": int(fields.output_times_years.size),
        "elevation_initial_min_m": float(np.nanmin(z_initial)),
        "elevation_initial_max_m": float(np.nanmax(z_initial)),
        "elevation_final_min_m": float(np.nanmin(z_final)),
        "elevation_final_max_m": float(np.nanmax(z_final)),
        "cumulative_erosion_final_min_m": float(np.nanmin(erosion_final)),
        "cumulative_erosion_final_median_m": float(np.nanmedian(erosion_final)),
        "cumulative_erosion_final_max_m": float(np.nanmax(erosion_final)),
        "mean_erosion_rate_final_min_mm_per_yr": float(np.nanmin(rate_final)),
        "mean_erosion_rate_final_median_mm_per_yr": float(np.nanmedian(rate_final)),
        "mean_erosion_rate_final_max_mm_per_yr": float(np.nanmax(rate_final)),
        "net_uplift_final_min_m": float(np.nanmin(fields.net_uplift[-1])),
        "net_uplift_final_max_m": float(np.nanmax(fields.net_uplift[-1])),
    }
    return metrics
