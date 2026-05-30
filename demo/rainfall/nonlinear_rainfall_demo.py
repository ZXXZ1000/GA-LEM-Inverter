"""Nonlinear rainfall/runoff demo for GA-LEM-Inverter.

This example combines:
- an eastward nonlinear gradient
- an elevation-dependent orographic enhancement
- a wetter early stage than late stage
"""

from __future__ import annotations

import numpy as np


def rainfall(x, y, z, t_ma, params):
    x_max = max(float(np.nanmax(x)), 1.0)
    x_norm = np.asarray(x, dtype=float) / x_max

    z = np.asarray(z, dtype=float)
    z_mean = float(np.nanmean(z))
    z_std = max(float(np.nanstd(z)), 1.0)
    z_norm = (z - z_mean) / z_std

    east_nonlinear = 0.55 * np.power(np.clip(x_norm, 0.0, 1.0), 1.8)
    orographic = 0.35 * np.tanh(1.2 * z_norm)
    wave = 0.18 * np.sin(2.5 * np.pi * x_norm)
    early_stage_boost = 0.25 if float(t_ma) > 5.0 else -0.10

    runoff = 1.0 + east_nonlinear + orographic + wave + early_stage_boost
    return np.clip(runoff, 0.2, 3.0)
