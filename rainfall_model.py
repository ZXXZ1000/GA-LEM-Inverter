"""User-editable rainfall/runoff model for GA-LEM-Inverter.

Edit ``rainfall`` when ``config.ini`` uses:

    [Rainfall]
    mode = python
    module_path = ./rainfall_model.py
    function = rainfall

The function must return either:
- a positive scalar, or
- a positive finite matrix with exactly the same shape as ``z``.

That shape is the active FastScape DEM grid, so it is also the grid used by
Ksp, uplift, drainage and the final terrain comparison. The workflow checks
the returned field before FastScape uses it and raises a clear error if the
shape is wrong or values contain NaN/Inf/zero/negative numbers.
"""

from __future__ import annotations

import numpy as np


def rainfall(x, y, z, t_ma, params):
    """Return runoff/rainfall factor p=f(x,y,z,t).

    Parameters
    ----------
    x, y
        Model-grid coordinates in meters, 2-D arrays.
    z
        Current topography/elevation in meters, 2-D array.
    t_ma
        Geologic time before present in Ma. ``0`` means present day.
    params
        String parameters read from ``[Rainfall]`` in config.ini.

    Returns
    -------
    array-like or scalar
        Positive runoff/rainfall factor. ``1.0`` means the same runoff as the
        uniform default. Larger values increase runoff; smaller positive values
        reduce runoff. This is passed to FastScape as ``FlowAccumulator.runoff``
        and does not modify Ksp.

    The default is uniform rainfall/runoff. For an elevation-dependent example:

        base = 1.0
        per_km = 0.25
        p = base + per_km * (z - np.nanmean(z)) / 1000.0
        return np.clip(p, 0.3, 3.0)

    For a time-dependent example:

        early = 1.4 if t_ma > 5.0 else 0.8
        return np.ones_like(z, dtype=float) * early
    """
    base = float(params.get("base", 1.0))
    return np.ones_like(z, dtype=float) * base
