"""正演结果绘图。

不依赖 ga_lem_inverter；只用 matplotlib + numpy。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forward_simulator.erosion_metrics import ErosionFields


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dem(dem: np.ndarray, path: Path, *, title: str, cmap: str = "terrain") -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(dem, cmap=cmap, origin="lower")
    fig.colorbar(im, ax=ax, label="elevation (m)")
    ax.set_title(title)
    ax.set_xlabel("x (cell)")
    ax.set_ylabel("y (cell)")
    return _save(fig, path)


def plot_uplift_input(
    base_field: np.ndarray,
    times_years: np.ndarray,
    multipliers: np.ndarray | None,
    path: Path,
) -> Path:
    has_curve = multipliers is not None
    fig, axes = plt.subplots(1, 2 if has_curve else 1, figsize=(12 if has_curve else 6, 5), dpi=150)
    if not has_curve:
        axes = [axes]
    im = axes[0].imshow(base_field, cmap="magma", origin="lower")
    fig.colorbar(im, ax=axes[0], label="U_base (mm/yr)")
    axes[0].set_title("Uplift base field U_base(x,y)")
    axes[0].set_xlabel("x (cell)")
    axes[0].set_ylabel("y (cell)")
    if has_curve:
        axes[1].plot(times_years / 1.0e6, multipliers, "-o", color="C3")
        axes[1].set_xlabel("elapsed time (Ma)")
        axes[1].set_ylabel("multiplier f(t)")
        axes[1].set_title("Uplift time function")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_history_grid(
    fields: ErosionFields,
    path: Path,
    *,
    max_panels: int = 9,
    title: str = "Topography evolution",
) -> Path:
    times = fields.output_times_years
    z = fields.topography_series
    n = z.shape[0]
    indices = np.linspace(0, n - 1, min(max_panels, n)).astype(int)
    cols = int(np.ceil(np.sqrt(len(indices))))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), dpi=150)
    axes = np.atleast_1d(axes).reshape(-1)
    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))
    for ax, idx in zip(axes, indices):
        im = ax.imshow(z[idx], cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {times[idx] / 1.0e6:.2f} Ma")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(indices):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[: len(indices)].tolist(), shrink=0.8, label="elevation (m)")
    fig.suptitle(title)
    return _save(fig, path)


def plot_field_final(field: np.ndarray, path: Path, *, title: str, label: str, cmap: str = "viridis") -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(field, cmap=cmap, origin="lower")
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel("x (cell)")
    ax.set_ylabel("y (cell)")
    return _save(fig, path)


def plot_erosion_history(fields: ErosionFields, path: Path) -> Path:
    times_ma = fields.output_times_years / 1.0e6
    cum = fields.cumulative_erosion
    rate = fields.mean_erosion_rate
    cum_med = np.array([np.nanmedian(frame) for frame in cum])
    cum_p10 = np.array([np.nanpercentile(frame, 10) for frame in cum])
    cum_p90 = np.array([np.nanpercentile(frame, 90) for frame in cum])
    rate_med = np.array([np.nanmedian(frame) for frame in rate])
    rate_p10 = np.array([np.nanpercentile(frame, 10) for frame in rate])
    rate_p90 = np.array([np.nanpercentile(frame, 90) for frame in rate])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    axes[0].fill_between(times_ma, cum_p10, cum_p90, color="C0", alpha=0.25, label="P10–P90")
    axes[0].plot(times_ma, cum_med, "-o", color="C0", label="median")
    axes[0].set_xlabel("elapsed time (Ma)")
    axes[0].set_ylabel("cumulative erosion (m)")
    axes[0].set_title("Cumulative erosion (region stats)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(times_ma, rate_p10, rate_p90, color="C3", alpha=0.25, label="P10–P90")
    axes[1].plot(times_ma, rate_med, "-o", color="C3", label="median")
    axes[1].set_xlabel("elapsed time (Ma)")
    axes[1].set_ylabel("mean erosion rate (mm/yr)")
    axes[1].set_title("Mean erosion rate (region stats)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return _save(fig, path)


def plot_rainfall_preview(field: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(field, cmap="Blues", origin="lower")
    fig.colorbar(im, ax=ax, label="runoff factor")
    ax.set_title("Rainfall / runoff preview at t=0")
    ax.set_xlabel("x (cell)")
    ax.set_ylabel("y (cell)")
    return _save(fig, path)
