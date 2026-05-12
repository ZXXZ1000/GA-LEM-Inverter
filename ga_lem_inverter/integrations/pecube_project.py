"""Build Pecube project directories from NumPy arrays."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PecubeProject:
    """Paths for one generated Pecube project."""

    project_dir: Path
    input_file: Path
    data_dir: Path
    dataset_dir: Path


@dataclass(frozen=True)
class PecubeProjectConfig:
    """Minimal Pecube input controls used by the Python glue layer."""

    dataset_name: str = "fastscape"
    nx: int | None = None
    ny: int | None = None
    dlon: float = 0.01
    dlat: float = 0.01
    lon0: float = 0.0
    lat0: float = 0.0
    total_time_myr: float = 1.0
    velocity_km_per_myr: float = 1.0
    age_ahe: bool = True
    echo_input_file: bool = True


class PecubeProjectBuilder:
    """Create Pecube project folders, input files, and data series."""

    def __init__(self, config: PecubeProjectConfig | None = None):
        self.config = config or PecubeProjectConfig()

    def build(
        self,
        *,
        project_dir: Path,
        topography_series: Iterable[np.ndarray],
        uplift_series: Iterable[np.ndarray],
        temperature_series: Iterable[np.ndarray] | None = None,
        sample_observations: Path | None = None,
    ) -> PecubeProject:
        topographies = [self._as_2d_array(array, "topography") for array in topography_series]
        uplifts = [self._as_2d_array(array, "uplift") for array in uplift_series]
        if not topographies:
            raise ValueError("topography_series 至少需要一个二维数组。")
        if len(topographies) != len(uplifts):
            raise ValueError("topography_series 和 uplift_series 的长度必须一致。")

        shape = topographies[0].shape
        for array in [*topographies, *uplifts]:
            if array.shape != shape:
                raise ValueError(f"Pecube 序列数组形状必须一致，期望 {shape}，得到 {array.shape}。")

        if temperature_series is None:
            temperatures = [np.zeros(shape, dtype=float) for _ in topographies]
        else:
            temperatures = [self._as_2d_array(array, "temperature") for array in temperature_series]
            if len(temperatures) != len(topographies):
                raise ValueError("temperature_series 的长度必须和 topography_series 一致。")
            for array in temperatures:
                if array.shape != shape:
                    raise ValueError(f"temperature_series 形状必须为 {shape}，得到 {array.shape}。")

        project_dir = Path(project_dir).resolve()
        if project_dir.exists():
            shutil.rmtree(project_dir)
        input_dir = project_dir / "input"
        data_dir = project_dir / "data"
        dataset_dir = data_dir / self.config.dataset_name
        input_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for index, array in enumerate(topographies):
            self._write_grid(dataset_dir / f"topo{index}", array)
        for index, array in enumerate(uplifts):
            self._write_grid(dataset_dir / f"uplift{index}", array)
        for index, array in enumerate(temperatures):
            self._write_grid(dataset_dir / f"temp{index}", array)

        has_observations = bool(sample_observations and Path(sample_observations).exists())
        if has_observations:
            samples_dir = data_dir / "observations"
            samples_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample_observations, samples_dir / Path(sample_observations).name)

        input_file = input_dir / "Pecube.in"
        input_file.write_text(
            self._render_input(shape=shape, ntime=len(topographies) - 1, has_observations=has_observations),
            encoding="utf-8",
        )
        return PecubeProject(project_dir=project_dir, input_file=input_file, data_dir=data_dir, dataset_dir=dataset_dir)

    @staticmethod
    def _as_2d_array(array: np.ndarray, label: str) -> np.ndarray:
        result = np.asarray(array, dtype=float)
        if result.ndim != 2:
            raise ValueError(f"{label} 必须是二维数组，得到 ndim={result.ndim}。")
        return result

    @staticmethod
    def _write_grid(path: Path, array: np.ndarray) -> None:
        # Pecube expects one column, ordered from bottom-left to top-right.
        values = np.flipud(np.asarray(array, dtype=float)).ravel()
        np.savetxt(path, values, fmt="%.8f")

    def _render_input(self, *, shape: tuple[int, int], ntime: int, has_observations: bool) -> str:
        rows, cols = shape
        nx = self.config.nx or cols
        ny = self.config.ny or rows
        total_time = max(float(self.config.total_time_myr), 1e-6)
        ntime = max(int(ntime), 1)
        step = total_time / ntime
        lines = [
            "GA-LEM-Inverter generated Pecube input",
            "FastScape/Pecube coupling smoke project",
            "",
            f"echo_input_file = {1 if self.config.echo_input_file else 0}",
            f"nx = {nx}",
            f"ny = {ny}",
            "nskip = 1",
            f"lon0 = {self.config.lon0}",
            f"lat0 = {self.config.lat0}",
            f"dlon = {self.config.dlon}",
            f"dlat = {self.config.dlat}",
            f"topo_file_name = {self.config.dataset_name}/",
            f"ntime = {ntime}",
        ]
        if has_observations:
            lines.append("data_folder = observations")
        for index in range(ntime):
            time_value = total_time - index * step
            lines.extend([
                f"time_topo{index + 1} = {time_value:.6f}",
                f"output{index + 1} = 0",
            ])
        lines.extend([
            "nfault = 1",
            "npoint1 = -1",
            "nstep1 = 1",
            f"time_start1_1 = {total_time:.6f}",
            "time_end1_1 = 0",
            f"velo1_1 = {self.config.velocity_km_per_myr:.6f}",
            f"age_AHe_flag = {1 if self.config.age_ahe else 0}",
            "save_PTT_paths = 0",
            "",
        ])
        return "\n".join(lines)
