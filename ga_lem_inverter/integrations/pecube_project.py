"""Build Pecube project directories from NumPy arrays."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    nskip: int = 4
    total_time_myr: float = 1.0
    velocity_km_per_myr: float = 1.0
    include_uniform_velocity_field: bool = False
    thickness: float = 35.0
    nz: int = 21
    thermal_diffusivity: float = 25.0
    basal_temperature: float = 700.0
    sea_level_temperature: float = 15.0
    lapse_rate: float = 6.5
    heat_production: float = 0.0
    erosional_time_scale: float = 0.0
    save_ptt_paths: bool = False
    age_ahe: bool = True
    age_zhe: bool = True
    age_aft: bool = True
    age_zft: bool = True
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
        normalized_observations: Iterable[Any] | None = None,
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
            temperatures = [surface_temperature_from_topography(array, self.config) for array in topographies]
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
            native_observation_path = samples_dir / "observations.csv"
            if normalized_observations is not None:
                self._write_native_observation_rows(normalized_observations, native_observation_path)
            else:
                self._write_native_observation_file(Path(sample_observations), native_observation_path)

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

    @staticmethod
    def _write_native_observation_file(source_path: Path, target_path: Path) -> None:
        grouped: dict[str, dict[str, str | float]] = {}
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"观测样品 CSV 没有表头: {source_path}")
            for row in reader:
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    continue
                lon_raw = row.get("lon") or row.get("longitude") or row.get("x")
                lat_raw = row.get("lat") or row.get("latitude") or row.get("y")
                if lon_raw in {None, ""} or lat_raw in {None, ""}:
                    raise ValueError(f"样品 {sample_id} 缺少 lon/lat，无法转换为 Pecube 原生观测文件。")
                system = str(row.get("system", "")).strip().lower().replace("-", "").replace("_", "")
                observed_age = str(row.get("observed_age", "")).strip()
                sigma = str(row.get("sigma", "")).strip()
                sample = grouped.setdefault(
                    sample_id,
                    {
                        "SAMPLE": sample_id,
                        "LON": float(lon_raw),
                        "LAT": float(lat_raw),
                        "HEIGHT": float(str(row.get("elevation", "0")).strip() or "0"),
                    },
                )
                mapping = {
                    "ahe": ("AHE", "DAHE"),
                    "heapatite": ("AHE", "DAHE"),
                    "apatitehe": ("AHE", "DAHE"),
                    "zhe": ("ZHE", "DZHE"),
                    "hezircon": ("ZHE", "DZHE"),
                    "zirconhe": ("ZHE", "DZHE"),
                    "aft": ("AFT", "DAFT"),
                    "ftapatite": ("AFT", "DAFT"),
                    "apatiteft": ("AFT", "DAFT"),
                    "zft": ("ZFT", "DZFT"),
                    "ftzircon": ("ZFT", "DZFT"),
                    "zirconft": ("ZFT", "DZFT"),
                }
                columns = mapping.get(system)
                if columns is None:
                    continue
                value_column, sigma_column = columns
                sample[value_column] = float(observed_age)
                sample[sigma_column] = float(sigma)

        if not grouped:
            raise ValueError(f"观测样品 CSV 没有可写入 Pecube 的样品行: {source_path}")

        fieldnames = [
            "SAMPLE",
            "LON",
            "LAT",
            "HEIGHT",
            "AHE",
            "DAHE",
            "AFT",
            "DAFT",
            "ZHE",
            "DZHE",
            "ZFT",
            "DZFT",
        ]
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id in sorted(grouped):
                writer.writerow(grouped[sample_id])

    @staticmethod
    def _write_native_observation_rows(observations: Iterable[Any], target_path: Path) -> None:
        grouped: dict[str, dict[str, str | float]] = {}
        mapping = {
            "ahe": ("AHE", "DAHE"),
            "heapatite": ("AHE", "DAHE"),
            "apatitehe": ("AHE", "DAHE"),
            "zhe": ("ZHE", "DZHE"),
            "hezircon": ("ZHE", "DZHE"),
            "zirconhe": ("ZHE", "DZHE"),
            "aft": ("AFT", "DAFT"),
            "ftapatite": ("AFT", "DAFT"),
            "apatiteft": ("AFT", "DAFT"),
            "zft": ("ZFT", "DZFT"),
            "ftzircon": ("ZFT", "DZFT"),
            "zirconft": ("ZFT", "DZFT"),
        }
        for observation in observations:
            sample_id = str(getattr(observation, "sample_id")).strip()
            system = str(getattr(observation, "system")).strip().lower().replace("-", "").replace("_", "")
            columns = mapping.get(system)
            if not sample_id or columns is None:
                continue
            sample = grouped.setdefault(
                sample_id,
                {
                    "SAMPLE": sample_id,
                    "LON": float(getattr(observation, "x")),
                    "LAT": float(getattr(observation, "y")),
                    "HEIGHT": float(getattr(observation, "elevation")),
                },
            )
            value_column, sigma_column = columns
            sample[value_column] = float(getattr(observation, "observed_age"))
            sample[sigma_column] = float(getattr(observation, "sigma"))
        fieldnames = [
            "SAMPLE",
            "LON",
            "LAT",
            "HEIGHT",
            "AHE",
            "DAHE",
            "AFT",
            "DAFT",
            "ZHE",
            "DZHE",
            "ZFT",
            "DZFT",
        ]
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id in sorted(grouped):
                writer.writerow(grouped[sample_id])

    def _render_input(self, *, shape: tuple[int, int], ntime: int, has_observations: bool) -> str:
        rows, cols = shape
        nx = self.config.nx or cols
        ny = self.config.ny or rows
        nskip = max(1, int(self.config.nskip))
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
            f"nskip = {nskip}",
            f"lon0 = {self.config.lon0}",
            f"lat0 = {self.config.lat0}",
            f"dlon = {self.config.dlon}",
            f"dlat = {self.config.dlat}",
            f"thickness = {self.config.thickness}",
            f"nz = {int(self.config.nz)}",
            f"thermal_diffusivity = {self.config.thermal_diffusivity}",
            f"basal_temperature = {self.config.basal_temperature}",
            f"sea_level_temperature = {self.config.sea_level_temperature}",
            f"lapse_rate = {self.config.lapse_rate}",
            f"heat_production = {self.config.heat_production}",
            f"erosional_time_scale = {self.config.erosional_time_scale}",
            f"topo_file_name = {self.config.dataset_name}/",
            f"ntime = {ntime}",
        ]
        if has_observations:
            lines.append("data_folder = observations")
        for index in range(ntime):
            time_value = total_time - index * step
            lines.extend([
                f"time_topo{index + 1} = {time_value:.6f}",
                f"output{index + 1} = 1",
            ])
        if self.config.include_uniform_velocity_field:
            lines.extend([
                "nfault = 1",
                "npoint1 = -1",
                "nstep1 = 1",
                f"time_start1_1 = {total_time:.6f}",
                "time_end1_1 = 0",
                f"velo1_1 = {self.config.velocity_km_per_myr:.6f}",
            ])
        else:
            lines.append("nfault = 0")
        lines.extend([
            f"age_AHe_flag = {1 if self.config.age_ahe else 0}",
            f"age_ZHe_flag = {1 if self.config.age_zhe else 0}",
            f"age_AFT_flag = {1 if self.config.age_aft else 0}",
            f"age_ZFT_flag = {1 if self.config.age_zft else 0}",
            f"save_PTT_paths = {1 if self.config.save_ptt_paths else 0}",
            "",
        ])
        return "\n".join(lines)


def surface_temperature_from_topography(topography: np.ndarray, config: PecubeProjectConfig) -> np.ndarray:
    """Build Pecube surface-temperature grids from topography.

    Pecube interprets topography in meters, converts it to km internally, and
    uses a positive lapse rate as colder temperature at higher elevation.
    """
    topo_km = np.asarray(topography, dtype=float) / 1000.0
    return float(config.sea_level_temperature) - float(config.lapse_rate) * topo_km
