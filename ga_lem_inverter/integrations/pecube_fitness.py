"""Pecube thermochronology constraint used by optimization workflows."""

from __future__ import annotations

import configparser
import csv
import json
import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform_bounds
from rasterio.warp import Resampling, reproject

from ga_lem_inverter.integrations.pecube import PecubeEngine
from ga_lem_inverter.integrations.pecube_parser import PecubeParsedOutput
from ga_lem_inverter.outputs import RunContext


REQUIRED_OBSERVATION_COLUMNS = ("sample_id", "elevation", "system", "observed_age", "sigma")
COORDINATE_COLUMN_SETS = (("x", "y"), ("lon", "lat"), ("longitude", "latitude"))

SYSTEM_TO_PECUBE_COLUMN = {
    "ahe": "HeApatite",
    "heapatite": "HeApatite",
    "apatitehe": "HeApatite",
    "zhe": "HeZircon",
    "hezircon": "HeZircon",
    "zirconhe": "HeZircon",
    "aft": "FTApatite",
    "ftapatite": "FTApatite",
    "apatiteft": "FTApatite",
    "zft": "FTZircon",
    "ftzircon": "FTZircon",
    "zirconft": "FTZircon",
    "arkfeldspar": "ArKFeldspar",
    "arbiotite": "ArBiotite",
    "armuscovite": "ArMuscovite",
    "arhornblend": "ArHornblend",
}


@dataclass(frozen=True)
class ThermochronologyObservation:
    sample_id: str
    x: float
    y: float
    elevation: float
    system: str
    observed_age: float
    sigma: float


@dataclass(frozen=True)
class ThermochronologyPrediction:
    sample_id: str
    x: float
    y: float
    elevation: float
    system: str
    observed_age: float
    predicted_age: float
    sigma: float
    residual: float
    normalized_residual: float
    pecube_column: str
    source_file: str


@dataclass(frozen=True)
class PecubeSpatialGrid:
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    crs: str
    source: str


@dataclass(frozen=True)
class PecubeSpatialAdapter:
    """Map DEM arrays into the regular geographic grid Pecube expects."""

    grid: PecubeSpatialGrid
    source_crs: str
    source_transform: tuple[float, float, float, float, float, float]
    source_shape: tuple[int, int]
    target_transform: tuple[float, float, float, float, float, float]
    target_shape: tuple[int, int]
    resample: bool

    def transform_array(self, array: np.ndarray, *, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
        array = np.asarray(array, dtype=float)
        if array.shape != self.source_shape:
            raise ValueError(f"Pecube 输入数组 shape={array.shape} 与 DEM shape={self.source_shape} 不一致。")
        if not self.resample:
            return array.copy()
        destination = np.full(self.target_shape, np.nan, dtype=float)
        reproject(
            source=array,
            destination=destination,
            src_transform=Affine(*self.source_transform),
            src_crs=self.source_crs,
            dst_transform=Affine(*self.target_transform),
            dst_crs="EPSG:4326",
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return destination


@dataclass(frozen=True)
class PecubeConstraintResult:
    enabled: bool
    terrain_loss: float
    thermo_loss: float | None
    total_loss: float
    n_observations: int
    message: str
    predictions: list[ThermochronologyPrediction] = field(default_factory=list)
    project_dir: Path | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        return {
            "terrain_loss": self.terrain_loss,
            "thermo_loss": self.thermo_loss,
            "total_loss": self.total_loss,
            "thermo_n_observations": self.n_observations,
            "pecube_message": self.message,
            **self.raw_metrics,
        }


class PecubeFitnessEvaluator:
    """Evaluate optional Pecube thermochronology loss for one candidate uplift field."""

    def __init__(
        self,
        *,
        config: configparser.ConfigParser,
        context: RunContext,
        target_dem: np.ndarray,
        ksp: np.ndarray,
        model_params: dict[str, Any],
    ):
        self.config = config
        self.context = context
        self.target_dem = np.asarray(target_dem, dtype=float)
        self.ksp = np.asarray(ksp, dtype=float)
        self.model_params = model_params
        self.engine = PecubeEngine.from_config(config)
        self.enabled = self._enabled()
        self.terrain_weight = self._float("Fitness", "terrain_loss_weight", 1.0)
        self.thermo_weight = self._float("Fitness", "thermo_loss_weight", 1.0)
        self.run_every = max(1, self._int("Pecube", "run_every", 1))
        self.max_evaluations = max(0, self._int("Pecube", "max_evaluations", 0))
        self.fail_strategy = self._str("Pecube", "fail_strategy", "penalty").lower()
        self.penalty_loss = self._float("Pecube", "penalty_loss", 1.0)
        self.thermo_loss_scale = max(self._float("Fitness", "thermo_loss_scale", 5.0), 1e-12)
        self.observation_coordinate_system = self._str("Pecube", "observation_coordinate_system", "pecube_output").lower()
        self.observation_crs = self._str("Pecube", "observation_crs", "auto")
        self.evaluation_count = 0
        self.best_result: PecubeConstraintResult | None = None
        self.history: list[dict[str, Any]] = []
        self.observations: list[ThermochronologyObservation] = []
        self.history_dir = self.context.root / "pecube" / "fitness_history"
        self.spatial_adapter: PecubeSpatialAdapter | None = None
        self.target_dem_pecube = self.target_dem

        if self.enabled:
            self.engine.validate()
            if self.engine.config.sample_observations is None:
                raise ValueError("[Pecube] 已启用热年代学约束，但 sample_observations 未配置。")
            if self._str("Pecube", "spatial_grid", "auto").strip().lower() in {"manual", "config"}:
                self.observations = load_observations(
                    self.engine.config.sample_observations,
                    self.target_dem.shape,
                    coordinate_system=self.observation_coordinate_system,
                    dlon=self.engine.config.dlon,
                    dlat=self.engine.config.dlat,
                    lon0=self.engine.config.lon0,
                    lat0=self.engine.config.lat0,
                    observation_crs=self.observation_crs,
                )

    @classmethod
    def from_config(
        cls,
        *,
        config: configparser.ConfigParser,
        context: RunContext,
        target_dem: np.ndarray,
        ksp: np.ndarray,
        model_params: dict[str, Any],
    ) -> "PecubeFitnessEvaluator":
        return cls(config=config, context=context, target_dem=target_dem, ksp=ksp, model_params=model_params)

    def apply_spatial_grid(self, spatial_grid: PecubeSpatialGrid) -> None:
        """Use DEM-derived coordinates for Pecube projects and sample matching."""
        self.engine = self.engine.with_spatial_grid(
            lon0=spatial_grid.lon0,
            lat0=spatial_grid.lat0,
            dlon=spatial_grid.dlon,
            dlat=spatial_grid.dlat,
        )
        if self.enabled and self.engine.config.sample_observations is not None:
            self.observations = load_observations(
                self.engine.config.sample_observations,
                self.target_dem.shape,
                coordinate_system=self.observation_coordinate_system,
                dlon=self.engine.config.dlon,
                dlat=self.engine.config.dlat,
                lon0=self.engine.config.lon0,
                lat0=self.engine.config.lat0,
                dem_crs=spatial_grid.crs,
                observation_crs=self.observation_crs,
            )

    def apply_spatial_adapter(self, spatial_adapter: PecubeSpatialAdapter) -> None:
        """Use a DEM-to-Pecube adapter and reproject arrays when needed."""
        self.spatial_adapter = spatial_adapter
        self.target_dem_pecube = spatial_adapter.transform_array(self.target_dem)
        self.engine = self.engine.with_spatial_grid(
            lon0=spatial_adapter.grid.lon0,
            lat0=spatial_adapter.grid.lat0,
            dlon=spatial_adapter.grid.dlon,
            dlat=spatial_adapter.grid.dlat,
        )
        if self.enabled and self.engine.config.sample_observations is not None:
            self.observations = load_observations(
                self.engine.config.sample_observations,
                spatial_adapter.target_shape,
                coordinate_system=self.observation_coordinate_system,
                dlon=self.engine.config.dlon,
                dlat=self.engine.config.dlat,
                lon0=self.engine.config.lon0,
                lat0=self.engine.config.lat0,
                dem_crs=spatial_adapter.source_crs,
                dem_transform=spatial_adapter.source_transform,
                source_dem_shape=spatial_adapter.source_shape,
                observation_crs=self.observation_crs,
            )

    def evaluate(
        self,
        *,
        terrain_loss: float,
        generated_dem: np.ndarray,
        uplift: np.ndarray,
        topography_series: np.ndarray | list[np.ndarray] | None = None,
    ) -> PecubeConstraintResult:
        if not self.enabled:
            return PecubeConstraintResult(
                enabled=False,
                terrain_loss=float(terrain_loss),
                thermo_loss=None,
                total_loss=float(terrain_loss),
                n_observations=0,
                message="Pecube 约束未启用。",
            )

        evaluation_id = self._next_evaluation_id()
        if not self.observations:
            raise ValueError(
                "Pecube 热年代学样品尚未完成坐标转换。"
                "请确认 [Pecube] spatial_grid=auto 且 DEM 有 CRS/transform，或设置 spatial_grid=manual 并填写 lon0/lat0/dlon/dlat。"
            )
        if evaluation_id % self.run_every != 0:
            terrain_loss_norm = normalize_unit_loss(terrain_loss)
            thermo_loss = 1.0
            total = weighted_unit_loss(terrain_loss_norm, thermo_loss, self.terrain_weight, self.thermo_weight)
            result = PecubeConstraintResult(
                enabled=True,
                terrain_loss=terrain_loss_norm,
                thermo_loss=thermo_loss,
                total_loss=total,
                n_observations=len(self.observations),
                message=f"按 run_every={self.run_every} 跳过本次 Pecube 评价，使用 penalty_loss。",
                raw_metrics={"terrain_loss_raw": float(terrain_loss), "thermo_loss_raw": self.penalty_loss},
            )
            self._record(result, evaluation_id=evaluation_id)
            return result
        if self.max_evaluations and evaluation_id > self.max_evaluations:
            terrain_loss_norm = normalize_unit_loss(terrain_loss)
            thermo_loss = 1.0
            total = weighted_unit_loss(terrain_loss_norm, thermo_loss, self.terrain_weight, self.thermo_weight)
            result = PecubeConstraintResult(
                enabled=True,
                terrain_loss=terrain_loss_norm,
                thermo_loss=thermo_loss,
                total_loss=total,
                n_observations=len(self.observations),
                message=f"已达到 max_evaluations={self.max_evaluations}，使用 penalty_loss。",
                raw_metrics={"terrain_loss_raw": float(terrain_loss), "thermo_loss_raw": self.penalty_loss},
            )
            self._record(result, evaluation_id=evaluation_id)
            return result

        eval_dir = self.context.root / "pecube" / "evaluations" / f"eval_{evaluation_id:05d}_pid_{os.getpid()}"
        try:
            generated_dem_pecube = self._to_pecube_grid(generated_dem)
            uplift_pecube = self._to_pecube_grid(uplift)
            pecube_topographies = self._prepare_topography_series(
                generated_dem_pecube=generated_dem_pecube,
                topography_series=topography_series,
            )
            pecube_uplifts = [uplift_pecube for _ in pecube_topographies]
            pecube_temperatures = [np.zeros_like(pecube_topographies[0]) for _ in pecube_topographies]
            result = self.engine.run(
                topography_series=pecube_topographies,
                uplift_series=pecube_uplifts,
                temperature_series=pecube_temperatures,
                sample_observations=self.engine.config.sample_observations,
                output_dir=eval_dir,
            )
            predictions = predictions_from_parsed(result.parsed, self.observations)
            thermo_loss_raw = normalized_rmse([p.normalized_residual for p in predictions])
            terrain_loss_norm = normalize_unit_loss(terrain_loss)
            thermo_loss = normalize_unit_loss(thermo_loss_raw / self.thermo_loss_scale)
            total = weighted_unit_loss(terrain_loss_norm, thermo_loss, self.terrain_weight, self.thermo_weight)
            constraint = PecubeConstraintResult(
                enabled=True,
                terrain_loss=terrain_loss_norm,
                thermo_loss=thermo_loss,
                total_loss=total,
                n_observations=len(predictions),
                message="Pecube 热年代学约束评价完成。",
                predictions=predictions,
                project_dir=result.project.project_dir,
                raw_metrics={
                    **{k: v for k, v in result.metrics.items() if k != "thermochronology_loss"},
                    "terrain_loss_raw": float(terrain_loss),
                    "thermo_loss_raw": thermo_loss_raw,
                    "thermo_loss_scale": self.thermo_loss_scale,
                },
            )
            self._record(constraint, evaluation_id=evaluation_id)
            return constraint
        except Exception as exc:
            failure_path = eval_dir / "pecube_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps({"error": str(exc), "evaluation": evaluation_id}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.context.add_artifact(failure_path)
            if self.fail_strategy == "raise":
                raise RuntimeError(f"Pecube 热年代学约束评价失败: {exc}") from exc
            logging.warning("Pecube 热年代学约束评价失败，使用 penalty_loss=%s: %s", self.penalty_loss, exc)
            terrain_loss_norm = normalize_unit_loss(terrain_loss)
            thermo_loss = 1.0
            total = weighted_unit_loss(terrain_loss_norm, thermo_loss, self.terrain_weight, self.thermo_weight)
            constraint = PecubeConstraintResult(
                enabled=True,
                terrain_loss=terrain_loss_norm,
                thermo_loss=thermo_loss,
                total_loss=total,
                n_observations=0,
                message=f"Pecube 评价失败，已使用惩罚项: {exc}",
                raw_metrics={"terrain_loss_raw": float(terrain_loss), "thermo_loss_raw": self.penalty_loss},
            )
            self._record(constraint, evaluation_id=evaluation_id)
            return constraint

    def save_best_outputs(
        self,
        *,
        generated_dem: np.ndarray | None = None,
        uplift: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"pecube_enabled": False}
        self._load_history_from_disk()
        history_path = self.context.tables_dir / "pecube_fitness_history.csv"
        _write_csv(history_path, self.history)
        self.context.add_artifact(history_path)
        if self.best_result is None:
            return {"pecube_enabled": True, "pecube_message": "没有成功记录 Pecube 评价。"}

        prediction_path = self.context.tables_dir / "predicted_thermochronology.csv"
        write_prediction_table(prediction_path, self.best_result.predictions)
        self.context.add_artifact(prediction_path)

        if self.best_result.project_dir:
            best_dir = self.context.root / "pecube" / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(self.best_result.project_dir.parent, best_dir)
            self.context.add_artifact(best_dir)

        age_fit_path = self.context.figure_path("pecube_observed_vs_predicted_ages.png")
        plot_age_fit(self.best_result.predictions, age_fit_path)
        self.context.add_artifact(age_fit_path)
        residual_path = self.context.figure_path("pecube_residual_spatial_map.png")
        plot_residual_map(self.best_result.predictions, residual_path)
        self.context.add_artifact(residual_path)

        age_elevation_path = self.context.figure_path("pecube_age_elevation.png")
        plot_age_elevation(self.best_result.predictions, age_elevation_path)
        self.context.add_artifact(age_elevation_path)

        age_surface_path = self.context.figure_path("pecube_age_surface_map.png")
        generated_dem_pecube = self._to_pecube_grid(generated_dem) if generated_dem is not None else self.target_dem_pecube
        uplift_pecube = self._to_pecube_grid(uplift) if uplift is not None else None
        plot_age_surface_map(
            self.best_result.predictions,
            age_surface_path,
            terrain=generated_dem_pecube,
            lon0=self.engine.config.lon0,
            lat0=self.engine.config.lat0,
            dlon=self.engine.config.dlon,
            dlat=self.engine.config.dlat,
        )
        self.context.add_artifact(age_surface_path)

        loss_history_path = self.context.figure_path("pecube_loss_history.png")
        plot_pecube_loss_history(self.history, loss_history_path)
        self.context.add_artifact(loss_history_path)

        dashboard_path = self.context.figure_path("pecube_dashboard.png")
        plot_pecube_dashboard(
            predictions=self.best_result.predictions,
            history=self.history,
            target_dem=self.target_dem_pecube,
            generated_dem=generated_dem_pecube if generated_dem is not None else None,
            uplift=uplift_pecube,
            path=dashboard_path,
        )
        self.context.add_artifact(dashboard_path)

        metrics = self.best_result.metrics()
        metrics["pecube_enabled"] = True
        metrics["pecube_history_rows"] = len(self.history)
        metrics["best_verified_pecube_total_loss"] = self.best_result.total_loss
        metrics["best_verified_pecube_project_dir"] = str(self.best_result.project_dir or "")
        return metrics

    def _next_evaluation_id(self) -> int:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        counter_path = self.history_dir / "counter.txt"
        lock_dir = self.history_dir / ".counter.lock"
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                time.sleep(0.05)
        try:
            if counter_path.exists():
                value = int(counter_path.read_text(encoding="utf-8").strip() or "0")
            else:
                value = 0
            value += 1
            counter_path.write_text(str(value), encoding="utf-8")
            self.evaluation_count = value
            return value
        finally:
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    def _to_pecube_grid(self, array: np.ndarray) -> np.ndarray:
        if self.spatial_adapter is None:
            return np.asarray(array, dtype=float)
        return self.spatial_adapter.transform_array(np.asarray(array, dtype=float))

    def _prepare_topography_series(
        self,
        *,
        generated_dem_pecube: np.ndarray,
        topography_series: np.ndarray | list[np.ndarray] | None,
    ) -> list[np.ndarray]:
        if topography_series is None:
            return [self.target_dem_pecube, generated_dem_pecube]
        series = [self._to_pecube_grid(array) for array in np.asarray(topography_series, dtype=float)]
        if len(series) < 2:
            raise ValueError("Pecube topography_series 至少需要两帧。")
        series[-1] = generated_dem_pecube
        return series

    def _record(self, result: PecubeConstraintResult, *, evaluation_id: int) -> None:
        row = {
            "evaluation": evaluation_id,
            "terrain_loss": result.terrain_loss,
            "thermo_loss": result.thermo_loss,
            "total_loss": result.total_loss,
            "terrain_loss_raw": result.raw_metrics.get("terrain_loss_raw", result.terrain_loss),
            "thermo_loss_raw": result.raw_metrics.get("thermo_loss_raw", result.thermo_loss),
            "n_observations": result.n_observations,
            "message": result.message,
            "project_dir": str(result.project_dir or ""),
        }
        self.history.append(row)
        self._write_result_record(result, row)
        if result.thermo_loss is None or not result.predictions:
            return
        if self.best_result is None or result.total_loss < self.best_result.total_loss:
            self.best_result = result

    def _write_result_record(self, result: PecubeConstraintResult, row: dict[str, Any]) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "row": row,
            "predictions": [prediction_to_row(item) for item in result.predictions],
            "project_dir": str(result.project_dir or ""),
            "raw_metrics": result.raw_metrics,
        }
        record_path = self.history_dir / f"eval_{int(row['evaluation']):05d}_{uuid.uuid4().hex}.json"
        record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_history_from_disk(self) -> None:
        if not self.history_dir.exists():
            return
        rows: list[dict[str, Any]] = []
        best: PecubeConstraintResult | None = None
        for path in sorted(self.history_dir.glob("eval_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = payload["row"]
                predictions = [prediction_from_row(item) for item in payload.get("predictions", [])]
                result = PecubeConstraintResult(
                    enabled=True,
                    terrain_loss=float(row["terrain_loss"]),
                    thermo_loss=None if row.get("thermo_loss") in {"", None} else float(row["thermo_loss"]),
                    total_loss=float(row["total_loss"]),
                    n_observations=int(row.get("n_observations") or len(predictions)),
                    message=str(row.get("message", "")),
                    predictions=predictions,
                    project_dir=Path(payload["project_dir"]) if payload.get("project_dir") else None,
                    raw_metrics=payload.get("raw_metrics", {}),
                )
            except Exception as exc:
                logging.warning("读取 Pecube fitness history 失败，跳过 %s: %s", path, exc)
                continue
            rows.append(row)
            if result.thermo_loss is not None and result.predictions:
                if best is None or result.total_loss < best.total_loss:
                    best = result
        if rows:
            self.history = sorted(rows, key=lambda item: int(item["evaluation"]))
        if best is not None:
            self.best_result = best

    def _enabled(self) -> bool:
        if "Pecube" not in self.config:
            return False
        raw = self.config.get("Pecube", "enabled", fallback="auto").strip().lower()
        if raw in {"auto", "default"}:
            return self.config.get("Pecube", "sample_observations", fallback="none").strip().lower() not in {
                "",
                "none",
                "null",
                "skip",
                "false",
                "0",
            }
        return self.config.getboolean("Pecube", "enabled", fallback=False)

    def _float(self, section: str, key: str, default: float) -> float:
        return self.config.getfloat(section, key, fallback=default) if section in self.config else default

    def _int(self, section: str, key: str, default: int) -> int:
        return self.config.getint(section, key, fallback=default) if section in self.config else default

    def _str(self, section: str, key: str, default: str) -> str:
        return self.config.get(section, key, fallback=default) if section in self.config else default


def load_observations(
    path: Path,
    dem_shape: tuple[int, int],
    *,
    coordinate_system: str = "pecube_output",
    dlon: float = 1.0,
    dlat: float = 1.0,
    lon0: float = 0.0,
    lat0: float = 0.0,
    dem_crs: str | None = None,
    dem_transform: tuple[float, float, float, float, float, float] | None = None,
    source_dem_shape: tuple[int, int] | None = None,
    observation_crs: str = "auto",
) -> list[ThermochronologyObservation]:
    if not Path(path).exists():
        raise FileNotFoundError(f"观测样品 CSV 不存在: {path}")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"观测样品 CSV 没有表头: {path}")
        missing = [name for name in REQUIRED_OBSERVATION_COLUMNS if name not in reader.fieldnames]
        has_coordinate_columns = any(all(name in reader.fieldnames for name in names) for names in COORDINATE_COLUMN_SETS)
        if missing:
            raise ValueError(f"观测样品 CSV 缺少列: {', '.join(missing)}。需要列: {', '.join(REQUIRED_OBSERVATION_COLUMNS)}")
        if not has_coordinate_columns:
            raise ValueError("观测样品 CSV 需要坐标列：x,y 或 lon,lat 或 longitude,latitude。")
        observations = []
        for line_number, row in enumerate(reader, start=2):
            observations.append(
                _parse_observation(
                    row,
                    line_number,
                    dem_shape,
                    coordinate_system=coordinate_system,
                    dlon=dlon,
                    dlat=dlat,
                    lon0=lon0,
                    lat0=lat0,
                    dem_crs=dem_crs,
                    dem_transform=dem_transform,
                    source_dem_shape=source_dem_shape,
                    observation_crs=observation_crs,
                )
            )
    if not observations:
        raise ValueError(f"观测样品 CSV 没有样品行: {path}")
    return observations


def _parse_observation(
    row: dict[str, str],
    line_number: int,
    dem_shape: tuple[int, int],
    *,
    coordinate_system: str,
    dlon: float,
    dlat: float,
    lon0: float,
    lat0: float,
    dem_crs: str | None,
    dem_transform: tuple[float, float, float, float, float, float] | None,
    source_dem_shape: tuple[int, int] | None,
    observation_crs: str,
) -> ThermochronologyObservation:
    sample_id = row["sample_id"].strip()
    system = row["system"].strip()
    if not sample_id:
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 sample_id 为空。")
    if _system_column(system) is None:
        supported = ", ".join(sorted(SYSTEM_TO_PECUBE_COLUMN))
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 system={system!r} 不支持。支持: {supported}")
    x, y, source_columns = _parse_coordinate_pair(row, line_number)
    elevation = _parse_float(row, "elevation", line_number)
    observed_age = _parse_float(row, "observed_age", line_number)
    sigma = _parse_float(row, "sigma", line_number)
    rows, cols = dem_shape
    if coordinate_system in {
        "pecube_output",
        "pecube",
        "lonlat",
        "geographic",
        "lon_lat",
        "longitude_latitude",
        "projected",
        "dem_crs",
    }:
        x, y = _coordinates_to_pecube(
            x,
            y,
            coordinate_system=coordinate_system,
            source_columns=source_columns,
            dem_crs=dem_crs,
            observation_crs=observation_crs,
        )
        min_x = float(lon0)
        min_y = float(lat0)
        max_x = float(lon0) + (cols - 1) * float(dlon)
        max_y = float(lat0) + (rows - 1) * float(dlat)
        if not (_between(x, min_x, max_x) and _between(y, min_y, max_y)):
            raise ValueError(
                f"观测样品 CSV 第 {line_number} 行坐标越界或单位不匹配: "
                f"x={x}, y={y}，当前 observation_coordinate_system=pecube_output，"
                f"允许范围 x=[{min_x},{max_x}], y=[{min_y},{max_y}]。"
            )
    elif coordinate_system in {"grid", "grid_index", "index"}:
        if dem_transform is not None and dem_crs:
            source_rows, source_cols = source_dem_shape or dem_shape
            if not (0 <= x <= source_cols - 1 and 0 <= y <= source_rows - 1):
                raise ValueError(
                    f"观测样品 CSV 第 {line_number} 行 DEM 网格坐标越界: "
                    f"x={x}, y={y}, source DEM shape={(source_rows, source_cols)}"
                )
            dem_x, dem_y = Affine(*dem_transform) * (x + 0.5, y + 0.5)
            x, y = _coordinates_to_pecube(
                float(dem_x),
                float(dem_y),
                coordinate_system="dem_crs",
                source_columns=source_columns,
                dem_crs=dem_crs,
                observation_crs=observation_crs,
            )
            min_x = float(lon0)
            min_y = float(lat0)
            max_x = float(lon0) + (cols - 1) * float(dlon)
            max_y = float(lat0) + (rows - 1) * float(dlat)
            if not (_between(x, min_x, max_x) and _between(y, min_y, max_y)):
                raise ValueError(
                    f"观测样品 CSV 第 {line_number} 行 DEM 网格坐标转换后越界: "
                    f"x={x}, y={y}，允许范围 x=[{min_x},{max_x}], y=[{min_y},{max_y}]。"
                )
        else:
            if not (0 <= x <= cols - 1 and 0 <= y <= rows - 1):
                raise ValueError(f"观测样品 CSV 第 {line_number} 行网格坐标越界: x={x}, y={y}, DEM shape={dem_shape}")
            x = float(lon0) + x * float(dlon)
            y = float(lat0) + y * float(dlat)
    else:
        raise ValueError(
            "不支持的 [Pecube] observation_coordinate_system="
            f"{coordinate_system!r}。可选: pecube_output 或 grid_index。"
        )
    if sigma <= 0:
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 sigma 必须大于 0。")
    return ThermochronologyObservation(sample_id, x, y, elevation, system, observed_age, sigma)


def _parse_coordinate_pair(row: dict[str, str], line_number: int) -> tuple[float, float, tuple[str, str]]:
    for x_key, y_key in COORDINATE_COLUMN_SETS:
        if x_key in row and y_key in row and row[x_key].strip() != "" and row[y_key].strip() != "":
            return _parse_float(row, x_key, line_number), _parse_float(row, y_key, line_number), (x_key, y_key)
    raise ValueError(f"观测样品 CSV 第 {line_number} 行缺少可用坐标。需要 x,y 或 lon,lat 或 longitude,latitude。")


def _coordinates_to_pecube(
    x: float,
    y: float,
    *,
    coordinate_system: str,
    source_columns: tuple[str, str],
    dem_crs: str | None,
    observation_crs: str,
) -> tuple[float, float]:
    if coordinate_system in {"pecube_output", "pecube", "lonlat", "geographic", "lon_lat", "longitude_latitude"}:
        return x, y
    if coordinate_system in {"projected", "dem_crs"}:
        if not dem_crs:
            raise ValueError("样品坐标声明为 DEM 投影坐标，但 DEM 缺少 CRS，无法转换到 Pecube 经纬度。")
        transformer = Transformer.from_crs(CRS.from_string(str(dem_crs)), "EPSG:4326", always_xy=True)
        return transformer.transform(x, y)
    if observation_crs.strip().lower() not in {"", "auto", "none", "epsg:4326", "4326"}:
        transformer = Transformer.from_crs(CRS.from_string(observation_crs), "EPSG:4326", always_xy=True)
        return transformer.transform(x, y)
    return x, y


def _between(value: float, left: float, right: float) -> bool:
    low = min(left, right)
    high = max(left, right)
    tolerance = max(abs(high - low) * 1e-9, 1e-12)
    return low - tolerance <= value <= high + tolerance


def pecube_spatial_adapter_from_dem_profile(profile: dict[str, Any], shape: tuple[int, int]) -> PecubeSpatialAdapter | None:
    """Derive a true regular geographic Pecube grid from a DEM profile."""
    transform = profile.get("transform")
    crs = profile.get("crs")
    if transform is None or crs is None:
        return None
    affine = Affine(*transform) if not isinstance(transform, Affine) else transform
    rows, cols = shape
    if rows < 2 or cols < 2:
        raise ValueError("DEM 网格太小，无法推导 Pecube 经纬度步长。")

    crs_obj = CRS.from_user_input(crs)
    if crs_obj.to_epsg() == 4326:
        target_transform = affine
        target_shape = (rows, cols)
        resample = False
    else:
        left, bottom, right, top = transform_bounds(
            crs_obj,
            "EPSG:4326",
            *_affine_outer_bounds(rows, cols, affine),
            densify_pts=21,
        )
        target_transform = from_bounds(left, bottom, right, top, cols, rows)
        target_shape = (rows, cols)
        resample = True

    grid = _grid_from_geographic_transform(
        target_transform,
        target_shape,
        source="dem_profile_reprojected" if resample else "dem_profile",
    )
    return PecubeSpatialAdapter(
        grid=grid,
        source_crs=crs_obj.to_string(),
        source_transform=tuple(float(value) for value in tuple(affine)[:6]),
        source_shape=(rows, cols),
        target_transform=tuple(float(value) for value in tuple(target_transform)[:6]),
        target_shape=target_shape,
        resample=resample,
    )


def _affine_outer_bounds(rows: int, cols: int, transform: Affine) -> tuple[float, float, float, float]:
    corners = [
        transform * (0, 0),
        transform * (cols, 0),
        transform * (0, rows),
        transform * (cols, rows),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def pecube_grid_from_dem_profile(profile: dict[str, Any], shape: tuple[int, int]) -> PecubeSpatialGrid | None:
    """Derive the Pecube lon/lat grid from a DEM profile."""
    adapter = pecube_spatial_adapter_from_dem_profile(profile, shape)
    return adapter.grid if adapter is not None else None


def _grid_from_geographic_transform(transform: Affine, shape: tuple[int, int], *, source: str) -> PecubeSpatialGrid:
    rows, cols = shape
    if rows < 2 or cols < 2:
        raise ValueError("Pecube 经纬度网格太小，无法推导步长。")
    left_center, top_center = transform * (0.5, 0.5)
    right_center, _ = transform * (cols - 0.5, 0.5)
    bottom_left_center, bottom_lat = transform * (0.5, rows - 0.5)

    dlon = (right_center - left_center) / (cols - 1)
    dlat = (top_center - bottom_lat) / (rows - 1)
    return PecubeSpatialGrid(
        lon0=float(bottom_left_center),
        lat0=float(bottom_lat),
        dlon=float(dlon),
        dlat=float(dlat),
        crs="EPSG:4326",
        source=source,
    )


def pecube_grid_from_observation_bounds(path: Path, shape: tuple[int, int]) -> PecubeSpatialGrid:
    """Build a synthetic Pecube grid that covers lon/lat sample observations."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"观测样品 CSV 没有表头: {path}")
        xs: list[float] = []
        ys: list[float] = []
        for line_number, row in enumerate(reader, start=2):
            x, y, _ = _parse_coordinate_pair(row, line_number)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        raise ValueError(f"观测样品 CSV 没有样品行: {path}")
    rows, cols = shape
    x_span = max(max(xs) - min(xs), 1e-6)
    y_span = max(max(ys) - min(ys), 1e-6)
    x_pad = x_span * 0.12
    y_pad = y_span * 0.12
    lon0 = min(xs) - x_pad
    lat0 = min(ys) - y_pad
    dlon = (x_span + 2 * x_pad) / max(cols - 1, 1)
    dlat = (y_span + 2 * y_pad) / max(rows - 1, 1)
    return PecubeSpatialGrid(
        lon0=float(lon0),
        lat0=float(lat0),
        dlon=float(dlon),
        dlat=float(dlat),
        crs="EPSG:4326",
        source="observation_bounds",
    )


def _parse_float(row: dict[str, str], key: str, line_number: int) -> float:
    raw = row[key].strip()
    if raw == "":
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 {key} 为空。")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 {key}={raw!r} 不是数字。") from exc
    if not math.isfinite(value):
        raise ValueError(f"观测样品 CSV 第 {line_number} 行 {key} 不是有限数。")
    return value


def predictions_from_parsed(
    parsed: PecubeParsedOutput,
    observations: list[ThermochronologyObservation],
) -> list[ThermochronologyPrediction]:
    rows, source_file = _find_prediction_rows(parsed)
    predictions = []
    for observation in observations:
        column = _system_column(observation.system)
        if column is None:
            raise ValueError(f"不支持的热年代学体系: {observation.system}")
        nearest = _nearest_prediction_row(rows, observation.x, observation.y)
        if column not in nearest:
            raise ValueError(f"Pecube 输出缺少预测列 {column}，无法匹配样品 {observation.sample_id}。")
        predicted_age = _float_value(nearest[column], f"Pecube 输出列 {column}")
        residual = predicted_age - observation.observed_age
        predictions.append(
            ThermochronologyPrediction(
                sample_id=observation.sample_id,
                x=observation.x,
                y=observation.y,
                elevation=observation.elevation,
                system=observation.system,
                observed_age=observation.observed_age,
                predicted_age=predicted_age,
                sigma=observation.sigma,
                residual=residual,
                normalized_residual=residual / observation.sigma,
                pecube_column=column,
                source_file=source_file,
            )
        )
    return predictions


def _find_prediction_rows(parsed: PecubeParsedOutput) -> tuple[list[dict[str, str]], str]:
    for name, rows in sorted(parsed.csv_files.items()):
        if rows and {"Longitude", "Latitude", "Height"}.issubset(rows[0]):
            return rows, name
    raise ValueError("Pecube 输出中没有找到包含 Longitude/Latitude/Height 的年龄预测 CSV。")


def _nearest_prediction_row(rows: list[dict[str, str]], x: float, y: float) -> dict[str, str]:
    best_row = None
    best_distance = float("inf")
    for row in rows:
        lon = _float_value(row.get("Longitude", ""), "Longitude")
        lat = _float_value(row.get("Latitude", ""), "Latitude")
        distance = (lon - x) ** 2 + (lat - y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_row = row
    if best_row is None:
        raise ValueError("Pecube 年龄预测 CSV 为空。")
    return best_row


def _system_column(system: str) -> str | None:
    key = system.strip().lower().replace("-", "").replace("_", "")
    return SYSTEM_TO_PECUBE_COLUMN.get(key) or SYSTEM_TO_PECUBE_COLUMN.get(system.strip().lower())


def _float_value(raw: Any, label: str) -> float:
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{label}={raw!r} 不是数字。") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} 不是有限数。")
    return value


def normalized_rmse(values: list[float]) -> float:
    if not values:
        raise ValueError("没有可用于计算 thermo_loss 的热年代学预测。")
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def normalize_unit_loss(value: float) -> float:
    """Clamp any finite loss value into the 0-1 range used by GA fitness."""
    value = float(value)
    if not math.isfinite(value):
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


def weighted_unit_loss(terrain_loss: float, thermo_loss: float, terrain_weight: float, thermo_weight: float) -> float:
    """Combine two normalized 0-1 losses as a weighted average."""
    terrain = normalize_unit_loss(terrain_loss)
    thermo = normalize_unit_loss(thermo_loss)
    terrain_w = max(float(terrain_weight), 0.0)
    thermo_w = max(float(thermo_weight), 0.0)
    weight_sum = terrain_w + thermo_w
    if weight_sum <= 0:
        return terrain
    return float((terrain_w * terrain + thermo_w * thermo) / weight_sum)


def write_prediction_table(path: Path, predictions: list[ThermochronologyPrediction]) -> None:
    rows = [prediction_to_row(p) for p in predictions]
    _write_csv(path, rows)


def prediction_to_row(p: ThermochronologyPrediction) -> dict[str, Any]:
    return {
        "sample_id": p.sample_id,
        "x": p.x,
        "y": p.y,
        "elevation": p.elevation,
        "system": p.system,
        "observed_age": p.observed_age,
        "predicted_age": p.predicted_age,
        "sigma": p.sigma,
        "residual": p.residual,
        "normalized_residual": p.normalized_residual,
        "pecube_column": p.pecube_column,
        "source_file": p.source_file,
    }


def prediction_from_row(row: dict[str, Any]) -> ThermochronologyPrediction:
    return ThermochronologyPrediction(
        sample_id=str(row["sample_id"]),
        x=float(row["x"]),
        y=float(row["y"]),
        elevation=float(row["elevation"]),
        system=str(row["system"]),
        observed_age=float(row["observed_age"]),
        predicted_age=float(row["predicted_age"]),
        sigma=float(row["sigma"]),
        residual=float(row["residual"]),
        normalized_residual=float(row["normalized_residual"]),
        pecube_column=str(row["pecube_column"]),
        source_file=str(row["source_file"]),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_age_fit(predictions: list[ThermochronologyPrediction], path: Path) -> None:
    observed = np.asarray([p.observed_age for p in predictions], dtype=float)
    predicted = np.asarray([p.predicted_age for p in predictions], dtype=float)
    sigma = np.asarray([p.sigma for p in predictions], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    pad = max((high - low) * 0.08, 0.1)
    xs = np.linspace(low - pad, high + pad, 100)
    ax.fill_between(xs, xs - np.mean(sigma), xs + np.mean(sigma), color="0.92", label="mean 1 sigma band")
    for system in _prediction_systems(predictions):
        subset = [p for p in predictions if p.system == system]
        ax.errorbar(
            [p.observed_age for p in subset],
            [p.predicted_age for p in subset],
            xerr=[p.sigma for p in subset],
            fmt=_system_marker(system),
            label=system,
            capsize=3,
            markersize=6,
            markeredgecolor="black",
            linewidth=1,
        )
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Observed age (Ma)")
    ax.set_ylabel("Predicted age (Ma)")
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    nrmse = normalized_rmse([p.normalized_residual for p in predictions])
    ax.set_title("Observed vs predicted ages")
    ax.text(
        0.02,
        0.98,
        f"RMSE={rmse:.3g} Ma\nnRMSE={nrmse:.3g}\nn={len(predictions)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_map(predictions: list[ThermochronologyPrediction], path: Path) -> None:
    residuals = np.asarray([p.residual for p in predictions], dtype=float)
    vmax = max(float(np.max(np.abs(residuals))), 1e-9)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sc = ax.scatter(
        [p.x for p in predictions],
        [p.y for p in predictions],
        c=residuals,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=90,
        edgecolors="black",
    )
    for p in predictions:
        ax.text(p.x + 0.4, p.y + 0.4, p.sample_id, fontsize=8)
    ax.set_xlabel("x index / Pecube longitude")
    ax.set_ylabel("y index / Pecube latitude")
    ax.set_title("Thermochronology residuals")
    fig.colorbar(sc, ax=ax, label="Predicted - observed (Ma)")
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_age_elevation(predictions: list[ThermochronologyPrediction], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for system in _prediction_systems(predictions):
        subset = [p for p in predictions if p.system == system]
        ax.errorbar(
            [p.observed_age for p in subset],
            [p.elevation for p in subset],
            xerr=[p.sigma for p in subset],
            fmt=_system_marker(system),
            color="black",
            markerfacecolor="white",
            label=f"{system} observed",
            capsize=3,
            markersize=6,
        )
        ax.scatter(
            [p.predicted_age for p in subset],
            [p.elevation for p in subset],
            marker=_system_marker(system),
            s=55,
            label=f"{system} predicted",
        )
        for p in subset:
            ax.plot([p.observed_age, p.predicted_age], [p.elevation, p.elevation], color="0.75", linewidth=0.9)
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Elevation")
    ax.set_title("Age-elevation fit")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_age_surface_map(
    predictions: list[ThermochronologyPrediction],
    path: Path,
    *,
    terrain: np.ndarray | None = None,
    lon0: float = 0.0,
    lat0: float = 0.0,
    dlon: float = 1.0,
    dlat: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    if terrain is not None:
        terrain_arr = np.asarray(terrain, dtype=float)
        extent = _terrain_extent(terrain_arr, lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat)
        im = ax.imshow(terrain_arr, cmap="terrain", origin="lower", extent=extent, alpha=0.88)
        fig.colorbar(im, ax=ax, label="Elevation", fraction=0.045, pad=0.03)
    sc = ax.scatter(
        [p.x for p in predictions],
        [p.y for p in predictions],
        c=[p.predicted_age for p in predictions],
        s=58,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.7,
    )
    label_dx = max(float(dlon) * 1.2, 0.002)
    label_dy = max(float(dlat) * 1.2, 0.002)
    for index, p in enumerate(predictions):
        dy = label_dy if index % 2 == 0 else -label_dy
        ax.annotate(
            f"{p.sample_id}\n{p.predicted_age:.2g} Ma",
            xy=(p.x, p.y),
            xytext=(p.x + label_dx, p.y + dy),
            fontsize=7,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.7},
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
    xs = [p.x for p in predictions]
    ys = [p.y for p in predictions]
    if terrain is not None:
        terrain_arr = np.asarray(terrain, dtype=float)
        extent = _terrain_extent(terrain_arr, lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        x_pad = max((max(xs) - min(xs)) * 0.10, float(dlon) * 2)
        y_pad = max((max(ys) - min(ys)) * 0.10, float(dlat) * 2)
        ax.set_xlim(max(0.0, min(xs) - x_pad), max(xs) + x_pad)
        ax.set_ylim(max(0.0, min(ys) - y_pad), max(ys) + y_pad)
    ax.set_xlabel("x / Pecube longitude")
    ax.set_ylabel("y / Pecube latitude")
    ax.set_title("Predicted ages on terrain")
    fig.colorbar(sc, ax=ax, label="Predicted age (Ma)", fraction=0.045, pad=0.03)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pecube_loss_history(history: list[dict[str, Any]], path: Path) -> None:
    rows = _numeric_history_rows(history)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    if rows:
        evaluations = [row["evaluation"] for row in rows]
        total_loss = [row["total_loss"] for row in rows]
        ax.plot(evaluations, total_loss, linewidth=2, color="tab:blue", label="Combined fitness")
        best_index = int(np.argmin(total_loss))
        ax.plot(
            evaluations[best_index],
            total_loss[best_index],
            "ro",
            label=f"Best: {total_loss[best_index]:.4f} at eval {int(evaluations[best_index])}",
        )
        ax_components = ax.twinx()
        ax_components.plot(evaluations, [row["terrain_loss"] for row in rows], "--", color="tab:green", alpha=0.75, label="terrain loss")
        ax_components.plot(evaluations, [row["thermo_loss"] for row in rows], "--", color="tab:orange", alpha=0.75, label="thermo loss")
        ax_components.set_ylabel("Component loss")
        lines, labels = ax.get_legend_handles_labels()
        component_lines, component_labels = ax_components.get_legend_handles_labels()
        ax.legend(lines + component_lines, labels + component_labels, frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Pecube loss history", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Combined fitness")
    ax.set_title("Pecube-coupled optimization history")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pecube_dashboard(
    *,
    predictions: list[ThermochronologyPrediction],
    history: list[dict[str, Any]],
    target_dem: np.ndarray,
    generated_dem: np.ndarray | None,
    uplift: np.ndarray | None,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    _imshow_panel(axes[0, 0], target_dem, "Target terrain", "terrain")
    if generated_dem is not None:
        _imshow_panel(axes[0, 1], generated_dem, "Generated terrain", "terrain")
    else:
        axes[0, 1].axis("off")
        axes[0, 1].set_title("Generated terrain unavailable")
    if uplift is not None:
        _imshow_panel(axes[0, 2], uplift, "Best uplift", "RdBu_r")
    else:
        axes[0, 2].axis("off")
        axes[0, 2].set_title("Uplift unavailable")
    _draw_age_fit_panel(axes[1, 0], predictions)
    _draw_age_elevation_panel(axes[1, 1], predictions)
    _draw_loss_history_panel(axes[1, 2], history)
    fig.suptitle("Pecube-coupled inversion summary", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _prediction_systems(predictions: list[ThermochronologyPrediction]) -> list[str]:
    return sorted({p.system for p in predictions})


def _system_marker(system: str) -> str:
    key = system.strip().lower()
    if key in {"ahe", "heapatite", "apatitehe"}:
        return "o"
    if key in {"zhe", "hezircon", "zirconhe"}:
        return "s"
    if key in {"aft", "ftapatite", "apatiteft"}:
        return "^"
    if key in {"zft", "ftzircon", "zirconft"}:
        return "D"
    return "o"


def _terrain_extent(
    terrain: np.ndarray,
    *,
    lon0: float = 0.0,
    lat0: float = 0.0,
    dlon: float,
    dlat: float,
) -> tuple[float, float, float, float]:
    max_x = float(lon0) + float(terrain.shape[1] - 1) * float(dlon)
    max_y = float(lat0) + float(terrain.shape[0] - 1) * float(dlat)
    return (float(lon0), max_x, float(lat0), max_y)


def _numeric_history_rows(history: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for row in history:
        try:
            rows.append({
                "evaluation": float(row["evaluation"]),
                "terrain_loss": float(row["terrain_loss"]),
                "thermo_loss": float(row["thermo_loss"]),
                "total_loss": float(row["total_loss"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _imshow_panel(ax: plt.Axes, data: np.ndarray, title: str, cmap: str) -> None:
    im = ax.imshow(np.asarray(data, dtype=float), cmap=cmap, origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _draw_age_fit_panel(ax: plt.Axes, predictions: list[ThermochronologyPrediction]) -> None:
    observed = np.asarray([p.observed_age for p in predictions], dtype=float)
    predicted = np.asarray([p.predicted_age for p in predictions], dtype=float)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    pad = max((high - low) * 0.08, 0.1)
    for system in _prediction_systems(predictions):
        subset = [p for p in predictions if p.system == system]
        ax.scatter([p.observed_age for p in subset], [p.predicted_age for p in subset], label=system, edgecolors="black")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Observed age (Ma)")
    ax.set_ylabel("Predicted age (Ma)")
    ax.set_title("Observed vs predicted")
    ax.legend(frameon=False, fontsize=8)


def _draw_age_elevation_panel(ax: plt.Axes, predictions: list[ThermochronologyPrediction]) -> None:
    for system in _prediction_systems(predictions):
        subset = [p for p in predictions if p.system == system]
        ax.scatter([p.observed_age for p in subset], [p.elevation for p in subset], facecolors="white", edgecolors="black", label=f"{system} obs")
        ax.scatter([p.predicted_age for p in subset], [p.elevation for p in subset], label=f"{system} pred")
        for p in subset:
            ax.plot([p.observed_age, p.predicted_age], [p.elevation, p.elevation], color="0.75", linewidth=0.8)
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Elevation")
    ax.set_title("Age-elevation")
    ax.legend(frameon=False, fontsize=7)


def _draw_loss_history_panel(ax: plt.Axes, history: list[dict[str, Any]]) -> None:
    rows = _numeric_history_rows(history)
    if not rows:
        ax.text(0.5, 0.5, "No loss history", ha="center", va="center", transform=ax.transAxes)
        return
    evaluations = [row["evaluation"] for row in rows]
    total_loss = [row["total_loss"] for row in rows]
    ax.plot(evaluations, total_loss, linewidth=2, color="tab:blue", label="combined")
    best_index = int(np.argmin(total_loss))
    ax.plot(evaluations[best_index], total_loss[best_index], "ro", markersize=5)
    ax_components = ax.twinx()
    ax_components.plot(evaluations, [row["terrain_loss"] for row in rows], "--", color="tab:green", alpha=0.75, label="terrain")
    ax_components.plot(evaluations, [row["thermo_loss"] for row in rows], "--", color="tab:orange", alpha=0.75, label="thermo")
    ax_components.set_ylabel("Components", fontsize=8)
    ax_components.tick_params(labelsize=8)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Combined fitness")
    ax.set_title("Optimization history")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    component_lines, component_labels = ax_components.get_legend_handles_labels()
    ax.legend(lines + component_lines, labels + component_labels, frameon=False, fontsize=8)
