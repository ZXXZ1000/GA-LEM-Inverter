"""User-configurable rainfall/runoff functions for FastScape."""

from __future__ import annotations

import importlib.util
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


RainfallFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, float, Mapping[str, Any]], np.ndarray]
_RAINFALL_FUNCTION_REGISTRY: dict[str, RainfallFunction] = {}


@dataclass(frozen=True)
class RainfallConfig:
    """A rainfall/runoff model ready to pass into the FastScape runner."""

    mode: str = "uniform"
    factor: float = 1.0
    function: RainfallFunction | None = None
    params: dict[str, Any] | None = None
    source_path: str | None = None
    function_name: str = "rainfall"
    dynamic: bool = True
    min_value: float | None = None
    max_value: float | None = None


def rainfall_from_config(config: Any, *, base_dir: str | Path | None = None) -> RainfallConfig:
    """Load a rainfall model from config.ini.

    ``[Rainfall] mode = uniform`` keeps the existing scalar behavior.
    ``[Rainfall] mode = python`` loads a user-editable Python function with
    signature ``rainfall(x, y, z, t_ma, params)``.
    """
    legacy_factor = config.getfloat("Model", "rainfall_factor", fallback=1.0)
    if not config.has_section("Rainfall"):
        return RainfallConfig(mode="uniform", factor=validate_rainfall_array(legacy_factor, label="rainfall_factor").item())

    mode = config.get("Rainfall", "mode", fallback="uniform").strip().lower()
    if mode in {"", "uniform", "constant"}:
        value = config.getfloat("Rainfall", "value", fallback=legacy_factor)
        return RainfallConfig(mode="uniform", factor=validate_rainfall_array(value, label="rainfall_factor").item())
    if mode != "python":
        raise ValueError(f"[Rainfall] mode={mode!r} 不支持。当前支持: uniform, python")

    module_raw = config.get("Rainfall", "module_path", fallback="./rainfall_model.py").strip()
    function_name = config.get("Rainfall", "function", fallback="rainfall").strip() or "rainfall"
    dynamic = config.getboolean("Rainfall", "dynamic", fallback=True)
    if not dynamic:
        raise ValueError("[Rainfall] mode=python 第一版只支持 dynamic=true。")
    module_path = _resolve_module_path(module_raw, base_dir=base_dir)
    rainfall_func = load_rainfall_function(module_path, function_name)
    params = {
        key: value
        for key, value in config["Rainfall"].items()
        if key not in {"mode", "module_path", "function", "dynamic", "min", "max"}
    }
    min_value = _optional_float(config.get("Rainfall", "min", fallback=""))
    max_value = _optional_float(config.get("Rainfall", "max", fallback=""))
    if min_value is not None and min_value <= 0:
        raise ValueError("[Rainfall] min 必须为正数。")
    if max_value is not None and max_value <= 0:
        raise ValueError("[Rainfall] max 必须为正数。")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("[Rainfall] min 不能大于 max。")

    return RainfallConfig(
        mode="python",
        factor=legacy_factor,
        function=rainfall_func,
        params=params,
        source_path=str(module_path),
        function_name=function_name,
        dynamic=dynamic,
        min_value=min_value,
        max_value=max_value,
    )


def load_rainfall_function(module_path: str | Path, function_name: str = "rainfall") -> RainfallFunction:
    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到降雨模型脚本: {path}")
    if not path.is_file():
        raise ValueError(f"降雨模型路径不是文件: {path}")

    spec = importlib.util.spec_from_file_location(f"ga_lem_user_rainfall_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载降雨模型脚本: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name, None)
    if not callable(func):
        raise AttributeError(f"{path} 中找不到可调用函数 {function_name!r}")
    return func


def register_rainfall_function(func: RainfallFunction) -> str:
    """Register a Python rainfall function and return a serializable key."""
    key = f"rainfall_{uuid.uuid4().hex}"
    _RAINFALL_FUNCTION_REGISTRY[key] = func
    return key


def get_registered_rainfall_function(key: str | None) -> RainfallFunction | None:
    if key is None or not str(key).strip():
        return None
    try:
        return _RAINFALL_FUNCTION_REGISTRY[str(key)]
    except KeyError as exc:
        raise KeyError(f"找不到已注册的降雨函数 key={key!r}。") from exc


def encode_rainfall_params(params: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(params or {}), ensure_ascii=False, sort_keys=True)


def decode_rainfall_params(raw: str | None) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        return {}
    data = json.loads(str(raw))
    if not isinstance(data, dict):
        raise ValueError("rainfall_params_json 必须解码为字典。")
    return data


def evaluate_rainfall(
    rainfall: RainfallConfig | float | int | None,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    elapsed_years: float,
    total_time_years: float,
) -> np.ndarray:
    """Evaluate a rainfall/runoff model to a positive matrix."""
    shape = tuple(np.asarray(z).shape)
    if rainfall is None:
        return np.ones(shape, dtype=float)
    if isinstance(rainfall, (float, int, np.floating, np.integer)):
        return validate_rainfall_array(rainfall, target_shape=shape)

    if rainfall.mode == "uniform" or rainfall.function is None:
        return validate_rainfall_array(rainfall.factor, target_shape=shape)

    t_ma = max(0.0, (float(total_time_years) - float(elapsed_years)) / 1.0e6)
    raw = rainfall.function(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        float(t_ma),
        dict(rainfall.params or {}),
    )
    values = validate_rainfall_array(raw, target_shape=shape)
    if rainfall.min_value is not None or rainfall.max_value is not None:
        values = np.clip(
            values,
            rainfall.min_value if rainfall.min_value is not None else np.nanmin(values),
            rainfall.max_value if rainfall.max_value is not None else np.nanmax(values),
        )
    return validate_rainfall_array(values, target_shape=shape)


def validate_rainfall_array(values: Any, *, target_shape: tuple[int, int] | None = None, label: str = "rainfall") -> np.ndarray:
    """Return a positive finite rainfall array, broadcasting scalars to target_shape."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError(f"{label} 必须为正数，当前为 {values!r}。")
        if target_shape is None:
            return np.asarray(scalar, dtype=float)
        return np.full(target_shape, scalar, dtype=float)
    if target_shape is not None and tuple(array.shape) != tuple(target_shape):
        raise ValueError(f"{label} shape {array.shape} 与目标网格 {target_shape} 不一致。")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{label} 必须全部为正且不能包含 NaN/Inf。")
    return array.astype(float, copy=True)


def preview_rainfall_fields(
    rainfall: RainfallConfig | float | int | None,
    *,
    shape: tuple[int, int],
    spacing: float,
    elevation: np.ndarray,
    total_time_years: float,
    times_ma: list[float] | tuple[float, ...] = (0.0,),
) -> dict[float, np.ndarray]:
    """Evaluate rainfall fields at selected geologic times before present."""
    rows, cols = shape
    elevation_array = np.asarray(elevation, dtype=float)
    if tuple(elevation_array.shape) != tuple(shape):
        raise ValueError(f"rainfall preview elevation shape {elevation_array.shape} 与目标网格 {shape} 不一致。")
    x = np.linspace(0.0, (cols - 1) * float(spacing), cols)
    y = np.linspace(0.0, (rows - 1) * float(spacing), rows)
    x_grid, y_grid = np.meshgrid(x, y)
    fields: dict[float, np.ndarray] = {}
    for time_ma in times_ma:
        elapsed = float(total_time_years) - float(time_ma) * 1.0e6
        fields[float(time_ma)] = evaluate_rainfall(
            rainfall,
            x=x_grid,
            y=y_grid,
            z=elevation_array,
            elapsed_years=elapsed,
            total_time_years=float(total_time_years),
        )
    return fields


def rainfall_metadata(rainfall: RainfallConfig | float | int | None) -> dict[str, Any]:
    if rainfall is None:
        return {"rainfall_mode": "uniform", "rainfall_factor": 1.0}
    if isinstance(rainfall, (float, int, np.floating, np.integer)):
        return {"rainfall_mode": "uniform", "rainfall_factor": float(rainfall)}
    data = {
        "rainfall_mode": rainfall.mode,
        "rainfall_factor": float(rainfall.factor),
        "rainfall_dynamic": bool(rainfall.dynamic),
    }
    if rainfall.source_path:
        data["rainfall_module_path"] = rainfall.source_path
        data["rainfall_function"] = rainfall.function_name
    if rainfall.min_value is not None:
        data["rainfall_min"] = float(rainfall.min_value)
    if rainfall.max_value is not None:
        data["rainfall_max"] = float(rainfall.max_value)
    return data


def _resolve_module_path(path_value: str, *, base_dir: str | Path | None) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path(__file__).resolve().parents[2]
        path = root / path
    return path.resolve()


def _optional_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text or text.lower() in {"none", "null", "skip"}:
        return None
    return float(text)
