"""读取并校验 forward_config.ini。

这一层只做配置文件 → 内部数据类的映射。FastScape 调用、剥蚀计算、绘图都在
其他模块。所有错误信息都用中文，方便非代码用户排查。
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_VALID_BOUNDARY = {"fixed_value", "core", "looped"}
_VALID_UPLIFT_MODE = {"none", "stages", "python", "array"}
_VALID_RAINFALL_MODE = {"uniform", "python"}


@dataclass
class UpliftTimeConfig:
    """uplift 在时间上的处理方式。"""

    mode: str = "none"
    module_path: str | None = None
    function_name: str = "uplift_time"
    params: dict[str, Any] = field(default_factory=dict)
    stage_times_ma: list[float] | None = None
    stage_multipliers: list[float] | None = None
    array_path: str | None = None
    array_times_years: list[float] | None = None


@dataclass
class RainfallConfig:
    mode: str = "uniform"
    value: float = 1.0
    module_path: str | None = None
    function_name: str = "rainfall"
    min_value: float | None = None
    max_value: float | None = None
    extra_params: dict[str, str] = field(default_factory=dict)


@dataclass
class ForwardConfig:
    """forward_config.ini 的完整内部表示。"""

    config_path: Path
    output_root: Path

    dem_path: Path
    uplift_base_path: Path | None
    uplift_value: float
    ksp_value: float
    ksp_path: Path | None

    time_total: float
    spacing: float
    output_steps: int
    boundary: list[str]
    area_exp: float
    slope_exp: float
    k_diff: float

    uplift_time: UpliftTimeConfig
    rainfall: RainfallConfig

    save_cumulative_erosion: bool
    save_mean_erosion_rate: bool
    save_net_uplift: bool
    save_topography_series: bool
    save_uplift_series: bool

    plot_history_grid: bool
    plot_erosion_history: bool

    raw_config: configparser.ConfigParser


def load_forward_config(config_path: str | Path) -> ForwardConfig:
    """读取 forward_config.ini 并返回 ForwardConfig。"""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"未找到正演配置文件: {config_path}")

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(config_path, encoding="utf-8")
    base_dir = config_path.parent

    output_root = _resolve_path(parser.get("Run", "output_root", fallback="./outputs/forward"), base_dir)

    dem_path = _required_path(parser, "Data", "dem_path", base_dir, label="DEM")
    uplift_base_path = _optional_path(parser, "Data", "uplift_base_path", base_dir)
    uplift_value = parser.getfloat("Data", "uplift_value", fallback=1.0)
    ksp_value = parser.getfloat("Data", "ksp_value", fallback=1.0e-5)
    ksp_path = _optional_path(parser, "Data", "ksp_path", base_dir)

    time_total = parser.getfloat("Model", "time_total", fallback=2.0e6)
    spacing = parser.getfloat("Model", "spacing", fallback=90.0)
    output_steps = parser.getint("Model", "output_steps", fallback=21)
    boundary = _read_boundary(parser)
    area_exp = parser.getfloat("Model", "area_exp", fallback=0.43)
    slope_exp = parser.getfloat("Model", "slope_exp", fallback=1.0)
    k_diff = parser.getfloat("Model", "k_diff", fallback=1.0)

    uplift_time = _read_uplift_time(parser, base_dir)
    rainfall = _read_rainfall(parser, base_dir)

    save_cum = parser.getboolean("Output", "save_cumulative_erosion", fallback=True)
    save_mean = parser.getboolean("Output", "save_mean_erosion_rate", fallback=True)
    save_net = parser.getboolean("Output", "save_net_uplift", fallback=True)
    save_topo = parser.getboolean("Output", "save_topography_series", fallback=True)
    save_uplift = parser.getboolean("Output", "save_uplift_series", fallback=True)
    plot_grid = parser.getboolean("Output", "plot_history_grid", fallback=True)
    plot_history = parser.getboolean("Output", "plot_erosion_history", fallback=True)

    if time_total <= 0:
        raise ValueError("[Model] time_total 必须为正数。")
    if spacing <= 0:
        raise ValueError("[Model] spacing 必须为正数。")
    if output_steps < 2:
        raise ValueError("[Model] output_steps 至少为 2。")
    if uplift_value <= 0 and uplift_base_path is None:
        raise ValueError("[Data] uplift_value 必须为正数，或填写 uplift_base_path。")
    if ksp_value <= 0 and ksp_path is None:
        raise ValueError("[Data] ksp_value 必须为正数，或填写 ksp_path。")

    return ForwardConfig(
        config_path=config_path,
        output_root=output_root,
        dem_path=dem_path,
        uplift_base_path=uplift_base_path,
        uplift_value=float(uplift_value),
        ksp_value=float(ksp_value),
        ksp_path=ksp_path,
        time_total=float(time_total),
        spacing=float(spacing),
        output_steps=int(output_steps),
        boundary=boundary,
        area_exp=float(area_exp),
        slope_exp=float(slope_exp),
        k_diff=float(k_diff),
        uplift_time=uplift_time,
        rainfall=rainfall,
        save_cumulative_erosion=save_cum,
        save_mean_erosion_rate=save_mean,
        save_net_uplift=save_net,
        save_topography_series=save_topo,
        save_uplift_series=save_uplift,
        plot_history_grid=plot_grid,
        plot_erosion_history=plot_history,
        raw_config=parser,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    text = (value or "").strip()
    if not text or text.lower() == "none":
        raise ValueError("路径不能为空。")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _required_path(parser: configparser.ConfigParser, section: str, key: str, base_dir: Path, *, label: str) -> Path:
    value = parser.get(section, key, fallback="").strip()
    if not value or value.lower() == "none":
        raise ValueError(f"[{section}] {key} 必填（{label}）。")
    return _resolve_path(value, base_dir)


def _optional_path(parser: configparser.ConfigParser, section: str, key: str, base_dir: Path) -> Path | None:
    value = parser.get(section, key, fallback="").strip()
    if not value or value.lower() == "none":
        return None
    return _resolve_path(value, base_dir)


def _read_boundary(parser: configparser.ConfigParser) -> list[str]:
    edges = []
    for key, default in (
        ("boundary_left", "fixed_value"),
        ("boundary_right", "fixed_value"),
        ("boundary_top", "fixed_value"),
        ("boundary_bottom", "core"),
    ):
        value = parser.get("Model", key, fallback=default).strip()
        if value not in _VALID_BOUNDARY:
            raise ValueError(
                f"[Model] {key}={value!r} 不合法，可选: {sorted(_VALID_BOUNDARY)}"
            )
        edges.append(value)
    if "fixed_value" not in edges:
        raise ValueError("FastScape 至少需要一条 fixed_value 边界作为数值约束。")
    return edges


def _read_uplift_time(parser: configparser.ConfigParser, base_dir: Path) -> UpliftTimeConfig:
    if not parser.has_section("UpliftTime"):
        return UpliftTimeConfig(mode="none")

    mode = parser.get("UpliftTime", "mode", fallback="none").strip().lower()
    if mode not in _VALID_UPLIFT_MODE:
        raise ValueError(
            f"[UpliftTime] mode={mode!r} 不合法，可选: {sorted(_VALID_UPLIFT_MODE)}"
        )
    if mode == "none":
        return UpliftTimeConfig(mode="none")

    if mode == "python":
        module_value = parser.get("UpliftTime", "module_path", fallback="./uplift_time.py").strip()
        module_path = str(_resolve_path(module_value, base_dir))
        function_name = parser.get("UpliftTime", "function", fallback="uplift_time").strip() or "uplift_time"
        reserved = {"mode", "module_path", "function", "stage_times_ma", "stage_multipliers",
                    "array_path", "array_times_years"}
        params = {
            key: value
            for key, value in parser["UpliftTime"].items()
            if key not in reserved
        }
        return UpliftTimeConfig(
            mode="python",
            module_path=module_path,
            function_name=function_name,
            params=params,
        )

    if mode == "stages":
        times_raw = parser.get("UpliftTime", "stage_times_ma", fallback="").strip()
        mult_raw = parser.get("UpliftTime", "stage_multipliers", fallback="").strip()
        if not times_raw or not mult_raw:
            raise ValueError("[UpliftTime] mode=stages 必须填写 stage_times_ma 和 stage_multipliers。")
        times = [float(part) for part in times_raw.replace(";", ",").split(",") if part.strip()]
        mults = [float(part) for part in mult_raw.replace(";", ",").split(",") if part.strip()]
        if len(times) < 2:
            raise ValueError("stage_times_ma 至少两个时间点。")
        if len(mults) != len(times) - 1:
            raise ValueError(
                f"stage_multipliers 数量必须等于 stage_times_ma-1: {len(mults)} != {len(times)-1}"
            )
        return UpliftTimeConfig(mode="stages", stage_times_ma=times, stage_multipliers=mults)

    if mode == "array":
        array_value = parser.get("UpliftTime", "array_path", fallback="").strip()
        if not array_value:
            raise ValueError("[UpliftTime] mode=array 必须填写 array_path。")
        array_path = str(_resolve_path(array_value, base_dir))
        times_raw = parser.get("UpliftTime", "array_times_years", fallback="").strip()
        if not times_raw:
            raise ValueError("[UpliftTime] mode=array 必须填写 array_times_years。")
        times = [float(part) for part in times_raw.replace(";", ",").split(",") if part.strip()]
        return UpliftTimeConfig(mode="array", array_path=array_path, array_times_years=times)

    raise ValueError(f"[UpliftTime] mode={mode!r} 解析失败。")  # pragma: no cover


def _read_rainfall(parser: configparser.ConfigParser, base_dir: Path) -> RainfallConfig:
    if not parser.has_section("Rainfall"):
        return RainfallConfig()
    mode = parser.get("Rainfall", "mode", fallback="uniform").strip().lower()
    if mode in {"", "uniform", "constant"}:
        value = parser.getfloat("Rainfall", "value", fallback=1.0)
        if value <= 0:
            raise ValueError("[Rainfall] value 必须为正数。")
        return RainfallConfig(mode="uniform", value=float(value))
    if mode != "python":
        raise ValueError(f"[Rainfall] mode={mode!r} 不支持，可选: uniform / python")
    module_value = parser.get("Rainfall", "module_path", fallback="./rainfall.py").strip()
    module_path = str(_resolve_path(module_value, base_dir))
    function_name = parser.get("Rainfall", "function", fallback="rainfall").strip() or "rainfall"
    min_value = _optional_float(parser.get("Rainfall", "min", fallback=""))
    max_value = _optional_float(parser.get("Rainfall", "max", fallback=""))
    if min_value is not None and min_value <= 0:
        raise ValueError("[Rainfall] min 必须为正数。")
    if max_value is not None and max_value <= 0:
        raise ValueError("[Rainfall] max 必须为正数。")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("[Rainfall] min 不能大于 max。")
    reserved = {"mode", "module_path", "function", "min", "max"}
    extra = {key: value for key, value in parser["Rainfall"].items() if key not in reserved}
    return RainfallConfig(
        mode="python",
        module_path=module_path,
        function_name=function_name,
        min_value=min_value,
        max_value=max_value,
        extra_params=extra,
    )


def _optional_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text or text.lower() == "none":
        return None
    return float(text)
