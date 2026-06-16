"""Configuration loading and user-facing validation."""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_MODES = {"main", "synthetic", "k_sensitivity", "pecube_coupled"}


class UserConfigError(ValueError):
    """A configuration error that should be shown directly to non-code users."""


@dataclass(frozen=True)
class AppConfig:
    """Loaded application configuration."""

    parser: configparser.ConfigParser
    path: Path
    mode: str


def _copy_legacy_sections(config: configparser.ConfigParser) -> None:
    """Keep old section names working while exposing simpler names to users."""
    aliases = {
        "Data": "Paths",
        "Optimization": "GeneticAlgorithm",
    }
    for new_section, legacy_section in aliases.items():
        if new_section not in config and legacy_section in config:
            config[new_section] = {}
            for key, value in config[legacy_section].items():
                config[new_section][key] = value
        if legacy_section not in config and new_section in config:
            config[legacy_section] = {}
            for key, value in config[new_section].items():
                config[legacy_section][key] = value

    if "Optimization" in config:
        opt = config["Optimization"]
        legacy = config["GeneticAlgorithm"]
        key_aliases = {
            "uplift_min": "lb",
            "uplift_max": "ub",
            "uplift_precision": "precision",
            "population_size": "ga_pop_size",
            "max_iterations": "ga_max_iter",
            "cross_probability": "ga_prob_cross",
            "mutation_probability": "ga_prob_mut",
            "min_population_size": "min_size_pop",
        }
        for simple_key, legacy_key in key_aliases.items():
            if simple_key in opt and legacy_key not in legacy:
                legacy[legacy_key] = opt[simple_key]
            if legacy_key in legacy and simple_key not in opt:
                opt[simple_key] = legacy[legacy_key]

    if "Preprocessing" not in config and "Optimization" in config:
        config["Preprocessing"] = {}
    if "Preprocessing" in config and "Optimization" in config:
        for key in ("scale_factor", "ratio", "target_crs", "smooth_sigma"):
            if key in config["Optimization"] and key not in config["Preprocessing"]:
                config["Preprocessing"][key] = config["Optimization"][key]
            if key in config["Preprocessing"] and key not in config["Optimization"]:
                config["Optimization"][key] = config["Preprocessing"][key]

    if "Fitness" not in config:
        config["Fitness"] = {"use_lpips": "true"}


def _clean_paths(config: configparser.ConfigParser) -> None:
    for section in ("Data", "Paths"):
        if section not in config:
            continue
        for key in config[section]:
            value = config[section][key].split(";")[0].strip()
            config[section][key] = value.replace("\\\\", "\\").replace("\\", "/")


def _resolve_paths(config: configparser.ConfigParser, base_dir: Path) -> None:
    path_keys = ("terrain_path", "fault_shp_path", "study_area_shp_path", "output_path")
    empty_markers = {"", "none", "null", "skip", "false", "0"}
    for section in ("Data", "Paths"):
        if section not in config:
            continue
        for key in path_keys:
            if key not in config[section]:
                continue
            value = config[section][key].strip()
            if value.lower() in empty_markers:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            config[section][key] = str(path.resolve())


def load_app_config(config_path: str | Path = "config.ini") -> AppConfig:
    """Load config.ini and normalize old/new section names."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise UserConfigError(f"找不到配置文件: {path}")

    config = configparser.ConfigParser()
    try:
        with path.open("r", encoding="utf-8") as f:
            config.read_file(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk") as f:
            config.read_file(f)

    if "Run" not in config:
        config["Run"] = {"mode": "main", "preset": "demo"}

    mode = config.get("Run", "mode", fallback="main").strip().lower()
    if mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise UserConfigError(f"[Run] mode={mode!r} 不支持。可选值: {valid}")
    config["Run"]["mode"] = mode

    _copy_legacy_sections(config)
    _clean_paths(config)
    project_root = Path(__file__).resolve().parents[1]
    # User-facing config paths are normally written relative to the repository
    # root. This also keeps temporary config copies from redirecting demo output
    # to /tmp on smoke tests or Windows launcher flows.
    _resolve_paths(config, project_root)
    _validate_minimal_config(config, mode)

    return AppConfig(parser=config, path=path, mode=mode)


def _validate_minimal_config(config: configparser.ConfigParser, mode: str) -> None:
    required_sections = ["Run", "Data", "Model", "Optimization"]
    for section in required_sections:
        if section not in config:
            raise UserConfigError(f"config.ini 缺少 [{section}] 配置段")

    if mode == "main":
        terrain = config.get("Data", "terrain_path", fallback="").strip()
        if not terrain:
            raise UserConfigError("[Data] terrain_path 不能为空。请填写 DEM 文件路径。")

    output = config.get("Data", "output_path", fallback="").strip()
    if not output:
        raise UserConfigError("[Data] output_path 不能为空。请填写输出目录。")

    _validate_model_parameters(config)
    _validate_pecube_time_alignment(config, mode)


def _validate_model_parameters(config: configparser.ConfigParser) -> None:
    if "rainfall_factor" in config["Model"]:
        try:
            rainfall_factor = config.getfloat("Model", "rainfall_factor")
        except ValueError as exc:
            raise UserConfigError("[Model] rainfall_factor 必须是数字。") from exc

        if not math.isfinite(rainfall_factor) or rainfall_factor <= 0:
            raise UserConfigError(
                "[Model] rainfall_factor 必须为正数。"
                f"当前值为 {config.get('Model', 'rainfall_factor')}。"
            )

    initial_topography = config.get("Model", "initial_topography", fallback="random").strip().lower()
    if initial_topography not in {"random", "flat"}:
        raise UserConfigError("[Model] initial_topography 必须是 random 或 flat。")

    try:
        initial_elevation = config.getfloat("Model", "initial_elevation", fallback=0.0)
    except ValueError as exc:
        raise UserConfigError("[Model] initial_elevation 必须是数字。") from exc
    if not math.isfinite(initial_elevation):
        raise UserConfigError("[Model] initial_elevation 不能是 NaN/Inf。")

    try:
        initial_seed = config.getint("Model", "initial_topography_seed", fallback=42)
    except ValueError as exc:
        raise UserConfigError("[Model] initial_topography_seed 必须是整数。") from exc
    if initial_seed < 0:
        raise UserConfigError("[Model] initial_topography_seed 必须是非负整数。")


def _config_value_is_empty(value: str) -> bool:
    return value.strip().lower() in {"", "none", "null", "skip", "false", "0"}


def _pecube_enabled_for_validation(config: configparser.ConfigParser, mode: str) -> bool:
    if "Pecube" not in config:
        return False

    raw = config.get("Pecube", "enabled", fallback="auto").strip().lower()
    if raw in {"false", "0", "no", "off", "none", "skip"}:
        return False
    if raw in {"true", "1", "yes", "on"}:
        return True

    sample_raw = config.get("Pecube", "sample_observations", fallback="none")
    return mode == "pecube_coupled" or not _config_value_is_empty(sample_raw)


def _validate_pecube_time_alignment(config: configparser.ConfigParser, mode: str) -> None:
    if "Pecube" not in config or "total_time_myr" not in config["Pecube"]:
        return
    if not _pecube_enabled_for_validation(config, mode):
        return

    try:
        model_time_myr = config.getfloat("Model", "time_total") / 1_000_000.0
        pecube_time_myr = config.getfloat("Pecube", "total_time_myr")
    except ValueError as exc:
        raise UserConfigError("[Model] time_total 和 [Pecube] total_time_myr 必须是数字。") from exc

    tolerance = max(1e-6, abs(model_time_myr) * 1e-6)
    if abs(model_time_myr - pecube_time_myr) <= tolerance:
        return

    raise UserConfigError(
        "[Pecube] total_time_myr 必须和 [Model] time_total 对齐。"
        f"当前 FastScape time_total={config.get('Model', 'time_total')} 年，"
        f"等于 {model_time_myr:g} Ma，但 Pecube total_time_myr={pecube_time_myr:g} Ma。"
        f"请把 [Pecube] total_time_myr 改为 {model_time_myr:g}，"
        "或删除该项让程序自动使用 time_total/1e6。"
    )


def get_bool(config: configparser.ConfigParser, section: str, key: str, default: bool) -> bool:
    if section not in config or key not in config[section]:
        return default
    return config.getboolean(section, key)


def get_int_list(config: configparser.ConfigParser, section: str, key: str, default: list[int]) -> list[int]:
    value = config.get(section, key, fallback="").strip()
    if not value:
        return default
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise UserConfigError(f"[{section}] {key} 必须是逗号分隔的整数列表，例如 4,6,8") from exc


def get_shape(config: configparser.ConfigParser, section: str, key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = config.get(section, key, fallback="").strip()
    if not value:
        return default
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise UserConfigError(f"[{section}] {key} 必须写成 64,64 或 64x64")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise UserConfigError(f"[{section}] {key} 必须写成整数尺寸，例如 64,64") from exc


def config_to_dict(config: configparser.ConfigParser) -> dict[str, dict[str, Any]]:
    return {section: dict(config[section]) for section in config.sections()}
