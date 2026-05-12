"""Structured run directory management."""

from __future__ import annotations

import configparser
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Paths and metadata for one experiment run."""

    mode: str
    root: Path
    started_at: str
    config_path: Path
    config: configparser.ConfigParser
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    figure_index: int = 0
    status: str = "running"
    message: str = ""

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def arrays_dir(self) -> Path:
        return self.root / "arrays"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def add_artifact(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        try:
            artifact = str(resolved.relative_to(self.root.resolve()))
        except ValueError:
            artifact = str(resolved)
        if artifact not in self.artifacts:
            self.artifacts.append(artifact)

    def figure_path(self, filename: str | Path) -> Path:
        """Return the next numbered figure path for this run."""
        self.figure_index += 1
        name = Path(filename).name
        return self.figures_dir / f"{self.figure_index:02d}_{name}"


def create_run_context(config_path: Path, config: configparser.ConfigParser, mode: str) -> RunContext:
    output_base = Path(config.get("Data", "output_path", fallback="./demo/outputs")).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_base):
        run_number = _next_run_number(output_base)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_root = output_base / f"{run_number:04d}_{timestamp}_{mode}"
        run_root.mkdir(parents=True, exist_ok=False)
    for child in ("figures", "arrays", "metrics", "logs"):
        (run_root / child).mkdir(parents=True, exist_ok=True)

    context = RunContext(
        mode=mode,
        root=run_root,
        started_at=timestamp,
        config_path=config_path,
        config=config,
    )

    shutil.copy2(config_path, run_root / "config_used.ini")
    context.add_artifact(run_root / "config_used.ini")
    return context


def _next_run_number(output_base: Path) -> int:
    max_number = 0
    for path in output_base.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            max_number = max(max_number, int(prefix))
    return max_number + 1


@contextmanager
def _output_lock(output_base: Path):
    """Serialize run-number allocation across concurrent runner processes."""
    lock_dir = output_base / ".run_number.lock"
    deadline = time.time() + 60
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"等待输出目录锁超时: {lock_dir}")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def write_metrics(context: RunContext, filename: str, metrics: dict[str, Any]) -> Path:
    path = context.metrics_dir / filename
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    context.metrics.update(metrics)
    context.add_artifact(path)
    return path


def write_summary(context: RunContext) -> Path:
    lines = [
        f"# GA-LEM-Inverter Run Summary",
        "",
        f"- 运行模式: `{context.mode}`",
        f"- 状态: `{context.status}`",
        f"- 输出目录: `{context.root}`",
        f"- 开始时间: `{context.started_at}`",
    ]
    if context.message:
        lines.append(f"- 说明: {context.message}")

    run = context.config["Run"] if "Run" in context.config else {}
    lines.extend([
        f"- 运行预设: `{run.get('preset', 'demo')}`",
        "",
        "## 关键参数",
        "",
        f"- DEM: `{context.config.get('Data', 'terrain_path', fallback='')}`",
        f"- 输出基目录: `{context.config.get('Data', 'output_path', fallback='')}`",
        f"- scale_factor/K: `{context.config.get('Optimization', 'scale_factor', fallback='')}`",
        f"- GA 种群: `{context.config.get('Optimization', 'population_size', fallback=context.config.get('GeneticAlgorithm', 'ga_pop_size', fallback=''))}`",
        f"- GA 迭代: `{context.config.get('Optimization', 'max_iterations', fallback=context.config.get('GeneticAlgorithm', 'ga_max_iter', fallback=''))}`",
        f"- n_jobs: `{context.config.get('Optimization', 'n_jobs', fallback=context.config.get('GeneticAlgorithm', 'n_jobs', fallback=''))}`",
        "",
        "## 指标",
        "",
    ])

    if context.metrics:
        for key, value in sorted(context.metrics.items()):
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- 暂无指标。")

    lines.extend([
        "",
        "## 主要输出",
        "",
        "- 图像: `figures/`，文件名按生成顺序自动编号",
        "- 数组: `arrays/`",
        "- 指标: `metrics/`",
        "- 日志: `logs/`",
        "- 配置副本: `config_used.ini`",
        "- 运行清单: `run_manifest.json`",
    ])

    path = context.root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    context.add_artifact(path)
    return path


def write_manifest(context: RunContext) -> Path:
    manifest_path = context.root / "run_manifest.json"
    context.add_artifact(manifest_path)
    manifest = {
        "mode": context.mode,
        "status": context.status,
        "message": context.message,
        "started_at": context.started_at,
        "output_dir": str(context.root),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
        "artifacts": sorted(context.artifacts),
        "config": {section: dict(context.config[section]) for section in context.config.sections()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def finalize_run(context: RunContext, status: str, message: str = "") -> None:
    context.status = status
    context.message = message
    write_summary(context)
    write_manifest(context)


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _dependency_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "xarray-simlab",
        "fastscape",
        "rasterio",
        "geopandas",
        "scikit-learn",
        "scikit-image",
        "torch",
        "lpips",
        "pykrige",
        "scikit-opt",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
