"""正演输出目录管理。

每次运行创建 ``<output_root>/NNNN_<时间戳>_forward/``，结构和反演工具风格一致：
``arrays/`` ``figures/`` ``logs/`` ``summary.md`` ``config_used.ini``。
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ForwardRunContext:
    """单次正演运行的所有路径和元数据。"""

    root: Path
    started_at: str
    config_path: Path
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
        self.figure_index += 1
        name = Path(filename).name
        return self.figures_dir / f"{self.figure_index:02d}_{name}"


def create_run_context(output_root: Path, config_path: Path) -> ForwardRunContext:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_root):
        run_number = _next_run_number(output_root)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_root = output_root / f"{run_number:04d}_{timestamp}_forward"
        run_root.mkdir(parents=True, exist_ok=False)
    for child in ("figures", "arrays", "logs"):
        (run_root / child).mkdir(parents=True, exist_ok=True)

    context = ForwardRunContext(
        root=run_root,
        started_at=timestamp,
        config_path=Path(config_path).resolve(),
    )
    shutil.copy2(config_path, run_root / "config_used.ini")
    context.add_artifact(run_root / "config_used.ini")
    return context


def _next_run_number(output_root: Path) -> int:
    max_number = 0
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            max_number = max(max_number, int(prefix))
    return max_number + 1


@contextmanager
def _output_lock(output_root: Path):
    lock_dir = output_root / ".forward_run_number.lock"
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


def write_summary(context: ForwardRunContext, info: dict[str, Any]) -> Path:
    """写 summary.md，info 是要展示的参数和指标。"""
    lines = [
        "# FastScape 正演运行摘要",
        "",
        f"- 状态: `{context.status}`",
        f"- 输出目录: `{context.root}`",
        f"- 开始时间: `{context.started_at}`",
    ]
    if context.message:
        lines.append(f"- 说明: {context.message}")

    lines += ["", "## 关键参数", ""]
    for key, value in info.get("parameters", {}).items():
        lines.append(f"- `{key}`: {value}")

    lines += ["", "## 指标", ""]
    metrics = info.get("metrics", {}) or context.metrics
    if metrics:
        for key, value in sorted(metrics.items()):
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- 暂无指标。")

    lines += [
        "",
        "## 主要输出",
        "",
        "- 数组: `arrays/`（topography_series / cumulative_erosion / mean_erosion_rate / net_uplift / uplift_series / output_times_years）",
        "- 图像: `figures/`，按生成顺序自动编号",
        "- 日志: `logs/forward.log`",
        "- 配置副本: `config_used.ini`",
        "- 运行清单: `run_manifest.json`",
    ]

    summary_path = context.root / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    context.add_artifact(summary_path)
    return summary_path


def write_manifest(context: ForwardRunContext, info: dict[str, Any]) -> Path:
    manifest_path = context.root / "run_manifest.json"
    context.add_artifact(manifest_path)
    manifest = {
        "tool": "forward_simulator",
        "status": context.status,
        "message": context.message,
        "started_at": context.started_at,
        "output_dir": str(context.root),
        "python": sys.version,
        "parameters": info.get("parameters", {}),
        "metrics": info.get("metrics", context.metrics),
        "artifacts": sorted(context.artifacts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest_path


def finalize_run(context: ForwardRunContext, status: str, info: dict[str, Any], message: str = "") -> None:
    context.status = status
    context.message = message
    write_summary(context, info)
    write_manifest(context, info)
