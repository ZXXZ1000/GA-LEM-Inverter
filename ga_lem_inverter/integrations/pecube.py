"""Pecube integration boundary.

This module is intentionally small for now. The current public tool focuses on
Fastscape + GA inversion; future Pecube coupling should enter through this
module instead of leaking Fortran/Pecube file handling into user-facing scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PecubeRunRequest:
    """Structured input for a future Pecube run."""

    uplift_field: Path
    topography: Path
    output_dir: Path
    sample_observations: Path | None = None
    parameters: dict[str, Any] | None = None


def run_pecube(request: PecubeRunRequest) -> dict[str, Any]:
    """Placeholder for the future Pecube coupling.

    The function exists so workflow code has a stable extension point. It should
    be implemented when Pecube input templates, executable discovery, and
    thermochronology observation formats are fixed.
    """

    raise NotImplementedError(
        "Pecube 集成尚未启用。当前版本先完成 Fastscape + GA 的统一入口；"
        "后续 Pecube 计算应通过 ga_lem_inverter.integrations.pecube 接入。"
    )
