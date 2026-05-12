"""Thermochronology loss helpers for Pecube outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ga_lem_inverter.integrations.pecube_parser import PecubeParsedOutput


@dataclass(frozen=True)
class ThermochronologyLoss:
    """Compare Pecube predictions against observed sample ages."""

    value: float | None
    n_observations: int
    message: str

    @classmethod
    def unavailable(cls, message: str) -> "ThermochronologyLoss":
        return cls(value=None, n_observations=0, message=message)

    @classmethod
    def compute(cls, parsed: PecubeParsedOutput, sample_observations: Path | None = None) -> "ThermochronologyLoss":
        if sample_observations is None:
            return cls.unavailable("未提供热年代学观测样品，跳过 Pecube loss。")
        if not Path(sample_observations).exists():
            return cls.unavailable(f"观测样品文件不存在: {sample_observations}")
        if not parsed.csv_files:
            return cls.unavailable("Pecube 未产生可解析 CSV 输出，无法计算 loss。")
        return cls.unavailable("Pecube loss 需要先定义观测 CSV 与预测 CSV 的列映射。")


def rmse(observed: list[float], predicted: list[float]) -> float:
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed 和 predicted 必须长度一致且非空。")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(observed, predicted)) / len(observed))

