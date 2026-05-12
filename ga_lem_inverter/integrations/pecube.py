"""Python API for the vendored Pecube engine."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ga_lem_inverter.integrations.pecube_loss import ThermochronologyLoss
from ga_lem_inverter.integrations.pecube_parser import PecubeOutputParser, PecubeParsedOutput
from ga_lem_inverter.integrations.pecube_project import PecubeProject, PecubeProjectBuilder, PecubeProjectConfig


@dataclass(frozen=True)
class PecubeEngineConfig:
    """Resolved Pecube engine paths and runtime switches."""

    enabled: bool = False
    pecube_root: Path = Path("vendor/pecube")
    pecube_bin_dir: Path = Path("vendor/pecube/bin")
    project_dir: Path = Path("vendor/pecube/projects")
    dataset_name: str = "fastscape"
    run_test: bool = True
    run_vtk: bool = False
    compute_loss: bool = True
    sample_observations: Path | None = None
    total_time_myr: float = 1.0
    velocity_km_per_myr: float = 1.0
    dlon: float = 0.01
    dlat: float = 0.01


@dataclass(frozen=True)
class PecubeCommandResult:
    """One Pecube subprocess result."""

    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PecubeResult:
    """Structured result returned by PecubeEngine.run."""

    project: PecubeProject
    commands: list[PecubeCommandResult]
    parsed: PecubeParsedOutput
    loss: ThermochronologyLoss
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return all(command.returncode == 0 for command in self.commands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "project_dir": str(self.project.project_dir),
            "input_file": str(self.project.input_file),
            "dataset_dir": str(self.project.dataset_dir),
            "commands": [
                {
                    "name": command.name,
                    "command": command.command,
                    "returncode": command.returncode,
                    "stdout": command.stdout[-4000:],
                    "stderr": command.stderr[-4000:],
                }
                for command in self.commands
            ],
            "parsed_output": PecubeOutputParser.to_dict(self.parsed),
            "loss": {
                "value": self.loss.value,
                "n_observations": self.loss.n_observations,
                "message": self.loss.message,
            },
            "metrics": self.metrics,
        }


class PecubeEngine:
    """High-level API for building, running, and parsing Pecube projects."""

    def __init__(self, config: PecubeEngineConfig):
        self.config = config
        self.parser = PecubeOutputParser()

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "PecubeEngine":
        root = Path(__file__).resolve().parents[2]
        section = config["Pecube"] if "Pecube" in config else {}

        def path_value(key: str, default: str) -> Path:
            raw = section.get(key, default) if hasattr(section, "get") else default
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = root / path
            return path.resolve()

        def bool_value(key: str, default: bool) -> bool:
            if "Pecube" not in config or key not in config["Pecube"]:
                return default
            raw = config.get("Pecube", key).strip().lower()
            if raw in {"auto", "default"}:
                return default
            return config.getboolean("Pecube", key)

        def float_value(key: str, default: float) -> float:
            if "Pecube" not in config or key not in config["Pecube"]:
                return default
            return config.getfloat("Pecube", key)

        sample_raw = section.get("sample_observations", "none") if hasattr(section, "get") else "none"
        sample_path: Path | None = None
        if str(sample_raw).strip().lower() not in {"", "none", "null", "skip", "false", "0"}:
            sample_path = path_value("sample_observations", str(sample_raw))

        mode = config.get("Run", "mode", fallback="")
        enabled_default = mode == "pecube_coupled"
        engine_config = PecubeEngineConfig(
            enabled=bool_value("enabled", enabled_default),
            pecube_root=path_value("pecube_root", "vendor/pecube"),
            pecube_bin_dir=path_value("pecube_bin_dir", "vendor/pecube/bin"),
            project_dir=path_value("project_dir", "vendor/pecube/projects"),
            dataset_name=str(section.get("dataset_name", "fastscape")) if hasattr(section, "get") else "fastscape",
            run_test=bool_value("run_test", True),
            run_vtk=bool_value("run_vtk", False),
            compute_loss=bool_value("compute_loss", True),
            sample_observations=sample_path,
            total_time_myr=float_value("total_time_myr", 1.0),
            velocity_km_per_myr=float_value("velocity_km_per_myr", 1.0),
            dlon=float_value("dlon", 0.01),
            dlat=float_value("dlat", 0.01),
        )
        return cls(engine_config)

    def validate(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("[Pecube] enabled=false，pecube_coupled 模式需要启用 Pecube。")
        missing = [name for name in ("Pecube", "Test") if not self._executable(name).exists()]
        if missing:
            raise FileNotFoundError(
                "Pecube 可执行文件缺失: "
                + ", ".join(missing)
                + f"。请先运行: bash tools/environment/build_pecube.sh"
            )

    def run(
        self,
        *,
        topography_series: Iterable[np.ndarray],
        uplift_series: Iterable[np.ndarray],
        temperature_series: Iterable[np.ndarray] | None = None,
        sample_observations: Path | None = None,
        output_dir: Path,
    ) -> PecubeResult:
        self.validate()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Pecube's Fortran command-line programs store the run folder in a
        # character*5 variable. Use a five-character project name so the path
        # is not silently truncated on any platform.
        project_dir = output_dir / "PGB01"
        parsed_dir = output_dir / "parsed_outputs"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        builder = PecubeProjectBuilder(
            PecubeProjectConfig(
                dataset_name=self.config.dataset_name,
                total_time_myr=self.config.total_time_myr,
                velocity_km_per_myr=self.config.velocity_km_per_myr,
                dlon=self.config.dlon,
                dlat=self.config.dlat,
            )
        )
        observations = sample_observations or self.config.sample_observations
        project = builder.build(
            project_dir=project_dir,
            topography_series=topography_series,
            uplift_series=uplift_series,
            temperature_series=temperature_series,
            sample_observations=observations,
        )

        commands: list[PecubeCommandResult] = []
        if self.config.run_test:
            commands.append(self._run_command("Test", project.project_dir))
        commands.append(self._run_command("Pecube", project.project_dir))
        if self.config.run_vtk:
            commands.append(self._run_command("Vtk", project.project_dir))

        parsed = self.parser.parse(project.project_dir)
        parsed_path = parsed_dir / "pecube_parsed_outputs.json"
        parsed_path.write_text(json.dumps(PecubeOutputParser.to_dict(parsed), indent=2, ensure_ascii=False), encoding="utf-8")

        loss = (
            ThermochronologyLoss.compute(parsed, observations)
            if self.config.compute_loss
            else ThermochronologyLoss.unavailable("[Pecube] compute_loss=false。")
        )
        metrics = {
            "pecube_success": all(command.returncode == 0 for command in commands),
            "pecube_csv_file_count": len(parsed.csv_files),
            "pecube_output_file_count": len(parsed.files),
            "thermochronology_loss": loss.value,
        }
        return PecubeResult(project=project, commands=commands, parsed=parsed, loss=loss, metrics=metrics)

    def _run_command(self, name: str, project_dir: Path) -> PecubeCommandResult:
        executable = self._executable(name)
        project_arg = project_dir.name
        command = [str(executable), project_arg]
        env = os.environ.copy()
        result = subprocess.run(
            command,
            cwd=project_dir.parent,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Pecube command failed: {' '.join(command)}\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        return PecubeCommandResult(
            name=name,
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _executable(self, name: str) -> Path:
        candidates = [self.config.pecube_bin_dir / name, self.config.pecube_bin_dir / f"{name}.exe"]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def copy_project_to_vendor(result: PecubeResult, vendor_project_dir: Path) -> Path:
        target = Path(vendor_project_dir).resolve()
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(result.project.project_dir, target)
        return target
