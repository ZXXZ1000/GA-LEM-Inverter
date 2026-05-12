"""Parse Pecube output files into lightweight Python structures."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PecubeParsedOutput:
    """Structured view of a Pecube output directory."""

    output_dir: Path
    csv_files: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


class PecubeOutputParser:
    """Read Pecube output CSV files when present."""

    def parse(self, project_dir: Path) -> PecubeParsedOutput:
        output_dir = Path(project_dir) / "output"
        if not output_dir.exists():
            return PecubeParsedOutput(output_dir=output_dir)

        csv_files: dict[str, list[dict[str, str]]] = {}
        files: list[str] = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            files.append(str(path.relative_to(output_dir)))
            if path.suffix.lower() != ".csv":
                continue
            csv_files[str(path.relative_to(output_dir))] = self._read_csv(path)
        return PecubeParsedOutput(output_dir=output_dir, csv_files=csv_files, files=files)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            with path.open("r", encoding="latin-1", newline="") as handle:
                return list(csv.DictReader(handle))
        except csv.Error:
            return []

    @staticmethod
    def to_dict(parsed: PecubeParsedOutput) -> dict[str, Any]:
        return {
            "output_dir": str(parsed.output_dir),
            "files": parsed.files,
            "csv_files": parsed.csv_files,
        }

