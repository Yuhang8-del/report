"""Build immutable final-report figures and tables from report_data.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.reporting.final_figures import FinalFigureError, write_final_figures
from fruit_ssod.reporting.final_tables import write_final_tables


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalFigureError(f"Problem: report-data input cannot be read. Likely cause: {error}. Remediation: pass the immutable report_data.json emitted by build_report_data.") from error
    if not isinstance(payload, Mapping):
        raise FinalFigureError("Problem: report-data input is not an object. Likely cause: the JSON root is malformed. Remediation: rebuild report_data.json from the verified evidence package.")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report_assets(report_data_path: Path, output: Path) -> Path:
    source = report_data_path.resolve(strict=True)
    destination = output.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FinalFigureError(f"Problem: report asset output already exists. Likely cause: {destination} is immutable evidence. Remediation: choose a fresh output directory.")
    report_data = _load(source)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
        figures = write_final_figures(report_data, temporary / "figures")
        tables = write_final_tables(report_data, temporary / "tables")
        manifest = {
            "schema_version": "1.0",
            "protocol": "fruit_ssod_final_report_assets_v1",
            "report_data": {"path": str(source), "sha256": _sha(source)},
            "figure_count": len(figures),
            "table_count": len(tables),
            "figures": [{"relative_path": item.relative_to(temporary).as_posix(), "sha256": _sha(item)} for item in figures],
            "tables": [{"relative_path": item.relative_to(temporary).as_posix(), "sha256": _sha(item)} for item in tables],
            "captions": {"relative_path": "figures/captions.json", "sha256": _sha(temporary / "figures" / "captions.json")},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate final-report figures and tables from sealed report_data.json.")
    parser.add_argument("--report-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = build_report_assets(args.report_data, args.output)
    except (FinalFigureError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "manifest": str(output / "manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
