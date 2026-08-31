"""Evidence-bound final-report table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.reporting.final_figures import FinalFigureError, report_aggregate_view
from fruit_ssod.reporting.result_tables import csv_text, fruitdet_rows, main_summary_rows, primary_result_rows


def write_final_tables(report_data: Mapping[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Write at most three CSV table projections without hiding failed rows."""
    aggregate = report_aggregate_view(report_data)
    projections = {
        "all_runs.csv": primary_result_rows(aggregate),
        "main_summary.csv": main_summary_rows(aggregate),
        "fruitdet_external.csv": fruitdet_rows(aggregate),
    }
    if len(projections) > 10:
        raise FinalFigureError("Problem: final table set exceeds the report limit. Likely cause: uncontrolled table creation. Remediation: reduce the controlled final table set to at most ten.")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, rows in projections.items():
        # An absent external protocol is a visible gap, not a fabricated table.
        if not rows:
            continue
        destination = output_dir / name
        destination.write_text(csv_text(rows), encoding="utf-8", newline="\n")
        paths.append(destination)
    return tuple(paths)
