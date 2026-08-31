"""Build the immutable evidence input required by final-report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.reporting.report_data import ReportDataError, build_report_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final report_data.json only from verified experiment, audit, and benchmark evidence.")
    parser.add_argument("--result-package", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--pseudo-audit", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = build_report_data(result_package=args.result_package, dataset_audit=args.dataset_audit, pseudo_audit=args.pseudo_audit, benchmark=args.benchmark, output=args.output)
    except ReportDataError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
