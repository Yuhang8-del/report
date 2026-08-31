"""Aggregate Task 12 supervised reference runs without hiding failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.training.supervised_matrix import SupervisedMatrixError, aggregate_supervised_matrix, write_supervised_matrix_aggregate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate supervised reference validation and fixed-test evidence.")
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="One run directory; repeat once per submitted matrix run.")
    parser.add_argument("--output", type=Path, help="Optional new JSON output path. Existing files are never overwritten.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = aggregate_supervised_matrix(args.run_dir)
        if args.output is not None:
            write_supervised_matrix_aggregate(result, args.output)
    except SupervisedMatrixError as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
