"""CLI for a read-only public-source breakdown of a completed test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.diagnostics.source_subsets import SourceSubsetDiagnosticError, run_source_subset_diagnostic


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose held-out test metrics by sealed public-data source.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory outside the formal run evidence.")
    parser.add_argument("--device", help="Optional Ultralytics device override.")
    parser.add_argument("--batch", type=int, default=4, help="Diagnostic evaluation batch size (default: 4).")
    parser.add_argument("--image-size", type=int, help="Optional diagnostic inference resolution; defaults to the recorded training resolution.")
    args = parser.parse_args(argv)
    try:
        result = run_source_subset_diagnostic(
            run_dir=args.run_dir.resolve(), output=args.output.resolve(), device=args.device, batch=args.batch, image_size=args.image_size,
        )
    except SourceSubsetDiagnosticError as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"output": str(result)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
