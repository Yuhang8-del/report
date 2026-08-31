"""Select the frozen v12 candidate using validation evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.evaluation.validation_selection import ValidationSelectionError, select_from_manifest, write_selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a v12 candidate from completed validation-only runs without reading fixed-test evidence.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-ap50-floor", type=float, default=0.50)
    args = parser.parse_args(argv)
    try:
        result = select_from_manifest(args.candidate_manifest, per_class_ap50_floor=args.per_class_ap50_floor)
        output = write_selection(result, args.output)
    except (ValidationSelectionError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "selected_candidate_id": result["selected_candidate_id"], "selection_status": result["selection_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
