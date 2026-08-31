"""Verify a full-label supervised snapshot without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.full_label_upper_bound import audit_full_label_upper_bound
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify every v12 snapshot image digest, JPEG marker and canonical label.")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_full_label_upper_bound(args.snapshot_root)
        if args.output is not None:
            output = args.output.resolve(strict=False)
            if output.exists():
                parser.error(f"Problem: audit output {output} already exists. Likely cause: audit evidence is immutable. Remediation: choose a fresh output path.")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    except (SupervisedDatasetError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
