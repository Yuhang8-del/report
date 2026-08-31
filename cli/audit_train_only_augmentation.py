"""Audit an immutable v13 train-only augmentation before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.supervised_dataset import SupervisedDatasetError
from fruit_ssod.data.train_only_augmentation import audit_train_only_augmentation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify added training images and unchanged sealed validation/test lists.")
    parser.add_argument("--augmentation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Problem: audit output {args.output} already exists. Likely cause: audit evidence is immutable. Remediation: choose a fresh audit output path.")
    try:
        result = audit_train_only_augmentation(args.augmentation_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (SupervisedDatasetError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
