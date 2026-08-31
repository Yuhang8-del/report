"""Create a sealed YOLO dataset snapshot for one supervised label budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.supervised_dataset import SupervisedDatasetError, materialize_supervised_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize one immutable supervised YOLO snapshot from sealed split labels.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_supervised_dataset(args.split_root, args.source_root, args.output_root, budget=args.budget)
    except SupervisedDatasetError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "dataset_yaml": str(result.dataset_yaml), "membership": str(result.membership), "image_count": result.image_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
