"""Create the auditable all-label supervised upper-bound snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.full_label_upper_bound import materialize_full_label_upper_bound
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore the hidden train labels for a separate supervised upper-bound snapshot while preserving protected splits.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-train-count", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_full_label_upper_bound(args.candidate_manifest, args.split_root, args.source_root, args.output_root, expected_train_count=args.expected_train_count)
    except SupervisedDatasetError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "dataset_yaml": str(result.dataset_yaml), "membership": str(result.membership), "image_count": result.image_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
