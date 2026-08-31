"""CLI for adding curated images only to supervised training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.supervised_dataset import SupervisedDatasetError
from fruit_ssod.data.train_only_augmentation import materialize_train_only_augmentation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add a cleaned candidate manifest to training while preserving sealed validation and test lists.")
    parser.add_argument("--base-training-root", type=Path, required=True)
    parser.add_argument("--added-candidate-manifest", type=Path, required=True)
    parser.add_argument("--added-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protected-near-hash-threshold", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = materialize_train_only_augmentation(
            args.base_training_root,
            args.added_candidate_manifest,
            args.added_source_root,
            args.output_root,
            protected_near_hash_threshold=args.protected_near_hash_threshold,
        )
    except SupervisedDatasetError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "dataset_yaml": str(result.dataset_yaml), "membership": str(result.membership), "base_train_exposure_count": result.base_train_exposure_count, "added_train_image_count": result.added_train_image_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
