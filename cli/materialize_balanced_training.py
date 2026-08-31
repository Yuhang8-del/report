"""Create a deterministic class-balanced view for v12 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.balanced_training import materialize_balanced_training_view
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a class-balanced training list while preserving the sealed validation and test partitions.")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-appearances-per-image", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = materialize_balanced_training_view(
            args.snapshot_root,
            args.output_root,
            seed=args.seed,
            max_appearances_per_image=args.max_appearances_per_image,
        )
    except SupervisedDatasetError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output_root": str(result.root),
                "dataset_yaml": str(result.dataset_yaml),
                "membership": str(result.membership),
                "base_image_count": result.base_image_count,
                "exposure_count": result.exposure_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
