"""Create deterministic train-only object-centric tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.object_centric_tiles import materialize_object_centric_tiles
from fruit_ssod.data.supervised_dataset import SupervisedDatasetError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create object-centric small-object tiles without changing validation or test membership.")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-training-list", type=Path)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--small-object-area", type=float, default=0.01)
    parser.add_argument("--minimum-visibility", type=float, default=0.5)
    parser.add_argument("--max-tiles-per-image", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = materialize_object_centric_tiles(
            args.snapshot_root,
            args.output_root,
            base_training_list=args.base_training_list,
            tile_size=args.tile_size,
            small_object_area=args.small_object_area,
            minimum_visibility=args.minimum_visibility,
            max_tiles_per_image=args.max_tiles_per_image,
        )
    except SupervisedDatasetError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "dataset_yaml": str(result.dataset_yaml), "membership": str(result.membership), "tile_count": result.tile_count, "exposure_count": result.exposure_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
