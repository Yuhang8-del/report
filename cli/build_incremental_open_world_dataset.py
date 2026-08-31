"""Build a reviewed incremental dataset with old-class replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.open_world.incremental import build_incremental_replay_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--known-train-list", type=Path, required=True)
    result.add_argument("--known-validation-list", type=Path, required=True)
    result.add_argument("--protected-novel-truth", type=Path, required=True)
    result.add_argument("--confirmed-category", action="append", required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--replay-images", type=int, default=2000)
    result.add_argument("--novel-validation-fraction", type=float, default=0.1)
    result.add_argument("--seed", type=int, default=42)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = build_incremental_replay_dataset(
        known_train_list=args.known_train_list.resolve(strict=True),
        known_validation_list=args.known_validation_list.resolve(strict=True),
        protected_novel_truth=args.protected_novel_truth.resolve(strict=True),
        confirmed_categories=args.confirmed_category,
        output_root=args.output_root.resolve(),
        replay_images=args.replay_images,
        novel_validation_fraction=args.novel_validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
