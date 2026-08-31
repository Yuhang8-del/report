"""Build the protected box-level OWOD protocol and known-only objectness dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.open_world.box_protocol import build_known_objectness_dataset, build_novel_box_protocol


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--known-train-list", type=Path, required=True)
    result.add_argument("--known-validation-list", type=Path, required=True)
    result.add_argument("--novel-source-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--holdout-fraction", type=float, default=0.2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = build_novel_box_protocol(
        args.novel_source_root.resolve(strict=True),
        args.output_root / "protocol",
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    objectness = build_known_objectness_dataset(
        args.known_train_list.resolve(strict=True),
        args.known_validation_list.resolve(strict=True),
        args.output_root / "objectness_dataset",
    )
    print(json.dumps({"protocol": protocol, "objectness_dataset": objectness}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
