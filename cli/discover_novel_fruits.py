"""Run the post-Student open-world novel-fruit discovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.open_world.discovery import NOVEL_CLASSES, run_discovery


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use the completed Student as the known five-class detector, then "
            "run self-supervised discovery on fruit categories outside it."
        )
    )
    parser.add_argument("--student-weights", type=Path, required=True, help="Completed Student best.pt.")
    parser.add_argument("--source-root", type=Path, required=True, help="DeepNIR yolov5 category root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Immutable open-world artifact directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--clusters", type=int, default=len(NOVEL_CLASSES))
    parser.add_argument("--epochs", type=int, default=10, help="Self-supervised adaptation epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--novelty-threshold", type=float, default=0.5)
    parser.add_argument("--known-test-list", type=Path, default=None, help="Optional sealed known-class test.txt for false-positive measurement.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.student_weights.is_file():
        parser.error(f"Student checkpoint does not exist: {args.student_weights}")
    if args.clusters < 2:
        parser.error("--clusters must be at least 2")
    if args.epochs < 0:
        parser.error("--epochs cannot be negative")
    try:
        result = run_discovery(
            student_weights=args.student_weights.resolve(),
            source_root=args.source_root.resolve(strict=True),
            output_dir=args.output_dir.resolve(),
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
            clusters=args.clusters,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            image_size=args.image_size,
            novelty_threshold=args.novelty_threshold,
            known_test_list=args.known_test_list.resolve() if args.known_test_list is not None else None,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "metrics": result["metrics"], "novelty": result["novelty"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
