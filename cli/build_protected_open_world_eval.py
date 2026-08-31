"""Build the protected novel-class evaluation-only YOLO view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.open_world.incremental import build_protected_holdout_eval_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--protected-truth", type=Path, required=True)
    result.add_argument("--class-registry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = build_protected_holdout_eval_dataset(
        protected_novel_truth=args.protected_truth,
        class_registry=args.class_registry,
        output_root=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
