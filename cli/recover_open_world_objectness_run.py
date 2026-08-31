"""Validate a saved OWOD objectness checkpoint after a non-training plotting failure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--image-size", type=int, default=640)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device", default="0")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from ultralytics import YOLO

    run_dir = args.run_dir.resolve(strict=True)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if not best.is_file() or not last.is_file() or not results_csv.is_file():
        raise FileNotFoundError("training checkpoint or results.csv is missing; recovery is not allowed")
    model = YOLO(str(best))
    metrics = model.val(
        data=str(args.dataset.resolve(strict=True)),
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        plots=False,
        verbose=False,
    )
    evidence = {
        "schema_version": "1.0",
        "artifact_type": "post_training_checkpoint_validation",
        "checkpoint": str(best),
        "dataset": str(args.dataset.resolve()),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "validated_unix": time.time(),
        "plots": False,
    }
    (run_dir / "open_world_checkpoint_validation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record_path = run_dir / "open_world_run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    original_error = record.get("error")
    record.update(
        {
            "status": "complete",
            "recovered_after_non_training_failure": True,
            "non_training_warning": original_error,
            "best_weights": str(best),
            "last_weights": str(last),
            "checkpoint_validation": evidence,
        }
    )
    record.pop("error", None)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
