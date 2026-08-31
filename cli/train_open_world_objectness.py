"""Train the known-only one-class Fruit objectness detector for OWOD proposals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--image-size", type=int, default=640)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="0")
    return result


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from ultralytics import YOLO

    run_dir = (args.project / args.name).resolve()
    record_path = run_dir / "open_world_run_record.json"
    started = time.time()
    record: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_type": "open_world_objectness_training_run",
        "status": "running",
        "weights": str(args.weights.resolve(strict=True)),
        "dataset": str(args.dataset.resolve(strict=True)),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "started_unix": started,
    }
    _write(record_path, record)
    try:
        model = YOLO(str(args.weights))
        model.train(
            data=str(args.dataset),
            project=str(args.project),
            name=args.name,
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch_size,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            device=args.device,
            amp=True,
            cache=False,
            close_mosaic=10,
            # Ultralytics 8.4.31 can fail in its optional result-plotting step on
            # this Windows/Polars combination after the weights are already saved.
            plots=False,
            exist_ok=True,
        )
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        record.update(
            {
                "status": "complete",
                "completed_unix": time.time(),
                "duration_seconds": time.time() - started,
                "best_weights": str(best) if best.is_file() else None,
                "last_weights": str(last) if last.is_file() else None,
            }
        )
        _write(record_path, record)
        return 0
    except Exception as error:
        # Ultralytics may finish training, strip/validate both checkpoints, and
        # only then fail while drawing results.csv (for example when an optional
        # Polars binary is unavailable).  Do not silently call that complete;
        # the separate recovery command must load and validate best.pt first.
        record.update(
            {
                "status": "failed",
                "completed_unix": time.time(),
                "duration_seconds": time.time() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write(record_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
