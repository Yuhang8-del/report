"""Train a reviewed expanded detector using old-class replay plus new classes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-weights", type=Path, required=True)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--epochs", type=int, default=40)
    result.add_argument("--image-size", type=int, default=768)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--patience", type=int, default=10)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="0")
    result.add_argument("--resume-from", type=Path)
    return result


def write_record(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from ultralytics import YOLO

    run_dir = (args.project / args.name).resolve()
    record_path = run_dir / "incremental_run_record.json"
    started = time.time()
    record: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_type": "reviewed_open_world_incremental_training",
        "status": "running",
        "base_weights": str(args.base_weights.resolve(strict=True)),
        "dataset": str(args.dataset.resolve(strict=True)),
        "training_policy": "old-class replay plus reviewed novel discovery labels",
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "started_unix": started,
        "resume_from": str(args.resume_from.resolve(strict=True)) if args.resume_from else None,
    }
    write_record(record_path, record)
    try:
        if args.resume_from:
            model = YOLO(str(args.resume_from))
            # The checkpoint restores the original dataset, optimizer, epoch,
            # save directory and plots=False configuration.
            model.train(resume=True, device=args.device)
        else:
            model = YOLO(str(args.base_weights))
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
                # Keep training independent of Ultralytics' optional plotting code;
                # plots are produced by the project's evaluation/report pipeline.
                plots=False,
                exist_ok=True,
            )
        record.update(
            {
                "status": "complete",
                "completed_unix": time.time(),
                "duration_seconds": time.time() - started,
                "best_weights": str(run_dir / "weights" / "best.pt"),
                "last_weights": str(run_dir / "weights" / "last.pt"),
            }
        )
        write_record(record_path, record)
        return 0
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "completed_unix": time.time(),
                "duration_seconds": time.time() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_record(record_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
