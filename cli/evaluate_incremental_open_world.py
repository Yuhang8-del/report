"""Evaluate the reviewed 11-class detector on the protected novel holdout."""

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
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--image-size", type=int, default=640)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--device", default="0")
    return result


def _values(value: object) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = YOLO(str(args.weights.resolve(strict=True)))
    metrics = model.val(
        data=str(args.dataset.resolve(strict=True)),
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        plots=False,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
    )
    names = {int(key): str(value) for key, value in metrics.names.items()}
    precision = _values(getattr(metrics.box, "p", None))
    recall = _values(getattr(metrics.box, "r", None))
    ap50 = _values(getattr(metrics.box, "ap50", None))
    maps = _values(getattr(metrics.box, "maps", None))
    active_ids = [int(value) for value in _values(getattr(metrics.box, "ap_class_index", None))]
    active_positions = {class_id: position for position, class_id in enumerate(active_ids)}
    per_class = {}
    for class_id, name in names.items():
        position = active_positions.get(class_id)
        per_class[name] = {
            "class_id": class_id,
            "present_in_holdout": position is not None,
            "precision": precision[position] if position is not None and position < len(precision) else None,
            "recall": recall[position] if position is not None and position < len(recall) else None,
            "map50": ap50[position] if position is not None and position < len(ap50) else None,
            "map50_95": maps[class_id] if class_id < len(maps) else None,
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "protected_novel_holdout_incremental_evaluation",
        "weights": str(args.weights.resolve(strict=True)),
        "dataset": str(args.dataset.resolve(strict=True)),
        "protected_holdout": True,
        "training_use_permitted": False,
        "image_size": args.image_size,
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "duration_seconds": time.time() - started,
    }
    path = args.output / "protected_novel_holdout_metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
