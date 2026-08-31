"""Create a reproducible offline GUI image-inference package for one checkpoint.

This helper intentionally uses the same detector adapter and result exporter as
the PySide6 application. It does not add camera or open-world functionality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.gui.result_exporter import export_inference_results
from fruit_ssod.gui.workers.image_worker import ImageInferenceResult, ImageInferenceSettings


CLASS_REGISTRY = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.images:
        paths = tuple(Path(item).expanduser().resolve() for item in args.images)
    else:
        source = Path(args.source_results).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        paths = tuple(Path(item["image_path"]).expanduser().resolve() for item in payload["results"])
    if not paths or any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise SystemExit(f"image input is missing or unreadable: {missing}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Five-class Ultralytics best.pt checkpoint.")
    parser.add_argument("--output", required=True, help="New, non-existing export directory.")
    parser.add_argument("--source-results", help="Existing GUI results.json used for its image list.")
    parser.add_argument("--images", nargs="*", help="Explicit image paths; overrides --source-results.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    if not args.images and not args.source_results:
        parser.error("provide --images or --source-results")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if output.exists():
        raise SystemExit(f"output directory already exists; refusing to overwrite: {output}")

    paths = _image_paths(args)
    settings = ImageInferenceSettings(confidence=args.confidence, nms_iou=args.iou)
    adapter = UltralyticsDetectorAdapter(weights_path=checkpoint, source_model=str(checkpoint))
    adapter.initialize()
    results: list[ImageInferenceResult] = []
    for path in paths:
        started = perf_counter()
        detections = tuple(adapter.predict(path, confidence=settings.confidence, nms_iou=settings.nms_iou))
        results.append(
            ImageInferenceResult(
                image_path=path,
                detections=detections,
                latency_ms=(perf_counter() - started) * 1000.0,
                confidence=settings.confidence,
                nms_iou=settings.nms_iou,
            )
        )
    export = export_inference_results(results, output)
    metadata = {
        "artifact_type": "fruit_ssod_gui_candidate_metadata",
        "run_id": checkpoint.parent.parent.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "image_count": len(results),
        "camera_enabled": False,
        "open_world_enabled": False,
        "class_registry": CLASS_REGISTRY,
        "export_manifest": str(export.manifest_path),
        "detections_per_image": [len(result.detections) for result in results],
    }
    (output / "v0_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
