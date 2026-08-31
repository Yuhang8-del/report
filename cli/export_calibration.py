"""Export aggregate aspect bounds and validation PR evidence for SSOD filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.detection.adapter import DetectorAdapterError
from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.pseudo.calibration import CalibrationError, write_calibration_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export immutable aggregate-only Trust Filter calibration artifacts.")
    parser.add_argument("--human-labels", type=Path, required=True, help="Task 8 20% human training labels.json; used only to calculate aggregate aspect-ratio bounds.")
    parser.add_argument("--validation-labels", type=Path, required=True, help="Task 8 protected validation_labels.json; used only for prediction/outcome PR matching.")
    parser.add_argument("--image-root", type=Path, required=True, help="Materialized root resolving validation file_path entries.")
    parser.add_argument("--weights", type=Path, required=True, help="Completed Teacher weights/best.pt.")
    parser.add_argument("--teacher-run-id", required=True, help="Completed supervised Teacher run ID.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New immutable calibration directory.")
    parser.add_argument("--confidence", type=float, default=0.001, help="Low inference confidence used to capture validation prediction/outcome rows.")
    parser.add_argument("--iou", type=float, default=0.50, help="Class-aware one-to-one validation matching IoU.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        detector = UltralyticsDetectorAdapter(weights_path=args.weights, source_model=str(args.weights))
        aspect, validation_pr = write_calibration_artifacts(
            human_labels=args.human_labels, validation_labels=args.validation_labels, image_root=args.image_root,
            detector=detector, teacher_run_id=args.teacher_run_id, output_dir=args.output_dir,
            confidence=args.confidence, iou_threshold=args.iou,
        )
    except (CalibrationError, DetectorAdapterError, ValueError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"aspect_ratio_bounds": str(aspect), "validation_pr": str(validation_pr)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
