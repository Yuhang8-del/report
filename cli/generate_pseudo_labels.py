"""Generate unfiltered, dual-view teacher candidates from a sealed unlabeled manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.detection.adapter import DetectorAdapterError
from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.pseudo.generator import PseudoGenerationError, PseudoLabelGenerator, load_unlabeled_manifest, write_pseudo_candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dual-view raw pseudo-label candidates from an explicit unlabeled manifest.")
    parser.add_argument("--unlabeled-manifest", type=Path, required=True, help="Task 8 unlabeled.json containing only no-label image records.")
    parser.add_argument("--split-manifest", type=Path, help="Paired Task 8 split_manifest.json; defaults to the file beside --unlabeled-manifest.")
    parser.add_argument("--weights", type=Path, required=True, help="Teacher checkpoint path passed to the detector adapter.")
    parser.add_argument("--teacher-run-id", required=True, help="Completed supervised teacher run ID retained in every candidate.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file for unfiltered candidates; it must not already exist.")
    parser.add_argument("--image-root", type=Path, required=True, help="Sealed base directory for relative image paths in the unlabeled manifest.")
    parser.add_argument("--confidence", type=float, help="Optional teacher confidence passed unchanged to the detector adapter.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        records = load_unlabeled_manifest(args.unlabeled_manifest.resolve(strict=False), split_manifest_path=args.split_manifest.resolve(strict=False) if args.split_manifest else None)
        detector = UltralyticsDetectorAdapter(weights_path=args.weights, source_model=str(args.weights))
        result = PseudoLabelGenerator(detector, teacher_run_id=args.teacher_run_id, confidence=args.confidence, image_root=args.image_root).generate(records)
        destination = write_pseudo_candidates(result, args.output)
    except (PseudoGenerationError, DetectorAdapterError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"output": str(destination), "teacher_run_id": result.teacher_run_id, "candidate_count": len(result.candidates)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
