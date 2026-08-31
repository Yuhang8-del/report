"""Recover validation-only evidence after a post-training Student runner fault.

The original run record remains immutable and failed.  This command never
changes it, never accepts a test split, and writes a separately identifiable
recovery envelope for the already-published ``weights/best.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError, validate_class_mapping
from fruit_ssod.training.run_record import RunRecordError, read_run_record
from fruit_ssod.training.supervised import SupervisedTrainingError, _serialize_metric_object, file_evidence


def _problem(problem: str, cause: str, remediation: str) -> SupervisedTrainingError:
    return SupervisedTrainingError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create validation-only recovery evidence for a preserved failed Student checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        run_dir = args.run_dir.resolve(strict=True)
        record = read_run_record(run_dir / "run_record.json")
        if record.status != "failed":
            raise _problem("validation recovery is only for a preserved failed run", f"run status is {record.status!r}", "use the normal evaluator for a completed run")
        snapshot = record.config_snapshot
        student = snapshot.get("student_dataset") if isinstance(snapshot, Mapping) else None
        if not isinstance(student, Mapping) or not isinstance(student.get("dataset_yaml"), str) or not isinstance(snapshot.get("image_size"), int):
            raise _problem("failed run lacks a sealed Student dataset", "config_snapshot has no Student dataset YAML or image size", "recover only a run created by train_student")
        data = Path(student["dataset_yaml"]).resolve(strict=True)
        weights = run_dir / "weights" / "best.pt"
        checkpoint = file_evidence(weights, description="recovered Student best checkpoint")
        output = run_dir / "evaluations" / "validation_recovery.json"
        if output.exists():
            raise _problem("validation recovery artifact already exists", str(output), "preserve immutable evidence and use the existing artifact")
        from ultralytics import YOLO
        model = YOLO(str(weights))
        validate_class_mapping(getattr(model, "names", None), DEFAULT_CLASS_REGISTRY)
        validation = model.val(data=str(data), split="val", imgsz=snapshot["image_size"], device=args.device, plots=False, verbose=False)
        metrics = _serialize_metric_object(validation).mapping()
        raw = getattr(validation, "results_dict", None)
        if not isinstance(raw, Mapping):
            raise _problem("Ultralytics validation has no raw result mapping", "the installed backend returned an incompatible result", "use a supported Ultralytics version")
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "student_validation_recovery_v1",
            "run_id": record.run_id,
            "original_run_status": record.status,
            "split": "validation",
            "checkpoint": checkpoint,
            "dataset_yaml": str(data),
            "dataset_yaml_sha256": student.get("dataset_yaml_sha256"),
            "metrics": metrics,
            "raw_results_dict": dict(raw),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (RunRecordError, SupervisedTrainingError, DetectorAdapterError, OSError, ValueError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"output": str(output), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
