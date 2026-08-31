from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.pseudo.calibration import (
    CalibrationError,
    aspect_ratio_artifact,
    derive_aspect_ratio_bounds,
    validation_pr_records,
    write_calibration_artifacts,
)
from fruit_ssod.pseudo.trust_filter import load_aspect_ratio_bounds_artifact


def _record(image_id: str, labels: list[dict[str, object]]) -> dict[str, object]:
    return {"source_image_id": image_id, "file_path": f"images/{image_id}.jpg", "width": 100, "height": 100, "labels": labels}


def _labels() -> list[dict[str, object]]:
    return [{"class_id": class_id, "xyxy": [10.0, 10.0, 30.0 + class_id, 30.0]} for class_id in range(5)]


class _Detector:
    def predict(self, image: Path, *, confidence: float | None = None, nms_iou: float | None = None) -> tuple[DetectionRecord, ...]:
        assert confidence == 0.001 and nms_iou is None
        return (
            DetectionRecord(0, "Apple", .90, (10, 10, 30, 30), False, "fake"),
            DetectionRecord(0, "Apple", .80, (60, 60, 80, 80), False, "fake"),
            DetectionRecord(4, "Pineapple", .70, (10, 10, 34, 30), False, "fake"),
        )


def test_aspect_ratio_artifact_is_aggregate_only_and_uses_every_class() -> None:
    records = (_record("one", _labels()),)
    bounds = derive_aspect_ratio_bounds(records)
    assert bounds[0] == (1.0, 1.0)
    assert bounds[4] == (1.2, 1.2)
    payload = aspect_ratio_artifact(records, human_labels_sha256="a" * 64)
    assert set(payload) == {"artifact_version", "artifact_type", "artifact_id", "class_registry_version", "classes", "provenance", "bounds"}
    assert '"labels":' not in json.dumps(payload)


def test_aspect_ratio_calibration_requires_all_canonical_training_classes() -> None:
    with pytest.raises(CalibrationError, match="no observations"):
        derive_aspect_ratio_bounds((_record("one", [{"class_id": 0, "xyxy": [1, 1, 2, 2]}]),))


def test_validation_pr_is_confidence_ordered_one_to_one_class_aware(tmp_path: Path) -> None:
    root = tmp_path / "root"; (root / "images").mkdir(parents=True)
    (root / "images" / "one.jpg").write_bytes(b"image")
    output = validation_pr_records((_record("one", _labels()),), detector=_Detector(), image_root=root)
    assert output == [
        {"class_id": 0, "confidence": .9, "is_true_positive": True, "source_split": "validation"},
        {"class_id": 0, "confidence": .8, "is_true_positive": False, "source_split": "validation"},
        {"class_id": 4, "confidence": .7, "is_true_positive": True, "source_split": "validation"},
    ]


def test_writer_publishes_filter_compatible_artifacts_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"; (root / "images").mkdir(parents=True)
    (root / "images" / "one.jpg").write_bytes(b"image")
    human, validation = tmp_path / "human.json", tmp_path / "validation.json"
    human.write_text(json.dumps({"records": [_record("one", _labels())]}), encoding="utf-8")
    validation.write_text(json.dumps({"records": [_record("one", _labels())]}), encoding="utf-8")
    aspect, pr = write_calibration_artifacts(human_labels=human, validation_labels=validation, image_root=root, detector=_Detector(), teacher_run_id="teacher-42", output_dir=tmp_path / "calibration")
    assert load_aspect_ratio_bounds_artifact(aspect).bounds[4] == (1.2, 1.2)
    assert json.loads(pr.read_text(encoding="utf-8"))["records"][0]["source_split"] == "validation"
    with pytest.raises(CalibrationError, match="already exists"):
        write_calibration_artifacts(human_labels=human, validation_labels=validation, image_root=root, detector=_Detector(), teacher_run_id="teacher-42", output_dir=tmp_path / "calibration")
