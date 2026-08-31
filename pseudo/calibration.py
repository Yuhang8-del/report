"""Immutable, aggregate-only calibration inputs for the Trust Filter.

The two published artifacts deliberately have different purposes.  Aspect
ratio bounds are aggregated from the approved human *training* budget, while
confidence evidence contains only detector prediction/outcome rows from the
sealed validation split.  Neither export may contain a ground-truth box,
image identifier, or image path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapter
from fruit_ssod.detection.types import DetectionRecord


class CalibrationError(ValueError):
    """Raised when a sealed split cannot safely produce calibration data."""


def _problem(problem: str, cause: str, remediation: str) -> CalibrationError:
    return CalibrationError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_CLASS_IDS = tuple(item.id for item in DEFAULT_CLASS_REGISTRY.classes)
_CLASSES = [{"id": item.id, "name": item.name} for item in DEFAULT_CLASS_REGISTRY.classes]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _problem("calibration input cannot be read", str(error), "restore the sealed split manifest and retry") from error


def load_labeled_image_records(path: Path, *, purpose: str) -> tuple[Mapping[str, Any], ...]:
    """Load only the image-local labels required for one calibration operation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{purpose} labels cannot be read", str(error), "use an untouched Task 8 JSON labels artifact") from error
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list) or not records or any(not isinstance(row, Mapping) for row in records):
        raise _problem(f"{purpose} labels are malformed", "records is absent, empty, or contains a non-object", "use an untouched Task 8 JSON labels artifact")
    return tuple(records)


def _labels(record: Mapping[str, Any], *, purpose: str) -> tuple[tuple[int, tuple[float, float, float, float]], ...]:
    width, height, values = record.get("width"), record.get("height"), record.get("labels")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise _problem(f"{purpose} record has invalid image dimensions", repr(record.get("source_image_id")), "restore the sealed Task 8 labels artifact")
    if not isinstance(values, list) or not values:
        raise _problem(f"{purpose} record has no labels", repr(record.get("source_image_id")), "use a labeled Task 8 artifact, not an image-only manifest")
    parsed: list[tuple[int, tuple[float, float, float, float]]] = []
    for item in values:
        if not isinstance(item, Mapping) or isinstance(item.get("class_id"), bool) or not isinstance(item.get("class_id"), int) or item["class_id"] not in _CLASS_IDS:
            raise _problem(f"{purpose} label has an invalid class", repr(record.get("source_image_id")), "restore canonical IDs 0 through 4")
        box = item.get("xyxy")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise _problem(f"{purpose} label has malformed XYXY", repr(record.get("source_image_id")), "restore a four-coordinate bounding box")
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError) as error:
            raise _problem(f"{purpose} label has nonnumeric XYXY", repr(record.get("source_image_id")), "restore finite bounding-box coordinates") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise _problem(f"{purpose} label is outside image bounds", repr(record.get("source_image_id")), "restore the sealed Task 8 labels artifact")
        parsed.append((item["class_id"], (x1, y1, x2, y2)))
    return tuple(parsed)


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Deterministic linear percentile without a NumPy runtime dependency."""
    ordered = sorted(values)
    if not ordered:
        raise _problem("aggregate statistic has no observations", "one canonical class has no approved training labels", "choose a human-label budget containing all five fruits")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def derive_aspect_ratio_bounds(records: Sequence[Mapping[str, Any]]) -> dict[int, tuple[float, float]]:
    """Derive inclusive P01/P99 box width-to-height ranges for all five classes."""
    values: dict[int, list[float]] = {class_id: [] for class_id in _CLASS_IDS}
    for record in records:
        for class_id, (x1, y1, x2, y2) in _labels(record, purpose="human training"):
            values[class_id].append((x2 - x1) / (y2 - y1))
    bounds: dict[int, tuple[float, float]] = {}
    for class_id in _CLASS_IDS:
        low, high = _quantile(values[class_id], 0.01), _quantile(values[class_id], 0.99)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise _problem("derived aspect-ratio bounds are invalid", f"class {class_id} has {low!r}, {high!r}", "inspect the sealed human training labels")
        bounds[class_id] = (low, high)
    return bounds


def aspect_ratio_artifact(records: Sequence[Mapping[str, Any]], *, human_labels_sha256: str) -> dict[str, object]:
    """Return the exact label-free schema accepted by the Trust Filter."""
    bounds = derive_aspect_ratio_bounds(records)
    artifact_id = "train-pool-ar-p01-p99-" + hashlib.sha256(_canonical_bytes({"human_labels_sha256": human_labels_sha256, "method": "width_over_height_p01_p99"})).hexdigest()[:16]
    return {
        "artifact_version": "1.0", "artifact_type": "sealed_aspect_ratio_bounds", "artifact_id": artifact_id,
        "class_registry_version": DEFAULT_CLASS_REGISTRY.version, "classes": _CLASSES,
        "provenance": {"source_split": "train_pool", "source_kind": "approved_aggregate_statistics", "contains_human_labels": False, "sealed": True},
        "bounds": {str(class_id): [bounds[class_id][0], bounds[class_id][1]] for class_id in _CLASS_IDS},
    }


def _safe_image(root: Path, relative: object, *, image_id: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise _problem("validation image path is unsafe", repr(relative), "restore the sealed validation labels artifact")
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise _problem("validation image is unavailable below image root", f"{image_id!r}: {error}", "use the materialized Open Images source root paired with validation labels") from error
    if not candidate.is_file():
        raise _problem("validation image is not a regular file", repr(candidate), "restore the materialized validation image")
    return candidate


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in first)
    a1, b1, a2, b2 = (float(value) for value in second)
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union > 0 else 0.0


def validation_pr_records(records: Sequence[Mapping[str, Any]], *, detector: DetectorAdapter, image_root: Path, confidence: float = 0.001, iou_threshold: float = 0.50) -> list[dict[str, object]]:
    """Match validation-only predictions greedily by class and descending confidence."""
    if not (isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and 0 <= float(confidence) <= 1):
        raise _problem("prediction confidence is invalid", repr(confidence), "provide a finite confidence in [0, 1]")
    if not (isinstance(iou_threshold, (int, float)) and not isinstance(iou_threshold, bool) and math.isfinite(iou_threshold) and 0 < float(iou_threshold) <= 1):
        raise _problem("matching IoU is invalid", repr(iou_threshold), "provide a finite IoU in (0, 1]")
    root = image_root.resolve(strict=True)
    output: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: str(item.get("source_image_id", ""))):
        truth = _labels(record, purpose="validation")
        image = _safe_image(root, record.get("file_path"), image_id=record.get("source_image_id"))
        predictions = detector.predict(image, confidence=float(confidence))
        if any(not isinstance(item, DetectionRecord) for item in predictions):
            raise _problem("detector returned an invalid validation prediction", repr(record.get("source_image_id")), "use the canonical detector adapter")
        for class_id in _CLASS_IDS:
            unmatched = [box for truth_class, box in truth if truth_class == class_id]
            candidates = sorted((item for item in predictions if item.class_id == class_id), key=lambda item: (-item.confidence, item.xyxy))
            for prediction in candidates:
                best_index, best_iou = -1, 0.0
                for index, box in enumerate(unmatched):
                    value = _iou(prediction.xyxy, box)
                    if value > best_iou:
                        best_index, best_iou = index, value
                matched = best_index >= 0 and best_iou >= float(iou_threshold)
                if matched:
                    unmatched.pop(best_index)
                output.append({"class_id": class_id, "confidence": prediction.confidence, "is_true_positive": matched, "source_split": "validation"})
    return output


def validation_pr_artifact(records: Sequence[Mapping[str, Any]], *, detector: DetectorAdapter, image_root: Path, teacher_run_id: str, validation_labels_sha256: str, confidence: float = 0.001, iou_threshold: float = 0.50) -> dict[str, object]:
    if not isinstance(teacher_run_id, str) or not teacher_run_id.strip():
        raise _problem("teacher run ID is missing", "validation prediction provenance would be untraceable", "supply the completed supervised Teacher run ID")
    return {
        "schema_version": "1.0", "artifact_type": "validation_pr_records", "teacher_run_id": teacher_run_id,
        "validation_labels_sha256": validation_labels_sha256, "matching": {"iou_threshold": float(iou_threshold), "prediction_confidence": float(confidence)},
        "records": validation_pr_records(records, detector=detector, image_root=image_root, confidence=confidence, iou_threshold=iou_threshold),
    }


def write_calibration_artifacts(*, human_labels: Path, validation_labels: Path, image_root: Path, detector: DetectorAdapter, teacher_run_id: str, output_dir: Path, confidence: float = 0.001, iou_threshold: float = 0.50) -> tuple[Path, Path]:
    """Publish both calibration inputs atomically into a previously absent directory."""
    destination = output_dir.resolve(strict=False)
    if destination.exists():
        raise _problem("calibration output already exists", str(destination), "choose a new output directory; immutable calibration artifacts are never overwritten")
    human_records = load_labeled_image_records(human_labels, purpose="human training")
    validation_records = load_labeled_image_records(validation_labels, purpose="validation")
    aspect = aspect_ratio_artifact(human_records, human_labels_sha256=_sha256(human_labels))
    validation_pr = validation_pr_artifact(validation_records, detector=detector, image_root=image_root, teacher_run_id=teacher_run_id, validation_labels_sha256=_sha256(validation_labels), confidence=confidence, iou_threshold=iou_threshold)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
        aspect_path, pr_path = temporary / "fixed_aspect_ratio_bounds.json", temporary / "fixed_validation_pr.json"
        aspect_path.write_text(json.dumps(aspect, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pr_path.write_text(json.dumps(validation_pr, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination); temporary = None
        return destination / aspect_path.name, destination / pr_path.name
    except OSError as error:
        raise _problem("calibration artifacts could not be published", str(error), "choose a new writable artifact directory") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
