"""Audit-only dual-view candidate generation for the sealed pseudo-audit set.

This module never participates in Student dataset composition.  It receives
only the already-sealed image metadata exposed by the audit module; labels are
used later by the audit scorer, never by teacher inference or filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fruit_ssod.detection.adapter import DetectorAdapter, DetectorAdapterError
from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.evaluation.pseudo_metrics import AuditImage
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.generator import PseudoGenerationError, PseudoGenerationResult
from fruit_ssod.pseudo.transforms import TransformError, horizontal_flip_image, horizontal_flip_xyxy


class AuditCandidateError(RuntimeError):
    """Raised when audit-only inference cannot preserve sealed image provenance."""


def _problem(problem: str, cause: str, remediation: str) -> AuditCandidateError:
    return AuditCandidateError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _load_image(path: Path) -> object:
    try:
        from PIL import Image

        with Image.open(path) as opened:
            opened.load()
            return opened.copy()
    except (OSError, ValueError) as error:
        raise _problem("pseudo-audit image cannot be decoded", str(error), "restore a Pillow-readable image below the approved audit image root") from error


def _safe_image(root: Path, image: AuditImage) -> Path:
    relative = Path(image.file_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise _problem("pseudo-audit image path is unsafe", repr(image.file_path), "restore the sealed pseudo-audit image record")
    unresolved = root / relative
    component = root
    for part in relative.parts:
        component = component / part
        if component.is_symlink():
            raise _problem("pseudo-audit image path redirects through a symbolic link", repr(image.file_path), "materialize a regular image directly below the audit image root")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise _problem("pseudo-audit image is unavailable below image root", str(error), "use the materialized image root paired with Task 8 audit images") from error
    if not path.is_file():
        raise _problem("pseudo-audit image is not a regular file", str(path), "restore the materialized audit image")
    return path


def _size(image: object) -> tuple[int, int] | None:
    value = getattr(image, "size", None)
    if not isinstance(value, tuple) or len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        return None
    return value


def _validated_box(detection: DetectionRecord, image: AuditImage) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = detection.xyxy
    if x1 < 0 or y1 < 0 or x2 > image.width or y2 > image.height:
        raise _problem("teacher box lies outside pseudo-audit image", image.source_image_id, "use an input-resolution-preserving detector checkpoint")
    return detection.xyxy


def _predict(detector: DetectorAdapter, image: object, confidence: float | None) -> tuple[DetectionRecord, ...]:
    try:
        records = detector.predict(image, confidence=confidence)
    except DetectorAdapterError as error:
        raise _problem("pseudo-audit teacher inference failed", str(error), "verify the checkpoint, adapter, and audit image") from error
    if not isinstance(records, tuple) or any(not isinstance(item, DetectionRecord) for item in records):
        raise _problem("pseudo-audit detector output is invalid", "the adapter did not return validated DetectionRecord objects", "use the canonical DetectorAdapter implementation")
    return records


def generate_audit_candidates(
    images: Mapping[str, AuditImage], *, detector: DetectorAdapter, teacher_run_id: str,
    image_root: Path, confidence: float | None = None,
) -> PseudoGenerationResult:
    """Run original/flip teacher inference using only sealed audit image metadata."""
    if not isinstance(teacher_run_id, str) or not teacher_run_id.strip():
        raise _problem("pseudo-audit teacher run ID is missing", "candidate provenance would be untraceable", "supply a completed Teacher run ID")
    if not isinstance(detector, DetectorAdapter):
        raise _problem("pseudo-audit detector is invalid", "a raw backend could bypass canonical output validation", "use the project DetectorAdapter")
    try:
        root = image_root.resolve(strict=True)
    except OSError as error:
        raise _problem("pseudo-audit image root cannot be resolved", str(error), "use an existing materialized image root") from error
    candidates: list[PseudoCandidate] = []
    for source_image_id, metadata in sorted(images.items()):
        if source_image_id != metadata.source_image_id:
            raise _problem("pseudo-audit image mapping is inconsistent", repr(source_image_id), "reload the sealed pseudo-audit artifact")
        image = _load_image(_safe_image(root, metadata))
        if _size(image) != (metadata.width, metadata.height):
            raise _problem("pseudo-audit image dimensions differ from sealed metadata", metadata.source_image_id, "restore the exact materialized Task 8 audit image")
        try:
            flipped = horizontal_flip_image(image)
        except TransformError as error:
            raise _problem("pseudo-audit image cannot be horizontally flipped", str(error), "restore a valid image and retry") from error
        if _size(flipped) != (metadata.width, metadata.height):
            raise _problem("pseudo-audit horizontal flip changed dimensions", metadata.source_image_id, "use a dimension-preserving image transform")
        for detection in _predict(detector, image, confidence):
            candidates.append(PseudoCandidate.from_detection(
                detection, teacher_run_id=teacher_run_id, source_image_id=metadata.source_image_id,
                source_file_path=metadata.file_path, view="original", xyxy=_validated_box(detection, metadata),
            ))
        for detection in _predict(detector, flipped, confidence):
            try:
                original = horizontal_flip_xyxy(detection.xyxy, width=metadata.width)
                _validated_box(DetectionRecord(
                    class_id=detection.class_id, class_name=detection.class_name, confidence=detection.confidence,
                    xyxy=original, is_unknown=False, source_model=detection.source_model,
                ), metadata)
            except (TransformError, ValueError) as error:
                raise _problem("flipped pseudo-audit teacher box cannot map to original coordinates", str(error), "use unscaled teacher coordinates") from error
            candidates.append(PseudoCandidate.from_detection(
                detection, teacher_run_id=teacher_run_id, source_image_id=metadata.source_image_id,
                source_file_path=metadata.file_path, view="horizontal_flip", xyxy=original,
            ))
    try:
        return PseudoGenerationResult(teacher_run_id, tuple(candidates))
    except PseudoGenerationError as error:
        raise _problem("pseudo-audit candidates cannot be sealed", str(error), "regenerate from the sealed audit image metadata") from error
