"""Strict JSON records for unfiltered dual-view pseudo-label candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fruit_ssod.detection.types import DetectionRecord


class PseudoCandidateError(ValueError):
    """Raised when raw pseudo-label evidence cannot be safely retained."""


def _problem(problem: str, cause: str, remediation: str) -> PseudoCandidateError:
    return PseudoCandidateError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} must be a nonempty string", "pseudo-label provenance omitted a required identifier", f"provide a nonempty {field}")
    return value


def _box(value: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise _problem(f"{field} must contain four coordinates", "raw detector evidence is not an XYXY box", "retain four finite XYXY coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise _problem(f"{field} must contain finite numeric coordinates", "raw detector evidence has non-finite or non-numeric coordinates", "retain finite numeric coordinates only")
    result = tuple(float(item) for item in value)
    if not result[0] < result[2] or not result[1] < result[3]:
        raise _problem(f"{field} must have positive area", "raw detector evidence has an inverted or empty box", "retain only x1 < x2 and y1 < y2 boxes")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class PseudoCandidate:
    """One raw teacher prediction, with both view-space and original-space boxes."""

    teacher_run_id: str
    source_image_id: str
    source_file_path: str
    view: str
    class_id: int
    class_name: str
    confidence: float
    raw_xyxy: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    source_model: str

    def __post_init__(self) -> None:
        for field in ("teacher_run_id", "source_image_id", "source_file_path", "source_model"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.view not in {"original", "horizontal_flip"}:
            raise _problem("view is not supported", "candidate provenance did not record the original or horizontal-flip view", "use exactly 'original' or 'horizontal_flip'")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or not 0 <= float(self.confidence) <= 1:
            raise _problem("confidence must be a finite probability", "teacher inference emitted an invalid score", "retain a finite confidence in [0, 1]")
        # Reconstructing this record guarantees the candidate preserves the
        # exact fixed five-class detector vocabulary rather than trusting a
        # hand-built id/name pair.
        DetectionRecord(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=float(self.confidence),
            xyxy=self.raw_xyxy,
            is_unknown=False,
            source_model=self.source_model,
        )
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "raw_xyxy", _box(self.raw_xyxy, "raw_xyxy"))
        object.__setattr__(self, "xyxy", _box(self.xyxy, "xyxy"))

    @classmethod
    def from_detection(
        cls,
        detection: DetectionRecord,
        *,
        teacher_run_id: str,
        source_image_id: str,
        source_file_path: str,
        view: str,
        xyxy: Sequence[float] | None = None,
    ) -> "PseudoCandidate":
        """Build a candidate from the validated detector boundary record."""
        return cls(
            teacher_run_id=teacher_run_id,
            source_image_id=source_image_id,
            source_file_path=source_file_path,
            view=view,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            raw_xyxy=detection.xyxy,
            xyxy=tuple(detection.xyxy if xyxy is None else xyxy),
            source_model=detection.source_model,
        )

    def mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation for later filtering."""
        return {
            "teacher_run_id": self.teacher_run_id,
            "source_image_id": self.source_image_id,
            "source_file_path": self.source_file_path,
            "view": self.view,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "raw_xyxy": list(self.raw_xyxy),
            "xyxy": list(self.xyxy),
            "source_model": self.source_model,
        }

    def to_json(self) -> str:
        return json.dumps(self.mapping(), ensure_ascii=False, sort_keys=True, allow_nan=False)
