"""Strict, JSON-friendly records shared by detector backends and consumers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

from fruit_ssod.data.class_mapping import ClassRegistry, DEFAULT_CLASS_REGISTRY


class DetectionValidationError(ValueError):
    """Raised when a detector result cannot safely cross an application boundary."""


def _fail(problem: str, cause: str, remediation: str) -> None:
    """Raise a consistent, user-actionable detector record error."""
    raise DetectionValidationError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


def _finite_number(value: object, field_name: str) -> float:
    """Convert a finite non-boolean numeric value to a JSON-safe float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(
            f"{field_name} must be a finite number",
            "the model output contains a non-numeric, boolean, NaN, or infinity value",
            f"provide a finite numeric {field_name} value",
        )
    return float(value)


def canonical_class_mapping(registry: ClassRegistry = DEFAULT_CLASS_REGISTRY) -> dict[int, str]:
    """Return the fixed detector taxonomy and reject registry substitution attempts."""
    expected = {item.id: item.name for item in DEFAULT_CLASS_REGISTRY.classes}
    try:
        received = {item.id: item.name for item in registry.classes}
    except (AttributeError, TypeError) as error:
        _fail(
            "registry does not match the canonical detector registry",
            "the supplied registry does not expose valid class definitions",
            "use the committed five-class canonical registry",
        )
        raise AssertionError("unreachable") from error
    if received != expected or len(registry.classes) != len(DEFAULT_CLASS_REGISTRY.classes):
        _fail(
            "registry does not match the canonical detector registry",
            f"expected {expected!r}, received {received!r}",
            "use exactly IDs 0-4 for Apple, Banana, Orange, Strawberry, and Pineapple",
        )
    return expected


@dataclass(frozen=True)
class DetectionRecord:
    """One validated known-fruit detection, ready for UI, metrics, or JSON export."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    is_unknown: bool
    source_model: str
    registry: ClassRegistry = DEFAULT_CLASS_REGISTRY

    def __post_init__(self) -> None:
        """Normalize primitive values and enforce the fixed five-class contract."""
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            _fail(
                "class_id must be a non-boolean integer",
                "the detector emitted an invalid class identifier",
                "configure the checkpoint with the approved canonical class registry",
            )
        class_by_id = canonical_class_mapping(self.registry)
        expected_name = class_by_id.get(self.class_id)
        if expected_name is None:
            _fail(
                "class_id is not an approved canonical class",
                "the checkpoint uses a class outside the five-fruit detector taxonomy",
                "use a checkpoint whose class IDs are 0-4 from the canonical registry",
            )
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            _fail(
                "class_name must be a nonempty string",
                "the detector did not provide a usable class label",
                "configure the checkpoint with the canonical class-name mapping",
            )
        if self.class_name != expected_name:
            _fail(
                "class_name does not match class_id",
                "the detector class mapping differs from the canonical registry",
                f"map class ID {self.class_id} to {expected_name!r}",
            )
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            _fail(
                "confidence must be between 0 and 1",
                "the model emitted a score outside probability bounds",
                "return a normalized confidence score in the inclusive range [0, 1]",
            )
        if not isinstance(self.xyxy, Sequence) or isinstance(self.xyxy, (str, bytes)) or len(self.xyxy) != 4:
            _fail(
                "xyxy must contain four coordinates",
                "the model output does not contain an XYXY bounding box",
                "return coordinates as (x1, y1, x2, y2)",
            )
        coordinates = tuple(_finite_number(value, "xyxy coordinate") for value in self.xyxy)
        x1, y1, x2, y2 = coordinates
        if not x1 < x2 or not y1 < y2:
            _fail(
                "xyxy must have positive area",
                "the predicted box has zero or negative width or height",
                "return a box with x1 < x2 and y1 < y2",
            )
        if not isinstance(self.is_unknown, bool):
            _fail(
                "is_unknown must be a boolean",
                "the open-world compatibility flag is not boolean",
                "set is_unknown to False for this known-class implementation",
            )
        if self.is_unknown:
            _fail(
                "is_unknown must be False in the first detector version",
                "open-world recognition is a reserved future interface and is not implemented yet",
                "set is_unknown to False until the open-world module is delivered",
            )
        if not isinstance(self.source_model, str) or not self.source_model.strip():
            _fail(
                "source_model must be a nonempty string",
                "the prediction cannot be traced to a model artifact",
                "provide a model filename, run identifier, or other nonempty provenance value",
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "xyxy", coordinates)
        # Do not retain a caller-owned registry whose classes/aliases may be mutable.
        object.__setattr__(self, "registry", DEFAULT_CLASS_REGISTRY)

    def to_dict(self) -> dict[str, object]:
        """Return only JSON primitives for downstream GUI, audit, and export code."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "is_unknown": self.is_unknown,
            "source_model": self.source_model,
        }

    def to_json(self) -> str:
        """Serialize the stable public representation without lossy coercion."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
