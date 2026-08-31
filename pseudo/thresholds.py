"""Validation-derived, bounded confidence thresholds for pseudo labels.

The validation input deliberately contains only prediction/outcome rows.  It
is never a label manifest, and it is not accepted for audit or test splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY


class ThresholdSelectionError(ValueError):
    """Raised when validation precision evidence is malformed or unsafe."""


def _problem(problem: str, cause: str, remediation: str) -> ThresholdSelectionError:
    return ThresholdSelectionError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_CLASS_IDS = frozenset(item.id for item in DEFAULT_CLASS_REGISTRY.classes)


def _score(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _problem(f"{field} must be a finite number", "validation PR evidence contains a non-numeric score", f"provide a finite {field} in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise _problem(f"{field} must be in [0, 1]", "validation PR evidence contains an invalid probability", f"provide a {field} in [0, 1]")
    return result


@dataclass(frozen=True)
class ValidationPRRecord:
    """One validation prediction matched against validation-only ground truth."""

    class_id: int
    confidence: float
    is_true_positive: bool
    source_split: str = "validation"

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id not in _CLASS_IDS:
            raise _problem("class_id is not one of the five canonical fruits", "validation evidence has an incompatible detector class", "supply class IDs 0 through 4 only")
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if not isinstance(self.is_true_positive, bool):
            raise _problem("is_true_positive must be boolean", "validation evidence does not identify prediction correctness", "supply explicit boolean precision-recall match outcomes")
        if self.source_split != "validation":
            raise _problem("threshold evidence is not validation data", f"received source_split={self.source_split!r}", "derive thresholds only from the sealed validation prediction/outcome export")

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ValidationPRRecord":
        allowed = {"class_id", "confidence", "is_true_positive", "source_split"}
        if set(row) - allowed:
            raise _problem("validation PR record has unsupported fields", "generic annotations or label-bearing inputs were supplied", "supply only class_id, confidence, is_true_positive, and source_split")
        try:
            return cls(
                class_id=row["class_id"],  # type: ignore[arg-type]
                confidence=row["confidence"],  # type: ignore[arg-type]
                is_true_positive=row["is_true_positive"],  # type: ignore[arg-type]
                source_split=row.get("source_split", "validation"),  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise _problem("validation PR record is incomplete", str(error), "supply class_id, confidence, and is_true_positive") from error


@dataclass(frozen=True)
class PerClassThresholds:
    """Immutable complete five-class confidence threshold table."""

    values: Mapping[int, float]
    target_precision: float = 0.90
    minimum: float = 0.50
    maximum: float = 0.85

    def __post_init__(self) -> None:
        target_precision = _score(self.target_precision, "target_precision")
        if target_precision <= 0.0:
            raise _problem("target_precision must be greater than zero", "a zero precision target has no reliability meaning", "use a target_precision in (0, 1]")
        minimum = _score(self.minimum, "minimum")
        maximum = _score(self.maximum, "maximum")
        if minimum > maximum:
            raise _problem("threshold clamp is inverted", "minimum exceeds maximum", "use minimum <= maximum")
        if not isinstance(self.values, Mapping) or set(self.values) != _CLASS_IDS:
            raise _problem("per-class threshold table is incomplete", "one or more canonical fruit classes are missing", "provide exactly thresholds for IDs 0, 1, 2, 3, and 4")
        normalized: dict[int, float] = {}
        for class_id, value in self.values.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in _CLASS_IDS:
                raise _problem("per-class threshold table has an invalid class ID", "the table does not match the fixed taxonomy", "provide integer class IDs 0 through 4")
            threshold = _score(value, f"threshold for class {class_id}")
            if not minimum <= threshold <= maximum:
                raise _problem("per-class threshold violates the configured clamp", f"class {class_id} has {threshold}", f"keep every threshold in [{minimum:.2f}, {maximum:.2f}]")
            normalized[class_id] = threshold
        object.__setattr__(self, "values", MappingProxyType(normalized))
        object.__setattr__(self, "target_precision", target_precision)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def for_class(self, class_id: int) -> float:
        try:
            return self.values[class_id]
        except KeyError as error:
            raise _problem("class threshold is missing", f"class {class_id} has no validated threshold", "use a complete PerClassThresholds table") from error

    def mapping(self) -> dict[str, object]:
        return {"target_precision": self.target_precision, "minimum": self.minimum, "maximum": self.maximum, "thresholds": {str(key): self.values[key] for key in sorted(self.values)}}


def select_per_class_thresholds(
    records: Sequence[ValidationPRRecord | Mapping[str, object]], *, target_precision: float = 0.90,
    minimum: float = 0.50, maximum: float = 0.85,
) -> PerClassThresholds:
    """Choose the lowest empirical score that keeps each class near target precision.

    If a class cannot reach the target with available validation predictions,
    its conservative upper clamp is used.  Thresholds are always bounded to
    the agreed [0.50, 0.85] range, independently of input ordering.
    """
    target = _score(target_precision, "target_precision")
    if target <= 0.0:
        raise _problem("target_precision must be greater than zero", "a zero precision target has no reliability meaning", "use a target_precision in (0, 1]")
    lower, upper = _score(minimum, "minimum"), _score(maximum, "maximum")
    if lower > upper:
        raise _problem("threshold clamp is inverted", "minimum exceeds maximum", "use minimum <= maximum")
    parsed = tuple(item if isinstance(item, ValidationPRRecord) else ValidationPRRecord.from_mapping(item) for item in records)
    by_class: dict[int, list[ValidationPRRecord]] = {class_id: [] for class_id in _CLASS_IDS}
    for item in parsed:
        by_class[item.class_id].append(item)
    values: dict[int, float] = {}
    for class_id in sorted(_CLASS_IDS):
        # Consider only attainable finite score cutoffs, plus the mandated
        # boundary. Sorting removes detector/output order as a source of drift.
        candidates = sorted({lower, upper, *(min(upper, max(lower, item.confidence)) for item in by_class[class_id])})
        valid: list[float] = []
        for cutoff in candidates:
            retained = [item for item in by_class[class_id] if item.confidence >= cutoff]
            if retained and sum(item.is_true_positive for item in retained) / len(retained) >= target:
                valid.append(cutoff)
        values[class_id] = min(valid) if valid else upper
    return PerClassThresholds(values, target, lower, upper)


# A descriptive alias kept for report/CLI code.
select_thresholds_from_validation_pr = select_per_class_thresholds
