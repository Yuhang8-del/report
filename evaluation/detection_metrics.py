"""Strict, framework-neutral serialization of object-detection metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY


class DetectionMetricsError(ValueError):
    """Raised when model evaluation output cannot be used as scientific evidence."""


def _problem(problem: str, cause: str, remediation: str) -> DetectionMetricsError:
    return DetectionMetricsError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= float(value) <= 1:
        raise _problem(f"{name} must be a finite value from 0 to 1", "the evaluator returned malformed metrics", "return normalized detection metrics before serializing")
    return float(value)


@dataclass(frozen=True)
class DetectionMetrics:
    """Result summary used by run records and later aggregation/reporting stages."""

    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    per_class_ap50: Mapping[int, float]

    def __post_init__(self) -> None:
        for name in ("map50", "map50_95", "precision", "recall", "f1"):
            object.__setattr__(self, name, _metric(getattr(self, name), name))
        expected = DEFAULT_CLASS_REGISTRY.class_ids
        if not isinstance(self.per_class_ap50, Mapping) or set(self.per_class_ap50) != expected:
            raise _problem("per_class_ap50 does not match the canonical five classes", "a class was dropped, added, or remapped", "store AP50 for exact IDs 0 through 4")
        if any(isinstance(key, bool) or not isinstance(key, int) for key in self.per_class_ap50):
            raise _problem("per_class_ap50 has invalid class IDs", "metric keys are not canonical integers", "use canonical integer IDs 0 through 4")
        object.__setattr__(self, "per_class_ap50", MappingProxyType({key: _metric(value, f"per_class_ap50[{key}]") for key, value in self.per_class_ap50.items()}))

    def mapping(self) -> dict[str, Any]:
        return {
            "map50": self.map50, "map50_95": self.map50_95, "precision": self.precision,
            "recall": self.recall, "f1": self.f1,
            "per_class_ap50": {str(key): value for key, value in sorted(self.per_class_ap50.items())},
        }

    def to_json(self) -> str:
        return json.dumps(self.mapping(), sort_keys=True, ensure_ascii=False, allow_nan=False)


def metrics_from_mapping(value: Mapping[str, Any]) -> DetectionMetrics:
    """Decode persisted evaluator results and reject omissions hidden by defaults."""
    try:
        per_class = value["per_class_ap50"]
        if not isinstance(per_class, Mapping):
            raise TypeError("per_class_ap50 is not an object")
        # JSON persistence uses the exact decimal canonical IDs.  Do not use
        # a permissive ``int(key)`` conversion here: aliases such as ``"00"``
        # and ``"+1"`` would otherwise make a manually altered record look
        # canonical after parsing.
        expected_keys = {str(class_id) for class_id in DEFAULT_CLASS_REGISTRY.class_ids}
        if set(per_class) != expected_keys:
            raise TypeError(f"per_class_ap50 keys must be exactly {sorted(expected_keys)!r}")
        normalized = {int(key): metric for key, metric in per_class.items()}
        return DetectionMetrics(value["map50"], value["map50_95"], value["precision"], value["recall"], value["f1"], normalized)
    except (KeyError, TypeError, ValueError, DetectionMetricsError) as error:
        if isinstance(error, DetectionMetricsError):
            raise
        raise _problem("serialized evaluation metrics are malformed", str(error), "write all required overall and five-class AP50 values") from error
