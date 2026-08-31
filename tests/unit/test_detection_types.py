"""Tests for validated, serializable detector output records."""

from __future__ import annotations

import json

import pytest

from fruit_ssod.data.class_mapping import (
    CanonicalClass,
    ClassRegistry,
    DEFAULT_CLASS_REGISTRY,
)
from fruit_ssod.detection.types import DetectionRecord, DetectionValidationError


def valid_detection(**overrides: object) -> DetectionRecord:
    """Build a valid canonical detector record for focused validation tests."""
    values: dict[str, object] = {
        "class_id": 0,
        "class_name": "Apple",
        "confidence": 0.91,
        "xyxy": (1.0, 2.0, 30.0, 40.0),
        "is_unknown": False,
        "source_model": "best.pt",
    }
    values.update(overrides)
    return DetectionRecord(**values)  # type: ignore[arg-type]


def test_detection_record_is_json_serializable_and_uses_canonical_class_names() -> None:
    """Consumers receive primitive JSON values and stable fruit vocabulary."""
    record = valid_detection()

    assert record.to_dict() == {
        "class_id": 0,
        "class_name": "Apple",
        "confidence": 0.91,
        "xyxy": [1.0, 2.0, 30.0, 40.0],
        "is_unknown": False,
        "source_model": "best.pt",
    }
    assert json.loads(record.to_json()) == record.to_dict()


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("class_id", 9, "approved canonical class"),
        ("class_name", "apple", "does not match"),
        ("confidence", 1.2, "confidence"),
        ("confidence", float("nan"), "confidence"),
        ("xyxy", (1, 2, 1, 4), "positive area"),
        ("xyxy", (1, 2, float("inf"), 4), "finite"),
        ("is_unknown", "False", "boolean"),
        ("source_model", "", "nonempty"),
    ],
)
def test_detection_record_rejects_malformed_values(
    field_name: str, value: object, match: str
) -> None:
    """Bad model outputs become an actionable domain error before UI/export use."""
    with pytest.raises(DetectionValidationError, match=match) as error:
        valid_detection(**{field_name: value})

    assert "Problem:" in str(error.value)
    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_detection_record_rejects_a_noncanonical_custom_registry() -> None:
    """Callers cannot create or serialize a Mango detector record through registry injection."""
    mango_registry = ClassRegistry(
        version="test",
        classes=(
            CanonicalClass(0, "Apple"),
            CanonicalClass(1, "Banana"),
            CanonicalClass(2, "Orange"),
            CanonicalClass(3, "Strawberry"),
            CanonicalClass(4, "Mango"),
        ),
        source_aliases={},
    )

    with pytest.raises(DetectionValidationError, match="registry does not match") as error:
        valid_detection(registry=mango_registry)

    assert "Remediation:" in str(error.value)


def test_detection_record_rejects_unknowns_until_the_open_world_module_exists() -> None:
    """The reserved unknown field cannot claim unsupported first-version behavior."""
    with pytest.raises(DetectionValidationError, match="must be False") as error:
        valid_detection(is_unknown=True)

    assert "Remediation:" in str(error.value)


def test_detection_record_canonicalizes_an_equivalent_but_mutable_registry() -> None:
    """Record serialization remains stable if a caller later mutates their registry list."""
    external_classes = list(DEFAULT_CLASS_REGISTRY.classes)
    equivalent_registry = ClassRegistry(
        version="test",
        classes=external_classes,  # type: ignore[arg-type]
        source_aliases={},
    )
    record = valid_detection(registry=equivalent_registry)
    external_classes[-1] = CanonicalClass(4, "Mango")

    assert record.registry is DEFAULT_CLASS_REGISTRY
    assert json.loads(record.to_json())["class_name"] == "Apple"
