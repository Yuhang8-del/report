"""Tests for the immutable Fruit SSOD class registry and annotation schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.data.class_mapping import (
    ClassMappingError,
    DEFAULT_CLASS_REGISTRY,
    load_class_registry,
    resolve_class_id,
)
from fruit_ssod.data.schema import (
    AnnotationValidationError,
    CanonicalAnnotation,
    LicenseMetadata,
    UnlabeledImageRecord,
)


def annotation_kwargs(**overrides: object) -> dict[str, object]:
    """Return a valid object annotation, with opt-in fields for negative tests."""
    values: dict[str, object] = {
        "source": "canonical",
        "source_category": "Apple",
        "source_image_id": "image-1",
        "file_path": "images/example.jpg",
        "width": 100,
        "height": 80,
        "class_id": 0,
        "xyxy": (10.0, 5.0, 90.0, 40.0),
        "split": "train_pool",
        "label_status": "labeled",
        "license_metadata": LicenseMetadata(name="CC BY 4.0"),
    }
    values.update(overrides)
    return values


def build_annotation(**overrides: object) -> CanonicalAnnotation:
    """Instantiate an object annotation while keeping test setup concise."""
    return CanonicalAnnotation(**annotation_kwargs(**overrides))  # type: ignore[arg-type]


def test_registry_has_the_fixed_canonical_class_ids_and_order() -> None:
    """The detector's class IDs remain stable across all data sources."""
    assert DEFAULT_CLASS_REGISTRY.version == "1.0.1"
    assert [(item.id, item.name) for item in DEFAULT_CLASS_REGISTRY.classes] == [
        (0, "Apple"),
        (1, "Banana"),
        (2, "Orange"),
        (3, "Strawberry"),
        (4, "Pineapple"),
    ]


@pytest.mark.parametrize(
    ("source", "label", "expected_id"),
    [
        ("canonical", "Apple", 0),
        ("canonical", "apple", 0),
        ("fruit_360", "Pineapple", 4),
        ("common_fruit_labels", "pineapples", 4),
    ],
)
def test_known_source_aliases_resolve_deterministically(
    source: str, label: str, expected_id: int
) -> None:
    """Only aliases explicitly approved for their source are accepted."""
    assert resolve_class_id(source, label) == expected_id


def test_unknown_source_or_unapproved_alias_has_an_actionable_error() -> None:
    """A missing source map or distinct fruit variety cannot silently relabel data."""
    with pytest.raises(ClassMappingError, match="Unknown source") as source_error:
        resolve_class_id("unreviewed_dataset", "Apple")

    with pytest.raises(ClassMappingError, match="Unknown category") as label_error:
        resolve_class_id("fruit_360", "Granny Smith")

    for error in (source_error.value, label_error.value):
        assert "Problem:" in str(error)
        assert "Likely cause:" in str(error)
        assert "Remediation:" in str(error)


def test_pineapple_is_retained_when_an_external_source_does_not_supply_it() -> None:
    """A source's limited taxonomy never removes a canonical detector class."""
    assert resolve_class_id("limited_external_set", "Apple") == 0
    assert [item.name for item in DEFAULT_CLASS_REGISTRY.classes][-1] == "Pineapple"
    with pytest.raises(ClassMappingError, match="Pineapple"):
        resolve_class_id("limited_external_set", "Pineapple")


@pytest.mark.parametrize(
    ("source_category", "expected_id"),
    [
        ("Apple", 0),
        ("Banana", 1),
        ("Orange", 2),
        ("Orange (fruit)", 2),
        ("Strawberry", 3),
        ("Pineapple", 4),
        ("apple", 0),
        ("banana", 1),
        ("orange", 2),
        ("strawberry", 3),
        ("pineapple", 4),
    ],
)
def test_shipped_registry_maps_only_approved_open_images_v7_labels(
    source_category: str, expected_id: int
) -> None:
    """Open Images canonical display labels resolve through the shipped registry."""
    registry_path = Path(__file__).parents[2] / "configs" / "class_registry.json"
    registry = load_class_registry(registry_path)

    assert resolve_class_id("open_images_v7", source_category, registry) == expected_id
    with pytest.raises(ClassMappingError, match="Unknown category") as error:
        resolve_class_id("open_images_v7", "Apple Red", registry)

    assert "Problem:" in str(error.value)
    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_annotation_schema_validates_fixture_and_rejects_invalid_box() -> None:
    """Fixture annotations instantiate without image files and boxes stay in bounds."""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "annotations" / "sample_annotations.json"
    rows = json.loads(fixture_path.read_text(encoding="utf-8"))

    annotations = [CanonicalAnnotation.from_mapping(row) for row in rows]

    assert annotations[0].class_id == 0
    assert annotations[-1].class_id == 4
    with pytest.raises(AnnotationValidationError, match="in bounds"):
        build_annotation(xyxy=(10.0, 5.0, 101.0, 40.0))


@pytest.mark.parametrize("invalid_class_id", [0.0, [0]])
def test_annotation_schema_rejects_non_integer_or_unhashable_class_ids(
    invalid_class_id: object,
) -> None:
    """Invalid IDs produce an actionable domain error before registry lookup."""
    with pytest.raises(AnnotationValidationError, match="class_id must be a non-bool integer") as error:
        build_annotation(class_id=invalid_class_id)

    assert "Problem:" in str(error.value)
    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_annotation_class_id_must_match_its_explicit_source_category() -> None:
    """A caller cannot bypass reviewed aliases by supplying a different class ID."""
    annotation = build_annotation(source_category="apple", class_id=0)

    assert annotation.class_id == 0
    with pytest.raises(AnnotationValidationError, match="does not match source/category"):
        build_annotation(class_id=4)
    with pytest.raises(AnnotationValidationError, match="Pineapple"):
        build_annotation(
            source="limited_external_set",
            source_category="Pineapple",
            class_id=4,
        )


@pytest.mark.parametrize(
    ("split", "label_status"),
    [
        ("train_pool", "labeled"),
        ("train_pool", "pseudo"),
        ("validation", "labeled"),
        ("test", "labeled"),
        ("pseudo_audit", "pseudo"),
        ("external_test", "labeled"),
    ],
)
def test_annotation_accepts_only_safe_labeled_and_pseudo_split_pairs(
    split: str, label_status: str
) -> None:
    """Every approved evaluation split remains expressible without pseudo leakage."""
    assert build_annotation(split=split, label_status=label_status).split == split


@pytest.mark.parametrize(
    ("split", "label_status"),
    [
        ("validation", "pseudo"),
        ("test", "pseudo"),
        ("external_test", "pseudo"),
        ("pseudo_audit", "labeled"),
        ("validation", "unlabeled"),
    ],
)
def test_annotation_rejects_unsafe_split_status_pairs(split: str, label_status: str) -> None:
    """Evaluation partitions cannot be contaminated by pseudo or unlabeled objects."""
    with pytest.raises(AnnotationValidationError, match="not an allowed object-annotation combination"):
        build_annotation(split=split, label_status=label_status)


def test_unlabeled_image_record_preserves_the_unlabeled_train_pool_state() -> None:
    """Unlabeled images have no object class or box and remain in the training pool."""
    record = UnlabeledImageRecord(
        source="canonical",
        source_image_id="unlabeled-1",
        file_path="images/unlabeled.jpg",
        width=100,
        height=80,
        split="train_pool",
        label_status="unlabeled",
        license_metadata=LicenseMetadata(name="CC BY 4.0"),
    )

    assert record.label_status == "unlabeled"
    with pytest.raises(AnnotationValidationError, match="not an allowed unlabeled-image combination"):
        UnlabeledImageRecord(
            source="canonical",
            source_image_id="unlabeled-2",
            file_path="images/unlabeled.jpg",
            width=100,
            height=80,
            split="test",
            label_status="unlabeled",
            license_metadata=LicenseMetadata(name="CC BY 4.0"),
        )


@pytest.mark.parametrize(("field_name", "invalid_value"), [("split", []), ("label_status", {})])
def test_schema_rejects_non_string_controlled_values(
    field_name: str, invalid_value: object
) -> None:
    """JSON arrays and objects fail as domain errors before set membership checks."""
    with pytest.raises(AnnotationValidationError, match=f"{field_name} must be a nonempty string"):
        build_annotation(**{field_name: invalid_value})


@pytest.mark.parametrize(("field_name", "invalid_value"), [("id", True), ("id", 0.0), ("name", ""), ("name", 1)])
def test_registry_rejects_malformed_class_definitions_before_comparison(
    tmp_path: Path, field_name: str, invalid_value: object
) -> None:
    """Bad primitive types in JSON cannot reach the fixed-class equality check."""
    registry_path = Path(__file__).parents[2] / "configs" / "class_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["classes"][0][field_name] = invalid_value
    corrupt_path = tmp_path / "class_registry.json"
    corrupt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ClassMappingError, match="class definition is malformed") as error:
        load_class_registry(corrupt_path)

    assert "Problem:" in str(error.value)
    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)
