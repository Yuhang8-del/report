"""Validated canonical annotation records for Fruit SSOD data curation.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from fruit_ssod.data.class_mapping import (
    ClassMappingError,
    ClassRegistry,
    DEFAULT_CLASS_REGISTRY,
    resolve_class_id,
)


ALLOWED_SPLITS = frozenset({"train_pool", "validation", "test", "pseudo_audit", "external_test"})
ALLOWED_LABEL_STATUSES = frozenset({"labeled", "unlabeled", "pseudo"})
ALLOWED_OBJECT_SPLIT_STATUSES = frozenset(
    {
        ("train_pool", "labeled"),
        ("train_pool", "pseudo"),
        ("validation", "labeled"),
        ("test", "labeled"),
        ("pseudo_audit", "pseudo"),
        ("external_test", "labeled"),
    }
)
ALLOWED_UNLABELED_IMAGE_SPLIT_STATUSES = frozenset({("train_pool", "unlabeled")})


class AnnotationValidationError(ValueError):
    """Raised when an annotation cannot safely enter the canonical data model."""


def _fail(problem: str, cause: str, remediation: str) -> None:
    """Raise a consistently actionable annotation validation failure."""
    raise AnnotationValidationError(
        f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."
    )


def _nonempty_text(value: object, field_name: str) -> str:
    """Require identifiers and paths to be meaningful text values."""
    if not isinstance(value, str) or not value.strip():
        _fail(
            f"{field_name} must be a nonempty string",
            "the source record omitted an identifier or path",
            f"provide a nonempty {field_name} value",
        )
    return value


def _controlled_text(value: object, field_name: str, allowed_values: frozenset[str]) -> str:
    """Validate a controlled string before safely testing it for membership."""
    value = _nonempty_text(value, field_name)
    if value not in allowed_values:
        _fail(
            f"{field_name} {value!r} is not controlled",
            "the record has an unsupported workflow value",
            f"use one of {sorted(allowed_values)}",
        )
    return value


@dataclass(frozen=True)
class LicenseMetadata:
    """License facts retained with each canonical annotation."""

    name: str
    url: str | None = None
    attribution: str | None = None

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "license_metadata.name")
        for field_name, value in (("license_metadata.url", self.url), ("license_metadata.attribution", self.attribution)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                _fail(
                    f"{field_name} must be nonempty when provided",
                    "license metadata contains an empty optional value",
                    "omit the optional value or provide meaningful text",
                )


@dataclass(frozen=True)
class CanonicalAnnotation:
    """One image-localized fruit label expressed in the canonical detector schema."""

    source: str
    source_category: str
    source_image_id: str
    file_path: str
    width: int
    height: int
    class_id: int
    xyxy: tuple[float, float, float, float]
    split: str
    label_status: str
    license_metadata: LicenseMetadata
    registry: ClassRegistry = field(default=DEFAULT_CLASS_REGISTRY, repr=False, compare=False)

    def __post_init__(self) -> None:
        _nonempty_text(self.source, "source")
        _nonempty_text(self.source_category, "source_category")
        _nonempty_text(self.source_image_id, "source_image_id")
        _nonempty_text(self.file_path, "file_path")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            _fail("width must be positive", "image width is absent or invalid", "provide a positive integer width")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            _fail("height must be positive", "image height is absent or invalid", "provide a positive integer height")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            _fail(
                "class_id must be a non-bool integer",
                "the annotation ID has an invalid type",
                "provide one of the integer IDs defined by the class registry",
            )
        try:
            derived_class_id = resolve_class_id(
                self.source, self.source_category, self.registry
            )
        except ClassMappingError as error:
            raise AnnotationValidationError(str(error)) from error
        if self.class_id != derived_class_id:
            _fail(
                "class_id does not match source/category",
                "the supplied ID differs from the reviewed source alias mapping",
                "use the class ID resolved from source and source_category",
            )
        split = _controlled_text(self.split, "split", ALLOWED_SPLITS)
        label_status = _controlled_text(
            self.label_status, "label_status", ALLOWED_LABEL_STATUSES
        )
        if (split, label_status) not in ALLOWED_OBJECT_SPLIT_STATUSES:
            _fail(
                "split and label_status are not an allowed object-annotation combination",
                "pseudo and unlabeled states are restricted to prevent evaluation contamination",
                f"use one of {sorted(ALLOWED_OBJECT_SPLIT_STATUSES)}",
            )
        self._validate_box()

    def _validate_box(self) -> None:
        """Ensure XYXY boxes are finite, bounded, and have positive area."""
        if not isinstance(self.xyxy, tuple) or len(self.xyxy) != 4:
            _fail("xyxy must contain four coordinates", "the source box is malformed", "provide (x1, y1, x2, y2)")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.xyxy):
            _fail("xyxy coordinates must be finite numbers", "the source box contains an invalid coordinate", "use finite numeric coordinates")
        x1, y1, x2, y2 = self.xyxy
        if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
            _fail(
                "xyxy must be in bounds and have non-zero area",
                "the box extends outside the image or has zero/negative area",
                "clamp and validate coordinates against width and height",
            )

    @classmethod
    def from_mapping(
        cls, row: Mapping[str, Any], registry: ClassRegistry = DEFAULT_CLASS_REGISTRY
    ) -> "CanonicalAnnotation":
        """Build a record from JSON-compatible values while preserving validation."""
        try:
            license_value = row["license_metadata"]
            if not isinstance(license_value, Mapping):
                raise TypeError("license_metadata must be an object")
            raw_box = row["xyxy"]
            if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
                raise TypeError("xyxy must be an array")
            return cls(
                source=row["source"],
                source_category=row["source_category"],
                source_image_id=row["source_image_id"],
                file_path=row["file_path"],
                width=row["width"],
                height=row["height"],
                class_id=row["class_id"],
                xyxy=tuple(raw_box),
                split=row["split"],
                label_status=row["label_status"],
                license_metadata=LicenseMetadata(
                    name=license_value["name"],
                    url=license_value.get("url"),
                    attribution=license_value.get("attribution"),
                ),
                registry=registry,
            )
        except (KeyError, TypeError) as error:
            _fail(
                "annotation mapping is missing a required canonical field",
                str(error),
                "supply all canonical annotation fields with JSON-compatible values",
            )


@dataclass(frozen=True)
class UnlabeledImageRecord:
    """An image manifest record for a training-pool image without object labels."""

    source: str
    source_image_id: str
    file_path: str
    width: int
    height: int
    split: str
    label_status: str
    license_metadata: LicenseMetadata

    def __post_init__(self) -> None:
        _nonempty_text(self.source, "source")
        _nonempty_text(self.source_image_id, "source_image_id")
        _nonempty_text(self.file_path, "file_path")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            _fail("width must be positive", "image width is absent or invalid", "provide a positive integer width")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            _fail("height must be positive", "image height is absent or invalid", "provide a positive integer height")
        split = _controlled_text(self.split, "split", ALLOWED_SPLITS)
        label_status = _controlled_text(
            self.label_status, "label_status", ALLOWED_LABEL_STATUSES
        )
        if (split, label_status) not in ALLOWED_UNLABELED_IMAGE_SPLIT_STATUSES:
            _fail(
                "split and label_status are not an allowed unlabeled-image combination",
                "unlabeled images must remain outside validation and test partitions",
                f"use one of {sorted(ALLOWED_UNLABELED_IMAGE_SPLIT_STATUSES)}",
            )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "UnlabeledImageRecord":
        """Build an unlabeled image manifest record from JSON-compatible values."""
        try:
            license_value = row["license_metadata"]
            if not isinstance(license_value, Mapping):
                raise TypeError("license_metadata must be an object")
            return cls(
                source=row["source"],
                source_image_id=row["source_image_id"],
                file_path=row["file_path"],
                width=row["width"],
                height=row["height"],
                split=row["split"],
                label_status=row["label_status"],
                license_metadata=LicenseMetadata(
                    name=license_value["name"],
                    url=license_value.get("url"),
                    attribution=license_value.get("attribution"),
                ),
            )
        except (KeyError, TypeError) as error:
            _fail(
                "unlabeled image mapping is missing a required field",
                str(error),
                "supply all unlabeled image fields with JSON-compatible values",
            )
