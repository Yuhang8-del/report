"""Local COCO-style FruitDet external-test importer.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "fruitdet"
MAPPING_SOURCE = "limited_external_set"
MAPPED_CLASS_IDS = (0, 1, 2, 3)
MAPPED_CLASS_NAMES = ("Apple", "Banana", "Orange", "Strawberry")


class FruitDetImportError(ValueError):
    """Raised when local COCO-style FruitDet data cannot remain safely external."""


@dataclass(frozen=True)
class FruitDetImportResult:
    """Validated external-test annotations and explicitly rejected source categories."""

    records: tuple[CanonicalAnnotation, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def _fail(problem: str, cause: str, remediation: str) -> FruitDetImportError:
    """Build FruitDet domain errors that identify a local correction."""
    return FruitDetImportError(_problem(problem, cause, remediation))


def _nonempty_text(value: object, field_name: str) -> str:
    """Reject missing COCO text fields before using them as paths or labels."""
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be nonempty text", "the COCO field is absent or malformed", f"provide a nonempty {field_name}")
    return value


def _coco_id(value: object, field_name: str) -> int:
    """COCO IDs must be non-bool integers to preserve source identity."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{field_name} must be a non-bool integer", "the COCO identifier has an invalid type", f"use a unique integer {field_name}")
    return value


def _positive_dimension(value: object, field_name: str) -> int:
    """Validate image dimensions before geometric conversion."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(f"{field_name} must be a positive integer", "the COCO image dimension is missing or invalid", f"provide a positive integer {field_name}")
    return value


def _contained_image_path(images_root: Path, file_name: str) -> str:
    """Require a relative path that resolves beneath the caller's image root."""
    raw_path = Path(file_name)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise _fail(
            f"COCO file_name {file_name!r} is not a safe relative path",
            "the annotation can escape the caller-provided images root",
            "use a relative file_name without '..' components",
        )
    resolved_root = images_root.resolve()
    resolved_image = (images_root / raw_path).resolve()
    try:
        resolved_image.relative_to(resolved_root)
    except ValueError as error:
        raise _fail(
            f"COCO file_name {file_name!r} escapes the supplied images root",
            "a path traversal or symlink points outside the local source directory",
            "keep every referenced image beneath --images-root",
        ) from error
    if not resolved_image.is_file():
        raise _fail(
            f"COCO image {file_name!r} is not an existing file beneath the supplied images root",
            "the local annotation references a missing path or a directory instead of an image file",
            "place the referenced image beneath --images-root or correct images[].file_name",
        )
    return raw_path.as_posix()


def _xyxy_from_coco(value: object, width: int, height: int, annotation_id: int) -> tuple[float, float, float, float]:
    """Strictly convert finite COCO XYWH geometry into canonical image-local XYXY."""
    if not isinstance(value, list) or len(value) != 4:
        raise _fail(f"Annotation {annotation_id} bbox must be a four-item XYWH array", "the COCO geometry is malformed", "provide [x, y, width, height]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise _fail(f"Annotation {annotation_id} bbox contains non-finite values", "the COCO geometry is not numeric", "provide finite numeric XYWH coordinates")
    x, y, box_width, box_height = (float(item) for item in value)
    x2, y2 = x + box_width, y + box_height
    if not (0 <= x < x2 <= width and 0 <= y < y2 <= height):
        raise _fail(
            f"Annotation {annotation_id} bbox is outside its image or has zero area",
            "COCO XYWH values are inverted, negative, or exceed image dimensions",
            "correct the annotation geometry; this importer does not clamp external-test boxes",
        )
    return (x, y, x2, y2)


def _load_payload(annotations_path: Path) -> Mapping[str, Any]:
    """Load a local JSON object without making any network request."""
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"FruitDet annotation JSON {annotations_path} could not be read", str(error), "provide a readable local UTF-8 COCO annotation JSON") from error
    if not isinstance(payload, Mapping):
        raise _fail("FruitDet annotation JSON must contain an object", "the JSON top level is not a COCO object", "provide images, categories, and annotations arrays")
    return payload


def _required_list(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """Validate a COCO collection before iterating over source mappings."""
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise _fail(f"COCO field {key!r} must be an array of objects", "the local annotation JSON has an incompatible structure", f"provide a valid {key} array")
    return value


def import_fruitdet(
    annotations_path: Path,
    images_root: Path,
    *,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
    split: str = "external_test",
) -> FruitDetImportResult:
    """Convert local FruitDet COCO annotations into labeled external-test records only."""
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("FruitDet source metadata is invalid", str(error), "provide source version, source page, and license metadata") from error
    if split != "external_test":
        raise _fail(
            "FruitDet can only be imported as external_test",
            "external reference labels must never merge into primary train/validation/test data",
            "use split='external_test' and keep its manifest separate from primary dataset manifests",
        )
    if not annotations_path.is_file():
        raise _fail(f"FruitDet annotations path {annotations_path} is not a file", "the caller supplied a missing local JSON file", "pass a readable local COCO annotation JSON")
    if not images_root.is_dir():
        raise _fail(f"FruitDet images root {images_root} is not a directory", "the caller supplied a missing local image root", "pass the local root that contains the COCO file_name paths")

    payload = _load_payload(annotations_path)
    raw_images = _required_list(payload, "images")
    raw_categories = _required_list(payload, "categories")
    raw_annotations = _required_list(payload, "annotations")

    images: dict[int, tuple[str, int, int]] = {}
    for image in raw_images:
        image_id = _coco_id(image.get("id"), "image.id")
        if image_id in images:
            raise _fail(f"Duplicate COCO image ID {image_id}", "two images share a source identifier", "make every images[].id unique")
        images[image_id] = (
            _contained_image_path(images_root, _nonempty_text(image.get("file_name"), "image.file_name")),
            _positive_dimension(image.get("width"), "image.width"),
            _positive_dimension(image.get("height"), "image.height"),
        )

    categories: dict[int, tuple[str, int] | None] = {}
    rejections: list[dict[str, str]] = []
    for category in raw_categories:
        category_id = _coco_id(category.get("id"), "category.id")
        if category_id in categories:
            raise _fail(f"Duplicate COCO category ID {category_id}", "two categories share a source identifier", "make every categories[].id unique")
        name = _nonempty_text(category.get("name"), "category.name")
        try:
            categories[category_id] = (name, resolve_class_id(MAPPING_SOURCE, name))
        except ClassMappingError as error:
            categories[category_id] = None
            rejections.append({"source_category_id": str(category_id), "source_category": name, "reason": str(error)})

    records: list[CanonicalAnnotation] = []
    record_rows: list[dict[str, Any]] = []
    annotation_ids: set[int] = set()
    for annotation in sorted(raw_annotations, key=lambda row: _coco_id(row.get("id"), "annotation.id")):
        annotation_id = _coco_id(annotation.get("id"), "annotation.id")
        if annotation_id in annotation_ids:
            raise _fail(f"Duplicate COCO annotation ID {annotation_id}", "two annotations share a source identifier", "make every annotations[].id unique")
        annotation_ids.add(annotation_id)
        image_id = _coco_id(annotation.get("image_id"), "annotation.image_id")
        category_id = _coco_id(annotation.get("category_id"), "annotation.category_id")
        if image_id not in images:
            raise _fail(f"Annotation {annotation_id} references unknown image ID {image_id}", "images[] is incomplete", "add the referenced image to images[]")
        if category_id not in categories:
            raise _fail(f"Annotation {annotation_id} references unknown category ID {category_id}", "categories[] is incomplete", "add the referenced category to categories[]")
        category = categories[category_id]
        if category is None:
            continue
        category_name, class_id = category
        file_path, width, height = images[image_id]
        xyxy = _xyxy_from_coco(annotation.get("bbox"), width, height, annotation_id)
        record = CanonicalAnnotation(
            source=MAPPING_SOURCE,
            source_category=category_name,
            source_image_id=str(image_id),
            file_path=file_path,
            width=width,
            height=height,
            class_id=class_id,
            xyxy=xyxy,
            split="external_test",
            label_status="labeled",
            license_metadata=license_metadata,
        )
        records.append(record)
        record_rows.append(
            {
                "record_type": "canonical_annotation",
                "source": record.source,
                "source_dataset": SOURCE_NAME,
                "source_image_id": record.source_image_id,
                "source_category_id": category_id,
                "source_category": record.source_category,
                "source_annotation_id": annotation_id,
                "file_path": record.file_path,
                "width": record.width,
                "height": record.height,
                "class_id": record.class_id,
                "xyxy": list(record.xyxy),
                "split": record.split,
                "label_status": record.label_status,
                "license_metadata": _license_mapping(record.license_metadata),
            }
        )

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)},
        "category_mapping_source": MAPPING_SOURCE,
        # Kept explicitly in the importer product so external evaluation can
        # prove its four-class protocol without inferring it from a detector
        # YAML or an arbitrary caller-supplied data directory.
        "mapped_class_ids": list(MAPPED_CLASS_IDS),
        "mapped_class_names": list(MAPPED_CLASS_NAMES),
        "split": "external_test",
        "label_status": "labeled",
        "records": record_rows,
        "rejections": rejections,
        "record_count": len(record_rows),
        "rejection_count": len(rejections),
    }
    return FruitDetImportResult(tuple(records), tuple(rejections), manifest)
