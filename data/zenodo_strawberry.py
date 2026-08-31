"""Strict local importer for Zenodo record 6126677 strawberry boxes.

The source archive stores YOLO labels beside its images under ``training`` and
``validation`` rather than under a conventional ``images``/``labels`` tree.
Its original partitions are deliberately discarded: accepted fruit records are
returned as ``train_pool`` so the project can create one leakage-safe split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem
from fruit_ssod.data.kaggle_fruit_detection import _dimensions, _source_classes, _source_image_id
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


SOURCE_NAME = "zenodo_strawberry_6126677"
_PARTITIONS = ("training", "validation")
_EXPECTED_CATEGORIES = frozenset({"ripe", "unripe", "peduncle"})


class ZenodoStrawberryImportError(ValueError):
    """Raised when the reviewed local Zenodo strawberry release is unsound."""


def _parse_or_clip_box(
    line: str,
    label: Path,
    line_number: int,
) -> tuple[int, float, float, float, float, bool] | None:
    """Parse a Zenodo YOLO box, clipping small border overflow when possible.

    The reviewed release contains a small number of zero-area boxes.  Those
    cannot describe a fruit and are rejected by the caller.  In contrast, a
    non-empty box that only extends beyond an image edge is clipped to the
    normalized image bounds and explicitly counted in the manifest.
    """
    parts = line.split()
    if len(parts) != 5:
        raise _fail("YOLO label row has an unsupported field count", f"{label}:{line_number}", "use class_id x_center y_center width height")
    try:
        class_id = int(parts[0])
        x, y, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
        raise _fail("YOLO label row is non-numeric", f"{label}:{line_number}", "use finite numeric normalized YOLO values") from error
    if class_id < 0 or any(not math.isfinite(value) for value in (x, y, width, height)):
        raise _fail("YOLO label row is invalid", f"{label}:{line_number}", "use a nonnegative class ID and finite values")
    if width <= 0 or height <= 0:
        return None
    raw = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
    clipped = (max(0.0, raw[0]), max(0.0, raw[1]), min(1.0, raw[2]), min(1.0, raw[3]))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    clipped_changed = clipped != raw
    clipped_width, clipped_height = clipped[2] - clipped[0], clipped[3] - clipped[1]
    return class_id, clipped[0] + clipped_width / 2, clipped[1] + clipped_height / 2, clipped_width, clipped_height, clipped_changed


@dataclass(frozen=True)
class ZenodoStrawberryImportResult:
    records: tuple[CanonicalAnnotation, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def _fail(problem: str, cause: str, remediation: str) -> ZenodoStrawberryImportError:
    return ZenodoStrawberryImportError(_problem(problem, cause, remediation))


def import_zenodo_strawberry(
    dataset_root: Path,
    data_yaml: Path,
    *,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
) -> ZenodoStrawberryImportResult:
    """Import reviewed fruit boxes while preserving rejected peduncle evidence."""
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("source metadata is invalid", str(error), "provide explicit source version, page and license metadata") from error
    if not dataset_root.is_dir() or not data_yaml.is_file():
        raise _fail("local Zenodo strawberry source is missing", f"dataset_root={dataset_root}, data_yaml={data_yaml}", "extract the reviewed archive and pass its strawberries directory and YAML")
    names = _source_classes(data_yaml)
    if set(names.values()) != _EXPECTED_CATEGORIES or len(names) != len(_EXPECTED_CATEGORIES):
        raise _fail("source class names do not match the reviewed Zenodo release", repr(names), "use record 6126677 with exactly ripe, unripe and peduncle classes")
    images = tuple(
        image
        for partition in _PARTITIONS
        for image in sorted((dataset_root / partition).glob("*"))
        if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise _fail("Zenodo strawberry source has no images", str(dataset_root), "provide the extracted strawberries/training and strawberries/validation directories")
    records: list[CanonicalAnnotation] = []
    rejections: list[dict[str, str]] = []
    clipped_box_count = 0
    for image in images:
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise _fail("strawberry image lacks its YOLO label", str(image), "restore the matching .txt annotation before importing")
        width, height = _dimensions(image)
        image_id = _source_image_id(image)
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise _fail("strawberry label cannot be read", str(label), "restore a readable UTF-8 YOLO label") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parsed = _parse_or_clip_box(line, label, line_number)
            if parsed is None:
                rejections.append({"source_image_id": image_id, "source_category": "unknown", "source_class_id": line.split()[0], "reason": "YOLO box is empty or lies wholly outside the normalized image bounds."})
                continue
            parsed_id, x, y, box_width, box_height, was_clipped = parsed
            category = names.get(parsed_id)
            if category is None:
                raise _fail("YOLO label references an absent class", f"{label}:{line_number} class={parsed_id}", "align source labels with strawberries.yaml")
            if category == "peduncle":
                rejections.append({"source_image_id": image_id, "source_category": category, "source_class_id": str(parsed_id), "reason": "Peduncle is not one of the five approved fruit detection classes."})
                continue
            try:
                canonical_id = resolve_class_id(SOURCE_NAME, category)
            except ClassMappingError as error:
                raise _fail("reviewed strawberry class is not mapped", str(error), "restore the explicit ripe/unripe Strawberry aliases") from error
            clipped_box_count += int(was_clipped)
            x1, y1 = (x - box_width / 2) * width, (y - box_height / 2) * height
            x2, y2 = (x + box_width / 2) * width, (y + box_height / 2) * height
            records.append(CanonicalAnnotation(source=SOURCE_NAME, source_category=category, source_image_id=image_id, file_path=image.resolve().relative_to(dataset_root.resolve()).as_posix(), width=width, height=height, class_id=canonical_id, xyxy=(x1, y1, x2, y2), split="train_pool", label_status="labeled", license_metadata=license_metadata))
    if not records:
        raise _fail("Zenodo strawberry source has no approved fruit boxes", str(dataset_root), "review annotations; only ripe and unripe boxes may map to Strawberry")
    rows = [{"record_type": "canonical_annotation", "source": row.source, "source_image_id": row.source_image_id, "source_category": row.source_category, "file_path": row.file_path, "width": row.width, "height": row.height, "class_id": row.class_id, "xyxy": list(row.xyxy), "split": row.split, "label_status": row.label_status, "license_metadata": _license_mapping(row.license_metadata)} for row in records]
    manifest = {"manifest_version": "1.0", "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)}, "source_data_yaml": str(data_yaml.resolve()), "source_layout": "images_and_yolo_labels_colocated_under_training_and_validation", "source_partitions_discarded": list(_PARTITIONS), "split": "train_pool", "label_status": "labeled", "geometry_conversion": {"source_box_count": len(rows), "polygon_to_enclosing_box_count": 0, "clipped_box_count": clipped_box_count}, "records": rows, "rejections": rejections, "record_count": len(rows), "rejection_count": len(rejections)}
    return ZenodoStrawberryImportResult(tuple(records), tuple(rejections), manifest)
