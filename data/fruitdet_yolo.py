"""Materialize FruitDet YOLO test files as a sealed external-only dataset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY, ClassMappingError, resolve_class_id
from fruit_ssod.data.fruitdet import MAPPED_CLASS_IDS, MAPPED_CLASS_NAMES, MAPPING_SOURCE, SOURCE_NAME
from fruit_ssod.data.fruits360 import SourceMetadata, _license_mapping, _problem, read_image_dimensions
from fruit_ssod.data.schema import LicenseMetadata


class FruitDetYoloError(ValueError):
    """Raised when a local FruitDet YOLO source cannot form external evidence."""


@dataclass(frozen=True)
class FruitDetYoloResult:
    root: Path
    dataset_yaml: Path
    manifest: Path
    image_count: int
    annotation_count: int


def _fail(problem: str, cause: str, remediation: str) -> FruitDetYoloError:
    return FruitDetYoloError(_problem(problem, cause, remediation))


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _fail("FruitDet source file cannot be hashed", str(error), "restore the readable local source file") from error


def _parse_yolo_labels(path: Path, *, width: int, height: int, category: str) -> list[tuple[float, float, float, float]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise _fail("FruitDet YOLO label cannot be read", str(error), "restore a UTF-8 label file beside its image") from error
    if not lines:
        raise _fail("FruitDet YOLO label is empty", str(path), "provide at least one bounding box for each external-test image")
    boxes: list[tuple[float, float, float, float]] = []
    for index, line in enumerate(lines, start=1):
        tokens = line.split()
        if len(tokens) != 5:
            raise _fail("FruitDet YOLO label has an invalid field count", f"{path}:{index} has {len(tokens)} fields", "use class_id center_x center_y width height")
        try:
            source_class = int(tokens[0])
            center_x, center_y, box_w, box_h = (float(value) for value in tokens[1:])
        except ValueError as error:
            raise _fail("FruitDet YOLO label is nonnumeric", f"{path}:{index}", "use finite numeric YOLO coordinates") from error
        if source_class != 0 or any(not math.isfinite(value) for value in (center_x, center_y, box_w, box_h)) or box_w <= 0 or box_h <= 0:
            raise _fail("FruitDet YOLO label is invalid", f"{path}:{index}", "use local class 0 and finite positive width/height")
        x1, y1 = (center_x - box_w / 2) * width, (center_y - box_h / 2) * height
        x2, y2 = (center_x + box_w / 2) * width, (center_y + box_h / 2) * height
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise _fail("FruitDet YOLO box is out of bounds", f"{path}:{index}", "correct normalized center/size values; external boxes are never clamped")
        boxes.append((x1, y1, x2, y2))
    return boxes


def materialize_fruitdet_yolo(dataset_root: Path, output_root: Path, *, source_version: str, source_page: str, license_metadata: LicenseMetadata) -> FruitDetYoloResult:
    """Copy the reviewed FruitDet test partition into canonical external-only YOLO evidence."""
    try:
        metadata = SourceMetadata(source_version, source_page, license_metadata)
    except ValueError as error:
        raise _fail("FruitDet source metadata is invalid", str(error), "provide source version, source page, and license metadata") from error
    source = dataset_root.resolve(strict=False)
    if not source.is_dir():
        raise _fail("FruitDet YOLO dataset root is unavailable", str(source), "pass the downloaded FruitDet repository root")
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _fail("FruitDet external output already exists", str(root), "preserve immutable external evidence and choose a fresh output directory")
    expected = {name.casefold(): name for name in MAPPED_CLASS_NAMES}
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    source_evidence: list[dict[str, str]] = []
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        image_dir, label_dir = temporary / "images" / "test", temporary / "labels" / "test"
        image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
        output_images: list[Path] = []
        for local_category, category in sorted(expected.items()):
            try:
                class_id = resolve_class_id(MAPPING_SOURCE, category)
            except ClassMappingError as error:  # pragma: no cover - committed registry invariant.
                raise _fail("FruitDet category mapping is invalid", str(error), "restore the limited_external_set registry") from error
            source_images = source / "data" / local_category / "images" / "test"
            source_labels = source / "data" / local_category / "labels" / "test"
            if not source_images.is_dir() or not source_labels.is_dir():
                rejections.append({"source_category": category, "reason": "source test directory is absent in this checked FruitDet revision"})
                continue
            images = sorted((path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda path: path.name)
            if not images:
                rejections.append({"source_category": category, "reason": "source test directory contains no images in this checked FruitDet revision"})
                continue
            for image in images:
                label = source_labels / f"{image.stem}.txt"
                if not label.is_file():
                    raise _fail("FruitDet test image has no matching label", str(image), "restore the matching YOLO .txt file")
                width, height = read_image_dimensions(image)
                boxes = _parse_yolo_labels(label, width=width, height=height, category=category)
                image_id = f"fruitdet-{local_category}-{image.stem}"
                destination_image = image_dir / f"{image_id}{image.suffix.lower()}"
                destination_label = label_dir / f"{image_id}.txt"
                shutil.copy2(image, destination_image)
                if _digest(image) != _digest(destination_image):
                    raise _fail("FruitDet copied image digest differs", str(image), "retry from stable local storage")
                canonical_lines: list[str] = []
                for annotation_index, (x1, y1, x2, y2) in enumerate(boxes):
                    canonical_lines.append(f"{class_id} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}")
                    records.append({"record_type": "canonical_annotation", "source": MAPPING_SOURCE, "source_dataset": SOURCE_NAME, "source_image_id": image_id, "source_category": category, "source_annotation_id": f"{local_category}/{image.name}:{annotation_index}", "source_file_path": str(image.resolve()), "file_path": str((root / destination_image.relative_to(temporary)).resolve()), "width": width, "height": height, "class_id": class_id, "xyxy": [x1, y1, x2, y2], "split": "external_test", "label_status": "labeled", "license_metadata": _license_mapping(license_metadata)})
                destination_label.write_text("\n".join(canonical_lines) + "\n", encoding="utf-8", newline="\n")
                output_images.append(destination_image)
                source_evidence.append({"source_image": str(image.resolve()), "source_label": str(label.resolve()), "source_image_sha256": _digest(image), "source_label_sha256": _digest(label), "snapshot_image": destination_image.relative_to(temporary).as_posix(), "snapshot_label": destination_label.relative_to(temporary).as_posix()})
        if not output_images or not records:
            raise _fail("FruitDet checked revision has no supported external test annotations", "none of Apple, Banana, Orange, or Strawberry supplied usable test image/label pairs", "use a revision containing a reviewed mapped category")
        # The snapshot is atomically renamed after staging; list entries must
        # point at their published locations rather than the temporary root.
        test_list = [str((root / path.relative_to(temporary)).resolve()) for path in output_images]
        (temporary / "test.txt").write_text("\n".join(test_list) + "\n", encoding="utf-8", newline="\n")
        dataset = {"path": str(temporary), "train": "test.txt", "val": "test.txt", "test": "test.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        manifest = {"manifest_version": "1.0", "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)}, "category_mapping_source": MAPPING_SOURCE, "mapped_class_ids": list(MAPPED_CLASS_IDS), "mapped_class_names": list(MAPPED_CLASS_NAMES), "split": "external_test", "label_status": "labeled", "records": records, "rejections": rejections, "record_count": len(records), "rejection_count": len(rejections), "source_evidence": source_evidence}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, root); temporary = None
        published_dataset = {"path": str(root), "train": "test.txt", "val": "test.txt", "test": "test.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (root / "dataset.yaml").write_text(yaml.safe_dump(published_dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        return FruitDetYoloResult(root, root / "dataset.yaml", root / "manifest.json", len(output_images), len(records))
    except OSError as error:
        raise _fail("FruitDet external dataset could not be materialized", str(error), "verify writable output storage and intact local source files") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
