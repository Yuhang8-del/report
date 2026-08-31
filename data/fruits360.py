"""Local-only Fruits-360 image manifest importer.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord


SOURCE_NAME = "fruit_360"
_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"})


class Fruits360ImportError(ValueError):
    """Raised when a local Fruits-360 directory cannot be safely indexed."""


@dataclass(frozen=True)
class SourceMetadata:
    """Caller-supplied provenance required for every auxiliary manifest."""

    version: str
    page: str
    license_metadata: LicenseMetadata

    def __post_init__(self) -> None:
        for field_name, value in (("source version", self.version), ("source page", self.page)):
            if not isinstance(value, str) or not value.strip():
                raise Fruits360ImportError(
                    _problem(
                        f"{field_name} must be nonempty text",
                        "required source provenance was omitted",
                        "provide the current source version and source-page URL explicitly",
                    )
                )


@dataclass(frozen=True)
class Fruits360ImportResult:
    """Deterministic local manifest rows plus recoverable category/image rejections."""

    records: tuple[UnlabeledImageRecord, ...]
    rejections: tuple[Mapping[str, str], ...]
    manifest: Mapping[str, Any]


def _problem(problem: str, cause: str, remediation: str) -> str:
    """Format local-data failures with an operator-facing remediation."""
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _normalized_sort_key(value: str) -> tuple[str, str]:
    """Sort paths case-insensitively while retaining an exact-text tie-breaker."""
    return (value.casefold(), value)


def _path_within(root: Path, candidate: Path) -> Path:
    """Resolve a local path and prevent a symlink or traversal from leaving ``root``."""
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise Fruits360ImportError(
            _problem(
                f"Image path {candidate} escapes the supplied images root",
                "a path traversal or symlink points outside the local source directory",
                "keep every imported image beneath --images-root",
            )
        ) from error
    return resolved


def _pixel_dimension(value: str | None, field_name: str) -> int:
    """Read a positive integral SVG dimension without accepting relative units."""
    if value is None:
        raise Fruits360ImportError(
            _problem(
                f"SVG {field_name} is missing",
                "the local SVG does not declare pixel dimensions",
                "add absolute width/height attributes or use a raster image format",
            )
        )
    numeric = value.strip().removesuffix("px")
    try:
        number = float(numeric)
    except ValueError as error:
        raise Fruits360ImportError(
            _problem(
                f"SVG {field_name} is not a pixel value",
                f"the value {value!r} is not a positive numeric dimension",
                "use a positive integer or px dimension",
            )
        ) from error
    if not number.is_integer() or number <= 0:
        raise Fruits360ImportError(
            _problem(
                f"SVG {field_name} must be a positive integer",
                "the image cannot be represented by the canonical pixel schema",
                "use a positive integral width and height",
            )
        )
    return int(number)


def _svg_dimensions(path: Path) -> tuple[int, int]:
    """Extract SVG dimensions without importing an image-processing dependency."""
    try:
        root = element_tree.parse(path).getroot()
    except (OSError, element_tree.ParseError) as error:
        raise Fruits360ImportError(
            _problem(
                f"Image {path} is not a readable SVG",
                str(error),
                "repair the local image or remove it from the import directory",
            )
        ) from error
    return (_pixel_dimension(root.get("width"), "width"), _pixel_dimension(root.get("height"), "height"))


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG SOF dimensions using only the standard library."""
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                raise ValueError("missing JPEG start marker")
            while True:
                marker_start = handle.read(1)
                while marker_start == b"\xff":
                    marker_start = handle.read(1)
                if not marker_start:
                    break
                marker = marker_start[0]
                if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                    continue
                raw_length = handle.read(2)
                if len(raw_length) != 2:
                    break
                length = struct.unpack(">H", raw_length)[0]
                if length < 2:
                    break
                if 0xC0 <= marker <= 0xC3 or 0xC5 <= marker <= 0xC7 or 0xC9 <= marker <= 0xCB or 0xCD <= marker <= 0xCF:
                    payload = handle.read(length - 2)
                    if len(payload) < 5:
                        break
                    height, width = struct.unpack(">HH", payload[1:5])
                    return width, height
                handle.seek(length - 2, 1)
    except OSError as error:
        raise Fruits360ImportError(
            _problem(f"Image {path} could not be read", str(error), "repair the local image and retry")
        ) from error
    raise Fruits360ImportError(
        _problem(
            f"Image {path} has no readable JPEG dimensions",
            "the file is truncated or not a baseline/progressive JPEG",
            "replace it with a valid local image",
        )
    )


def read_image_dimensions(path: Path) -> tuple[int, int]:
    """Return positive pixel dimensions for common local image formats."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            return _svg_dimensions(path)
        data = path.read_bytes()
        if suffix == ".png" and data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if suffix == ".gif" and len(data) >= 10 and data[:3] == b"GIF":
            return struct.unpack("<HH", data[6:10])
        if suffix == ".bmp" and len(data) >= 26 and data[:2] == b"BM":
            return abs(struct.unpack("<i", data[18:22])[0]), abs(struct.unpack("<i", data[22:26])[0])
        if suffix in {".jpg", ".jpeg"}:
            return _jpeg_dimensions(path)
        if suffix == ".webp" and len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            if data[12:16] == b"VP8X":
                return (1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little"))
    except (OSError, struct.error) as error:
        raise Fruits360ImportError(
            _problem(f"Image {path} could not be read", str(error), "repair the local image and retry")
        ) from error
    raise Fruits360ImportError(
        _problem(
            f"Image {path} uses an unsupported or malformed image encoding",
            "its header does not contain readable pixel dimensions",
            "use a valid PNG, JPEG, GIF, BMP, WEBP, or SVG image",
        )
    )


def _license_mapping(metadata: LicenseMetadata) -> dict[str, str | None]:
    """Keep manifest provenance JSON-compatible without mutating caller metadata."""
    return {"name": metadata.name, "url": metadata.url, "attribution": metadata.attribution}


def _record_mapping(record: UnlabeledImageRecord, source_category: str) -> dict[str, Any]:
    """Render only unlabeled image facts; this importer never manufactures boxes."""
    return {
        "record_type": "unlabeled_image",
        "source": record.source,
        "source_category": source_category,
        "source_image_id": record.source_image_id,
        "file_path": record.file_path,
        "width": record.width,
        "height": record.height,
        "split": record.split,
        "label_status": record.label_status,
        "license_metadata": _license_mapping(record.license_metadata),
    }


def import_fruits360(
    images_root: Path,
    *,
    source_version: str,
    source_page: str,
    license_metadata: LicenseMetadata,
    split: str = "train_pool",
) -> Fruits360ImportResult:
    """Index local category directories as unlabeled training-pool images only."""
    metadata = SourceMetadata(source_version, source_page, license_metadata)
    if split != "train_pool":
        raise Fruits360ImportError(
            _problem(
                "Fruits-360 image-only import must target train_pool",
                "unlabeled images cannot safely enter evaluation partitions",
                "use split='train_pool' and curate labels separately",
            )
        )
    if not images_root.is_dir():
        raise Fruits360ImportError(
            _problem(
                f"Images root {images_root} is not a readable directory",
                "the caller supplied a missing local source path",
                "pass an existing local directory containing category subdirectories",
            )
        )

    root = images_root.resolve()
    records: list[UnlabeledImageRecord] = []
    record_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    categories = sorted(
        (item for item in images_root.iterdir() if item.is_dir()),
        key=lambda item: _normalized_sort_key(item.name),
    )
    for category_dir in categories:
        category = category_dir.name
        try:
            _path_within(root, category_dir)
            resolve_class_id(SOURCE_NAME, category)
        except (ClassMappingError, Fruits360ImportError) as error:
            rejections.append({"source_category": category, "reason": str(error)})
            continue
        image_paths = sorted(
            (item for item in category_dir.rglob("*") if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS),
            key=lambda item: _normalized_sort_key(item.relative_to(images_root).as_posix()),
        )
        for image_path in image_paths:
            try:
                _path_within(root, image_path)
                width, height = read_image_dimensions(image_path)
                relative_path = image_path.relative_to(images_root).as_posix()
                record = UnlabeledImageRecord(
                    source=SOURCE_NAME,
                    source_image_id=relative_path,
                    file_path=relative_path,
                    width=width,
                    height=height,
                    split="train_pool",
                    label_status="unlabeled",
                    license_metadata=license_metadata,
                )
            except (Fruits360ImportError, ValueError) as error:
                rejections.append({"source_category": category, "source_image_id": image_path.name, "reason": str(error)})
                continue
            records.append(record)
            record_rows.append(_record_mapping(record, category))

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "source": {"name": SOURCE_NAME, "version": metadata.version, "page": metadata.page, "license": _license_mapping(metadata.license_metadata)},
        "split": "train_pool",
        "label_status": "unlabeled",
        "records": record_rows,
        "rejections": rejections,
        "record_count": len(record_rows),
        "rejection_count": len(rejections),
    }
    return Fruits360ImportResult(tuple(records), tuple(rejections), manifest)
