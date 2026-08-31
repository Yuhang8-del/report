"""Turn a downloaded Open Images conversion into canonical annotation rows."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY


class OpenImagesManifestError(ValueError):
    """Raised when downloaded Open Images artifacts cannot prove their provenance."""


@dataclass(frozen=True)
class CanonicalManifestResult:
    """Counts emitted by the immutable canonical-manifest writer."""

    image_count: int
    annotation_count: int


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _read_url_metadata(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ImageID", "License", "Author", "OriginalURL"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise OpenImagesManifestError(_problem("selected image URL CSV is incomplete", f"missing columns {sorted(missing)!r}", "use select_open_images output without modification"))
            result: dict[str, dict[str, str]] = {}
            for row in reader:
                image_id = (row.get("ImageID") or "").strip()
                if not image_id or image_id in result:
                    raise OpenImagesManifestError(_problem("selected image URL CSV has an empty or duplicate ImageID", "selection provenance is malformed", "regenerate a fresh selection output"))
                result[image_id] = {key: (value or "").strip() for key, value in row.items()}
            return result
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise OpenImagesManifestError(_problem(f"selected image URL CSV {path} cannot be read", str(error), "restore the immutable selection output")) from error


def _safe_child(root: Path, relative: Path, *, description: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise OpenImagesManifestError(_problem(f"{description} has an unsafe relative path", str(relative), "restore a generated Open Images conversion directory"))
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise OpenImagesManifestError(_problem(f"{description} escapes the converted root", str(candidate), "restore a generated Open Images conversion directory")) from error
    return candidate


def _image_path(image_root: Path, image_id: str) -> Path:
    matches = [image_root / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp") if (image_root / f"{image_id}{suffix}").is_file()]
    if len(matches) != 1:
        raise OpenImagesManifestError(_problem(f"downloaded image {image_id!r} is missing or ambiguous", f"found {len(matches)} matching image files", "complete the resumable download and keep one generated image per source ID"))
    return matches[0]


def _label_rows(path: Path, image_id: str, *, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} cannot be read", str(error), "restore the generated YOLO label file")) from error
    if not lines:
        raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} is empty", "the downloaded conversion has no retained boxes", "regenerate this conversion from the selected annotations"))
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} has malformed line {line_number}", repr(line), "regenerate labels with download_open_images"))
        try:
            class_id = int(parts[0]); cx, cy, box_w, box_h = (float(value) for value in parts[1:])
        except ValueError as error:
            raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} has nonnumeric line {line_number}", repr(line), "regenerate labels with download_open_images")) from error
        if class_id not in range(len(DEFAULT_CLASS_REGISTRY.class_names)) or any(not math.isfinite(value) for value in (cx, cy, box_w, box_h)):
            raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} has an invalid class or geometry", repr(line), "regenerate labels with the fixed five-class converter"))
        x1, y1 = (cx - box_w / 2) * width, (cy - box_h / 2) * height
        x2, y2 = (cx + box_w / 2) * width, (cy + box_h / 2) * height
        # The converter persists normalized YOLO values at six decimals.
        # Reconstructing a box that originally touched an image edge can
        # therefore miss that edge by a few thousandths of a pixel.  Permit
        # only that quantified serialization error and retain hard rejection
        # for genuinely invalid geometry.
        tolerance = max(width, height) * 1e-5
        if -tolerance <= x1 <= 0:
            x1 = 0.0
        if -tolerance <= y1 <= 0:
            y1 = 0.0
        if width <= x2 <= width + tolerance:
            x2 = float(width)
        if height <= y2 <= height + tolerance:
            y2 = float(height)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise OpenImagesManifestError(_problem(f"label file for image {image_id!r} has out-of-bounds line {line_number}", repr(line), "regenerate labels from valid normalized source boxes"))
        result.append((class_id, (x1, y1, x2, y2)))
    return result


def build_canonical_open_images_manifest(converted_root: Path, selection_url_csv: Path, output: Path) -> CanonicalManifestResult:
    """Write canonical object rows that retain only approved downloaded evidence."""
    if output.exists():
        raise OpenImagesManifestError(_problem(f"canonical manifest output {output} already exists", "dataset evidence must not be overwritten", "choose a fresh output path"))
    root = converted_root.resolve(strict=True)
    manifest_path = _safe_child(root, Path("manifest.jsonl"), description="converted manifest")
    image_root = _safe_child(root, Path("images"), description="converted images")
    label_root = _safe_child(root, Path("labels"), description="converted labels")
    if not manifest_path.is_file() or not image_root.is_dir() or not label_root.is_dir():
        raise OpenImagesManifestError(_problem("converted Open Images root is incomplete", str(root), "use a completed download_open_images output directory"))
    metadata = _read_url_metadata(selection_url_csv.resolve(strict=True))
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    try:
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise OpenImagesManifestError(_problem("converted manifest cannot be read", str(error), "restore the generated manifest.jsonl")) from error
    for line_number, line in enumerate(manifest_lines, start=1):
        try:
            row = json.loads(line)
            image_id = row["source_image_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise OpenImagesManifestError(_problem(f"converted manifest line {line_number} is malformed", str(error), "regenerate the conversion output")) from error
        if not isinstance(image_id, str) or not image_id or image_id in seen_ids or image_id not in metadata:
            raise OpenImagesManifestError(_problem(f"converted manifest has invalid provenance for image {image_id!r}", "an image ID is duplicate, missing, or absent from the frozen selection", "use matching unmodified selection and conversion outputs"))
        seen_ids.add(image_id)
        image = _image_path(image_root, image_id)
        try:
            with Image.open(image) as decoded:
                width, height = decoded.size
        except (OSError, UnidentifiedImageError) as error:
            raise OpenImagesManifestError(_problem(f"downloaded image {image_id!r} cannot be decoded", str(error), "rerun the resumable download for this source image")) from error
        labels = _label_rows(label_root / f"{image_id}.txt", image_id, width=width, height=height)
        source_meta = metadata[image_id]
        for class_id, xyxy in labels:
            records.append({"source": "open_images_v7", "source_category": DEFAULT_CLASS_REGISTRY.class_names[class_id], "source_image_id": image_id, "file_path": image.relative_to(root).as_posix(), "width": width, "height": height, "class_id": class_id, "xyxy": list(xyxy), "split": "train_pool", "label_status": "labeled", "license_metadata": {"name": "Open Images recorded image license", "url": source_meta["License"] or None, "attribution": source_meta["Author"] or None}, "source_url": source_meta["OriginalURL"], "source_original_url": source_meta.get("SourceOriginalURL") or None, "source_landing_url": source_meta.get("OriginalLandingURL") or None})
    if not records:
        raise OpenImagesManifestError(_problem("converted manifest contains no retained annotations", "no valid downloaded image/label pair was available", "review the selection and conversion reports"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "1.0", "source": "open_images_v7", "records": records}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return CanonicalManifestResult(len(seen_ids), len(records))
