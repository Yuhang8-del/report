"""Build a bounded, provenance-preserving Open Images fruit selection.

This module reads the official CSV exports incrementally.  It never downloads
pixels itself: the resulting two CSV files are explicit inputs to the existing
resumable ``download_open_images`` command.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY, resolve_class_id
from fruit_ssod.data.open_images import OpenImagesConversionError, _read_class_descriptions


SOURCE_NAME = "open_images_v7"
OFFICIAL_BUCKET_BASE = "https://open-images-dataset.s3.amazonaws.com"
_IMAGE_SPLITS = frozenset({"train", "validation", "test"})


class OpenImagesSelectionError(ValueError):
    """Raised when official CSV inputs cannot produce a sound selection."""


@dataclass(frozen=True)
class OpenImagesSelection:
    """One deterministic subset plus the official image provenance needed later."""

    image_ids: tuple[str, ...]
    class_image_counts: Mapping[str, int]
    annotation_count: int


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _required_columns(reader: csv.DictReader, required: set[str], path: Path) -> None:
    fields = set(reader.fieldnames or ())
    missing = sorted(required - fields)
    if missing:
        raise OpenImagesSelectionError(
            _problem(f"CSV {path} is missing required columns {missing}", "an incompatible or truncated official export was supplied", "download the matching Open Images metadata file again")
        )


def _read_rows(path: Path, required: set[str]) -> Iterable[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _required_columns(reader, required, path)
            for row in reader:
                yield {key: (value or "") for key, value in row.items()}
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise OpenImagesSelectionError(
            _problem(f"CSV {path} cannot be read", str(error), "restore a readable UTF-8 official CSV export")
        ) from error


def _flagged(row: Mapping[str, str]) -> bool:
    return any(row.get(field, "").strip() in {"1", "true", "True"} for field in ("IsDepiction", "IsInside", "IsGroupOf"))


def _target_label_ids(class_descriptions: Path) -> dict[str, tuple[int, str]]:
    descriptions = _read_class_descriptions(class_descriptions)
    targets: dict[str, tuple[int, str]] = {}
    for label_id, display_name in descriptions.items():
        try:
            class_id = resolve_class_id(SOURCE_NAME, display_name)
        except Exception:
            continue
        targets[label_id] = (class_id, display_name)
    expected = set(range(len(DEFAULT_CLASS_REGISTRY.class_names)))
    found = {class_id for class_id, _ in targets.values()}
    if found != expected:
        missing = [DEFAULT_CLASS_REGISTRY.class_names[index] for index in sorted(expected - found)]
        raise OpenImagesSelectionError(
            _problem("official class descriptions do not contain every requested fruit", f"missing canonical names: {missing}", "check the Open Images V7 boxable class-description file and source alias registry")
        )
    return targets


def _class_targets(per_class: int | Mapping[str, int]) -> dict[int, int]:
    """Normalize a uniform or explicit class-cap request."""
    names = DEFAULT_CLASS_REGISTRY.class_names
    if isinstance(per_class, int) and not isinstance(per_class, bool):
        if per_class <= 0:
            raise OpenImagesSelectionError(_problem("per_class must be positive", "a zero or negative cap was supplied", "use a positive integer such as 25 for a smoke run"))
        return {class_id: per_class for class_id in range(len(names))}
    if not isinstance(per_class, Mapping) or set(per_class) != set(names):
        raise OpenImagesSelectionError(_problem("class caps are invalid", f"expected exactly {list(names)!r}", "provide one positive integer cap for each canonical fruit class"))
    normalized: dict[int, int] = {}
    for class_id, name in enumerate(names):
        value = per_class[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OpenImagesSelectionError(_problem("class caps are invalid", f"{name}={value!r}", "provide a positive integer cap for every canonical class"))
        normalized[class_id] = value
    return normalized


def _select_ids(rows_by_image: Mapping[str, list[dict[str, str]]], targets: Mapping[str, tuple[int, str]], requested: Mapping[int, int]) -> tuple[str, ...]:
    """Greedily select image groups, prioritizing whichever fruit is rarest."""
    remaining = dict(requested)
    selected: list[str] = []
    available = set(rows_by_image)
    while available and any(value > 0 for value in remaining.values()):
        def rank(image_id: str) -> tuple[int, int, str]:
            present = {targets[row["LabelName"]][0] for row in rows_by_image[image_id]}
            unmet = sum(remaining[class_id] for class_id in present)
            coverage = sum(1 for class_id in present if remaining[class_id] > 0)
            return (unmet, coverage, image_id)
        image_id = max(available, key=rank)
        present = {targets[row["LabelName"]][0] for row in rows_by_image[image_id]}
        if not any(remaining[class_id] > 0 for class_id in present):
            break
        available.remove(image_id)
        selected.append(image_id)
        for class_id in present:
            remaining[class_id] = max(0, remaining[class_id] - 1)
    if any(value > 0 for value in remaining.values()):
        missing = {DEFAULT_CLASS_REGISTRY.class_names[class_id]: count for class_id, count in remaining.items() if count > 0}
        raise OpenImagesSelectionError(
            _problem("Open Images annotations cannot satisfy the requested per-class cap", f"insufficient valid source images for {missing}", "reduce --per-class or obtain a complete matching train annotation export")
        )
    return tuple(sorted(selected))


def build_open_images_selection(
    annotations: Path,
    class_descriptions: Path,
    image_metadata: Path,
    output_dir: Path,
    *,
    per_class: int | Mapping[str, int],
    image_split: str = "train",
    excluded_image_ids: Iterable[str] = (),
) -> OpenImagesSelection:
    """Write selected annotation/URL metadata without changing source CSVs."""
    requested = _class_targets(per_class)
    if output_dir.exists():
        raise OpenImagesSelectionError(_problem(f"selection output {output_dir} already exists", "selection evidence is immutable once written", "choose a fresh output directory instead of overwriting it"))
    if image_split not in _IMAGE_SPLITS:
        raise OpenImagesSelectionError(
            _problem("Open Images image split is unsupported", repr(image_split), "use one of train, validation, or test")
        )
    targets = _target_label_ids(class_descriptions)
    excluded_ids = frozenset(image_id.strip() for image_id in excluded_image_ids if image_id.strip())
    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(annotations, {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsDepiction", "IsInside", "IsGroupOf"}):
        image_id, label_id = row["ImageID"].strip(), row["LabelName"].strip()
        if image_id and image_id not in excluded_ids and label_id in targets and not _flagged(row):
            rows_by_image[image_id].append(row)
    selected_ids = _select_ids(rows_by_image, targets, requested)
    selected_set = set(selected_ids)
    metadata_by_id: dict[str, dict[str, str]] = {}
    for row in _read_rows(image_metadata, {"ImageID", "OriginalURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL", "Thumbnail300KURL"}):
        image_id = row["ImageID"].strip()
        if image_id in selected_set:
            # Original Flickr links and even generated thumbnails can
            # disappear after the dataset release.  The official V7 manual
            # downloader retrieves this exact object from the public
            # ``open-images-dataset`` bucket instead.  Use its HTTPS form so
            # the project's resumable downloader remains dependency-free.
            row["SourceOriginalURL"] = row["OriginalURL"]
            row["SourceThumbnailURL"] = row["Thumbnail300KURL"]
            row["OriginalURL"] = f"{OFFICIAL_BUCKET_BASE}/{image_split}/{image_id}.jpg"
            metadata_by_id[image_id] = row
    missing_urls = sorted(selected_set - set(metadata_by_id))
    if missing_urls:
        raise OpenImagesSelectionError(_problem("selected image IDs have no official URL metadata", f"missing IDs include {missing_urls[:5]!r}", "use the image URL CSV matching the annotation split"))
    output_dir.mkdir(parents=True)
    annotation_path = output_dir / "annotations.csv"
    url_path = output_dir / "image-urls.csv"
    source_path = output_dir / "source-metadata.json"
    fields = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
    written_annotations = 0
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for image_id in selected_ids:
            for row in rows_by_image[image_id]:
                writer.writerow(row)
                written_annotations += 1
    with url_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "OriginalURL", "SourceOriginalURL", "SourceThumbnailURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL"], extrasaction="ignore")
        writer.writeheader()
        for image_id in selected_ids:
            writer.writerow(metadata_by_id[image_id])
    class_counts = {name: 0 for name in DEFAULT_CLASS_REGISTRY.class_names}
    for image_id in selected_ids:
        for class_id in {targets[row["LabelName"]][0] for row in rows_by_image[image_id]}:
            class_counts[DEFAULT_CLASS_REGISTRY.class_names[class_id]] += 1
    requested_metadata: int | dict[str, int] = per_class if isinstance(per_class, int) else {name: requested[class_id] for class_id, name in enumerate(DEFAULT_CLASS_REGISTRY.class_names)}
    source_path.write_text(json.dumps({"schema_version": "1.0", "source": SOURCE_NAME, "source_image_split": image_split, "per_class_requested": requested_metadata, "excluded_source_image_count": len(excluded_ids), "selected_image_ids": list(selected_ids), "class_image_counts": class_counts, "annotation_count": written_annotations, "source_files": {"annotations": str(annotations.resolve()), "class_descriptions": str(class_descriptions.resolve()), "image_metadata": str(image_metadata.resolve())}}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return OpenImagesSelection(selected_ids, class_counts, written_annotations)
