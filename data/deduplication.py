"""Deterministic exact/perceptual duplicate detection without source mutation."""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import imagehash
from PIL import Image, UnidentifiedImageError

from fruit_ssod.data.cleaning import QuarantineRecord


SPLIT_PRIORITY = ("test", "validation", "pseudo_audit", "train_pool")
_SPLIT_RANK = {split: index for index, split in enumerate(SPLIT_PRIORITY)}


class DeduplicationError(ValueError):
    """Raised for invalid deduplication input or policy values."""


@dataclass(frozen=True)
class ImageFingerprint:
    source_image_id: str
    file_path: str
    split: str
    sha256: str
    perceptual_hash: str


@dataclass(frozen=True)
class RecordImageMapping:
    """Links an object-level input row to its single image-level fingerprint."""

    record_index: int
    image_key: str
    source_image_id: str
    file_path: str


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    kind: str
    members: tuple[ImageFingerprint, ...]


@dataclass(frozen=True)
class DuplicateResolution:
    group_id: str
    retained_split: str
    retained_image_ids: tuple[str, ...]
    excluded_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeduplicationResult:
    fingerprints: tuple[ImageFingerprint, ...]
    record_to_image: tuple[RecordImageMapping, ...]
    rejected: tuple[QuarantineRecord, ...]
    exact_groups: tuple[DuplicateGroup, ...]
    near_groups: tuple[DuplicateGroup, ...]
    resolutions: tuple[DuplicateResolution, ...]


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _record_value(record: Mapping[str, Any] | object, name: str) -> object:
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name, None)


def _text(record: Mapping[str, Any] | object, name: str) -> str:
    value = _record_value(record, name)
    if not isinstance(value, str) or not value:
        raise DeduplicationError(_problem(f"duplicate record has no nonempty {name}", "the manifest omits an image identity, path, or split", f"provide a nonempty {name} for every record"))
    return value


def _path(file_path: str, image_root: Path | None) -> Path:
    candidate = Path(file_path)
    return candidate if candidate.is_absolute() or image_root is None else image_root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(source_image_id: str, file_path: str, split: str, path: Path) -> ImageFingerprint:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                perceptual = str(imagehash.phash(image.convert("RGB")))
            sha256 = _sha256(path)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DeduplicationError(_problem(f"image {path} cannot be fingerprinted", str(error), "replace the local image or keep its quarantine record")) from error
    return ImageFingerprint(source_image_id, file_path, split, sha256, perceptual)


def _sort_fingerprint(item: ImageFingerprint) -> tuple[str, str, str]:
    return (item.source_image_id.casefold(), item.source_image_id, item.file_path)


def _image_key(source_image_id: str, file_path: str) -> str:
    """Keep image identity distinct from object-level annotation rows."""
    return f"{source_image_id}\t{file_path}"


def _group_id(kind: str, members: Iterable[ImageFingerprint]) -> str:
    identities = "\n".join(f"{item.source_image_id}\t{item.file_path}" for item in sorted(members, key=_sort_fingerprint))
    return f"{kind}-{hashlib.sha256(identities.encode('utf-8')).hexdigest()[:16]}"


def _components(fingerprints: list[ImageFingerprint], linked: Iterable[tuple[int, int]]) -> list[list[ImageFingerprint]]:
    parents = list(range(len(fingerprints)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in linked:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root
    grouped: dict[int, list[ImageFingerprint]] = defaultdict(list)
    for index, item in enumerate(fingerprints):
        grouped[find(index)].append(item)
    return [sorted(group, key=_sort_fingerprint) for group in grouped.values() if len(group) > 1]


def _resolution(group: list[ImageFingerprint], group_id: str) -> DuplicateResolution:
    retained_split = min((item.split for item in group), key=lambda split: _SPLIT_RANK[split])
    retained = tuple(item.source_image_id for item in group if item.split == retained_split)
    excluded = tuple(item.source_image_id for item in group if item.split != retained_split)
    return DuplicateResolution(group_id, retained_split, retained, excluded)


def _near_hash_pairs(fingerprints: list[ImageFingerprint], threshold: int) -> list[tuple[int, int]]:
    """Return every perceptual-hash pair within ``threshold`` bits without all-pairs scanning.

    Dividing a 64-bit hash into ``threshold + 1`` blocks guarantees that two
    hashes whose Hamming distance is at most ``threshold`` match exactly in at
    least one block.  Candidate pairs are still checked with the full hash, so
    this only reduces work; it never changes the duplicate decision.
    """
    if threshold >= 64:
        return [(left, right) for left in range(len(fingerprints)) for right in range(left + 1, len(fingerprints))]
    block_count = threshold + 1
    base_width, extra_bits = divmod(64, block_count)
    blocks: list[tuple[int, int]] = []
    shift = 0
    for index in range(block_count):
        width = base_width + (1 if index < extra_bits else 0)
        blocks.append((shift, (1 << width) - 1))
        shift += width
    values = [int(fingerprint.perceptual_hash, 16) for fingerprint in fingerprints]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        for block_index, (offset, mask) in enumerate(blocks):
            key = (block_index, (value >> offset) & mask)
            candidates.update((prior, index) for prior in buckets[key])
            buckets[key].append(index)
    return [pair for pair in sorted(candidates) if (values[pair[0]] ^ values[pair[1]]).bit_count() <= threshold]


def deduplicate_records(records: Iterable[Mapping[str, Any] | object], *, image_root: Path | None = None, near_hash_threshold: int = 4) -> DeduplicationResult:
    """Fingerprint local images and resolve duplicate *split assignments* deterministically.

    The returned resolution is a review record only: this function never changes
    labels, records, files, or split fields.
    """
    if isinstance(near_hash_threshold, bool) or not isinstance(near_hash_threshold, int) or near_hash_threshold < 0:
        raise DeduplicationError(_problem("near_hash_threshold must be a non-negative integer", "the configured perceptual distance is invalid", "provide an integer threshold of zero or greater"))
    fingerprints: list[ImageFingerprint] = []
    record_to_image: list[RecordImageMapping] = []
    rejected: list[QuarantineRecord] = []
    images: dict[str, tuple[str, str, str, Path]] = {}
    for record_index, record in enumerate(records):
        source_image_id = _text(record, "source_image_id")
        file_path = _text(record, "file_path")
        split = _text(record, "split")
        if split not in _SPLIT_RANK:
            raise DeduplicationError(_problem(f"split {split!r} is not covered by the duplicate policy", "the record has an unknown or unsafe split value", f"use one of {list(SPLIT_PRIORITY)} before deduplication"))
        image_key = _image_key(source_image_id, file_path)
        record_to_image.append(RecordImageMapping(record_index, image_key, source_image_id, file_path))
        path = _path(file_path, image_root)
        existing = images.get(image_key)
        if existing is not None:
            if existing[2] != split:
                raise DeduplicationError(_problem(f"image {source_image_id!r} has conflicting split assignments", "object-level rows for one source image do not agree on split", "assign every annotation for the same image to one split before deduplication"))
            continue
        images[image_key] = (source_image_id, file_path, split, path)
    for source_image_id, file_path, split, path in images.values():
        if not path.is_file():
            rejected.append(QuarantineRecord("image", source_image_id, file_path, "IMAGE_MISSING", {"problem": f"image {path} is missing", "likely_cause": "the manifest points to a nonexistent local file", "remediation": "restore the image or correct file_path; no source file was changed"}))
            continue
        try:
            fingerprints.append(_fingerprint(source_image_id, file_path, split, path))
        except DeduplicationError as error:
            rejected.append(QuarantineRecord("image", source_image_id, file_path, "IMAGE_UNDECODABLE", {"problem": f"image {path} cannot be fingerprinted", "likely_cause": str(error), "remediation": "replace the local image or retain this record in quarantine"}))
    fingerprints.sort(key=_sort_fingerprint)
    exact_indices: list[tuple[int, int]] = []
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        by_sha[fingerprint.sha256].append(index)
    exact_groups = tuple(
        DuplicateGroup(_group_id("exact", [fingerprints[index] for index in indices]), "exact", tuple(fingerprints[index] for index in indices))
        for _, indices in sorted(by_sha.items()) if len(indices) > 1
    )
    for indices in by_sha.values():
        exact_indices.extend((indices[0], index) for index in indices[1:])
    near_edges = list(exact_indices)
    near_edges.extend(_near_hash_pairs(fingerprints, near_hash_threshold))
    near_groups_list: list[DuplicateGroup] = []
    for group in _components(fingerprints, near_edges):
        if any(left_item.sha256 != right_item.sha256 and imagehash.hex_to_hash(left_item.perceptual_hash) - imagehash.hex_to_hash(right_item.perceptual_hash) <= near_hash_threshold for position, left_item in enumerate(group) for right_item in group[position + 1:]):
            near_groups_list.append(DuplicateGroup(_group_id("near", group), "near", tuple(group)))
    near_groups = tuple(sorted(near_groups_list, key=lambda group: group.group_id))
    resolution_components = _components(fingerprints, near_edges)
    resolutions = tuple(sorted((_resolution(group, _group_id("resolution", group)) for group in resolution_components), key=lambda record: record.group_id))
    return DeduplicationResult(tuple(fingerprints), tuple(record_to_image), tuple(rejected), exact_groups, near_groups, resolutions)
