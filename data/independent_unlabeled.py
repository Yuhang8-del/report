"""Seal an image-only pool that is demonstrably independent of a Teacher.

This importer intentionally scans pixels and filenames only.  It never opens
annotation files and its public unlabeled manifest has the strict schema
accepted by :mod:`fruit_ssod.pseudo.generator`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord
from fruit_ssod.pseudo.generator import canonical_unlabeled_fingerprint


class IndependentUnlabeledError(ValueError):
    """Raised when an independent image-only pool cannot be safely sealed."""


@dataclass(frozen=True)
class IndependentUnlabeledResult:
    root: Path
    unlabeled_manifest: Path
    split_manifest: Path
    evidence: Path
    record_count: int
    split_fingerprint: str


def _problem(problem: str, cause: str, remediation: str) -> IndependentUnlabeledError:
    return IndependentUnlabeledError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise _problem("image cannot be hashed", str(error), "restore a readable regular image") from error
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise _problem("evidence cannot be canonicalized", str(error), "use only JSON-compatible sealed inputs") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the original UTF-8 JSON artifact") from error
    if not isinstance(value, dict):
        raise _problem(f"{description} is malformed", "expected a JSON object", "use an importer-generated split manifest")
    return value


def _images(root: Path) -> tuple[Path, ...]:
    accepted = {".jpg", ".jpeg", ".png"}
    paths = tuple(sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in accepted), key=lambda path: path.as_posix()))
    if not paths:
        raise _problem("independent image directory is empty", str(root), "copy reviewed image files before sealing the pool")
    return paths


def _teacher_hashes(root: Path) -> set[str]:
    if not root.is_dir():
        raise _problem("Teacher dataset root is missing", str(root), "supply the materialized dataset used to train the Teacher")
    return {_sha256(path) for path in _images(root)}


def _record_mapping(record: UnlabeledImageRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "source_image_id": record.source_image_id,
        "file_path": record.file_path,
        "width": record.width,
        "height": record.height,
        "split": record.split,
        "label_status": record.label_status,
        "license_metadata": {
            "name": record.license_metadata.name,
            "url": record.license_metadata.url,
            "attribution": record.license_metadata.attribution,
        },
    }


def seal_independent_unlabeled_pool(
    *,
    base_split_manifest: Path,
    image_directory: Path,
    source_root: Path,
    relative_prefix: str,
    teacher_dataset_root: Path,
    output_root: Path,
    source: str = "open_images_v7",
    license_metadata: LicenseMetadata | None = None,
) -> IndependentUnlabeledResult:
    """Publish a fresh label-free membership sealed against a Teacher snapshot.

    ``image_directory`` must live below ``source_root``.  Images are compared
    by SHA-256 with every image in ``teacher_dataset_root``; any equality is a
    hard failure, so this function cannot quietly create a pseudo-label pool
    from images already seen by the selected Teacher.
    """
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem("independent pool output already exists", str(root), "preserve immutable evidence and choose a fresh output directory")
    base = _read_object(base_split_manifest.resolve(strict=True), description="base split manifest")
    split_ids = base.get("split_image_ids")
    fingerprints = base.get("fingerprints")
    if not isinstance(split_ids, Mapping) or set(split_ids) != {"validation", "test", "pseudo_audit", "external_test"} or not isinstance(fingerprints, Mapping):
        raise _problem("base split manifest lacks protected membership", "validation, test, pseudo-audit, or fingerprints are unavailable", "use the immutable Task 8 split manifest")
    source_base = source_root.resolve(strict=True)
    directory = image_directory.resolve(strict=True)
    try:
        local = directory.relative_to(source_base)
    except ValueError as error:
        raise _problem("independent image directory escapes source root", str(directory), "copy images below the shared Student source root") from error
    requested_prefix = Path(relative_prefix.replace("\\", "/"))
    if requested_prefix.is_absolute() or ".." in requested_prefix.parts or requested_prefix.as_posix().strip("./") != local.as_posix():
        raise _problem("relative image prefix does not match image directory", relative_prefix, "use the exact safe path below source_root")
    teacher = _teacher_hashes(teacher_dataset_root.resolve(strict=True))
    metadata = license_metadata or LicenseMetadata(name="Open Images V7 image license", url="https://creativecommons.org/licenses/by/2.0/", attribution="Open Images V7")
    protected = {str(identifier) for rows in split_ids.values() if isinstance(rows, list) for identifier in rows}
    records: list[UnlabeledImageRecord] = []
    evidence_rows: list[dict[str, str]] = []
    for image in _images(directory):
        digest = _sha256(image)
        if digest in teacher:
            raise _problem("candidate image overlaps Teacher training snapshot", image.name, "exclude the image and rebuild the independent image-only pool")
        relative = image.relative_to(source_base).as_posix()
        original_id = image.relative_to(directory).with_suffix("").as_posix()
        composed_id = "independent-" + hashlib.sha256(f"{source}\0{relative}".encode("utf-8")).hexdigest()[:24]
        if composed_id in protected:
            raise _problem("candidate identity overlaps a protected split", composed_id, "use a fresh independent image source")
        try:
            with Image.open(image) as opened:
                width, height = opened.size
        except (OSError, ValueError) as error:
            raise _problem("candidate image cannot be decoded", str(image), "restore a Pillow-readable image") from error
        records.append(UnlabeledImageRecord(source, composed_id, relative, width, height, "train_pool", "unlabeled", metadata))
        evidence_rows.append({"source_image_relative_path": relative, "source_image_local_id": original_id, "source_image_sha256": digest, "composed_source_image_id": composed_id})
    if len({record.source_image_id for record in records}) != len(records):
        raise _problem("candidate image identifiers collide", "hash-derived membership IDs are not unique", "use a fresh image source and report the collision")
    combined_fingerprint = canonical_unlabeled_fingerprint(tuple(records))
    evidence = {
        "schema_version": "1.0",
        "protocol": "independent_image_only_teacher_pool_v1",
        "base_split_manifest_sha256": _sha256(base_split_manifest),
        "teacher_dataset_root": str(teacher_dataset_root.resolve(strict=True)),
        "teacher_snapshot_image_count": len(teacher),
        "image_directory": str(directory),
        "record_count": len(records),
        "source": source,
        "label_access": "no annotation files opened; records derived only from image bytes, filenames, and fixed source license metadata",
        "unlabeled_fingerprint": combined_fingerprint,
        "records": evidence_rows,
    }
    split = dict(base)
    revised = dict(fingerprints)
    revised["unlabeled"] = combined_fingerprint
    split["unlabeled_image_ids"] = [record.source_image_id for record in records]
    split["independent_unlabeled_pool"] = {key: value for key, value in evidence.items() if key != "records"}
    revised["split_protocol"] = _canonical_sha({"base": fingerprints.get("split_protocol"), "independent_unlabeled_pool": split["independent_unlabeled_pool"]})
    split["fingerprints"] = revised
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        (temporary / "unlabeled.json").write_text(json.dumps({"records": [_record_mapping(record) for record in records]}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (temporary / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (temporary / "independence_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root)
        temporary = None
        return IndependentUnlabeledResult(root, root / "unlabeled.json", root / "split_manifest.json", root / "independence_evidence.json", len(records), revised["split_protocol"])
    except OSError as error:
        raise _problem("independent pool cannot be published", str(error), "choose a fresh writable output directory") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
