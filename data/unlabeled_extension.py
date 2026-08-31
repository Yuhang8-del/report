"""Seal a reviewed auxiliary image-only source into Task 8's no-label pool.

The primary Open Images split remains immutable.  This module produces a new,
explicitly named protocol extension that retains every protected/budget binding
while adding only ``train_pool``/``unlabeled`` records from an auxiliary
manifest.  It never reads or creates object annotations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.schema import AnnotationValidationError, UnlabeledImageRecord
from fruit_ssod.pseudo.generator import PseudoGenerationError, canonical_unlabeled_fingerprint, load_unlabeled_manifest


class UnlabeledExtensionError(ValueError):
    """Raised when an auxiliary image-only protocol extension is unsafe."""


_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True)
class UnlabeledExtensionResult:
    root: Path
    unlabeled_manifest: Path
    split_manifest: Path
    record_count: int
    auxiliary_record_count: int
    split_fingerprint: str


def _problem(problem: str, cause: str, remediation: str) -> UnlabeledExtensionError:
    return UnlabeledExtensionError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the original UTF-8 JSON artifact") from error
    if not isinstance(payload, Mapping):
        raise _problem(f"{description} is not a JSON object", "the top-level payload is malformed", "use the importer or split output without manual edits")
    return payload


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _problem("source artifact cannot be hashed", str(error), "restore a readable input artifact") from error


def _canonical_sha(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _problem("extension evidence is not canonical JSON", str(error), "use JSON-compatible source metadata") from error
    return hashlib.sha256(encoded).hexdigest()


def _safe_prefix(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized or not _SAFE_PREFIX.fullmatch(normalized) or ".." in Path(normalized).parts:
        raise _problem("auxiliary path prefix is unsafe", repr(value), "use a relative directory below the configured raw data root")
    return normalized


def _record_mapping(record: UnlabeledImageRecord) -> dict[str, Any]:
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


def _auxiliary_records(manifest_path: Path, *, raw_prefix: str) -> tuple[tuple[UnlabeledImageRecord, ...], list[dict[str, str]]]:
    manifest = _read_json(manifest_path, description="auxiliary unlabeled manifest")
    rows = manifest.get("records")
    if manifest.get("label_status") != "unlabeled" or manifest.get("split") != "train_pool" or not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise _problem("auxiliary manifest is not an unlabeled train-pool artifact", "it must contain nonempty image-only records with split=train_pool", "import Fruits-360 with import_auxiliary_data before extending the pool")
    result: list[UnlabeledImageRecord] = []
    mappings: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise _problem("auxiliary manifest record is malformed", repr(row), "restore the importer-generated manifest")
        try:
            original = UnlabeledImageRecord.from_mapping(row)
        except (AnnotationValidationError, TypeError, ValueError) as error:
            raise _problem("auxiliary manifest record is invalid", str(error), "restore the importer-generated image-only manifest") from error
        if original.split != "train_pool" or original.label_status != "unlabeled":
            raise _problem("auxiliary record is not unlabeled train-pool data", repr(original.source_image_id), "do not mix evaluation or human-labeled data into pseudo-label inputs")
        local = Path(original.file_path)
        if local.is_absolute() or ".." in local.parts:
            raise _problem("auxiliary record file path is unsafe", repr(original.file_path), "use a relative importer file path beneath the approved source root")
        # Student snapshots use source_image_id in filenames, so source IDs
        # must never contain a path separator.  The mapping remains explicit
        # in extension.json for auditability.
        digest = hashlib.sha256(f"{original.source}\0{original.source_image_id}".encode("utf-8")).hexdigest()
        composed_id = f"aux-{original.source}-{digest[:24]}"
        composed_path = f"{raw_prefix}/{local.as_posix()}"
        composed = UnlabeledImageRecord(
            source=original.source,
            source_image_id=composed_id,
            file_path=composed_path,
            width=original.width,
            height=original.height,
            split="train_pool",
            label_status="unlabeled",
            license_metadata=original.license_metadata,
        )
        result.append(composed)
        mappings.append({"auxiliary_source": original.source, "auxiliary_source_image_id": original.source_image_id, "composed_source_image_id": composed_id, "composed_file_path": composed_path})
    if len({record.source_image_id for record in result}) != len(result):
        raise _problem("auxiliary records produce duplicate source IDs", "hash-derived IDs unexpectedly collide", "use a fresh, unmodified auxiliary manifest and report the collision")
    return tuple(result), mappings


def extend_unlabeled_pool(base_unlabeled: Path, base_split: Path, auxiliary_manifest: Path, output_root: Path, *, raw_prefix: str) -> UnlabeledExtensionResult:
    """Atomically seal a primary no-label pool plus a reviewed auxiliary source."""
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem("unlabeled extension output already exists", str(root), "preserve immutable evidence and choose a fresh output directory")
    prefix = _safe_prefix(raw_prefix)
    try:
        base = load_unlabeled_manifest(base_unlabeled.resolve(strict=True), split_manifest_path=base_split.resolve(strict=True))
    except (OSError, PseudoGenerationError) as error:
        raise _problem("base Task 8 unlabeled membership is invalid", str(error), "use the unmodified paired unlabeled.json and split_manifest.json") from error
    split = dict(_read_json(base_split, description="base split manifest"))
    fingerprints = split.get("fingerprints")
    protected_ids = split.get("split_image_ids")
    if not isinstance(fingerprints, Mapping) or not isinstance(protected_ids, Mapping) or not isinstance(fingerprints.get("split_protocol"), str):
        raise _problem("base split manifest lacks sealed protocol facts", "fingerprints or protected memberships are missing", "regenerate Task 8 split outputs before extending them")
    auxiliary, id_mapping = _auxiliary_records(auxiliary_manifest.resolve(strict=True), raw_prefix=prefix)
    base_ids = {record.source_image_id for record in base.records}
    protected = {value for rows in protected_ids.values() if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) for value in rows}
    if base_ids & {record.source_image_id for record in auxiliary} or protected & {record.source_image_id for record in auxiliary}:
        raise _problem("auxiliary extension overlaps existing split identities", "an auxiliary record collides with primary or protected membership", "choose a fresh auxiliary source or correct its manifest")
    combined = (*base.records, *auxiliary)
    combined_fingerprint = canonical_unlabeled_fingerprint(combined)
    extension_evidence = {
        "schema_version": "1.0",
        "protocol": "auxiliary_unlabeled_extension_v1",
        "base_split_protocol_sha256": fingerprints["split_protocol"],
        "base_unlabeled_sha256": _sha256_file(base_unlabeled),
        "base_split_sha256": _sha256_file(base_split),
        "auxiliary_manifest_sha256": _sha256_file(auxiliary_manifest),
        "auxiliary_member_count": len(auxiliary),
        "auxiliary_path_prefix": prefix,
        "combined_unlabeled_fingerprint": combined_fingerprint,
    }
    extension_fingerprint = _canonical_sha(extension_evidence)
    revised_fingerprints = dict(fingerprints)
    revised_fingerprints["unlabeled"] = combined_fingerprint
    revised_fingerprints["split_protocol"] = extension_fingerprint
    split["unlabeled_image_ids"] = [record.source_image_id for record in combined]
    split["fingerprints"] = revised_fingerprints
    split["unlabeled_extension"] = extension_evidence
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        (temporary / "unlabeled.json").write_text(json.dumps({"records": [_record_mapping(record) for record in combined]}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "fingerprints.json").write_text(json.dumps({"fingerprints": revised_fingerprints}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "extension.json").write_text(json.dumps({**extension_evidence, "source_id_mapping": id_mapping}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, root)
        temporary = None
        return UnlabeledExtensionResult(root, root / "unlabeled.json", root / "split_manifest.json", len(combined), len(auxiliary), extension_fingerprint)
    except OSError as error:
        raise _problem("unlabeled extension could not be published", str(error), "choose a writable empty output directory") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
