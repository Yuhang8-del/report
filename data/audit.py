"""Dataset protocol audit and immutable report-ready summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from fruit_ssod.reporting.dataset_figures import render_annotation_montage


class DatasetAuditError(ValueError):
    """Raised for malformed audit input or unsafe output requests."""


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _freeze(value: Any) -> Any:
    """Copy JSON-compatible public values into recursively immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class AuditFinding:
    """One stable, machine-readable protocol finding."""

    code: str
    severity: str
    message: str
    details: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze(dict(self.details)))

    def mapping(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class DatasetAuditResult:
    """Audit result deliberately keeps both human and report-building data."""

    findings: tuple[AuditFinding, ...]
    manifest_rows: tuple[dict[str, Any], ...]
    class_box_counts: Mapping[str, Mapping[str, Mapping[str, int]]]
    label_budget_membership: Mapping[str, Mapping[str, Any]]
    source_license_summary: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "manifest_rows", tuple(_freeze(dict(row)) for row in self.manifest_rows))
        object.__setattr__(self, "class_box_counts", _freeze(dict(self.class_box_counts)))
        object.__setattr__(self, "label_budget_membership", _freeze(dict(self.label_budget_membership)))
        object.__setattr__(self, "source_license_summary", tuple(_freeze(dict(row)) for row in self.source_license_summary))

    @property
    def critical_finding_count(self) -> int:
        return sum(finding.severity == "critical" for finding in self.findings)

    def mapping(self, *, montage_sample_count: int | None = None) -> dict[str, Any]:
        output = {
            "manifest_version": "1.0",
            "critical_finding_count": self.critical_finding_count,
            "finding_count": len(self.findings),
            "findings": [finding.mapping() for finding in self.findings],
            "class_box_counts": _thaw(self.class_box_counts),
            "label_budget_membership": _thaw(self.label_budget_membership),
            "source_license_summary": _thaw(self.source_license_summary),
        }
        if montage_sample_count is not None:
            output["sample_annotation_montage_image_count"] = montage_sample_count
        return output


def _as_rows(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DatasetAuditError(_problem("an annotation record is not an object", f"record {index} is {type(record).__name__}", "provide an array of JSON annotation objects"))
        rows.append(dict(record))
    return tuple(rows)


def _text(row: Mapping[str, Any], name: str, fallback: str = "<missing>") -> str:
    value = row.get(name)
    return value if isinstance(value, str) and value else fallback


def _class_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("class_id")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _box_is_legal(row: Mapping[str, Any]) -> bool:
    box = row.get("xyxy")
    width, height = row.get("width"), row.get("height")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return False
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0 or not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        return False
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in box):
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _image_hash(row: Mapping[str, Any]) -> str | None:
    for key in ("image_hash", "image_sha256", "sha256"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _file_sha256(file_path: str, image_root: Path | None) -> str | None:
    path = Path(file_path)
    if not path.is_absolute():
        if image_root is None:
            return None
        path = image_root / path
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _budget_summary(split_manifest: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if split_manifest is None:
        return {}
    values = split_manifest.get("budget_image_ids")
    if not isinstance(values, Mapping):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for budget, image_ids in values.items():
        if not isinstance(budget, str) or not budget.isdigit() or not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes)) or any(not isinstance(item, str) for item in image_ids):
            raise DatasetAuditError(_problem("label budget membership is malformed", "budget names must be decimal percentages and members must be image IDs", "use a create_splits split_manifest.json with numeric budget keys"))
        output[budget] = {"image_count": len(image_ids), "source_image_ids": list(image_ids)}
    return dict(sorted(output.items(), key=lambda item: int(item[0])))


def _split_membership(split_manifest: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Read the Task 8 image-ID protocol without trusting stale record splits."""
    if split_manifest is None or "split_image_ids" not in split_manifest:
        return None
    protected = split_manifest["split_image_ids"]
    train_pool = split_manifest.get("train_pool_image_ids", [])
    unlabeled = split_manifest.get("unlabeled_image_ids", [])
    if not isinstance(protected, Mapping) or not isinstance(train_pool, Sequence) or isinstance(train_pool, (str, bytes)) or not isinstance(unlabeled, Sequence) or isinstance(unlabeled, (str, bytes)):
        raise DatasetAuditError(_problem("split manifest membership is malformed", "split_image_ids, train_pool_image_ids, or unlabeled_image_ids has an unsupported type", "provide the Task 8 split_manifest.json unchanged"))
    assignments: dict[str, str] = {}

    def add(split: str, ids: object) -> None:
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or any(not isinstance(item, str) or not item for item in ids):
            raise DatasetAuditError(_problem("split manifest membership is malformed", f"{split} does not contain an image-ID array", "regenerate the Task 8 split manifest"))
        for image_id in ids:
            prior = assignments.get(image_id)
            if prior is not None and prior != split:
                raise DatasetAuditError(_problem("an image belongs to multiple protocol splits", f"{image_id!r} is in both {prior} and {split}", "regenerate split artifacts so every image has exactly one effective split"))
            assignments[image_id] = split

    for split, image_ids in protected.items():
        if not isinstance(split, str):
            raise DatasetAuditError(_problem("split manifest has a non-string split name", "a JSON object key was not a protocol split", "regenerate the Task 8 split manifest"))
        add(split, image_ids)
    add("train_pool", train_pool)
    add("unlabeled", unlabeled)
    return assignments


def verify_sealed_label_artifacts(split_output_root: Path, split_manifest: Mapping[str, Any]) -> tuple[AuditFinding, ...]:
    """Verify Task 8's test and pseudo-audit labels against saved fingerprints.

    The validator reports integrity defects as audit findings rather than
    treating unavailable artifacts as a successful verification.
    """
    fingerprints = split_manifest.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        return (AuditFinding("SEALED_LABEL_FINGERPRINTS_MISSING", "critical", "The split manifest has no protected-label fingerprints.", {"split_output_root": str(split_output_root)}),)
    findings: list[AuditFinding] = []
    for split in ("test", "pseudo_audit"):
        expected = fingerprints.get(f"protected/{split}")
        path = split_output_root / "protected_splits" / f"{split}_labels.json"
        if not isinstance(expected, str) or not expected:
            findings.append(AuditFinding("SEALED_LABEL_FINGERPRINT_MISSING", "critical", "The split manifest lacks a protected-label fingerprint.", {"split": split, "path": str(path)}))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            findings.append(AuditFinding("SEALED_LABEL_ARTIFACT_UNAVAILABLE", "critical", "A sealed protected-label artifact cannot be read.", {"split": split, "path": str(path), "cause": str(error)}))
            continue
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
            findings.append(AuditFinding("SEALED_LABEL_ARTIFACT_MALFORMED", "critical", "A sealed protected-label artifact has no records array.", {"split": split, "path": str(path)}))
            continue
        try:
            canonical_records = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            findings.append(AuditFinding("SEALED_LABEL_ARTIFACT_MALFORMED", "critical", "A sealed protected-label artifact contains values outside the canonical JSON contract.", {"split": split, "path": str(path), "cause": str(error)}))
            continue
        actual = hashlib.sha256(canonical_records.encode("utf-8")).hexdigest()
        if actual != expected:
            findings.append(AuditFinding("SEALED_LABEL_FINGERPRINT_MISMATCH", "critical", "A sealed protected-label artifact differs from the Task 8 fingerprint.", {"split": split, "path": str(path), "expected": expected, "actual": actual}))
    return tuple(findings)


def audit_annotations(
    records: Iterable[Mapping[str, Any]], *, expected_class_ids: Sequence[int] = (0, 1, 2, 3, 4), required_splits: Sequence[str] = ("train_pool", "validation", "test", "pseudo_audit"), split_manifest: Mapping[str, Any] | None = None, image_hashes: Mapping[tuple[str, str], str] | None = None, require_hashes: bool = False, image_root: Path | None = None, additional_findings: Iterable[AuditFinding] = ()
) -> DatasetAuditResult:
    """Audit raw canonical annotation rows without silently fixing violations."""
    rows = _as_rows(records)
    memberships = _split_membership(split_manifest)
    expected = tuple(expected_class_ids)
    findings: list[AuditFinding] = list(additional_findings)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_hash: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    license_images: dict[tuple[str, str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    license_boxes: dict[tuple[str, str, str, str], int] = defaultdict(int)
    manifest_rows: list[dict[str, Any]] = []
    missing_hash_images: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        source, image_id = _text(row, "source"), _text(row, "source_image_id")
        split = memberships.get(image_id, "<unassigned>") if memberships is not None else _text(row, "split")
        if memberships is not None and image_id not in memberships:
            findings.append(AuditFinding("SPLIT_MEMBERSHIP_MISSING", "critical", "A cleaned annotation image is absent from the fixed split protocol.", {"record_index": str(index), "source_image_id": image_id}))
        by_split[split].append(row)
        class_id = _class_id(row)
        is_unlabeled = split == "unlabeled"
        if class_id is not None and not is_unlabeled:
            counts[split][source][str(class_id)] += 1
        license_data = row.get("license_metadata")
        license_data = license_data if isinstance(license_data, Mapping) else {}
        license_key = (source, _text(license_data, "name"), _text(license_data, "url", ""), _text(license_data, "attribution", ""))
        license_images[license_key].add((source, image_id, _text(row, "file_path")))
        if not is_unlabeled:
            license_boxes[license_key] += 1
        image_hash = _image_hash(row)
        if image_hash is None and image_hashes is not None:
            image_hash = image_hashes.get((image_id, _text(row, "file_path")))
        if image_hash is None:
            image_hash = _file_sha256(_text(row, "file_path", ""), image_root)
        image_key = (source, image_id)
        if image_hash is None and image_key not in missing_hash_images:
            missing_hash_images.add(image_key)
            findings.append(AuditFinding("IMAGE_HASH_UNAVAILABLE", "critical", "No image hash is available, so exact cross-split duplicate detection cannot be trusted.", {"source": source, "source_image_id": image_id, "required_from_cleaned_manifest": str(require_hashes).lower()}))
        if image_hash:
            by_hash[image_hash].add(split)
        if not is_unlabeled and not _box_is_legal(row):
            findings.append(AuditFinding("ILLEGAL_BBOX", "critical", "Annotation has an invalid or out-of-bounds xyxy box.", {"record_index": str(index), "source_image_id": image_id, "split": split}))
        manifest_rows.append({
            "source": source, "source_image_id": image_id, "original_source_image_id": image_id, "file_path": _text(row, "file_path"), "effective_split": split,
            "label_status": "unlabeled" if is_unlabeled else _text(row, "label_status"), "class_id": "" if class_id is None or is_unlabeled else class_id,
            "width": row.get("width", ""), "height": row.get("height", ""), "xyxy": json.dumps(None if is_unlabeled else row.get("xyxy"), ensure_ascii=False),
            "image_hash": image_hash or "", "license_name": _text(license_data, "name"), "license_url": _text(license_data, "url", ""), "attribution": _text(license_data, "attribution", ""),
            "cleaning_status": "unlabeled_label_excluded" if is_unlabeled else ("accepted" if _box_is_legal(row) else "audit_failed_illegal_bbox"),
        })
    for split in required_splits:
        split_rows = by_split.get(split, [])
        if not split_rows:
            findings.append(AuditFinding("EMPTY_SPLIT", "critical", "A required protocol split has no annotations.", {"split": split}))
            continue
        present = {_class_id(row) for row in split_rows}
        for class_id in expected:
            if class_id not in present:
                findings.append(AuditFinding("MISSING_CLASS", "critical", "A required class is absent from a protocol split.", {"split": split, "class_id": str(class_id)}))
    for image_hash, splits in sorted(by_hash.items()):
        if len(splits) > 1:
            findings.append(AuditFinding("DUPLICATE_HASH_CROSS_SPLIT", "critical", "The same image hash occurs in more than one split.", {"image_hash": image_hash, "splits": ",".join(sorted(splits))}))
    budget_membership = _budget_summary(split_manifest)
    if memberships is not None:
        for budget, summary in budget_membership.items():
            eligible: list[str] = []
            excluded: list[str] = []
            for image_id in summary["source_image_ids"]:
                if memberships.get(image_id) == "train_pool":
                    eligible.append(image_id)
                else:
                    excluded.append(image_id)
                    findings.append(AuditFinding("BUDGET_MEMBER_OUTSIDE_LABELED_TRAIN_POOL", "critical", "A label-budget member is not in the labeled train pool.", {"budget": budget, "source_image_id": image_id, "effective_split": memberships.get(image_id, "<unassigned>")}))
            summary["source_image_ids"] = eligible
            summary["image_count"] = len(eligible)
            if excluded:
                summary["excluded_non_labeled_image_ids"] = excluded
    normalized_counts = {split: {source: dict(sorted(classes.items(), key=lambda item: int(item[0]))) for source, classes in sorted(sources.items())} for split, sources in sorted(counts.items())}
    license_summary = tuple({"source": source, "license_name": name, "license_url": url, "attribution": attribution, "image_count": len(license_images[key]), "box_count": license_boxes[key]} for key in sorted(license_images) for source, name, url, attribution in (key,))
    return DatasetAuditResult(tuple(findings), tuple(manifest_rows), normalized_counts, budget_membership, license_summary)


def _manifest_by_image(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["source_image_id"]), str(row["file_path"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, annotations in sorted(grouped.items()):
        first = annotations[0]
        classes = sorted({str(row["class_id"]) for row in annotations if row["class_id"] != ""}, key=int)
        output.append({
            "source": first["source"], "source_image_id": first["source_image_id"], "original_source_image_id": first["original_source_image_id"], "file_path": first["file_path"], "image_hash": first["image_hash"],
            "effective_split": first["effective_split"], "classes": "" if first["effective_split"] == "unlabeled" else ";".join(classes), "class_count": 0 if first["effective_split"] == "unlabeled" else len(classes), "box_count": 0 if first["effective_split"] == "unlabeled" else len(annotations), "license_name": first["license_name"], "license_url": first["license_url"], "attribution": first["attribution"],
            "cleaning_status": "unlabeled_label_excluded" if first["effective_split"] == "unlabeled" else ("accepted" if all(row["cleaning_status"] == "accepted" for row in annotations) else "audit_failed_illegal_bbox"),
        })
    return tuple(output)


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ["source", "source_image_id", "original_source_image_id", "file_path", "image_hash", "effective_split", "classes", "class_count", "box_count", "license_name", "license_url", "attribution", "cleaning_status"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_audit_outputs(result: DatasetAuditResult, output_root: Path, *, image_root: Path | None = None) -> tuple[Path, ...]:
    """Atomically publish report-ready audit artifacts to a new output directory."""
    root = output_root.resolve(strict=False)
    if root.exists():
        raise DatasetAuditError(_problem("audit output root already exists", f"{root} would be overwritten", "choose a new empty output directory"))
    ancestor = root.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise DatasetAuditError(_problem("audit output has a file as an ancestor", f"{ancestor} is not a directory", "choose an output path whose parent hierarchy is directories"))
        ancestor = ancestor.parent
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        montage_count = render_annotation_montage(result.manifest_rows, temporary / "sample_annotation_montage.png", image_root=image_root)
        (temporary / "data_manifest.csv").write_text(_csv(_manifest_by_image(result.manifest_rows)), encoding="utf-8", newline="")
        (temporary / "dataset_audit.json").write_text(json.dumps(result.mapping(montage_sample_count=montage_count), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, root)
    except OSError as error:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise DatasetAuditError(_problem("audit outputs could not be written atomically", str(error), "ensure the output parent is writable and retry with a new output directory")) from error
    return tuple(root / name for name in ("data_manifest.csv", "dataset_audit.json", "sample_annotation_montage.png"))
