"""Aggregate cleaned object rows into deterministic image-level split inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


class CandidateManifestError(ValueError):
    """Raised when a cleaned manifest cannot be safely converted to image groups."""


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _load_cleaned(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateManifestError(_problem(f"cleaned manifest {path} cannot be read", str(error), "use clean_dataset output without modification")) from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list) or not isinstance(payload.get("deduplication"), Mapping):
        raise CandidateManifestError(_problem("cleaned manifest is incomplete", "records or deduplication evidence is missing", "use a successful clean_dataset output"))
    return payload


def _duplicate_groups(deduplication: Mapping[str, Any]) -> dict[str, str]:
    """Build connected duplicate components, retaining all exact/near evidence."""
    parent: dict[str, str] = {}

    def root(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = root(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for collection_name in ("exact_groups", "near_groups"):
        groups = deduplication.get(collection_name, [])
        if not isinstance(groups, list):
            raise CandidateManifestError(_problem("deduplication group evidence is malformed", f"{collection_name} is not an array", "regenerate the cleaned manifest"))
        for group in groups:
            if not isinstance(group, Mapping) or not isinstance(group.get("member_image_ids"), list):
                raise CandidateManifestError(_problem("deduplication group evidence is malformed", f"{collection_name} contains an invalid group", "regenerate the cleaned manifest"))
            members = group["member_image_ids"]
            if any(not isinstance(member, str) or not member for member in members):
                raise CandidateManifestError(_problem("deduplication group has invalid image ID", repr(members), "regenerate the cleaned manifest"))
            if members:
                for member in members[1:]:
                    union(members[0], member)
    return {image_id: f"duplicate:{root(image_id)}" for image_id in parent}


def build_candidate_manifest(cleaned_manifest: Path, output: Path) -> int:
    """Write one CandidateImageRecord-compatible row per source image."""
    if output.exists():
        raise CandidateManifestError(_problem(f"candidate manifest output {output} already exists", "split inputs are immutable once constructed", "choose a fresh output path"))
    payload = _load_cleaned(cleaned_manifest.resolve(strict=True))
    records = payload["records"]
    groups = _duplicate_groups(payload["deduplication"])
    images: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise CandidateManifestError(_problem("cleaned manifest record is malformed", f"record {index} is not an object", "regenerate the cleaned manifest"))
        try:
            source, image_id, file_path = row["source"], row["source_image_id"], row["file_path"]
            width, height, class_id, xyxy = row["width"], row["height"], row["class_id"], row["xyxy"]
            license_metadata = row["license_metadata"]
        except KeyError as error:
            raise CandidateManifestError(_problem("cleaned record omits a canonical field", str(error), "regenerate the cleaned manifest")) from error
        if not all(isinstance(value, str) and value for value in (source, image_id, file_path)) or not isinstance(width, int) or not isinstance(height, int) or not isinstance(class_id, int) or not isinstance(xyxy, list) or not isinstance(license_metadata, Mapping):
            raise CandidateManifestError(_problem("cleaned record has invalid canonical types", f"record {index}", "regenerate the cleaned manifest"))
        key = (source, image_id)
        target = images.setdefault(key, {"source": source, "source_image_id": image_id, "file_path": file_path, "width": width, "height": height, "class_presence": set(), "labels": {}, "duplicate_group_id": groups.get(image_id, f"unique:{source}:{image_id}"), "license_metadata": dict(license_metadata)})
        if target["width"] != width or target["height"] != height or target["license_metadata"] != dict(license_metadata):
            raise CandidateManifestError(_problem("object rows for one image disagree on image metadata", image_id, "regenerate the cleaned manifest from consistent source files"))
        target["file_path"] = min(target["file_path"], file_path)
        target["class_presence"].add(class_id)
        target["labels"][(class_id, tuple(xyxy))] = {"class_id": class_id, "xyxy": xyxy}
    if not images:
        raise CandidateManifestError(_problem("cleaned manifest has no accepted records", "there are no images left after cleaning", "resolve source-data quarantine findings before splitting"))
    output_rows = []
    for item in sorted(images.values(), key=lambda value: (value["source"], value["source_image_id"], value["file_path"])):
        item["class_presence"] = sorted(item["class_presence"])
        item["labels"] = sorted(item["labels"].values(), key=lambda value: (value["class_id"], value["xyxy"]))
        output_rows.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": "1.0", "images": output_rows, "source_cleaned_manifest": str(cleaned_manifest.resolve()), "image_count": len(output_rows)}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return len(output_rows)
