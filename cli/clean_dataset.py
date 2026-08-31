"""Clean an explicit local annotation manifest without mutating source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fruit_ssod.data.cleaning import (
    DataCleaningError,
    annotation_mapping,
    clean_manifest_rows,
    write_quarantine_manifest,
)
from fruit_ssod.data.deduplication import DeduplicationError, DeduplicationResult, deduplicate_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode, clean, and deduplicate an explicit local annotation manifest. Source files are never modified.")
    parser.add_argument("--input-manifest", required=True, type=Path, help="JSON manifest containing an array of canonical annotation objects, or an object with records.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Caller-designated JSON output path for accepted records and duplicate review results.")
    parser.add_argument("--quarantine-manifest", required=True, type=Path, help="Caller-designated JSONL output path for rejected images/annotations.")
    parser.add_argument("--image-root", type=Path, help="Optional root used only to resolve relative file_path values.")
    parser.add_argument("--near-hash-threshold", default=4, type=int, help="Maximum perceptual-hash Hamming distance for a near-duplicate group (default: 4).")
    parser.add_argument("--dry-run", action="store_true", help="Report the result without creating output files or directories.")
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataCleaningError(f"Problem: input manifest {path} cannot be read as UTF-8 JSON. Likely cause: {error}. Remediation: provide a readable JSON manifest file.") from error
    rows = value.get("records") if isinstance(value, dict) else value
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise DataCleaningError("Problem: input manifest must contain an array of annotation objects. Likely cause: the JSON root is neither an array nor an object with a records array. Remediation: export canonical annotation rows as JSON objects.")
    return rows


def _resolve(path: Path, argument_name: str) -> Path:
    """Resolve a caller path before any output operation can touch it."""
    try:
        return path.resolve()
    except OSError as error:
        raise DataCleaningError(f"Problem: {argument_name} path {path} cannot be resolved. Likely cause: {error}. Remediation: provide a local path without an unreadable or cyclic link.") from error


def _referenced_source_paths(rows: Sequence[dict[str, Any]], image_root: Path | None) -> set[Path]:
    """Resolve every local source image explicitly named by the input manifest."""
    paths: set[Path] = set()
    for row in rows:
        file_path = row.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            continue
        candidate = Path(file_path)
        if not candidate.is_absolute() and image_root is not None:
            candidate = image_root / candidate
        paths.add(_resolve(candidate, "referenced source image"))
    return paths


def _validate_output_collisions(input_manifest: Path, output_manifest: Path, quarantine_manifest: Path, rows: Sequence[dict[str, Any]], image_root: Path | None) -> None:
    """Reject aliasing output paths before creating a directory or opening an output file."""
    if output_manifest == input_manifest:
        raise DataCleaningError("Problem: --output-manifest collides with --input-manifest. Likely cause: the same local path was supplied for source and generated output. Remediation: choose a distinct output manifest path.")
    if quarantine_manifest == input_manifest:
        raise DataCleaningError("Problem: --quarantine-manifest collides with --input-manifest. Likely cause: the same local path was supplied for source and generated quarantine output. Remediation: choose a distinct quarantine manifest path.")
    if output_manifest == quarantine_manifest:
        raise DataCleaningError("Problem: --output-manifest collides with --quarantine-manifest. Likely cause: both generated manifests were assigned the same path. Remediation: provide two distinct output paths.")
    source_paths = _referenced_source_paths(rows, image_root)
    if output_manifest in source_paths or quarantine_manifest in source_paths:
        raise DataCleaningError("Problem: a generated output collides with a referenced source image. Likely cause: --output-manifest or --quarantine-manifest names a file_path from the input manifest. Remediation: choose output paths outside all source image paths.")


def _deduplication_mapping(result: DeduplicationResult) -> dict[str, Any]:
    return {
        "fingerprints": [
            {"source_image_id": item.source_image_id, "file_path": item.file_path, "split": item.split, "sha256": item.sha256, "perceptual_hash": item.perceptual_hash}
            for item in result.fingerprints
        ],
        "record_to_image": [
            {"record_index": item.record_index, "image_key": item.image_key, "source_image_id": item.source_image_id, "file_path": item.file_path}
            for item in result.record_to_image
        ],
        "exact_groups": [{"group_id": group.group_id, "member_image_ids": [item.source_image_id for item in group.members]} for group in result.exact_groups],
        "near_groups": [{"group_id": group.group_id, "member_image_ids": [item.source_image_id for item in group.members]} for group in result.near_groups],
        "split_resolutions": [
            {"group_id": record.group_id, "retained_split": record.retained_split, "retained_image_ids": list(record.retained_image_ids), "excluded_image_ids": list(record.excluded_image_ids)}
            for record in result.resolutions
        ],
    }


def _write_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run cleaning/deduplication and return zero for valid quarantine findings."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.near_hash_threshold < 0:
        parser.error("--near-hash-threshold must be zero or greater")
    try:
        input_manifest = _resolve(args.input_manifest, "--input-manifest")
        output_manifest = _resolve(args.output_manifest, "--output-manifest")
        quarantine_manifest = _resolve(args.quarantine_manifest, "--quarantine-manifest")
        image_root = _resolve(args.image_root, "--image-root") if args.image_root is not None else None
        rows = _load_rows(input_manifest)
        _validate_output_collisions(input_manifest, output_manifest, quarantine_manifest, rows, image_root)
        cleaned = clean_manifest_rows(rows, image_root=image_root)
        deduplicated = deduplicate_records(cleaned.accepted, image_root=image_root, near_hash_threshold=args.near_hash_threshold)
    except (DataCleaningError, DeduplicationError) as error:
        parser.error(str(error))
    quarantine = (*cleaned.rejected, *deduplicated.rejected)
    summary = {
        "input_count": len(rows),
        "accepted_count": len(cleaned.accepted),
        "quarantine_count": len(quarantine),
        "exact_duplicate_group_count": len(deduplicated.exact_groups),
        "near_duplicate_group_count": len(deduplicated.near_groups),
        "split_resolution_count": len(deduplicated.resolutions),
        "dry_run": args.dry_run,
    }
    output = {"manifest_version": "1.0", "records": [annotation_mapping(record) for record in cleaned.accepted], "deduplication": _deduplication_mapping(deduplicated), "summary": summary}
    if not args.dry_run:
        _write_output(output_manifest, output)
        write_quarantine_manifest(quarantine_manifest, quarantine)
    print(json.dumps({"output_manifest": str(output_manifest), "quarantine_manifest": str(quarantine_manifest), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
