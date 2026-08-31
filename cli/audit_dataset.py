"""Run a local Fruit SSOD dataset protocol audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.audit import DatasetAuditError, audit_annotations, verify_sealed_label_artifacts, write_audit_outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a prepared local Fruit SSOD canonical annotation manifest.")
    parser.add_argument("--annotations", required=True, type=Path, help="JSON array of canonical annotations or an object with a records array.")
    parser.add_argument("--output-root", required=True, type=Path, help="New directory for CSV, JSON, and sample montage audit artifacts.")
    parser.add_argument("--split-manifest", type=Path, help="Optional split_manifest.json used for label-budget membership reporting.")
    parser.add_argument("--split-output-root", type=Path, help="Optional Task 8 split-output directory; verifies sealed test and pseudo-audit labels against its split manifest.")
    parser.add_argument("--image-root", type=Path, help="Optional root used only to resolve relative image paths for the sample montage.")
    return parser


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetAuditError(f"Problem: JSON input {path} cannot be read. Likely cause: {error}. Remediation: provide a readable UTF-8 JSON manifest.") from error


def _resolve_existing_file(path: Path, argument_name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DatasetAuditError(f"Problem: {argument_name} file {path} cannot be resolved. Likely cause: {error}. Remediation: provide an existing readable local JSON file.") from error
    if not resolved.is_file():
        raise DatasetAuditError(f"Problem: {argument_name} path {resolved} is not a file. Likely cause: a directory was supplied. Remediation: provide the required local JSON file.")
    return resolved


def _resolve_existing_directory(path: Path, argument_name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DatasetAuditError(f"Problem: {argument_name} directory {path} cannot be resolved. Likely cause: {error}. Remediation: provide an existing Task 8 split-output directory.") from error
    if not resolved.is_dir():
        raise DatasetAuditError(f"Problem: {argument_name} path {resolved} is not a directory. Likely cause: a file was supplied. Remediation: provide the Task 8 split-output directory.")
    return resolved


def _resolve_output_root(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError as error:
        raise DatasetAuditError(f"Problem: --output-root path {path} cannot be resolved. Likely cause: {error}. Remediation: provide a writable local output path.") from error


def _load_annotations(value: Any) -> list[Mapping[str, Any]]:
    rows = value.get("records") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DatasetAuditError("Problem: annotation manifest has no records array. Likely cause: it is not a canonical annotation JSON array/object. Remediation: provide an array or {'records': [...]} of annotation objects.")
    return rows


def _dedup_hashes(cleaned_manifest: Any) -> tuple[dict[tuple[str, str], str] | None, bool]:
    """Extract exact Task 7 SHA-256 facts without recomputing or guessing them."""
    if not isinstance(cleaned_manifest, Mapping):
        return None, False
    deduplication = cleaned_manifest.get("deduplication")
    if not isinstance(deduplication, Mapping):
        return None, False
    fingerprints = deduplication.get("fingerprints")
    if not isinstance(fingerprints, list):
        raise DatasetAuditError("Problem: cleaned manifest deduplication.fingerprints is malformed. Likely cause: the Task 7 output was edited or incomplete. Remediation: rerun clean_dataset before the audit.")
    output: dict[tuple[str, str], str] = {}
    for index, fingerprint in enumerate(fingerprints):
        if not isinstance(fingerprint, Mapping):
            raise DatasetAuditError(f"Problem: cleaned manifest fingerprint {index} is not an object. Likely cause: the Task 7 output was edited or incomplete. Remediation: rerun clean_dataset before the audit.")
        image_id, file_path, sha256 = fingerprint.get("source_image_id"), fingerprint.get("file_path"), fingerprint.get("sha256")
        if not isinstance(image_id, str) or not image_id or not isinstance(file_path, str) or not file_path or not isinstance(sha256, str) or not sha256:
            raise DatasetAuditError(f"Problem: cleaned manifest fingerprint {index} is incomplete. Likely cause: source image identity or SHA-256 is missing. Remediation: rerun clean_dataset before the audit.")
        key = (image_id, file_path)
        previous = output.get(key)
        if previous is not None and previous != sha256:
            raise DatasetAuditError(f"Problem: cleaned manifest assigns conflicting SHA-256 values to {image_id!r}. Likely cause: the deduplication artifact was merged incorrectly. Remediation: rerun clean_dataset and use its untouched output.")
        output[key] = sha256
    return output, True


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        annotations = _resolve_existing_file(args.annotations, "--annotations")
        output_root = _resolve_output_root(args.output_root)
        image_root = _resolve_output_root(args.image_root) if args.image_root else annotations.parent
        if args.split_manifest and args.split_output_root:
            raise DatasetAuditError("Problem: --split-manifest and --split-output-root cannot be combined. Likely cause: sealed verification must use the manifest stored beside its protected artifacts. Remediation: use --split-output-root alone, or use --split-manifest alone without sealed verification.")
        split_output_root = _resolve_existing_directory(args.split_output_root, "--split-output-root") if args.split_output_root else None
        split_manifest_path = _resolve_existing_file(split_output_root / "split_manifest.json", "Task 8 split manifest") if split_output_root else (_resolve_existing_file(args.split_manifest, "--split-manifest") if args.split_manifest else None)
        split_manifest = _load_json(split_manifest_path) if split_manifest_path else None
        if split_manifest is not None and not isinstance(split_manifest, Mapping):
            raise DatasetAuditError("Problem: split manifest is not an object. Likely cause: a JSON array was supplied. Remediation: provide the split_manifest.json emitted by create_splits.")
        raw_manifest = _load_json(annotations)
        hashes, require_hashes = _dedup_hashes(raw_manifest)
        sealed_findings = verify_sealed_label_artifacts(split_output_root, split_manifest) if split_output_root is not None and split_manifest is not None else ()
        result = audit_annotations(_load_annotations(raw_manifest), split_manifest=split_manifest, image_hashes=hashes, require_hashes=require_hashes, image_root=image_root, additional_findings=sealed_findings)
        written = write_audit_outputs(result, output_root, image_root=image_root)
    except DatasetAuditError as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"output_root": str(output_root), "critical_finding_count": result.critical_finding_count, "written_count": len(written)}, sort_keys=True))
    return 1 if result.critical_finding_count else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
