"""Publish the immutable Task 18 result package."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from fruit_ssod.evaluation.acceptance import AcceptanceError, evaluate_acceptance
from fruit_ssod.evaluation.aggregate import ResultAggregationError, aggregate_results, canonical_json, thaw
from fruit_ssod.reporting.result_figures import write_result_figures
from fruit_ssod.reporting.result_tables import csv_text, fruitdet_rows, main_summary_rows, primary_result_rows


def _problem(problem: str, cause: str, remediation: str) -> ResultAggregationError:
    return ResultAggregationError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _file_evidence(path: Path, *, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    import hashlib
    return {"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _cell(value: object) -> str:
    if value is None:
        return "<c t=\"inlineStr\"><is><t></t></is></c>"
    return f"<c t=\"inlineStr\"><is><t>{escape(str(value))}</t></is></c>"


def _xlsx(rows_by_sheet: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> None:
    """Write a tiny standards-compliant XLSX without adding a runtime dependency."""
    sheets: list[tuple[str, list[str], Sequence[Mapping[str, Any]]]] = []
    for name, rows in rows_by_sheet.items():
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        sheets.append((name, keys, rows))
    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">', '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', '<Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for index in range(1, len(sheets) + 1):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append("</Types>")
    workbook = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    for index, (name, _, _) in enumerate(sheets, 1):
        workbook.append(f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
    workbook.append("</sheets></workbook>")
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">', '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>', "</Relationships>"]
    workbook_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for index in range(1, len(sheets) + 1):
        workbook_rels.append(f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>')
    workbook_rels.append("</Relationships>")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", "".join(rels))
        archive.writestr("xl/workbook.xml", "".join(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for index, (_, keys, rows) in enumerate(sheets, 1):
            xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
            for number, row in enumerate(([{key: key for key in keys}] + list(rows)), 1):
                xml.append(f'<row r="{number}">')
                xml.extend(_cell(row.get(key)) for key in keys)
                xml.append("</row>")
            xml.append("</sheetData></worksheet>")
            archive.writestr(f"xl/worksheets/sheet{index}.xml", "".join(xml))


def publish_result_package(aggregate: Mapping[str, Any], acceptance: Mapping[str, Any], output_dir: Path | str) -> Path:
    """Atomically publish an all-or-nothing, never-overwritten result directory."""
    # Acceptance is derived evidence, not caller-controlled package metadata.
    # Keep the argument for CLI/API compatibility, but make an attempted
    # forged pass (or stale recalculation) fail before any output is staged.
    calculated_acceptance = evaluate_acceptance(aggregate)
    try:
        acceptance_matches = canonical_json(acceptance) == canonical_json(calculated_acceptance)
    except (TypeError, ValueError) as error:
        raise _problem("supplied acceptance evidence is malformed", str(error), "pass the exact evaluate_acceptance result for this aggregate") from error
    if not acceptance_matches:
        raise _problem("supplied acceptance evidence differs from the aggregate", "acceptance must be recomputed from the immutable aggregate at publication", "do not edit acceptance.json; call evaluate_acceptance on this aggregate")
    acceptance = calculated_acceptance
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise _problem("result package already exists", str(destination), "preserve immutable evidence or choose a new output directory")
    ancestor = destination.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise _problem("result package parent is not a directory", str(ancestor), "choose a destination below a writable directory")
        ancestor = ancestor.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        tables = staging / "tables"
        _write_text(staging / "aggregate.json", canonical_json(aggregate) + "\n")
        _write_text(staging / "acceptance.json", canonical_json(acceptance) + "\n")
        primary = primary_result_rows(aggregate)
        summary = main_summary_rows(aggregate)
        fruitdet = fruitdet_rows(aggregate)
        _write_text(tables / "primary_results.csv", csv_text(primary))
        _write_text(tables / "main_group_summary.csv", csv_text(summary))
        _write_text(tables / "fruitdet_mapped_results.csv", csv_text(fruitdet))
        _xlsx({"primary_results": primary, "main_summary": summary, "fruitdet_mapped": fruitdet}, tables / "results.xlsx")
        figure_paths = write_result_figures(aggregate, staging / "figures")
        artifacts = [_file_evidence(path, root=staging) for path in sorted(staging.rglob("*")) if path.is_file()]
        manifest = {"schema_version": "1.0", "protocol": "task18_result_package_v1", "artifacts": artifacts, "acceptance_status": acceptance["status"], "figures": [path.relative_to(staging).as_posix() for path in figure_paths]}
        _write_text(staging / "manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        # Windows refuses to rename over an existing destination, which is the
        # desired immutable-publication behavior.  The complete directory is
        # staged before this single namespace operation.
        os.rename(staging, destination)
    except FileExistsError as error:
        raise _problem("result package already exists", str(destination), "preserve immutable evidence or choose a new output directory") from error
    except OSError as error:
        raise _problem("result package cannot be published", str(error), "choose a writable new destination and retry") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination


def verify_result_package(path: Path | str) -> Mapping[str, Any]:
    """Verify the byte identity of every sealed result-package artifact."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise _problem("result package root is not a real directory", str(root), "verify the original non-symlink result package directory")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("result package manifest cannot be read", str(error), "restore the immutable manifest.json") from error
    if not isinstance(manifest, Mapping) or manifest.get("protocol") != "task18_result_package_v1" or not isinstance(manifest.get("artifacts"), list):
        raise _problem("result package manifest is malformed", "protocol or artifacts list is missing", "restore the original immutable result package")
    seen: set[str] = set()
    for item in manifest["artifacts"]:
        if not isinstance(item, Mapping):
            raise _problem("result package manifest has an invalid artifact", "an artifact entry is not an object", "restore the original immutable result package")
        rel, expected_bytes, expected_sha = item.get("path"), item.get("bytes"), item.get("sha256")
        if not isinstance(rel, str) or not rel or rel == "manifest.json" or Path(rel).is_absolute() or ".." in Path(rel).parts or rel in seen:
            raise _problem("result package manifest has an unsafe artifact path", repr(rel), "restore the original immutable manifest")
        seen.add(rel)
        target = root / rel
        if target.is_symlink() or not target.is_file() or not isinstance(expected_bytes, int) or expected_bytes < 0 or not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise _problem("result package artifact is missing or malformed", rel, "restore the complete immutable package")
        actual = _file_evidence(target, root=root)
        if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha.lower():
            raise _problem("result package artifact digest differs", rel, "restore the untouched immutable package")
    actual_files: set[str] = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in subdirectories:
            if (current / name).is_symlink():
                raise _problem("result package contains a symbolic link", str((current / name).relative_to(root)), "restore the original package with ordinary files only")
        for name in filenames:
            candidate = current / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file():
                raise _problem("result package contains a non-regular file", relative, "restore the original package with ordinary files only")
            actual_files.add(relative)
    expected_files = seen | {"manifest.json"}
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise _problem("result package file set differs from its manifest", f"extra={extra!r}, missing={missing!r}", "remove no files manually; restore or republish the complete immutable package")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate immutable fruit-SSOD results and evaluate acceptance.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Run artifact directory; repeat for every submitted experiment.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New immutable output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        aggregate = aggregate_results(args.run_dir)
        acceptance = evaluate_acceptance(aggregate)
        output = publish_result_package(aggregate, acceptance, args.output_dir)
    except (ResultAggregationError, AcceptanceError, OSError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"output_dir": str(output), "acceptance": thaw(acceptance)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
