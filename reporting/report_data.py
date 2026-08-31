"""Build one immutable, evidence-bound input for the final report.

No report table may be assembled from hand-entered values.  This module joins
the verified Task 18 package, dataset audit, sealed pseudo-label audit, and
RTX 3080 benchmark only after their essential protocol bindings are checked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.cli.aggregate_results import verify_result_package
from fruit_ssod.evaluation.acceptance import evaluate_acceptance
from fruit_ssod.evaluation.aggregate import canonical_json, thaw
from fruit_ssod.evaluation.benchmark import BenchmarkConfig, BenchmarkError, BenchmarkSummary


class ReportDataError(ValueError):
    """Raised when a final report would be based on incomplete evidence."""


def _problem(problem: str, cause: str, remediation: str) -> ReportDataError:
    return ReportDataError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_symlink():
            raise OSError("symbolic links are not accepted as final evidence")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{label} cannot be read", str(error), "restore the immutable JSON artifact and retry") from error
    if not isinstance(payload, Mapping):
        raise _problem(f"{label} is not a JSON object", "the artifact was truncated or manually replaced", "regenerate it from the canonical pipeline")
    return payload


def _evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        raw = resolved.read_bytes()
    except OSError as error:
        raise _problem(f"{label} cannot be hashed", str(error), "restore the readable immutable artifact") from error
    return {"path": str(resolved), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _require(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _problem(f"{label} is missing or malformed", "the required report evidence is not an object", "regenerate the source artifact with the current pipeline")
    return value


def _completed_experiment_evidence(aggregate: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str, str]:
    if aggregate.get("protocol") != "task18_result_aggregation_v1":
        raise _problem("result aggregate protocol is unsupported", repr(aggregate.get("protocol")), "build report data from a verified Task 18 result package")
    rows = aggregate.get("rows")
    summary = _require(aggregate.get("summary"), "aggregate summary")
    groups = _require(summary.get("main_groups"), "aggregate main groups")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise _problem("result aggregate has no run rows", "completed and failed experiments are not visible", "aggregate the complete experiment queue before report generation")
    for name in ("supervised_20", "trust_main"):
        group = _require(groups.get(name), f"main group {name}")
        comparable = _require(group.get("comparability"), f"main group {name} comparability")
        if group.get("complete") is not True or comparable.get("compatible") is not True:
            raise _problem("main experiment group is incomplete or incomparable", name, "complete all required seeds using one fixed split and primary-test protocol")
    fingerprints: set[str] = set()
    dataset_digests: set[str] = set()
    complete_rows: list[Mapping[str, Any]] = []
    for row in rows:
        status, run_id = row.get("status"), row.get("run_id")
        if status in {"unreadable", "incomplete"} or not isinstance(run_id, str) or not run_id:
            raise _problem("experiment queue has a missing or unreadable run", repr(run_id), "restore the run record or record an explicit failed run before report generation")
        if status == "failed":
            if not isinstance(row.get("failure"), Mapping):
                raise _problem("failed experiment has no failure evidence", repr(run_id), "preserve a structured failure record in the result aggregate")
            continue
        if status != "complete" or row.get("evaluation_status") != "complete":
            raise _problem("experiment is not fully evaluated", repr(run_id), "complete its fixed primary-test evaluation or retain it as an explicit failed run")
        protocol = _require(row.get("primary_test_protocol"), f"primary-test protocol for {run_id}")
        fingerprint, digest = row.get("split_fingerprint"), protocol.get("dataset_yaml_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or not isinstance(digest, str) or len(digest) != 64:
            raise _problem("completed experiment lacks immutable split/test identity", repr(run_id), "regenerate the sealed evaluation evidence")
        fingerprints.add(fingerprint); dataset_digests.add(digest)
        complete_rows.append(row)
    if len(fingerprints) != 1 or len(dataset_digests) != 1:
        raise _problem("completed experiments use inconsistent primary protocols", f"split fingerprints={sorted(fingerprints)!r}, test datasets={sorted(dataset_digests)!r}", "rerun only the incompatible experiments under one fixed protocol")
    return complete_rows, next(iter(fingerprints)), next(iter(dataset_digests))


def _dataset_summary(audit: Mapping[str, Any], *, audit_path: Path) -> dict[str, Any]:
    if audit.get("critical_finding_count") != 0:
        raise _problem("dataset audit has critical findings", repr(audit.get("critical_finding_count")), "resolve every critical dataset finding and regenerate the audit")
    for name in ("class_box_counts", "source_license_summary", "label_budget_membership"):
        _require(audit.get(name), f"dataset audit {name}")
    budgets = audit["label_budget_membership"]
    budget_counts = {str(key): value.get("image_count") for key, value in budgets.items() if isinstance(value, Mapping)}
    if not budget_counts or any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in budget_counts.values()):
        raise _problem("dataset audit budget membership is incomplete", repr(budget_counts), "regenerate the Task 9 audit from sealed Task 8 split outputs")
    montage = audit_path.resolve(strict=True).with_name("sample_annotation_montage.png")
    if not montage.is_file():
        raise _problem("dataset annotation montage is missing", str(montage), "rerun the dataset audit and retain its sample_annotation_montage.png beside dataset_audit.json")
    return {
        "critical_finding_count": 0,
        "class_box_counts": thaw(audit["class_box_counts"]),
        "source_license_summary": thaw(audit["source_license_summary"]),
        "label_budget_image_counts": budget_counts,
        "sample_annotation_montage": _evidence(montage, "dataset annotation montage"),
    }


def _pseudo_summary(audit: Mapping[str, Any], *, split_fingerprint: str) -> dict[str, Any]:
    if audit.get("schema_version") != "1.0" or not isinstance(audit.get("teacher_run_id"), str) or not audit["teacher_run_id"]:
        raise _problem("pseudo-label audit identity is incomplete", "schema version or teacher run ID is absent", "regenerate the sealed pseudo-label audit")
    provenance, metrics = _require(audit.get("provenance"), "pseudo-label provenance"), _require(audit.get("metrics"), "pseudo-label metrics")
    pseudo_fingerprint = provenance.get("pseudo_audit_split_fingerprint")
    if not isinstance(pseudo_fingerprint, str) or len(pseudo_fingerprint) != 64:
        raise _problem("pseudo-label audit has no sealed split fingerprint", repr(pseudo_fingerprint), "regenerate the audit from a sealed pseudo-audit split")
    # Pseudo-audit is a distinct protected partition, so it need not equal the
    # primary split-protocol digest; retain both identities explicitly.
    after = _require(metrics.get("after_filter"), "post-filter pseudo metrics")
    overall = _require(after.get("overall"), "post-filter pseudo overall metrics")
    precision = overall.get("precision")
    if isinstance(precision, bool) or not isinstance(precision, (int, float)) or not math.isfinite(float(precision)) or not 0 <= float(precision) <= 1:
        raise _problem("post-filter pseudo precision is invalid", repr(precision), "regenerate pseudo-label metrics from sealed audit boxes")
    refresh = _require(audit.get("pseudo_refresh"), "pseudo refresh gate")
    if not isinstance(refresh.get("allowed"), bool) or not isinstance(refresh.get("reason"), str):
        raise _problem("pseudo refresh gate is malformed", repr(refresh), "regenerate the pseudo-label audit with the current gate")
    return {"teacher_run_id": audit["teacher_run_id"], "primary_split_fingerprint": split_fingerprint, "pseudo_audit_split_fingerprint": pseudo_fingerprint, "metrics": thaw(metrics), "pseudo_refresh": thaw(refresh), "filter_policy": thaw(audit.get("filter_policy"))}


def _benchmark_summary(payload: Mapping[str, Any], *, expected_checkpoint_sha256: str) -> dict[str, Any]:
    try:
        summary = BenchmarkSummary(
            config=BenchmarkConfig(**_require(payload.get("config"), "benchmark config")),
            latency_ms=_require(payload.get("latency_ms"), "benchmark latency"), fps=payload.get("fps"),
            peak_allocated_bytes=payload.get("peak_allocated_bytes"), model=_require(payload.get("model"), "benchmark model"),
            environment=_require(payload.get("environment"), "benchmark environment"),
            schema_version=payload.get("schema_version", ""), protocol=payload.get("protocol", ""),
        )
    except (BenchmarkError, TypeError, ValueError) as error:
        raise _problem("deployment benchmark is invalid", str(error), "rerun the sealed RTX 3080 benchmark after all training jobs finish") from error
    mapped = summary.mapping()
    if mapped["model"]["weights_sha256"] != expected_checkpoint_sha256:
        raise _problem("benchmark checkpoint differs from the designated final model", f"benchmark={mapped['model']['weights_sha256']}, final={expected_checkpoint_sha256}", "benchmark the exact checkpoint used for the designated final fixed-test result")
    return mapped


def build_report_data(*, result_package: Path, dataset_audit: Path, pseudo_audit: Path, benchmark: Path, output: Path) -> Path:
    """Publish one non-overwriting report-data JSON after full evidence checks."""
    try:
        verify_result_package(result_package)
    except ValueError as error:
        raise _problem("result package verification failed", str(error), "restore the immutable Task 18 package before building report data") from error
    aggregate = _load_json(result_package / "aggregate.json", "result aggregate")
    acceptance = _load_json(result_package / "acceptance.json", "acceptance evidence")
    expected_acceptance = thaw(evaluate_acceptance(aggregate))
    if canonical_json(acceptance) != canonical_json(expected_acceptance):
        raise _problem("acceptance evidence differs from the aggregate", "the report package was manually changed or stale", "republish Task 18 results and rerun report-data construction")
    complete_rows, split_fingerprint, _ = _completed_experiment_evidence(aggregate)
    final_rows = [row for row in complete_rows if row.get("method") == "trust_main" and row.get("seed") == 42]
    if len(final_rows) != 1:
        raise _problem("designated final trust run is unavailable", repr([row.get("run_id") for row in final_rows]), "complete exactly one trust_main seed-42 fixed-test run")
    final_protocol = _require(final_rows[0].get("primary_test_protocol"), "designated final trust protocol")
    checkpoint = final_protocol.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise _problem("designated final trust checkpoint is missing", repr(checkpoint), "restore its sealed primary-test evaluation evidence")
    audit_payload, pseudo_payload, benchmark_payload = _load_json(dataset_audit, "dataset audit"), _load_json(pseudo_audit, "pseudo-label audit"), _load_json(benchmark, "deployment benchmark")
    payload = {
        "schema_version": "1.0", "protocol": "fruit_ssod_final_report_data_v1",
        "datasets": _dataset_summary(audit_payload, audit_path=dataset_audit),
        "methods": thaw(aggregate["summary"]),
        "metrics": {"rows": thaw(aggregate["rows"]), "designated_final_run_id": final_rows[0]["run_id"]},
        "pseudo_label_quality": _pseudo_summary(pseudo_payload, split_fingerprint=split_fingerprint),
        "deployment": _benchmark_summary(benchmark_payload, expected_checkpoint_sha256=checkpoint),
        "acceptance": expected_acceptance,
        "provenance": {"result_package": str(result_package.resolve()), "aggregate": _evidence(result_package / "aggregate.json", "result aggregate"), "acceptance": _evidence(result_package / "acceptance.json", "acceptance evidence"), "dataset_audit": _evidence(dataset_audit, "dataset audit"), "pseudo_audit": _evidence(pseudo_audit, "pseudo-label audit"), "benchmark": _evidence(benchmark, "deployment benchmark")},
    }
    destination = output.resolve(strict=False)
    if destination.exists():
        raise _problem("report-data output already exists", str(destination), "preserve the immutable report_data.json or choose a fresh output path")
    temporary: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(canonical_json(payload) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, destination)
        return destination
    except FileExistsError as error:
        raise _problem("report-data output already exists", str(destination), "preserve existing evidence or choose a fresh output path") from error
    except OSError as error:
        raise _problem("report-data output cannot be published", str(error), "choose a writable fresh output path") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
