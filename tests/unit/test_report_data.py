from __future__ import annotations

import json
from pathlib import Path

import pytest

import fruit_ssod.reporting.report_data as report_data
from fruit_ssod.evaluation.acceptance import evaluate_acceptance
from fruit_ssod.evaluation.aggregate import thaw
from fruit_ssod.evaluation.benchmark import BenchmarkConfig, file_evidence, summarize_timings


_FINGERPRINT = "a" * 64
_DATASET = "b" * 64
_CHECKPOINT = "c" * 64


def _environment() -> dict[str, object]:
    return {"gpu_name": "NVIDIA GeForce RTX 3080", "nvidia_smi_gpu_name": "NVIDIA GeForce RTX 3080", "gpu_index": 0, "cuda_logical_index": 0, "nvidia_smi_physical_index": 0, "gpu_uuid": "GPU-3080", "torch_gpu_uuid": "GPU-3080", "driver_version": "555.85", "torch_version": "2.5.1+cu121", "cuda_runtime": "12.1", "ultralytics_version": "8.4.0", "benchmark_process_id": 1234, "active_compute_pids": [], "foreign_compute_pids": [], "gpu_isolation_policy": "no_foreign_compute_processes"}


def _aggregate() -> dict[str, object]:
    def group() -> dict[str, object]:
        return {"complete": True, "comparability": {"compatible": True}, "metrics": {"map50": {"mean": 0.82}}}
    rows = []
    for method, prefix in (("supervised_20", "supervised"), ("trust_main", "ssod_trust")):
        for seed in (42, 3407, 2026):
            rows.append({"run_id": f"{prefix}_{seed}", "status": "complete", "evaluation_status": "complete", "method": method, "seed": seed, "split_fingerprint": _FINGERPRINT, "primary_test_protocol": {"dataset_yaml_sha256": _DATASET, "checkpoint_sha256": _CHECKPOINT}})
    return {"schema_version": "1.0", "protocol": "task18_result_aggregation_v1", "rows": rows, "summary": {"main_groups": {"supervised_20": group(), "trust_main": group()}}}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    package = tmp_path / "package"; package.mkdir()
    aggregate = _aggregate()
    (package / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    (package / "acceptance.json").write_text(json.dumps(thaw(evaluate_acceptance(aggregate))), encoding="utf-8")
    dataset = tmp_path / "dataset_audit.json"
    dataset.write_text(json.dumps({"critical_finding_count": 0, "class_box_counts": {"train_pool": {"source": {"0": 1}}}, "source_license_summary": {"source": {"license": "CC BY"}}, "label_budget_membership": {"20": {"image_count": 5}}}), encoding="utf-8")
    (tmp_path / "sample_annotation_montage.png").write_bytes(b"fixture montage")
    pseudo = tmp_path / "pseudo_audit.json"
    pseudo.write_text(json.dumps({"schema_version": "1.0", "teacher_run_id": "supervised_20_seed42", "provenance": {"pseudo_audit_split_fingerprint": "d" * 64}, "metrics": {"after_filter": {"overall": {"precision": 0.91}}}, "pseudo_refresh": {"allowed": True, "reason": "precision_at_or_above_threshold"}}), encoding="utf-8")
    model = tmp_path / "best.pt"; model.write_bytes(b"checkpoint")
    evidence = file_evidence(model); evidence["weights_sha256"] = _CHECKPOINT
    benchmark = tmp_path / "benchmark.json"
    summary = summarize_timings(durations_seconds=(0.01,), config=BenchmarkConfig(measured_iterations=1), peak_allocated_bytes=0, model=evidence, environment=_environment())
    benchmark.write_text(json.dumps(summary.mapping()), encoding="utf-8")
    return package, dataset, pseudo, benchmark


def test_build_report_data_requires_verified_complete_evidence_and_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, dataset, pseudo, benchmark = _write_inputs(tmp_path)
    monkeypatch.setattr(report_data, "verify_result_package", lambda _: {"protocol": "task18_result_package_v1"})
    output = report_data.build_report_data(result_package=package, dataset_audit=dataset, pseudo_audit=pseudo, benchmark=benchmark, output=tmp_path / "report_data.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"] == "fruit_ssod_final_report_data_v1"
    assert payload["metrics"]["designated_final_run_id"] == "ssod_trust_42"
    assert payload["deployment"]["model"]["weights_sha256"] == _CHECKPOINT
    with pytest.raises(report_data.ReportDataError, match="already exists"):
        report_data.build_report_data(result_package=package, dataset_audit=dataset, pseudo_audit=pseudo, benchmark=benchmark, output=output)


def test_build_report_data_rejects_incompatible_primary_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, dataset, pseudo, benchmark = _write_inputs(tmp_path)
    aggregate_path = package / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["rows"][0]["split_fingerprint"] = "e" * 64
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(report_data, "verify_result_package", lambda _: {"protocol": "task18_result_package_v1"})
    with pytest.raises(report_data.ReportDataError, match="inconsistent primary protocols"):
        report_data.build_report_data(result_package=package, dataset_audit=dataset, pseudo_audit=pseudo, benchmark=benchmark, output=tmp_path / "report_data.json")
