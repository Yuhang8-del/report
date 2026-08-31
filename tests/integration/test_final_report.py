from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from fruit_ssod.cli.build_report_assets import build_report_assets


_MODULE_PATH = Path(__file__).parents[2] / "reports" / "final_report" / "build_report.py"
_SPEC = importlib.util.spec_from_file_location("fruit_ssod_final_report", _MODULE_PATH)
assert _SPEC and _SPEC.loader
final_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(final_report)


def _report_data() -> dict[str, object]:
    group = {"complete": True, "metrics": {"map50": {"mean": 0.82, "std": 0.01}}}
    return {
        "protocol": "fruit_ssod_final_report_data_v1",
        "datasets": {"label_budget_image_counts": {"20": 100}},
        "methods": {"main_groups": {"supervised_20": group, "trust_main": group}},
        "metrics": {"designated_final_run_id": "trust_42", "rows": [{"run_id": "trust_42", "method": "trust_main", "evaluation_status": "complete"}]},
        "pseudo_label_quality": {"metrics": {"after_filter": {"overall": {"precision": 0.9}}}},
        "deployment": {"fps": 25.0},
        "acceptance": {},
        "provenance": {},
    }


def _assets(tmp_path: Path, report_data: Path) -> Path:
    root = tmp_path / "assets"; root.mkdir()
    figure = root / "figures" / "result.png"; figure.parent.mkdir(); Image.new("RGB", (2, 2), "white").save(figure)
    captions = root / "figures" / "captions.json"; captions.write_text(json.dumps({"result.png": "Figure: fixture."}), encoding="utf-8")
    digest = __import__("hashlib").sha256(figure.read_bytes()).hexdigest()
    manifest = {"protocol": "fruit_ssod_final_report_assets_v1", "figure_count": 1, "table_count": 0, "report_data": {"sha256": __import__("hashlib").sha256(report_data.read_bytes()).hexdigest()}, "figures": [{"relative_path": "figures/result.png", "sha256": digest}], "tables": [], "captions": {"relative_path": "figures/captions.json", "sha256": __import__("hashlib").sha256(captions.read_bytes()).hexdigest()}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _full_report_data(tmp_path: Path) -> dict[str, object]:
    montage = tmp_path / "audit_montage.png"
    Image.new("RGB", (24, 24), "white").save(montage)
    metric_values = {
        "map50": (0.82, 0.01), "map50_95": (0.50, 0.01), "precision": (0.80, 0.01),
        "recall": (0.70, 0.01), "f1": (0.75, 0.01),
    }
    group = {
        "complete": True,
        "metrics": {name: {"mean": value, "std": spread, "n": 3} for name, (value, spread) in metric_values.items()},
        "per_class_ap50": {str(index): {"mean": 0.8, "std": 0.01, "n": 3} for index in range(5)},
        "observed_seeds": [42, 2026, 3407],
    }
    primary = {name: value for name, (value, _) in metric_values.items()} | {"per_class_ap50": {str(index): 0.8 for index in range(5)}}
    rows = [
        {"run_id": "supervised_20_seed42", "method": "supervised_20", "status": "complete", "evaluation_status": "complete", "seed": 42, "label_budget_percent": 20, "split_fingerprint": "a" * 64, "run_dir": "run", "issues": [], "primary_test": primary},
        {"run_id": "trust_42", "method": "trust_main", "status": "complete", "evaluation_status": "complete", "seed": 42, "label_budget_percent": 20, "split_fingerprint": "a" * 64, "run_dir": "run", "issues": [], "primary_test": primary},
    ]
    return {
        "schema_version": "1.0", "protocol": "fruit_ssod_final_report_data_v1",
        "datasets": {"label_budget_image_counts": {"20": 100}, "source_license_summary": {"fixture": "CC BY"}, "sample_annotation_montage": {"path": str(montage), "bytes": montage.stat().st_size, "sha256": hashlib.sha256(montage.read_bytes()).hexdigest()}},
        "methods": {"main_groups": {"supervised_20": group, "trust_main": group}},
        "metrics": {"designated_final_run_id": "trust_42", "rows": rows},
        "pseudo_label_quality": {"teacher_run_id": "supervised_20_seed42", "metrics": {"after_filter": {"overall": {"precision": 0.9}}}, "pseudo_refresh": {"allowed": True, "reason": "fixture"}},
        "deployment": {"fps": 25.0, "peak_allocated_mib": 100.0, "latency_ms": {"mean": 40.0}, "environment": {"gpu_name": "NVIDIA GeForce RTX 3080", "cuda_runtime": "12.1", "torch_version": "2.5.1"}},
        "acceptance": {}, "provenance": {},
    }


def test_final_report_preflight_requires_completed_matching_evidence(tmp_path: Path) -> None:
    report_data = tmp_path / "report_data.json"; report_data.write_text(json.dumps(_report_data()), encoding="utf-8")
    assets = _assets(tmp_path, report_data)
    assert final_report.validate_final_report_inputs(report_data, assets)["protocol"] == "fruit_ssod_final_report_data_v1"
    data = _report_data(); data["methods"]["main_groups"]["trust_main"]["complete"] = False
    invalid = tmp_path / "invalid.json"; invalid.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(final_report.FinalReportError, match="three-seed"):
        final_report.validate_final_report_inputs(invalid, assets)


def test_final_report_docx_is_generated_only_after_preflight(tmp_path: Path) -> None:
    report_data = tmp_path / "report_data.json"; report_data.write_text(json.dumps(_report_data()), encoding="utf-8")
    output = final_report.build_final_report(report_data, _assets(tmp_path, report_data), tmp_path / "Final_Report.docx")
    assert output.read_bytes().startswith(b"PK")
    with pytest.raises(final_report.FinalReportError, match="already exists"):
        final_report.build_final_report(report_data, output.parent / "assets", output)


def test_full_asset_package_is_embedded_in_the_report_docx(tmp_path: Path) -> None:
    report_data = tmp_path / "report_data.json"
    report_data.write_text(json.dumps(_full_report_data(tmp_path)), encoding="utf-8")
    assets = build_report_assets(report_data, tmp_path / "assets")
    output = final_report.build_final_report(report_data, assets, tmp_path / "Final_Report.docx")
    with zipfile.ZipFile(output) as document:
        media = [name for name in document.namelist() if name.startswith("word/media/")]
    assert len(media) == 10
