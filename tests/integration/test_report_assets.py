from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from fruit_ssod.cli.build_report_assets import build_report_assets
from fruit_ssod.reporting.final_figures import FinalFigureError


def _report_data(tmp_path: Path) -> dict[str, object]:
    montage = tmp_path / "montage.png"
    Image.new("RGB", (2, 2), "white").save(montage)
    group = {
        "complete": True,
        "metrics": {
            "map50": {"mean": 0.82, "std": 0.01, "n": 3},
            "map50_95": {"mean": 0.50, "std": 0.01, "n": 3},
            "precision": {"mean": 0.80, "std": 0.01, "n": 3},
            "recall": {"mean": 0.70, "std": 0.01, "n": 3},
            "f1": {"mean": 0.75, "std": 0.01, "n": 3},
        },
        "per_class_ap50": {str(index): {"mean": 0.8, "std": 0.01, "n": 3} for index in range(5)},
        "observed_seeds": [42, 2026, 3407],
    }
    return {
        "schema_version": "1.0",
        "protocol": "fruit_ssod_final_report_data_v1",
        "datasets": {"sample_annotation_montage": {"path": str(montage), "bytes": montage.stat().st_size, "sha256": hashlib.sha256(montage.read_bytes()).hexdigest()}},
        "methods": {"main_groups": {"supervised_20": group, "trust_main": group}},
        "metrics": {
            "rows": [
                {"run_id": "supervised_20_seed42", "method": "supervised_20", "status": "complete", "evaluation_status": "complete", "seed": 42, "label_budget_percent": 20, "split_fingerprint": "a" * 64, "run_dir": "run", "issues": [], "primary_test": {"map50": 0.70, "map50_95": 0.4, "precision": 0.7, "recall": 0.6, "f1": 0.65, "per_class_ap50": {str(index): 0.7 for index in range(5)}}},
                {"run_id": "ssod_trust_seed42", "method": "trust_main", "status": "complete", "evaluation_status": "complete", "seed": 42, "label_budget_percent": 20, "split_fingerprint": "a" * 64, "run_dir": "run", "issues": [], "primary_test": {"map50": 0.82, "map50_95": 0.5, "precision": 0.8, "recall": 0.7, "f1": 0.75, "per_class_ap50": {str(index): 0.8 for index in range(5)}}},
            ]
        },
    }


def test_report_assets_are_bounded_evidence_derived_and_immutable(tmp_path: Path) -> None:
    report_data = tmp_path / "report_data.json"
    report_data.write_text(json.dumps(_report_data(tmp_path)), encoding="utf-8")

    output = build_report_assets(report_data, tmp_path / "assets")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "fruit_ssod_final_report_assets_v1"
    assert manifest["figure_count"] == 10
    assert manifest["table_count"] == 2
    assert len(manifest["captions"]["sha256"]) == 64
    assert (output / "figures" / "method_comparison.png").is_file()
    assert (output / "figures" / "workflow.png").is_file()
    assert (output / "figures" / "pseudo_label_quality.png").is_file()
    assert (output / "figures" / "deployment.png").is_file()
    assert (output / "figures" / "dataset_annotation_examples.png").is_file()
    assert (output / "tables" / "all_runs.csv").is_file()
    assert (output / "figures" / "method_comparison.png").stat().st_size > 0
    with pytest.raises(FinalFigureError, match="already exists"):
        build_report_assets(report_data, output)
