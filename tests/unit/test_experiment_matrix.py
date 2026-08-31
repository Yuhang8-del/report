"""Task 17 controlled SSOD matrix contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fruit_ssod.cli.validate_ssod_matrix import main as matrix_cli
from fruit_ssod.evaluation.detection_metrics import DetectionMetrics
from fruit_ssod.training.run_record import complete_run_record, create_run_record, write_run_record
from fruit_ssod.training.ssod_matrix import SsodMatrixError, matrix_entries, matrix_queue, validate_ssod_matrix
from fruit_ssod.training.student_dataset import compose_student_dataset
from tests.unit.test_student_dataset import _inputs


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"


def _metrics() -> dict[str, object]:
    return DetectionMetrics(0.8, 0.6, 0.8, 0.7, 0.7467, {item: 0.8 for item in range(5)}).mapping()


def _copy_matrix(tmp_path: Path) -> Path:
    configs = tmp_path / "configs" / "experiments"; configs.mkdir(parents=True)
    models = tmp_path / "configs" / "models"; models.mkdir(parents=True)
    (models / "yolov8s_640.yaml").write_text((ROOT / "configs" / "models" / "yolov8s_640.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    for entry in matrix_entries():
        (configs / entry.filename).write_text((CONFIGS / entry.filename).read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("supervised_20_seed42.yaml", "supervised_20_seed3407.yaml", "supervised_20_seed2026.yaml"):
        (configs / name).write_text((CONFIGS / name).read_text(encoding="utf-8"), encoding="utf-8")
    return configs


def test_matrix_contains_one_global_three_trust_and_four_named_ablations() -> None:
    entries = matrix_entries()
    assert [entry.experiment_name for entry in entries] == [
        "ssod_global_seed42", "ssod_trust_seed42", "ssod_trust_seed3407", "ssod_trust_seed2026",
        "ablation_no_class_threshold", "ablation_no_view_consistency", "ablation_no_size_filter", "ablation_no_human_resampling",
    ]
    assert sum(entry.role == "global_baseline" for entry in entries) == 1
    assert [entry.seed for entry in entries if entry.role == "trust_main"] == [42, 3407, 2026]
    assert sum(entry.role.startswith("ablation_") for entry in entries) == 4
    assert validate_ssod_matrix(CONFIGS) == tuple(CONFIGS / entry.filename for entry in entries)


def test_matrix_controls_model_split_evaluation_and_initialization_protocol() -> None:
    configs = [yaml.safe_load((CONFIGS / entry.filename).read_text(encoding="utf-8")) for entry in matrix_entries()]
    for field in ("model_config", "pretrained_weights", "split_manifest", "human_images", "human_labels", "validation_labels", "unlabeled_manifest", "evaluation_protocol", "image_size", "amp", "batch", "epochs"):
        assert {json.dumps(item[field], sort_keys=True) if isinstance(item[field], dict) else item[field] for item in configs} == {json.dumps(configs[0][field], sort_keys=True) if isinstance(configs[0][field], dict) else configs[0][field]}
    assert {tuple(item["initialization_policy"][key] for key in ("policy_id", "model_initialization", "comparison_group")) for item in configs} == {("ssod_student_init_v1", "shared_pretrained_weights", "ssod_main_seed42")}


def test_matrix_rejects_an_uncontrolled_model_or_evaluation_change(tmp_path: Path) -> None:
    copied = _copy_matrix(tmp_path)
    target = copied / "ssod_trust_seed3407.yaml"
    target.write_text(target.read_text(encoding="utf-8").replace("image_size: 640", "image_size: 512"), encoding="utf-8")
    with pytest.raises(SsodMatrixError, match="not controlled"):
        validate_ssod_matrix(copied)
    target.write_text(target.read_text(encoding="utf-8").replace("image_size: 512", "image_size: 640").replace("split: test", "split: val"), encoding="utf-8")
    with pytest.raises(SsodMatrixError, match="evaluation protocol"):
        validate_ssod_matrix(copied)


def test_each_ablation_changes_only_its_declared_factor() -> None:
    trust = yaml.safe_load((CONFIGS / "ssod_trust_seed42.yaml").read_text(encoding="utf-8"))
    fields = ("use_per_class_thresholds", "require_view_consistency", "require_size_filter")
    expected = {
        "ablation_no_class_threshold": ({"use_per_class_thresholds"}, "balanced_50_50"),
        "ablation_no_view_consistency": ({"require_view_consistency"}, "balanced_50_50"),
        "ablation_no_size_filter": ({"require_size_filter"}, "balanced_50_50"),
        "ablation_no_human_resampling": (set(), "natural_unresampled"),
    }
    for name, (changed_filter, strategy) in expected.items():
        config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        actual = {field for field in fields if config["pseudo_filter"][field] != trust["pseudo_filter"][field]}
        assert actual == changed_filter
        assert config["sampling_strategy"] == strategy


def test_natural_unresampled_ablation_keeps_each_training_source_once(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = compose_student_dataset(replace(inputs, sampling_strategy="natural_unresampled"), tmp_path / "natural")
    plan = json.loads(result.sampling_plan.read_text(encoding="utf-8"))
    assert plan["sampling_strategy"] == "natural_unresampled"
    assert plan["human_occurrences"] == 1 and plan["pseudo_occurrences"] == 1


def test_resume_skips_only_complete_record_with_current_config_and_split_fingerprints(tmp_path: Path, monkeypatch) -> None:
    configs = _copy_matrix(tmp_path)
    data_root = tmp_path / "data"; split = data_root / "fruit_ssod" / "manifests" / "ssod_unlabeled_pool_fruits360_v2" / "split_manifest.json"
    split.parent.mkdir(parents=True); split.write_text(json.dumps({"fingerprints": {"split_protocol": "a" * 64}}), encoding="utf-8")
    monkeypatch.setenv("FRUIT_SSOD_DATA_ROOT", str(data_root)); monkeypatch.setenv("FRUIT_SSOD_ARTIFACT_ROOT", str(tmp_path / "artifacts")); monkeypatch.setenv("FRUIT_SSOD_PRETRAINED_WEIGHTS", str(tmp_path / "weights.pt"))
    entry = matrix_entries()[0]; config = configs / entry.filename
    pseudo = tmp_path / "artifacts" / "pseudo" / entry.experiment_name
    artifacts = {
        # Seed-42 global and ablation policies reuse the same immutable
        # Teacher candidate envelope; only their filter/audit outputs differ.
        "candidate_sha256": tmp_path / "artifacts" / "pseudo" / "ssod_trust_seed42" / "candidates.json",
        "filter_audit_sha256": pseudo / "filter" / "audit.jsonl",
        "filter_decision_manifest_sha256": pseudo / "filter" / "decision_manifest.json",
            "pseudo_audit_report_sha256": pseudo / "audit_report" / "pseudo_audit.json",
    }
    for index, artifact in enumerate(artifacts.values()):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"artifact-{index}\n", encoding="utf-8")
    provenance = {name: hashlib.sha256(artifact.read_bytes()).hexdigest() for name, artifact in artifacts.items()}
    run_dir = tmp_path / "artifacts" / "runs" / entry.experiment_name; run_dir.mkdir(parents=True)
    snapshot = {"experiment_name": entry.experiment_name, "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "student_dataset": {"provenance": provenance}}
    running = create_run_record(config_snapshot=snapshot, split_fingerprint="a" * 64, command=("python", "-m", "fruit_ssod.cli.train_student"), environment={"python": "3.10"}, run_id=entry.experiment_name)
    write_run_record(complete_run_record(running, _metrics()), run_dir / "run_record.json")
    queue = matrix_queue(configs, artifact_root=tmp_path / "artifacts", resume=True)
    assert queue[0]["action"] == "skip", queue[0]
    config.write_text(config.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    queue = matrix_queue(configs, artifact_root=tmp_path / "artifacts", resume=True)
    assert queue[0]["action"] == "run" and "configuration fingerprint" in str(queue[0]["reason"])


def test_matrix_rejects_source_root_or_calibration_change_outside_named_ablation(tmp_path: Path) -> None:
    copied = _copy_matrix(tmp_path)
    target = copied / "ssod_trust_seed3407.yaml"
    target.write_text(target.read_text(encoding="utf-8").replace("source_root: ${FRUIT_SSOD_DATA_ROOT}/fruit_ssod/processed/ssod_unlabeled_pool_fruits360_v2/image_root_materialized_v2", "source_root: ${FRUIT_SSOD_DATA_ROOT}/another_root"), encoding="utf-8")
    with pytest.raises(SsodMatrixError, match="not controlled"):
        validate_ssod_matrix(copied)
    target.write_text(target.read_text(encoding="utf-8").replace("global_confidence: 0.50", "global_confidence: 0.51"), encoding="utf-8")
    with pytest.raises(SsodMatrixError, match="not controlled"):
        validate_ssod_matrix(copied)


def test_launcher_prints_full_queue_before_train_student_and_uses_conda_vectors() -> None:
    script = (ROOT / "scripts" / "run_ssod_matrix.ps1").read_text(encoding="utf-8")
    assert "fruit_ssod.cli.validate_ssod_matrix" in script and "Full SSOD queue" in script
    assert "--verify-preparation" in script
    assert script.index("Full SSOD queue") < script.index("fruit_ssod.cli.train_student")
    assert "if ($DryRun)" in script and "$Arguments += '--dry-run'" in script
    assert "$Arguments += @('--run-id', $Entry.experiment_name, '--device', $Device)" in script
    assert "& $CondaCommand @Arguments" in script and "Invoke-Expression" not in script


def test_matrix_validation_cli_reports_all_entries() -> None:
    assert matrix_cli(["--config-directory", str(CONFIGS), "--queue"]) == 0
