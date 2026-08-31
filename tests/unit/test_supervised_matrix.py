"""Task 12 deterministic reference-matrix and evidence-aggregation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fruit_ssod.evaluation.detection_metrics import DetectionMetrics
from fruit_ssod.training.run_record import complete_run_record, create_run_record, fail_run_record, write_run_record
from fruit_ssod.training.supervised_matrix import (
    SupervisedMatrixError,
    aggregate_supervised_matrix,
    conda_train_command,
    matrix_entries,
    render_reference_matrix,
    validate_reference_configs,
    write_supervised_matrix_aggregate,
)
from fruit_ssod.training.supervised import load_supervised_experiment
from fruit_ssod.cli.validate_supervised_matrix import main as validate_matrix_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs" / "experiments"
TEMPLATE = CONFIG_DIRECTORY / "supervised_reference_template.yaml"


def _metrics(map50: float = 0.9) -> dict[str, object]:
    return DetectionMetrics(map50, 0.6, 0.8, 0.7, 0.7467, {index: map50 for index in range(5)}).mapping()


def _test_evidence(*, run_id: str, metrics: dict[str, object]) -> dict[str, object]:
    """Fixture implementation of the public evaluator's evidence envelope."""
    content = json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": "1.0",
        "metrics": metrics,
        "protocol": {
            "schema": "fruit_ssod_evaluation_evidence_v1",
            "run_id": run_id,
            "checkpoint_sha256": "d" * 64,
            "dataset_yaml_sha256": "c" * 64,
            "split": "test",
            "metrics_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "canonical_classes": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"],
        },
    }


def _record(
    directory: Path, *, budget: int, seed: int, status: str = "complete", test_map50: float | None = 0.9,
    run_id: str | None = None,
) -> Path:
    name = f"supervised_{budget}_seed{seed}"
    snapshot = {
        "experiment_name": name, "label_budget_percent": budget, "seed": seed,
        "matrix_template_id": "supervised_reference_v1", "model_config_sha256": "b" * 64,
        "dataset_yaml_sha256": "c" * 64, "model_config_effective": {"model": "yolov8s.yaml", "names": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]},
        "dataset_yaml_effective": {"names": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]},
        "split_manifest": "fixture/split_manifest.json", "canonical_classes": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"],
    }
    running = create_run_record(config_snapshot=snapshot, split_fingerprint="a" * 64, command=("python", "-m", "fruit_ssod.cli.train_supervised"), environment={"python": "3.10"}, run_id=run_id or name)
    if status == "complete":
        record = complete_run_record(running, _metrics())
    elif status == "failed":
        record = fail_run_record(running, problem="fixture failed", cause="fixture", remediation="fix fixture")
    else:
        record = running
    directory.mkdir(parents=True)
    write_run_record(record, directory / "run_record.json")
    (directory / "checkpoint_evidence.json").write_text(json.dumps({"best.pt": {"sha256": "d" * 64}}), encoding="utf-8")
    if test_map50 is not None:
        test = directory / "evaluations"
        test.mkdir()
        metrics = _metrics(test_map50)
        (test / "test.json").write_text(json.dumps(_test_evidence(run_id=record.run_id, metrics=metrics)), encoding="utf-8")
    return directory


def test_reference_configs_are_deterministic_renders_of_one_template() -> None:
    expected_names = (
        "supervised_10_seed42.yaml", "supervised_20_seed42.yaml", "supervised_20_seed3407.yaml",
        "supervised_20_seed2026.yaml", "supervised_40_seed42.yaml", "supervised_100_seed42.yaml",
    )
    assert tuple(entry.filename for entry in matrix_entries()) == expected_names
    assert tuple(render_reference_matrix(TEMPLATE)) == expected_names
    assert validate_reference_configs(TEMPLATE, CONFIG_DIRECTORY) == tuple(CONFIG_DIRECTORY / name for name in expected_names)


def test_reference_config_drift_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "configs"
    copied.mkdir()
    for source in CONFIG_DIRECTORY.glob("supervised_*_seed*.yaml"):
        (copied / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    altered = copied / "supervised_40_seed42.yaml"
    altered.write_text(altered.read_text(encoding="utf-8").replace("epochs: 100", "epochs: 1"), encoding="utf-8")
    with pytest.raises(SupervisedMatrixError, match="diverges"):
        validate_reference_configs(TEMPLATE, copied)
    with pytest.raises(SystemExit) as status:
        validate_matrix_main(["--template", str(TEMPLATE), "--config-directory", str(copied)])
    assert status.value.code == 2


def test_conda_dry_run_command_is_argument_safe_and_complete() -> None:
    command = conda_train_command(Path("E:/a path/supervised_20_seed42.yaml"), conda_executable="E:/Conda Space/conda.exe", environment_name="fruit-ssod", dry_run=True)
    config_path = str(Path("E:/a path/supervised_20_seed42.yaml"))
    assert command == (
        "E:/Conda Space/conda.exe", "run", "--no-capture-output", "--name", "fruit-ssod", "python", "-m",
        "fruit_ssod.cli.train_supervised", "--config", config_path, "--dry-run",
    )
    assert conda_train_command("config.yaml", run_id="fixed-run")[-2:] == ("--run-id", "fixed-run")


def test_powershell_launcher_constructs_argument_vector_for_dry_run() -> None:
    script = (REPOSITORY_ROOT / "scripts" / "run_supervised_matrix.ps1").read_text(encoding="utf-8")
    assert "$Arguments = @(" in script
    assert "if ($DryRun)" in script
    assert "$Arguments += '--dry-run'" in script
    assert "$Arguments += @('--run-id', $ExperimentName)" in script
    assert "if ($DryRun)" in script
    assert "& $CondaCommand @Arguments" in script
    assert "fruit_ssod.cli.validate_supervised_matrix" in script
    assert "& $CondaCommand @ValidationArguments" in script
    assert "Invoke-Expression" not in script


def test_matrix_budget_and_template_are_frozen_in_the_training_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model.yaml"
    model.write_text("model: yolov8s.yaml\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: .\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    split = tmp_path / "split_manifest.json"
    split.write_text('{"fingerprints": {"split_protocol": "' + "a" * 64 + '"}}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join((
            "template_id: supervised_reference_v1", "experiment_name: supervised_20_seed42",
            f"model_config: '{model.as_posix()}'", f"dataset_yaml: '{dataset.as_posix()}'",
            f"split_manifest: '{split.as_posix()}'", f"artifact_root: '{(tmp_path / 'artifacts').as_posix()}'",
            "seed: 42", "label_budget_percent: 20",
        )) + "\n",
        encoding="utf-8",
    )
    snapshot = load_supervised_experiment(config).snapshot()
    assert snapshot["label_budget_percent"] == 20
    assert snapshot["matrix_template_id"] == "supervised_reference_v1"


def test_disposable_dry_run_does_not_claim_the_real_matrix_run_id() -> None:
    entry = matrix_entries()[1]
    dry_command = conda_train_command(entry.filename, dry_run=True)
    real_command = conda_train_command(entry.filename, run_id=entry.experiment_name)
    assert "--run-id" not in dry_command
    assert real_command[-2:] == ("--run-id", "supervised_20_seed42")


def test_aggregation_keeps_failed_rows_and_applies_100_percent_gate(tmp_path: Path) -> None:
    complete = _record(tmp_path / "complete", budget=20, seed=42, test_map50=0.88)
    failed = _record(tmp_path / "failed", budget=20, seed=3407, status="failed", test_map50=None)
    upper = _record(tmp_path / "upper", budget=100, seed=42, test_map50=0.84)

    aggregate = aggregate_supervised_matrix((complete, failed, upper))
    assert aggregate["summary"] == {"submitted_runs": 3, "complete_runs": 2, "failed_runs": 1, "noncomplete_runs": 1}
    by_id = {row["run_id"]: row for row in aggregate["rows"]}
    assert by_id["supervised_20_seed3407"]["status"] == "failed"
    assert by_id["supervised_20_seed3407"]["failure"]["problem"] == "fixture failed"
    assert by_id["supervised_20_seed42"]["validation"]["map50"] == 0.9
    assert by_id["supervised_20_seed42"]["fixed_test"]["map50"] == 0.88
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] == 0.84
    assert aggregate["upper_bound_gate"]["data_quality_investigation_required"] is True


def test_aggregation_reports_missing_100_percent_fixed_test_without_fabrication(tmp_path: Path) -> None:
    upper = _record(tmp_path / "upper", budget=100, seed=42, test_map50=None)
    aggregate = aggregate_supervised_matrix((upper,))
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] is None
    assert aggregate["upper_bound_gate"]["status"] == "missing_fixed_test_evidence"
    assert aggregate["rows"][0]["fixed_test"] is None
    assert "fixed-test evaluation is missing" in aggregate["rows"][0]["issues"]


def test_noncanonical_high_score_cannot_mask_canonical_100_percent_gate(tmp_path: Path) -> None:
    canonical = _record(tmp_path / "canonical", budget=100, seed=42, test_map50=0.84)
    fake = _record(tmp_path / "fake", budget=100, seed=42, test_map50=0.99, run_id="dry-run-uuid")
    aggregate = aggregate_supervised_matrix((fake, canonical))
    by_id = {row["run_id"]: row for row in aggregate["rows"]}
    assert by_id["dry-run-uuid"]["canonical_protocol"] is False
    assert "run_id does not equal" in by_id["dry-run-uuid"]["issues"][0]
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] == 0.84
    assert aggregate["upper_bound_gate"]["data_quality_investigation_required"] is True


def test_fake_class_contract_high_score_cannot_mask_canonical_100_percent_gate(tmp_path: Path) -> None:
    canonical = _record(tmp_path / "canonical", budget=100, seed=42, test_map50=0.84)
    fake = _record(tmp_path / "fake", budget=100, seed=42, test_map50=0.99, run_id="invalid-class-contract")
    fake_record = json.loads((fake / "run_record.json").read_text(encoding="utf-8"))
    fake_record["config_snapshot"]["canonical_classes"] = ["fake"]
    # Rebuild a valid immutable record rather than bypassing the record reader:
    # all generic run-record invariants hold, but Task 12's stronger protocol
    # gate must reject this scientifically incompatible class contract.
    from fruit_ssod.training.run_record import read_run_record
    from dataclasses import replace
    original = read_run_record(fake / "run_record.json")
    changed = replace(original, config_snapshot=fake_record["config_snapshot"])
    (fake / "run_record.json").unlink()
    write_run_record(changed, fake / "run_record.json")
    aggregate = aggregate_supervised_matrix((fake, canonical))
    by_id = {row["run_dir"]: row for row in aggregate["rows"]}
    fake_row = by_id[str(fake.resolve())]
    assert fake_row["canonical_protocol"] is False
    assert any("canonical_classes" in issue for issue in fake_row["issues"])
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] == 0.84


def test_duplicate_canonical_matrix_identity_is_protocol_error_and_cannot_gate(tmp_path: Path) -> None:
    """Two sources claiming one fixed run make the upper-bound evidence invalid."""
    low = _record(tmp_path / "low", budget=100, seed=42, test_map50=0.2)
    duplicate = _record(tmp_path / "duplicate", budget=100, seed=42, test_map50=0.99)
    aggregate = aggregate_supervised_matrix((low, duplicate))
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] is None
    assert aggregate["upper_bound_gate"]["status"] == "missing_fixed_test_evidence"
    assert all(row["canonical_protocol"] is False for row in aggregate["rows"])
    assert all(any("duplicate canonical matrix identity" in issue for issue in row["issues"]) for row in aggregate["rows"])


def test_replaced_or_unbound_test_json_is_excluded_from_the_gate(tmp_path: Path) -> None:
    run = _record(tmp_path / "upper", budget=100, seed=42, test_map50=0.9)
    output = run / "evaluations" / "test.json"
    fake = _test_evidence(run_id="other-run", metrics=_metrics(0.99))
    output.write_text(json.dumps(fake), encoding="utf-8")
    aggregate = aggregate_supervised_matrix((run,))
    row = aggregate["rows"][0]
    assert row["fixed_test"] is None
    assert any("does not bind to run_record" in issue for issue in row["issues"])
    assert aggregate["upper_bound_gate"]["fixed_test_map50"] is None


def test_metrics_content_hash_detects_replaced_metrics_even_with_original_envelope(tmp_path: Path) -> None:
    run = _record(tmp_path / "upper", budget=100, seed=42, test_map50=0.9)
    output = run / "evaluations" / "test.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["metrics"]["map50"] = 0.99
    output.write_text(json.dumps(payload), encoding="utf-8")
    aggregate = aggregate_supervised_matrix((run,))
    assert aggregate["rows"][0]["fixed_test"] is None
    assert any("metrics_sha256" in issue for issue in aggregate["rows"][0]["issues"])


def test_aggregate_output_is_immutable(tmp_path: Path) -> None:
    result = aggregate_supervised_matrix(())
    output = tmp_path / "aggregate.json"
    assert write_supervised_matrix_aggregate(result, output) == output
    with pytest.raises(SupervisedMatrixError, match="already exists"):
        write_supervised_matrix_aggregate(result, output)


def test_aggregate_rejects_file_parent_without_partial_output(tmp_path: Path) -> None:
    result = aggregate_supervised_matrix(())
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("x", encoding="utf-8")
    destination = parent_file / "aggregate.json"
    with pytest.raises(SupervisedMatrixError, match="cannot be written"):
        write_supervised_matrix_aggregate(result, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".aggregate.json.*.tmp"))
