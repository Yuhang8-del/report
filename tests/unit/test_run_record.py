"""Tests for immutable, reproducible training run records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.training.run_record import (
    RunRecordError,
    complete_run_record,
    create_run_record,
    read_run_record,
    split_fingerprint_from_manifest,
    write_run_record,
)
from fruit_ssod.training.supervised import (
    SupervisedTrainingError,
    SupervisedExperiment,
    SupervisedTrainingRunner,
    TrainingExecution,
    load_supervised_experiment,
)
from fruit_ssod.evaluation.detection_metrics import DetectionMetrics


def _record():
    return create_run_record(
        config_snapshot={"epochs": 1, "nested": {"batch": 4}},
        split_fingerprint="a" * 64,
        command=("python", "-m", "fruit_ssod.cli.train_supervised", "--dry-run"),
        environment={"python": "3.10"},
        run_id="supervised-20-test",
    )


def _canonical_result() -> dict[str, object]:
    return DetectionMetrics(
        0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)}
    ).mapping()


def test_run_id_snapshot_and_fingerprint_are_immutable() -> None:
    snapshot = {"epochs": 1, "nested": {"batch": 4}}
    record = create_run_record(
        config_snapshot=snapshot,
        split_fingerprint="a" * 64,
        command=["python", "-m", "fruit_ssod.cli.train_supervised"],
        environment={"python": "3.10"},
        run_id="supervised-20-test",
    )

    snapshot["nested"]["batch"] = 99

    assert record.run_id == "supervised-20-test"
    assert record.config_snapshot["nested"]["batch"] == 4
    assert record.command == ("python", "-m", "fruit_ssod.cli.train_supervised")
    with pytest.raises(TypeError):
        record.config_snapshot["epochs"] = 2  # type: ignore[index]
    with pytest.raises(Exception):
        record.run_id = "changed"  # type: ignore[misc]


def test_run_record_serializes_and_only_allows_terminal_transition(tmp_path: Path) -> None:
    path = tmp_path / "run_record.json"
    running = _record()
    write_run_record(running, path)
    complete = complete_run_record(running, _canonical_result())
    write_run_record(complete, path, allow_status_update=True)

    loaded = read_run_record(path)

    assert loaded.status == "complete"
    assert loaded.run_id == running.run_id
    assert loaded.result["map50"] == 0.81
    assert json.loads(path.read_text(encoding="utf-8"))["config_snapshot"]["nested"]["batch"] == 4
    with pytest.raises(RunRecordError, match="terminal"):
        write_run_record(complete, path, allow_status_update=True)


def test_complete_run_records_require_a_full_canonical_detection_result(tmp_path: Path) -> None:
    running = _record()
    incomplete = {"map50": 0.81, "per_class_ap50": {str(index): 0.8 for index in range(5)}}

    with pytest.raises(RunRecordError, match="serialized evaluation metrics"):
        complete_run_record(running, incomplete)

    missing_class = _canonical_result()
    missing_class["per_class_ap50"].pop("4")  # type: ignore[index]
    with pytest.raises(RunRecordError, match="per_class_ap50"):
        complete_run_record(running, missing_class)

    remapped_class = _canonical_result()
    remapped_class["per_class_ap50"]["00"] = remapped_class["per_class_ap50"].pop("0")  # type: ignore[index]
    with pytest.raises(RunRecordError, match="per_class_ap50"):
        complete_run_record(running, remapped_class)

    complete = complete_run_record(running, _canonical_result())
    assert complete.result == _canonical_result()

    path = tmp_path / "run_record.json"
    payload = complete.mapping()
    payload["result"].pop("recall")  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunRecordError, match="serialized evaluation metrics"):
        read_run_record(path)


def test_split_manifest_fingerprint_is_stable_and_detects_invalid_payload(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"fingerprints": {"split_protocol": "b" * 64}}), encoding="utf-8")

    assert split_fingerprint_from_manifest(manifest) == "b" * 64

    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(RunRecordError, match="fingerprints"):
        split_fingerprint_from_manifest(manifest)


def test_real_executor_contract_persists_weights_curves_and_raw_validation(tmp_path: Path) -> None:
    model = tmp_path / "model.yaml"
    model.write_text("model: yolov8s.yaml\n", encoding="utf-8")
    data = tmp_path / "dataset.yaml"
    data.write_text(
        "path: .\ntrain: train\nval: val\ntest: test\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n",
        encoding="utf-8",
    )
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"fingerprints": {"split_protocol": "d" * 64}}), encoding="utf-8")
    experiment = SupervisedExperiment(
        source_config=tmp_path / "experiment.yaml", experiment_name="fixture", model_config=model,
        dataset_yaml=data, split_manifest=split, artifact_root=tmp_path / "artifacts", seed=42,
        device="cpu", epochs=1,
    )

    def executor(invocation):
        weights = invocation.run_dir / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        metrics = DetectionMetrics(0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)})
        return TrainingExecution(metrics=metrics, raw_validation={"raw": "validation-output"}, curves={"epoch": [1]})

    record, run_dir = SupervisedTrainingRunner(executor=executor).run(
        experiment, command=("python", "-m", "fruit_ssod.cli.train_supervised")
    )

    assert record.status == "complete"
    assert (run_dir / "weights" / "best.pt").is_file()
    assert (run_dir / "weights" / "last.pt").is_file()
    assert json.loads((run_dir / "training_curves.json").read_text(encoding="utf-8")) == {"epoch": [1]}
    assert json.loads((run_dir / "validation_raw.json").read_text(encoding="utf-8")) == {"raw": "validation-output"}
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["map50"] == 0.81
    checkpoints = json.loads((run_dir / "checkpoint_evidence.json").read_text(encoding="utf-8"))
    assert checkpoints["best.pt"]["bytes"] == 4
    assert len(checkpoints["best.pt"]["sha256"]) == 64
    assert checkpoints["last.pt"]["bytes"] == 4


def _write_experiment_config(tmp_path: Path, *, names: str = "[Apple, Banana, Orange, Strawberry, Pineapple]") -> tuple[Path, Path, Path, Path]:
    model = tmp_path / "model.yaml"
    model.write_text("model: yolov8s.yaml\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(f"path: .\ntrain: train\nval: val\ntest: test\nnames: {names}\n", encoding="utf-8")
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"fingerprints": {"split_protocol": "e" * 64}}), encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            (
                "experiment_name: snapshot_fixture",
                f"model_config: '{model.as_posix()}'",
                f"dataset_yaml: '{dataset.as_posix()}'",
                f"split_manifest: '{split.as_posix()}'",
                f"artifact_root: '{(tmp_path / 'artifacts').as_posix()}'",
            )
        ) + "\n",
        encoding="utf-8",
    )
    return config, model, dataset, split


def test_snapshot_captures_full_effective_yaml_and_resume_rejects_content_change(tmp_path: Path) -> None:
    config, model, dataset, split = _write_experiment_config(tmp_path)
    experiment = load_supervised_experiment(config)
    snapshot = experiment.snapshot()

    assert snapshot["model_reference"] == "yolov8s.yaml"
    assert snapshot["model_config_effective"]["names"] == list(experiment.snapshot()["canonical_classes"])
    assert snapshot["dataset_yaml_effective"]["names"] == list(experiment.snapshot()["canonical_classes"])
    assert set(snapshot["dataset_paths"]) == {"train", "val", "test"}
    assert len(snapshot["model_config_sha256"]) == len(snapshot["dataset_yaml_sha256"]) == 64

    run_dir = experiment.artifact_root / "runs" / "interrupted"
    (run_dir / "weights").mkdir(parents=True)
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.write_bytes(b"last")
    write_run_record(
        create_run_record(
            config_snapshot=snapshot, split_fingerprint="e" * 64,
            command=("python", "-m", "fruit_ssod.cli.train_supervised"), environment={"python": "3.10"}, run_id="interrupted",
        ),
        run_dir / "run_record.json",
    )
    model.write_text("model: changed.yaml\nnames: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")

    with pytest.raises(SupervisedTrainingError, match="resume provenance"):
        SupervisedTrainingRunner().run(experiment, command=("python",), resume=checkpoint)


def test_dataset_yaml_requires_exact_canonical_names_during_load(tmp_path: Path) -> None:
    config, _, _, _ = _write_experiment_config(tmp_path, names="[Apple, Banana, Orange, Strawberry, pear]")

    with pytest.raises(SupervisedTrainingError, match="dataset YAML class names"):
        load_supervised_experiment(config)


def test_missing_or_non_json_safe_evidence_marks_run_failed(tmp_path: Path) -> None:
    config, _, _, _ = _write_experiment_config(tmp_path)
    experiment = load_supervised_experiment(config)

    def missing_executor(invocation):
        weights = invocation.run_dir / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        metrics = DetectionMetrics(0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)})
        return TrainingExecution(metrics=metrics, raw_validation={}, curves={"epoch": [1]})

    runner = SupervisedTrainingRunner(executor=missing_executor)
    with pytest.raises(SupervisedTrainingError, match="raw validation payload"):
        runner.run(experiment, command=("python",), run_id="missing-evidence")

    payload = json.loads((experiment.artifact_root / "runs" / "missing-evidence" / "run_record.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "raw validation payload" in payload["failure"]["cause"]

    def non_json_executor(invocation):
        weights = invocation.run_dir / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        metrics = DetectionMetrics(0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)})
        return TrainingExecution(metrics=metrics, raw_validation={"value": float("nan")}, curves={"epoch": [1]})

    with pytest.raises(SupervisedTrainingError, match="run artifact cannot be serialized"):
        SupervisedTrainingRunner(executor=non_json_executor).run(experiment, command=("python",), run_id="non-json-evidence")
    non_json_payload = json.loads((experiment.artifact_root / "runs" / "non-json-evidence" / "run_record.json").read_text(encoding="utf-8"))
    assert non_json_payload["status"] == "failed"


def test_static_artifact_failure_is_recorded_as_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, _ = _write_experiment_config(tmp_path)
    experiment = load_supervised_experiment(config)
    runner = SupervisedTrainingRunner()

    def fail_static(*_args, **_kwargs):
        raise SupervisedTrainingError("Problem: static artifact cannot be saved. Likely cause: fixture. Remediation: fix fixture.")

    monkeypatch.setattr(runner, "_write_static_artifacts", fail_static)
    with pytest.raises(SupervisedTrainingError, match="static artifact"):
        runner.run(experiment, command=("python",), dry_run=True, run_id="static-failure")

    payload = json.loads((experiment.artifact_root / "runs" / "static-failure" / "run_record.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "static artifact" in payload["failure"]["cause"]


def test_resume_requires_the_prior_regular_nonempty_last_checkpoint_and_appends_evidence(tmp_path: Path) -> None:
    config, _, _, _ = _write_experiment_config(tmp_path)
    experiment = load_supervised_experiment(config)
    snapshot = experiment.snapshot()
    run_dir = experiment.artifact_root / "runs" / "interrupted"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"checkpoint-v1")
    write_run_record(
        create_run_record(
            config_snapshot=snapshot, split_fingerprint="e" * 64,
            command=("original-command",), environment={"python": "3.10"}, run_id="interrupted",
        ),
        run_dir / "run_record.json",
    )
    (run_dir / "command.txt").write_text("original-command\n", encoding="utf-8")
    previous_event = {"command": ["prior-resume"], "checkpoint": {"path": "old", "bytes": 1, "sha256": "0" * 64}}
    (run_dir / "resume_history.jsonl").write_text(json.dumps(previous_event) + "\n", encoding="utf-8")

    def executor(invocation):
        (weights / "best.pt").write_bytes(b"best-v2")
        (weights / "last.pt").write_bytes(b"last-v2")
        return TrainingExecution(
            metrics=DetectionMetrics(0.81, 0.61, 0.8, 0.7, 0.7467, {index: 0.8 for index in range(5)}),
            raw_validation={"raw": "validation-output"}, curves={"epoch": [2]},
        )

    completed, _ = SupervisedTrainingRunner(executor=executor).run(
        experiment, command=("resumed-command", "--resume", str(last)), resume=last,
    )

    assert completed.status == "complete"
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == "original-command\n"
    events = [json.loads(line) for line in (run_dir / "resume_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert events[0] == previous_event
    assert events[1]["command"] == ["resumed-command", "--resume", str(last)]
    assert events[1]["checkpoint"]["bytes"] == 13
    assert events[1]["checkpoint"]["path"] == str(last.resolve())
    assert len(events[1]["checkpoint"]["sha256"]) == 64

    invalid = run_dir / "weights" / "best.pt"
    with pytest.raises(SupervisedTrainingError, match="unsafe location"):
        SupervisedTrainingRunner().run(experiment, command=("python",), resume=invalid)

    empty_run = experiment.artifact_root / "runs" / "empty"
    (empty_run / "weights").mkdir(parents=True)
    empty = empty_run / "weights" / "last.pt"
    empty.write_bytes(b"")
    with pytest.raises(SupervisedTrainingError, match="is empty"):
        SupervisedTrainingRunner().run(experiment, command=("python",), resume=empty)


def test_noncanonical_training_metrics_cannot_complete_a_run(tmp_path: Path) -> None:
    config, _, _, _ = _write_experiment_config(tmp_path)
    experiment = load_supervised_experiment(config)

    def executor(invocation):
        weights = invocation.run_dir / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        return TrainingExecution(  # type: ignore[arg-type]
            metrics=object(), raw_validation={"raw": "validation-output"}, curves={"epoch": [1]},
        )

    with pytest.raises(SupervisedTrainingError, match="not DetectionMetrics"):
        SupervisedTrainingRunner(executor=executor).run(experiment, command=("python",), run_id="invalid-metrics")
    payload = json.loads((experiment.artifact_root / "runs" / "invalid-metrics" / "run_record.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
