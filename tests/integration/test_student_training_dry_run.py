"""CLI dry run must compose data without GPU/model execution."""

from __future__ import annotations

import os
import json
import sys
import types
from pathlib import Path

import pytest

from fruit_ssod.cli.train_student import main
from fruit_ssod.training.semi_supervised import SemiSupervisedTrainingError, StudentExperiment, StudentInvocation, UltralyticsStudentExecutor, load_student_experiment
from fruit_ssod.training.supervised import file_evidence
from fruit_ssod.training.student_dataset import StudentDatasetError, compose_student_dataset
from tests.unit.test_student_dataset import _inputs


def test_student_cli_dry_run_writes_sealed_snapshot(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    model = Path(__file__).parents[2] / "configs" / "models" / "yolov8s_640.yaml"
    weights = tmp_path / "shared-pretrained.pt"
    weights.write_bytes(b"fixture-pretrained-weights")
    teacher = tmp_path / "teacher.yaml"
    teacher.write_text("\n".join((
        "experiment_name: teacher", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}",
        "initialization_policy:", "  policy_id: ssod_student_init_v1",
        "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture",
    )), encoding="utf-8")
    config = tmp_path / "student.yaml"
    config.write_text("\n".join([f"experiment_name: student_fixture", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}", f"artifact_root: {(tmp_path / 'artifacts').as_posix()}", f"source_root: {inputs.source_root.as_posix()}", f"split_manifest: {inputs.split_manifest.as_posix()}", f"human_images: {inputs.human_images.as_posix()}", f"human_labels: {inputs.human_labels.as_posix()}", f"validation_labels: {inputs.validation_labels.as_posix()}", f"unlabeled_manifest: {inputs.unlabeled_manifest.as_posix()}", f"candidates: {inputs.candidates.as_posix()}", f"filter_audit: {inputs.filter_audit.as_posix()}", f"filter_decision_manifest: {inputs.filter_decision_manifest.as_posix()}", f"pseudo_audit_report: {inputs.pseudo_audit_report.as_posix()}", "seed: 42", "label_budget_percent: 20", "human_sample_probability: 0.5", "initialization_policy:", "  policy_id: ssod_student_init_v1", "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture", f"  teacher_experiment_config: {teacher.as_posix()}", "  teacher_run_id: teacher", "epochs: 1"]), encoding="utf-8")
    assert main(["--config", str(config), "--dry-run", "--run-id", "student-dry-run"]) == 0
    roots = list((tmp_path / "artifacts" / "runs").glob("dry-run-student-dry-run-*"))
    assert len(roots) == 1
    root = roots[0]
    assert (root / "student_dataset" / "dataset.yaml").is_file()
    snapshot = (root / "config_snapshot.json").read_text(encoding="utf-8")
    assert "pseudo_audit_labels" not in snapshot and "test_labels" not in snapshot
    assert "model_config_sha256" in snapshot and "dataset_yaml_sha256" in snapshot
    assert (root / "environment.json").is_file() and (root / "command.txt").is_file()
    # Retain a second disposable snapshot instead of occupying the fixed ID
    # that a later real run may use.
    assert main(["--config", str(config), "--dry-run", "--run-id", "student-dry-run"]) == 0
    roots = list((tmp_path / "artifacts" / "runs").glob("dry-run-student-dry-run-*"))
    assert len(roots) == 2
    assert not (tmp_path / "artifacts" / "runs" / "student-dry-run").exists()


def test_student_rejects_candidate_teacher_other_than_declared_config(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    from dataclasses import replace

    with pytest.raises(StudentDatasetError, match="declared Teacher configuration"):
        compose_student_dataset(replace(inputs, expected_teacher_run_id="another-teacher"), tmp_path / "out")


def test_external_teacher_checkpoint_policy_binds_completed_run_and_checkpoint(tmp_path: Path) -> None:
    """An exploratory Teacher must be explicit, completed, and byte-bound."""
    inputs = _inputs(tmp_path)
    model = Path(__file__).parents[2] / "configs" / "models" / "yolov8s_640.yaml"
    checkpoint = tmp_path / "teacher-best.pt"; checkpoint.write_bytes(b"teacher-best")
    original = tmp_path / "teacher-start.pt"; original.write_bytes(b"teacher-start")
    teacher = tmp_path / "teacher.yaml"
    teacher.write_text("\n".join(("experiment_name: full_teacher", f"model_config: {model.as_posix()}", f"pretrained_weights: {original.as_posix()}")), encoding="utf-8")
    record = tmp_path / "run_record.json"
    record.write_text(json.dumps({"run_id": "teacher-run", "status": "complete", "config_snapshot": {"experiment_name": "full_teacher"}}), encoding="utf-8")
    config = tmp_path / "student.yaml"
    config.write_text("\n".join((
        "experiment_name: exploratory_student", f"model_config: {model.as_posix()}", f"pretrained_weights: {checkpoint.as_posix()}", f"artifact_root: {(tmp_path / 'artifacts').as_posix()}", f"source_root: {inputs.source_root.as_posix()}", f"split_manifest: {inputs.split_manifest.as_posix()}", f"human_images: {inputs.human_images.as_posix()}", f"human_labels: {inputs.human_labels.as_posix()}", f"validation_labels: {inputs.validation_labels.as_posix()}", f"unlabeled_manifest: {inputs.unlabeled_manifest.as_posix()}", f"candidates: {inputs.candidates.as_posix()}", f"filter_audit: {inputs.filter_audit.as_posix()}", f"filter_decision_manifest: {inputs.filter_decision_manifest.as_posix()}", f"pseudo_audit_report: {inputs.pseudo_audit_report.as_posix()}",
        "initialization_policy:", "  policy_id: external_teacher_self_training_v1", "  model_initialization: teacher_checkpoint", "  comparison_group: exploratory", f"  teacher_experiment_config: {teacher.as_posix()}", "  teacher_run_id: teacher-run", f"  teacher_checkpoint: {checkpoint.as_posix()}", f"  teacher_run_record: {record.as_posix()}",
    )), encoding="utf-8")
    experiment = load_student_experiment(config)
    assert experiment.initialization_evidence["teacher_run_id"] == "teacher-run"  # type: ignore[index]
    assert experiment.initialization_evidence["teacher_checkpoint"]["sha256"] == experiment.initialization_evidence["student_pretrained_weights"]["sha256"]  # type: ignore[index]


def test_student_rejects_candidate_checkpoint_other_than_external_teacher(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    from dataclasses import replace

    with pytest.raises(StudentDatasetError, match="checkpoint differs"):
        compose_student_dataset(replace(inputs, expected_teacher_source_model="another-teacher.pt"), tmp_path / "out")


def test_student_dry_run_truncates_a_maximum_legal_requested_run_id(tmp_path: Path) -> None:
    # Reuse the CLI fixture setup, then issue a max-length legal ID.  The
    # disposable prefix/UUID must remain inside RunRecord's 160-character
    # Windows-safe bound.
    inputs = _inputs(tmp_path)
    model = Path(__file__).parents[2] / "configs" / "models" / "yolov8s_640.yaml"
    weights = tmp_path / "shared-pretrained.pt"; weights.write_bytes(b"fixture-pretrained-weights")
    teacher = tmp_path / "teacher.yaml"
    teacher.write_text("\n".join(("experiment_name: teacher", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}", "initialization_policy:", "  policy_id: ssod_student_init_v1", "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture")), encoding="utf-8")
    config = tmp_path / "student.yaml"
    config.write_text("\n".join(("experiment_name: student_fixture", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}", f"artifact_root: {(tmp_path / 'artifacts').as_posix()}", f"source_root: {inputs.source_root.as_posix()}", f"split_manifest: {inputs.split_manifest.as_posix()}", f"human_images: {inputs.human_images.as_posix()}", f"human_labels: {inputs.human_labels.as_posix()}", f"validation_labels: {inputs.validation_labels.as_posix()}", f"unlabeled_manifest: {inputs.unlabeled_manifest.as_posix()}", f"candidates: {inputs.candidates.as_posix()}", f"filter_audit: {inputs.filter_audit.as_posix()}", f"filter_decision_manifest: {inputs.filter_decision_manifest.as_posix()}", f"pseudo_audit_report: {inputs.pseudo_audit_report.as_posix()}", "seed: 42", "label_budget_percent: 20", "human_sample_probability: 0.5", "initialization_policy:", "  policy_id: ssod_student_init_v1", "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture", f"  teacher_experiment_config: {teacher.as_posix()}", "  teacher_run_id: teacher", "epochs: 1")), encoding="utf-8")
    requested = "x" * 160
    assert main(["--config", str(config), "--dry-run", "--run-id", requested]) == 0
    roots = list((tmp_path / "artifacts" / "runs").iterdir())
    assert len(roots) == 1 and len(roots[0].name) <= 160


def test_student_executor_rehashes_pretrained_weights_immediately_before_loading(tmp_path: Path, monkeypatch) -> None:
    # Build a validated experiment, then alter the local checkpoint after its
    # evidence is sealed.  The executor must reject it before YOLO.load().
    inputs = _inputs(tmp_path)
    model = Path(__file__).parents[2] / "configs" / "models" / "yolov8s_640.yaml"
    weights = tmp_path / "shared-pretrained.pt"; weights.write_bytes(b"before")
    teacher = tmp_path / "teacher.yaml"
    teacher.write_text("\n".join(("experiment_name: teacher", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}", "initialization_policy:", "  policy_id: ssod_student_init_v1", "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture")), encoding="utf-8")
    config = tmp_path / "student.yaml"
    config.write_text("\n".join(("experiment_name: student_fixture", f"model_config: {model.as_posix()}", f"pretrained_weights: {weights.as_posix()}", f"artifact_root: {(tmp_path / 'artifacts').as_posix()}", f"source_root: {inputs.source_root.as_posix()}", f"split_manifest: {inputs.split_manifest.as_posix()}", f"human_images: {inputs.human_images.as_posix()}", f"human_labels: {inputs.human_labels.as_posix()}", f"validation_labels: {inputs.validation_labels.as_posix()}", f"unlabeled_manifest: {inputs.unlabeled_manifest.as_posix()}", f"candidates: {inputs.candidates.as_posix()}", f"filter_audit: {inputs.filter_audit.as_posix()}", f"filter_decision_manifest: {inputs.filter_decision_manifest.as_posix()}", f"pseudo_audit_report: {inputs.pseudo_audit_report.as_posix()}", "seed: 42", "label_budget_percent: 20", "human_sample_probability: 0.5", "initialization_policy:", "  policy_id: ssod_student_init_v1", "  model_initialization: shared_pretrained_weights", "  comparison_group: fixture", f"  teacher_experiment_config: {teacher.as_posix()}", "  teacher_run_id: teacher", "epochs: 1")), encoding="utf-8")
    experiment = load_student_experiment(config)
    weights.write_bytes(b"after")
    loaded: list[str] = []
    class FakeYolo:
        def __init__(self, _model: str) -> None: pass
        def load(self, path: str) -> None: loaded.append(path)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    with pytest.raises(SemiSupervisedTrainingError, match="changed after configuration validation"):
        UltralyticsStudentExecutor()(StudentInvocation(tmp_path / "run", experiment, object()))  # type: ignore[arg-type]
    assert not loaded


def test_student_executor_validates_the_published_best_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """Final Student metrics must come from weights/best.pt, not train state."""
    model = Path(__file__).parents[2] / "configs" / "models" / "yolov8s_640.yaml"
    weights = tmp_path / "initial.pt"; weights.write_bytes(b"initial")
    dataset_yaml = tmp_path / "dataset.yaml"; dataset_yaml.write_text("names: [Apple, Banana, Orange, Strawberry, Pineapple]\n", encoding="utf-8")
    run = tmp_path / "run"; run.mkdir()
    experiment = StudentExperiment(
        source_config=model, experiment_name="student", model_config=model, artifact_root=tmp_path,
        dataset_inputs=object(), seed=42, epochs=1, patience=1, pretrained_weights=weights,
        initialization_evidence={"student_pretrained_weights": file_evidence(weights, description="fixture")},
    )
    constructed: list[str] = []
    train_kwargs: dict[str, object] = {}
    class FakeYolo:
        names = {0: "Apple", 1: "Banana", 2: "Orange", 3: "Strawberry", 4: "Pineapple"}
        def __init__(self, value: str) -> None:
            constructed.append(value)
            self.callbacks: dict[str, list] = {}
        def add_callback(self, event: str, callback) -> None:
            self.callbacks.setdefault(event, []).append(callback)
        def load(self, _path: str) -> None: pass
        def train(self, **kwargs: object) -> None:
            train_kwargs.update(kwargs)
            (run / "weights").mkdir(); (run / "weights" / "best.pt").write_bytes(b"best")
            (run / "results.csv").write_text("epoch\n0\n", encoding="utf-8")
            trainer = types.SimpleNamespace(best_fitness=.8, fitness=.8, best=run / "weights" / "best.pt", epoch=0)
            for callback in self.callbacks["on_model_save"]:
                callback(trainer)
        def val(self, **_kwargs: object):
            box = types.SimpleNamespace(all_ap=[[.1], [.2], [.3], [.4], [.5]], mp=.6, mr=.7, map50=.8, map=.5)
            return types.SimpleNamespace(box=box, results_dict={"metrics/mAP50(B)": .8}, speed={})
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    result = UltralyticsStudentExecutor()(StudentInvocation(run, experiment, types.SimpleNamespace(dataset_yaml=dataset_yaml)))
    assert constructed == ["yolov8s.yaml", str(run / "weights" / "best.pt")]
    assert train_kwargs["workers"] == 0
    assert result.metrics.map50 == .8
    assert result.curves["best_weights"]["sha256"] == file_evidence(run / "weights" / "best.pt", description="fixture")["sha256"]
    assert result.curves["curve_best_capture"]["epoch"] == 1
    assert (run / "weights" / "curve_best_pre_final.pt").is_file()
