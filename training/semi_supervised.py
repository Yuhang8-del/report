"""Auditable Student training orchestration for the first SSOD iteration."""

from __future__ import annotations

import json
import os
import re
import shutil
import hashlib
import math
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError, validate_class_mapping
from fruit_ssod.evaluation.detection_metrics import DetectionMetrics, DetectionMetricsError, metrics_from_mapping
from fruit_ssod.training.run_record import RunRecord, complete_run_record, create_run_record, fail_run_record, split_fingerprint_from_manifest, write_run_record
from fruit_ssod.training.student_dataset import StudentDatasetInputs, StudentDatasetResult, StudentDatasetError, _normalize_pseudo_filter_policy, compose_student_dataset
from fruit_ssod.training.supervised import TrainingExecution, _serialize_metric_object, _yaml_evidence, environment_details, file_evidence


class SemiSupervisedTrainingError(RuntimeError):
    """Raised for an actionable Student-training protocol violation."""


def _problem(problem: str, cause: str, remediation: str) -> SemiSupervisedTrainingError:
    return SemiSupervisedTrainingError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _path(value: object, *, base: Path, field: str, exists: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} is missing", "Student experiment YAML omits a required path", f"set {field} to a readable path")
    def expand(match: re.Match[str]) -> str:
        item = os.environ.get(match.group(1))
        if not item:
            raise _problem(f"{field} references unset {match.group(1)}", "machine-local storage was not configured", f"set {match.group(1)} before launching Student training")
        return item
    raw = Path(_VARIABLE.sub(expand, value))
    resolved = (raw if raw.is_absolute() else base / raw).resolve(strict=exists)
    if exists and not resolved.is_file():
        raise _problem(f"{field} is not a file", f"{resolved} is missing or a directory", f"set {field} to the required artifact")
    return resolved


def _positive(value: object, field: str, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _problem(f"{field} is invalid", f"received {value!r}", f"set {field} to a positive integer")
    return value


def _nonnegative(value: object, field: str, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _problem(f"{field} is invalid", f"received {value!r}", f"set {field} to a nonnegative integer")
    return value


def _rate(value: object, field: str, default: float) -> float:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
        raise _problem(f"{field} is invalid", f"received {value!r}", f"set {field} to a finite learning rate in (0, 1]")
    return float(value)


def _fraction(value: object, field: str, default: float) -> float:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise _problem(f"{field} is invalid", f"received {value!r}", "set it to a finite value in [0, 1]")
    return float(value)


def _bool(value: object, field: str, default: bool) -> bool:
    value = default if value is None else value
    if not isinstance(value, bool):
        raise _problem(f"{field} is invalid", f"received {value!r}", "set it to true or false")
    return value


def _save_period(value: object) -> int:
    value = -1 if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise _problem("save_period is invalid", f"received {value!r}", "set save_period to -1 or a nonnegative epoch interval")
    return value


def _student_dataset_evidence(path: Path) -> tuple[dict[str, Any], str]:
    """Freeze the effective two-partition Student dataset YAML.

    The derived Student snapshot deliberately has train/validation only; held
    out test and pseudo-audit labels must never appear in it.  This mirrors the
    supervised YAML evidence pattern without inventing a test partition.
    """
    payload, digest = _yaml_evidence(path, "Student dataset YAML")
    if payload.get("names") != list(DEFAULT_CLASS_REGISTRY.class_names):
        raise _problem("Student dataset YAML violates the five-class contract", "the derived dataset names differ from the canonical registry", "regenerate the sealed Student dataset")
    root_value = payload.get("path")
    if not isinstance(root_value, str) or not root_value.strip():
        raise _problem("Student dataset YAML path is invalid", "the derived dataset root is missing", "regenerate the sealed Student dataset")
    root = Path(root_value)
    root = root if root.is_absolute() else path.parent / root
    try:
        root = root.resolve(strict=False)
    except OSError as error:
        raise _problem("Student dataset YAML path cannot be resolved", str(error), "regenerate the sealed Student dataset on a readable filesystem") from error
    effective = dict(payload)
    effective["path"] = str(root)
    for field in ("train", "val"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise _problem(f"Student dataset YAML {field} is invalid", "the derived dataset has no usable split reference", "regenerate the sealed Student dataset")
        candidate = Path(value)
        try:
            effective[field] = str((candidate if candidate.is_absolute() else root / candidate).resolve(strict=False))
        except OSError as error:
            raise _problem(f"Student dataset YAML {field} cannot be resolved", str(error), "regenerate the sealed Student dataset") from error
    return effective, digest


def _shared_initialization_evidence(*, source: Path, model_config: Path, weights: Path, policy: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    """Validate that the Teacher and Student start from identical local bytes."""
    teacher_config = _path(policy.get("teacher_experiment_config"), base=source.parent, field="initialization_policy.teacher_experiment_config")
    try:
        teacher_payload, teacher_config_digest = _yaml_evidence(teacher_config, "Teacher experiment config")
    except Exception as error:
        if isinstance(error, SemiSupervisedTrainingError):
            raise
        raise _problem("Teacher experiment config cannot be read", str(error), "provide the exact supervised Teacher configuration") from error
    teacher_model = _path(teacher_payload.get("model_config"), base=teacher_config.parent, field="Teacher model_config")
    student_model, student_model_digest = _yaml_evidence(model_config, "Student model config")
    teacher_model_payload, teacher_model_digest = _yaml_evidence(teacher_model, "Teacher model config")
    if student_model_digest != teacher_model_digest:
        raise _problem("Teacher and Student model configurations differ", "the shared initialization comparison would not use the same five-class architecture", "point both configurations to the same canonical YOLOv8s model YAML")
    student_weights_evidence = file_evidence(weights, description="Student pretrained weights")
    if policy["model_initialization"] == "teacher_checkpoint":
        checkpoint = _path(policy.get("teacher_checkpoint"), base=source.parent, field="initialization_policy.teacher_checkpoint")
        checkpoint_evidence = file_evidence(checkpoint, description="Teacher checkpoint")
        if checkpoint_evidence["sha256"] != student_weights_evidence["sha256"]:
            raise _problem("Student initialization does not equal the declared Teacher checkpoint", "pretrained_weights and initialization_policy.teacher_checkpoint have different SHA-256 values", "initialize the exploratory Student from the exact Teacher checkpoint")
        run_record = _path(policy.get("teacher_run_record"), base=source.parent, field="initialization_policy.teacher_run_record")
        try:
            completed = json.loads(run_record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _problem("Teacher run record cannot be read", str(error), "provide the completed Teacher run_record.json") from error
        snapshot = completed.get("config_snapshot") if isinstance(completed, Mapping) else None
        if not isinstance(completed, Mapping) or completed.get("status") != "complete" or completed.get("run_id") != policy["teacher_run_id"] or not isinstance(snapshot, Mapping) or snapshot.get("experiment_name") != teacher_payload.get("experiment_name"):
            raise _problem("Teacher run record is not bound to the declared completed Teacher", "run ID, completion state, or experiment configuration differs", "use the run_record.json from the declared completed Teacher run")
        return teacher_config, {
            "policy": {key: policy[key] for key in ("policy_id", "model_initialization", "comparison_group")},
            "student_pretrained_weights": student_weights_evidence,
            "teacher_checkpoint": checkpoint_evidence,
            "teacher_run_record": file_evidence(run_record, description="Teacher run record"),
            "teacher_run_id": policy["teacher_run_id"],
            "teacher_experiment_config": str(teacher_config),
            "teacher_experiment_config_sha256": teacher_config_digest,
            "student_model_config_sha256": student_model_digest,
            "teacher_model_config_sha256": teacher_model_digest,
            "model_reference": student_model.get("model"),
        }
    teacher_weights = _path(teacher_payload.get("pretrained_weights"), base=teacher_config.parent, field="Teacher pretrained_weights")
    teacher_weights_evidence = file_evidence(teacher_weights, description="Teacher pretrained weights")
    if student_weights_evidence["sha256"] != teacher_weights_evidence["sha256"]:
        raise _problem("Teacher and Student pretrained weights differ", "their SHA-256 evidence is not identical", "configure both runs with the same local pretrained checkpoint; this workflow never downloads weights")
    teacher_policy = teacher_payload.get("initialization_policy")
    if not isinstance(teacher_policy, Mapping) or {key: teacher_policy.get(key) for key in ("policy_id", "model_initialization", "comparison_group")} != {key: policy[key] for key in ("policy_id", "model_initialization", "comparison_group")}:
        raise _problem("Teacher and Student initialization policies differ", "the comparable runs do not declare the same initialization protocol", "make policy_id, model_initialization, and comparison_group identical")
    teacher_run_id = policy["teacher_run_id"]
    if teacher_payload.get("experiment_name") != teacher_run_id:
        raise _problem(
            "declared teacher_run_id does not match the Teacher configuration",
            f"initialization_policy declares {teacher_run_id!r}, but the Teacher experiment_name is {teacher_payload.get('experiment_name')!r}",
            "set initialization_policy.teacher_run_id to the exact completed Teacher experiment run ID",
        )
    return teacher_config, {
        "policy": {key: policy[key] for key in ("policy_id", "model_initialization", "comparison_group")},
        "shared_pretrained_weights_sha256": student_weights_evidence["sha256"],
        "student_pretrained_weights": student_weights_evidence,
        "teacher_pretrained_weights": teacher_weights_evidence,
        "teacher_experiment_config": str(teacher_config),
        "teacher_experiment_config_sha256": teacher_config_digest,
        "teacher_run_id": teacher_run_id,
        "student_model_config_sha256": student_model_digest,
        "teacher_model_config_sha256": teacher_model_digest,
        "model_reference": student_model.get("model"),
    }


@dataclass(frozen=True)
class StudentExperiment:
    source_config: Path
    experiment_name: str
    model_config: Path
    artifact_root: Path
    dataset_inputs: StudentDatasetInputs
    seed: int
    image_size: int = 640
    amp: bool = True
    batch: int = 4
    epochs: int = 100
    patience: int = 20
    learning_rate: float = 0.01
    freeze: int = 0
    save_period: int = -1
    device: str = "cuda:0"
    # Windows validation can exhaust host RAM when Ultralytics starts several
    # DataLoader workers.  Keep this explicit so low-memory retries are
    # reproducible instead of relying on the library default.
    workers: int = 0
    optimizer: str = "auto"
    cos_lr: bool = False
    close_mosaic: int = 10
    mosaic: float = 1.0
    mixup: float = 0.0
    initialization_policy: Mapping[str, str] | None = None
    pretrained_weights: Path | None = None
    teacher_experiment_config: Path | None = None
    initialization_evidence: Mapping[str, Any] | None = None

    def snapshot(self, dataset: StudentDatasetResult) -> dict[str, Any]:
        # The snapshot intentionally contains no Task 8 test/pseudo_audit
        # label paths.  Their influence is only the Task 15 hash gate.
        model_config, model_digest = _yaml_evidence(self.model_config, "Student model config")
        dataset_config, dataset_digest = _student_dataset_evidence(dataset.dataset_yaml)
        source_config, source_digest = _yaml_evidence(self.source_config, "Student experiment config")
        return {"experiment_name": self.experiment_name, "protocol": "student_ssod_v1", "source_config": str(self.source_config), "source_config_sha256": source_digest, "source_config_effective": source_config, "model_config": str(self.model_config), "model_config_sha256": model_digest, "model_config_effective": model_config, "model_reference": model_config["model"], "artifact_root": str(self.artifact_root), "seed": self.seed, "image_size": self.image_size, "amp": self.amp, "batch": self.batch, "epochs": self.epochs, "patience": self.patience, "learning_rate": self.learning_rate, "freeze": self.freeze, "save_period": self.save_period, "device": self.device, "workers": self.workers, "optimizer": self.optimizer, "cos_lr": self.cos_lr, "close_mosaic": self.close_mosaic, "mosaic": self.mosaic, "mixup": self.mixup, "initialization_policy": dict(self.initialization_policy or {}), "initialization_evidence": dict(self.initialization_evidence or {}), "student_dataset": {"dataset_yaml": str(dataset.dataset_yaml), "dataset_yaml_sha256": dataset_digest, "dataset_yaml_effective": dataset_config, "dataset_paths": {key: dataset_config[key] for key in ("train", "val")}, "training_membership": str(dataset.membership), "training_membership_sha256": hashlib.sha256(dataset.membership.read_bytes()).hexdigest(), "provenance": dict(dataset.provenance)}, "canonical_classes": list(DEFAULT_CLASS_REGISTRY.class_names)}


def load_student_experiment(path: Path | str) -> StudentExperiment:
    source = Path(path).resolve(strict=True)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem("Student experiment YAML cannot be read", str(error), "provide a readable UTF-8 YAML configuration") from error
    if not isinstance(raw, Mapping):
        raise _problem("Student experiment YAML is not an object", "the top-level YAML value is malformed", "use key/value experiment settings")
    name = raw.get("experiment_name")
    if not isinstance(name, str) or not name.strip():
        raise _problem("experiment_name is missing", "the Student run could not be uniquely identified", "set a nonempty experiment_name")
    model = _path(raw.get("model_config"), base=source.parent, field="model_config")
    try:
        model_payload = yaml.safe_load(model.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem("model_config cannot be read", str(error), "use the canonical five-class YOLO model YAML") from error
    if not isinstance(model_payload, Mapping) or model_payload.get("names") != list(DEFAULT_CLASS_REGISTRY.class_names) or not isinstance(model_payload.get("model"), str):
        raise _problem("model_config violates the five-class detector contract", "model names or architecture reference differ", "use configs/models/yolov8s_640.yaml")
    artifact = _path(raw.get("artifact_root"), base=source.parent, field="artifact_root", exists=False)
    policy = raw.get("initialization_policy")
    common_policy_fields = {"policy_id", "model_initialization", "comparison_group", "teacher_experiment_config", "teacher_run_id"}
    if not isinstance(policy, Mapping) or any(not isinstance(value, str) or not value.strip() for value in policy.values()):
        raise _problem("initialization_policy is incomplete", "comparable SSOD runs must name one immutable initialization policy", "set policy_id, model_initialization, and comparison_group")
    mode = policy.get("model_initialization")
    expected_policy_fields = common_policy_fields if mode == "shared_pretrained_weights" else common_policy_fields | {"teacher_checkpoint", "teacher_run_record"} if mode == "teacher_checkpoint" else set()
    if set(policy) != expected_policy_fields:
        raise _problem("initialization_policy is unsupported", "policy fields do not match its declared initialization mode", "use shared_pretrained_weights or teacher_checkpoint with Teacher checkpoint/run-record evidence")
    weights = _path(raw.get("pretrained_weights"), base=source.parent, field="pretrained_weights")
    policy_values = {key: policy[key] for key in sorted(policy)}
    teacher_config, initialization_evidence = _shared_initialization_evidence(source=source, model_config=model, weights=weights, policy=policy_values)
    fields = {name: _path(raw.get(name), base=source.parent, field=name) for name in ("split_manifest", "human_images", "human_labels", "validation_labels", "unlabeled_manifest", "candidates", "filter_audit", "filter_decision_manifest", "pseudo_audit_report")}
    pseudo_filter_raw = raw.get("pseudo_filter")
    # Task-16 standalone fixture/legacy experiments predate the Task-17
    # matrix.  They remain readable, but every Task-17 config is required by
    # matrix validation to carry the executable policy and therefore gets the
    # strict manifest binding below.
    if pseudo_filter_raw is None:
        pseudo_filter_policy = None
    else:
        try:
            pseudo_filter_policy = _normalize_pseudo_filter_policy(pseudo_filter_raw)
        except StudentDatasetError as error:
            raise _problem("pseudo_filter is invalid", str(error), "restore the executable Task 17 pseudo_filter policy") from error
    source_root = _path(raw.get("source_root"), base=source.parent, field="source_root", exists=False)
    if not source_root.is_dir():
        raise _problem("source_root is not a directory", f"{source_root} is not an image root", "set source_root to the Task 8 source-image tree")
    seed = _positive(raw.get("seed"), "seed", 42)
    sampling_strategy = raw.get("sampling_strategy", "balanced_50_50")
    if not isinstance(sampling_strategy, str):
        raise _problem("sampling_strategy is invalid", "sampling_strategy must be a named protocol string", "use balanced_50_50 or natural_unresampled")
    expected_filter_policy_evidence = None
    if pseudo_filter_policy is not None and raw.get("filter_calibration") is not None:
        # Matrix Students consume the full sealed Task-14 policy, including
        # PR calibration/artifact digests and every non-toggle guard.
        from fruit_ssod.training.ssod_matrix import SsodMatrixError, effective_filter_policy
        try:
            expected_filter_policy_evidence = effective_filter_policy(raw, source)
        except SsodMatrixError as error:
            raise _problem("matrix filter policy cannot be verified", str(error), "restore the exact calibration PR and aggregate bounds artifacts before Student training") from error
    allow_below_precision_gate = raw.get("allow_below_precision_gate", False)
    if not isinstance(allow_below_precision_gate, bool):
        raise _problem("allow_below_precision_gate is invalid", "the exploratory override must be an explicit boolean", "set it to true only for the customer-authorized v0 run")
    inputs = StudentDatasetInputs(**fields, source_root=source_root, label_budget=_positive(raw.get("label_budget_percent"), "label_budget_percent", 20), seed=seed, human_sample_probability=raw.get("human_sample_probability", 0.5), sampling_strategy=sampling_strategy, pseudo_filter_policy=pseudo_filter_policy, expected_filter_policy_evidence=expected_filter_policy_evidence, expected_teacher_source_model=str(weights.resolve(strict=True)) if mode == "teacher_checkpoint" else None, allow_below_precision_gate=allow_below_precision_gate)
    amp = raw.get("amp", model_payload.get("amp", True))
    if not isinstance(amp, bool):
        raise _problem("amp is invalid", "AMP must be a boolean", "set amp to true or false")
    device = raw.get("device", "cuda:0")
    if not isinstance(device, str) or not device.strip():
        raise _problem("device is invalid", "device must be a nonempty Ultralytics selector", "use cuda:0 or cpu")
    workers = _nonnegative(raw.get("workers"), "workers", 0)
    optimizer = raw.get("optimizer", "auto")
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise _problem("optimizer is invalid", f"received {optimizer!r}", "use auto, SGD, Adam, or AdamW")
    return StudentExperiment(source, name, model, artifact, inputs, seed, _positive(raw.get("image_size", model_payload.get("image_size")), "image_size", 640), amp, _positive(raw.get("batch", model_payload.get("batch")), "batch", 4), _positive(raw.get("epochs"), "epochs", 100), _positive(raw.get("patience"), "patience", 20), _rate(raw.get("learning_rate"), "learning_rate", 0.01), _nonnegative(raw.get("freeze"), "freeze", 0), _save_period(raw.get("save_period")), device, workers, optimizer, _bool(raw.get("cos_lr"), "cos_lr", False), _nonnegative(raw.get("close_mosaic"), "close_mosaic", 10), _fraction(raw.get("mosaic"), "mosaic", 1.0), _fraction(raw.get("mixup"), "mixup", 0.0), policy_values, weights, teacher_config, initialization_evidence)


@dataclass(frozen=True)
class StudentInvocation:
    run_dir: Path
    experiment: StudentExperiment
    dataset: StudentDatasetResult


class UltralyticsStudentExecutor:
    """Delayed real executor; dry-runs never import Ultralytics or CUDA."""
    def __call__(self, invocation: StudentInvocation) -> TrainingExecution:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise _problem("Ultralytics is not installed", "the active Conda environment lacks the locked training dependency", "install the project environment before a real Student smoke run") from error
        try:
            model_yaml = yaml.safe_load(invocation.experiment.model_config.read_text(encoding="utf-8"))
            model = YOLO(str(model_yaml["model"]))
            if invocation.experiment.pretrained_weights is None:
                raise _problem("Student pretrained weights are unavailable", "the validated experiment lost its required initialization path", "reload a configuration with a local shared pretrained checkpoint")
            expected = (invocation.experiment.initialization_evidence or {}).get("student_pretrained_weights")
            if not isinstance(expected, Mapping) or not isinstance(expected.get("sha256"), str) or not isinstance(expected.get("bytes"), int):
                raise _problem("Student pretrained-weight evidence is incomplete", "the validated initialization snapshot lost its immutable byte evidence", "reload the Student configuration before training")
            # Configuration loading may precede a real GPU run by hours.  Hash
            # the exact local file again at the last safe point before loading
            # it, so a replaced checkpoint cannot silently change Student
            # initialization after its Teacher/Student comparison was sealed.
            actual = file_evidence(invocation.experiment.pretrained_weights, description="Student pretrained weights immediately before model load")
            if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                raise _problem("Student pretrained weights changed after configuration validation", "the local checkpoint no longer matches the sealed shared-initialization evidence", "restore the exact audited pretrained weights and reload the experiment")
            model.load(str(invocation.experiment.pretrained_weights))
            # Ultralytics final_eval() strips and rewrites ``best.pt`` after
            # the epoch loop.  Capture the curve-selected checkpoint while it
            # is still the trainer's live EMA serialization, then publish
            # that immutable copy back as the canonical ``best.pt`` after
            # train() returns.  This keeps downstream evidence/evaluation on
            # the conventional path while preventing a post-loop rewrite from
            # silently changing the measured model.
            best_weights = invocation.run_dir / "weights" / "best.pt"
            captured_weights = invocation.run_dir / "weights" / "curve_best_pre_final.pt"
            capture: dict[str, Any] = {}

            def capture_curve_best(trainer: Any) -> None:
                if trainer.best_fitness != trainer.fitness:
                    return
                source = Path(trainer.best)
                if not source.is_file():
                    raise RuntimeError("trainer best checkpoint missing while capturing curve-selected Student weights")
                captured_weights.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, captured_weights)
                capture.update({
                    "epoch": int(trainer.epoch) + 1,
                    "fitness": float(trainer.fitness),
                    "source": file_evidence(source, description="live curve-selected Student checkpoint"),
                    "captured": file_evidence(captured_weights, description="pre-final curve-selected Student checkpoint"),
                })

            model.add_callback("on_model_save", capture_curve_best)
            model.train(data=str(invocation.dataset.dataset_yaml), epochs=invocation.experiment.epochs, patience=invocation.experiment.patience, lr0=invocation.experiment.learning_rate, optimizer=invocation.experiment.optimizer, cos_lr=invocation.experiment.cos_lr, close_mosaic=invocation.experiment.close_mosaic, mosaic=invocation.experiment.mosaic, mixup=invocation.experiment.mixup, freeze=invocation.experiment.freeze, save_period=invocation.experiment.save_period, imgsz=invocation.experiment.image_size, amp=invocation.experiment.amp, batch=invocation.experiment.batch, workers=invocation.experiment.workers, device=invocation.experiment.device, seed=invocation.experiment.seed, project=str(invocation.run_dir.parent), name=invocation.run_dir.name, exist_ok=True, verbose=False, plots=False)
            if not captured_weights.is_file() or not capture:
                raise RuntimeError("no curve-selected Student checkpoint was captured during training")
            shutil.copy2(captured_weights, best_weights)
            # Do not reuse the training model object for the final validation:
            # resumed Ultralytics runs can otherwise validate a different
            # in-memory state from the curve-selected best checkpoint.
            selected = YOLO(str(best_weights))
            validation = selected.val(data=str(invocation.dataset.dataset_yaml), split="val", imgsz=invocation.experiment.image_size, device=invocation.experiment.device, plots=False)
            validate_class_mapping(getattr(selected, "names", None), DEFAULT_CLASS_REGISTRY)
            curves = invocation.run_dir / "results.csv"
            if not curves.is_file():
                raise RuntimeError("results.csv missing")
            metrics = _serialize_metric_object(validation)
            raw = getattr(validation, "results_dict", None)
            if not isinstance(raw, Mapping):
                raise RuntimeError("validation results_dict missing")
            curve_evidence = file_evidence(curves, description="Student training curves")
            return TrainingExecution(metrics, {"results_dict": dict(raw), "speed": dict(getattr(validation, "speed", {}))}, {"results_csv": {"path": str(curves), "sha256": curve_evidence["sha256"]}, "best_weights": file_evidence(best_weights, description="selected Student best checkpoint"), "curve_best_capture": capture})
        except DetectorAdapterError as error:
            raise _problem("trained Student class mapping is incompatible", str(error), "use only the canonical five-class Student dataset") from error
        except SemiSupervisedTrainingError:
            raise
        except Exception as error:
            raise _problem("Student training or validation failed", str(error), "check dataset paths, GPU memory, and the one-epoch smoke command") from error


class StudentTrainingRunner:
    def __init__(self, *, executor: Callable[[StudentInvocation], TrainingExecution] | None = None) -> None:
        self._executor = executor or UltralyticsStudentExecutor()

    def run(self, experiment: StudentExperiment, *, command: Sequence[str], dry_run: bool = False, run_id: str | None = None) -> tuple[RunRecord, Path]:
        if not command:
            raise _problem("exact command is empty", "Student provenance would not be reproducible", "pass the complete train_student command")
        split_fingerprint = split_fingerprint_from_manifest(experiment.dataset_inputs.split_manifest)
        effective_run_id = run_id
        if dry_run:
            # A dry run is retained as useful provenance but must never seize
            # the fixed identifier later used by a real experiment.
            base = run_id or experiment.experiment_name
            # RunRecord intentionally limits IDs to 160 characters for safe
            # Windows artifact paths.  Preserve the disposable prefix and
            # full UUID uniqueness while trimming only the human-readable
            # requested identifier.
            suffix = uuid.uuid4().hex
            # Also leave room for the run's own evidence files under normal
            # Windows MAX_PATH handling.  The RunRecord ceiling alone is not
            # enough when artifact_root itself is deeply nested (as it is in
            # many pytest and managed-workspace paths).
            # ``compose_student_dataset`` atomically creates a sibling such
            # as ``.student_dataset.tmp-XXXXXXXX`` below this run.  Budget
            # that 32-character child suffix as well, and target 220 rather
            # than 260 to leave headroom for its files on legacy Windows APIs.
            max_run_id_length = min(160, 220 - len(str(experiment.artifact_root / "runs")) - 1 - 32)
            fixed_length = len("dry-run-") + len("-") + len(suffix)
            if max_run_id_length < fixed_length:
                raise _problem("artifact_root is too deep for a disposable dry run", f"only {max_run_id_length} run-ID characters remain under the Windows path budget", "shorten artifact_root before creating a dry-run snapshot")
            max_base_length = max_run_id_length - fixed_length
            effective_run_id = f"dry-run-{base[:max_base_length]}-{suffix}"
        provisional = create_run_record(config_snapshot={"experiment_name": experiment.experiment_name, "protocol": "student_ssod_v1", "initialization_policy": dict(experiment.initialization_policy or {}), "dry_run_requested": dry_run}, split_fingerprint=split_fingerprint, command=command, environment=environment_details(probe_cuda=not dry_run), run_id=effective_run_id, status="running")
        run_dir = experiment.artifact_root / "runs" / provisional.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            expected_teacher_run_id = (experiment.initialization_evidence or {}).get("teacher_run_id")
            if not isinstance(expected_teacher_run_id, str) or not expected_teacher_run_id:
                raise _problem("Student Teacher run binding is unavailable", "initialization evidence lacks teacher_run_id", "reload a configuration with a declared Teacher run")
            dataset_inputs = replace(experiment.dataset_inputs, expected_teacher_run_id=expected_teacher_run_id)
            dataset = compose_student_dataset(dataset_inputs, run_dir / "student_dataset")
            record = create_run_record(config_snapshot=experiment.snapshot(dataset), split_fingerprint=split_fingerprint, command=command, environment=provisional.environment, run_id=provisional.run_id, status="running")
            write_run_record(record, run_dir / "run_record.json")
            self._write_static_artifacts(record, run_dir)
            if dry_run:
                terminal = replace(record, status="dry_run")
                write_run_record(terminal, run_dir / "run_record.json", allow_status_update=True)
                return terminal, run_dir
            execution = self._executor(StudentInvocation(run_dir, experiment, dataset))
            if not isinstance(execution, TrainingExecution):
                raise _problem("Student executor returned an unsupported result", "the executor did not return TrainingExecution", "use the project Student executor or conforming test executor")
            metrics = metrics_from_mapping(execution.metrics.mapping())
            checkpoints = {
                name: {
                    "relative_path": (run_dir / "weights" / name).relative_to(run_dir).as_posix(),
                    **file_evidence(run_dir / "weights" / name, description=f"Student checkpoint {name}"),
                }
                for name in ("best.pt", "last.pt")
            }
            (run_dir / "result.json").write_text(json.dumps(metrics.mapping(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (run_dir / "checkpoint_evidence.json").write_text(json.dumps(checkpoints, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            terminal = complete_run_record(record, metrics.mapping())
            write_run_record(terminal, run_dir / "run_record.json", allow_status_update=True)
            return terminal, run_dir
        except Exception as error:
            if (run_dir / "run_record.json").is_file():
                try:
                    # Existing records are the source of immutable config;
                    # only append a terminal diagnostic once.
                    failed = fail_run_record(record, problem="Student training failed", cause=str(error), remediation="inspect run_record.json and sealed Task 8/14/15 artifacts")
                    write_run_record(failed, run_dir / "run_record.json", allow_status_update=True)
                except Exception:
                    pass
            if isinstance(error, (SemiSupervisedTrainingError, StudentDatasetError, ValueError)):
                raise
            raise _problem("Student training failed", repr(error), "inspect run artifacts and correct the underlying issue") from error

    @staticmethod
    def _write_static_artifacts(record: RunRecord, run_dir: Path) -> None:
        """Mirror supervised evidence files for report and reproducibility tooling."""
        serialized = record.mapping()
        try:
            (run_dir / "config_snapshot.json").write_text(json.dumps(serialized["config_snapshot"], ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            (run_dir / "environment.json").write_text(json.dumps(serialized["environment"], ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            (run_dir / "command.txt").write_text(" ".join(record.command) + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as error:
            raise _problem("Student static run evidence cannot be written", str(error), "ensure the run directory is writable before training") from error
