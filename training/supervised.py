"""Configuration-driven supervised YOLO training with auditable artifacts.

Ultralytics is imported only inside the default executor.  Unit and CLI dry
runs therefore remain CPU-safe and network-free, while a real run has one
place where its model-framework interaction is made explicit.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.detection.adapter import DetectorAdapterError, validate_class_mapping
from fruit_ssod.evaluation.detection_metrics import DetectionMetrics, DetectionMetricsError, metrics_from_mapping
from fruit_ssod.training.run_record import (
    RunRecord,
    RunRecordError,
    canonical_snapshot_fingerprint,
    complete_run_record,
    create_run_record,
    fail_run_record,
    read_run_record,
    split_fingerprint_from_manifest,
    write_run_record,
)


class SupervisedTrainingError(RuntimeError):
    """Raised for safe-to-display supervised training configuration/runtime errors."""


def _problem(problem: str, cause: str, remediation: str) -> SupervisedTrainingError:
    return SupervisedTrainingError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _read_yaml_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem(f"{description} cannot be read", str(error), f"provide a readable UTF-8 {description} path") from error
    if not isinstance(payload, Mapping):
        raise _problem(f"{description} is not a YAML object", "the file is empty or has the wrong top-level shape", "use key/value experiment settings")
    return payload


def _normalize_yaml_value(value: Any, *, field: str) -> Any:
    """Return a deterministic JSON-compatible copy of parsed YAML evidence."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _problem(f"{field} has a non-string key", "YAML keys would not have an unambiguous JSON representation", "use string keys only in the model and dataset YAML files")
        return {key: _normalize_yaml_value(item, field=field) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_yaml_value(item, field=field) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise _problem(f"{field} contains NaN or infinity", "non-finite YAML values cannot be preserved in immutable JSON evidence", "use finite numeric configuration values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise _problem(f"{field} is not JSON-compatible", f"YAML parsed an unsupported {type(value).__name__} value", "use mappings, arrays, strings, finite numbers, booleans, and null")


def _yaml_evidence(path: Path, description: str) -> tuple[dict[str, Any], str]:
    """Load a YAML object and keep a normalized content digest for resume safety."""
    payload = _normalize_yaml_value(_read_yaml_mapping(path, description), field=description)
    assert isinstance(payload, dict)  # `_read_yaml_mapping` guarantees an object.
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return payload, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: str, field: str) -> str:
    """Support the same ``${NAME}`` spelling as the typed project config."""
    def replace_variable(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise _problem(f"{field} references unset environment variable {name}", "machine-specific storage was not configured", f"set {name} or replace the variable in the experiment YAML")
        return resolved
    return os.path.expandvars(_ENVIRONMENT_VARIABLE.sub(replace_variable, value))


def _value_path(value: object, *, field: str, base: Path, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} is missing", "the experiment YAML omitted a required path", f"set {field} to a readable path")
    raw = Path(_expand_environment(value, field))
    path = raw if raw.is_absolute() else base / raw
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise _problem(f"{field} cannot be resolved", str(error), f"correct {field} and ensure it exists") from error
    if must_exist and not resolved.is_file():
        raise _problem(f"{field} is not a file", f"{resolved} is missing or a directory", f"set {field} to the required file")
    return resolved


def _positive_int(value: object, field: str, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _problem(f"{field} must be a positive integer", f"received {value!r}", f"set {field} to a positive integer")
    return value


def _nonnegative_int(value: object, field: str, default: int) -> int:
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _problem(f"{field} must be a non-negative integer", f"received {value!r}", f"set {field} to zero or a positive integer")
    return value


def _rate(value: object, field: str, default: float) -> float:
    """Validate a finite Ultralytics rate such as lr0 or mixup."""
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise _problem(f"{field} must be a finite value in [0, 1]", f"received {value!r}", f"set {field} to a finite numeric value in [0, 1]")
    return float(value)


def _device(value: object, default: str = "cuda:0") -> str:
    value = default if value is None else value
    if not isinstance(value, str) or not value.strip():
        raise _problem("device must be a nonempty string", f"received {value!r}", "use cpu, cuda:0, or another valid Ultralytics device selector")
    return value


def _bool(value: object, field: str, default: bool) -> bool:
    value = default if value is None else value
    if not isinstance(value, bool):
        raise _problem(f"{field} must be true or false", f"received {value!r}", f"set {field} to true or false")
    return value


def _dataset_reference(value: object, *, field: str, dataset_root: Path) -> str | list[str]:
    """Resolve each declared dataset partition without requiring images at dry-run time."""
    entries = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else None
    if not entries or any(not isinstance(item, str) or not item.strip() for item in entries):
        raise _problem(f"dataset YAML {field} path is missing or malformed", f"received {value!r}", "set a nonempty path for train, val, and test")
    resolved: list[str] = []
    for entry in entries:
        expanded = _expand_environment(entry, f"dataset YAML {field}")
        candidate = Path(expanded)
        target = candidate if candidate.is_absolute() else dataset_root / candidate
        try:
            resolved.append(str(target.resolve(strict=False)))
        except OSError as error:
            raise _problem(f"dataset YAML {field} path cannot be resolved", str(error), "use a valid local or shared dataset path") from error
    return resolved[0] if isinstance(value, str) else resolved


def _dataset_evidence(path: Path) -> tuple[dict[str, Any], str]:
    """Validate canonical labels and record the fully effective dataset YAML."""
    dataset, digest = _yaml_evidence(path, "dataset YAML")
    expected_names = list(DEFAULT_CLASS_REGISTRY.class_names)
    if dataset.get("names") != expected_names:
        raise _problem("dataset YAML class names do not match the canonical registry", f"expected {expected_names!r}, received {dataset.get('names')!r}", "use Apple, Banana, Orange, Strawberry, Pineapple in IDs 0 through 4")
    root_value = dataset.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise _problem("dataset YAML path is malformed", f"received {root_value!r}", "set path to a nonempty dataset root")
    root_candidate = Path(_expand_environment(root_value, "dataset YAML path"))
    dataset_root = root_candidate if root_candidate.is_absolute() else path.parent / root_candidate
    try:
        dataset_root = dataset_root.resolve(strict=False)
    except OSError as error:
        raise _problem("dataset YAML root cannot be resolved", str(error), "set path to a valid local or shared dataset root") from error
    effective = dict(dataset)
    effective["path"] = str(dataset_root)
    effective["train"] = _dataset_reference(dataset.get("train"), field="train", dataset_root=dataset_root)
    effective["val"] = _dataset_reference(dataset.get("val"), field="val", dataset_root=dataset_root)
    effective["test"] = _dataset_reference(dataset.get("test"), field="test", dataset_root=dataset_root)
    return effective, digest


def file_evidence(path: Path, *, description: str) -> dict[str, Any]:
    """Return immutable evidence for a real, nonempty checkpoint file.

    ``Path.is_file`` follows symbolic links, which is unsuitable for an
    auditable run directory: a later link retarget could silently change the
    model that a record appears to describe.  Inspect the link itself and
    hash the exact bytes before allowing a resume or successful completion.
    """
    try:
        metadata = path.lstat()
    except OSError as error:
        raise _problem(f"{description} cannot be inspected", str(error), f"provide a readable regular {description} file") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise _problem(f"{description} is not a regular file", f"{path} is a link, directory, or special file", "use the actual non-symlink checkpoint file")
    if metadata.st_size <= 0:
        raise _problem(f"{description} is empty", f"{path} has zero bytes", "restore a nonempty checkpoint before resuming or completing the run")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _problem(f"{description} cannot be hashed", str(error), "ensure the checkpoint remains readable and retry") from error
    return {"path": str(path), "bytes": metadata.st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class SupervisedExperiment:
    """Normalized supervised experiment settings before any model is constructed."""

    source_config: Path
    experiment_name: str
    model_config: Path
    dataset_yaml: Path
    split_manifest: Path
    artifact_root: Path
    seed: int
    image_size: int = 640
    amp: bool = True
    batch: int = 4
    epochs: int = 100
    patience: int = 100
    workers: int = 0
    device: str = "cuda:0"
    learning_rate: float = 0.01
    optimizer: str = "auto"
    cos_lr: bool = False
    close_mosaic: int = 10
    mosaic: float = 1.0
    mixup: float = 0.0
    # Ultralytics normally saves only the final checkpoints.  Interrupted
    # Windows GPU jobs need a positive period so an external chunked runner
    # can resume safely after a session/process timeout.
    save_period: int = -1
    label_budget_percent: int | None = None
    matrix_template_id: str | None = None
    pretrained_weights: Path | None = None
    initialization_policy: Mapping[str, str] | None = None

    def snapshot(self) -> dict[str, Any]:
        model_config, model_digest = _yaml_evidence(self.model_config, "model config")
        dataset_config, dataset_digest = _dataset_evidence(self.dataset_yaml)
        snapshot = {
            "experiment_name": self.experiment_name,
            "model_config": str(self.model_config),
            "model_config_sha256": model_digest,
            "model_config_effective": model_config,
            "model_reference": model_config["model"],
            "dataset_yaml": str(self.dataset_yaml),
            "dataset_yaml_sha256": dataset_digest,
            "dataset_yaml_effective": dataset_config,
            "dataset_paths": {key: dataset_config[key] for key in ("train", "val", "test")},
            "split_manifest": str(self.split_manifest),
            "artifact_root": str(self.artifact_root),
            "seed": self.seed,
            "image_size": self.image_size,
            "amp": self.amp,
            "batch": self.batch,
            "epochs": self.epochs,
            "patience": self.patience,
            "workers": self.workers,
            "device": self.device,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "save_period": self.save_period,
            "canonical_classes": list(DEFAULT_CLASS_REGISTRY.class_names),
        }
        # Generic supervised smoke experiments need not participate in the
        # Task 12 label-budget matrix.  When they do, freeze both protocol
        # facts so aggregation can never infer a budget from a run name.
        if self.label_budget_percent is not None:
            snapshot["label_budget_percent"] = self.label_budget_percent
        if self.matrix_template_id is not None:
            snapshot["matrix_template_id"] = self.matrix_template_id
        if self.pretrained_weights is not None:
            snapshot["pretrained_weights"] = file_evidence(self.pretrained_weights, description="supervised pretrained weights")
        if self.initialization_policy is not None:
            snapshot["initialization_policy"] = dict(self.initialization_policy)
        return snapshot


def load_supervised_experiment(path: Path | str) -> SupervisedExperiment:
    """Read one experiment YAML while preserving the canonical class contract."""
    source = Path(path)
    try:
        source = source.resolve(strict=True)
    except OSError as error:
        raise _problem("experiment config cannot be resolved", str(error), "pass an existing supervised experiment YAML") from error
    raw = _read_yaml_mapping(source, "supervised experiment config")
    name = raw.get("experiment_name")
    if not isinstance(name, str) or not name.strip():
        raise _problem("experiment_name is missing", "the experiment cannot be identified in saved runs", "set a nonempty experiment_name")
    model_config = _value_path(raw.get("model_config"), field="model_config", base=source.parent)
    dataset_yaml = _value_path(raw.get("dataset_yaml"), field="dataset_yaml", base=source.parent)
    split_manifest = _value_path(raw.get("split_manifest"), field="split_manifest", base=source.parent)
    model, _ = _yaml_evidence(model_config, "model config")
    model_reference = model.get("model")
    if not isinstance(model_reference, str) or not model_reference.strip():
        raise _problem("model config has no model", "the model YAML does not identify a YOLO architecture or checkpoint", "set model to a local YOLO YAML or checkpoint path")
    configured_names = model.get("names")
    if configured_names is not None:
        expected = list(DEFAULT_CLASS_REGISTRY.class_names)
        if configured_names != expected:
            raise _problem("model config class names do not match the canonical registry", f"expected {expected!r}, received {configured_names!r}", "use Apple, Banana, Orange, Strawberry, Pineapple in IDs 0 through 4")
    # Validate at config-load time, including a CLI dry-run, so a malformed
    # dataset can never produce a seemingly valid provenance record.
    _dataset_evidence(dataset_yaml)
    artifact_value = raw.get("artifact_root")
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        raise _problem("artifact_root is missing", "training needs an explicit non-repository artifact location", "set artifact_root to the configured shared or local artifact directory")
    artifact = Path(_expand_environment(artifact_value, "artifact_root"))
    if not artifact.is_absolute():
        artifact = source.parent / artifact
    try:
        artifact = artifact.resolve(strict=False)
    except OSError as error:
        raise _problem("artifact_root cannot be resolved", str(error), "set artifact_root to a writable local/shared directory") from error
    label_budget_raw = raw.get("label_budget_percent")
    label_budget: int | None = None
    if label_budget_raw is not None:
        label_budget = _positive_int(label_budget_raw, "label_budget_percent", 1)
        if label_budget > 100:
            raise _problem("label_budget_percent exceeds 100", f"received {label_budget}", "use an integer percentage from 1 through 100")
    template_id_raw = raw.get("template_id")
    if template_id_raw is not None and (not isinstance(template_id_raw, str) or not template_id_raw.strip()):
        raise _problem("template_id is malformed", f"received {template_id_raw!r}", "use a nonempty template identifier or omit it for an ad-hoc experiment")
    if template_id_raw is not None and label_budget is None:
        raise _problem("template_id has no label budget", "a matrix template was declared without label_budget_percent", "add the exact budget or omit template_id for a non-matrix run")
    weights_raw = raw.get("pretrained_weights")
    pretrained_weights = None if weights_raw is None else _value_path(weights_raw, field="pretrained_weights", base=source.parent)
    policy_raw = raw.get("initialization_policy")
    initialization_policy: Mapping[str, str] | None = None
    if policy_raw is not None:
        required_policy = {"policy_id", "model_initialization", "comparison_group"}
        if not isinstance(policy_raw, Mapping) or set(policy_raw) != required_policy or any(not isinstance(value, str) or not value.strip() for value in policy_raw.values()):
            raise _problem("initialization_policy is incomplete", "a comparable Teacher must declare the immutable shared initialization protocol", "set policy_id, model_initialization, and comparison_group")
        if policy_raw["model_initialization"] != "shared_pretrained_weights" or pretrained_weights is None:
            raise _problem("initialization_policy cannot be satisfied", "shared_pretrained_weights requires a readable local pretrained_weights checkpoint", "configure the approved local checkpoint; this workflow never downloads weights")
        initialization_policy = {key: policy_raw[key] for key in sorted(policy_raw)}
    elif pretrained_weights is not None:
        raise _problem("pretrained_weights has no initialization policy", "the Teacher checkpoint would not be scientifically comparable", "add the matching shared_pretrained_weights initialization_policy")
    save_period_raw = raw.get("save_period", -1)
    if isinstance(save_period_raw, bool) or not isinstance(save_period_raw, int) or save_period_raw < -1:
        raise _problem("save_period must be -1 or a non-negative integer", f"received {save_period_raw!r}", "use -1 for final-only checkpoints or a positive period for resumable training")
    return SupervisedExperiment(
        source_config=source, experiment_name=name, model_config=model_config, dataset_yaml=dataset_yaml,
        split_manifest=split_manifest, artifact_root=artifact,
        seed=_positive_int(raw.get("seed"), "seed", 42),
        image_size=_positive_int(raw.get("image_size", model.get("image_size")), "image_size", 640),
        amp=_bool(raw.get("amp", model.get("amp")), "amp", True),
        batch=_positive_int(raw.get("batch", model.get("batch")), "batch", 4),
        epochs=_positive_int(raw.get("epochs"), "epochs", 100),
        patience=_positive_int(raw.get("patience"), "patience", 100),
        workers=_nonnegative_int(raw.get("workers"), "workers", 0),
        device=_device(raw.get("device")),
        learning_rate=_rate(raw.get("learning_rate"), "learning_rate", 0.01),
        optimizer=raw.get("optimizer", "auto") if isinstance(raw.get("optimizer", "auto"), str) and str(raw.get("optimizer", "auto")).strip() else "auto",
        cos_lr=_bool(raw.get("cos_lr"), "cos_lr", False),
        close_mosaic=_nonnegative_int(raw.get("close_mosaic"), "close_mosaic", 10),
        mosaic=_rate(raw.get("mosaic"), "mosaic", 1.0),
        mixup=_rate(raw.get("mixup"), "mixup", 0.0),
        save_period=save_period_raw,
        label_budget_percent=label_budget, matrix_template_id=template_id_raw, pretrained_weights=pretrained_weights,
        initialization_policy=initialization_policy,
    )


def environment_details(*, probe_cuda: bool = True) -> dict[str, Any]:
    """Capture framework and CUDA facts without making either a hard import dependency."""
    details: dict[str, Any] = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform()}
    for distribution in ("ultralytics", "torch"):
        try:
            details[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            details[distribution] = None
    if not probe_cuda:
        details["cuda_probe"] = "skipped for dry-run; no GPU context was requested"
        return details
    try:
        import torch
        details["cuda_available"] = bool(torch.cuda.is_available())
        details["cuda_version"] = torch.version.cuda
        details["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:  # Environment capture must not prevent a CPU dry run.
        details["cuda_probe_error"] = str(error)
    return details


@dataclass(frozen=True)
class TrainingInvocation:
    """All side-effecting executor inputs, allowing a fake executor in tests."""

    run_dir: Path
    experiment: SupervisedExperiment
    resume: Path | None = None


@dataclass(frozen=True)
class TrainingExecution:
    """Normalized output expected from an executor after a real training attempt."""

    metrics: DetectionMetrics
    raw_validation: Mapping[str, Any]
    curves: Mapping[str, Any]


def _serialize_metric_object(metrics: Any) -> DetectionMetrics:
    """Extract the documented Ultralytics detection metric attributes defensively."""
    box = getattr(metrics, "box", None)
    try:
        # The framework's maps field is mAP50-95, not AP50.  The report
        # contract explicitly names AP50, so use column zero of all_ap (or
        # the direct per-class ap50 property) and never silently substitute
        # the wrong metric.
        all_ap = getattr(box, "all_ap", None)
        if all_ap is not None:
            rows = all_ap.tolist() if hasattr(all_ap, "tolist") else list(all_ap)
            per_class_values = [row[0] for row in rows]
        else:
            direct_ap50 = getattr(box, "ap50", None)
            if direct_ap50 is None:
                raise ValueError("Ultralytics metric object exposes no per-class AP50 values")
            per_class_values = list(direct_ap50.tolist() if hasattr(direct_ap50, "tolist") else direct_ap50)
        if len(per_class_values) != len(DEFAULT_CLASS_REGISTRY.classes):
            raise ValueError("per-class metric count differs from canonical classes")
        precision = float(getattr(box, "mp"))
        recall = float(getattr(box, "mr"))
        return DetectionMetrics(
            map50=float(getattr(box, "map50")), map50_95=float(getattr(box, "map")), precision=precision,
            recall=recall, f1=(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            per_class_ap50={index: float(value) for index, value in enumerate(per_class_values)},
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _problem("Ultralytics validation output is incompatible", str(error), "use a supported Ultralytics detection version with five-class box metrics") from error


class UltralyticsSupervisedExecutor:
    """Delayed Ultralytics implementation used only by a non-dry-run command."""

    def __call__(self, invocation: TrainingInvocation) -> TrainingExecution:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise _problem("Ultralytics is not installed", "the active Conda environment lacks the training dependency", "install the locked project dependencies in fruit-ssod") from error
        model_settings = _read_yaml_mapping(invocation.experiment.model_config, "model config")
        model = YOLO(str(model_settings["model"]))
        if invocation.experiment.pretrained_weights is not None:
            try:
                model.load(str(invocation.experiment.pretrained_weights))
            except Exception as error:
                raise _problem("supervised pretrained weights cannot be loaded", str(error), "restore the declared local checkpoint; no weights are downloaded automatically") from error
        # Initial architecture may not yet expose labels; after training val must.
        kwargs: dict[str, Any] = {
            "data": str(invocation.experiment.dataset_yaml), "epochs": invocation.experiment.epochs,
            "imgsz": invocation.experiment.image_size, "amp": invocation.experiment.amp,
            "batch": invocation.experiment.batch, "device": invocation.experiment.device,
            "seed": invocation.experiment.seed, "patience": invocation.experiment.patience,
            "workers": invocation.experiment.workers,
            "lr0": invocation.experiment.learning_rate,
            "optimizer": invocation.experiment.optimizer,
            "cos_lr": invocation.experiment.cos_lr,
            "close_mosaic": invocation.experiment.close_mosaic,
            "mosaic": invocation.experiment.mosaic,
            "mixup": invocation.experiment.mixup,
            "save_period": invocation.experiment.save_period,
            "project": str(invocation.run_dir.parent),
            "name": invocation.run_dir.name, "exist_ok": True, "verbose": False, "plots": False,
        }
        if invocation.resume is not None:
            kwargs["resume"] = str(invocation.resume)
        try:
            model.train(**kwargs)
            # Ultralytics otherwise writes its explicit post-training
            # validation plots under the caller's ``runs/detect`` directory.
            # Keep every generated validation artifact beside the immutable
            # run evidence instead.
            validation = model.val(
                data=str(invocation.experiment.dataset_yaml),
                split="val",
                imgsz=invocation.experiment.image_size,
                device=invocation.experiment.device,
                project=str(invocation.run_dir),
                name="post_train_validation",
                exist_ok=True,
                plots=False,
            )
            validate_class_mapping(getattr(model, "names", None), DEFAULT_CLASS_REGISTRY)
        except DetectorAdapterError as error:
            raise _problem("trained model class mapping is incompatible", str(error), "train only with the canonical five-class dataset YAML") from error
        except Exception as error:
            raise _problem("Ultralytics training or validation failed", str(error), "check dataset YAML, GPU memory, paths, and resume checkpoint") from error
        metrics = _serialize_metric_object(validation)
        curves_path = invocation.run_dir / "results.csv"
        if not curves_path.is_file():
            raise _problem("training curve evidence is missing", f"Ultralytics did not produce {curves_path}", "retain results.csv in the run directory before a run can be marked complete")
        try:
            curve_text = curves_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise _problem("training curve evidence cannot be read", str(error), "ensure results.csv is readable UTF-8 evidence in the run directory") from error
        raw_results = getattr(validation, "results_dict", None)
        if not isinstance(raw_results, Mapping):
            raise _problem("raw validation payload is missing", "Ultralytics returned no results_dict mapping", "use a supported Ultralytics detection version that exposes raw validation results")
        raw_validation = {
            "results_dict": _normalize_yaml_value(raw_results, field="raw validation payload"),
            "speed": _normalize_yaml_value(getattr(validation, "speed", {}), field="raw validation payload"),
        }
        curves: Mapping[str, Any] = {
            "results_csv": {
                "path": str(curves_path),
                "sha256": hashlib.sha256(curve_text.encode("utf-8")).hexdigest(),
                "content": curve_text,
            }
        }
        return TrainingExecution(metrics=metrics, raw_validation=raw_validation, curves=curves)


class SupervisedTrainingRunner:
    """Creates evidence first, then invokes an injected real training executor."""

    def __init__(self, *, executor: Callable[[TrainingInvocation], TrainingExecution] | None = None) -> None:
        self._executor = executor or UltralyticsSupervisedExecutor()

    @staticmethod
    def _run_dir(experiment: SupervisedExperiment, run_id: str) -> Path:
        return experiment.artifact_root / "runs" / run_id

    def run(self, experiment: SupervisedExperiment, *, command: Sequence[str], dry_run: bool = False, resume: Path | None = None, run_id: str | None = None) -> tuple[RunRecord, Path]:
        """Run or prepare a training invocation without exposing test labels to training."""
        if not command:
            raise _problem("exact command is empty", "the invocation would not be reproducible", "pass sys.argv including train_supervised arguments")
        split_fingerprint = split_fingerprint_from_manifest(experiment.split_manifest)
        is_resume = resume is not None
        if resume is not None:
            # Do not resolve before validating the path shape: resolve follows
            # a link and would make a symlink checkpoint look legitimate.
            original_resume = Path(resume)
            if original_resume.name != "last.pt" or original_resume.parent.name != "weights":
                raise _problem("resume checkpoint has an unsafe location", f"{original_resume} is not exactly <run>/weights/last.pt", "pass the prior run's weights/last.pt checkpoint")
            checkpoint_evidence = file_evidence(original_resume, description="resume checkpoint")
            try:
                resume = original_resume.resolve(strict=True)
            except OSError as error:
                raise _problem("resume checkpoint cannot be resolved", str(error), "pass a readable prior weights/last.pt checkpoint") from error
            inferred_dir = resume.parent.parent
            record_path = inferred_dir / "run_record.json"
            try:
                record = read_run_record(record_path)
            except OSError as error:
                raise _problem("resume run record cannot be read", str(error), "restore the prior run_record.json beside weights/last.pt") from error
            if record.status != "running":
                raise _problem("resume run is not active", f"run {record.run_id} is {record.status}", "resume only an interrupted running run or start a new run")
            if (
                record.split_fingerprint != split_fingerprint
                or canonical_snapshot_fingerprint(record.config_snapshot) != canonical_snapshot_fingerprint(experiment.snapshot())
            ):
                raise _problem("resume provenance does not match the requested experiment", "the split fingerprint or configuration differs", "use the original experiment config and split manifest or start a new run")
            run_dir = inferred_dir
        else:
            # A dry run starts as `running` too: if writing any required
            # provenance evidence fails, it must become an auditable failed
            # record instead of a terminal dry_run with missing artifacts.
            record = create_run_record(config_snapshot=experiment.snapshot(), split_fingerprint=split_fingerprint, command=command, environment=environment_details(probe_cuda=not dry_run), run_id=run_id, status="running")
            run_dir = self._run_dir(experiment, record.run_id)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise _problem("run artifact directory already exists", f"{run_dir} would be reused", "choose a new generated run ID or use --resume with its last.pt") from error
            except OSError as error:
                raise _problem("run artifact directory cannot be created", str(error), "choose a writable configured artifact_root") from error
            write_run_record(record, run_dir / "run_record.json")
        try:
            if is_resume:
                # The original command/configuration are immutable run
                # provenance.  Each retry is separately appended instead of
                # overwriting command.txt or the original snapshot.
                self._append_resume_history(run_dir, command=command, checkpoint=checkpoint_evidence)
            else:
                self._write_static_artifacts(record, run_dir)
            if dry_run:
                completed_dry_run = replace(record, status="dry_run")
                write_run_record(completed_dry_run, run_dir / "run_record.json", allow_status_update=True)
                return completed_dry_run, run_dir
            execution = self._executor(TrainingInvocation(run_dir=run_dir, experiment=experiment, resume=resume))
            if not isinstance(execution, TrainingExecution):
                raise _problem("training executor returned an unsupported result", "the executor did not return TrainingExecution", "use the project Ultralytics executor or a conforming injected executor")
            weights = run_dir / "weights"
            checkpoints = {}
            for name in ("best.pt", "last.pt"):
                evidence = file_evidence(weights / name, description=f"required training checkpoint {name}")
                # The run folder may be moved or copied for review.  Bind the
                # evidence to a stable, portable artifact-relative location
                # rather than treating the original absolute workstation path
                # as the identity of the checkpoint.
                evidence["relative_path"] = (weights / name).relative_to(run_dir).as_posix()
                checkpoints[name] = evidence
            if not isinstance(execution.raw_validation, Mapping) or not execution.raw_validation:
                raise _problem("raw validation payload is missing", "the executor returned no nonempty validation evidence", "return the raw JSON-safe validation payload before completing the run")
            if not isinstance(execution.curves, Mapping) or not execution.curves:
                raise _problem("training curve evidence is missing", "the executor returned no nonempty curve evidence", "return results.csv evidence or structured per-epoch curves before completing the run")
            if not isinstance(execution.metrics, DetectionMetrics):
                raise _problem("training metrics are not DetectionMetrics", f"received {type(execution.metrics).__name__}", "return a fully validated five-class DetectionMetrics object")
            try:
                # Decode the serialized form as well as checking its class.
                # This establishes the canonical IDs/finite bounds immediately
                # before the lifecycle transition to `complete`.
                validated_metrics = metrics_from_mapping(execution.metrics.mapping())
            except DetectionMetricsError as error:
                raise _problem("training metrics violate the canonical contract", str(error), "return AP50 for exactly canonical class IDs 0 through 4") from error
            self._write_json(run_dir / "validation_raw.json", dict(execution.raw_validation))
            self._write_json(run_dir / "training_curves.json", dict(execution.curves))
            self._write_json(run_dir / "checkpoint_evidence.json", checkpoints)
            self._write_json(run_dir / "result.json", validated_metrics.mapping())
            completed = complete_run_record(record, validated_metrics.mapping())
            write_run_record(completed, run_dir / "run_record.json", allow_status_update=True)
            return completed, run_dir
        except Exception as error:
            if isinstance(error, SupervisedTrainingError):
                problem, cause = "supervised training failed", str(error)
            else:
                problem, cause = "supervised training failed", repr(error)
            failed = fail_run_record(record, problem=problem, cause=cause, remediation="inspect run_record.json, dataset paths, GPU memory, and the saved command before retrying")
            write_run_record(failed, run_dir / "run_record.json", allow_status_update=True)
            if isinstance(error, SupervisedTrainingError):
                raise
            raise _problem(problem, cause, "inspect run_record.json and retry only after correcting the underlying issue") from error

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        try:
            path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as error:
            raise _problem("run artifact cannot be serialized", str(error), "use JSON-safe metrics and ensure the run directory is writable") from error

    def _write_static_artifacts(self, record: RunRecord, run_dir: Path) -> None:
        """Save independent files so report tooling never has to parse console output."""
        serialized = record.mapping()
        self._write_json(run_dir / "config_snapshot.json", serialized["config_snapshot"])
        self._write_json(run_dir / "environment.json", serialized["environment"])
        try:
            (run_dir / "command.txt").write_text(" ".join(record.command) + "\n", encoding="utf-8")
        except OSError as error:
            raise _problem("exact command cannot be saved", str(error), "ensure the run directory is writable") from error

    @staticmethod
    def _append_resume_history(run_dir: Path, *, command: Sequence[str], checkpoint: Mapping[str, Any]) -> None:
        """Append one JSONL retry event without modifying prior provenance."""
        history = run_dir / "resume_history.jsonl"
        event = {"command": list(command), "checkpoint": dict(checkpoint)}
        try:
            encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            with history.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as error:
            raise _problem("resume history cannot be appended", str(error), "ensure the interrupted run directory is writable before retrying") from error
