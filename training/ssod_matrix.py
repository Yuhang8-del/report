"""Fixed Task 17 SSOD/ablation matrix and conservative resume checks.

This module schedules no GPU work.  It establishes that the committed matrix
is comparable before the PowerShell launcher can request a device, and it
answers a deliberately narrow resume question: may this exact config and Task
8 split reuse an existing *complete* Student record?  Any unreadable or
incomplete evidence is queued again rather than silently skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.pseudo.thresholds import PerClassThresholds, ThresholdSelectionError, select_per_class_thresholds
from fruit_ssod.training.run_record import RunRecordError, read_run_record, split_fingerprint_from_manifest
from fruit_ssod.training.student_dataset import StudentDatasetError, _normalize_pseudo_filter_policy


class SsodMatrixError(ValueError):
    """Raised when a Task 17 matrix would not be a fair comparison."""


def _problem(problem: str, cause: str, remediation: str) -> SsodMatrixError:
    return SsodMatrixError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


@dataclass(frozen=True)
class SsodMatrixEntry:
    filename: str
    role: str
    seed: int

    @property
    def experiment_name(self) -> str:
        return Path(self.filename).stem


# Keep published order stable: baseline, replicated main method, then the
# single-seed sensitivity studies.
SSOD_EXPERIMENT_MATRIX: tuple[SsodMatrixEntry, ...] = (
    SsodMatrixEntry("ssod_global_seed42.yaml", "global_baseline", 42),
    SsodMatrixEntry("ssod_trust_seed42.yaml", "trust_main", 42),
    SsodMatrixEntry("ssod_trust_seed3407.yaml", "trust_main", 3407),
    SsodMatrixEntry("ssod_trust_seed2026.yaml", "trust_main", 2026),
    SsodMatrixEntry("ablation_no_class_threshold.yaml", "ablation_no_class_threshold", 42),
    SsodMatrixEntry("ablation_no_view_consistency.yaml", "ablation_no_view_consistency", 42),
    SsodMatrixEntry("ablation_no_size_filter.yaml", "ablation_no_size_filter", 42),
    SsodMatrixEntry("ablation_no_human_resampling.yaml", "ablation_no_human_resampling", 42),
)

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_COMMON_FIELDS = (
    "model_config", "pretrained_weights", "split_manifest", "human_images",
    "human_labels", "validation_labels", "unlabeled_manifest", "label_budget_percent",
    "image_size", "amp", "batch", "epochs", "evaluation_protocol", "source_root",
    "filter_calibration",
)
_POLICY_COMPARISON_FIELDS = ("policy_id", "model_initialization", "comparison_group")
_FILTER_CALIBRATION_KEYS = frozenset({
    "global_confidence", "cross_view_iou", "min_pixels_at_640", "max_area_fraction",
    "min_aspect_ratio", "max_aspect_ratio", "max_boxes_per_image", "nms_iou",
    "target_precision", "threshold_minimum", "threshold_maximum", "validation_pr",
    "aspect_ratio_bounds",
})
_FILTER_EXPECTATIONS: Mapping[str, Mapping[str, object]] = {
    "global_baseline": {"policy_id": "global_threshold_v1", "use_per_class_thresholds": False, "require_view_consistency": False, "require_size_filter": False, "sampling_strategy": "balanced_50_50"},
    "trust_main": {"policy_id": "trust_filter_v1", "use_per_class_thresholds": True, "require_view_consistency": True, "require_size_filter": True, "sampling_strategy": "balanced_50_50"},
    "ablation_no_class_threshold": {"policy_id": "trust_without_class_threshold_v1", "use_per_class_thresholds": False, "require_view_consistency": True, "require_size_filter": True, "sampling_strategy": "balanced_50_50"},
    "ablation_no_view_consistency": {"policy_id": "trust_without_view_consistency_v1", "use_per_class_thresholds": True, "require_view_consistency": False, "require_size_filter": True, "sampling_strategy": "balanced_50_50"},
    "ablation_no_size_filter": {"policy_id": "trust_without_size_filter_v1", "use_per_class_thresholds": True, "require_view_consistency": True, "require_size_filter": False, "sampling_strategy": "balanced_50_50"},
    "ablation_no_human_resampling": {"policy_id": "trust_filter_v1", "use_per_class_thresholds": True, "require_view_consistency": True, "require_size_filter": True, "sampling_strategy": "natural_unresampled"},
}


def matrix_entries() -> tuple[SsodMatrixEntry, ...]:
    return SSOD_EXPERIMENT_MATRIX


def _read_yaml(path: Path, description: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise _problem(f"{description} cannot be read", str(error), "restore the checked-in UTF-8 YAML source") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _problem(f"{description} is not a string-keyed object", "the YAML is empty or malformed", "use a top-level YAML object")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _problem("matrix configuration cannot be hashed", str(error), "restore the checked-in configuration") from error


def _resolve(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _problem(f"{field} is invalid", "a required path is missing", "restore the canonical SSOD matrix YAML")

    def expand(match: re.Match[str]) -> str:
        name = match.group(1)
        item = os.environ.get(name)
        if not item:
            raise _problem(f"{field} cannot be resolved", f"{name} is not set", f"set {name} before asking the launcher to resume a run")
        return item

    raw = Path(_VARIABLE.sub(expand, value))
    return (raw if raw.is_absolute() else base / raw).resolve(strict=False)


def _model_evidence(config: Mapping[str, Any], path: Path) -> tuple[str, Mapping[str, Any]]:
    model_path = _resolve(config.get("model_config"), base=path.parent, field="model_config")
    model = _read_yaml(model_path, "matrix model configuration")
    if model.get("names") != list(DEFAULT_CLASS_REGISTRY.class_names) or model.get("image_size") != 640 or model.get("model") != "yolov8s.yaml":
        raise _problem("matrix model violates the common detector contract", "model/classes/image size differ from the five-class YOLOv8s-640 protocol", "use configs/models/yolov8s_640.yaml unchanged")
    return _sha256(model_path), model


def _number(value: object, *, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _problem("matrix filter calibration is invalid", f"{field} is not numeric", "restore the canonical filter_calibration values")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")} or (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise _problem("matrix filter calibration is invalid", f"{field}={value!r} is outside its allowed range", "restore the canonical filter_calibration values")
    return result


def _filter_calibration(config: Mapping[str, Any]) -> dict[str, object]:
    """Strictly parse all non-toggle Task-14 parameters.

    The three gates in ``pseudo_filter`` are the only SSOD filter factors
    allowed to vary in named ablations.  Everything else, including the
    validation PR source, is declared here rather than being an unrecorded
    CLI default.
    """
    value = config.get("filter_calibration")
    if not isinstance(value, Mapping) or set(value) != _FILTER_CALIBRATION_KEYS:
        raise _problem("matrix filter_calibration is incomplete", "the full effective Task 14 configuration is not declared", "restore every canonical filter_calibration field")
    normalized: dict[str, object] = {}
    for field in ("global_confidence", "cross_view_iou", "max_area_fraction", "nms_iou", "target_precision", "threshold_minimum", "threshold_maximum"):
        normalized[field] = _number(value[field], field=field, minimum=0.0, maximum=1.0)
    for field in ("min_pixels_at_640", "min_aspect_ratio", "max_aspect_ratio"):
        normalized[field] = _number(value[field], field=field, minimum=0.0)
        if normalized[field] == 0.0:
            raise _problem("matrix filter calibration is invalid", f"{field} must be positive", "restore the canonical filter_calibration values")
    max_boxes = value["max_boxes_per_image"]
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes <= 0:
        raise _problem("matrix filter calibration is invalid", "max_boxes_per_image must be a positive integer", "restore the canonical filter_calibration values")
    normalized["max_boxes_per_image"] = max_boxes
    if normalized["min_aspect_ratio"] > normalized["max_aspect_ratio"] or normalized["threshold_minimum"] > normalized["threshold_maximum"]:
        raise _problem("matrix filter calibration is invalid", "a declared lower bound exceeds its upper bound", "restore the canonical filter_calibration bounds")
    for field in ("validation_pr", "aspect_ratio_bounds"):
        source = value[field]
        if not isinstance(source, str) or not source.strip():
            raise _problem("matrix filter calibration is invalid", f"{field} is not a readable source reference", "restore the canonical calibration artifact path")
        normalized[field] = source
    return normalized


def effective_filter_policy(config: Mapping[str, Any], path: Path) -> dict[str, object]:
    """Build the byte-bound Task-14 policy expected for one matrix entry.

    This is intentionally richer than the four executable gate switches.  It
    seals every effective numeric guard plus the exact validation-PR and
    aggregate-bounds inputs from which confidence calibration was derived.
    """
    try:
        gates = _normalize_pseudo_filter_policy(config["pseudo_filter"])
    except (KeyError, StudentDatasetError) as error:
        raise _problem("matrix pseudo_filter is not executable", str(error), "restore the exact declared global/Trust/ablation policy") from error
    calibration = _filter_calibration(config)
    sources: dict[str, dict[str, str]] = {}
    for field in ("validation_pr", "aspect_ratio_bounds"):
        source = _resolve(calibration[field], base=path.parent, field=f"filter_calibration.{field}")
        if not source.is_file():
            raise _problem("matrix calibration artifact is missing", f"{field} is not a file at {source}", "restore the sealed calibration artifact before preparing or resuming the matrix")
        sources[field] = {"path": str(source), "sha256": _sha256(source)}
    if gates["policy_id"] == "global_threshold_v1":
        thresholds = PerClassThresholds(
            {class_id: float(calibration["global_confidence"]) for class_id in range(5)},
            minimum=0.0,
            maximum=1.0,
        )
    else:
        try:
            validation_payload = json.loads(Path(sources["validation_pr"]["path"]).read_text(encoding="utf-8"))
            records = validation_payload.get("records") if isinstance(validation_payload, Mapping) else None
            if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
                raise ValueError("validation PR source has no records array")
            thresholds = select_per_class_thresholds(
                records,
                target_precision=float(calibration["target_precision"]),
                minimum=float(calibration["threshold_minimum"]),
                maximum=float(calibration["threshold_maximum"]),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ThresholdSelectionError, ValueError) as error:
            raise _problem("matrix validation PR cannot calibrate thresholds", str(error), "restore the sealed validation-only PR export used by every comparison") from error
    return {
        **gates,
        "filter_config": {field: calibration[field] for field in (
            "global_confidence", "cross_view_iou", "min_pixels_at_640", "max_area_fraction",
            "min_aspect_ratio", "max_aspect_ratio", "max_boxes_per_image", "nms_iou",
        )},
        "threshold_calibration": {
            "target_precision": calibration["target_precision"],
            "minimum": calibration["threshold_minimum"],
            "maximum": calibration["threshold_maximum"],
            "validation_pr": sources["validation_pr"],
        },
        "aspect_ratio_bounds": sources["aspect_ratio_bounds"],
        "resolved_thresholds": thresholds.mapping(),
    }


def load_effective_filter_policy(path: Path | str) -> dict[str, object]:
    """Load one matrix config's complete, byte-bound Task-14 policy.

    Task-14's CLI uses this helper so it cannot quietly fall back to its own
    default guardrails while publishing an artifact consumed by Task 17.
    """
    config_path = Path(path).resolve(strict=True)
    config = _read_yaml(config_path, f"SSOD matrix config {config_path.name}")
    return effective_filter_policy(config, config_path)


def _teacher_evidence(config: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    policy = config.get("initialization_policy")
    if not isinstance(policy, Mapping) or set(policy) != {"policy_id", "model_initialization", "comparison_group", "teacher_experiment_config", "teacher_run_id"}:
        raise _problem("matrix initialization policy is incomplete", "Student/Teacher initialization cannot be compared", "restore all five initialization_policy fields")
    if policy.get("model_initialization") != "shared_pretrained_weights":
        raise _problem("matrix initialization policy is unsupported", "comparison would not start from shared local weights", "use shared_pretrained_weights")
    teacher_path = _resolve(policy.get("teacher_experiment_config"), base=path.parent, field="teacher_experiment_config")
    teacher = _read_yaml(teacher_path, "matrix Teacher configuration")
    if teacher.get("experiment_name") != policy.get("teacher_run_id"):
        raise _problem("Teacher run ID is not bound", "initialization_policy names a different Teacher config", "set teacher_run_id to the Teacher experiment_name")
    if teacher.get("model_config") != config.get("model_config") or teacher.get("pretrained_weights") != config.get("pretrained_weights"):
        raise _problem("Teacher and Student do not share model initialization", "their model config or local pretrained-weight reference differs", "use the same YOLOv8s config and FRUIT_SSOD_PRETRAINED_WEIGHTS")
    teacher_policy = teacher.get("initialization_policy")
    if not isinstance(teacher_policy, Mapping) or any(teacher_policy.get(key) != policy.get(key) for key in _POLICY_COMPARISON_FIELDS):
        raise _problem("Teacher and Student policy differs", "policy_id/model_initialization/comparison_group are not identical", "restore the common initialization policy")
    return {key: policy[key] for key in _POLICY_COMPARISON_FIELDS}


def _validate_entry(entry: SsodMatrixEntry, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_yaml(path, f"SSOD matrix config {path.name}")
    if config.get("experiment_name") != entry.experiment_name or config.get("matrix_role") != entry.role or config.get("seed") != entry.seed:
        raise _problem("matrix entry identity is inconsistent", f"{path.name} does not bind its filename, role, and seed", "restore the canonical experiment_name, matrix_role, and seed")
    if config.get("label_budget_percent") != 20 or config.get("human_sample_probability") != 0.5:
        raise _problem("matrix label budget or human probability differs", "all main and ablation comparisons start from the sealed 20% budget", "use label_budget_percent: 20 and human_sample_probability: 0.5")
    if config.get("evaluation_protocol") != {"protocol_id": "fixed_primary_test_v1", "split": "test", "dataset_role": "primary_fixed_test"}:
        raise _problem("matrix evaluation protocol differs", "held-out primary-test evidence would not be comparable", "restore fixed_primary_test_v1 with split test")
    filter_config = config.get("pseudo_filter")
    expected = _FILTER_EXPECTATIONS[entry.role]
    try:
        parsed_filter = _normalize_pseudo_filter_policy(filter_config)  # type: ignore[arg-type]
    except StudentDatasetError as error:
        raise _problem("matrix pseudo_filter is not executable", str(error), "restore the exact declared global/Trust/ablation policy") from error
    if any(parsed_filter.get(key) != value for key, value in expected.items() if key != "sampling_strategy") or config.get("sampling_strategy") != expected["sampling_strategy"]:
        raise _problem("matrix ablation does not change exactly its named factor", f"role {entry.role!r} has incompatible pseudo-filter or sampling settings", "restore the canonical one-factor ablation settings")
    _filter_calibration(config)
    model_hash, _model = _model_evidence(config, path)
    policy = _teacher_evidence(config, path)
    return config, {"model_config_sha256": model_hash, "initialization_policy": policy}


def validate_ssod_matrix(config_directory: Path | str) -> tuple[Path, ...]:
    """Validate every required entry and all common comparison invariants."""
    directory = Path(config_directory)
    payloads: list[tuple[SsodMatrixEntry, Path, dict[str, Any], dict[str, Any]]] = []
    for entry in matrix_entries():
        path = directory / entry.filename
        config, evidence = _validate_entry(entry, path)
        payloads.append((entry, path, config, evidence))
    expected_names = {entry.filename for entry in matrix_entries()}
    # Customer-authorized exploratory continuations are deliberately outside
    # the fixed eight-entry comparable matrix.  They use a separately sealed
    # Teacher/provenance policy and must not make the standard matrix appear to
    # have an uncontrolled ninth experiment.
    exploratory_prefixes = ("ssod_exploratory_", "ssod_v0_", "ssod_v1_", "ssod_v3_")
    actual_names = ({item.name for item in directory.glob("ssod_*.yaml") if not item.name.startswith(exploratory_prefixes)}
                    | {item.name for item in directory.glob("ablation_*.yaml")})
    if actual_names != expected_names:
        raise _problem("SSOD matrix file set differs from the published protocol", f"missing={sorted(expected_names - actual_names)!r}, unexpected={sorted(actual_names - expected_names)!r}", "add exactly the eight committed matrix configs")
    reference = payloads[0][2]
    reference_evidence = payloads[0][3]
    for entry, path, config, evidence in payloads[1:]:
        unequal = [field for field in _COMMON_FIELDS if config.get(field) != reference.get(field)]
        if unequal:
            raise _problem("SSOD matrix comparison is not controlled", f"{path.name} differs in {unequal!r}", "keep base model, image settings, Task 8 split, training schedule, and evaluation protocol identical")
        if evidence != reference_evidence:
            raise _problem("SSOD matrix initialization evidence differs", f"{path.name} changes model hash or shared initialization policy", "use the same five-class model and Teacher/Student initialization policy")
    trust_seeds = tuple(entry.seed for entry, _, _, _ in payloads if entry.role == "trust_main")
    if trust_seeds != (42, 3407, 2026):
        raise _problem("Trust Filter replication seeds are incomplete", f"received {trust_seeds!r}", "run Trust Filter at seeds 42, 3407, and 2026")
    return tuple(path for _, path, _, _ in payloads)


def _read_json_artifact(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), "run the sealed Task 13–15 preparation path before Student training") from error
    if not isinstance(value, Mapping):
        raise _problem(f"{description} is malformed", "the sealed artifact is not a JSON object", "regenerate the matching Task 13–15 artifact")
    return value


def _canonical_sha(value: object) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _problem("sealed policy cannot be canonicalized", str(error), "regenerate the Task 14 decision manifest") from error
    return hashlib.sha256(raw).hexdigest()


def verify_prepared_ssod_artifacts(config_directory: Path | str) -> tuple[Path, ...]:
    """Gate the launcher on a sealed Task 13→14→15 policy chain.

    This deliberately does not generate predictions (that requires a completed
    Teacher and real data).  It proves that the already-published candidate,
    filtering and audit artifacts are byte-linked and that the executable
    Task-14 policy is exactly the one declared by each matrix YAML.
    """
    paths = validate_ssod_matrix(config_directory)
    for entry, path in zip(matrix_entries(), paths, strict=True):
        config = _read_yaml(path, f"SSOD matrix config {path.name}")
        policy = effective_filter_policy(config, path)
        artifact_paths = {
            field: _resolve(config.get(field), base=path.parent, field=field)
            for field in ("candidates", "filter_audit", "filter_decision_manifest", "pseudo_audit_report")
        }
        for field, artifact in artifact_paths.items():
            if not artifact.is_file():
                raise _problem("SSOD preparation artifact is missing", f"{entry.experiment_name} has no {field} at {artifact}", "run Task 13, Task 14, and Task 15 for this exact matrix policy before GPU training")
        candidates = _read_json_artifact(artifact_paths["candidates"], "Task 13 candidate envelope")
        teacher = config["initialization_policy"]["teacher_run_id"]
        if candidates.get("teacher_run_id") != teacher:
            raise _problem("Task 13 Teacher differs from matrix configuration", f"{entry.experiment_name} expects {teacher!r}", "generate candidates with the Teacher declared in initialization_policy")
        manifest = _read_json_artifact(artifact_paths["filter_decision_manifest"], "Task 14 decision manifest")
        audit_digest = _sha256(artifact_paths["filter_audit"])
        if (manifest.get("artifact_type") != "sealed_task14_filter_decisions"
                or manifest.get("teacher_run_id") != teacher
                or manifest.get("candidate_artifact_sha256") != _sha256(artifact_paths["candidates"])
                or manifest.get("decision_records_sha256") != audit_digest
                or manifest.get("filter_policy") != policy
                or manifest.get("filter_policy_sha256") != _canonical_sha(policy)):
            raise _problem("Task 14 filter policy does not match matrix configuration", f"{entry.experiment_name} decision manifest is not byte-bound to its declared executable policy", "rerun Task 14 with this config's pseudo_filter switches and preserve the sealed outputs")
        report = _read_json_artifact(artifact_paths["pseudo_audit_report"], "Task 15 pseudo audit report")
        provenance = report.get("provenance")
        # The audit candidate envelope intentionally contains only the
        # protected pseudo-audit image membership; it cannot and must not be
        # byte-identical to the train-pool candidate envelope used by Student
        # composition.  Comparability is instead proven by binding both to
        # the same full executable filter policy.
        if (report.get("teacher_run_id") != teacher or not isinstance(provenance, Mapping)
                or not isinstance(provenance.get("candidate_artifact_sha256"), str)
                or len(provenance["candidate_artifact_sha256"]) != 64
                or not isinstance(provenance.get("filter_audit_sha256"), str)
                or len(provenance["filter_audit_sha256"]) != 64
                or not isinstance(provenance.get("filter_decision_manifest_sha256"), str)
                or len(provenance["filter_decision_manifest_sha256"]) != 64
                or not isinstance(report.get("pseudo_refresh"), Mapping)
                or report["pseudo_refresh"].get("allowed") is not True
                or report.get("filter_policy") != policy
                or report.get("filter_policy_sha256") != _canonical_sha(policy)):
            raise _problem("Task 15 audit does not approve this prepared policy", f"{entry.experiment_name} lacks a passing audit bound to its executable Task 14 policy", "rerun Task 15 for this exact policy and correct pseudo quality before Student training")
    return paths


def _record_matches(entry: SsodMatrixEntry, config_path: Path, artifact_root: Path) -> tuple[bool, str]:
    record_path = artifact_root / "runs" / entry.experiment_name / "run_record.json"
    if not record_path.is_file():
        return False, "no prior run record"
    try:
        record = read_run_record(record_path)
    except (RunRecordError, OSError) as error:
        return False, f"unreadable prior run record: {error}"
    if record.status != "complete":
        return False, f"prior run status is {record.status}"
    if record.run_id != entry.experiment_name:
        return False, "prior run ID differs from the fixed matrix ID"
    snapshot = record.config_snapshot
    if snapshot.get("experiment_name") != entry.experiment_name or snapshot.get("source_config_sha256") != _sha256(config_path):
        return False, "prior run configuration fingerprint differs"
    try:
        config = _read_yaml(config_path, f"SSOD matrix config {config_path.name}")
        split_path = _resolve(config.get("split_manifest"), base=config_path.parent, field="split_manifest")
        current_split = split_fingerprint_from_manifest(split_path)
    except (SsodMatrixError, RunRecordError) as error:
        return False, f"current split fingerprint cannot be verified: {error}"
    if record.split_fingerprint != current_split:
        return False, "prior run split fingerprint differs"
    # A complete Student run is only reusable if its Task 13→15 byte inputs
    # remain the exact sealed chain recorded in the Student dataset snapshot.
    # Configuration/split equality alone cannot detect regenerated candidates
    # or a changed filter/audit decision set.
    try:
        dataset = snapshot.get("student_dataset")
        provenance = dataset.get("provenance") if isinstance(dataset, Mapping) else None
        if not isinstance(provenance, Mapping):
            return False, "prior run lacks Task 13-15 provenance"
        current = {
            "candidate_sha256": _sha256(_resolve(config.get("candidates"), base=config_path.parent, field="candidates")),
            "filter_audit_sha256": _sha256(_resolve(config.get("filter_audit"), base=config_path.parent, field="filter_audit")),
            "filter_decision_manifest_sha256": _sha256(_resolve(config.get("filter_decision_manifest"), base=config_path.parent, field="filter_decision_manifest")),
            "pseudo_audit_report_sha256": _sha256(_resolve(config.get("pseudo_audit_report"), base=config_path.parent, field="pseudo_audit_report")),
        }
    except SsodMatrixError as error:
        return False, f"current Task 13-15 artifacts cannot be verified: {error}"
    if any(provenance.get(field) != digest for field, digest in current.items()):
        return False, "prior run Task 13-15 artifact fingerprint differs"
    return True, "complete record matches config and split fingerprints"


def matrix_queue(config_directory: Path | str, *, artifact_root: Path | str | None = None, resume: bool = False, verify_preparation: bool = False) -> tuple[dict[str, object], ...]:
    """Return all entries before GPU work; never skip on uncertain evidence."""
    paths = verify_prepared_ssod_artifacts(config_directory) if verify_preparation else validate_ssod_matrix(config_directory)
    root = Path(artifact_root).resolve(strict=False) if artifact_root is not None else None
    rows: list[dict[str, object]] = []
    for entry, path in zip(matrix_entries(), paths, strict=True):
        skip, reason = (False, "resume not requested") if not resume or root is None else _record_matches(entry, path, root)
        rows.append({"experiment_name": entry.experiment_name, "config": str(path), "role": entry.role, "seed": entry.seed, "action": "skip" if skip else "run", "reason": reason})
    return tuple(rows)
