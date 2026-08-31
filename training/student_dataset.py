"""Build a sealed, human-first YOLO dataset for Student SSOD training.

The task deliberately creates the labels itself instead of accepting a loose
``labels/`` directory.  This keeps pseudo confidence/source evidence out of
YOLO text files while making the training membership auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import yaml

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.data.yolo_format import YoloFormatError, format_yolo_label, xyxy_normalized_to_yolo
from fruit_ssod.pseudo.candidates import PseudoCandidate, PseudoCandidateError
from fruit_ssod.pseudo.generator import PseudoGenerationError, load_unlabeled_manifest
from fruit_ssod.pseudo.transforms import TransformError, horizontal_flip_xyxy
from fruit_ssod.pseudo.trust_filter import box_iou, load_candidate_envelope


class StudentDatasetError(RuntimeError):
    """Raised when Student data would lose its Task 8/14/15 provenance."""


def _problem(problem: str, cause: str, remediation: str) -> StudentDatasetError:
    return StudentDatasetError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _problem("sealed input cannot be hashed", str(error), "restore the original Task 8, Task 14, or Task 15 artifact") from error
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _problem("dataset evidence is not canonical JSON", str(error), "use the exact JSON artifacts emitted by the prior task") from error
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem(f"{description} cannot be read", str(error), f"provide the original readable {description}") from error
    if not isinstance(value, Mapping):
        raise _problem(f"{description} is not a JSON object", "the artifact has a malformed top-level value", "regenerate the immutable prior-task output")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value in {".", ".."} or any(c in value for c in "\\/:"):
        raise _problem(f"{field} is unsafe", "an ID could escape the Student snapshot", "use the exact non-path Task 8 source_image_id")
    return value


def _safe_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} is missing", "source-image provenance has no path", "restore the Task 8 image record")
    candidate = value.replace("\\", "/")
    windows, posix = PureWindowsPath(value), PurePosixPath(candidate)
    if windows.is_absolute() or windows.drive or posix.is_absolute() or any(part in {"", ".", ".."} for part in candidate.split("/")):
        raise _problem(f"{field} is unsafe", "an absolute or traversing input path could escape source_root", "use the exact relative Task 8 file_path")
    return candidate


def _records(path: Path, *, description: str) -> tuple[Mapping[str, Any], ...]:
    payload = _read_json(path, description=description)
    records = payload.get("records")
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise _problem(f"{description} has no records array", "the artifact does not use the Task 8 image-record schema", "supply the original Task 8 labels/images JSON")
    return tuple(records)


def _image_record(row: Mapping[str, Any], *, require_labels: bool) -> dict[str, Any]:
    required = {"source_image_id", "file_path", "width", "height"}
    if require_labels:
        required.add("labels")
    if not required <= set(row):
        raise _problem("Task 8 image record is incomplete", f"missing {sorted(required - set(row))!r}", "restore the original Task 8 budget or validation record")
    image_id = _safe_id(row["source_image_id"], field="source_image_id")
    file_path = _safe_relative(row["file_path"], field="file_path")
    width, height = row["width"], row["height"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in (width, height)):
        raise _problem("Task 8 image dimensions are invalid", "width/height must be positive integers", "restore the original image record")
    labels = row.get("labels", ())
    if require_labels and (not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or any(not isinstance(item, Mapping) for item in labels)):
        raise _problem("Task 8 human labels are malformed", "the budget record has no mapping label array", "restore the original Task 8 budget labels artifact")
    return {"source_image_id": image_id, "file_path": file_path, "width": width, "height": height, "labels": tuple(labels)}


def _yolo_lines(labels: Sequence[Mapping[str, Any]], *, width: int, height: int) -> tuple[str, ...]:
    output: list[str] = []
    for label in labels:
        class_id, xyxy = label.get("class_id"), label.get("xyxy")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in DEFAULT_CLASS_REGISTRY.class_ids:
            raise _problem("label class is outside the five-class taxonomy", f"received {class_id!r}", "retain only canonical class IDs 0 through 4")
        if not isinstance(xyxy, Sequence) or isinstance(xyxy, (str, bytes)) or len(xyxy) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in xyxy):
            raise _problem("label XYXY is malformed", "a human or accepted pseudo box is not four finite coordinates", "regenerate the upstream artifact")
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise _problem("label XYXY is outside the sealed image", "the annotation dimensions and image record differ", "regenerate upstream labels against the exact Task 8 image")
        try:
            output.append(format_yolo_label(class_id, xyxy_normalized_to_yolo(x1 / width, y1 / height, x2 / width, y2 / height)))
        except YoloFormatError as error:
            raise _problem("YOLO label conversion failed", str(error), "repair the upstream box before Student training") from error
    return tuple(sorted(output))


@dataclass(frozen=True)
class StudentDatasetInputs:
    split_manifest: Path
    human_images: Path
    human_labels: Path
    validation_labels: Path
    unlabeled_manifest: Path
    candidates: Path
    filter_audit: Path
    filter_decision_manifest: Path
    pseudo_audit_report: Path
    source_root: Path
    label_budget: int = 20
    seed: int = 42
    human_sample_probability: float = 0.5
    sampling_strategy: str = "balanced_50_50"
    expected_teacher_run_id: str | None = None
    expected_teacher_source_model: str | None = None
    pseudo_filter_policy: Mapping[str, object] | None = None
    expected_filter_policy_evidence: Mapping[str, object] | None = None
    allow_below_precision_gate: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.label_budget, bool) or not isinstance(self.label_budget, int) or not 1 <= self.label_budget <= 100:
            raise _problem("label_budget is invalid", "Student data must point at one Task 8 percentage budget", "use an integer from 1 through 100")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise _problem("Student sampling seed is invalid", "deterministic sampling requires an integer", "set seed to the experiment seed")
        if not isinstance(self.allow_below_precision_gate, bool):
            raise _problem("allow_below_precision_gate is invalid", "the exploratory override must be an explicit boolean", "set it to true only for the customer-authorized v0 run")
        value = self.human_sample_probability
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < float(value) < 1:
            raise _problem("human_sample_probability is invalid", "Student resampling must have a nontrivial human proportion", "use a finite value in (0, 1), normally 0.5")
        object.__setattr__(self, "human_sample_probability", float(value))
        if self.sampling_strategy not in {"balanced_50_50", "natural_unresampled"}:
            raise _problem(
                "Student sampling_strategy is unsupported",
                f"received {self.sampling_strategy!r}",
                "use balanced_50_50 for the main protocol or natural_unresampled only for the named ablation",
            )
        if self.expected_teacher_run_id is not None:
            object.__setattr__(
                self,
                "expected_teacher_run_id",
                _safe_id(self.expected_teacher_run_id, field="declared teacher_run_id"),
            )
        if self.pseudo_filter_policy is not None:
            policy = _normalize_pseudo_filter_policy(self.pseudo_filter_policy)
            object.__setattr__(self, "pseudo_filter_policy", policy)
        if self.expected_filter_policy_evidence is not None:
            if not isinstance(self.expected_filter_policy_evidence, Mapping):
                raise _problem("expected Task 14 policy evidence is invalid", "the matrix policy is not a mapping", "recompute the effective policy from the current matrix configuration")
            object.__setattr__(self, "expected_filter_policy_evidence", json.loads(json.dumps(self.expected_filter_policy_evidence, sort_keys=True, allow_nan=False)))


@dataclass(frozen=True)
class StudentDatasetResult:
    root: Path
    dataset_yaml: Path
    train_list: Path
    sampling_plan: Path
    membership: Path
    provenance: Mapping[str, Any]


_CANDIDATE_EVENT_FIELDS = frozenset({
    "teacher_run_id", "source_image_id", "source_file_path", "view", "class_id",
    "class_name", "confidence", "raw_xyxy", "xyxy", "source_model", "decision",
    "reason_code", "paired_with_view", "paired_with_confidence", "filter_provenance",
})


_PSEUDO_FILTER_POLICY_KEYS = frozenset({
    "policy_id", "use_per_class_thresholds", "require_view_consistency", "require_size_filter",
})


def _normalize_pseudo_filter_policy(value: Mapping[str, object]) -> dict[str, object]:
    """Parse the executable Task 17 policy before any pseudo labels are read."""
    if not isinstance(value, Mapping) or set(value) != _PSEUDO_FILTER_POLICY_KEYS:
        raise _problem("pseudo_filter policy is incomplete", "the configuration does not declare exactly the three executable Trust gates", "set policy_id, use_per_class_thresholds, require_view_consistency, and require_size_filter")
    policy_id = value.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise _problem("pseudo_filter policy_id is invalid", "the sealed Task 14 policy has no stable identity", "use the canonical Task 17 policy ID")
    flags = {key: value[key] for key in _PSEUDO_FILTER_POLICY_KEYS if key != "policy_id"}
    if any(not isinstance(item, bool) for item in flags.values()):
        raise _problem("pseudo_filter gate is invalid", "an ablation gate is not boolean", "use explicit true or false for every executable pseudo_filter gate")
    # Global is deliberately *only* a global threshold baseline.  It cannot
    # accidentally retain class/view/geometry Trust filtering under a name
    # that suggests otherwise.
    if policy_id == "global_threshold_v1" and any(flags.values()):
        raise _problem("global pseudo_filter retains Trust gates", "the global baseline must not apply class, view, or size filters", "set all three pseudo_filter gates to false")
    return {"policy_id": policy_id, **flags}


def _filter_policy_gates(value: object) -> dict[str, object]:
    """Read executable gates from legacy or full Task-17 policy evidence.

    Task 14 now seals a complete calibration/configuration record in addition
    to its four switches.  Student only executes the switches, while Task 17
    separately verifies the full manifest; accepting the four-field subset
    here preserves that division without treating arbitrary extra fields as a
    loose policy.
    """
    if not isinstance(value, Mapping):
        raise _problem("Task 14 filter policy is malformed", "the decision manifest has no mapping policy", "restore the paired sealed Task 14 decision manifest")
    if set(value) == _PSEUDO_FILTER_POLICY_KEYS:
        return _normalize_pseudo_filter_policy(value)
    if not _PSEUDO_FILTER_POLICY_KEYS <= set(value):
        raise _problem("Task 14 filter policy is incomplete", "the decision manifest lacks executable gate fields", "restore the paired sealed Task 14 decision manifest")
    return _normalize_pseudo_filter_policy({key: value[key] for key in _PSEUDO_FILTER_POLICY_KEYS})


def _candidate_identity(candidate: PseudoCandidate) -> tuple[object, ...]:
    """Identity of one Task 13 candidate occurrence, excluding Task 14's decision."""
    return (
        candidate.teacher_run_id, candidate.source_image_id, candidate.source_file_path,
        candidate.view, candidate.class_id, candidate.class_name, candidate.confidence,
        candidate.raw_xyxy, candidate.xyxy, candidate.source_model,
    )


def _finite_probability(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= float(value) <= 1


def _validate_decision_shape(row: Mapping[str, Any], *, expected_provenance: object, require_view_consistency: bool = True) -> None:
    """Reject loose Task 14 records before they can contribute Student labels."""
    if set(row) != _CANDIDATE_EVENT_FIELDS:
        raise _problem("Task 14 decision record has unsupported fields", "a loose or label-bearing audit row was supplied", "use the exact Task 14 audit.jsonl emitted for this candidate envelope")
    if row.get("filter_provenance") != expected_provenance:
        raise _problem("Task 14 decision provenance differs from its manifest", "a decision was paired with another filtering policy", "restore the exact audit.jsonl and decision_manifest.json pair")
    decision, reason = row.get("decision"), row.get("reason_code")
    if decision not in {"accepted", "rejected"} or not isinstance(reason, str) or not reason:
        raise _problem("Task 14 decision is malformed", "decision/reason evidence is not a complete filter result", "regenerate Task 14 filter decisions")
    paired_view, paired_confidence = row.get("paired_with_view"), row.get("paired_with_confidence")
    if paired_view is not None and paired_view not in {"original", "horizontal_flip"}:
        raise _problem("Task 14 paired view is malformed", "the audit record names an unsupported view", "regenerate Task 14 filter decisions")
    if paired_confidence is not None and not _finite_probability(paired_confidence):
        raise _problem("Task 14 paired confidence is malformed", "the audit record has a non-finite paired score", "regenerate Task 14 filter decisions")
    if decision == "accepted" and require_view_consistency:
        if reason != "accepted" or paired_view not in {"original", "horizontal_flip"} or paired_confidence is None:
            raise _problem("accepted Task 14 decision lacks a cross-view pair", "Student labels require a fully recorded accepted original/flip pair", "regenerate Task 14 filter decisions")
    if decision == "accepted" and not require_view_consistency and (reason != "accepted" or paired_view is not None or paired_confidence is not None):
        raise _problem("view-disabled accepted Task 14 decision is malformed", "the original-view representative must not claim a cross-view pair", "regenerate Task 14 with require_view_consistency: false")


def _validate_accepted_geometry(candidate: PseudoCandidate, *, width: int, height: int) -> None:
    """Recheck Task 13's original-coordinate claim for a retained candidate."""
    for box in (candidate.raw_xyxy, candidate.xyxy):
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise _problem("accepted pseudo candidate is outside sealed image geometry", "Task 13 coordinates do not match the sealed unlabeled record", "regenerate Tasks 13 and 14 from the exact unlabeled membership")
    if candidate.view == "original":
        if candidate.raw_xyxy != candidate.xyxy:
            raise _problem("accepted original pseudo candidate has inconsistent coordinates", "raw and original-coordinate boxes differ", "regenerate Task 13 dual-view candidates")
        return
    try:
        mapped = horizontal_flip_xyxy(candidate.raw_xyxy, width=width)
    except TransformError as error:
        raise _problem("accepted flip pseudo candidate cannot map to original coordinates", str(error), "regenerate Task 13 dual-view candidates") from error
    if mapped != candidate.xyxy:
        raise _problem("accepted flip pseudo candidate has inconsistent coordinates", "the raw flip box does not map to its declared original box", "regenerate Task 13 dual-view candidates")


def _load_split(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path, description="Task 8 split manifest")
    fingerprints, budget_ids, split_ids = payload.get("fingerprints"), payload.get("budget_image_ids"), payload.get("split_image_ids")
    if not isinstance(fingerprints, Mapping) or not isinstance(budget_ids, Mapping) or not isinstance(split_ids, Mapping):
        raise _problem("Task 8 split manifest is incomplete", "fingerprints, budget_image_ids, or split_image_ids are missing", "use the unmodified Task 8 split_manifest.json")
    if set(split_ids) != {"validation", "test", "pseudo_audit", "external_test"}:
        raise _problem("Task 8 protected memberships are invalid", "the split manifest does not seal all protected partitions", "regenerate Task 8 split artifacts")
    return payload


def _sealed_human(inputs: StudentDatasetInputs, split: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    image_rows = tuple(_image_record(row, require_labels=False) for row in _records(inputs.human_images, description="Task 8 budget images"))
    label_rows = tuple(_image_record(row, require_labels=True) for row in _records(inputs.human_labels, description="Task 8 budget labels"))
    if [row["source_image_id"] for row in image_rows] != [row["source_image_id"] for row in label_rows]:
        raise _problem("Task 8 human images and labels do not align", "the budget image order/membership differs from the budget label artifact", "use the paired budgets/<percent>/images.json and labels.json files")
    budget_key = str(inputs.label_budget)
    expected_ids = split["budget_image_ids"].get(budget_key) if isinstance(split["budget_image_ids"], Mapping) else None
    if not isinstance(expected_ids, list) or expected_ids != [row["source_image_id"] for row in label_rows]:
        raise _problem("human budget membership differs from Task 8", "the labels are not exactly the sealed requested budget", "use the paired Task 8 budget artifacts for this label budget")
    expected = split["fingerprints"].get(f"budget/{budget_key}") if isinstance(split["fingerprints"], Mapping) else None
    if not isinstance(expected, str) or expected != _canonical_sha([dict(row) for row in _records(inputs.human_labels, description="Task 8 budget labels")]):
        raise _problem("human budget fingerprint differs from Task 8", "the labels were edited or paired with a different split manifest", "restore the original Task 8 budget labels JSON")
    return label_rows


def _sealed_validation(inputs: StudentDatasetInputs, split: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = tuple(_image_record(row, require_labels=True) for row in _records(inputs.validation_labels, description="Task 8 validation labels"))
    expected_ids = split["split_image_ids"].get("validation") if isinstance(split["split_image_ids"], Mapping) else None
    expected_digest = split["fingerprints"].get("protected/validation") if isinstance(split["fingerprints"], Mapping) else None
    raw_records = _records(inputs.validation_labels, description="Task 8 validation labels")
    if expected_ids != [row["source_image_id"] for row in rows] or not isinstance(expected_digest, str) or expected_digest != _canonical_sha([dict(row) for row in raw_records]):
        raise _problem("validation membership differs from Task 8", "validation labels were edited or are not the sealed protected split", "restore protected_splits/validation_labels.json")
    return rows


def _accepted_pseudo(inputs: StudentDatasetInputs, split: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    # Reuse Task 13's verifier: this proves the candidates are exactly the
    # Task 8 no-label pool and supplies dimensions/source paths without ever
    # reading protected labels.
    try:
        membership = load_unlabeled_manifest(inputs.unlabeled_manifest, split_manifest_path=inputs.split_manifest)
    except PseudoGenerationError as error:
        raise _problem("Task 8 unlabeled membership is unsafe", str(error), "restore the paired unlabeled.json and split_manifest.json without protected members") from error
    by_id = {row.source_image_id: row for row in membership.records}
    # The decisions have authority only when every one of their candidate
    # fields can be joined back to the exact Task 13 envelope.  Do not infer
    # labels from an audit row alone: it could otherwise substitute a box,
    # view, source image, or teacher while retaining a plausible digest.
    try:
        envelope_teacher, candidates = load_candidate_envelope(inputs.candidates)
    except Exception as error:
        raise _problem("Task 13 candidate envelope is invalid", str(error), "restore the original Task 13 candidates JSON envelope") from error
    if inputs.expected_teacher_run_id is not None and envelope_teacher != inputs.expected_teacher_run_id:
        raise _problem(
            "Task 13 teacher run differs from the declared Teacher configuration",
            f"candidate envelope declares {envelope_teacher!r}, but Student configuration declares {inputs.expected_teacher_run_id!r}",
            "generate pseudo labels from the configured completed Teacher run or correct initialization_policy.teacher_run_id",
        )
    if inputs.expected_teacher_source_model is not None:
        source_models = {candidate.source_model for candidate in candidates}
        if source_models != {inputs.expected_teacher_source_model}:
            raise _problem(
                "Task 13 candidate checkpoint differs from the declared Teacher checkpoint",
                f"candidate source_model values {sorted(source_models)!r} do not match {inputs.expected_teacher_source_model!r}",
                "regenerate pseudo labels with the immutable Teacher checkpoint declared by the Student configuration",
            )
    manifest = _read_json(inputs.filter_decision_manifest, description="Task 14 decision manifest")
    required = {"schema_version", "artifact_type", "teacher_run_id", "candidate_artifact_sha256", "decision_record_count", "decision_records_sha256", "filter_provenance", "filter_provenance_sha256"}
    policy_required = required | {"filter_policy", "filter_policy_sha256"}
    allowed_manifest = policy_required if inputs.pseudo_filter_policy is not None else required
    if (set(manifest) != allowed_manifest or manifest.get("schema_version") != "1.0"
            or manifest.get("artifact_type") != "sealed_task14_filter_decisions"
            or manifest.get("teacher_run_id") != envelope_teacher
            or manifest.get("candidate_artifact_sha256") != _sha256(inputs.candidates)
            or manifest.get("filter_provenance_sha256") != _canonical_sha(manifest.get("filter_provenance"))):
        raise _problem("Task 14 decision manifest is not bound to candidates", "the candidate envelope or decision manifest was substituted", "use the exact paired Task 13 and Task 14 outputs")
    if inputs.pseudo_filter_policy is not None:
        full_policy = manifest.get("filter_policy")
        actual_policy = _filter_policy_gates(full_policy)
        if actual_policy != inputs.pseudo_filter_policy or manifest.get("filter_policy_sha256") != _canonical_sha(full_policy):
            raise _problem("Task 14 filter policy does not match Student configuration", "the sealed decision manifest was produced by another global/Trust/ablation policy", "prepare Task 13–15 artifacts with this exact pseudo_filter configuration")
    if inputs.pseudo_filter_policy is not None and inputs.expected_filter_policy_evidence is not None and full_policy != inputs.expected_filter_policy_evidence:
        raise _problem("Task 14 full filter policy does not match Student configuration", "the policy's calibration sources, thresholds, or non-toggle guards differ from the current matrix config", "regenerate Task 14 with --matrix-config for this exact Student experiment")
    try:
        audit_bytes = inputs.filter_audit.read_bytes()
    except OSError as error:
        raise _problem("Task 14 decision audit cannot be read", str(error), "restore the original audit.jsonl") from error
    if manifest.get("decision_records_sha256") != hashlib.sha256(audit_bytes).hexdigest():
        raise _problem("Task 14 decision audit differs from its manifest", "accepted/rejected decisions may have been edited", "restore audit.jsonl and decision_manifest.json as a pair")
    try:
        decoded = audit_bytes.decode("utf-8")
        lines = decoded.splitlines()
        if any(not line.strip() for line in lines):
            raise _problem("Task 14 decision audit contains a blank line", "immutable JSONL decision evidence was truncated or edited", "restore the original Task 14 audit.jsonl")
        events = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("Task 14 audit JSONL is malformed", str(error), "restore the original UTF-8 audit.jsonl") from error
    if (isinstance(manifest.get("decision_record_count"), bool)
            or not isinstance(manifest.get("decision_record_count"), int)
            or manifest.get("decision_record_count") < 0
            or not all(isinstance(row, Mapping) for row in events)
            or manifest.get("decision_record_count") != len(events)):
        raise _problem("Task 14 decision audit count is invalid", "the JSONL event stream differs from its sealed manifest", "restore the original Task 14 outputs")
    report = _read_json(inputs.pseudo_audit_report, description="Task 15 pseudo audit report")
    refresh = report.get("pseudo_refresh")
    provenance = report.get("provenance")
    if not isinstance(refresh, Mapping) or (refresh.get("allowed") is not True and not inputs.allow_below_precision_gate) or not isinstance(provenance, Mapping):
        raise _problem("Task 15 pseudo audit does not permit refresh", "post-filter precision is below the gate or the audit report is incomplete", "improve filtering/teacher quality and rerun Task 15 before Student training")
    if inputs.allow_below_precision_gate and refresh.get("allowed") is not True and refresh.get("reason") != "stopped_precision_below_threshold":
        raise _problem("exploratory pseudo-refresh override is not bound", "the report does not explicitly record a below-threshold precision decision", "use the exact Task 15 report with reason stopped_precision_below_threshold")
    exact_audit = (
        provenance.get("candidate_artifact_sha256") == _sha256(inputs.candidates)
        and provenance.get("filter_audit_sha256") == _sha256(inputs.filter_audit)
        and provenance.get("filter_decision_manifest_sha256") == _sha256(inputs.filter_decision_manifest)
    )
    # A protected pseudo-audit must use different image IDs, so its candidate
    # envelope cannot be byte-identical to the independent unlabeled pool.
    # Matrix-driven runs may therefore bind the two artifacts through one full
    # executable policy rather than pretending their candidate hashes match.
    matched_policy_audit = (
        inputs.expected_filter_policy_evidence is not None
        and report.get("filter_policy") == full_policy
        and report.get("filter_policy_sha256") == _canonical_sha(full_policy)
    )
    if report.get("teacher_run_id") != manifest.get("teacher_run_id") or not (exact_audit or matched_policy_audit):
        raise _problem("Task 15 audit provenance differs from Task 14", "the acceptance report neither audits these exact decisions nor proves the same executable filter policy on the protected pseudo-audit split", "supply paired Task 14/15 artifacts or a passing protected audit bound to this exact policy")
    protected = set().union(*(set(ids) for ids in split["split_image_ids"].values()))
    candidate_slots: dict[tuple[object, ...], list[PseudoCandidate]] = defaultdict(list)
    for candidate in candidates:
        image_id = _safe_id(candidate.source_image_id, field="Task 13 candidate source_image_id")
        source = by_id.get(image_id)
        if source is None or image_id in protected or candidate.source_file_path != source.file_path:
            raise _problem("Task 13 candidate lies outside the sealed unlabeled pool", "a candidate image ID or path is not exactly the Task 8 no-label member", "regenerate Task 13 from the paired Task 8 unlabeled manifest")
        candidate_slots[_candidate_identity(candidate)].append(candidate)
    # Pop one candidate occurrence for every audit row.  A Counter alone
    # proves multiplicity but cannot retain the corresponding representative.
    consumed: Counter[tuple[object, ...]] = Counter()
    bound_events: list[tuple[PseudoCandidate, Mapping[str, Any]]] = []
    require_view_consistency = inputs.pseudo_filter_policy is None or bool(inputs.pseudo_filter_policy["require_view_consistency"])
    for event in events:
        _validate_decision_shape(event, expected_provenance=manifest.get("filter_provenance"), require_view_consistency=require_view_consistency)
        try:
            candidate = PseudoCandidate(**{key: event[key] for key in (
                "teacher_run_id", "source_image_id", "source_file_path", "view", "class_id",
                "class_name", "confidence", "raw_xyxy", "xyxy", "source_model",
            )})
        except (KeyError, TypeError, PseudoCandidateError) as error:
            raise _problem("Task 14 decision candidate provenance is invalid", str(error), "restore the unmodified Task 14 audit.jsonl") from error
        identity = _candidate_identity(candidate)
        if consumed[identity] >= len(candidate_slots.get(identity, ())):
            raise _problem("Task 14 decisions are not one-to-one with Task 13 candidates", "a decision was injected, substituted, duplicated, or paired with another teacher envelope", "use the exact Task 13 candidates and Task 14 audit outputs")
        consumed[identity] += 1
        if candidate.teacher_run_id != envelope_teacher:
            raise _problem("Task 14 decision teacher differs from Task 13", "mixed teacher evidence was supplied", "use one paired Task 13/14 teacher run")
        bound_events.append((candidate, event))
    expected_counts = Counter({identity: len(slots) for identity, slots in candidate_slots.items()})
    # The expression above deliberately compares occurrence counts, including
    # duplicate byte-identical candidates that Task 14 may later reject.
    if consumed != expected_counts:
        raise _problem("Task 14 decisions do not cover every Task 13 candidate", "the audit JSONL was truncated or a candidate was silently omitted", "restore the complete paired Task 13 and Task 14 artifacts")

    accepted_originals: list[tuple[PseudoCandidate, Mapping[str, Any]]] = []
    accepted_flips: list[tuple[PseudoCandidate, Mapping[str, Any]]] = []
    for candidate, event in bound_events:
        if event["decision"] != "accepted":
            continue
        source = by_id[candidate.source_image_id]
        _validate_accepted_geometry(candidate, width=source.width, height=source.height)
        if candidate.view == "original":
            accepted_originals.append((candidate, event))
        else:
            accepted_flips.append((candidate, event))

    # Pair accepted decisions one-to-one when the executable policy requires
    # it.  A no-view ablation accepts original-view representatives directly;
    # flip candidates remain recorded as non-selected evidence in Task 14.
    # opposite view and its confidence, so both rows must affirm the same
    # cross-view relation.  Only its original-view member becomes a Student
    # label; a flip is evidence, never a second training target.
    # TrustFilterConfig's protocol default is deliberately repeated here at
    # the consumer boundary.  Pair-view/confidence fields alone can be made
    # reciprocal while their mapped original-coordinate boxes do not overlap.
    cross_view_iou = 0.60
    def pair_order(item: tuple[PseudoCandidate, Mapping[str, Any]]) -> tuple[object, ...]:
        candidate, _event = item
        return (-candidate.confidence, candidate.source_image_id, candidate.class_id, *candidate.xyxy, candidate.source_model)
    accepted_originals.sort(key=pair_order)
    accepted_flips.sort(key=pair_order)
    if not require_view_consistency:
        if accepted_flips:
            raise _problem("view-disabled Task 14 policy accepted flip labels", "the no-view ablation must still use original views as the sole Student representatives", "regenerate Task 14 with require_view_consistency: false")
        if not accepted_originals:
            raise _problem("Task 14 has no accepted pseudo labels", "Student training would not be semi-supervised", "tune the teacher/filter and rerun Tasks 13 through 15")
        result: dict[str, list[tuple[PseudoCandidate, Mapping[str, Any]]]] = {}
        for candidate, event in accepted_originals:
            result.setdefault(candidate.source_image_id, []).append((candidate, event))
        return _pseudo_records(result, by_id)
    eligible: dict[int, list[int]] = {}
    for original_index, (original, original_event) in enumerate(accepted_originals):
        choices: list[int] = []
        for flip_index, (flip, flip_event) in enumerate(accepted_flips):
            if (original.source_image_id == flip.source_image_id
                    and original.source_file_path == flip.source_file_path
                    and original.teacher_run_id == flip.teacher_run_id
                    and original.class_id == flip.class_id
                    and original.class_name == flip.class_name
                    and original.source_model == flip.source_model
                    and original_event["paired_with_view"] == "horizontal_flip"
                    and flip_event["paired_with_view"] == "original"
                    and float(original_event["paired_with_confidence"]) == flip.confidence
                    and float(flip_event["paired_with_confidence"]) == original.confidence
                    and box_iou(original.xyxy, flip.xyxy) >= cross_view_iou):
                choices.append(flip_index)
        eligible[original_index] = choices

    matched_flip: dict[int, int] = {}
    def match(original_index: int, seen: set[int]) -> bool:
        for flip_index in eligible[original_index]:
            if flip_index in seen:
                continue
            seen.add(flip_index)
            prior = matched_flip.get(flip_index)
            if prior is None or match(prior, seen):
                matched_flip[flip_index] = original_index
                return True
        return False
    for original_index in range(len(accepted_originals)):
        if not match(original_index, set()):
            raise _problem("accepted pseudo decision lacks a valid cross-view counterpart", "an accepted original/flip pair is malformed, mismatched, or duplicated", "regenerate Task 14 filter decisions from the exact Task 13 envelope")
    if len(matched_flip) != len(accepted_flips):
        raise _problem("accepted pseudo flip lacks a valid original representative", "a flip-view acceptance would become an arbitrary Student label", "regenerate Task 14 filter decisions from the exact Task 13 envelope")

    result: dict[str, list[tuple[PseudoCandidate, Mapping[str, Any]]]] = {}
    for original_index in sorted(matched_flip.values()):
        candidate, event = accepted_originals[original_index]
        result.setdefault(candidate.source_image_id, []).append((candidate, event))
    if not result:
        raise _problem("Task 14 has no accepted pseudo labels", "Student training would not be semi-supervised", "tune the teacher/filter and rerun Tasks 13 through 15")
    return _pseudo_records(result, by_id)


def _pseudo_records(result: Mapping[str, Sequence[tuple[PseudoCandidate, Mapping[str, Any]]]], by_id: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for image_id in sorted(result):
        source = by_id[image_id]
        labels: list[dict[str, Any]] = []
        reliability: list[dict[str, Any]] = []
        for candidate, event in result[image_id]:
            labels.append({"class_id": candidate.class_id, "xyxy": candidate.xyxy})
            reliability.append({"confidence": candidate.confidence, "teacher_run_id": candidate.teacher_run_id, "source_model": candidate.source_model, "filter_provenance": event["filter_provenance"]})
        records.append({"source_image_id": image_id, "file_path": _safe_relative(source.file_path, field="unlabeled file_path"), "width": source.width, "height": source.height, "labels": tuple(labels), "reliability": reliability})
    return tuple(records)


def _source_path(source_root: Path, relative: str) -> Path:
    root = source_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise _problem("source image escapes source_root", "a resolved path or symlink points outside the approved image root", "restore Task 8 paths under the configured source_root") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise _problem("source image is not a regular file", f"{candidate} is missing or a symlink", "restore a regular source image under source_root")
    return candidate


def _copy_snapshot_image(source: Path, destination: Path) -> str:
    """Copy (never hard-link) one source image into an immutable run snapshot.

    A hard link looks economical, but it makes a published training snapshot
    change when a mutable source file is overwritten in place.  Copy the bytes
    and bind both sides to their digest before exposing the snapshot.
    """
    source_digest = _sha256(source)
    try:
        shutil.copy2(source, destination)
    except OSError as error:
        raise _problem("source image cannot be copied into Student snapshot", str(error), "ensure the source image is readable and the run directory is writable") from error
    destination_digest = _sha256(destination)
    if destination_digest != source_digest:
        raise _problem("Student snapshot image hash differs from source", "the copied image bytes changed during snapshot publication", "retry from stable source storage and inspect filesystem integrity")
    return destination_digest


def _deterministic_sample(
    human: Sequence[str], pseudo: Sequence[str], *, seed: int, probability: float, strategy: str
) -> tuple[str, ...]:
    if not human or not pseudo:
        raise _problem("Student training has an empty source type", "approximately 50% human resampling requires both human and accepted pseudo examples", "provide sealed human budget labels and accepted pseudo labels")
    if strategy == "natural_unresampled":
        # The ablation intentionally removes human/pseudo oversampling.  Keep
        # every distinct input once, but retain a deterministic order so it
        # remains a reproducible experimental condition rather than an input
        # filesystem accident.
        return tuple(sorted((*human, *pseudo), key=lambda value: hashlib.sha256(f"{seed}:natural:{value}".encode()).hexdigest()))
    # Equal source occurrences implement the specified 50/50 policy exactly;
    # sorted hash order makes repetition independent of input ordering.
    target = max(len(human), len(pseudo))
    def expand(values: Sequence[str], kind: str) -> list[str]:
        ordered = sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest())
        return [ordered[index % len(ordered)] for index in range(target)]
    left, right = expand(human, "human"), expand(pseudo, "pseudo")
    values = [entry for pair in zip(left, right) for entry in pair]
    # The probability is explicit evidence even though this first protocol is
    # intentionally exact at 0.5.  Other probabilities are rejected here so
    # comparable SSOD main runs cannot silently change resampling.
    if probability != 0.5:
        raise _problem("human_sample_probability is not comparable", "the approved Task 16 protocol fixes equal human/pseudo sampling", "use exactly 0.5 for all comparable SSOD runs")
    return tuple(values)


def compose_student_dataset(inputs: StudentDatasetInputs, output_root: Path) -> StudentDatasetResult:
    """Atomically compose a derived YOLO snapshot from sealed Task 8/14/15 evidence."""
    root = output_root.resolve(strict=False)
    if root.exists():
        raise _problem("Student dataset output already exists", f"{root} would be overwritten", "choose a new output directory for each Student run")
    split = _load_split(inputs.split_manifest)
    human, validation, pseudo = _sealed_human(inputs, split), _sealed_validation(inputs, split), _accepted_pseudo(inputs, split)
    human_ids = {row["source_image_id"] for row in human}
    pseudo = tuple(row for row in pseudo if row["source_image_id"] not in human_ids)  # Human labels always win if inputs are compromised/overlap.
    if not pseudo:
        raise _problem("human-label precedence removed every pseudo image", "accepted pseudo IDs overlap the human budget", "regenerate Task 14 from Task 8's disjoint unlabeled manifest")
    protected = set().union(*(set(ids) for ids in split["split_image_ids"].values()))
    training_ids = human_ids | {row["source_image_id"] for row in pseudo}
    if training_ids & protected:
        raise _problem("protected labels leaked into Student training", f"overlap {sorted(training_ids & protected)!r}", "use only the Task 8 human budget and no-label pseudo pool")
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        images, labels = temporary / "images", temporary / "labels"
        images.mkdir(); labels.mkdir()
        members: list[dict[str, Any]] = []
        paths: dict[str, str] = {}
        for kind, rows in (("human", human), ("pseudo", pseudo), ("validation", validation)):
            for row in rows:
                image_id = row["source_image_id"]
                extension = Path(row["file_path"]).suffix.lower() or ".jpg"
                stem = f"{kind}__{image_id}"
                source = _source_path(inputs.source_root, row["file_path"])
                image_dest, label_dest = images / f"{stem}{extension}", labels / f"{stem}.txt"
                image_digest = _copy_snapshot_image(source, image_dest)
                label_dest.write_text("\n".join(_yolo_lines(row["labels"], width=row["width"], height=row["height"])) + "\n", encoding="utf-8", newline="\n")
                paths[f"{kind}:{image_id}"] = image_dest.relative_to(temporary).as_posix()
                metadata: dict[str, Any] = {"source_image_id": image_id, "source": kind, "source_file_path": row["file_path"], "snapshot_image": paths[f"{kind}:{image_id}"], "snapshot_image_sha256": image_digest, "snapshot_label": label_dest.relative_to(temporary).as_posix()}
                if kind == "pseudo":
                    metadata["reliability"] = row["reliability"]
                members.append(metadata)
        # Ultralytics resolves each line in a text split independently of the
        # dataset YAML's ``path``.  Relative entries therefore resolve against
        # the process working directory on Windows and make every label appear
        # missing.  Keep relative paths in immutable membership evidence, but
        # publish absolute snapshot paths in the executable split lists.
        relative_train_entries = _deterministic_sample([paths[f"human:{row['source_image_id']}"] for row in human], [paths[f"pseudo:{row['source_image_id']}"] for row in pseudo], seed=inputs.seed, probability=inputs.human_sample_probability, strategy=inputs.sampling_strategy)
        relative_val_entries = [paths[f"validation:{row['source_image_id']}"] for row in validation]
        train_entries = [str((root / entry).resolve(strict=False)) for entry in relative_train_entries]
        val_entries = [str((root / entry).resolve(strict=False)) for entry in relative_val_entries]
        (temporary / "train.txt").write_text("\n".join(train_entries) + "\n", encoding="utf-8", newline="\n")
        (temporary / "val.txt").write_text("\n".join(val_entries) + "\n", encoding="utf-8", newline="\n")
        dataset = {"path": str(temporary), "train": "train.txt", "val": "val.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (temporary / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        image_evidence = [{"source": member["source"], "source_image_id": member["source_image_id"], "snapshot_image": member["snapshot_image"], "sha256": member["snapshot_image_sha256"]} for member in members]
        provenance = {"schema_version": "1.0", "artifact_type": "sealed_student_dataset", "split_fingerprint": split["fingerprints"].get("split_protocol"), "budget_fingerprint": split["fingerprints"].get(f"budget/{inputs.label_budget}"), "unlabeled_fingerprint": split["fingerprints"].get("unlabeled"), "candidate_sha256": _sha256(inputs.candidates), "filter_audit_sha256": _sha256(inputs.filter_audit), "filter_decision_manifest_sha256": _sha256(inputs.filter_decision_manifest), "pseudo_audit_report_sha256": _sha256(inputs.pseudo_audit_report), "pseudo_filter_policy": dict(inputs.pseudo_filter_policy or {}), "pseudo_filter_policy_sha256": _canonical_sha(dict(inputs.pseudo_filter_policy or {})), "allow_below_precision_gate": inputs.allow_below_precision_gate, "seed": inputs.seed, "human_sample_probability": inputs.human_sample_probability, "sampling_strategy": inputs.sampling_strategy, "canonical_classes": list(DEFAULT_CLASS_REGISTRY.class_names), "training_image_ids": sorted(training_ids), "validation_image_ids": sorted(row["source_image_id"] for row in validation), "snapshot_image_evidence_sha256": _canonical_sha(image_evidence)}
        membership = {"schema_version": "1.0", "members": members, "provenance": provenance}
        (temporary / "training_membership.json").write_text(json.dumps(membership, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "sampling_plan.json").write_text(json.dumps({"human_occurrences": sum(item.startswith("images/human__") for item in relative_train_entries), "pseudo_occurrences": sum(item.startswith("images/pseudo__") for item in relative_train_entries), "train_entries": list(train_entries), "relative_train_entries": list(relative_train_entries), "seed": inputs.seed, "human_sample_probability": inputs.human_sample_probability, "sampling_strategy": inputs.sampling_strategy}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root); temporary = None
        # Dataset path was staged, so rewrite only its path after publication.
        published = {"path": str(root), "train": "train.txt", "val": "val.txt", "names": list(DEFAULT_CLASS_REGISTRY.class_names)}
        (root / "dataset.yaml").write_text(yaml.safe_dump(published, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        return StudentDatasetResult(root, root / "dataset.yaml", root / "train.txt", root / "sampling_plan.json", root / "training_membership.json", provenance)
    except StudentDatasetError:
        raise
    except (OSError, ValueError, YoloFormatError) as error:
        raise _problem("Student dataset could not be composed", str(error), "check source images and choose a new writable output path") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
