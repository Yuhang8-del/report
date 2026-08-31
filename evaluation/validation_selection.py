"""Fail-closed validation-only candidate selection for the v12 recovery."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.evaluation.detection_metrics import DetectionMetrics, DetectionMetricsError, metrics_from_mapping
from fruit_ssod.training.run_record import RunRecordError, read_run_record
from fruit_ssod.training.supervised import SupervisedTrainingError, file_evidence


class ValidationSelectionError(ValueError):
    """Raised when a candidate cannot be compared as validation evidence."""


def _problem(problem: str, cause: str, remediation: str) -> ValidationSelectionError:
    return ValidationSelectionError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


@dataclass(frozen=True)
class ValidationCandidate:
    candidate_id: str
    run_dir: Path
    inference: Mapping[str, Any]
    metrics: DetectionMetrics
    checkpoint_sha256: str
    dataset_yaml_sha256: str
    validation_membership_sha256: str
    split_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip() or any(character.isspace() for character in self.candidate_id):
            raise _problem("candidate_id is invalid", repr(self.candidate_id), "use a nonempty whitespace-free stable identifier")
        if not isinstance(self.inference, Mapping) or not self.inference:
            raise _problem("candidate inference protocol is missing", repr(self.inference), "declare the direct or sliced validation inference settings")
        if not isinstance(self.checkpoint_sha256, str) or len(self.checkpoint_sha256) != 64:
            raise _problem("candidate checkpoint digest is invalid", repr(self.checkpoint_sha256), "use a completed run with checkpoint evidence")
        if not isinstance(self.dataset_yaml_sha256, str) or len(self.dataset_yaml_sha256) != 64:
            raise _problem("candidate dataset digest is invalid", repr(self.dataset_yaml_sha256), "use a completed run with a frozen dataset YAML")
        if not isinstance(self.validation_membership_sha256, str) or len(self.validation_membership_sha256) != 64:
            raise _problem("candidate validation membership digest is invalid", repr(self.validation_membership_sha256), "use a completed run with a sealed validation image list")
        if not isinstance(self.split_fingerprint, str) or len(self.split_fingerprint) != 64:
            raise _problem("candidate split fingerprint is invalid", repr(self.split_fingerprint), "use a completed run with a sealed split manifest")


def _threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise _problem("per-class AP50 floor is invalid", repr(value), "set a normalized floor from 0 through 1")
    return float(value)


def select_validation_candidate(candidates: Sequence[ValidationCandidate], *, per_class_ap50_floor: float = 0.50) -> dict[str, Any]:
    """Select from validation evidence with predeclared map/AP/recall ordering.

    Candidates that satisfy every class AP50 floor are preferred as an
    eligibility group. Within that group (or, if empty, across all candidates),
    ranking is higher mAP50 then higher recall then lexicographic ID. This
    prevents a high aggregate score from silently masking a missing class when
    a viable all-class candidate exists.
    """
    floor = _threshold(per_class_ap50_floor)
    if not candidates:
        raise _problem("no validation candidates were supplied", "selection cannot choose an empty set", "provide at least one completed validation-only candidate")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise _problem("candidate IDs are ambiguous", repr([candidate.candidate_id for candidate in candidates]), "use one unique ID for each candidate")
    if len({candidate.split_fingerprint for candidate in candidates}) != 1:
        raise _problem("candidate split fingerprints differ", repr([candidate.split_fingerprint for candidate in candidates]), "compare candidates on exactly one sealed validation split")
    if len({candidate.validation_membership_sha256 for candidate in candidates}) != 1:
        raise _problem("candidate validation memberships differ", repr([candidate.validation_membership_sha256 for candidate in candidates]), "compare candidates using the same sealed validation image list")

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        per_class = {str(class_id): candidate.metrics.per_class_ap50[class_id] for class_id in sorted(candidate.metrics.per_class_ap50)}
        minimum = min(per_class.values())
        rows.append({
            "candidate_id": candidate.candidate_id,
            "run_dir": str(candidate.run_dir),
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "dataset_yaml_sha256": candidate.dataset_yaml_sha256,
            "validation_membership_sha256": candidate.validation_membership_sha256,
            "inference": dict(candidate.inference),
            "metrics": candidate.metrics.mapping(),
            "minimum_per_class_ap50": minimum,
            "per_class_ap50_floor": floor,
            "per_class_floor_pass": minimum >= floor,
        })
    eligible = [row for row in rows if row["per_class_floor_pass"]]
    pool = eligible if eligible else rows
    ranked = sorted(pool, key=lambda row: (-float(row["metrics"]["map50"]), -float(row["metrics"]["recall"]), str(row["candidate_id"])))
    selected = ranked[0]
    all_ranked = sorted(rows, key=lambda row: (not bool(row["per_class_floor_pass"]), -float(row["metrics"]["map50"]), -float(row["metrics"]["recall"]), str(row["candidate_id"])))
    return {
        "schema_version": "1.0",
        "protocol": "v12_validation_only_selection_v1",
        "selection_criteria": {
            "primary": "maximum map50",
            "per_class_ap50_floor": floor,
            "floor_policy": "prefer all-class-floor candidates; use the highest-map50 fallback only when none satisfies the floor",
            "tie_breaker": "maximum recall, then candidate_id",
            "fixed_test_access": "forbidden before selection",
        },
        "split_fingerprint": candidates[0].split_fingerprint,
        "validation_membership_sha256": candidates[0].validation_membership_sha256,
        "candidate_count": len(rows),
        "floor_eligible_candidate_count": len(eligible),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "candidates_ranked": all_ranked,
        "selection_status": "all_class_floor_pass" if eligible else "no_candidate_met_per_class_floor",
    }


def _candidate_from_entry(entry: Mapping[str, Any]) -> ValidationCandidate:
    candidate_id = entry.get("candidate_id")
    raw_run_dir = entry.get("run_dir")
    inference = entry.get("inference")
    if not isinstance(candidate_id, str) or not isinstance(raw_run_dir, str) or not isinstance(inference, Mapping):
        raise _problem("candidate manifest entry is malformed", repr(entry), "declare candidate_id, run_dir and inference for every candidate")
    try:
        run_dir = Path(raw_run_dir).resolve(strict=True)
        record = read_run_record(run_dir / "run_record.json")
    except (OSError, RunRecordError) as error:
        raise _problem("candidate run record cannot be read", str(error), "use a completed immutable training run") from error
    if record.status != "complete":
        raise _problem("candidate run is not complete", f"{record.run_id} is {record.status}", "complete training before validation selection")
    if (run_dir / "evaluations" / "test.json").exists():
        raise _problem("candidate has already accessed the fixed test", str(run_dir / "evaluations" / "test.json"), "do not use a fixed-test-informed run in validation-only selection")
    try:
        metrics = metrics_from_mapping(record.result or {})
    except DetectionMetricsError as error:
        raise _problem("candidate has no canonical post-training validation metrics", str(error), "complete a run whose post-training validation records all five AP50 values") from error
    snapshot = record.config_snapshot
    dataset_digest = snapshot.get("dataset_yaml_sha256")
    if not isinstance(dataset_digest, str):
        raise _problem("candidate run lacks a dataset digest", repr(dataset_digest), "use a run produced by the auditable supervised runner")
    checkpoint = file_evidence(run_dir / "weights" / "best.pt", description="candidate best checkpoint")
    paths = snapshot.get("dataset_paths")
    validation_reference = paths.get("val") if isinstance(paths, Mapping) else None
    if not isinstance(validation_reference, str) or not validation_reference:
        raise _problem("candidate run lacks a concrete validation image list", repr(validation_reference), "use the auditable supervised dataset YAML with an explicit validation .txt list")
    try:
        validation_path = Path(validation_reference).resolve(strict=True)
        if validation_path.suffix.lower() != ".txt":
            raise ValueError("validation membership must be an explicit .txt image list")
        validation_evidence = file_evidence(validation_path, description="candidate validation membership list")
    except (OSError, SupervisedTrainingError, ValueError) as error:
        raise _problem("candidate validation membership cannot be sealed", str(error), "use a readable immutable validation .txt list") from error
    return ValidationCandidate(candidate_id, run_dir, dict(inference), metrics, str(checkpoint["sha256"]), dataset_digest, str(validation_evidence["sha256"]), record.split_fingerprint)


def select_from_manifest(manifest_path: Path, *, per_class_ap50_floor: float = 0.50) -> dict[str, Any]:
    """Load only completed, fixed-test-unseen runs from a candidate manifest."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("candidate manifest cannot be read", str(error), "provide a valid JSON candidate manifest") from error
    entries = manifest.get("candidates") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list) or any(not isinstance(entry, Mapping) for entry in entries):
        raise _problem("candidate manifest has no valid candidates array", repr(entries), "provide one candidate object per validation-only run")
    return select_validation_candidate([_candidate_from_entry(entry) for entry in entries], per_class_ap50_floor=per_class_ap50_floor)


def write_selection(result: Mapping[str, Any], output: Path) -> Path:
    """Atomically publish selection evidence exactly once."""
    destination = output.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise _problem("selection output already exists", str(destination), "preserve original selection evidence or choose a new output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temporary = None
    try:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False)
        temporary = Path(handle.name)
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary, destination)
        temporary = None
        return destination
    except OSError as error:
        raise _problem("selection evidence cannot be written", str(error), "choose a fresh writable output path") from error
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
