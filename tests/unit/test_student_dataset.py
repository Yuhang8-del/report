"""Contract tests for Task 16 human-first Student composition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fruit_ssod.training.student_dataset import StudentDatasetError, StudentDatasetInputs, compose_student_dataset


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _record(image_id: str, path: str, *, labels: bool = True) -> dict[str, object]:
    value: dict[str, object] = {"source": "fixture", "source_image_id": image_id, "file_path": path, "width": 20, "height": 10, "duplicate_group_id": image_id, "class_presence": [0], "license_metadata": {"name": "fixture"}}
    if labels:
        value["labels"] = [{"class_id": 0, "xyxy": [1, 1, 10, 8]}]
    return value


def _inputs(tmp_path: Path) -> StudentDatasetInputs:
    source = tmp_path / "source"; source.mkdir()
    for name in ("human.jpg", "validation.jpg", "unlabeled.jpg"):
        (source / name).write_bytes(b"fixture-image")
    human = _record("human", "human.jpg")
    validation = _record("validation", "validation.jpg")
    unlabeled = {"source": "fixture", "source_image_id": "unlabeled", "file_path": "unlabeled.jpg", "width": 20, "height": 10, "split": "train_pool", "label_status": "unlabeled", "license_metadata": {"name": "fixture"}}
    split = {"fingerprints": {"split_protocol": "a" * 64, "budget/20": _sha([human]), "protected/validation": _sha([validation]), "unlabeled": _sha([{key: unlabeled[key] for key in ("source", "source_image_id", "file_path", "width", "height", "split", "label_status")}])}, "budget_image_ids": {"20": ["human"]}, "unlabeled_image_ids": ["unlabeled"], "split_image_ids": {"validation": ["validation"], "test": [], "pseudo_audit": [], "external_test": []}}
    paths = {name: tmp_path / name for name in ("split.json", "human_images.json", "human_labels.json", "validation.json", "unlabeled.json", "candidates.json", "audit.jsonl", "decision.json", "pseudo_audit.json")}
    paths["split.json"].write_text(json.dumps(split), encoding="utf-8")
    paths["human_images.json"].write_text(json.dumps({"records": [{key: value for key, value in human.items() if key != "labels"}]}), encoding="utf-8")
    paths["human_labels.json"].write_text(json.dumps({"records": [human]}), encoding="utf-8")
    paths["validation.json"].write_text(json.dumps({"records": [validation]}), encoding="utf-8")
    paths["unlabeled.json"].write_text(json.dumps({"records": [unlabeled]}), encoding="utf-8")
    original = {"teacher_run_id": "teacher", "source_image_id": "unlabeled", "source_file_path": "unlabeled.jpg", "view": "original", "class_id": 0, "class_name": "Apple", "confidence": 0.95, "raw_xyxy": [1, 1, 10, 8], "xyxy": [1, 1, 10, 8], "source_model": "teacher.pt"}
    flipped = {"teacher_run_id": "teacher", "source_image_id": "unlabeled", "source_file_path": "unlabeled.jpg", "view": "horizontal_flip", "class_id": 0, "class_name": "Apple", "confidence": 0.94, "raw_xyxy": [10, 1, 19, 8], "xyxy": [1, 1, 10, 8], "source_model": "teacher.pt"}
    paths["candidates.json"].write_text(json.dumps({"manifest_version": "1.0", "teacher_run_id": "teacher", "candidate_count": 2, "candidates": [original, flipped]}), encoding="utf-8")
    event = {**original, "decision": "accepted", "reason_code": "accepted", "paired_with_view": "horizontal_flip", "paired_with_confidence": 0.94, "filter_provenance": None}
    flip_event = {**flipped, "decision": "accepted", "reason_code": "accepted", "paired_with_view": "original", "paired_with_confidence": 0.95, "filter_provenance": None}
    audit_bytes = (json.dumps(event, sort_keys=True) + "\n" + json.dumps(flip_event, sort_keys=True) + "\n").encode()
    paths["audit.jsonl"].write_bytes(audit_bytes)
    manifest = {"schema_version": "1.0", "artifact_type": "sealed_task14_filter_decisions", "teacher_run_id": "teacher", "candidate_artifact_sha256": hashlib.sha256(paths["candidates.json"].read_bytes()).hexdigest(), "decision_record_count": 2, "decision_records_sha256": hashlib.sha256(audit_bytes).hexdigest(), "filter_provenance": None, "filter_provenance_sha256": _sha(None)}
    paths["decision.json"].write_text(json.dumps(manifest), encoding="utf-8")
    report = {"teacher_run_id": "teacher", "pseudo_refresh": {"allowed": True}, "provenance": {"candidate_artifact_sha256": hashlib.sha256(paths["candidates.json"].read_bytes()).hexdigest(), "filter_audit_sha256": hashlib.sha256(paths["audit.jsonl"].read_bytes()).hexdigest(), "filter_decision_manifest_sha256": hashlib.sha256(paths["decision.json"].read_bytes()).hexdigest()}}
    paths["pseudo_audit.json"].write_text(json.dumps(report), encoding="utf-8")
    return StudentDatasetInputs(paths["split.json"], paths["human_images.json"], paths["human_labels.json"], paths["validation.json"], paths["unlabeled.json"], paths["candidates.json"], paths["audit.jsonl"], paths["decision.json"], paths["pseudo_audit.json"], source)


def _rewrite_audit_bindings(inputs: StudentDatasetInputs, rows: list[dict[str, object]]) -> None:
    """Keep Task 14/15 byte digests coherent to exercise semantic checks."""
    audit = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    inputs.filter_audit.write_bytes(audit)
    decision = json.loads(inputs.filter_decision_manifest.read_text(encoding="utf-8"))
    decision["decision_record_count"] = len(rows)
    decision["decision_records_sha256"] = hashlib.sha256(audit).hexdigest()
    inputs.filter_decision_manifest.write_text(json.dumps(decision), encoding="utf-8")
    report = json.loads(inputs.pseudo_audit_report.read_text(encoding="utf-8"))
    report["provenance"]["filter_audit_sha256"] = hashlib.sha256(audit).hexdigest()
    report["provenance"]["filter_decision_manifest_sha256"] = hashlib.sha256(inputs.filter_decision_manifest.read_bytes()).hexdigest()
    inputs.pseudo_audit_report.write_text(json.dumps(report), encoding="utf-8")


def test_student_snapshot_keeps_human_precedence_metadata_and_balanced_sampling(tmp_path: Path) -> None:
    result = compose_student_dataset(_inputs(tmp_path), tmp_path / "out")
    membership = json.loads(result.membership.read_text(encoding="utf-8"))
    sources = [row["source"] for row in membership["members"]]
    assert sources.count("human") == 1 and sources.count("pseudo") == 1 and sources.count("validation") == 1
    assert all("reliability" not in row for row in membership["members"] if row["source"] != "pseudo")
    assert membership["provenance"]["training_image_ids"] == ["human", "unlabeled"]
    assert set(__import__("yaml").safe_load(result.dataset_yaml.read_text(encoding="utf-8"))) == {"path", "train", "val", "names"}
    plan = json.loads(result.sampling_plan.read_text(encoding="utf-8"))
    assert plan["human_occurrences"] == plan["pseudo_occurrences"]
    train_paths = [Path(value) for value in result.train_list.read_text(encoding="utf-8").splitlines()]
    assert train_paths and all(path.is_absolute() and path.is_file() for path in train_paths)
    assert len((result.root / "labels" / "pseudo__unlabeled.txt").read_text(encoding="utf-8").splitlines()) == 1


def test_student_snapshot_copies_images_and_binds_content_hashes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = compose_student_dataset(inputs, tmp_path / "out")
    membership = json.loads(result.membership.read_text(encoding="utf-8"))
    pseudo = next(row for row in membership["members"] if row["source"] == "pseudo")
    snapshot = result.root / pseudo["snapshot_image"]
    original_hash = hashlib.sha256(b"fixture-image").hexdigest()
    assert pseudo["snapshot_image_sha256"] == original_hash
    # Overwriting the mutable source must not alter an already-published run
    # snapshot (which would happen with a hard link).
    (inputs.source_root / "unlabeled.jpg").write_bytes(b"mutated-after-publication")
    assert snapshot.read_bytes() == b"fixture-image"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == original_hash


def test_student_rejects_pseudo_refresh_below_audit_gate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = json.loads(inputs.pseudo_audit_report.read_text(encoding="utf-8")); report["pseudo_refresh"]["allowed"] = False
    inputs.pseudo_audit_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="does not permit refresh"):
        compose_student_dataset(inputs, tmp_path / "out")


def test_exploratory_override_records_below_gate_decision(tmp_path: Path) -> None:
    inputs = replace(_inputs(tmp_path), allow_below_precision_gate=True)
    report = json.loads(inputs.pseudo_audit_report.read_text(encoding="utf-8"))
    report["pseudo_refresh"] = {"allowed": False, "reason": "stopped_precision_below_threshold"}
    inputs.pseudo_audit_report.write_text(json.dumps(report), encoding="utf-8")
    result = compose_student_dataset(inputs, tmp_path / "out")
    provenance = json.loads(result.membership.read_text(encoding="utf-8"))["provenance"]
    assert provenance["allow_below_precision_gate"] is True


def test_student_rejects_any_protected_pseudo_member(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    split = json.loads(inputs.split_manifest.read_text(encoding="utf-8")); split["split_image_ids"]["test"] = ["unlabeled"]
    inputs.split_manifest.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="unlabeled membership is unsafe"):
        compose_student_dataset(inputs, tmp_path / "out")


def test_student_rejects_accepted_pseudo_without_reciprocal_cross_view_pair(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs.filter_audit.read_text(encoding="utf-8").splitlines()]
    rows[1]["paired_with_confidence"] = 0.01
    _rewrite_audit_bindings(inputs, rows)
    with pytest.raises(StudentDatasetError, match="cross-view counterpart"):
        compose_student_dataset(inputs, tmp_path / "out")


def test_student_rejects_reciprocal_accepted_views_with_nonoverlapping_mapped_boxes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs.filter_audit.read_text(encoding="utf-8").splitlines()]
    # Counterpart views/confidences remain reciprocal.  Only Task 14's
    # mapped XYXY agreement has been corrupted, which Student must recheck.
    rows[1]["xyxy"] = [11, 1, 19, 8]
    rows[1]["raw_xyxy"] = [1, 1, 9, 8]
    candidates = json.loads(inputs.candidates.read_text(encoding="utf-8"))
    candidates["candidates"][1]["xyxy"] = [11, 1, 19, 8]
    candidates["candidates"][1]["raw_xyxy"] = [1, 1, 9, 8]
    inputs.candidates.write_text(json.dumps(candidates), encoding="utf-8")
    decision = json.loads(inputs.filter_decision_manifest.read_text(encoding="utf-8"))
    decision["candidate_artifact_sha256"] = hashlib.sha256(inputs.candidates.read_bytes()).hexdigest()
    inputs.filter_decision_manifest.write_text(json.dumps(decision), encoding="utf-8")
    _rewrite_audit_bindings(inputs, rows)
    report = json.loads(inputs.pseudo_audit_report.read_text(encoding="utf-8"))
    report["provenance"]["candidate_artifact_sha256"] = hashlib.sha256(inputs.candidates.read_bytes()).hexdigest()
    report["provenance"]["filter_decision_manifest_sha256"] = hashlib.sha256(inputs.filter_decision_manifest.read_bytes()).hexdigest()
    inputs.pseudo_audit_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(StudentDatasetError, match="cross-view counterpart"):
        compose_student_dataset(inputs, tmp_path / "out")


def test_student_rejects_audit_candidate_substitution_even_when_hashes_are_updated(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs.filter_audit.read_text(encoding="utf-8").splitlines()]
    rows[0]["xyxy"] = [2, 1, 10, 8]
    _rewrite_audit_bindings(inputs, rows)
    with pytest.raises(StudentDatasetError, match="not one-to-one"):
        compose_student_dataset(inputs, tmp_path / "out")
