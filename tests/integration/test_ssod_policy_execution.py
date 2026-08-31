"""Task 17 integration contracts: YAML-style gates change real membership."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import yaml

from fruit_ssod.cli.filter_pseudo_labels import main as filter_pseudo_main
from fruit_ssod.cli.train_student import main as train_student_main
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.thresholds import PerClassThresholds
from fruit_ssod.pseudo.transforms import horizontal_flip_xyxy
from fruit_ssod.pseudo.trust_filter import ImageGeometry, TrustFilter, TrustFilterConfig, write_trust_filter_outputs
from fruit_ssod.training.student_dataset import compose_student_dataset
from tests.unit.test_student_dataset import _inputs


def _pair(image_id: str, class_id: int, confidence: float, box: tuple[float, float, float, float], *, matching_flip: bool = True) -> tuple[PseudoCandidate, PseudoCandidate]:
    raw_flip = horizontal_flip_xyxy(box, width=640) if matching_flip else (0.0, box[1], box[2] - box[0], box[3])
    flip_xyxy = box if matching_flip else horizontal_flip_xyxy(raw_flip, width=640)
    original = PseudoCandidate("teacher", image_id, f"{image_id}.jpg", "original", class_id, ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")[class_id], confidence, box, box, "teacher.pt")
    flipped = PseudoCandidate("teacher", image_id, f"{image_id}.jpg", "horizontal_flip", class_id, ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")[class_id], confidence, raw_flip, flip_xyxy, "teacher.pt")
    return original, flipped


def _membership(config: TrustFilterConfig) -> set[str]:
    candidates = (
        *_pair("good", 3, .90, (100., 100., 200., 200.)),
        # Only the size ablation/global baseline admits this 10×10 pair.
        *_pair("small", 0, .55, (10., 10., 20., 20.)),
        # Only the class-threshold ablation/global baseline admits this pair.
        *_pair("class_gate", 1, .55, (250., 100., 350., 200.)),
        # Only the view ablation/global baseline admits this original.
        *_pair("view_gate", 2, .90, (400., 100., 500., 200.), matching_flip=False),
    )
    geometry = {item: ImageGeometry(item, 640, 640) for item in ("good", "small", "class_gate", "view_gate")}
    thresholds = PerClassThresholds({0: .50, 1: .70, 2: .50, 3: .50, 4: .50})
    result = TrustFilter(thresholds, config=config).filter("teacher", candidates, geometry)
    return {label.source_image_id for label in result.accepted}


def test_each_task17_ablation_changes_only_its_named_pseudo_membership_gate() -> None:
    trust = _membership(TrustFilterConfig(policy_id="trust_filter_v1"))
    no_class = _membership(TrustFilterConfig(policy_id="trust_without_class_threshold_v1", use_per_class_thresholds=False))
    no_view = _membership(TrustFilterConfig(policy_id="trust_without_view_consistency_v1", require_view_consistency=False))
    no_size = _membership(TrustFilterConfig(policy_id="trust_without_size_filter_v1", require_size_filter=False))
    global_only = _membership(TrustFilterConfig(policy_id="global_threshold_v1", use_per_class_thresholds=False, require_view_consistency=False, require_size_filter=False))

    assert trust == {"good"}
    assert no_class == trust | {"class_gate"}
    assert no_view == trust | {"view_gate"}
    assert no_size == trust | {"small"}
    assert global_only == trust | {"class_gate", "view_gate", "small"}


def _actual_task14_then_student(tmp_path: Path, config: TrustFilterConfig) -> tuple[object, dict[str, object]]:
    """Publish real Task-14 bytes, then pass the exact sealed chain to Student."""
    inputs = _inputs(tmp_path)
    geometry = {"unlabeled": ImageGeometry("unlabeled", 20, 10)}
    thresholds = PerClassThresholds({index: .50 for index in range(5)})
    filtered = TrustFilter(thresholds, config=config).filter_envelope(inputs.candidates, geometry)
    output = tmp_path / "actual-task14"
    _, audit = write_trust_filter_outputs(filtered, output)
    decision = audit.with_name("decision_manifest.json")
    report = {
        "teacher_run_id": "teacher",
        "pseudo_refresh": {"allowed": True},
        "provenance": {
            "candidate_artifact_sha256": hashlib.sha256(inputs.candidates.read_bytes()).hexdigest(),
            "filter_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "filter_decision_manifest_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
        },
    }
    report_path = tmp_path / "actual-task15.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    sealed = replace(
        inputs,
        filter_audit=audit,
        filter_decision_manifest=decision,
        pseudo_audit_report=report_path,
        expected_teacher_run_id="teacher",
        pseudo_filter_policy=dict(config.policy_mapping()),
    )
    result = compose_student_dataset(sealed, tmp_path / "student")
    membership = json.loads(result.membership.read_text(encoding="utf-8"))
    return result, membership


def test_actual_task14_trust_output_composes_student_with_paired_accepted_labels(tmp_path: Path) -> None:
    result, membership = _actual_task14_then_student(tmp_path, TrustFilterConfig(policy_id="trust_filter_v1"))
    pseudo = [row for row in membership["members"] if row["source"] == "pseudo"]
    assert len(pseudo) == 1 and pseudo[0]["source_image_id"] == "unlabeled"
    assert (result.root / "labels" / "pseudo__unlabeled.txt").is_file()
    assert membership["provenance"]["pseudo_filter_policy"]["require_view_consistency"] is True


def test_actual_task14_no_view_and_global_output_import_only_original_representative(tmp_path: Path) -> None:
    for name, config in (
        ("no-view", TrustFilterConfig(policy_id="trust_without_view_consistency_v1", require_view_consistency=False)),
        ("global", TrustFilterConfig(policy_id="global_threshold_v1", use_per_class_thresholds=False, require_view_consistency=False, require_size_filter=False)),
    ):
        case_root = tmp_path / name
        case_root.mkdir()
        result, membership = _actual_task14_then_student(case_root, config)
        pseudo = [row for row in membership["members"] if row["source"] == "pseudo"]
        assert len(pseudo) == 1 and pseudo[0]["source_image_id"] == "unlabeled"
        assert (result.root / "labels" / "pseudo__unlabeled.txt").is_file()
        assert membership["provenance"]["pseudo_filter_policy"]["require_view_consistency"] is False


def _sealed_bounds_payload() -> dict[str, object]:
    return {
        "artifact_version": "1.0",
        "artifact_type": "sealed_aspect_ratio_bounds",
        "artifact_id": "matrix-fixture-bounds-v1",
        "class_registry_version": "1.0.1",
        "classes": [
            {"id": 0, "name": "Apple"}, {"id": 1, "name": "Banana"},
            {"id": 2, "name": "Orange"}, {"id": 3, "name": "Strawberry"},
            {"id": 4, "name": "Pineapple"},
        ],
        "provenance": {
            "source_split": "train_pool",
            "source_kind": "approved_aggregate_statistics",
            "contains_human_labels": False,
            "sealed": True,
        },
        "bounds": {str(index): [0.1, 10.0] for index in range(5)},
    }


def test_matrix_config_task14_output_composes_and_dry_runs_student(tmp_path: Path) -> None:
    """The full matrix policy hash must remain consumable by Student.

    This exercises actual Task-14 CLI publication with ``--matrix-config``;
    it is deliberately not a hand-written manifest.  The following Student
    dry run proves the complete policy hash is checked before its executable
    gates are extracted and compared with the matrix configuration.
    """
    inputs = _inputs(tmp_path)
    root = Path(__file__).resolve().parents[2]
    model = root / "configs" / "models" / "yolov8s_640.yaml"
    weights = tmp_path / "shared-pretrained.pt"
    weights.write_bytes(b"fixture-pretrained-weights")
    teacher = tmp_path / "teacher.yaml"
    teacher.write_text(yaml.safe_dump({
        "experiment_name": "teacher",
        "model_config": model.as_posix(),
        "pretrained_weights": weights.as_posix(),
        "initialization_policy": {
            "policy_id": "ssod_student_init_v1",
            "model_initialization": "shared_pretrained_weights",
            "comparison_group": "matrix-fixture",
        },
    }, sort_keys=False), encoding="utf-8")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    validation_pr = calibration / "validation-pr.json"
    validation_pr.write_text(json.dumps({"records": [
        {"class_id": class_id, "confidence": 0.95, "is_true_positive": True, "source_split": "validation"}
        for class_id in range(5)
    ]}), encoding="utf-8")
    bounds = calibration / "aspect-bounds.json"
    bounds.write_text(json.dumps(_sealed_bounds_payload()), encoding="utf-8")
    filter_root = tmp_path / "pseudo" / "filter"
    audit_report = tmp_path / "pseudo" / "audit" / "pseudo_audit.json"
    config = tmp_path / "student-matrix.yaml"
    config.write_text(yaml.safe_dump({
        "experiment_name": "matrix-student",
        "model_config": model.as_posix(),
        "pretrained_weights": weights.as_posix(),
        "artifact_root": (tmp_path / "artifacts").as_posix(),
        "source_root": inputs.source_root.as_posix(),
        "split_manifest": inputs.split_manifest.as_posix(),
        "human_images": inputs.human_images.as_posix(),
        "human_labels": inputs.human_labels.as_posix(),
        "validation_labels": inputs.validation_labels.as_posix(),
        "unlabeled_manifest": inputs.unlabeled_manifest.as_posix(),
        "candidates": inputs.candidates.as_posix(),
        "filter_audit": (filter_root / "audit.jsonl").as_posix(),
        "filter_decision_manifest": (filter_root / "decision_manifest.json").as_posix(),
        "pseudo_audit_report": audit_report.as_posix(),
        "seed": 42,
        "label_budget_percent": 20,
        "human_sample_probability": 0.5,
        "sampling_strategy": "balanced_50_50",
        "initialization_policy": {
            "policy_id": "ssod_student_init_v1",
            "model_initialization": "shared_pretrained_weights",
            "comparison_group": "matrix-fixture",
            "teacher_experiment_config": teacher.as_posix(),
            "teacher_run_id": "teacher",
        },
        "epochs": 1,
        "image_size": 640,
        "amp": True,
        "batch": 1,
        "pseudo_filter": {
            "policy_id": "trust_filter_v1",
            "use_per_class_thresholds": True,
            "require_view_consistency": True,
            "require_size_filter": True,
        },
        "filter_calibration": {
            "global_confidence": 0.50,
            "cross_view_iou": 0.60,
            "min_pixels_at_640": 1.0,
            "max_area_fraction": 0.90,
            "min_aspect_ratio": 0.10,
            "max_aspect_ratio": 10.0,
            "max_boxes_per_image": 20,
            "nms_iou": 0.60,
            "target_precision": 0.90,
            "threshold_minimum": 0.50,
            "threshold_maximum": 0.85,
            "validation_pr": validation_pr.as_posix(),
            "aspect_ratio_bounds": bounds.as_posix(),
        },
    }, sort_keys=False), encoding="utf-8")
    assert filter_pseudo_main([
        "--candidates", str(inputs.candidates),
        "--unlabeled-manifest", str(inputs.unlabeled_manifest),
        "--split-manifest", str(inputs.split_manifest),
        "--output", str(filter_root),
        "--matrix-config", str(config),
    ]) == 0
    decision = filter_root / "decision_manifest.json"
    audit = filter_root / "audit.jsonl"
    manifest = json.loads(decision.read_text(encoding="utf-8"))
    assert set(manifest["filter_policy"]) > {
        "policy_id", "use_per_class_thresholds", "require_view_consistency", "require_size_filter",
    }
    assert manifest["filter_policy_sha256"] == hashlib.sha256(json.dumps(
        manifest["filter_policy"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    audit_report.parent.mkdir(parents=True)
    audit_report.write_text(json.dumps({
        "teacher_run_id": "teacher",
        "pseudo_refresh": {"allowed": True},
        "provenance": {
            "candidate_artifact_sha256": hashlib.sha256(inputs.candidates.read_bytes()).hexdigest(),
            "filter_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "filter_decision_manifest_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
        },
    }), encoding="utf-8")
    assert train_student_main([
        "--config", str(config), "--dry-run", "--run-id", "matrix-filter-e2e",
    ]) == 0
    runs = list((tmp_path / "artifacts" / "runs").glob("dry-run-matrix-filter-e2e-*"))
    assert len(runs) == 1
