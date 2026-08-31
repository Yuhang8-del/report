from __future__ import annotations

import pytest

from fruit_ssod.evaluation.detection_metrics import DetectionMetrics
from fruit_ssod.evaluation.validation_selection import ValidationCandidate, ValidationSelectionError, select_validation_candidate


def _candidate(candidate_id: str, *, map50: float, recall: float, ap: float) -> ValidationCandidate:
    metrics = DetectionMetrics(map50, 0.4, 0.7, recall, 0.7, {index: ap for index in range(5)})
    return ValidationCandidate(candidate_id, __import__("pathlib").Path(f"C:/{candidate_id}"), {"mode": "direct", "image_size": 1024}, metrics, "a" * 64, "b" * 64, "c" * 64, "d" * 64)


def test_selection_prefers_floor_eligible_candidate_before_map50() -> None:
    result = select_validation_candidate(
        [_candidate("aggregate_better_but_missing_class", map50=0.84, recall=0.80, ap=0.49), _candidate("all_classes", map50=0.82, recall=0.70, ap=0.51)],
        per_class_ap50_floor=0.50,
    )

    assert result["selected_candidate_id"] == "all_classes"
    assert result["selection_status"] == "all_class_floor_pass"


def test_selection_uses_map50_then_recall_when_no_candidate_meets_floor() -> None:
    result = select_validation_candidate(
        [_candidate("higher_recall", map50=0.80, recall=0.80, ap=0.40), _candidate("lower_recall", map50=0.80, recall=0.70, ap=0.40)],
        per_class_ap50_floor=0.50,
    )

    assert result["selected_candidate_id"] == "higher_recall"
    assert result["selection_status"] == "no_candidate_met_per_class_floor"


def test_selection_rejects_cross_split_comparison() -> None:
    first = _candidate("one", map50=0.8, recall=0.7, ap=0.6)
    second = ValidationCandidate("two", first.run_dir, first.inference, first.metrics, first.checkpoint_sha256, first.dataset_yaml_sha256, first.validation_membership_sha256, "e" * 64)

    with pytest.raises(ValidationSelectionError, match="split fingerprints differ"):
        select_validation_candidate([first, second])


def test_selection_allows_different_training_yaml_when_validation_membership_matches() -> None:
    first = _candidate("full_images", map50=0.80, recall=0.70, ap=0.60)
    tiled = ValidationCandidate("tiled", first.run_dir, {"mode": "direct", "image_size": 1024}, first.metrics, first.checkpoint_sha256, "f" * 64, first.validation_membership_sha256, first.split_fingerprint)

    assert select_validation_candidate([first, tiled])["candidate_count"] == 2


def test_selection_rejects_changed_validation_membership() -> None:
    first = _candidate("one", map50=0.8, recall=0.7, ap=0.6)
    second = ValidationCandidate("two", first.run_dir, first.inference, first.metrics, first.checkpoint_sha256, first.dataset_yaml_sha256, "e" * 64, first.split_fingerprint)

    with pytest.raises(ValidationSelectionError, match="validation memberships differ"):
        select_validation_candidate([first, second])
