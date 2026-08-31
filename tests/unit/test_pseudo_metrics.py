from __future__ import annotations

import pytest

from fruit_ssod.evaluation.pseudo_metrics import (
    AuditBox,
    PseudoAuditError,
    calculate_pseudo_metrics,
    one_to_one_match,
    pseudo_refresh_allowed,
)


def _box(image: str, class_id: int, xyxy: tuple[float, float, float, float]) -> AuditBox:
    return AuditBox(image, class_id, xyxy, .9)


def test_one_to_one_matching_counts_duplicate_prediction_as_false_positive() -> None:
    truth = (_box("audit-a", 0, (10., 10., 50., 50.)), _box("audit-a", 1, (60., 10., 90., 50.)))
    predictions = (
        _box("audit-a", 0, (10., 10., 50., 50.)),
        _box("audit-a", 0, (11., 11., 49., 49.)),
        _box("audit-a", 2, (60., 10., 90., 50.)),
    )
    match = one_to_one_match(predictions, truth, iou_threshold=.5)
    assert match.tp == 1 and match.fp == 2 and match.fn == 1
    assert match.matched_prediction_indices == (0,)
    metrics, _ = calculate_pseudo_metrics(predictions, truth)
    assert metrics.per_class[0]["tp"] == 1 and metrics.per_class[0]["fp"] == 1
    assert metrics.per_class[1]["fn"] == 1 and metrics.per_class[2]["fp"] == 1
    assert metrics.overall == {"tp": 1, "fp": 2, "fn": 1, "precision": pytest.approx(1 / 3), "recall": pytest.approx(.5), "f1": pytest.approx(.4)}


def test_matching_maximizes_true_positive_count_before_iou() -> None:
    """A greedy best-IoU pair would strand the second prediction here.

    ``prediction[0]`` overlaps both labels and has the single best IoU with
    ``truth[0]``.  Taking that edge leaves only one TP, whereas assigning it
    to ``truth[1]`` allows ``prediction[1]`` to match ``truth[0]`` for two
    true positives.
    """
    truth = (
        _box("audit-a", 0, (0., 0., 10., 10.)),
        _box("audit-a", 0, (4., 0., 14., 10.)),
    )
    predictions = (
        _box("audit-a", 0, (0., 0., 12., 10.)),  # IoU .833 to first, .571 to second
        _box("audit-a", 0, (0., 0., 7., 10.)),   # IoU .700 to first only
    )
    match = one_to_one_match(predictions, truth, iou_threshold=.5)
    assert match.tp == 2 and match.fp == 0 and match.fn == 0
    assert match.matched_prediction_indices == (0, 1)
    assert match.matched_ground_truth_indices == (0, 1)


def test_matching_is_class_and_image_aware_and_reports_all_five_classes() -> None:
    metrics, match = calculate_pseudo_metrics((_box("other", 0, (0., 0., 10., 10.)),), (_box("audit", 0, (0., 0., 10., 10.)),))
    assert match.tp == 0 and match.fp == 1 and match.fn == 1
    assert set(metrics.per_class) == {0, 1, 2, 3, 4}
    assert metrics.per_class[4]["precision"] == 0.0


def test_pseudo_refresh_gate_stops_below_ninety_percent_precision() -> None:
    passing, _ = calculate_pseudo_metrics((_box("audit", 0, (0., 0., 10., 10.)),), (_box("audit", 0, (0., 0., 10., 10.)),))
    failing, _ = calculate_pseudo_metrics((_box("audit", 0, (20., 20., 30., 30.)),), (_box("audit", 0, (0., 0., 10., 10.)),))
    assert pseudo_refresh_allowed(passing)
    assert not pseudo_refresh_allowed(failing)
    with pytest.raises(PseudoAuditError, match="minimum audit precision"):
        pseudo_refresh_allowed(passing, minimum_precision=1.1)


def test_matching_rejects_loose_or_invalid_threshold_input() -> None:
    with pytest.raises(PseudoAuditError, match="IoU threshold"):
        one_to_one_match((), (), iou_threshold=0)
    with pytest.raises(PseudoAuditError, match="loose boxes"):
        one_to_one_match((object(),), ())  # type: ignore[arg-type]
