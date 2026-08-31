from __future__ import annotations

import pytest

from fruit_ssod.pseudo.thresholds import ThresholdSelectionError, select_per_class_thresholds


def _row(class_id: int, confidence: float, is_true_positive: bool) -> dict[str, object]:
    return {"class_id": class_id, "confidence": confidence, "is_true_positive": is_true_positive, "source_split": "validation"}


def test_thresholds_target_precision_and_are_clamped_per_class() -> None:
    records = [_row(0, 0.51, True), _row(0, 0.50, False), _row(1, 0.99, False)]
    thresholds = select_per_class_thresholds(records, target_precision=0.90)
    assert thresholds.for_class(0) == pytest.approx(0.51)
    assert thresholds.for_class(1) == 0.85
    assert all(0.50 <= thresholds.for_class(class_id) <= 0.85 for class_id in range(5))


@pytest.mark.parametrize("row", [
    {"class_id": 0, "confidence": 0.8, "is_true_positive": True, "source_split": "test"},
    {"class_id": 0, "confidence": 0.8, "is_true_positive": True, "annotation_path": "hidden.json"},
])
def test_threshold_selection_rejects_non_validation_or_label_bearing_evidence(row: dict[str, object]) -> None:
    with pytest.raises(ThresholdSelectionError, match="validation|unsupported"):
        select_per_class_thresholds([row])


def test_threshold_selection_is_input_order_independent() -> None:
    rows = [_row(3, 0.72, True), _row(3, 0.71, True), _row(3, 0.70, False)]
    assert select_per_class_thresholds(rows).values == select_per_class_thresholds(list(reversed(rows))).values
