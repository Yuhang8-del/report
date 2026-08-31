from __future__ import annotations

import pytest

from fruit_ssod.evaluation.acceptance import AcceptanceError, evaluate_acceptance


def _aggregate(trust: float | None, baseline: float | None, *, complete: bool = True) -> dict:
    def group(value: float | None) -> dict:
        return {"complete": complete, "metrics": {"map50": {"mean": value}}}
    return {"summary": {"main_groups": {"trust_main": group(trust), "supervised_20": group(baseline)}}}


def test_acceptance_passes_only_with_both_required_margins() -> None:
    result = evaluate_acceptance(_aggregate(0.83, 0.80))
    assert result["status"] == "pass"
    assert result["requirements"]["improvement_over_supervised_20_map50"]["observed"] == pytest.approx(0.03)


def test_acceptance_fails_closed_for_incomplete_or_low_result() -> None:
    incomplete = evaluate_acceptance(_aggregate(None, 0.80, complete=False))
    assert incomplete["status"] == "fail"
    below = evaluate_acceptance(_aggregate(0.79, 0.70))
    assert below["status"] == "fail"


def test_acceptance_rejects_wrong_aggregate_shape() -> None:
    with pytest.raises(AcceptanceError, match="aggregation is incomplete"):
        evaluate_acceptance({})


def test_acceptance_rejects_boolean_as_a_metric() -> None:
    result = evaluate_acceptance(_aggregate(True, 0.80))
    assert result["status"] == "fail"
