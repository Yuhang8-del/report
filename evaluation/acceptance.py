"""Explicit project acceptance gate evaluated only from Task 18 summaries."""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

from fruit_ssod.evaluation.aggregate import ResultAggregationError, _freeze


class AcceptanceError(ValueError):
    """Raised when the aggregate does not have the required Task 18 shape."""


def evaluate_acceptance(aggregate: Mapping[str, Any], *, target_map50: float = 0.80, minimum_improvement: float = 0.03) -> Mapping[str, Any]:
    """Return a fail-closed, immutable acceptance evidence envelope.

    The requested 80% accuracy is interpreted by the agreed primary metric,
    held-out primary-test mean mAP@0.5.  A missing seed is a visible failure,
    never an average over whichever runs happened to succeed.
    """
    try:
        groups = aggregate["summary"]["main_groups"]
        trust = groups["trust_main"]
        baseline = groups["supervised_20"]
        trust_mean = trust["metrics"]["map50"]["mean"]
        baseline_mean = baseline["metrics"]["map50"]["mean"]
        trust_complete = trust["complete"] is True
        baseline_complete = baseline["complete"] is True
    except (KeyError, TypeError) as error:
        raise AcceptanceError("Problem: aggregation is incomplete. Likely cause: Task 18 fields are missing. Remediation: aggregate immutable canonical run evidence before checking acceptance.") from error
    valid_numbers = (
        isinstance(trust_mean, (int, float)) and not isinstance(trust_mean, bool)
        and isinstance(baseline_mean, (int, float)) and not isinstance(baseline_mean, bool)
        and math.isfinite(float(trust_mean)) and math.isfinite(float(baseline_mean))
        and 0.0 <= float(trust_mean) <= 1.0 and 0.0 <= float(baseline_mean) <= 1.0
    )
    improvement = float(trust_mean) - float(baseline_mean) if valid_numbers else None
    threshold_pass = bool(trust_complete and valid_numbers and float(trust_mean) >= target_map50)
    # Decimal threshold literals such as 0.03 are not represented exactly by
    # binary floats.  Treat values equal within a tiny numerical round-off as
    # satisfying the published boundary, without relaxing a meaningful score.
    improvement_pass = bool(trust_complete and baseline_complete and improvement is not None and improvement + 1e-12 >= minimum_improvement)
    passed = threshold_pass and improvement_pass
    return _freeze({
        "schema_version": "1.0", "protocol": "task18_acceptance_v1",
        "status": "pass" if passed else "fail",
        "requirements": {
            "trust_main_three_seed_complete": {"required": True, "observed": trust_complete, "passed": trust_complete},
            "supervised_20_three_seed_complete": {"required": True, "observed": baseline_complete, "passed": baseline_complete},
            "final_mean_map50": {"threshold": target_map50, "observed": trust_mean, "passed": threshold_pass},
            "improvement_over_supervised_20_map50": {"threshold": minimum_improvement, "observed": improvement, "passed": improvement_pass},
        },
        "message": "All project acceptance requirements passed." if passed else "Acceptance failed or is incomplete; retain all failed/incomplete rows in the result package.",
    })
