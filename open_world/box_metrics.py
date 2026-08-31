"""Evaluation metrics for box-level unknown fruit proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.open_world.box_proposals import box_iou, known_overlap_count
from fruit_ssod.open_world.contracts import UnknownProposal


@dataclass(frozen=True)
class UnknownGroundTruth:
    image_id: str
    category: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class UnknownEvaluationSample:
    image_id: str
    proposals: tuple[UnknownProposal, ...]
    ground_truth: tuple[UnknownGroundTruth, ...]
    known_detections: tuple[DetectionRecord, ...] = ()


def _match_sample(
    sample: UnknownEvaluationSample,
    *,
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int, dict[str, tuple[int, int]]]:
    unmatched = set(range(len(sample.ground_truth)))
    scored: list[tuple[float, bool]] = []
    true_by_category: dict[str, int] = {}
    total_by_category: dict[str, int] = {}
    for truth in sample.ground_truth:
        total_by_category[truth.category] = total_by_category.get(truth.category, 0) + 1
    for proposal in sorted(sample.proposals, key=lambda item: item.novelty_score, reverse=True):
        candidates = [
            (box_iou(proposal.xyxy, sample.ground_truth[index].xyxy), index)
            for index in unmatched
        ]
        if candidates:
            best_iou, best_index = max(candidates)
            if best_iou >= iou_threshold:
                unmatched.remove(best_index)
                category = sample.ground_truth[best_index].category
                true_by_category[category] = true_by_category.get(category, 0) + 1
                scored.append((proposal.novelty_score, True))
                continue
        scored.append((proposal.novelty_score, False))
    category_counts = {
        category: (true_by_category.get(category, 0), total)
        for category, total in total_by_category.items()
    }
    return scored, len(unmatched), category_counts


def _average_precision(scored: Sequence[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        return 0.0
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    precision: list[float] = []
    recall: list[float] = []
    for _, is_true in ranked:
        if is_true:
            true_positives += 1
        else:
            false_positives += 1
        precision.append(true_positives / max(1, true_positives + false_positives))
        recall.append(true_positives / ground_truth_count)
    return sum(max((p for p, r in zip(precision, recall) if r >= threshold), default=0.0) for threshold in [i / 100 for i in range(101)]) / 101.0


def evaluate_unknown_boxes(
    samples: Iterable[UnknownEvaluationSample],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Return U-Precision, U-Recall, U-F1, U-AP50, A-OSE and category recall."""
    sample_list = tuple(samples)
    scored: list[tuple[float, bool]] = []
    false_negatives = 0
    ground_truth_count = 0
    absolute_open_set_error = 0
    per_category: dict[str, list[int]] = {}
    for sample in sample_list:
        current, missing, category_counts = _match_sample(sample, iou_threshold=iou_threshold)
        scored.extend(current)
        false_negatives += missing
        ground_truth_count += len(sample.ground_truth)
        absolute_open_set_error += known_overlap_count(
            sample.known_detections,
            (item.xyxy for item in sample.ground_truth),
            iou_threshold=iou_threshold,
        )
        for category, (matched, total) in category_counts.items():
            aggregate = per_category.setdefault(category, [0, 0])
            aggregate[0] += matched
            aggregate[1] += total
    true_positives = sum(is_true for _, is_true in scored)
    false_positives = len(scored) - true_positives
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "schema_version": "1.0",
        "iou_threshold": iou_threshold,
        "image_count": len(sample_list),
        "ground_truth_boxes": ground_truth_count,
        "proposal_boxes": len(scored),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "u_precision": precision,
        "u_recall": recall,
        "u_f1": f1,
        "u_ap50": _average_precision(scored, ground_truth_count),
        "a_ose": absolute_open_set_error,
        "per_category_recall": {
            category: {"matched": values[0], "total": values[1], "recall": values[0] / max(1, values[1])}
            for category, values in sorted(per_category.items())
        },
        "samples": [
            {
                "image_id": sample.image_id,
                "proposal_count": len(sample.proposals),
                "ground_truth_count": len(sample.ground_truth),
                "known_detection_count": len(sample.known_detections),
            }
            for sample in sample_list
        ],
    }
