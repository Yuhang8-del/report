"""Run and evaluate the complete known/unknown box-level fruit pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.open_world.box_clustering import BoxClusterInput, BoxClusterer, fit_box_cluster_model
from fruit_ssod.open_world.box_metrics import UnknownEvaluationSample, UnknownGroundTruth, evaluate_unknown_boxes
from fruit_ssod.open_world.box_proposals import UltralyticsObjectnessProposalProvider, box_iou
from fruit_ssod.open_world.contracts import UnknownProposal, UnknownProposalRequest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--student-weights", type=Path, required=True)
    result.add_argument("--objectness-weights", type=Path, required=True)
    result.add_argument("--encoder-checkpoint", type=Path, required=True)
    result.add_argument("--public-manifest", type=Path, required=True)
    result.add_argument("--protected-truth", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--known-confidence", type=float, default=0.50)
    result.add_argument("--objectness-threshold", type=float, default=0.15)
    result.add_argument("--known-iou-threshold", type=float, default=0.35)
    result.add_argument("--evaluation-iou", type=float, default=0.50)
    result.add_argument("--image-size", type=int, default=768)
    result.add_argument("--clusters", type=int, default=6)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="0")
    return result


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _truth_boxes(record: dict, width: int, height: int) -> tuple[UnknownGroundTruth, ...]:
    result = []
    for box in record["boxes"]:
        xc, yc = float(box["x_center"]), float(box["y_center"])
        bw, bh = float(box["width"]), float(box["height"])
        result.append(
            UnknownGroundTruth(
                image_id=record["image_id"],
                category=record["category"],
                xyxy=(
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ),
            )
        )
    return tuple(result)


def _best_truth(proposal: UnknownProposal, truth: Sequence[UnknownGroundTruth], threshold: float) -> str | None:
    candidates = [(box_iou(proposal.xyxy, item.xyxy), item.category) for item in truth]
    if not candidates:
        return None
    overlap, category = max(candidates)
    return category if overlap >= threshold else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = _json(args.public_manifest.resolve(strict=True))["records"]
    protected = _json(args.protected_truth.resolve(strict=True))["records"]
    protected_by_id = {record["image_id"]: record for record in protected}

    known_detector = UltralyticsDetectorAdapter(weights_path=args.student_weights.resolve(strict=True))
    proposal_provider = UltralyticsObjectnessProposalProvider(
        weights_path=args.objectness_weights.resolve(strict=True),
        objectness_threshold=args.objectness_threshold,
        known_iou_threshold=args.known_iou_threshold,
        image_size=args.image_size,
        device=args.device,
    )
    samples: list[UnknownEvaluationSample] = []
    cluster_inputs: list[BoxClusterInput] = []
    proposal_by_id: dict[str, UnknownProposal] = {}
    truth_by_image: dict[str, tuple[UnknownGroundTruth, ...]] = {}
    predictions_path = args.output_dir / "box_predictions.jsonl"
    # Persist each completed image immediately so a long Windows inference run
    # still leaves auditable progress if an external interruption occurs.
    predictions_path.write_text("", encoding="utf-8")
    for index, member in enumerate(public, start=1):
        image_path = Path(member["image_path"]).resolve(strict=True)
        known = known_detector.predict(image_path, confidence=args.known_confidence, nms_iou=0.6)
        request = UnknownProposalRequest(
            image_path=image_path,
            known_detections=known,
            source_run_id=args.output_dir.name,
            evidence={"image_id": member["image_id"], "public_manifest": str(args.public_manifest.resolve())},
        )
        proposals = proposal_provider.propose_unknowns(request)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            truth = _truth_boxes(protected_by_id[member["image_id"]], image.width, image.height)
        truth_by_image[member["image_id"]] = truth
        for proposal in proposals:
            proposal_by_id[proposal.proposal_id] = proposal
            cluster_inputs.append(
                BoxClusterInput(
                    proposal_id=proposal.proposal_id,
                    image_id=proposal.image_id,
                    image_path=str(image_path),
                    xyxy=proposal.xyxy,
                    split=member["split"],
                    novelty_score=proposal.novelty_score,
                )
            )
        if member["split"] == "holdout":
            samples.append(UnknownEvaluationSample(member["image_id"], proposals, truth, known))
        output_row = {
                "image_id": member["image_id"],
                "image_path": str(image_path),
                "split": member["split"],
                "known_detections": [
                    {
                        "class_id": item.class_id,
                        "class_name": item.class_name,
                        "confidence": item.confidence,
                        "xyxy": list(item.xyxy),
                    }
                    for item in known
                ],
                "unknown_proposals": [
                    {
                        "proposal_id": item.proposal_id,
                        "xyxy": list(item.xyxy),
                        "novelty_score": item.novelty_score,
                        "evidence": dict(item.evidence),
                    }
                    for item in proposals
                ],
                "progress": {"completed": index, "total": len(public)},
            }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        if index % 25 == 0 or index == len(public):
            print(json.dumps({"completed": index, "total": len(public)}, ensure_ascii=False), flush=True)

    metrics = evaluate_unknown_boxes(samples, iou_threshold=args.evaluation_iou)
    (args.output_dir / "unknown_box_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    cluster_dir = args.output_dir / "box_clusters"
    discovery_assignments = fit_box_cluster_model(
        cluster_inputs,
        encoder_checkpoint=args.encoder_checkpoint.resolve(strict=True),
        output_dir=cluster_dir,
        clusters=args.clusters,
        seed=args.seed,
        device=f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device),
    )
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    for assignment in discovery_assignments:
        proposal = proposal_by_id[assignment.proposal_id]
        category = _best_truth(proposal, truth_by_image[proposal.image_id], args.evaluation_iou)
        if category is not None:
            votes[assignment.cluster_id][category] += 1
    candidate_names = {cluster: counts.most_common(1)[0][0] for cluster, counts in votes.items() if counts}
    (cluster_dir / "posthoc_cluster_names.json").write_text(
        json.dumps({str(key): value for key, value in sorted(candidate_names.items())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    clusterer = BoxClusterer(
        encoder_checkpoint=args.encoder_checkpoint.resolve(strict=True),
        cluster_model=cluster_dir / "box_cluster_model.npz",
        candidate_names=candidate_names,
        device=f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device),
    )
    holdout_inputs = [record for record in cluster_inputs if record.split == "holdout"]
    holdout_assignments = clusterer.assign(holdout_inputs)
    semantic_total = 0
    semantic_correct = 0
    assignment_rows = []
    for assignment in holdout_assignments:
        proposal = proposal_by_id[assignment.proposal_id]
        truth_name = _best_truth(proposal, truth_by_image[proposal.image_id], args.evaluation_iou)
        if truth_name is not None:
            semantic_total += 1
            semantic_correct += assignment.candidate_name == truth_name
        assignment_rows.append({**asdict(assignment), "matched_truth": truth_name})
    (cluster_dir / "holdout_box_cluster_assignments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in assignment_rows), encoding="utf-8"
    )
    final = {
        "schema_version": "1.0",
        "artifact_type": "box_level_open_world_fruit_evaluation",
        "known_registry": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"],
        "unknown_evaluation_categories": ["Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon"],
        "unknown_box_metrics": metrics,
        "box_clustering": {
            "discovery_proposals": len(discovery_assignments),
            "holdout_proposals": len(holdout_assignments),
            "posthoc_cluster_names": candidate_names,
            "matched_holdout_proposals": semantic_total,
            "posthoc_semantic_accuracy": semantic_correct / max(1, semantic_total),
        },
        "limitations": [
            "Unknown semantic names are post-hoc cluster mappings and require review before registry updates.",
            "The class-agnostic proposal baseline is trained only on the five known fruit classes.",
        ],
    }
    (args.output_dir / "open_world_box_results.json").write_text(
        json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
