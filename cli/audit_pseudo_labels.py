"""Audit pseudo labels against Task 8's sealed pseudo_audit partition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fruit_ssod.detection.adapter import DetectorAdapterError
from fruit_ssod.detection.ultralytics_backend import UltralyticsDetectorAdapter
from fruit_ssod.evaluation.audit_candidates import AuditCandidateError, generate_audit_candidates
from fruit_ssod.evaluation.pseudo_metrics import (
    AuditBox,
    PseudoAuditError,
    calculate_pseudo_metrics,
    file_sha256,
    load_audit_candidates,
    load_bound_filter_predictions,
    load_sealed_pseudo_audit,
    original_view_predictions,
    pseudo_refresh_allowed,
)
from fruit_ssod.pseudo.generator import PseudoGenerationError, write_pseudo_candidates
from fruit_ssod.pseudo.thresholds import PerClassThresholds, ThresholdSelectionError, select_per_class_thresholds
from fruit_ssod.pseudo.trust_filter import (
    ImageGeometry,
    TrustFilter,
    TrustFilterConfig,
    TrustFilterError,
    load_aspect_ratio_bounds_artifact,
    write_trust_filter_outputs,
)
from fruit_ssod.reporting.pseudo_figures import render_pseudo_audit_examples


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit raw and filtered pseudo labels against Task 8 sealed pseudo_audit labels.")
    parser.add_argument("--audit-labels", type=Path, required=True, help="Task 8 protected_splits/pseudo_audit_labels.json; accepted by this audit command only.")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Paired immutable Task 8 split_manifest.json.")
    parser.add_argument("--candidates", type=Path, help="Existing Task 13-format raw candidate envelope, or new output path with --prepare-from-teacher.")
    parser.add_argument("--filter-audit", type=Path, help="Paired Task 14 complete audit.jsonl decisions for --candidates.")
    parser.add_argument("--filter-decision-manifest", type=Path, help="Paired Task 14 decision_manifest.json; defaults beside --filter-audit.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for immutable audit metrics and diagnostic figures.")
    parser.add_argument("--image-root", type=Path, help="Optional sealed pseudo-audit image root used only to render examples.")
    parser.add_argument("--prepare-from-teacher", action="store_true", help="Audit-only preparation: infer dual-view candidates and apply one matrix policy before scoring.")
    parser.add_argument("--weights", type=Path, help="Completed Teacher weights/best.pt; required with --prepare-from-teacher.")
    parser.add_argument("--teacher-run-id", help="Completed Teacher run ID; required with --prepare-from-teacher.")
    parser.add_argument("--matrix-config", type=Path, help="Task 17 YAML supplying the exact executable filter policy; required with --prepare-from-teacher.")
    parser.add_argument("--filter-output", type=Path, help="New Task 14-format output directory for audit-only filter decisions; required with --prepare-from-teacher.")
    parser.add_argument("--confidence", type=float, default=0.01, help="Teacher confidence for audit-only candidate inference.")
    parser.add_argument("--iou", type=float, default=0.50, help="One-to-one same-class IoU matching threshold.")
    parser.add_argument("--minimum-precision", type=float, default=0.90, help="Pseudo refresh is stopped below this post-filter audit precision.")
    return parser


def _example_rows(before: tuple[AuditBox, ...], after: tuple[AuditBox, ...], truth: tuple[AuditBox, ...], *, iou: float, images: Mapping[str, Any]) -> dict[str, tuple[tuple[Any, AuditBox], ...]]:
    """Select deterministic diagnostics from matching results without labels leaking elsewhere."""
    _, before_match = calculate_pseudo_metrics(before, truth, iou_threshold=iou)
    _, after_match = calculate_pseudo_metrics(after, truth, iou_threshold=iou)
    kept = tuple((images[item.source_image_id], item) for item in after)
    # ``AuditBox`` values can be byte-identical (for example, two detections
    # later split by NMS).  Set-like ``item not in after`` loses occurrences
    # and could hide a rejected duplicate.  Consume one retained occurrence
    # at a time so every raw prediction remains diagnosable.
    retained_counts = Counter(after)
    rejected_rows: list[tuple[Any, AuditBox]] = []
    for item in before:
        if retained_counts[item]:
            retained_counts[item] -= 1
        else:
            rejected_rows.append((images[item.source_image_id], item))
    rejected = tuple(rejected_rows)
    false_positive = tuple((images[after[index].source_image_id], after[index]) for index in after_match.unmatched_prediction_indices)
    missed = tuple((images[truth[index].source_image_id], truth[index]) for index in after_match.unmatched_ground_truth_indices)
    # ``before_match`` is intentionally evaluated too: it makes one-to-one
    # matching a required computation on both sides of the filter, even if the
    # before diagnostic displays all raw original-view boxes.
    _ = before_match
    return {"kept": kept, "rejected": rejected, "false_positive": false_positive, "missed": missed}


def _preflight_output(root: Path) -> Path:
    destination = root.resolve(strict=False)
    if destination.exists():
        raise PseudoAuditError(f"Problem: pseudo audit output already exists. Likely cause: {destination} would be overwritten. Remediation: choose a new empty output directory.")
    ancestor = destination.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise PseudoAuditError(f"Problem: pseudo audit output has a file ancestor. Likely cause: {ancestor} is not a directory. Remediation: choose an output path below directories only.")
        ancestor = ancestor.parent
    return destination


def _write_atomic(payload: Mapping[str, Any], output: Path, examples: Mapping[str, tuple[tuple[Any, AuditBox], ...]], image_root: Path | None) -> Path:
    root = _preflight_output(output)
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        (temporary / "pseudo_audit.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Renderer refuses existing dirs; use its dedicated staging subtree.
        render_pseudo_audit_examples(examples, temporary / "examples", image_root=image_root)
        os.replace(temporary, root); temporary = None
        return root / "pseudo_audit.json"
    except PseudoAuditError:
        raise
    except OSError as error:
        raise PseudoAuditError(f"Problem: pseudo audit outputs could not be published atomically. Likely cause: {error}. Remediation: choose a new writable output path.") from error
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _read_pr(path: Path) -> list[Mapping[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PseudoAuditError(f"Problem: validation PR input cannot be read. Likely cause: {error}. Remediation: restore the sealed validation prediction/outcome export.") from error
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise PseudoAuditError("Problem: validation PR input has no records array. Likely cause: calibration evidence is malformed. Remediation: restore the sealed validation prediction/outcome export.")
    return [dict(row) for row in records]


def _prepare_filter_for_audit(*, matrix_config: Path, candidates: Path, images: Mapping[str, Any], output: Path) -> Path:
    """Apply the exact matrix policy to audit-only candidates without exposing labels."""
    from fruit_ssod.training.ssod_matrix import SsodMatrixError, load_effective_filter_policy

    try:
        policy = load_effective_filter_policy(matrix_config)
    except SsodMatrixError as error:
        raise PseudoAuditError(str(error)) from error
    try:
        gates = {key: policy[key] for key in ("policy_id", "use_per_class_thresholds", "require_view_consistency", "require_size_filter")}
        filter_config, calibration = policy["filter_config"], policy["threshold_calibration"]
        if not isinstance(filter_config, Mapping) or not isinstance(calibration, Mapping):
            raise ValueError("matrix policy has no filter configuration")
        global_policy = gates["policy_id"] == "global_threshold_v1"
        if global_policy:
            thresholds = PerClassThresholds({class_id: float(filter_config["global_confidence"]) for class_id in range(5)}, minimum=0.0, maximum=1.0)
            bounds = None
        else:
            validation_pr, aspect = calibration.get("validation_pr"), policy.get("aspect_ratio_bounds")
            if not isinstance(validation_pr, Mapping) or not isinstance(validation_pr.get("path"), str) or not isinstance(aspect, Mapping) or not isinstance(aspect.get("path"), str):
                raise ValueError("matrix policy has no sealed calibration paths")
            thresholds = select_per_class_thresholds(
                _read_pr(Path(validation_pr["path"])),
                target_precision=float(calibration["target_precision"]),
                minimum=float(calibration["minimum"]),
                maximum=float(calibration["maximum"]),
            )
            bounds = load_aspect_ratio_bounds_artifact(Path(aspect["path"]))
        geometries = {source_id: ImageGeometry(source_id, image.width, image.height) for source_id, image in images.items()}
        config = TrustFilterConfig(
            global_confidence=float(filter_config["global_confidence"]), cross_view_iou=float(filter_config["cross_view_iou"]),
            min_pixels_at_640=float(filter_config["min_pixels_at_640"]), max_area_fraction=float(filter_config["max_area_fraction"]),
            min_aspect_ratio=float(filter_config["min_aspect_ratio"]), max_aspect_ratio=float(filter_config["max_aspect_ratio"]),
            max_boxes_per_image=int(filter_config["max_boxes_per_image"]), nms_iou=float(filter_config["nms_iou"]),
            aspect_ratio_bounds_artifact=bounds, policy_id=str(gates["policy_id"]),
            use_per_class_thresholds=bool(gates["use_per_class_thresholds"]),
            require_view_consistency=bool(gates["require_view_consistency"]),
            require_size_filter=bool(gates["require_size_filter"]), effective_policy=policy,
        )
        result = TrustFilter(thresholds, config=config).filter_envelope(candidates, geometries)
        _, audit_path = write_trust_filter_outputs(result, output)
        return audit_path
    except (TrustFilterError, ThresholdSelectionError, ValueError) as error:
        raise PseudoAuditError(str(error)) from error


def _matrix_teacher_run_id(path: Path) -> str:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise PseudoAuditError(f"Problem: SSOD matrix configuration cannot be read. Likely cause: {error}. Remediation: restore the checked-in Task 17 YAML.") from error
    policy = payload.get("initialization_policy") if isinstance(payload, Mapping) else None
    teacher = policy.get("teacher_run_id") if isinstance(policy, Mapping) else None
    if not isinstance(teacher, str) or not teacher:
        raise PseudoAuditError("Problem: SSOD matrix has no Teacher run ID. Likely cause: initialization policy is malformed. Remediation: restore the canonical Task 17 YAML.")
    return teacher


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        audit = load_sealed_pseudo_audit(args.audit_labels.resolve(strict=False), args.split_manifest.resolve(strict=False))
        candidate_path, filter_audit = args.candidates, args.filter_audit
        if args.prepare_from_teacher:
            required = {"--candidates": candidate_path, "--weights": args.weights, "--teacher-run-id": args.teacher_run_id, "--matrix-config": args.matrix_config, "--filter-output": args.filter_output, "--image-root": args.image_root}
            missing = [name for name, value in required.items() if value is None]
            if missing:
                parser.error(f"{', '.join(missing)} is required with --prepare-from-teacher")
            assert candidate_path is not None and args.weights is not None and args.teacher_run_id is not None and args.matrix_config is not None and args.filter_output is not None and args.image_root is not None
            if args.teacher_run_id != _matrix_teacher_run_id(args.matrix_config):
                raise PseudoAuditError("Problem: audit Teacher differs from matrix configuration. Likely cause: --teacher-run-id does not match initialization_policy.teacher_run_id. Remediation: use the Teacher declared by the exact Task 17 config.")
            detector = UltralyticsDetectorAdapter(weights_path=args.weights, source_model=str(args.weights))
            generated = generate_audit_candidates(audit.images, detector=detector, teacher_run_id=args.teacher_run_id, image_root=args.image_root, confidence=args.confidence)
            candidate_path = write_pseudo_candidates(generated, candidate_path)
            filter_audit = _prepare_filter_for_audit(matrix_config=args.matrix_config, candidates=candidate_path, images=audit.images, output=args.filter_output)
        elif candidate_path is not None and filter_audit is None and args.matrix_config is not None and args.filter_output is not None:
            # A prior audit-only candidate publication may have completed
            # before a recoverable policy/configuration failure.  Reuse that
            # immutable envelope rather than rerunning or overwriting it.
            filter_audit = _prepare_filter_for_audit(matrix_config=args.matrix_config, candidates=candidate_path, images=audit.images, output=args.filter_output)
        if candidate_path is None or filter_audit is None:
            parser.error("--candidates and --filter-audit are required unless --prepare-from-teacher is used")
        teacher_run_id, candidates, candidate_sha256 = load_audit_candidates(candidate_path.resolve(strict=False), audit)
        before = original_view_predictions(candidates, audit)
        after = load_bound_filter_predictions(
            filter_audit.resolve(strict=False), candidates, audit,
            teacher_run_id=teacher_run_id, candidate_artifact_sha256=candidate_sha256,
            decision_manifest_path=args.filter_decision_manifest.resolve(strict=False) if args.filter_decision_manifest else None,
        )
        before_metrics, _ = calculate_pseudo_metrics(before, audit.labels, iou_threshold=args.iou)
        after_metrics, _ = calculate_pseudo_metrics(after, audit.labels, iou_threshold=args.iou)
        refresh_allowed = pseudo_refresh_allowed(after_metrics, minimum_precision=args.minimum_precision)
        decision_manifest = args.filter_decision_manifest.resolve(strict=False) if args.filter_decision_manifest else filter_audit.resolve(strict=False).with_name("decision_manifest.json")
        try:
            decision_payload = json.loads(decision_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PseudoAuditError(f"Problem: pseudo-audit decision manifest cannot be read. Likely cause: {error}. Remediation: restore the Task 14 decision manifest paired with --filter-audit.") from error
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "teacher_run_id": teacher_run_id,
            "iou_threshold": args.iou,
            "minimum_precision": args.minimum_precision,
            "pseudo_refresh": {"allowed": refresh_allowed, "reason": "precision_at_or_above_threshold" if refresh_allowed else "stopped_precision_below_threshold"},
            "metrics": {"before_filter": before_metrics.mapping(), "after_filter": after_metrics.mapping()},
            "provenance": {
                "pseudo_audit_split_fingerprint": audit.split_fingerprint,
                "pseudo_audit_labels_sha256": audit.labels_sha256,
                "candidate_artifact_sha256": candidate_sha256,
                "filter_audit_sha256": file_sha256(filter_audit.resolve(strict=False)),
                "filter_decision_manifest_sha256": file_sha256(decision_manifest),
                "candidate_count": len(candidates), "before_original_prediction_count": len(before), "after_accepted_prediction_count": len(after),
            },
        }
        if isinstance(decision_payload, Mapping) and isinstance(decision_payload.get("filter_policy"), Mapping):
            payload["filter_policy"] = dict(decision_payload["filter_policy"])
            payload["filter_policy_sha256"] = decision_payload.get("filter_policy_sha256")
        output = _write_atomic(payload, args.output, _example_rows(before, after, audit.labels, iou=args.iou, images=audit.images), args.image_root.resolve(strict=False) if args.image_root else None)
    except (PseudoAuditError, AuditCandidateError, PseudoGenerationError, DetectorAdapterError, ValueError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"output": str(output), "refresh_allowed": refresh_allowed, "post_filter_precision": after_metrics.overall["precision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
