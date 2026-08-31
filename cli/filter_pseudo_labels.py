"""Filter one Task 13 candidate envelope into auditable pseudo YOLO labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from fruit_ssod.pseudo.generator import PseudoGenerationError, load_unlabeled_manifest
from fruit_ssod.pseudo.thresholds import PerClassThresholds, ThresholdSelectionError, select_per_class_thresholds
from fruit_ssod.pseudo.trust_filter import (
    ImageGeometry,
    TrustFilter,
    TrustFilterConfig,
    TrustFilterError,
    load_aspect_ratio_bounds_artifact,
    write_trust_filter_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply global or validation-calibrated Trust Filter rules to Task 13 candidates.")
    parser.add_argument("--candidates", type=Path, required=True, help="Unmodified Task 13 candidates JSON envelope.")
    parser.add_argument("--unlabeled-manifest", type=Path, required=True, help="Task 8 no-label unlabeled.json used only for image dimensions.")
    parser.add_argument("--split-manifest", type=Path, help="Paired Task 8 split_manifest.json; defaults beside unlabeled manifest.")
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory for labels/, audit.jsonl, and sealed decision_manifest.json.")
    parser.add_argument("--mode", choices=("global", "trust"), default="trust", help="global uses one threshold; trust calibrates five validation thresholds.")
    parser.add_argument("--global-confidence", type=float, default=0.50, help="Global candidate confidence gate and global baseline threshold in [0, 1]; global mode does not use the trust-mode 0.50-0.85 per-class clamp.")
    parser.add_argument("--validation-pr", type=Path, help="Validation prediction/outcome JSON with records array; required in trust mode.")
    parser.add_argument("--aspect-ratio-bounds", type=Path, help="Sealed aggregate-only five-class aspect-ratio bounds JSON; required in trust mode and recorded in audit.jsonl.")
    parser.add_argument("--target-precision", type=float, default=0.90, help="Per-class validation precision target in trust mode.")
    parser.add_argument("--policy-id", default="trust_filter_v1", help="Stable Task 17 policy ID bound into the sealed decision manifest.")
    parser.add_argument("--no-per-class-thresholds", action="store_true", help="Disable only validation-derived per-class confidence gates.")
    parser.add_argument("--no-view-consistency", action="store_true", help="Disable only original/flip agreement; original views remain representatives.")
    parser.add_argument("--no-size-filter", action="store_true", help="Disable only size/area/aspect-ratio gates.")
    parser.add_argument("--matrix-config", type=Path, help="Task 17 matrix YAML. When supplied, seals its full effective filter configuration and calibration input digests into Task 14 output.")
    return parser


def _read_pr(path: Path) -> list[Mapping[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrustFilterError(f"Problem: validation PR input cannot be read. Likely cause: {error}. Remediation: provide a UTF-8 validation prediction/outcome JSON file.") from error
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise TrustFilterError("Problem: validation PR input has no records array. Likely cause: it is not the evaluation prediction/outcome export. Remediation: provide {'records': [...]} with validation PR rows only.")
    return [dict(row) for row in records]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        membership = load_unlabeled_manifest(args.unlabeled_manifest.resolve(strict=False), split_manifest_path=args.split_manifest.resolve(strict=False) if args.split_manifest else None)
        geometries = {record.source_image_id: ImageGeometry(record.source_image_id, record.width, record.height) for record in membership.records}
        effective_policy: Mapping[str, object] | None = None
        if args.matrix_config is not None:
            # Import lazily: the normal Task-14 CLI remains independent of
            # Task-17 matrix files, while matrix-driven artifacts use no
            # invisible defaults or unrecorded calibration inputs.
            from fruit_ssod.training.ssod_matrix import SsodMatrixError, load_effective_filter_policy
            try:
                effective_policy = load_effective_filter_policy(args.matrix_config)
            except SsodMatrixError as error:
                raise TrustFilterError(str(error)) from error
            gates = {key: effective_policy[key] for key in ("policy_id", "use_per_class_thresholds", "require_view_consistency", "require_size_filter")}
            if args.policy_id != "trust_filter_v1" and args.policy_id != gates["policy_id"]:
                raise TrustFilterError("Problem: --policy-id differs from --matrix-config. Likely cause: a CLI override would produce an unsealed comparison. Remediation: omit --policy-id or use the matrix policy_id.")
            filter_config = effective_policy["filter_config"]
            calibration = effective_policy["threshold_calibration"]
            if not isinstance(filter_config, Mapping) or not isinstance(calibration, Mapping):  # defensive schema guard
                raise TrustFilterError("Problem: matrix effective filter policy is malformed. Likely cause: required calibration sections are absent. Remediation: restore the canonical Task 17 configuration.")
            is_global = gates["policy_id"] == "global_threshold_v1"
            if is_global:
                thresholds = PerClassThresholds({class_id: float(filter_config["global_confidence"]) for class_id in range(5)}, minimum=0.0, maximum=1.0)
            else:
                pr = calibration.get("validation_pr")
                aspect = effective_policy.get("aspect_ratio_bounds")
                if not isinstance(pr, Mapping) or not isinstance(pr.get("path"), str) or not isinstance(aspect, Mapping) or not isinstance(aspect.get("path"), str):
                    raise TrustFilterError("Problem: matrix calibration evidence is malformed. Likely cause: validation PR or aspect bounds source is absent. Remediation: restore the canonical Task 17 calibration artifacts.")
                thresholds = select_per_class_thresholds(_read_pr(Path(pr["path"])), target_precision=float(calibration["target_precision"]), minimum=float(calibration["minimum"]), maximum=float(calibration["maximum"]))
                bounds_artifact = load_aspect_ratio_bounds_artifact(Path(aspect["path"]))
            config = TrustFilterConfig(
                global_confidence=float(filter_config["global_confidence"]), cross_view_iou=float(filter_config["cross_view_iou"]), min_pixels_at_640=float(filter_config["min_pixels_at_640"]), max_area_fraction=float(filter_config["max_area_fraction"]), min_aspect_ratio=float(filter_config["min_aspect_ratio"]), max_aspect_ratio=float(filter_config["max_aspect_ratio"]), max_boxes_per_image=int(filter_config["max_boxes_per_image"]), nms_iou=float(filter_config["nms_iou"]), aspect_ratio_bounds_artifact=None if is_global else bounds_artifact,
                policy_id=str(gates["policy_id"]), use_per_class_thresholds=bool(gates["use_per_class_thresholds"]), require_view_consistency=bool(gates["require_view_consistency"]), require_size_filter=bool(gates["require_size_filter"]), effective_policy=effective_policy,
            )
        elif args.mode == "global":
            # A global baseline is one explicit [0, 1] confidence gate.  It is
            # deliberately not a per-class calibration and therefore must not
            # inherit the trust-mode [0.50, 0.85] calibration clamp.
            thresholds = PerClassThresholds({class_id: args.global_confidence for class_id in range(5)}, minimum=0.0, maximum=1.0)
            config = TrustFilterConfig(
                global_confidence=args.global_confidence, policy_id=args.policy_id,
                use_per_class_thresholds=False, require_view_consistency=False, require_size_filter=False,
            )
        else:
            if args.validation_pr is None:
                parser.error("--validation-pr is required in trust mode")
            if args.aspect_ratio_bounds is None:
                parser.error("--aspect-ratio-bounds is required in trust mode")
            thresholds = select_per_class_thresholds(_read_pr(args.validation_pr), target_precision=args.target_precision)
            bounds_artifact = load_aspect_ratio_bounds_artifact(args.aspect_ratio_bounds)
            config = TrustFilterConfig(
                global_confidence=args.global_confidence,
                aspect_ratio_bounds_artifact=bounds_artifact,
                policy_id=args.policy_id,
                use_per_class_thresholds=not args.no_per_class_thresholds,
                require_view_consistency=not args.no_view_consistency,
                require_size_filter=not args.no_size_filter,
            )
        result = TrustFilter(thresholds, config=config).filter_envelope(args.candidates, geometries)
        labels, audit = write_trust_filter_outputs(result, args.output)
    except (TrustFilterError, ThresholdSelectionError, PseudoGenerationError, ValueError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"labels": str(labels), "audit": str(audit), "decision_manifest": str(audit.with_name("decision_manifest.json")), "accepted": len(result.accepted), "rejected": sum(item["decision"] == "rejected" for item in result.audit)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
