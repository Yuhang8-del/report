"""Deterministic, auditable trust filtering for Task 13 candidate envelopes."""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any, Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.pseudo.candidates import PseudoCandidate, PseudoCandidateError
from fruit_ssod.pseudo.thresholds import PerClassThresholds
from fruit_ssod.pseudo.transforms import TransformError, horizontal_flip_xyxy


class TrustFilterError(RuntimeError):
    """Raised when candidate filtering cannot retain a reproducible audit trail."""


def _problem(problem: str, cause: str, remediation: str) -> TrustFilterError:
    return TrustFilterError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_CLASS_IDS = frozenset(item.id for item in DEFAULT_CLASS_REGISTRY.classes)
_CANDIDATE_KEYS = frozenset({"teacher_run_id", "source_image_id", "source_file_path", "view", "class_id", "class_name", "confidence", "raw_xyxy", "xyxy", "source_model"})
_CANONICAL_CLASSES = tuple(
    MappingProxyType({"id": item.id, "name": item.name})
    for item in DEFAULT_CLASS_REGISTRY.classes
)
_ASPECT_BOUNDS_ARTIFACT_KEYS = frozenset({"artifact_version", "artifact_type", "artifact_id", "class_registry_version", "classes", "provenance", "bounds"})
_ASPECT_BOUNDS_PROVENANCE_KEYS = frozenset({"source_split", "source_kind", "contains_human_labels", "sealed"})


def _freeze_json(value: object) -> object:
    """Recursively freeze JSON-compatible metadata retained in audit evidence."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise _problem("audit metadata has a non-string key", "JSON audit evidence would silently coerce a mapping key", "use JSON-compatible mappings with string keys only")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _immutable_canonical_classes() -> tuple[Mapping[str, object], ...]:
    """Return a fresh immutable five-class tuple for public audit evidence."""
    return tuple(MappingProxyType(dict(item)) for item in _CANONICAL_CLASSES)


def _plain_json(value: object) -> object:
    """Return a serializable copy without exposing frozen internal mappings."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return XYXY IoU with no coordinate coercion or rounding."""
    if len(first) != 4 or len(second) != 4:
        raise _problem("IoU input is not XYXY", "candidate boxes have an invalid coordinate count", "use four-coordinate Task 13 candidates")
    ax1, ay1, ax2, ay2 = (float(item) for item in first)
    bx1, by1, bx2, by2 = (float(item) for item in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class ImageGeometry:
    """No-label image geometry used only for scale and area guardrails."""

    source_image_id: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_image_id, str) or not self.source_image_id.strip():
            raise _problem("image geometry has no source_image_id", "candidate dimensions cannot be joined safely", "provide a no-label image ID")
        for field, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _problem(f"image geometry {field} must be positive", "candidate scale cannot be measured", f"provide a positive integer {field}")


@dataclass(frozen=True, init=False)
class AspectRatioBoundsArtifact:
    """Sealed aggregate-only aspect-ratio bounds for the five fruit classes.

    This is intentionally not an annotation export: it carries only reviewed
    aggregate bounds and provenance.  The exact schema prevents labels, boxes,
    image IDs, or audit annotations from being smuggled into trust filtering.
    """

    artifact_id: str
    artifact_sha256: str
    artifact_type: str
    class_registry_version: str
    bounds: Mapping[int, tuple[float, float]]
    provenance: Mapping[str, object]

    @classmethod
    def _from_sealed_payload(cls, payload: Mapping[str, object], raw_bytes: bytes) -> "AspectRatioBoundsArtifact":
        """Build one artifact only after the exact on-disk schema is validated.

        There is deliberately no public constructor.  A free-form bounds
        mapping has no authority to influence Trust mode: it must originate
        from the exact aggregate-only JSON artifact whose byte hash is copied
        into every audit event.
        """
        if not isinstance(payload.get("artifact_id"), str) or not payload["artifact_id"].strip():
            raise _problem("aspect-ratio bounds artifact has no ID", "the aggregate input is not traceable", "supply a nonempty artifact_id")
        if payload.get("artifact_type") != "sealed_aspect_ratio_bounds" or payload.get("class_registry_version") != DEFAULT_CLASS_REGISTRY.version:
            raise _problem("aspect-ratio bounds artifact provenance is invalid", "the artifact type or class registry version is not approved", "use the committed sealed aggregate-only five-class artifact")
        if payload.get("classes") != [dict(item) for item in _CANONICAL_CLASSES]:
            raise _problem("aspect-ratio bounds artifact taxonomy differs", "the artifact does not name the exact canonical five fruit classes", "regenerate it for the committed class registry and fixed class order")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != _ASPECT_BOUNDS_PROVENANCE_KEYS:
            raise _problem("aspect-ratio bounds provenance is incomplete", "the aggregate source cannot be verified as label-free and sealed", "supply the exact sealed aggregate-only provenance schema")
        if provenance != {
            "source_split": "train_pool",
            "source_kind": "approved_aggregate_statistics",
            "contains_human_labels": False,
            "sealed": True,
        }:
            raise _problem("aspect-ratio bounds provenance is unsafe", "the input is not sealed aggregate-only train-pool statistics", "use sealed approved aggregate statistics without labels, boxes, or image IDs")
        raw_bounds = payload.get("bounds")
        if not isinstance(raw_bounds, Mapping) or set(raw_bounds) != {str(item) for item in _CLASS_IDS}:
            raise _problem("aspect-ratio bounds artifact is incomplete", "the bounds must contain string keys 0 through 4", "supply exactly five canonical class bounds")
        bounds = _normalize_aspect_ratio_bounds({int(class_id): values for class_id, values in raw_bounds.items()})
        instance = object.__new__(cls)
        object.__setattr__(instance, "artifact_id", payload["artifact_id"])
        object.__setattr__(instance, "artifact_sha256", hashlib.sha256(raw_bytes).hexdigest())
        object.__setattr__(instance, "artifact_type", payload["artifact_type"])
        object.__setattr__(instance, "class_registry_version", payload["class_registry_version"])
        object.__setattr__(instance, "bounds", MappingProxyType(bounds))
        object.__setattr__(instance, "provenance", _freeze_json(provenance))
        return instance

    def audit_mapping(self) -> Mapping[str, object]:
        """Non-label provenance copied into each published filter decision."""
        return MappingProxyType({
            "artifact_id": self.artifact_id,
            "sha256": self.artifact_sha256,
            "artifact_type": self.artifact_type,
            "class_registry_version": self.class_registry_version,
            "canonical_classes": _immutable_canonical_classes(),
            "provenance": self.provenance,
            "bounds": MappingProxyType({str(class_id): bounds for class_id, bounds in self.bounds.items()}),
        })


def _normalize_aspect_ratio_bounds(bounds: Mapping[int, Sequence[float]]) -> dict[int, tuple[float, float]]:
    if not isinstance(bounds, Mapping) or set(bounds) != _CLASS_IDS:
        raise _problem("labeled aspect-ratio bounds are incomplete", "one or more fruit classes are missing", "provide non-label distribution bounds for every class ID 0 through 4")
    normalized: dict[int, tuple[float, float]] = {}
    for class_id, values in bounds.items():
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in _CLASS_IDS:
            raise _problem("labeled aspect-ratio bounds have invalid class IDs", "the bounds do not match the fixed five-class taxonomy", "provide integer class IDs 0 through 4")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
            raise _problem("labeled aspect-ratio bounds are malformed", "a class does not have lower and upper bounds", "provide (lower, upper) for each class")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise _problem("labeled aspect-ratio bounds are invalid", "a distribution bound is not numeric", "provide finite numeric lower and upper bounds")
        low, high = (float(item) for item in values)
        if not all(math.isfinite(item) and item > 0 for item in (low, high)) or low > high:
            raise _problem("labeled aspect-ratio bounds are invalid", "a distribution bound is nonpositive, nonfinite, or inverted", "provide finite positive lower <= upper bounds")
        normalized[class_id] = (low, high)
    return normalized


def load_aspect_ratio_bounds_artifact(path: Path) -> AspectRatioBoundsArtifact:
    """Load a sealed, aggregate-only five-class distribution-bounds artifact."""
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("aspect-ratio bounds artifact cannot be read", str(error), "supply a readable UTF-8 sealed aggregate-statistics JSON artifact") from error
    if not isinstance(payload, Mapping) or set(payload) != _ASPECT_BOUNDS_ARTIFACT_KEYS:
        raise _problem("aspect-ratio bounds artifact has unsupported or label-bearing fields", "the input is not the sealed aggregate-only bounds schema", "supply only the approved five-class bounds artifact without annotations, images, or labels")
    if payload.get("artifact_version") != "1.0" or payload.get("artifact_type") != "sealed_aspect_ratio_bounds":
        raise _problem("aspect-ratio bounds artifact schema is unsupported", "its version or approved aggregate type is not recognized", "use artifact_version 1.0 and an approved aggregate-statistics artifact type")
    return AspectRatioBoundsArtifact._from_sealed_payload(payload, raw_bytes)


@dataclass(frozen=True)
class TrustFilterConfig:
    """Fixed Task 14 guardrails, with explicit changes required for experiments."""

    global_confidence: float = 0.50
    cross_view_iou: float = 0.60
    min_pixels_at_640: float = 16.0
    max_area_fraction: float = 0.90
    min_aspect_ratio: float = 0.10
    max_aspect_ratio: float = 10.0
    aspect_ratio_bounds_artifact: AspectRatioBoundsArtifact | None = None
    max_boxes_per_image: int = 20
    nms_iou: float = 0.60
    # These are executable Task 17 policy switches, not descriptive YAML.
    # Coordinate provenance, global confidence, NMS and the per-image cap are
    # always retained; the three switches name the only removable Trust gates.
    policy_id: str = "trust_filter_v1"
    use_per_class_thresholds: bool = True
    require_view_consistency: bool = True
    require_size_filter: bool = True
    # Optional Task-17 evidence generated from the matrix YAML.  When
    # supplied, it replaces the legacy four-field policy record and is copied
    # byte-for-byte into the sealed Task-14 decision manifest.
    effective_policy: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field in ("global_confidence", "cross_view_iou", "max_area_fraction", "nms_iou"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= float(value) <= 1:
                raise _problem(f"{field} must be finite in [0, 1]", "trust-filter configuration is invalid", "supply a finite probability")
            object.__setattr__(self, field, float(value))
        for field in ("min_pixels_at_640", "min_aspect_ratio", "max_aspect_ratio"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0:
                raise _problem(f"{field} must be finite and positive", "trust-filter configuration is invalid", "supply a positive finite value")
            object.__setattr__(self, field, float(value))
        if self.min_aspect_ratio > self.max_aspect_ratio:
            raise _problem("aspect-ratio bounds are inverted", "minimum exceeds maximum", "use min_aspect_ratio <= max_aspect_ratio")
        if isinstance(self.max_boxes_per_image, bool) or not isinstance(self.max_boxes_per_image, int) or self.max_boxes_per_image <= 0:
            raise _problem("max_boxes_per_image must be positive", "the output cap is invalid", "use a positive integer")
        if self.aspect_ratio_bounds_artifact is not None and not isinstance(self.aspect_ratio_bounds_artifact, AspectRatioBoundsArtifact):
            raise _problem("aspect-ratio bounds artifact is not sealed", "Trust mode received a raw bounds mapping or unverified provenance", "load the artifact with load_aspect_ratio_bounds_artifact and pass it as aspect_ratio_bounds_artifact")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise _problem("policy_id is invalid", "the published filtering policy has no stable identity", "provide the declared Task 17 policy_id")
        for field in ("use_per_class_thresholds", "require_view_consistency", "require_size_filter"):
            if not isinstance(getattr(self, field), bool):
                raise _problem(f"{field} must be boolean", "an ablation gate is ambiguous", "use explicit true or false in the Task 17 pseudo_filter policy")
        if self.effective_policy is not None:
            if not isinstance(self.effective_policy, Mapping):
                raise _problem("effective policy is invalid", "the Task 17 policy evidence is not a mapping", "supply the sealed effective filter policy generated from the matrix configuration")
            evidence = _freeze_json(self.effective_policy)
            if not isinstance(evidence, Mapping) or any(evidence.get(field) != getattr(self, field) for field in ("policy_id", "use_per_class_thresholds", "require_view_consistency", "require_size_filter")):
                raise _problem("effective policy differs from executable filter", "the sealed Task 17 evidence does not describe these active gates", "regenerate policy evidence from the same matrix configuration")
            object.__setattr__(self, "effective_policy", evidence)

    @property
    def labeled_aspect_ratio_bounds(self) -> Mapping[int, tuple[float, float]] | None:
        """Read-only compatibility view; raw mappings are never accepted."""
        return self.aspect_ratio_bounds_artifact.bounds if self.aspect_ratio_bounds_artifact else None

    @property
    def aspect_ratio_bounds_provenance(self) -> Mapping[str, object] | None:
        return self.aspect_ratio_bounds_artifact.audit_mapping() if self.aspect_ratio_bounds_artifact else None

    def bounds_for(self, class_id: int) -> tuple[float, float]:
        lower, upper = self.min_aspect_ratio, self.max_aspect_ratio
        if self.labeled_aspect_ratio_bounds is not None:
            labeled_lower, labeled_upper = self.labeled_aspect_ratio_bounds[class_id]
            lower, upper = max(lower, labeled_lower), min(upper, labeled_upper)
        return lower, upper

    def policy_mapping(self) -> Mapping[str, object]:
        """Canonical, serializable policy evidence bound to Task 14 outputs."""
        if self.effective_policy is not None:
            return self.effective_policy
        return MappingProxyType({
            "policy_id": self.policy_id,
            "use_per_class_thresholds": self.use_per_class_thresholds,
            "require_view_consistency": self.require_view_consistency,
            "require_size_filter": self.require_size_filter,
        })


@dataclass(frozen=True)
class TrustedPseudoLabel:
    source_image_id: str
    source_file_path: str
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    teacher_run_id: str
    source_model: str

    def mapping(self) -> dict[str, Any]:
        return {"source_image_id": self.source_image_id, "source_file_path": self.source_file_path, "class_id": self.class_id, "class_name": self.class_name, "confidence": self.confidence, "xyxy": list(self.xyxy), "teacher_run_id": self.teacher_run_id, "source_model": self.source_model}


@dataclass(frozen=True)
class TrustFilterResult:
    teacher_run_id: str
    accepted: tuple[TrustedPseudoLabel, ...]
    audit: tuple[Mapping[str, Any], ...]
    image_geometries: Mapping[str, ImageGeometry]
    filter_provenance: Mapping[str, object] | None = None
    candidate_artifact_sha256: str | None = None
    filter_policy: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.teacher_run_id, str) or not self.teacher_run_id.strip():
            raise _problem("trust result has no teacher run ID", "filter provenance was lost", "filter one validated Task 13 envelope at a time")
        object.__setattr__(self, "accepted", tuple(self.accepted))
        if any(not isinstance(item, Mapping) for item in self.audit):
            raise _problem("trust result audit is invalid", "an audit event is not a JSON mapping", "retain only mapping records produced by TrustFilter.filter")
        object.__setattr__(self, "audit", tuple(_freeze_json(dict(item)) for item in self.audit))
        geometries = dict(self.image_geometries)
        if any(not isinstance(value, ImageGeometry) or value.source_image_id != key for key, value in geometries.items()):
            raise _problem("trust result image geometry is invalid", "accepted labels are not bound to a matching no-label geometry", "retain ImageGeometry values keyed by source_image_id")
        if any(label.source_image_id not in geometries for label in self.accepted):
            raise _problem("trust result lacks accepted image geometry", "YOLO normalization would be ambiguous", "filter candidates with ImageGeometry for every accepted image")
        object.__setattr__(self, "image_geometries", MappingProxyType(geometries))
        if self.filter_provenance is not None:
            object.__setattr__(self, "filter_provenance", _freeze_json(dict(self.filter_provenance)))
        if self.candidate_artifact_sha256 is not None:
            digest = self.candidate_artifact_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise _problem("candidate artifact digest is malformed", "the Task 13 source artifact cannot be bound to this decision set", "filter an on-disk Task 13 candidate envelope without modifying it")
        if self.filter_policy is not None:
            object.__setattr__(self, "filter_policy", _freeze_json(dict(self.filter_policy)))


def load_candidate_envelope(path: Path) -> tuple[str, tuple[PseudoCandidate, ...]]:
    """Load exactly the Task 13 candidate envelope; reject labels and loose lists."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("candidate envelope cannot be read", str(error), "supply the UTF-8 JSON output of generate_pseudo_labels") from error
    if not isinstance(payload, Mapping) or set(payload) != {"manifest_version", "teacher_run_id", "candidate_count", "candidates"}:
        raise _problem("candidate envelope is not a Task 13 manifest", "input is a generic label/audit manifest or has unsupported fields", "supply the unmodified Task 13 candidates JSON envelope")
    if payload["manifest_version"] != "1.0" or not isinstance(payload["teacher_run_id"], str) or isinstance(payload["candidate_count"], bool) or not isinstance(payload["candidate_count"], int) or payload["candidate_count"] < 0 or not isinstance(payload["candidates"], list):
        raise _problem("candidate envelope is malformed", "Task 13 provenance fields are missing or invalid", "supply the unmodified Task 13 candidates JSON envelope")
    if payload["candidate_count"] != len(payload["candidates"]):
        raise _problem("candidate envelope count differs", "candidate evidence may have been edited", "regenerate candidates from the Task 13 generator")
    candidates: list[PseudoCandidate] = []
    for row in payload["candidates"]:
        if not isinstance(row, Mapping) or set(row) != _CANDIDATE_KEYS:
            raise _problem("candidate record has unsupported or label-bearing fields", "candidate provenance no longer matches the Task 13 schema", "supply only unmodified Task 13 candidate records")
        try:
            candidate = PseudoCandidate(**dict(row))
        except (TypeError, PseudoCandidateError) as error:
            raise _problem("candidate record is invalid", str(error), "regenerate the Task 13 candidate envelope") from error
        if candidate.teacher_run_id != payload["teacher_run_id"]:
            raise _problem("candidate envelope mixes teacher runs", "candidate provenance was modified or combined", "filter exactly one teacher run at a time")
        candidates.append(candidate)
    return payload["teacher_run_id"], tuple(candidates)


def _candidate_key(candidate: PseudoCandidate) -> tuple[Any, ...]:
    return (-candidate.confidence, candidate.view, candidate.source_image_id, candidate.class_id, *candidate.xyxy, candidate.source_model)


def _audit(candidate: PseudoCandidate, decision: str, reason: str | None = None, *, paired_with: PseudoCandidate | None = None) -> dict[str, Any]:
    # Every accepted Task 14 row has an explicit, stable reason.  The paired
    # fields encode cross-view evidence when enabled and remain null only for
    # the intentional no-view/global policies.
    if decision == "accepted" and reason is None:
        reason = "accepted"
    record = candidate.mapping()
    record.update({"decision": decision, "reason_code": reason, "paired_with_view": paired_with.view if paired_with else None, "paired_with_confidence": paired_with.confidence if paired_with else None})
    return record


def _coordinate_provenance_reason(candidate: PseudoCandidate, geometry: ImageGeometry | None) -> str | None:
    """Reject an envelope whose view-space and original-space evidence disagree.

    Task 13 records both representations so that Task 14 can verify the
    transform instead of trusting a caller-provided ``xyxy`` box.  These are
    rejection reason codes, not loader errors: the whole evidence envelope is
    retained in the output audit for later diagnosis.
    """
    if candidate.view == "original":
        return None if candidate.raw_xyxy == candidate.xyxy else "original_raw_xyxy_mismatch"
    if geometry is None:
        return "missing_image_geometry"
    try:
        mapped = horizontal_flip_xyxy(candidate.raw_xyxy, width=geometry.width)
    except TransformError:
        return "flip_raw_xyxy_invalid"
    return None if mapped == candidate.xyxy else "flip_raw_xyxy_mapping_mismatch"


def _geometry_reason(candidate: PseudoCandidate, geometry: ImageGeometry | None, config: TrustFilterConfig) -> str | None:
    if geometry is None:
        return "missing_image_geometry"
    x1, y1, x2, y2 = candidate.xyxy
    if x1 < 0 or y1 < 0 or x2 > geometry.width or y2 > geometry.height:
        return "box_out_of_bounds"
    width, height = x2 - x1, y2 - y1
    if width * 640 / geometry.width < config.min_pixels_at_640 or height * 640 / geometry.height < config.min_pixels_at_640:
        return "too_small_at_640"
    if width * height / (geometry.width * geometry.height) > config.max_area_fraction:
        return "area_too_large"
    ratio = width / height
    lower, upper = config.bounds_for(candidate.class_id)
    if ratio < config.min_aspect_ratio or ratio > config.max_aspect_ratio:
        return "aspect_ratio_outside_global_bounds"
    if ratio < lower or ratio > upper:
        return "aspect_ratio_outside_labeled_distribution"
    return None


class TrustFilter:
    """Apply configuration-only guards to raw dual-view Task 13 candidates."""

    def __init__(self, thresholds: PerClassThresholds, *, config: TrustFilterConfig | None = None) -> None:
        if not isinstance(thresholds, PerClassThresholds):
            raise _problem("per-class thresholds are not validated", "filter construction bypassed validation threshold selection", "pass a PerClassThresholds instance")
        self.thresholds = thresholds
        self.config = config or TrustFilterConfig()
        if self.config.effective_policy is not None:
            resolved = self.config.effective_policy.get("resolved_thresholds")
            if resolved != thresholds.mapping():
                raise _problem("effective policy threshold table differs", "the sealed Task 17 policy does not match the actual calibrated thresholds", "recompute thresholds from the declared validation PR and regenerate Task 14 output")

    def filter(self, teacher_run_id: str, candidates: Sequence[PseudoCandidate], geometries: Mapping[str, ImageGeometry] | None = None) -> TrustFilterResult:
        if not isinstance(teacher_run_id, str) or not teacher_run_id.strip() or any(not isinstance(candidate, PseudoCandidate) for candidate in candidates):
            raise _problem("filter input is not validated Task 13 candidates", "a loose label mapping or invalid candidate was supplied", "call load_candidate_envelope before TrustFilter.filter")
        if any(candidate.teacher_run_id != teacher_run_id for candidate in candidates):
            raise _problem("filter input mixes teacher run IDs", "candidate provenance does not match the envelope", "filter one Task 13 teacher envelope at a time")
        geometry_by_id = dict(geometries or {})
        if any(not isinstance(item, ImageGeometry) or item.source_image_id != key for key, item in geometry_by_id.items()):
            raise _problem("image geometry mapping is invalid", "a geometry is loose or keyed to a different image ID", "provide ImageGeometry values keyed by their source_image_id")
        audit: list[dict[str, Any]] = []
        eligible: list[PseudoCandidate] = []
        for candidate in sorted(candidates, key=_candidate_key):
            coordinate_reason = _coordinate_provenance_reason(candidate, geometry_by_id.get(candidate.source_image_id))
            if coordinate_reason:
                audit.append(_audit(candidate, "rejected", coordinate_reason)); continue
            if candidate.confidence < self.config.global_confidence:
                audit.append(_audit(candidate, "rejected", "below_global_confidence")); continue
            if self.config.use_per_class_thresholds and candidate.confidence < self.thresholds.for_class(candidate.class_id):
                audit.append(_audit(candidate, "rejected", "below_class_threshold")); continue
            if self.config.require_size_filter:
                reason = _geometry_reason(candidate, geometry_by_id.get(candidate.source_image_id), self.config)
                if reason:
                    audit.append(_audit(candidate, "rejected", reason)); continue
            eligible.append(candidate)
        groups: dict[tuple[str, int], list[PseudoCandidate]] = {}
        for candidate in eligible:
            groups.setdefault((candidate.source_image_id, candidate.class_id), []).append(candidate)
        paired: list[tuple[PseudoCandidate, PseudoCandidate, float]] = []
        paired_ids: set[int] = set()
        for key in sorted(groups):
            originals = sorted((item for item in groups[key] if item.view == "original"), key=_candidate_key)
            flips = sorted((item for item in groups[key] if item.view == "horizontal_flip"), key=_candidate_key)
            if not self.config.require_view_consistency:
                # A no-view ablation still uses one canonical original-view
                # representative per detection.  Flips are alternative model
                # inputs, never a second label for the same source image.
                for original in originals:
                    paired.append((original, original, 1.0)); paired_ids.add(id(original))
                for flip in flips:
                    audit.append(_audit(flip, "rejected", "view_not_selected"))
                continue
            available = list(flips)
            for original in originals:
                matches = [(box_iou(original.xyxy, flip.xyxy), flip) for flip in available]
                matches = [(iou, flip) for iou, flip in matches if iou >= self.config.cross_view_iou]
                if not matches:
                    audit.append(_audit(original, "rejected", "no_cross_view_match")); continue
                iou, flip = max(matches, key=lambda item: (item[0], tuple(-value for value in item[1].xyxy), item[1].confidence, item[1].source_model))
                available.remove(flip)
                paired.append((original, flip, iou)); paired_ids.update((id(original), id(flip)))
            for flip in available:
                audit.append(_audit(flip, "rejected", "no_cross_view_match"))
        # Represent a paired detection by the original-view coordinates. Both
        # source view records are nevertheless retained in the audit trail.
        preliminary: list[tuple[TrustedPseudoLabel, PseudoCandidate, PseudoCandidate]] = []
        for original, flip, _iou in paired:
            label = TrustedPseudoLabel(original.source_image_id, original.source_file_path, original.class_id, original.class_name, min(original.confidence, flip.confidence), original.xyxy, original.teacher_run_id, original.source_model)
            preliminary.append((label, original, flip))
        preliminary.sort(key=lambda item: (-item[0].confidence, item[0].source_image_id, item[0].class_id, *item[0].xyxy, item[0].source_model))
        retained: list[TrustedPseudoLabel] = []
        per_image: dict[str, list[TrustedPseudoLabel]] = {}
        for label, original, flip in preliminary:
            peers = per_image.setdefault(label.source_image_id, [])
            if any(existing.class_id == label.class_id and box_iou(existing.xyxy, label.xyxy) >= self.config.nms_iou for existing in peers):
                if original is flip:
                    audit.append(_audit(original, "rejected", "nms_duplicate"))
                else:
                    audit.extend((_audit(original, "rejected", "nms_duplicate", paired_with=flip), _audit(flip, "rejected", "nms_duplicate", paired_with=original)))
                continue
            if len(peers) >= self.config.max_boxes_per_image:
                if original is flip:
                    audit.append(_audit(original, "rejected", "max_boxes_per_image"))
                else:
                    audit.extend((_audit(original, "rejected", "max_boxes_per_image", paired_with=flip), _audit(flip, "rejected", "max_boxes_per_image", paired_with=original)))
                continue
            peers.append(label); retained.append(label)
            if self.config.require_view_consistency:
                audit.extend((_audit(original, "accepted", paired_with=flip), _audit(flip, "accepted", paired_with=original)))
            else:
                audit.append(_audit(original, "accepted"))
        provenance: Mapping[str, object] | None = None
        if self.config.aspect_ratio_bounds_provenance is not None:
            provenance = MappingProxyType({"aspect_ratio_bounds": self.config.aspect_ratio_bounds_provenance})
        return TrustFilterResult(teacher_run_id, tuple(sorted(retained, key=lambda item: (item.source_image_id, item.class_id, *item.xyxy, -item.confidence))), tuple(audit), geometry_by_id, provenance, None, self.config.policy_mapping())

    def filter_envelope(self, path: Path, geometries: Mapping[str, ImageGeometry] | None = None) -> TrustFilterResult:
        teacher_run_id, candidates = load_candidate_envelope(path)
        result = self.filter(teacher_run_id, candidates, geometries)
        # Bind production decisions to the exact Task 13 bytes, not merely to
        # candidate values that could be reconstructed in another envelope.
        return TrustFilterResult(
            result.teacher_run_id, result.accepted, result.audit, result.image_geometries,
            result.filter_provenance, _file_sha256(path), result.filter_policy,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _problem("candidate envelope cannot be hashed", str(error), "keep the original Task 13 candidate artifact readable until filtering completes") from error
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _problem("filter provenance is not canonical JSON", str(error), "use only immutable JSON-compatible filter provenance") from error
    return hashlib.sha256(encoded).hexdigest()


def write_trust_filter_outputs(result: TrustFilterResult, output_root: Path) -> tuple[Path, Path]:
    """Atomically publish labels, decisions, and their sealed binding manifest.

    ``decision_manifest.json`` sits beside ``audit.jsonl``.  It hashes both
    the exact candidate envelope and the exact serialized decision records so
    Task 15 can reject a changed decision, reason, or filter provenance.
    """
    if not isinstance(result, TrustFilterResult):
        raise _problem("trust-filter output is invalid", "publication did not receive a validated filter result", "pass TrustFilter.filter output")
    if result.candidate_artifact_sha256 is None:
        raise _problem("trust-filter output lacks a sealed candidate binding", "a direct in-memory filter result has no immutable Task 13 artifact to hash", "load candidates with TrustFilter.filter_envelope before publishing decisions")
    root = output_root.resolve(strict=False)
    labels_root, audit_path = root / "labels", root / "audit.jsonl"
    if root.exists():
        raise _problem("trust-filter output root already exists", f"{root} would be overwritten", "choose a new empty output directory")
    ancestor = root.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise _problem("trust-filter output has a file ancestor", f"{ancestor} is not a directory", "choose an output path below directories only")
        ancestor = ancestor.parent
    temporary: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        (temporary / "labels").mkdir()
        for image_id, labels in _group_labels(result.accepted).items():
            geometry = result.image_geometries[image_id]
            lines: list[str] = []
            for label in sorted(labels, key=lambda item: (item.class_id, *item.xyxy, -item.confidence)):
                x1, y1, x2, y2 = label.xyxy
                center_x, center_y = ((x1 + x2) / 2 / geometry.width, (y1 + y2) / 2 / geometry.height)
                width, height = ((x2 - x1) / geometry.width, (y2 - y1) / geometry.height)
                lines.append(f"{label.class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
            (temporary / "labels" / _safe_label_name(image_id)).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        provenance = _plain_json(result.filter_provenance) if result.filter_provenance is not None else None
        rows: list[dict[str, Any]] = []
        for event in sorted(result.audit, key=lambda item: (str(item.get("source_image_id", "")), str(item.get("view", "")), int(item.get("class_id", -1)), tuple(item.get("xyxy", ())), str(item.get("decision", "")), str(item.get("reason_code", "")))):
            published = dict(event)
            # Always write the field, including null for global mode.  A
            # decision file then has one unambiguous, manifest-bound policy.
            published["filter_provenance"] = provenance
            rows.append(published)
        audit_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode("utf-8")
        (temporary / "audit.jsonl").write_bytes(audit_bytes)
        manifest = {
            "schema_version": "1.0",
            "artifact_type": "sealed_task14_filter_decisions",
            "teacher_run_id": result.teacher_run_id,
            "candidate_artifact_sha256": result.candidate_artifact_sha256,
            "decision_record_count": len(rows),
            "decision_records_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "filter_provenance": provenance,
            "filter_provenance_sha256": _canonical_json_sha256(provenance),
            "filter_policy": _plain_json(result.filter_policy),
            "filter_policy_sha256": _canonical_json_sha256(_plain_json(result.filter_policy)),
        }
        (temporary / "decision_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, root)
        temporary = None
        return labels_root, audit_path
    except TrustFilterError:
        raise
    except OSError as error:
        raise _problem("trust-filter outputs could not be written atomically", str(error), "ensure the output parent is writable and use a new path") from error
    finally:
        if temporary is not None and temporary.exists():
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)


def _group_labels(labels: Sequence[TrustedPseudoLabel]) -> dict[str, list[TrustedPseudoLabel]]:
    result: dict[str, list[TrustedPseudoLabel]] = {}
    for label in labels:
        result.setdefault(label.source_image_id, []).append(label)
    return result


def _safe_label_name(source_image_id: str) -> str:
    if not source_image_id or any(character in source_image_id for character in "\\/:") or source_image_id in {".", ".."}:
        raise _problem("source_image_id cannot form a safe YOLO filename", "an image ID can escape the labels directory", "use a plain nonempty image identifier without path separators")
    return f"{source_image_id}.txt"
