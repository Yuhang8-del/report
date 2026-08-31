"""Sealed pseudo-audit metrics for offline semi-supervised experiments.

This module deliberately has no training dependency.  It can read the labels
reserved by Task 8 only after the audit CLI proves their split fingerprint,
and it exposes only prediction/ground-truth metric records to callers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from fruit_ssod.data.class_mapping import DEFAULT_CLASS_REGISTRY
from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.trust_filter import box_iou, load_candidate_envelope


class PseudoAuditError(RuntimeError):
    """Raised when pseudo-label audit evidence is malformed or leaky."""


def _problem(problem: str, cause: str, remediation: str) -> PseudoAuditError:
    return PseudoAuditError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


_CLASS_IDS = frozenset(DEFAULT_CLASS_REGISTRY.class_ids)
_AUDIT_EVENT_KEYS = frozenset({
    "teacher_run_id", "source_image_id", "source_file_path", "view", "class_id",
    "class_name", "confidence", "raw_xyxy", "xyxy", "source_model", "decision",
    "reason_code", "paired_with_view", "paired_with_confidence",
})
_DECISION_MANIFEST_KEYS = frozenset({
    "schema_version", "artifact_type", "teacher_run_id", "candidate_artifact_sha256",
    "decision_record_count", "decision_records_sha256", "filter_provenance",
    "filter_provenance_sha256",
})
_DECISION_MANIFEST_POLICY_KEYS = _DECISION_MANIFEST_KEYS | frozenset({"filter_policy", "filter_policy_sha256"})
_CANONICAL_SHA256 = frozenset("0123456789abcdef")


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _problem("audit evidence is not canonical JSON", str(error), "regenerate the immutable Task 8 artifact") from error
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash the precise input artifact retained in the output provenance."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _problem("audit artifact cannot be read", str(error), "restore the immutable pseudo-audit artifact") from error
    return digest.hexdigest()


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _problem(f"{field} must be a nonempty relative path", "audit image provenance is missing", f"store a safe relative {field}")
    windows, posix = PureWindowsPath(value), PurePosixPath(value.replace("\\", "/"))
    parts = tuple(part for part in value.replace("\\", "/").split("/") if part)
    if windows.is_absolute() or windows.drive or posix.is_absolute() or not parts or any(part in {".", ".."} for part in parts):
        raise _problem(f"{field} is unsafe", "an absolute, drive-qualified, or traversing image path could escape the sealed audit root", "use the exact relative file_path from Task 8")
    return value


def _box(value: object, *, width: int, height: int, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise _problem(f"{field} must be an XYXY array", "audit geometry is malformed", "provide four finite in-bounds coordinates")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise _problem(f"{field} contains invalid coordinates", "audit geometry has a non-finite value", "provide finite numeric XYXY coordinates")
    result = tuple(float(item) for item in value)
    if not (0 <= result[0] < result[2] <= width and 0 <= result[1] < result[3] <= height):
        raise _problem(f"{field} is outside sealed image geometry", "boxes and audit membership do not describe the same decoded image", "regenerate predictions or labels against the exact pseudo-audit image")
    return result  # type: ignore[return-value]


def _class_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _CLASS_IDS:
        raise _problem("class_id is not one of the fixed five classes", "audit labels or predictions use an incompatible taxonomy", "use canonical IDs 0 through 4")
    return value


@dataclass(frozen=True)
class AuditImage:
    """One sealed pseudo-audit image, including only its image provenance."""

    source_image_id: str
    file_path: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_image_id, str) or not self.source_image_id.strip():
            raise _problem("pseudo-audit image has no source_image_id", "membership cannot be joined safely", "restore the Task 8 protected split record")
        object.__setattr__(self, "file_path", _safe_relative_path(self.file_path, "file_path"))
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise _problem(f"pseudo-audit image {name} is invalid", "membership has no usable decoded dimensions", "restore positive integer dimensions from Task 8")


@dataclass(frozen=True)
class AuditBox:
    """A class-aware box, always joined to an immutable pseudo-audit image."""

    source_image_id: str
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_image_id, str) or not self.source_image_id.strip():
            raise _problem("audit box has no source_image_id", "box matching would cross image boundaries", "retain the sealed image ID on every box")
        object.__setattr__(self, "class_id", _class_id(self.class_id))
        if not isinstance(self.xyxy, tuple) or len(self.xyxy) != 4 or any(not isinstance(item, float) or not math.isfinite(item) for item in self.xyxy):
            raise _problem("audit box is not a finite XYXY tuple", "a loose geometry record bypassed audit parsing", "construct AuditBox records through sealed loaders")
        if not self.xyxy[0] < self.xyxy[2] or not self.xyxy[1] < self.xyxy[3]:
            raise _problem("audit box has no positive area", "XYXY coordinates are inverted or empty", "retain only x1 < x2 and y1 < y2 boxes")
        if self.confidence is not None and (isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or not 0 <= float(self.confidence) <= 1):
            raise _problem("prediction confidence is invalid", "candidate provenance has an invalid score", "use a finite confidence in [0, 1]")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class MatchResult:
    """One-to-one class-aware IoU match result."""

    matched_prediction_indices: tuple[int, ...]
    matched_ground_truth_indices: tuple[int, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]

    @property
    def tp(self) -> int:
        return len(self.matched_prediction_indices)

    @property
    def fp(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def fn(self) -> int:
        return len(self.unmatched_ground_truth_indices)


@dataclass(frozen=True)
class PseudoLabelMetrics:
    """Five-class, per-class and aggregate precision/recall/F1 evidence."""

    per_class: Mapping[int, Mapping[str, float | int]]
    overall: Mapping[str, float | int]

    def __post_init__(self) -> None:
        if set(self.per_class) != _CLASS_IDS:
            raise _problem("pseudo metrics do not cover the five canonical classes", "a class was omitted or remapped", "report classes 0 through 4, including zero-count classes")
        frozen: dict[int, Mapping[str, float | int]] = {}
        for class_id, values in self.per_class.items():
            if not isinstance(values, Mapping) or set(values) != {"tp", "fp", "fn", "precision", "recall", "f1"}:
                raise _problem("per-class pseudo metrics are malformed", "metric evidence lacks a required field", "retain TP, FP, FN, Precision, Recall and F1 for every class")
            frozen[class_id] = MappingProxyType(_validated_metric_values(values))
        if not isinstance(self.overall, Mapping) or set(self.overall) != {"tp", "fp", "fn", "precision", "recall", "f1"}:
            raise _problem("overall pseudo metrics are malformed", "aggregate metric evidence lacks a required field", "retain TP, FP, FN, Precision, Recall and F1")
        object.__setattr__(self, "per_class", MappingProxyType(frozen))
        object.__setattr__(self, "overall", MappingProxyType(_validated_metric_values(self.overall)))

    def mapping(self) -> dict[str, Any]:
        return {
            "per_class": {str(class_id): dict(values) for class_id, values in sorted(self.per_class.items())},
            "overall": dict(self.overall),
        }


def _validated_metric_values(values: Mapping[str, float | int]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key in ("tp", "fp", "fn"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _problem(f"metric {key} is invalid", "count metrics must be nonnegative integers", "recompute metrics with one-to-one matching")
        output[key] = value
    for key in ("precision", "recall", "f1"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= float(value) <= 1:
            raise _problem(f"metric {key} is invalid", "score metrics must be finite values in [0, 1]", "recompute normalized precision, recall and F1")
        output[key] = float(value)
    return output


def _maximum_cardinality(edges: Mapping[int, tuple[int, ...]]) -> int:
    """Return a deterministic maximum bipartite matching cardinality.

    This first phase deliberately ignores IoU scores.  A descending-IoU
    greedy pass can consume the only label reachable by a second prediction,
    even where a different eligible pairing would produce another true
    positive.
    """
    truth_to_prediction: dict[int, int] = {}

    def augment(prediction: int, seen: set[int]) -> bool:
        for truth in edges[prediction]:
            if truth in seen:
                continue
            seen.add(truth)
            previous = truth_to_prediction.get(truth)
            if previous is None or augment(previous, seen):
                truth_to_prediction[truth] = prediction
                return True
        return False

    for prediction in sorted(edges):
        augment(prediction, set())
    return len(truth_to_prediction)


def _maximum_weight_matching(edges: Mapping[int, tuple[tuple[int, float], ...]], cardinality: int) -> tuple[tuple[int, int], ...]:
    """Find a cardinality-fixed maximum-total-IoU matching.

    Successive shortest augmenting paths solve the small per-image/class
    bipartite flow problem.  ``Decimal(str(iou))`` keeps the comparison based
    on the detector's reported IoU rather than a coarse integer quantisation.
    Equal-cost paths are traversed in canonical index order and never replace
    an earlier equal path, giving stable ties without changing the primary
    cardinality or IoU objectives.
    """
    if cardinality == 0:
        return ()

    # Nodes are namespaced so integer prediction and truth indices can never
    # collide.  A residual edge is [destination, residual_capacity, cost].
    source: tuple[str, int] = ("source", 0)
    sink: tuple[str, int] = ("sink", 0)
    graph: dict[tuple[str, int], list[list[object]]] = {}

    def add_edge(start: tuple[str, int], end: tuple[str, int], cost: Decimal) -> None:
        graph.setdefault(start, [])
        graph.setdefault(end, [])
        forward: list[object] = [end, 1, cost, len(graph[end])]
        reverse: list[object] = [start, 0, -cost, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(reverse)

    for prediction in sorted(edges):
        pred_node = ("prediction", prediction)
        add_edge(source, pred_node, Decimal("0"))
        for truth, iou in edges[prediction]:
            add_edge(pred_node, ("truth", truth), -Decimal(str(iou)))
    truth_indices = sorted({truth for entries in edges.values() for truth, _ in entries})
    for truth in truth_indices:
        add_edge(("truth", truth), sink, Decimal("0"))

    # Bellman-Ford is intentional here: residual reverse edges can be
    # negative.  The flow is bounded by the number of boxes in one audit
    # image/class group, where exact and auditable behaviour matters more than
    # a specialised approximate optimiser.
    vertices = tuple(sorted(graph))
    for _ in range(cardinality):
        distance: dict[tuple[str, int], Decimal] = {source: Decimal("0")}
        predecessor: dict[tuple[str, int], tuple[tuple[str, int], int]] = {}
        for _ in range(len(vertices) - 1):
            changed = False
            for start in vertices:
                start_distance = distance.get(start)
                if start_distance is None:
                    continue
                for edge_index, edge in enumerate(graph[start]):
                    end, capacity, cost, _reverse_index = edge
                    if capacity != 1:
                        continue
                    assert isinstance(end, tuple) and isinstance(cost, Decimal)
                    candidate = start_distance + cost
                    # Strictly-better only preserves the first canonical path
                    # when total IoU is tied.
                    if end not in distance or candidate < distance[end]:
                        distance[end] = candidate
                        predecessor[end] = (start, edge_index)
                        changed = True
            if not changed:
                break
        if sink not in predecessor:
            raise _problem("maximum matching flow is incomplete", "eligible pseudo-audit pairs could not be connected", "regenerate the immutable audit inputs and rerun matching")
        node = sink
        while node != source:
            start, edge_index = predecessor[node]
            edge = graph[start][edge_index]
            end, capacity, _cost, reverse_index = edge
            assert end == node and capacity == 1 and isinstance(reverse_index, int)
            edge[1] = 0
            reverse = graph[node][reverse_index]
            reverse[1] = 1
            node = start

    pairs: list[tuple[int, int]] = []
    for prediction in sorted(edges):
        node = ("prediction", prediction)
        for end, capacity, _cost, _reverse_index in graph[node]:
            if isinstance(end, tuple) and end[0] == "truth" and capacity == 0:
                pairs.append((prediction, end[1]))
    return tuple(pairs)


def one_to_one_match(predictions: Sequence[AuditBox], ground_truth: Sequence[AuditBox], *, iou_threshold: float = 0.50) -> MatchResult:
    """Match by image/class for maximum TP count, then maximum total IoU.

    Every prediction and label participates in at most one pair.  Matching is
    solved separately for every ``(source_image_id, class_id)`` group so no
    cross-image or cross-class pairing is possible.  It first maximises true
    positive count and, among those solutions, maximises total IoU.  Stable
    index-order ties keep the result reproducible for a fixed input sequence.
    """
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)) or not math.isfinite(iou_threshold) or not 0 < float(iou_threshold) <= 1:
        raise _problem("IoU threshold is invalid", "the audit matching protocol is outside (0, 1]", "use a finite IoU threshold such as 0.50")
    if any(not isinstance(item, AuditBox) for item in (*predictions, *ground_truth)):
        raise _problem("audit matching received loose boxes", "unvalidated mappings could bypass class and geometry checks", "match AuditBox objects constructed from sealed artifacts")
    grouped: dict[tuple[str, int], dict[int, list[tuple[int, float]]]] = {}
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            if prediction.source_image_id != truth.source_image_id or prediction.class_id != truth.class_id:
                continue
            iou = box_iou(prediction.xyxy, truth.xyxy)
            if iou >= float(iou_threshold):
                key = (prediction.source_image_id, prediction.class_id)
                grouped.setdefault(key, {}).setdefault(pred_index, []).append((truth_index, iou))
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for key in sorted(grouped):
        edges = {
            prediction: tuple(sorted(options, key=lambda item: item[0]))
            for prediction, options in grouped[key].items()
        }
        cardinality = _maximum_cardinality({prediction: tuple(truth for truth, _ in options) for prediction, options in edges.items()})
        for prediction, truth in _maximum_weight_matching(edges, cardinality):
            matched_predictions.add(prediction)
            matched_truth.add(truth)
    return MatchResult(
        tuple(sorted(matched_predictions)), tuple(sorted(matched_truth)),
        tuple(index for index in range(len(predictions)) if index not in matched_predictions),
        tuple(index for index in range(len(ground_truth)) if index not in matched_truth),
    )


def calculate_pseudo_metrics(predictions: Sequence[AuditBox], ground_truth: Sequence[AuditBox], *, iou_threshold: float = 0.50) -> tuple[PseudoLabelMetrics, MatchResult]:
    """Calculate canonical per-class/overall audit metrics from sealed boxes."""
    result = one_to_one_match(predictions, ground_truth, iou_threshold=iou_threshold)
    values: dict[int, dict[str, int]] = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in _CLASS_IDS}
    for index in result.matched_prediction_indices:
        values[predictions[index].class_id]["tp"] += 1
    for index in result.unmatched_prediction_indices:
        values[predictions[index].class_id]["fp"] += 1
    for index in result.unmatched_ground_truth_indices:
        values[ground_truth[index].class_id]["fn"] += 1
    per_class = {class_id: _scores(counts) for class_id, counts in values.items()}
    total = {key: sum(counts[key] for counts in values.values()) for key in ("tp", "fp", "fn")}
    return PseudoLabelMetrics(per_class, _scores(total)), result


def _scores(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


@dataclass(frozen=True)
class SealedPseudoAudit:
    """Audit-only authority for Task 8's pseudo_audit label partition."""

    images: Mapping[str, AuditImage]
    labels: tuple[AuditBox, ...]
    split_fingerprint: str
    labels_sha256: str

    def __post_init__(self) -> None:
        images = dict(self.images)
        if not images or any(not isinstance(key, str) or key != value.source_image_id or not isinstance(value, AuditImage) for key, value in images.items()):
            raise _problem("sealed pseudo-audit membership is invalid", "image labels are not bound to a valid Task 8 pseudo_audit partition", "restore the paired Task 8 protected labels and split manifest")
        labels = tuple(self.labels)
        if any(not isinstance(label, AuditBox) or label.source_image_id not in images for label in labels):
            raise _problem("pseudo-audit labels do not match sealed membership", "a box refers to an image outside the pseudo_audit split", "restore the paired protected labels artifact")
        if not isinstance(self.split_fingerprint, str) or len(self.split_fingerprint) != 64 or any(char not in _CANONICAL_SHA256 for char in self.split_fingerprint):
            raise _problem("pseudo-audit fingerprint is malformed", "split provenance lacks a canonical SHA-256 digest", "use Task 8's fingerprints.protected/pseudo_audit")
        object.__setattr__(self, "images", MappingProxyType(images)); object.__setattr__(self, "labels", labels)


def load_sealed_pseudo_audit(labels_path: Path, split_manifest_path: Path) -> SealedPseudoAudit:
    """Open protected audit labels only after proving their Task 8 fingerprint.

    This function lives in the audit module and is intentionally imported by
    ``audit_pseudo_labels`` only.  Training, generation and filtering receive
    the no-label manifest APIs instead.
    """
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("sealed pseudo-audit input cannot be read", str(error), "supply the paired UTF-8 Task 8 pseudo_audit labels and split_manifest artifacts") from error
    if not isinstance(payload, Mapping) or set(payload) != {"records"} or not isinstance(payload.get("records"), list):
        raise _problem("audit labels are not the Task 8 protected labels artifact", "a generic labels file could expose data outside the sealed protocol", "use protected_splits/pseudo_audit_labels.json exactly")
    if not isinstance(split_payload, Mapping) or not isinstance(split_payload.get("split_image_ids"), Mapping) or not isinstance(split_payload.get("fingerprints"), Mapping):
        raise _problem("paired split manifest is malformed", "pseudo-audit membership cannot be proven", "use the unmodified Task 8 split_manifest.json")
    expected_ids = split_payload["split_image_ids"].get("pseudo_audit")
    expected_fp = split_payload["fingerprints"].get("protected/pseudo_audit")
    if not isinstance(expected_ids, list) or any(not isinstance(item, str) or not item for item in expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise _problem("pseudo-audit membership is malformed", "Task 8 did not seal unique pseudo_audit image IDs", "regenerate deterministic split outputs")
    if not isinstance(expected_fp, str) or len(expected_fp) != 64 or any(char not in _CANONICAL_SHA256 for char in expected_fp):
        raise _problem("pseudo-audit split fingerprint is malformed", "the paired manifest lacks fingerprints.protected/pseudo_audit", "regenerate deterministic split outputs")
    if _canonical_sha256(payload["records"]) != expected_fp:
        raise _problem("pseudo-audit labels differ from the sealed Task 8 split", "the protected labels were edited, substituted, or paired with a different split manifest", "use the exact matching pseudo_audit_labels.json and split_manifest.json")
    records = payload["records"]
    record_ids: list[str] = []
    images: dict[str, AuditImage] = {}; labels: list[AuditBox] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise _problem("pseudo-audit record is not an object", "protected labels have malformed image rows", "regenerate Task 8 outputs")
        required = {"source_image_id", "file_path", "width", "height", "labels"}
        if not required <= set(record) or not isinstance(record.get("labels"), list):
            raise _problem("pseudo-audit record lacks image or label fields", "protected labels are not Task 8 image-level records", "use the original protected pseudo_audit labels artifact")
        image = AuditImage(record["source_image_id"], record["file_path"], record["width"], record["height"])
        if image.source_image_id in images:
            raise _problem("pseudo-audit records duplicate an image ID", "one audit image appears more than once", "regenerate the Task 8 protected labels artifact")
        images[image.source_image_id] = image; record_ids.append(image.source_image_id)
        for raw_label in record["labels"]:
            if not isinstance(raw_label, Mapping):
                raise _problem("pseudo-audit label is not an object", "a protected label row is malformed", "regenerate the Task 8 labels artifact")
            class_id = _class_id(raw_label.get("class_id"))
            labels.append(AuditBox(image.source_image_id, class_id, _box(raw_label.get("xyxy"), width=image.width, height=image.height, field="pseudo-audit label xyxy")))
    if set(record_ids) != set(expected_ids) or len(record_ids) != len(expected_ids):
        raise _problem("pseudo-audit records do not match sealed membership", "a protected image was added, removed, or substituted", "use the exact Task 8 pseudo_audit labels artifact")
    return SealedPseudoAudit(images, tuple(labels), expected_fp, file_sha256(labels_path))


def load_audit_candidates(path: Path, audit: SealedPseudoAudit) -> tuple[str, tuple[PseudoCandidate, ...], str]:
    """Load Task 13-format evidence only when every path joins sealed audit IDs."""
    try:
        teacher_run_id, candidates = load_candidate_envelope(path)
    except Exception as error:
        raise _problem("audit candidates are not a validated candidate envelope", str(error), "supply an immutable Task 13-format candidate artifact") from error
    if not candidates:
        # An empty audit is valid; it must still have a known teacher identity.
        return teacher_run_id, (), file_sha256(path)
    for candidate in candidates:
        image = audit.images.get(candidate.source_image_id)
        if image is None:
            raise _problem("candidate refers outside pseudo_audit membership", "candidate source_image_id is not sealed for audit", "run audit inference only on Task 8 pseudo_audit images")
        if candidate.source_file_path != image.file_path or _safe_relative_path(candidate.source_file_path, "candidate source_file_path") != image.file_path:
            raise _problem("candidate image path differs from sealed pseudo_audit membership", "an absolute, traversing, or substituted path could leak another split", "use the exact Task 8 pseudo_audit file_path")
        _box(candidate.xyxy, width=image.width, height=image.height, field="candidate xyxy")
        _box(candidate.raw_xyxy, width=image.width, height=image.height, field="candidate raw_xyxy")
    return teacher_run_id, candidates, file_sha256(path)


def _sha256_field(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _CANONICAL_SHA256 for character in value):
        raise _problem(f"{field} is malformed", "sealed decision provenance lacks a canonical SHA-256 digest", "restore the unmodified Task 14 decision manifest")
    return value


def _load_decision_manifest(path: Path, *, teacher_run_id: str, candidate_sha256: str, audit_path: Path) -> tuple[object, int]:
    """Verify Task 14's immutable decision/provenance binding before rows load."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _problem("sealed Task 14 decision manifest cannot be read", str(error), "supply the decision_manifest.json published beside the paired audit.jsonl") from error
    if not isinstance(payload, Mapping) or set(payload) not in {_DECISION_MANIFEST_KEYS, _DECISION_MANIFEST_POLICY_KEYS}:
        raise _problem("sealed Task 14 decision manifest is malformed", "it has unsupported, missing, or label-bearing fields", "use the exact decision_manifest.json published by Task 14")
    if payload.get("schema_version") != "1.0" or payload.get("artifact_type") != "sealed_task14_filter_decisions":
        raise _problem("sealed Task 14 decision manifest has an unsupported schema", "the decision evidence is not a Task 14 output", "use Task 14 schema version 1.0 output")
    if payload.get("teacher_run_id") != teacher_run_id:
        raise _problem("sealed Task 14 decision manifest teacher differs", "decisions were paired with another teacher run", "use decisions generated from the exact audit candidate envelope")
    if _sha256_field(payload.get("candidate_artifact_sha256"), "candidate artifact digest") != candidate_sha256:
        raise _problem("sealed Task 14 decisions are bound to another candidate artifact", "the candidate envelope was substituted or the manifest was edited", "use the paired unmodified Task 13 candidates and Task 14 outputs")
    record_digest = _sha256_field(payload.get("decision_records_sha256"), "decision records digest")
    if record_digest != file_sha256(audit_path):
        raise _problem("sealed Task 14 decision records were modified", "a decision, reason, candidate field, or provenance differs from its immutable manifest", "restore the exact paired audit.jsonl output")
    count = payload.get("decision_record_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _problem("sealed Task 14 decision count is malformed", "the manifest cannot prove complete decision coverage", "restore the unmodified Task 14 decision manifest")
    provenance = payload.get("filter_provenance")
    provenance_digest = _sha256_field(payload.get("filter_provenance_sha256"), "filter provenance digest")
    if provenance_digest != _canonical_sha256(provenance):
        raise _problem("sealed Task 14 filter provenance was modified", "the policy provenance no longer matches its manifest digest", "restore the unmodified Task 14 decision manifest")
    if "filter_policy" in payload and _sha256_field(payload.get("filter_policy_sha256"), "filter policy digest") != _canonical_sha256(payload.get("filter_policy")):
        raise _problem("sealed Task 14 filter policy was modified", "the executable global/Trust/ablation policy no longer matches its manifest digest", "restore the unmodified Task 14 decision manifest")
    return provenance, count


def load_bound_filter_predictions(path: Path, candidates: Sequence[PseudoCandidate], audit: SealedPseudoAudit, *, teacher_run_id: str, candidate_artifact_sha256: str, decision_manifest_path: Path | None = None) -> tuple[AuditBox, ...]:
    """Read Task 14 decisions only when a sealed manifest binds every row.

    The companion manifest hashes the original Task 13 envelope and the exact
    JSONL bytes.  This makes an altered decision, reason code, or policy
    provenance unusable for the Task 15 refresh gate.
    """
    candidate_artifact_sha256 = _sha256_field(candidate_artifact_sha256, "candidate artifact digest")
    manifest_path = decision_manifest_path if decision_manifest_path is not None else path.with_name("decision_manifest.json")
    expected_provenance, expected_count = _load_decision_manifest(manifest_path, teacher_run_id=teacher_run_id, candidate_sha256=candidate_artifact_sha256, audit_path=path)
    # A detector can emit byte-identical boxes.  Preserve multiplicity rather
    # than collapsing these records in a set: the paired Task 14 audit must
    # account for every candidate occurrence, including duplicates later
    # rejected by NMS.
    candidate_rows = Counter((candidate.source_image_id, candidate.view, candidate.class_id, candidate.class_name, candidate.confidence, candidate.raw_xyxy, candidate.xyxy, candidate.source_model, candidate.teacher_run_id, candidate.source_file_path) for candidate in candidates)
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise _problem("filter audit contains a blank line", f"line {line_number} is not immutable JSONL evidence", "use the exact Task 14 audit.jsonl output")
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise _problem("filter audit record is not an object", f"line {line_number} is malformed", "use the exact Task 14 audit.jsonl output")
                if set(row) != (_AUDIT_EVENT_KEYS | {"filter_provenance"}):
                    raise _problem("filter audit record has unsupported fields", f"line {line_number} is not a Task 14 decision", "use an unmodified Task 14 audit.jsonl output")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, PseudoAuditError):
            raise
        raise _problem("filter audit cannot be read", str(error), "supply a readable Task 14 audit.jsonl output") from error
    if len(rows) != expected_count:
        raise _problem("sealed Task 14 decision count differs", "the audit JSONL was truncated or the manifest was substituted", "restore the exact paired Task 14 outputs")
    seen: Counter[tuple[object, ...]] = Counter(); accepted: list[AuditBox] = []
    for row in rows:
        if row["filter_provenance"] != expected_provenance:
            raise _problem("sealed Task 14 filter provenance differs", "an event carries policy provenance that differs from its immutable manifest", "restore the exact paired Task 14 audit.jsonl output")
        try:
            candidate = PseudoCandidate(**{key: row[key] for key in ("teacher_run_id", "source_image_id", "source_file_path", "view", "class_id", "class_name", "confidence", "raw_xyxy", "xyxy", "source_model")})
        except Exception as error:
            raise _problem("filter audit candidate provenance is invalid", str(error), "use unmodified Task 14 audit evidence") from error
        identity = (candidate.source_image_id, candidate.view, candidate.class_id, candidate.class_name, candidate.confidence, candidate.raw_xyxy, candidate.xyxy, candidate.source_model, candidate.teacher_run_id, candidate.source_file_path)
        if candidate_rows[identity] <= seen[identity]:
            raise _problem("filter audit is not bound one-to-one to candidate artifact", "a decision was injected, altered, duplicated, or comes from a different teacher artifact", "supply the paired unmodified Task 13 candidates and Task 14 audit.jsonl files")
        seen[identity] += 1
        if row.get("decision") not in {"accepted", "rejected"} or row.get("reason_code") is not None and not isinstance(row.get("reason_code"), str):
            raise _problem("filter audit decision is malformed", "decision/reason evidence is not a Task 14 record", "use an unmodified Task 14 audit.jsonl output")
        image = audit.images.get(candidate.source_image_id)
        if image is None or candidate.source_file_path != image.file_path:
            raise _problem("filter audit path leaks outside pseudo_audit membership", "filter decisions do not join sealed audit image provenance", "audit only the paired pseudo_audit artifacts")
        if row["decision"] == "accepted" and candidate.view == "original":
            accepted.append(AuditBox(candidate.source_image_id, candidate.class_id, _box(candidate.xyxy, width=image.width, height=image.height, field="filtered candidate xyxy"), candidate.confidence))
    if seen != candidate_rows:
        raise _problem("filter audit does not cover every candidate", "filter output was truncated or paired with a different candidate file", "supply the complete Task 14 audit.jsonl for this candidate artifact")
    return tuple(sorted(accepted, key=lambda item: (item.source_image_id, item.class_id, item.xyxy, -(item.confidence or 0.0))))


def original_view_predictions(candidates: Iterable[PseudoCandidate], audit: SealedPseudoAudit) -> tuple[AuditBox, ...]:
    """Use each original-view candidate exactly once for pre-filter metrics."""
    output: list[AuditBox] = []
    for candidate in candidates:
        if candidate.view != "original":
            continue
        image = audit.images[candidate.source_image_id]
        output.append(AuditBox(candidate.source_image_id, candidate.class_id, _box(candidate.xyxy, width=image.width, height=image.height, field="raw candidate xyxy"), candidate.confidence))
    return tuple(sorted(output, key=lambda item: (item.source_image_id, item.class_id, item.xyxy, -(item.confidence or 0.0))))


def pseudo_refresh_allowed(metrics: PseudoLabelMetrics, *, minimum_precision: float = 0.90) -> bool:
    """Return whether an audit supports one more pseudo-label refresh."""
    if not isinstance(metrics, PseudoLabelMetrics):
        raise _problem("pseudo refresh gate lacks validated metrics", "a loose score could authorize a contaminated refresh", "pass PseudoLabelMetrics from calculate_pseudo_metrics")
    if isinstance(minimum_precision, bool) or not isinstance(minimum_precision, (int, float)) or not math.isfinite(minimum_precision) or not 0 <= float(minimum_precision) <= 1:
        raise _problem("minimum audit precision is invalid", "refresh policy is outside [0, 1]", "use a precision threshold such as 0.90")
    return float(metrics.overall["precision"]) >= float(minimum_precision)
