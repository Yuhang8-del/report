"""Deterministic, image-group-level split construction for Fruit SSOD.

The allocator never sees one object annotation as a sampling unit.  It first
forms duplicate groups, then greedily chooses groups whose class-presence sets
improve the least represented classes.  A seeded SHA-256 tie breaker makes the
otherwise deterministic greedy order reproducible without depending on input
row order.  Group sizes can make an exact fraction impossible; in that case a
whole group is retained and the decision is recorded rather than split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord


class SplitError(ValueError):
    """Raised when a split would be invalid, leaky, or unsafe to write."""


def _problem(problem: str, cause: str, remediation: str) -> str:
    return f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}."


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitError(_problem(f"{field} must be a nonempty string", "an image identity or path was omitted", f"provide a nonempty {field} for every candidate image"))
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SplitError(_problem("label payload mapping has a non-string key", "JSON object keys must be strings and coercion could silently collide with an existing key", "replace every label mapping key with a unique string"))
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SplitError(_problem("label payload contains a non-finite number", "NaN or infinity cannot be represented in a canonical split fingerprint", "replace it with a finite JSON number before creating splits"))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SplitError(_problem("label payload is not JSON-compatible", f"unsupported value {type(value).__name__} was supplied", "use only JSON objects, arrays, strings, numbers, booleans, and null in labels"))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateImageRecord:
    """One image and every original label belonging to it, never one box row."""

    source_image_id: str
    file_path: str
    class_presence: frozenset[int]
    labels: tuple[Mapping[str, Any], ...]
    duplicate_group_id: str
    source: str = "unknown_source"
    width: int = 1
    height: int = 1
    license_metadata: LicenseMetadata = field(default_factory=lambda: LicenseMetadata(name="unspecified"))
    protected_split: str | None = None

    def __post_init__(self) -> None:
        _text(self.source_image_id, "source_image_id")
        _text(self.file_path, "file_path")
        _text(self.duplicate_group_id, "duplicate_group_id")
        _text(self.source, "source")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise SplitError(_problem("width must be a positive integer", "the image record has invalid dimensions", "provide the source image width"))
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise SplitError(_problem("height must be a positive integer", "the image record has invalid dimensions", "provide the source image height"))
        classes = frozenset(self.class_presence)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in classes):
            raise SplitError(_problem("class_presence must contain non-negative integer IDs", "a candidate has an invalid class-presence set", "provide canonical integer class IDs for the image"))
        if not isinstance(self.labels, tuple):
            raise SplitError(_problem("labels must be a tuple of image labels", "object rows were passed directly instead of one image record", "group object labels into the candidate image's labels array"))
        frozen_labels = tuple(_freeze(label) for label in self.labels)
        if any(not isinstance(label, Mapping) for label in frozen_labels):
            raise SplitError(_problem("labels must contain objects", "a label payload is not a JSON object", "provide a list of object-label mappings for each image"))
        if self.protected_split not in (None, "external_test"):
            raise SplitError(_problem("protected_split is unsupported", "only external_test is immutable in this split protocol", "use external_test or omit protected_split for source-pool images"))
        object.__setattr__(self, "class_presence", classes)
        object.__setattr__(self, "labels", frozen_labels)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateImageRecord":
        try:
            labels = value["labels"]
            classes = value["class_presence"]
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
                raise TypeError("labels must be an array")
            if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)):
                raise TypeError("class_presence must be an array")
            license_value = value.get("license_metadata", {"name": "unspecified"})
            if not isinstance(license_value, Mapping):
                raise TypeError("license_metadata must be an object")
            protected = value.get("protected_split")
            if protected is None and value.get("split") == "external_test":
                protected = "external_test"
            return cls(
                source_image_id=value["source_image_id"], file_path=value["file_path"],
                class_presence=frozenset(classes), labels=tuple(labels), duplicate_group_id=value["duplicate_group_id"],
                source=value.get("source", "unknown_source"), width=value.get("width", 1), height=value.get("height", 1),
                protected_split=protected,
                license_metadata=LicenseMetadata(name=license_value["name"], url=license_value.get("url"), attribution=license_value.get("attribution")),
            )
        except (KeyError, TypeError) as error:
            raise SplitError(_problem("candidate image mapping is missing required image-level fields", str(error), "supply source_image_id, file_path, class_presence, labels, and duplicate_group_id")) from error

    def mapping(self) -> dict[str, Any]:
        return {
            "source": self.source, "source_image_id": self.source_image_id, "file_path": self.file_path,
            "width": self.width, "height": self.height, "class_presence": sorted(self.class_presence),
            "labels": _thaw(self.labels), "duplicate_group_id": self.duplicate_group_id,
            "protected_split": self.protected_split,
            "license_metadata": {"name": self.license_metadata.name, "url": self.license_metadata.url, "attribution": self.license_metadata.attribution},
        }

    def unlabeled(self) -> UnlabeledImageRecord:
        """Return the deliberately label-free public representation."""
        return UnlabeledImageRecord(self.source, self.source_image_id, self.file_path, self.width, self.height, "train_pool", "unlabeled", self.license_metadata)


@dataclass(frozen=True)
class SplitProtocol:
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    pseudo_audit_fraction: float = 0.05
    unlabeled_fraction: float = 0.20
    split_seed: int = 42
    budget_seed: int = 3407
    unlabeled_seed: int = 2026
    budgets: tuple[int, ...] = (10, 20, 40, 100)

    def __post_init__(self) -> None:
        fractions = (self.validation_fraction, self.test_fraction, self.pseudo_audit_fraction, self.unlabeled_fraction)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in fractions):
            raise SplitError(_problem("split fractions must be numbers from zero through one", "a fraction is negative, over one, or not numeric", "use decimal fractions such as 0.10"))
        if self.validation_fraction + self.test_fraction + self.pseudo_audit_fraction >= 1:
            raise SplitError(_problem("protected source-pool fractions sum to one or more", "validation, test, and pseudo_audit leave no train pool", "reduce their sum below 1.0"))
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in (self.split_seed, self.budget_seed, self.unlabeled_seed)):
            raise SplitError(_problem("seeds must be integers", "a random seed has an invalid type", "supply integer values for split, budget, and unlabeled seeds"))
        budgets = tuple(self.budgets)
        if not budgets or budgets[-1] != 100 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > 100 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
            raise SplitError(_problem("budgets must be unique ascending percentages ending in 100", "the labelled-budget protocol is not nested", "use values such as 10,20,40,100"))
        object.__setattr__(self, "budgets", budgets)


@dataclass(frozen=True)
class DuplicateGroupDecision:
    group_id: str
    split: str
    source_image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.group_id, "duplicate_group_decision.group_id")
        _text(self.split, "duplicate_group_decision.split")
        if not isinstance(self.source_image_ids, Sequence) or isinstance(self.source_image_ids, (str, bytes)):
            raise SplitError(_problem("duplicate_group_decision.source_image_ids must be a sequence", "the duplicate decision omitted its image identifiers", "provide a list or tuple of nonempty source image IDs"))
        source_image_ids = tuple(self.source_image_ids)
        if not source_image_ids:
            raise SplitError(_problem("duplicate_group_decision.source_image_ids must not be empty", "a duplicate group decision has no member images", "provide every source image ID in the duplicate group"))
        for source_image_id in source_image_ids:
            _text(source_image_id, "duplicate_group_decision.source_image_ids item")
        object.__setattr__(self, "source_image_ids", source_image_ids)


@dataclass(frozen=True)
class SplitResult:
    """Frozen split result; the only unlabeled API is label-free by construction."""

    protocol: SplitProtocol
    protected_splits: Mapping[str, tuple[CandidateImageRecord, ...]]
    train_pool: tuple[CandidateImageRecord, ...]
    budgets: Mapping[str, tuple[CandidateImageRecord, ...]]
    unlabeled: tuple[UnlabeledImageRecord, ...]
    duplicate_group_decisions: tuple[DuplicateGroupDecision, ...]
    fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        def record_tuple(values: object, field_name: str, expected_type: type[object]) -> tuple[Any, ...]:
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise SplitError(_problem(f"{field_name} must be a record sequence", "the result has a malformed mutable container", "provide a list or tuple of validated records"))
            normalized = tuple(values)
            if any(not isinstance(record, expected_type) for record in normalized):
                raise SplitError(_problem(f"{field_name} contains an invalid record", f"{field_name} records have an unsupported type", "construct the result from the corresponding validated record model"))
            return normalized

        if not isinstance(self.protocol, SplitProtocol):
            raise SplitError(_problem("protocol has an invalid type", "the split result was not built with SplitProtocol", "provide a validated SplitProtocol instance"))
        if not isinstance(self.protected_splits, Mapping):
            raise SplitError(_problem("protected_splits must be a mapping", "a mutable or malformed result container was supplied", "provide validation, test, pseudo_audit, and external_test record sequences"))
        required_splits = {"validation", "test", "pseudo_audit", "external_test"}
        if set(self.protected_splits) != required_splits:
            raise SplitError(_problem("protected_splits has invalid keys", "a required protected split is missing or an unknown split was supplied", "provide exactly validation, test, pseudo_audit, and external_test"))
        protected = {name: record_tuple(records, f"protected_splits.{name}", CandidateImageRecord) for name, records in self.protected_splits.items()}
        if not isinstance(self.budgets, Mapping) or any(not isinstance(name, str) for name in self.budgets):
            raise SplitError(_problem("budgets must be a string-keyed mapping", "the budget container has invalid keys or type", "provide named budget record sequences"))
        budget_values = {name: record_tuple(records, f"budgets.{name}", CandidateImageRecord) for name, records in self.budgets.items()}
        train_pool = record_tuple(self.train_pool, "train_pool", CandidateImageRecord)
        unlabeled = record_tuple(self.unlabeled, "unlabeled", UnlabeledImageRecord)
        decisions = record_tuple(self.duplicate_group_decisions, "duplicate_group_decisions", DuplicateGroupDecision)
        if not isinstance(self.fingerprints, Mapping) or any(not isinstance(name, str) or not isinstance(value, str) for name, value in self.fingerprints.items()):
            raise SplitError(_problem("fingerprints must be a string mapping", "the result fingerprint container is malformed", "provide SHA-256 fingerprint strings keyed by artifact name"))
        object.__setattr__(self, "protected_splits", MappingProxyType(protected))
        object.__setattr__(self, "train_pool", train_pool)
        object.__setattr__(self, "budgets", MappingProxyType(budget_values))
        object.__setattr__(self, "unlabeled", unlabeled)
        object.__setattr__(self, "duplicate_group_decisions", decisions)
        object.__setattr__(self, "fingerprints", MappingProxyType(dict(self.fingerprints)))

    def unlabeled_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple({"source": item.source, "source_image_id": item.source_image_id, "file_path": item.file_path, "width": item.width, "height": item.height, "split": item.split, "label_status": item.label_status, "license_metadata": {"name": item.license_metadata.name, "url": item.license_metadata.url, "attribution": item.license_metadata.attribution}} for item in self.unlabeled)

    @property
    def validation(self) -> tuple[CandidateImageRecord, ...]:
        return self.protected_splits["validation"]

    @property
    def test(self) -> tuple[CandidateImageRecord, ...]:
        return self.protected_splits["test"]

    @property
    def pseudo_audit(self) -> tuple[CandidateImageRecord, ...]:
        return self.protected_splits["pseudo_audit"]

    @property
    def external_test(self) -> tuple[CandidateImageRecord, ...]:
        return self.protected_splits["external_test"]


def _group_records(records: Iterable[CandidateImageRecord]) -> dict[str, tuple[CandidateImageRecord, ...]]:
    groups: dict[str, list[CandidateImageRecord]] = {}
    seen_ids: set[str] = set()
    for record in records:
        if record.source_image_id in seen_ids:
            raise SplitError(_problem(f"source_image_id {record.source_image_id!r} occurs more than once", "candidate records are not one record per source image", "consolidate all labels for an image into one candidate record"))
        seen_ids.add(record.source_image_id)
        groups.setdefault(record.duplicate_group_id, []).append(record)
    output: dict[str, tuple[CandidateImageRecord, ...]] = {}
    for group_id, members in groups.items():
        protected = {item.protected_split for item in members}
        if len(protected) != 1:
            raise SplitError(_problem(f"duplicate group {group_id!r} spans conflicting protected inputs", "some near-duplicate images are external_test while others are source-pool candidates", "place the complete duplicate group in one protected state before splitting"))
        output[group_id] = tuple(sorted(members, key=lambda item: (item.source_image_id.casefold(), item.source_image_id, item.file_path)))
    return output


def _tie(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def _select_groups(groups: Mapping[str, tuple[CandidateImageRecord, ...]], target_images: int, seed: int) -> set[str]:
    """Select whole groups with deterministic proportional class-presence balance.

    Earlier selection maximized rarity on every draw.  That is suitable for a
    one-off audit sample, but it repeatedly exhausted the rarest classes
    before the nested 10/20/40 percent label budgets were drawn.  The result
    can satisfy an image-level coverage check while giving a class only one
    object in the actual Teacher budget.  This allocator instead tracks the
    class-presence distribution of its *current input* and greedily reduces
    the distance to the proportional target for every class.
    """
    if target_images <= 0 or not groups:
        return set()
    available = set(groups)
    selected: set[str] = set()
    class_total: dict[int, int] = {}
    group_presence: dict[str, dict[int, int]] = {}
    for members in groups.values():
        for item in members:
            for class_id in item.class_presence:
                class_total[class_id] = class_total.get(class_id, 0) + 1
    for group_id, members in groups.items():
        counts: dict[int, int] = {}
        for item in members:
            for class_id in item.class_presence:
                counts[class_id] = counts.get(class_id, 0) + 1
        group_presence[group_id] = counts
    population = sum(len(members) for members in groups.values())
    targets = {
        class_id: target_images * class_total[class_id] / population
        for class_id in class_total
    }
    selected_count = 0
    selected_class: dict[int, int] = {}
    # A protected split with enough capacity must be diagnostically useful for
    # every canonical class.  Pure global rarity scoring can otherwise spend a
    # small pseudo-audit split entirely on multi-label groups and silently omit
    # a class.  Seed one whole duplicate group per class before filling the
    # remaining quota; if the requested split is smaller than the class count,
    # retain the original rarity-only behaviour rather than exceed its size.
    if target_images >= len(class_total):
        for class_id in sorted(class_total, key=lambda value: (class_total[value], value)):
            candidates = [group_id for group_id in available if any(class_id in item.class_presence for item in groups[group_id])]
            if not candidates:
                continue
            def coverage_score(group_id: str) -> tuple[int, float, str, str]:
                present = group_presence[group_id]
                uncovered = sum(1 for value in present if selected_class.get(value, 0) == 0)
                deficit = sum(max(0.0, targets[value] - selected_class.get(value, 0)) / max(targets[value], 1.0) for value in present)
                return (uncovered, deficit, _tie(seed, group_id), group_id)
            chosen = max(candidates, key=coverage_score)
            available.remove(chosen)
            selected.add(chosen)
            selected_count += len(groups[chosen])
            for value, count in group_presence[chosen].items():
                selected_class[value] = selected_class.get(value, 0) + count
    while available and selected_count < target_images:
        def score(group_id: str) -> tuple[float, str, str]:
            counts = group_presence[group_id]
            gain = sum(
                (
                    min(float(count), max(0.0, targets[class_id] - selected_class.get(class_id, 0)))
                    / max(targets[class_id], 1e-9)
                    / class_total[class_id]
                )
                for class_id, count in counts.items()
            )
            overshoot = sum(
                max(0.0, selected_class.get(class_id, 0) + count - targets[class_id]) / max(targets[class_id], 1.0)
                for class_id, count in counts.items()
            )
            # The small penalty prevents one very common class from consuming
            # the entire remaining quota once its proportional target is met.
            return (gain - 0.05 * overshoot, _tie(seed, group_id), group_id)
        group_id = max(available, key=score)
        available.remove(group_id)
        selected.add(group_id)
        selected_count += len(groups[group_id])
        for class_id, count in group_presence[group_id].items():
            selected_class[class_id] = selected_class.get(class_id, 0) + count
    return selected


def _records_for(groups: Mapping[str, tuple[CandidateImageRecord, ...]], selected: Iterable[str]) -> tuple[CandidateImageRecord, ...]:
    return tuple(record for group_id in sorted(selected) for record in groups[group_id])


def _as_candidates(records: Iterable[CandidateImageRecord | Mapping[str, Any]]) -> tuple[CandidateImageRecord, ...]:
    return tuple(record if isinstance(record, CandidateImageRecord) else CandidateImageRecord.from_mapping(record) for record in records)


def split_records(records: Iterable[CandidateImageRecord | Mapping[str, Any]], *, validation_fraction: float = 0.10, test_fraction: float = 0.10, pseudo_audit_fraction: float = 0.05, unlabeled_fraction: float = 0.20, split_seed: int = 42, budget_seed: int = 3407, unlabeled_seed: int = 2026, budgets: Sequence[int] = (10, 20, 40, 100)) -> SplitResult:
    """Create a deterministic protocol from image candidates without writing files."""
    protocol = SplitProtocol(validation_fraction, test_fraction, pseudo_audit_fraction, unlabeled_fraction, split_seed, budget_seed, unlabeled_seed, tuple(budgets))
    grouped = _group_records(_as_candidates(records))
    external_groups = {group_id: members for group_id, members in grouped.items() if members[0].protected_split == "external_test"}
    source_groups = {group_id: members for group_id, members in grouped.items() if members[0].protected_split is None}
    source_count = sum(len(members) for members in source_groups.values())
    remaining = dict(source_groups)
    selected_by_split: dict[str, set[str]] = {}
    for name, fraction in (("validation", protocol.validation_fraction), ("test", protocol.test_fraction), ("pseudo_audit", protocol.pseudo_audit_fraction)):
        selected = _select_groups(remaining, round(source_count * fraction), protocol.split_seed)
        selected_by_split[name] = selected
        remaining = {group_id: members for group_id, members in remaining.items() if group_id not in selected}
    train_groups = remaining
    unlabeled_groups = _select_groups(train_groups, round(sum(len(members) for members in train_groups.values()) * protocol.unlabeled_fraction), protocol.unlabeled_seed)
    labeled_train_groups = {group_id: members for group_id, members in train_groups.items() if group_id not in unlabeled_groups}
    labeled_count = sum(len(members) for members in labeled_train_groups.values())
    selected_budgets: dict[str, tuple[CandidateImageRecord, ...]] = {}
    for percentage in protocol.budgets:
        # A seed-specific global group order makes every percentage a prefix of 100%.
        chosen = _select_groups(labeled_train_groups, round(labeled_count * percentage / 100), protocol.budget_seed)
        selected_budgets[str(percentage)] = _records_for(labeled_train_groups, chosen)
    protected = {name: _records_for(source_groups, selected_by_split[name]) for name in ("validation", "test", "pseudo_audit")}
    protected["external_test"] = _records_for(external_groups, external_groups)
    train_pool = _records_for(labeled_train_groups, labeled_train_groups)
    unlabeled = tuple(record.unlabeled() for record in _records_for(train_groups, unlabeled_groups))
    decisions = tuple(sorted((DuplicateGroupDecision(group_id, "external_test" if members[0].protected_split else next((name for name, chosen in selected_by_split.items() if group_id in chosen), "unlabeled" if group_id in unlabeled_groups else "train_pool"), tuple(item.source_image_id for item in members)) for group_id, members in grouped.items()), key=lambda item: item.group_id))
    result_without_fingerprints = {
        "protocol": {"validation_fraction": protocol.validation_fraction, "test_fraction": protocol.test_fraction, "pseudo_audit_fraction": protocol.pseudo_audit_fraction, "unlabeled_fraction": protocol.unlabeled_fraction, "split_seed": protocol.split_seed, "budget_seed": protocol.budget_seed, "unlabeled_seed": protocol.unlabeled_seed, "budgets": list(protocol.budgets)},
        "protected": {name: [record.mapping() for record in values] for name, values in protected.items()},
        "train_pool": [record.mapping() for record in train_pool], "budgets": {name: [record.mapping() for record in values] for name, values in selected_budgets.items()},
        "unlabeled": [{"source": item.source, "source_image_id": item.source_image_id, "file_path": item.file_path, "width": item.width, "height": item.height, "split": item.split, "label_status": item.label_status} for item in unlabeled],
        "decisions": [{"group_id": item.group_id, "split": item.split, "source_image_ids": list(item.source_image_ids)} for item in decisions],
    }
    fingerprints = {"split_protocol": _fingerprint(result_without_fingerprints)}
    for name, values in protected.items():
        fingerprints[f"protected/{name}"] = _fingerprint([record.mapping() for record in values])
    for name, values in selected_budgets.items():
        fingerprints[f"budget/{name}"] = _fingerprint([record.mapping() for record in values])
    fingerprints["unlabeled"] = _fingerprint(result_without_fingerprints["unlabeled"])
    fingerprints["duplicate_group_decisions"] = _fingerprint(result_without_fingerprints["decisions"])
    return SplitResult(protocol, protected, train_pool, selected_budgets, unlabeled, decisions, fingerprints)


def _output_payloads(result: SplitResult) -> dict[str, Any]:
    split_image_ids = {name: [record.source_image_id for record in records] for name, records in result.protected_splits.items()}
    return {
        "split_manifest.json": {"manifest_version": "1.0", "protocol": {"validation_fraction": result.protocol.validation_fraction, "test_fraction": result.protocol.test_fraction, "pseudo_audit_fraction": result.protocol.pseudo_audit_fraction, "unlabeled_fraction": result.protocol.unlabeled_fraction, "split_seed": result.protocol.split_seed, "budget_seed": result.protocol.budget_seed, "unlabeled_seed": result.protocol.unlabeled_seed, "budgets": list(result.protocol.budgets)}, "split_image_ids": split_image_ids, "train_pool_image_ids": [record.source_image_id for record in result.train_pool], "budget_image_ids": {name: [record.source_image_id for record in records] for name, records in result.budgets.items()}, "unlabeled_image_ids": [record.source_image_id for record in result.unlabeled], "fingerprints": dict(result.fingerprints)},
        "unlabeled.json": {"records": list(result.unlabeled_manifest())},
        "duplicate_group_decisions.json": {"decisions": [{"group_id": item.group_id, "split": item.split, "source_image_ids": list(item.source_image_ids)} for item in result.duplicate_group_decisions]},
        "fingerprints.json": {"fingerprints": dict(result.fingerprints)},
        **{f"protected_splits/{name}_labels.json": {"records": [record.mapping() for record in records]} for name, records in result.protected_splits.items()},
        **{f"budgets/{name}/images.json": {"records": [{"source": record.source, "source_image_id": record.source_image_id, "file_path": record.file_path, "width": record.width, "height": record.height, "duplicate_group_id": record.duplicate_group_id} for record in records]} for name, records in result.budgets.items()},
        **{f"budgets/{name}/labels.json": {"records": [record.mapping() for record in records]} for name, records in result.budgets.items()},
    }


def write_split_outputs(result: SplitResult, output_root: Path, *, input_manifest: Path, dry_run: bool = False, source_root: Path | None = None) -> tuple[Path, ...]:
    """Write only the explicit output tree after collision checks; dry-run writes none."""
    root = output_root.resolve(strict=False)
    input_path = input_manifest.resolve(strict=False)
    payloads = _output_payloads(result)
    source_base = source_root.resolve(strict=False) if source_root is not None else input_path.parent
    source_paths = {(source_base / record.file_path).resolve(strict=False) if not Path(record.file_path).is_absolute() else Path(record.file_path).resolve(strict=False) for records in result.protected_splits.values() for record in records}
    source_paths.update((source_base / record.file_path).resolve(strict=False) if not Path(record.file_path).is_absolute() else Path(record.file_path).resolve(strict=False) for record in result.train_pool)
    # Unlabeled paths are present only in the safe records, so include them separately.
    source_paths.update((source_base / record.file_path).resolve(strict=False) if not Path(record.file_path).is_absolute() else Path(record.file_path).resolve(strict=False) for record in result.unlabeled)
    paths = tuple((root / relative).resolve(strict=False) for relative in payloads)
    if root == input_path or root in source_paths or input_path in paths or any(path in source_paths for path in paths):
        raise SplitError(_problem("generated split output collides with an input file or source image", "the explicit output root aliases protected input content", "choose an empty output root distinct from the manifest and every source image path"))
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise SplitError(_problem("generated split artifact already exists", f"{existing} would be overwritten", "choose a new empty output root or archive the prior generated artifacts before rerunning"))
    if root.exists():
        raise SplitError(_problem("output root already exists", f"{root} cannot be atomically published without replacing existing content", "choose a new output root that does not yet exist"))
    for path in paths:
        ancestor = path.parent
        while ancestor != ancestor.parent:
            if ancestor.exists() and not ancestor.is_dir():
                raise SplitError(_problem("an output artifact has a file as an ancestor", f"{ancestor} is a file, not a directory", "choose an output root whose parent hierarchy consists only of directories"))
            ancestor = ancestor.parent
    if dry_run:
        return ()
    temporary_root: Path | None = None
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        for path, payload in zip(paths, payloads.values()):
            temporary_path = temporary_root / path.relative_to(root)
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary_root, root)
    except OSError as error:
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise SplitError(_problem("split outputs could not be written atomically", str(error), "ensure the output parent is writable and retry with a new empty output root")) from error
    return paths
