"""Tests for deterministic, leakage-safe image-group splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.cli.create_splits import main
from fruit_ssod.data.splitting import CandidateImageRecord, DuplicateGroupDecision, SplitError, SplitResult, split_records, write_split_outputs


def _record(image_id: str, group: str, classes: set[int], *, protected: str | None = None) -> dict[str, object]:
    return {
        "source": "fixture",
        "source_image_id": image_id,
        "file_path": f"images/{image_id}.jpg",
        "width": 12,
        "height": 10,
        "class_presence": sorted(classes),
        "labels": [{"class_id": class_id, "xyxy": [1, 1, 4, 4]} for class_id in sorted(classes)],
        "duplicate_group_id": group,
        "protected_split": protected,
        "license_metadata": {"name": "fixture"},
    }


def _records() -> list[dict[str, object]]:
    rows = [_record(f"id-{index}", f"group-{index}", {index % 3, (index + 1) % 3}) for index in range(20)]
    rows.extend([_record("dup-a", "duplicate-pair", {0, 2}), _record("dup-b", "duplicate-pair", {0, 2})])
    rows.append(_record("external", "external-group", {1}, protected="external_test"))
    return rows


def test_image_group_splits_are_deterministic_nested_and_safe() -> None:
    first = split_records(_records())
    second = split_records(_records())

    assert first.fingerprints == second.fingerprints
    assigned = [record.source_image_id for records in first.protected_splits.values() for record in records]
    assigned.extend(record.source_image_id for record in first.train_pool)
    assigned.extend(record.source_image_id for record in first.unlabeled)
    assert len(assigned) == len(set(assigned))
    decisions = {decision.group_id: decision for decision in first.duplicate_group_decisions}
    assert set(decisions["duplicate-pair"].source_image_ids) == {"dup-a", "dup-b"}
    assert len({decisions["duplicate-pair"].split}) == 1
    budget_ids = {name: {record.source_image_id for record in records} for name, records in first.budgets.items()}
    assert budget_ids["10"] <= budget_ids["20"] <= budget_ids["40"] <= budget_ids["100"]
    assert all(record.label_status == "unlabeled" for record in first.unlabeled)
    assert all(not hasattr(record, "labels") and not hasattr(record, "class_presence") for record in first.unlabeled)
    assert not ({record.source_image_id for record in first.unlabeled} & {record.source_image_id for record in first.protected_splits["pseudo_audit"]})


def test_conflicting_duplicate_protection_and_bad_identifiers_are_actionable() -> None:
    rows = [_record("a", "same", {0}, protected="external_test"), _record("b", "same", {0})]
    with pytest.raises(SplitError, match="Problem:") as error:
        split_records(rows)
    assert "Remediation:" in str(error.value)
    with pytest.raises(SplitError, match="duplicate_group_id"):
        split_records([_record("a", "", {0})])


def test_multilabel_seed_protocol_and_unlabeled_serialization_are_deterministic() -> None:
    result = split_records(_records(), split_seed=42, budget_seed=3407, unlabeled_seed=2026)
    assert result.protocol.split_seed == 42
    assert result.protocol.budget_seed == 3407
    assert result.protocol.unlabeled_seed == 2026
    payload = result.unlabeled_manifest()
    assert all("labels" not in row and "class_id" not in row and "xyxy" not in row for row in payload)
    assert result.fingerprints == split_records(_records(), split_seed=42, budget_seed=3407, unlabeled_seed=2026).fingerprints


def test_multilabel_stratification_prioritizes_a_rare_class_for_small_target() -> None:
    rows = [_record(f"common-{index}", f"common-{index}", {0, 1}) for index in range(5)]
    rows.append(_record("rare", "rare", {2}))

    result = split_records(rows, validation_fraction=1 / 6, test_fraction=0, pseudo_audit_fraction=0, unlabeled_fraction=0)

    assert [record.source_image_id for record in result.validation] == ["rare"]


def test_protected_splits_cover_every_available_class_when_their_capacity_allows() -> None:
    """Small audit/test partitions cannot silently omit a canonical class."""
    rows = [_record(f"class-{class_id}-{index}", f"class-{class_id}-{index}", {class_id}) for class_id in range(5) for index in range(8)]

    result = split_records(rows, validation_fraction=.2, test_fraction=.2, pseudo_audit_fraction=.2, unlabeled_fraction=0)

    for records in (result.validation, result.test, result.pseudo_audit):
        assert set().union(*(record.class_presence for record in records)) == set(range(5))


def test_nested_budget_preserves_each_class_proportion_instead_of_exhausting_rare_classes() -> None:
    rows = [
        _record(f"class-{class_id}-{index}", f"class-{class_id}-{index}", {class_id})
        for class_id in range(5) for index in range(40)
    ]
    result = split_records(
        rows, validation_fraction=.1, test_fraction=.1, pseudo_audit_fraction=.05,
        unlabeled_fraction=.2, budgets=(20, 100),
    )
    budget = result.budgets["20"]
    counts = {class_id: sum(class_id in record.class_presence for record in budget) for class_id in range(5)}
    # Each source class has the same 20% allocation after protected/unlabeled
    # removal; no class may be reduced to coverage-only status.
    assert min(counts.values()) >= 4
    assert max(counts.values()) - min(counts.values()) <= 1


def test_writer_dry_run_and_collision_safety(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"images": _records()}), encoding="utf-8")
    result = split_records(_records())
    output = tmp_path / "output"
    written = write_split_outputs(result, output, input_manifest=source, dry_run=True)
    assert written == ()
    assert not output.exists()
    with pytest.raises(SplitError, match="collides"):
        write_split_outputs(result, source, input_manifest=source)
    assert source.exists()


def test_writer_refuses_existing_generated_artifacts_without_overwriting(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"images": _records()}), encoding="utf-8")
    result = split_records(_records())
    output = tmp_path / "output"
    write_split_outputs(result, output, input_manifest=source)
    manifest = output / "split_manifest.json"
    original = manifest.read_bytes()

    with pytest.raises(SplitError, match="already exists") as error:
        write_split_outputs(result, output, input_manifest=source)

    assert "Remediation:" in str(error.value)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_label_values_are_rejected_before_fingerprinting(bad_number: float) -> None:
    rows = _records()
    rows[0]["labels"] = [{"class_id": 0, "xyxy": [1, bad_number, 4, 4]}]
    with pytest.raises(SplitError, match="non-finite") as error:
        split_records(rows)
    assert "Problem:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_label_mapping_keys_are_not_coerced_or_silently_collided() -> None:
    with pytest.raises(SplitError, match="non-string key") as error:
        CandidateImageRecord("image", "image.jpg", frozenset({0}), ({1: "a", "1": "b"},), "group")
    assert "Remediation:" in str(error.value)


def test_split_result_copies_supplied_sequences_and_mappings_into_immutable_values() -> None:
    original = split_records(_records())
    protected = {name: list(records) for name, records in original.protected_splits.items()}
    train_pool = list(original.train_pool)
    budgets = {name: list(records) for name, records in original.budgets.items()}
    unlabeled = list(original.unlabeled)
    mutable_decision_ids = list(original.duplicate_group_decisions[0].source_image_ids)
    decisions = [
        DuplicateGroupDecision(
            original.duplicate_group_decisions[0].group_id,
            original.duplicate_group_decisions[0].split,
            mutable_decision_ids,
        ),
        *original.duplicate_group_decisions[1:],
    ]
    fingerprints = dict(original.fingerprints)
    result = SplitResult(original.protocol, protected, train_pool, budgets, unlabeled, decisions, fingerprints)
    expected_validation = result.validation
    expected_train = result.train_pool
    expected_budget = result.budgets["100"]
    expected_decision_ids = result.duplicate_group_decisions[0].source_image_ids

    protected["validation"].clear()
    train_pool.clear()
    budgets["100"].clear()
    unlabeled.clear()
    decisions.clear()
    fingerprints.clear()
    mutable_decision_ids.append("caller-added-id")

    assert result.validation == expected_validation
    assert result.train_pool == expected_train
    assert result.budgets["100"] == expected_budget
    assert result.duplicate_group_decisions[0].source_image_ids == expected_decision_ids
    assert isinstance(result.validation, tuple)
    with pytest.raises(TypeError):
        result.protected_splits["validation"] = ()  # type: ignore[index]


def test_cli_validates_fractions_and_writes_explicit_root_only(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"images": _records()}), encoding="utf-8")
    output = tmp_path / "output"
    assert main(["--input-manifest", str(source), "--output-root", str(output), "--dry-run"]) == 0
    assert not output.exists()
    with pytest.raises(SystemExit) as exit_status:
        main(["--input-manifest", str(source), "--output-root", str(output), "--validation-fraction", "0.8", "--test-fraction", "0.3"])
    assert exit_status.value.code == 2
    assert main(["--input-manifest", str(source), "--output-root", str(output)]) == 0
    assert (output / "split_manifest.json").is_file()


def test_cli_rejects_non_finite_local_json_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "nonfinite.json"
    rows = _records()
    rows[0]["labels"] = [{"class_id": 0, "xyxy": [1, float("nan"), 4, 4]}]
    source.write_text(json.dumps({"images": rows}), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(SystemExit) as exit_status:
        main(["--input-manifest", str(source), "--output-root", str(output)])

    assert exit_status.value.code == 2
    assert not output.exists()


def test_writer_preflights_file_parent_and_cleans_up_after_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"images": _records()}), encoding="utf-8")
    result = split_records(_records())
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    blocked_root = blocker / "output"

    with pytest.raises(SplitError, match="file as an ancestor"):
        write_split_outputs(result, blocked_root, input_manifest=source)
    assert not blocked_root.exists()

    output = tmp_path / "output"

    def write_failure(self: Path, *args: object, **kwargs: object) -> int:
        raise OSError("synthetic output failure")

    monkeypatch.setattr(Path, "write_text", write_failure)
    with pytest.raises(SplitError, match="atomically") as error:
        write_split_outputs(result, output, input_manifest=source)
    assert "Remediation:" in str(error.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))
