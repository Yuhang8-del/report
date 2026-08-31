from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.pseudo.candidates import PseudoCandidate
from fruit_ssod.pseudo.thresholds import PerClassThresholds
from fruit_ssod.pseudo.trust_filter import (
    AspectRatioBoundsArtifact, ImageGeometry, TrustFilter, TrustFilterConfig, TrustFilterError,
    load_aspect_ratio_bounds_artifact, load_candidate_envelope,
    write_trust_filter_outputs,
)
from fruit_ssod.pseudo.transforms import horizontal_flip_xyxy


def _candidate(*, view: str, box: tuple[float, float, float, float] = (100, 100, 200, 200), confidence: float = .9, class_id: int = 0, image: str = "image-a") -> PseudoCandidate:
    raw_box = box if view == "original" else horizontal_flip_xyxy(box, width=640)
    return PseudoCandidate("teacher-20", image, f"{image}.jpg", view, class_id, ("Apple", "Banana", "Orange", "Strawberry", "Pineapple")[class_id], confidence, raw_box, box, "teacher.pt")


def _filter(**config: object) -> TrustFilter:
    return TrustFilter(PerClassThresholds({index: .5 for index in range(5)}), config=TrustFilterConfig(**config))


def _geometry() -> dict[str, ImageGeometry]:
    return {"image-a": ImageGeometry("image-a", 640, 640)}


def _filter_envelope(tmp_path: Path, candidates: tuple[PseudoCandidate, ...], **config: object):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({
        "manifest_version": "1.0", "teacher_run_id": "teacher-20",
        "candidate_count": len(candidates), "candidates": [candidate.mapping() for candidate in candidates],
    }), encoding="utf-8")
    return _filter(**config).filter_envelope(path, _geometry())


def _bounds_payload() -> dict[str, object]:
    return {
        "artifact_version": "1.0",
        "artifact_type": "sealed_aspect_ratio_bounds",
        "artifact_id": "train-pool-ar-v1",
        "class_registry_version": "1.0.1",
        "classes": [
            {"id": 0, "name": "Apple"}, {"id": 1, "name": "Banana"},
            {"id": 2, "name": "Orange"}, {"id": 3, "name": "Strawberry"},
            {"id": 4, "name": "Pineapple"},
        ],
        "provenance": {
            "source_split": "train_pool", "source_kind": "approved_aggregate_statistics",
            "contains_human_labels": False, "sealed": True,
        },
        "bounds": {str(index): [.1, 10.0] for index in range(5)},
    }


def test_filter_requires_same_class_cross_view_iou_and_records_all_rejections() -> None:
    original = _candidate(view="original")
    mismatch = _candidate(view="horizontal_flip", box=(300, 300, 400, 400))
    result = _filter().filter("teacher-20", (original, mismatch), _geometry())
    assert not result.accepted
    assert [row["reason_code"] for row in result.audit] == ["no_cross_view_match", "no_cross_view_match"]


@pytest.mark.parametrize("box,reason", [
    ((0, 0, 15, 100), "too_small_at_640"),
    ((0, 0, 620, 620), "area_too_large"),
    ((1, 1, 2, 200), "too_small_at_640"),
])
def test_filter_rejects_scale_and_area_guards(box: tuple[float, float, float, float], reason: str) -> None:
    result = _filter().filter("teacher-20", (_candidate(view="original", box=box), _candidate(view="horizontal_flip", box=box)), _geometry())
    assert {row["reason_code"] for row in result.audit} == {reason}


def test_filter_enforces_distribution_aspect_bounds_and_global_threshold(tmp_path: Path) -> None:
    payload = _bounds_payload(); payload["bounds"]["0"] = [0.8, 1.2]  # type: ignore[index]
    path = tmp_path / "bounds.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    config = {"aspect_ratio_bounds_artifact": load_aspect_ratio_bounds_artifact(path)}
    result = _filter(**config).filter("teacher-20", (_candidate(view="original", box=(100, 100, 300, 150)), _candidate(view="horizontal_flip", box=(100, 100, 300, 150))), _geometry())
    assert {row["reason_code"] for row in result.audit} == {"aspect_ratio_outside_labeled_distribution"}
    result = _filter(global_confidence=.8).filter("teacher-20", (_candidate(view="original", confidence=.7), _candidate(view="horizontal_flip", confidence=.7)), _geometry())
    assert {row["reason_code"] for row in result.audit} == {"below_global_confidence"}


def test_filter_is_deterministic_and_applies_nms_and_twenty_box_cap() -> None:
    candidates: list[PseudoCandidate] = []
    for index in range(22):
        x = index * 20
        candidates.extend((_candidate(view="original", box=(x, 100, x + 18, 130), image="image-a"), _candidate(view="horizontal_flip", box=(x, 100, x + 18, 130), image="image-a")))
    result = _filter().filter("teacher-20", tuple(reversed(candidates)), _geometry())
    assert len(result.accepted) == 20
    assert sum(row["reason_code"] == "max_boxes_per_image" for row in result.audit) == 4
    duplicated = (_candidate(view="original", box=(100, 100, 200, 200), confidence=.8), _candidate(view="horizontal_flip", box=(100, 100, 200, 200), confidence=.8), _candidate(view="original", box=(101, 101, 201, 201), confidence=.8), _candidate(view="horizontal_flip", box=(101, 101, 201, 201), confidence=.8))
    assert sum(row["reason_code"] == "nms_duplicate" for row in _filter().filter("teacher-20", duplicated, _geometry()).audit) == 2


def test_no_view_consistency_rejects_each_original_candidate_once_after_nms() -> None:
    candidates = (
        _candidate(view="original", confidence=.9),
        _candidate(view="horizontal_flip", confidence=.9),
        _candidate(view="original", box=(101, 101, 201, 201), confidence=.8),
        _candidate(view="horizontal_flip", box=(101, 101, 201, 201), confidence=.8),
    )
    result = _filter(require_view_consistency=False).filter("teacher-20", candidates, _geometry())
    nms = [row for row in result.audit if row["reason_code"] == "nms_duplicate"]
    assert len(nms) == 1 and nms[0]["view"] == "original"


def test_filter_exports_yolo_and_complete_jsonl_audit_atomically(tmp_path: Path) -> None:
    result = _filter_envelope(tmp_path, (_candidate(view="original"), _candidate(view="horizontal_flip")))
    labels, audit = write_trust_filter_outputs(result, tmp_path / "filtered")
    assert labels.joinpath("image-a.txt").read_text(encoding="utf-8") == "0 0.23437500 0.23437500 0.15625000 0.15625000\n"
    rows = [json.loads(row) for row in audit.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {row["decision"] for row in rows} == {"accepted"}
    with pytest.raises(TrustFilterError, match="already exists"):
        write_trust_filter_outputs(result, tmp_path / "filtered")
    unbound = _filter().filter("teacher-20", (_candidate(view="original"), _candidate(view="horizontal_flip")), _geometry())
    with pytest.raises(TrustFilterError, match="sealed candidate binding"):
        write_trust_filter_outputs(unbound, tmp_path / "unbound")


def test_filter_rejects_tampered_original_and_flip_coordinate_provenance() -> None:
    original = PseudoCandidate(
        "teacher-20", "image-a", "image-a.jpg", "original", 0, "Apple", .9,
        (100, 100, 200, 200), (101, 100, 201, 200), "teacher.pt",
    )
    flipped = PseudoCandidate(
        "teacher-20", "image-a", "image-a.jpg", "horizontal_flip", 0, "Apple", .9,
        (440, 100, 540, 200), (99, 100, 199, 200), "teacher.pt",
    )
    result = _filter().filter("teacher-20", (original, flipped), _geometry())
    assert not result.accepted
    assert {event["reason_code"] for event in result.audit} == {
        "original_raw_xyxy_mismatch", "flip_raw_xyxy_mapping_mismatch",
    }


def test_sealed_aspect_ratio_bounds_are_five_class_label_free_immutable_and_audited(tmp_path: Path) -> None:
    artifact_path = tmp_path / "bounds.json"
    artifact_path.write_text(json.dumps(_bounds_payload()), encoding="utf-8")
    artifact = load_aspect_ratio_bounds_artifact(artifact_path)
    with pytest.raises(TypeError):
        artifact.bounds[0] = (.2, .8)  # type: ignore[index]
    with pytest.raises(TypeError):
        artifact.provenance["sealed"] = False  # type: ignore[index]
    config = TrustFilterConfig(
        aspect_ratio_bounds_artifact=artifact,
    )
    with pytest.raises(TypeError):
        config.labeled_aspect_ratio_bounds[0] = (.2, .8)  # type: ignore[index,union-attr]
    result = _filter_envelope(tmp_path, (_candidate(view="original"), _candidate(view="horizontal_flip")), aspect_ratio_bounds_artifact=artifact)
    _, audit_path = write_trust_filter_outputs(result, tmp_path / "filtered")
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["filter_provenance"]["aspect_ratio_bounds"]["artifact_id"] == "train-pool-ar-v1" for row in rows)
    assert all(row["filter_provenance"]["aspect_ratio_bounds"]["provenance"]["contains_human_labels"] is False for row in rows)


def test_bounds_cannot_be_constructed_or_injected_as_raw_config(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        AspectRatioBoundsArtifact("id", "0" * 64, "sealed_aspect_ratio_bounds", "1.0.0", {}, {})  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TrustFilterConfig(labeled_aspect_ratio_bounds={0: (.1, 1.0)})  # type: ignore[call-arg]
    with pytest.raises(TrustFilterError, match="not sealed"):
        TrustFilterConfig(aspect_ratio_bounds_artifact=object())  # type: ignore[arg-type]

    path = tmp_path / "bounds.json"; path.write_text(json.dumps(_bounds_payload()), encoding="utf-8")
    artifact = load_aspect_ratio_bounds_artifact(path)
    audit = artifact.audit_mapping()
    with pytest.raises(TypeError):
        audit["canonical_classes"][0]["name"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        audit["bounds"]["0"] = (.2, .8)  # type: ignore[index]


def test_result_audit_and_provenance_are_deeply_immutable(tmp_path: Path) -> None:
    path = tmp_path / "bounds.json"; path.write_text(json.dumps(_bounds_payload()), encoding="utf-8")
    artifact = load_aspect_ratio_bounds_artifact(path)
    result = _filter(aspect_ratio_bounds_artifact=artifact).filter(
        "teacher-20", (_candidate(view="original"), _candidate(view="horizontal_flip")), _geometry(),
    )
    with pytest.raises(TypeError):
        result.audit[0]["decision"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.audit[0]["raw_xyxy"][0] = 0.0  # type: ignore[index]
    assert result.filter_provenance is not None
    with pytest.raises(TypeError):
        result.filter_provenance["aspect_ratio_bounds"]["provenance"]["sealed"] = False  # type: ignore[index]


@pytest.mark.parametrize("mutation", [
    lambda payload: payload.__setitem__("labels", []),
    lambda payload: payload["provenance"].__setitem__("contains_human_labels", True),  # type: ignore[index]
    lambda payload: payload.__setitem__("classes", list(reversed(payload["classes"]))),  # type: ignore[index]
])
def test_bounds_loader_rejects_label_bearing_or_noncanonical_artifacts(tmp_path: Path, mutation: object) -> None:
    payload = _bounds_payload()
    mutation(payload)  # type: ignore[operator]
    path = tmp_path / "bad-bounds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrustFilterError, match="label-bearing|unsafe|taxonomy"):
        load_aspect_ratio_bounds_artifact(path)


def test_loader_accepts_only_task13_envelope(tmp_path: Path) -> None:
    candidate = _candidate(view="original").mapping()
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"manifest_version": "1.0", "teacher_run_id": "teacher-20", "candidate_count": 1, "candidates": [candidate]}), encoding="utf-8")
    run, candidates = load_candidate_envelope(path)
    assert run == "teacher-20" and candidates == (_candidate(view="original"),)
    path.write_text(json.dumps({"records": []}), encoding="utf-8")
    with pytest.raises(TrustFilterError, match="Task 13"):
        load_candidate_envelope(path)
