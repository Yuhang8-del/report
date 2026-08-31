"""End-to-end no-label dual-view candidate generation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from fruit_ssod.data.schema import LicenseMetadata, UnlabeledImageRecord
from fruit_ssod.detection.adapter import DetectorAdapter
from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.pseudo.generator import (
    canonical_unlabeled_fingerprint,
    PseudoGenerationError,
    PseudoGenerationResult,
    PseudoLabelGenerator,
    SealedUnlabeledMembership,
    load_unlabeled_manifest,
    write_pseudo_candidates,
)
from fruit_ssod.pseudo.candidates import PseudoCandidate


def _record(path: Path) -> UnlabeledImageRecord:
    return UnlabeledImageRecord(source="fruit_360", source_image_id="unlabeled-1", file_path=path.name, width=100, height=60, split="train_pool", label_status="unlabeled", license_metadata=LicenseMetadata(name="CC BY"))


def _detection(box: tuple[float, float, float, float]) -> DetectionRecord:
    return DetectionRecord(class_id=0, class_name="Apple", confidence=0.91, xyxy=box, is_unknown=False, source_model="teacher-best.pt")


def _sealed(record: UnlabeledImageRecord, *, protected: frozenset[str] = frozenset()) -> SealedUnlabeledMembership:
    records = (record,)
    return SealedUnlabeledMembership(records, protected, canonical_unlabeled_fingerprint(records))


def _mapping(record: UnlabeledImageRecord) -> dict[str, Any]:
    return {
        "source": record.source, "source_image_id": record.source_image_id,
        "file_path": record.file_path, "width": record.width, "height": record.height,
        "split": record.split, "label_status": record.label_status,
        "license_metadata": {"name": record.license_metadata.name},
    }


def _write_paired_task8_manifests(root: Path, record: UnlabeledImageRecord, *, unlabeled_ids: list[str] | None = None, protected: dict[str, list[str]] | None = None) -> Path:
    manifest = root / "unlabeled.json"
    manifest.write_text(json.dumps({"records": [_mapping(record)]}), encoding="utf-8")
    split = {
        "unlabeled_image_ids": [record.source_image_id] if unlabeled_ids is None else unlabeled_ids,
        "split_image_ids": protected or {"validation": [], "test": [], "pseudo_audit": [], "external_test": []},
        "fingerprints": {"unlabeled": canonical_unlabeled_fingerprint((record,))},
    }
    (root / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    return manifest


class _DualViewFakeDetector(DetectorAdapter):
    def __init__(self) -> None:
        self.images: list[Any] = []

    def predict(self, image: object, *, confidence: float | None = None) -> tuple[DetectionRecord, ...]:
        self.images.append(image)
        # The original red pixel starts at the left edge; after PIL mirroring it
        # appears at the right edge.  This proves the second call receives an
        # actual transformed view, not the original twice.
        left = image.getpixel((0, 0))  # type: ignore[union-attr]
        return (_detection((10.0, 5.0, 30.0, 20.0) if left[0] else (70.0, 5.0, 90.0, 20.0)),)


def test_generator_runs_original_and_horizontal_flip_and_preserves_raw_provenance(tmp_path: Path) -> None:
    image_path = tmp_path / "unlabeled.png"
    image = Image.new("RGB", (100, 60), color=(0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(image_path)
    detector = _DualViewFakeDetector()

    result = PseudoLabelGenerator(detector, teacher_run_id="teacher-20-seed42", image_root=tmp_path).generate(_sealed(_record(image_path)))

    assert len(detector.images) == 2
    assert [item.view for item in result.candidates] == ["original", "horizontal_flip"]
    assert result.candidates[0].raw_xyxy == (10.0, 5.0, 30.0, 20.0)
    assert result.candidates[0].xyxy == (10.0, 5.0, 30.0, 20.0)
    assert result.candidates[1].raw_xyxy == (70.0, 5.0, 90.0, 20.0)
    assert result.candidates[1].xyxy == (10.0, 5.0, 30.0, 20.0)
    assert all(item.teacher_run_id == "teacher-20-seed42" for item in result.candidates)
    assert all(item.source_image_id == "unlabeled-1" and item.confidence == 0.91 for item in result.candidates)

    destination = write_pseudo_candidates(result, tmp_path / "candidates.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 2
    assert payload["candidates"][1]["raw_xyxy"] == [70.0, 5.0, 90.0, 20.0]
    with pytest.raises(PseudoGenerationError, match="already exists"):
        write_pseudo_candidates(result, destination)


@pytest.mark.parametrize("forbidden", ["labels", "annotation_path", "human_label", "xyxy", "class_id"])
def test_manifest_loader_rejects_every_label_bearing_input_field(tmp_path: Path, forbidden: str) -> None:
    row: dict[str, Any] = {
        "source": "fruit_360", "source_image_id": "unlabeled-1", "file_path": "unlabeled.png",
        "width": 100, "height": 60, "split": "train_pool", "label_status": "unlabeled",
        "license_metadata": {"name": "CC BY"},
    }
    row[forbidden] = []
    manifest = tmp_path / "unlabeled.json"
    manifest.write_text(json.dumps({"records": [row]}), encoding="utf-8")

    with pytest.raises(PseudoGenerationError, match="label-bearing"):
        load_unlabeled_manifest(manifest)


def test_generator_refuses_annotation_objects_even_if_their_image_fields_are_valid(tmp_path: Path) -> None:
    class _UnusedDetector(DetectorAdapter):
        def predict(self, image: object, *, confidence: float | None = None) -> tuple[DetectionRecord, ...]:
            return ()

    with pytest.raises(PseudoGenerationError, match="sealed unlabeled membership"):
        PseudoLabelGenerator(_UnusedDetector(), teacher_run_id="teacher", image_root=tmp_path).generate([{"source_image_id": "x"}])  # type: ignore[arg-type]


def test_loader_requires_exact_task8_membership_and_rejects_protected_ids(tmp_path: Path) -> None:
    record = _record(tmp_path / "unlabeled.png")
    manifest = _write_paired_task8_manifests(tmp_path, record, unlabeled_ids=["other"])
    with pytest.raises(PseudoGenerationError, match="membership differs"):
        load_unlabeled_manifest(manifest)

    manifest = _write_paired_task8_manifests(
        tmp_path, record,
        protected={"validation": [record.source_image_id], "test": [], "pseudo_audit": [], "external_test": []},
    )
    with pytest.raises(PseudoGenerationError, match="protected image IDs"):
        load_unlabeled_manifest(manifest)


def test_loader_returns_sealed_membership_from_paired_task8_manifests(tmp_path: Path) -> None:
    record = _record(tmp_path / "unlabeled.png")
    manifest = _write_paired_task8_manifests(tmp_path, record)
    membership = load_unlabeled_manifest(manifest)
    assert isinstance(membership, SealedUnlabeledMembership)
    assert membership.records == (record,)
    assert membership.protected_image_ids == frozenset()
    assert membership.unlabeled_fingerprint == canonical_unlabeled_fingerprint((record,))


@pytest.mark.parametrize("fingerprint", [None, "A" * 64, "not-a-hash"])
def test_loader_rejects_missing_or_noncanonical_task8_unlabeled_fingerprint(tmp_path: Path, fingerprint: object) -> None:
    record = _record(tmp_path / "unlabeled.png")
    manifest = _write_paired_task8_manifests(tmp_path, record)
    split = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    if fingerprint is None:
        split.pop("fingerprints")
    else:
        split["fingerprints"]["unlabeled"] = fingerprint
    (tmp_path / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(PseudoGenerationError, match="fingerprint"):
        load_unlabeled_manifest(manifest)


@pytest.mark.parametrize("field,value", [("file_path", "other.png"), ("width", 99), ("height", 59), ("source", "other_source")])
def test_loader_rejects_any_task8_fingerprint_normalized_field_mutation(tmp_path: Path, field: str, value: object) -> None:
    record = _record(tmp_path / "unlabeled.png")
    manifest = _write_paired_task8_manifests(tmp_path, record)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0][field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PseudoGenerationError, match="fingerprint"):
        load_unlabeled_manifest(manifest)


@pytest.mark.parametrize("file_path", ["../outside.png", "/absolute/outside.png"])
def test_generator_rejects_unsealed_image_paths(tmp_path: Path, file_path: str) -> None:
    record = UnlabeledImageRecord("fruit_360", "unlabeled-1", file_path, 100, 60, "train_pool", "unlabeled", LicenseMetadata(name="CC BY"))

    class _UnusedDetector(DetectorAdapter):
        def predict(self, image: object, *, confidence: float | None = None) -> tuple[DetectionRecord, ...]:
            return ()

    with pytest.raises(PseudoGenerationError, match="absolute|traverses"):
        PseudoLabelGenerator(_UnusedDetector(), teacher_run_id="teacher", image_root=tmp_path).generate(_sealed(record))


def test_generator_rejects_a_legal_relative_path_redirected_to_protected_content_inside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A symlink can look legal after lexical path validation but is not authority."""
    record = _record(tmp_path / "unlabeled.png")

    class _UnusedDetector(DetectorAdapter):
        def predict(self, image: object, *, confidence: float | None = None) -> tuple[DetectionRecord, ...]:
            return ()

    link_path = (tmp_path / record.file_path).resolve(strict=False)
    original_is_symlink = Path.is_symlink

    def redirected(path: Path) -> bool:
        return path.resolve(strict=False) == link_path or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", redirected)
    with pytest.raises(PseudoGenerationError, match="symbolic link"):
        PseudoLabelGenerator(_UnusedDetector(), teacher_run_id="teacher", image_root=tmp_path).generate(_sealed(record))


def test_generator_requires_exact_original_and_flipped_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "unlabeled.png"
    Image.new("RGB", (100, 60)).save(image_path)
    record = _record(image_path)

    class _UnusedDetector(DetectorAdapter):
        def predict(self, image: object, *, confidence: float | None = None) -> tuple[DetectionRecord, ...]:
            return ()

    generator = PseudoLabelGenerator(_UnusedDetector(), teacher_run_id="teacher", image_root=tmp_path)
    with pytest.raises(PseudoGenerationError, match="no valid dimensions"):
        PseudoLabelGenerator(_UnusedDetector(), teacher_run_id="teacher", image_root=tmp_path, image_loader=lambda _: object()).generate(_sealed(record))
    monkeypatch.setattr("fruit_ssod.pseudo.generator.horizontal_flip_image", lambda _: Image.new("RGB", (99, 60)))
    with pytest.raises(PseudoGenerationError, match="horizontal-flip image dimensions"):
        generator.generate(_sealed(record))
    monkeypatch.setattr("fruit_ssod.pseudo.generator.horizontal_flip_image", lambda _: object())
    with pytest.raises(PseudoGenerationError, match="no valid dimensions"):
        generator.generate(_sealed(record))


def test_generation_result_and_writer_reject_mixed_or_unvalidated_candidates(tmp_path: Path) -> None:
    candidate = PseudoCandidate.from_detection(_detection((1.0, 1.0, 2.0, 2.0)), teacher_run_id="teacher-a", source_image_id="image-a", source_file_path="image-a.png", view="original")
    with pytest.raises(PseudoGenerationError, match="mixes teacher run IDs"):
        PseudoGenerationResult("teacher-b", (candidate,))
    with pytest.raises(PseudoGenerationError, match="invalid candidate"):
        PseudoGenerationResult("teacher-a", (object(),))  # type: ignore[arg-type]
    with pytest.raises(PseudoGenerationError, match="not a validated generation result"):
        write_pseudo_candidates({"candidates": []}, tmp_path / "bad.json")  # type: ignore[arg-type]
