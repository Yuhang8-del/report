"""Tests for content and perceptual duplicate detection split safeguards."""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest
from PIL import Image

import fruit_ssod.data.deduplication as deduplication
from fruit_ssod.data.deduplication import DeduplicationError, deduplicate_records
from fruit_ssod.data.schema import CanonicalAnnotation, LicenseMetadata


def _record(image_id: str, filename: str, split: str) -> dict[str, str]:
    return {"source_image_id": image_id, "file_path": filename, "split": split}


def test_deduplication_finds_exact_near_and_cross_split_groups_without_false_exact_matches(tmp_path: Path) -> None:
    """Byte-identical and visually-close images form separate deterministic groups."""
    exact_a = tmp_path / "exact-a.png"
    exact_b = tmp_path / "exact-b.png"
    near = tmp_path / "near.png"
    unrelated = tmp_path / "unrelated.png"
    base = Image.new("L", (32, 32), 0)
    base.putpixel((4, 4), 255)
    base.save(exact_a)
    exact_b.write_bytes(exact_a.read_bytes())
    changed = base.copy()
    changed.putpixel((5, 4), 255)
    changed.save(near)
    Image.new("L", (32, 32), 255).save(unrelated)

    result = deduplicate_records(
        [
            _record("a", "exact-a.png", "train_pool"),
            _record("b", "exact-b.png", "test"),
            _record("c", "near.png", "validation"),
            _record("d", "unrelated.png", "train_pool"),
        ],
        image_root=tmp_path,
        near_hash_threshold=8,
    )

    assert [[item.source_image_id for item in group.members] for group in result.exact_groups] == [["a", "b"]]
    assert [[item.source_image_id for item in group.members] for group in result.near_groups] == [["a", "b", "c"]]
    assert result.resolutions[0].retained_split == "test"
    assert result.resolutions[0].excluded_image_ids == ("a", "c")
    assert result.rejected == ()
    assert all(fingerprint.sha256 != result.fingerprints[-1].sha256 for fingerprint in result.fingerprints[:-1])


def test_deduplication_reports_missing_files_and_rejects_unknown_split(tmp_path: Path) -> None:
    """Invalid inputs identify the affected image instead of silently skipping it."""
    with pytest.raises(DeduplicationError, match="Problem:") as error:
        deduplicate_records([_record("unknown", "missing.png", "development")], image_root=tmp_path)
    assert "Remediation:" in str(error.value)

    result = deduplicate_records([_record("missing", "missing.png", "train_pool")], image_root=tmp_path)
    assert result.fingerprints == ()
    assert result.rejected[0].reason_code == "IMAGE_MISSING"


def test_multiple_object_rows_for_one_image_are_not_self_duplicates(tmp_path: Path) -> None:
    """Canonical object annotations share one image fingerprint and never self-resolve."""
    image = tmp_path / "multi-object.png"
    Image.new("RGB", (16, 16), "red").save(image)
    common = {
        "source": "open_images_v7",
        "source_category": "Apple",
        "source_image_id": "one-image",
        "file_path": "multi-object.png",
        "width": 16,
        "height": 16,
        "class_id": 0,
        "split": "train_pool",
        "label_status": "labeled",
        "license_metadata": LicenseMetadata(name="fixture"),
    }

    result = deduplicate_records(
        [
            CanonicalAnnotation(xyxy=(1.0, 1.0, 4.0, 4.0), **common),
            CanonicalAnnotation(xyxy=(8.0, 8.0, 12.0, 12.0), **common),
        ],
        image_root=tmp_path,
    )

    assert len(result.fingerprints) == 1
    assert result.exact_groups == ()
    assert result.near_groups == ()
    assert result.resolutions == ()
    assert len(result.record_to_image) == 2
    assert {mapping.image_key for mapping in result.record_to_image} == {"one-image\tmulti-object.png"}


def test_deduplication_quarantines_a_decompression_bomb_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow's safety exception is recorded rather than escaping as a raw exception."""
    image = tmp_path / "safe-placeholder.png"
    image.write_bytes(b"placeholder")

    def bomb(_: Path) -> object:
        raise Image.DecompressionBombError("synthetic bomb")

    monkeypatch.setattr(deduplication.Image, "open", bomb)
    result = deduplicate_records([_record("bomb", "safe-placeholder.png", "train_pool")], image_root=tmp_path)

    assert result.rejected[0].reason_code == "IMAGE_UNDECODABLE"
    assert "synthetic bomb" in result.rejected[0].details["likely_cause"]


def test_deduplication_quarantines_a_decompression_bomb_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow's warning-only safety signal is promoted while producing perceptual hashes."""
    image = tmp_path / "safe-placeholder.png"
    image.write_bytes(b"placeholder")

    def warning(_: Path) -> object:
        warnings.warn("synthetic bomb warning", Image.DecompressionBombWarning)
        raise AssertionError("warning should have been promoted to an exception")

    monkeypatch.setattr(deduplication.Image, "open", warning)
    result = deduplicate_records([_record("bomb-warning", "safe-placeholder.png", "train_pool")], image_root=tmp_path)

    assert result.rejected[0].reason_code == "IMAGE_UNDECODABLE"
    assert "synthetic bomb warning" in result.rejected[0].details["likely_cause"]


def test_deduplication_quarantines_a_hash_read_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source disappearing after decode becomes an actionable quarantine finding."""
    image = tmp_path / "valid.png"
    Image.new("RGB", (16, 16), "red").save(image)

    def hash_race(_: Path) -> str:
        raise OSError("synthetic hash read race")

    monkeypatch.setattr(deduplication, "_sha256", hash_race)
    result = deduplicate_records([_record("race", "valid.png", "train_pool")], image_root=tmp_path)

    assert result.rejected[0].reason_code == "IMAGE_UNDECODABLE"
    assert "synthetic hash read race" in result.rejected[0].details["likely_cause"]
