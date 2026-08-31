"""Tests for local-only auxiliary Fruit-360 and FruitDet importers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fruit_ssod.cli.import_auxiliary_data import main
from fruit_ssod.data.class_mapping import ClassMappingError, resolve_class_id
from fruit_ssod.data.fruitdet import FruitDetImportError, import_fruitdet
import fruit_ssod.data.fruits360 as fruits360_module
from fruit_ssod.data.fruits360 import _normalized_sort_key, import_fruits360
from fruit_ssod.data.schema import LicenseMetadata


FIXTURES = Path(__file__).parents[1] / "fixtures" / "auxiliary"
METADATA = {
    "source_version": "fixture-1",
    "source_page": "https://example.invalid/source-page",
    "license_metadata": LicenseMetadata(name="Fixture license", url="https://example.invalid/license"),
}


def test_auxiliary_sources_resolve_only_their_reviewed_aliases() -> None:
    """Fruit-360 permits all canonical classes; FruitDet's source omits Pineapple."""
    assert resolve_class_id("fruit_360", "Pineapple") == 4
    assert resolve_class_id("limited_external_set", "Apple") == 0
    with pytest.raises(ClassMappingError, match="Unknown category.*Pineapple"):
        resolve_class_id("limited_external_set", "Pineapple")


def test_fruits360_reviewed_official_directory_variants_are_curation_only_aliases() -> None:
    """Reviewed 100x100 directory names validate without becoming labels."""
    assert [
        resolve_class_id("fruit_360", category)
        for category in ("Apple 6", "Banana 1", "Orange 1", "Strawberry 1", "Pineapple 1")
    ] == [0, 1, 2, 3, 4]


def test_fruits360_scans_deterministically_without_fabricating_boxes_or_labels() -> None:
    """Directory category names are retained in the manifest, not turned into detections."""
    result = import_fruits360(FIXTURES / "fruits360", **METADATA)

    assert [record.source_image_id for record in result.records] == [
        "Apple/apple.svg",
        "Pineapple/pineapple.svg",
    ]
    assert all(record.split == "train_pool" and record.label_status == "unlabeled" for record in result.records)
    assert [entry["source_category"] for entry in result.manifest["records"]] == ["Apple", "Pineapple"]
    assert result.manifest["record_count"] == 2
    assert result.manifest["rejection_count"] == 1
    assert "MysteryFruit" in result.rejections[0]["reason"]


def test_fruits360_sort_key_breaks_case_only_ties_deterministically() -> None:
    """Case-fold-equivalent category names and paths retain a stable exact-name tie-breaker."""
    assert sorted(["apple.svg", "Apple.svg"], key=_normalized_sort_key) == [
        "Apple.svg",
        "apple.svg",
    ]


def test_fruits360_importer_uses_stable_sorting_for_reversed_categories_and_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importer ordering stays identical when local directory traversal arrives reversed."""
    images_root = tmp_path / "images"
    for category in ("Apple", "Pineapple", "MysteryFruit", "Zucchini"):
        for image_name in ("alpha.svg", "beta.svg"):
            image_path = images_root / category / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="20"></svg>',
                encoding="utf-8",
            )

    original_iterdir = Path.iterdir
    original_rglob = Path.rglob
    original_key = fruits360_module._normalized_sort_key
    observed_sort_values: list[str] = []

    def reversed_iterdir(path: Path):  # type: ignore[no-untyped-def]
        return iter(reversed(list(original_iterdir(path))))

    def reversed_rglob(path: Path, pattern: str):  # type: ignore[no-untyped-def]
        return iter(reversed(list(original_rglob(path, pattern))))

    def recording_key(value: str) -> tuple[str, str]:
        observed_sort_values.append(value)
        return original_key(value)

    monkeypatch.setattr(Path, "iterdir", reversed_iterdir)
    monkeypatch.setattr(Path, "rglob", reversed_rglob)
    monkeypatch.setattr(fruits360_module, "_normalized_sort_key", recording_key)

    first = import_fruits360(images_root, **METADATA)
    second = import_fruits360(images_root, **METADATA)

    expected_ids = [
        "Apple/alpha.svg",
        "Apple/beta.svg",
        "Pineapple/alpha.svg",
        "Pineapple/beta.svg",
    ]
    assert [record.source_image_id for record in first.records] == expected_ids
    assert [record.source_image_id for record in second.records] == expected_ids
    assert first.rejections == second.rejections
    assert [rejection["source_category"] for rejection in first.rejections] == [
        "MysteryFruit",
        "Zucchini",
    ]
    assert all("Unknown category" in rejection["reason"] for rejection in first.rejections)
    assert first.manifest == second.manifest
    assert json.dumps(first.manifest, sort_keys=True) == json.dumps(second.manifest, sort_keys=True)
    assert first.manifest["rejections"] == list(first.rejections)
    assert [rejection["source_category"] for rejection in first.manifest["rejections"]] == [
        "MysteryFruit",
        "Zucchini",
    ]
    assert {"Apple", "Pineapple", *expected_ids}.issubset(observed_sort_values)


def test_fruitdet_rejects_absent_pineapple_and_stays_external_test() -> None:
    """FruitDet annotations never enter a primary training or validation partition."""
    result = import_fruitdet(
        FIXTURES / "fruitdet" / "annotations.json",
        FIXTURES / "fruitdet" / "images",
        **METADATA,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert (record.source, record.source_category, record.class_id) == ("limited_external_set", "Apple", 0)
    assert record.xyxy == (10.0, 5.0, 50.0, 25.0)
    assert (record.split, record.label_status) == ("external_test", "labeled")
    assert result.manifest["split"] == "external_test"
    assert result.manifest["category_mapping_source"] == "limited_external_set"
    assert result.manifest["mapped_class_ids"] == [0, 1, 2, 3]
    assert result.manifest["mapped_class_names"] == ["Apple", "Banana", "Orange", "Strawberry"]
    assert result.manifest["record_count"] == 1
    assert result.manifest["rejection_count"] == 1
    assert result.rejections[0]["source_category"] == "Pineapple"


def test_fruitdet_rejects_a_coco_image_path_that_is_missing_locally() -> None:
    """External-test records cannot point at a missing local source image."""
    with pytest.raises(FruitDetImportError, match="not an existing file") as error:
        import_fruitdet(
            FIXTURES / "fruitdet" / "annotations_missing_image.json",
            FIXTURES / "fruitdet" / "images",
            **METADATA,
        )

    assert "Likely cause:" in str(error.value)
    assert "Remediation:" in str(error.value)


def test_fruitdet_cli_refuses_primary_split_and_writes_only_requested_manifest(tmp_path: Path) -> None:
    """The command cannot accidentally make FruitDet part of a primary split."""
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "fruitdet",
                "--annotations",
                str(FIXTURES / "fruitdet" / "annotations.json"),
                "--images-root",
                str(FIXTURES / "fruitdet" / "images"),
                "--output-dir",
                str(tmp_path / "bad"),
                "--source-version",
                "fixture-1",
                "--source-page",
                "https://example.invalid/source-page",
                "--license-name",
                "Fixture license",
                "--split",
                "train_pool",
            ]
        )

    output_dir = tmp_path / "manifest"
    assert main(
        [
            "fruitdet",
            "--annotations",
            str(FIXTURES / "fruitdet" / "annotations.json"),
            "--images-root",
            str(FIXTURES / "fruitdet" / "images"),
            "--output-dir",
            str(output_dir),
            "--source-version",
            "fixture-1",
            "--source-page",
            "https://example.invalid/source-page",
            "--license-name",
            "Fixture license",
        ]
    ) == 0
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["source"]["name"] == "fruitdet"
    assert payload["split"] == "external_test"
    assert sorted(path.name for path in output_dir.iterdir()) == ["manifest.json"]
