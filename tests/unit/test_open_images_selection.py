from __future__ import annotations

import csv
import json
from pathlib import Path

from fruit_ssod.data.open_images_selection import OFFICIAL_BUCKET_BASE, build_open_images_selection


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_validation_selection_emits_matching_official_bucket_urls(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    annotations = tmp_path / "annotations.csv"
    metadata = tmp_path / "metadata.csv"
    names = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]
    _write_csv(classes, ["LabelName", "DisplayName"], [{"LabelName": f"/fruit/{index}", "DisplayName": name} for index, name in enumerate(names)])
    annotation_fields = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
    _write_csv(annotations, annotation_fields, [{"ImageID": f"image-{index}", "Source": "xclick", "LabelName": f"/fruit/{index}", "Confidence": "1", "XMin": "0.1", "XMax": "0.9", "YMin": "0.1", "YMax": "0.9", "IsOccluded": "0", "IsTruncated": "0", "IsGroupOf": "0", "IsDepiction": "0", "IsInside": "0"} for index in range(5)])
    metadata_fields = ["ImageID", "OriginalURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL", "Thumbnail300KURL"]
    _write_csv(metadata, metadata_fields, [{"ImageID": f"image-{index}", "OriginalURL": "https://example.invalid/original.jpg", "License": "CC BY", "Author": "fixture", "AuthorProfileURL": "", "OriginalLandingURL": "", "Thumbnail300KURL": "https://example.invalid/thumb.jpg"} for index in range(5)])

    output = tmp_path / "selection"
    result = build_open_images_selection(annotations, classes, metadata, output, per_class=1, image_split="validation")

    assert len(result.image_ids) == 5
    urls = list(csv.DictReader((output / "image-urls.csv").open(encoding="utf-8", newline="")))
    assert {row["OriginalURL"] for row in urls} == {f"{OFFICIAL_BUCKET_BASE}/validation/image-{index}.jpg" for index in range(5)}
    source = json.loads((output / "source-metadata.json").read_text(encoding="utf-8"))
    assert source["source_image_split"] == "validation"


def test_selection_excludes_previously_used_source_image_ids(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    annotations = tmp_path / "annotations.csv"
    metadata = tmp_path / "metadata.csv"
    names = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]
    _write_csv(classes, ["LabelName", "DisplayName"], [{"LabelName": f"/fruit/{index}", "DisplayName": name} for index, name in enumerate(names)])
    fields = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
    rows = []
    for index in range(5):
        for suffix in ("old", "new"):
            rows.append({"ImageID": f"{suffix}-{index}", "Source": "xclick", "LabelName": f"/fruit/{index}", "Confidence": "1", "XMin": "0.1", "XMax": "0.9", "YMin": "0.1", "YMax": "0.9", "IsOccluded": "0", "IsTruncated": "0", "IsGroupOf": "0", "IsDepiction": "0", "IsInside": "0"})
    _write_csv(annotations, fields, rows)
    metadata_fields = ["ImageID", "OriginalURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL", "Thumbnail300KURL"]
    _write_csv(metadata, metadata_fields, [{"ImageID": f"{suffix}-{index}", "OriginalURL": "https://example.invalid/original.jpg", "License": "CC BY", "Author": "fixture", "AuthorProfileURL": "", "OriginalLandingURL": "", "Thumbnail300KURL": "https://example.invalid/thumb.jpg"} for index in range(5) for suffix in ("old", "new")])

    result = build_open_images_selection(annotations, classes, metadata, tmp_path / "selection", per_class=1, excluded_image_ids=[f"old-{index}" for index in range(5)])

    assert result.image_ids == tuple(f"new-{index}" for index in range(5))
    source = json.loads((tmp_path / "selection" / "source-metadata.json").read_text(encoding="utf-8"))
    assert source["excluded_source_image_count"] == 5


def test_selection_supports_explicit_imbalanced_class_caps(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"; annotations = tmp_path / "annotations.csv"; metadata = tmp_path / "metadata.csv"
    names = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]
    _write_csv(classes, ["LabelName", "DisplayName"], [{"LabelName": f"/fruit/{index}", "DisplayName": name} for index, name in enumerate(names)])
    fields = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
    rows = [{"ImageID": f"image-{class_id}-{copy}", "Source": "xclick", "LabelName": f"/fruit/{class_id}", "Confidence": "1", "XMin": "0.1", "XMax": "0.9", "YMin": "0.1", "YMax": "0.9", "IsOccluded": "0", "IsTruncated": "0", "IsGroupOf": "0", "IsDepiction": "0", "IsInside": "0"} for class_id in range(5) for copy in range(2)]
    _write_csv(annotations, fields, rows)
    metadata_fields = ["ImageID", "OriginalURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL", "Thumbnail300KURL"]
    _write_csv(metadata, metadata_fields, [{"ImageID": row["ImageID"], "OriginalURL": "https://example.invalid/original.jpg", "License": "CC BY", "Author": "fixture", "AuthorProfileURL": "", "OriginalLandingURL": "", "Thumbnail300KURL": "https://example.invalid/thumb.jpg"} for row in rows])
    caps = {"Apple": 2, "Banana": 1, "Orange": 1, "Strawberry": 1, "Pineapple": 1}
    result = build_open_images_selection(annotations, classes, metadata, tmp_path / "selection", per_class=caps)
    assert result.class_image_counts == caps
    source = json.loads((tmp_path / "selection" / "source-metadata.json").read_text(encoding="utf-8"))
    assert source["per_class_requested"] == caps
