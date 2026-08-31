"""Tests for canonical manifest generation from downloaded Open Images data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from fruit_ssod.data.open_images_manifest import build_canonical_open_images_manifest


def test_build_canonical_manifest_preserves_downloaded_provenance(tmp_path: Path) -> None:
    root = tmp_path / "converted"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (100, 50), "red").save(root / "images" / "apple.jpg")
    (root / "labels" / "apple.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (root / "manifest.jsonl").write_text(json.dumps({"source": "open_images_v7", "source_image_id": "apple", "url": "https://official.example/apple.jpg", "class_ids": [0]}) + "\n", encoding="utf-8")
    urls = tmp_path / "selection.csv"
    with urls.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "OriginalURL", "License", "Author", "AuthorProfileURL", "OriginalLandingURL"])
        writer.writeheader()
        writer.writerow({"ImageID": "apple", "OriginalURL": "https://official.example/apple.jpg", "License": "https://creativecommons.org/licenses/by/2.0/", "Author": "Example Author", "AuthorProfileURL": "", "OriginalLandingURL": "https://example.invalid/landing"})

    output = tmp_path / "interim" / "canonical.json"
    result = build_canonical_open_images_manifest(root, urls, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.image_count == 1 and result.annotation_count == 1
    assert payload["records"][0]["file_path"] == "images/apple.jpg"
    assert payload["records"][0]["xyxy"] == [30.0, 15.0, 70.0, 35.0]
    assert payload["records"][0]["license_metadata"]["attribution"] == "Example Author"
