from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fruit_ssod.data.object_centric_tiles import materialize_object_centric_tiles


def _snapshot(root: Path) -> None:
    members: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for partition, image_id in (("train", "train-a"), ("val", "val-a"), ("test", "test-a")):
        image = root / "images" / partition / f"{image_id}.jpg"
        label = root / "labels" / partition / f"{image_id}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1000, 800), "green").save(image)
        # The train fixture is a 20x16-pixel small object. Protected fixtures
        # are deliberately valid but must never produce training tiles.
        label.write_text("4 0.75 0.25 0.02 0.02\n", encoding="utf-8")
        (root / f"{partition}.txt").write_text(str(image.resolve()) + "\n", encoding="utf-8")
        members[partition].append(
            {
                "source_image_id": image_id,
                "snapshot_image": image.relative_to(root).as_posix(),
                "snapshot_label": label.relative_to(root).as_posix(),
            }
        )
    (root / "membership.json").write_text(
        json.dumps({"artifact_type": "sealed_full_label_supervised_upper_bound", "members": members}),
        encoding="utf-8",
    )


def test_object_centric_tiles_keep_parent_split_and_valid_boxes(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot)

    result = materialize_object_centric_tiles(
        snapshot,
        tmp_path / "tiles",
        tile_size=400,
        small_object_area=0.01,
        minimum_visibility=0.5,
        max_tiles_per_image=2,
    )

    evidence = json.loads(result.membership.read_text(encoding="utf-8"))
    assert result.tile_count == 1
    assert evidence["tiles"][0]["parent_image_id"] == "train-a"
    tile_image = result.root / evidence["tiles"][0]["tile_image"]
    tile_label = result.root / evidence["tiles"][0]["tile_label"]
    assert Image.open(tile_image).size == (400, 400)
    row = tile_label.read_text(encoding="utf-8").split()
    assert row[0] == "4"
    assert all(0.0 <= float(value) <= 1.0 for value in row[1:])
    train_lines = (result.root / "train_with_tiles.txt").read_text(encoding="utf-8").splitlines()
    assert str((snapshot / "images" / "train" / "train-a.jpg").resolve()) in train_lines
    assert str(tile_image.resolve()) in train_lines
    assert not any("val-a" in line or "test-a" in line for line in train_lines)


def test_object_centric_tiles_can_retain_balanced_full_image_exposures(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot)
    balanced = tmp_path / "balanced.txt"
    source_image = str((snapshot / "images" / "train" / "train-a.jpg").resolve())
    balanced.write_text(f"{source_image}\n{source_image}\n", encoding="utf-8")

    result = materialize_object_centric_tiles(
        snapshot,
        tmp_path / "tiles",
        base_training_list=balanced,
        tile_size=400,
        max_tiles_per_image=1,
    )

    lines = (result.root / "train_with_tiles.txt").read_text(encoding="utf-8").splitlines()
    assert lines.count(source_image) == 2
    assert len(lines) == 3
