from __future__ import annotations

import json
from pathlib import Path

import yaml

from fruit_ssod.data.balanced_training import materialize_balanced_training_view


def _write_snapshot(root: Path) -> None:
    members: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    classes = {
        "apple-a": 0,
        "apple-b": 0,
        "banana-a": 1,
        "orange-a": 2,
        "strawberry-a": 3,
        "pineapple-a": 4,
    }
    for partition, image_ids in {
        "train": tuple(classes),
        "val": ("val-a",),
        "test": ("test-a",),
    }.items():
        (root / "images" / partition).mkdir(parents=True, exist_ok=True)
        (root / "labels" / partition).mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for image_id in image_ids:
            image = root / "images" / partition / f"{image_id}.jpg"
            label = root / "labels" / partition / f"{image_id}.txt"
            image.write_bytes(b"fixture")
            class_id = classes.get(image_id, 1)
            label.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            paths.append(str(image.resolve()))
            members[partition].append(
                {
                    "source_image_id": image_id,
                    "snapshot_image": image.relative_to(root).as_posix(),
                    "snapshot_label": label.relative_to(root).as_posix(),
                }
            )
        (root / f"{partition}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (root / "membership.json").write_text(
        json.dumps({"artifact_type": "sealed_full_label_supervised_upper_bound", "members": members}),
        encoding="utf-8",
    )


def test_balanced_training_view_preserves_unique_train_and_protected_splits(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_snapshot(snapshot)

    result = materialize_balanced_training_view(snapshot, tmp_path / "balanced", seed=42, max_appearances_per_image=3)

    evidence = json.loads(result.membership.read_text(encoding="utf-8"))
    lines = (result.root / "train_balanced.txt").read_text(encoding="utf-8").splitlines()
    assert evidence["base_train_image_count"] == 6
    assert evidence["balanced_training_exposure_count"] == len(lines)
    assert set(lines) == {str(path.resolve()) for path in (snapshot / "images" / "train").glob("*.jpg")}
    assert evidence["class_image_exposure_before"] == {"Apple": 2, "Banana": 1, "Orange": 1, "Strawberry": 1, "Pineapple": 1}
    assert evidence["class_image_exposure_after"]["Banana"] == 2
    assert evidence["class_image_exposure_after"]["Pineapple"] == 2
    assert evidence["class_image_exposure_after"]["Orange"] == 2
    assert evidence["class_image_exposure_after"]["Strawberry"] == 2
    dataset = yaml.safe_load(result.dataset_yaml.read_text(encoding="utf-8"))
    assert dataset["val"] == str((snapshot / "val.txt").resolve())
    assert dataset["test"] == str((snapshot / "test.txt").resolve())


def test_balanced_training_view_is_deterministic(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_snapshot(snapshot)

    first = materialize_balanced_training_view(snapshot, tmp_path / "first", seed=7, max_appearances_per_image=3)
    second = materialize_balanced_training_view(snapshot, tmp_path / "second", seed=7, max_appearances_per_image=3)

    first_evidence = json.loads(first.membership.read_text(encoding="utf-8"))
    second_evidence = json.loads(second.membership.read_text(encoding="utf-8"))
    for evidence in (first_evidence, second_evidence):
        evidence.pop("output_root")
    assert first_evidence == second_evidence
    assert (first.root / "train_balanced.txt").read_text(encoding="utf-8") == (second.root / "train_balanced.txt").read_text(encoding="utf-8")
