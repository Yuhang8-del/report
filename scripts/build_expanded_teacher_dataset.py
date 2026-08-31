"""Build a larger leakage-safe supervised pool for Teacher refinement.

The current v2 split is deliberately small.  This utility combines its
labelled training images with the older full-label training pool, removes any
image stem appearing in the protected v2 validation/test sets, and de-
duplicates by image stem.  It writes absolute-path list files so Ultralytics
can train without copying the source images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CLASS_IDS = frozenset(range(5))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_list(path: Path) -> list[Path]:
    rows = [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty image list: {path}")
    missing = [str(item) for item in rows if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"missing image(s) in {path}: {missing[:3]}")
    return rows


def _label_path(image: Path) -> Path:
    # .../<dataset>/images/<split>/<stem>.jpg -> .../<dataset>/labels/<split>/<stem>.txt
    if len(image.parents) < 3 or image.parent.parent.name != "images":
        raise ValueError(f"cannot infer YOLO label path from {image}")
    return image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"


def _validate_label(image: Path) -> None:
    label = _label_path(image)
    if not label.is_file():
        raise FileNotFoundError(f"missing label for {image}: {label}")
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise ValueError(f"invalid class id in {label}: {fields[0]!r}") from error
        if class_id not in CLASS_IDS:
            raise ValueError(f"non-canonical class id {class_id} in {label}")


def _write_lines(path: Path, rows: list[Path]) -> None:
    path.write_text("\n".join(str(row) for row in rows) + "\n", encoding="utf-8")


def build(*, v2_root: Path, v12_train: Path, output_root: Path, split_manifest: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    v2_train_rows = _read_list(v2_root / "train.txt")
    v2_val_rows = _read_list(v2_root / "val.txt")
    v2_test_rows = _read_list(v2_root / "test.txt")
    v12_rows = _read_list(v12_train)

    protected_stems = {row.stem for row in (*v2_val_rows, *v2_test_rows)}
    selected: list[Path] = []
    seen_stems: set[str] = set()
    excluded_protected: list[Path] = []
    excluded_duplicate: list[Path] = []
    source_counts = {"v2_train": len(v2_train_rows), "v12_train_list": len(v12_rows)}

    # Keep current v2 training membership first, then add the older pool.
    for row in (*v2_train_rows, *v12_rows):
        if row.stem in protected_stems:
            excluded_protected.append(row)
            continue
        if row.stem in seen_stems:
            excluded_duplicate.append(row)
            continue
        _validate_label(row)
        seen_stems.add(row.stem)
        selected.append(row)

    selected.sort(key=lambda item: (item.stem, str(item)))
    _write_lines(output_root / "train.txt", selected)
    _write_lines(output_root / "val.txt", v2_val_rows)
    _write_lines(output_root / "test.txt", v2_test_rows)
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "path: " + str(output_root).replace("\\", "/") + "\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        "- Apple\n- Banana\n- Orange\n- Strawberry\n- Pineapple\n",
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_type": "expanded_teacher_training_pool",
        "protocol": "v2_fixed_test_with_v12_train_pool_extension",
        "output_root": str(output_root),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "source_counts": source_counts,
        "selected_unique_images": len(selected),
        "v2_validation_images": len(v2_val_rows),
        "v2_test_images": len(v2_test_rows),
        "excluded_protected_members": len(excluded_protected),
        "excluded_duplicate_members": len(excluded_duplicate),
        "dataset_yaml_sha256": _sha256(dataset_yaml),
        "train_list_sha256": _sha256(output_root / "train.txt"),
        "validation_list_sha256": _sha256(output_root / "val.txt"),
        "test_list_sha256": _sha256(output_root / "test.txt"),
        "protected_stem_count": len(protected_stems),
    }
    (output_root / "expansion_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v12-train", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        v2_root=args.v2_root.resolve(),
        v12_train=args.v12_train.resolve(),
        output_root=args.output_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
