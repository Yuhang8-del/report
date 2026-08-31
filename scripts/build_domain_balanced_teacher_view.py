"""Build a deterministic domain-balanced Teacher training view.

The expanded Teacher pool contains the v2 fixed-domain training images only
once, while the older auxiliary pool contributes most exposures.  This script
keeps every original training member and adds reproducible extra exposures of
the v2 training list.  Validation and test lists are copied byte-for-byte from
the expanded pool, so no protected image is added to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty image list: {path}")
    return values


def _assert_disjoint(train: list[str], protected: list[str]) -> None:
    train_ids = {Path(value).stem for value in train}
    protected_ids = {Path(value).stem for value in protected}
    overlap = sorted(train_ids & protected_ids)
    if overlap:
        raise ValueError(f"training/protected overlap detected: {overlap[:5]}")


def build(base_root: Path, v2_root: Path, output_root: Path, *, repeats: int, seed: int) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    base_root = base_root.resolve(strict=True)
    v2_root = v2_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite published view: {output_root}")

    base_train = _lines(base_root / "train.txt")
    base_val = _lines(base_root / "val.txt")
    base_test = _lines(base_root / "test.txt")
    v2_train = _lines(v2_root / "train.txt")
    v2_val = _lines(v2_root / "val.txt")
    v2_test = _lines(v2_root / "test.txt")
    _assert_disjoint(base_train, base_val + base_test)
    _assert_disjoint(v2_train, v2_val + v2_test)
    v2_ids = {Path(value).stem for value in v2_train}
    base_v2 = [value for value in base_train if Path(value).stem in v2_ids]
    if {Path(value).stem for value in base_v2} != v2_ids:
        missing = sorted(v2_ids - {Path(value).stem for value in base_v2})
        raise ValueError(f"base expanded pool is missing v2 train members: {missing[:5]}")

    rng = random.Random(seed)
    extra: list[str] = []
    for appearance in range(repeats):
        block = list(base_v2)
        rng.shuffle(block)
        extra.extend(block)
    train_out = base_train + extra
    _assert_disjoint(train_out, v2_val + v2_test)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    train_path = output_root / "train_domain_balanced.txt"
    val_path = output_root / "val.txt"
    test_path = output_root / "test.txt"
    train_path.write_text("\n".join(train_out) + "\n", encoding="utf-8", newline="\n")
    val_path.write_text("\n".join(base_val) + "\n", encoding="utf-8", newline="\n")
    test_path.write_text("\n".join(base_test) + "\n", encoding="utf-8", newline="\n")
    dataset = {
        "path": str(output_root),
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "names": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"],
    }
    dataset_path = output_root / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    evidence = {
        "schema_version": "1.0",
        "artifact_type": "deterministic_domain_balanced_teacher_view",
        "algorithm": "v2-fixed-domain-repeat-v1",
        "seed": seed,
        "additional_v2_appearances": repeats,
        "base_training_root": str(base_root),
        "base_train_list": str((base_root / "train.txt").resolve()),
        "base_train_list_sha256": _sha256(base_root / "train.txt"),
        "v2_training_root": str(v2_root),
        "v2_train_list": str((v2_root / "train.txt").resolve()),
        "v2_train_list_sha256": _sha256(v2_root / "train.txt"),
        "output_root": str(output_root),
        "base_train_exposures": len(base_train),
        "v2_train_unique_members": len(v2_train),
        "added_v2_exposures": len(extra),
        "total_train_exposures": len(train_out),
        "validation_images": len(base_val),
        "test_images": len(base_test),
        "dataset_yaml_sha256": _sha256(dataset_path),
        "train_list_sha256": _sha256(train_path),
        "validation_list_sha256": _sha256(val_path),
        "test_list_sha256": _sha256(test_path),
        "protected_overlap_count": 0,
    }
    (output_root / "membership.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"output_root": str(output_root), "dataset_yaml": str(dataset_path), **{key: evidence[key] for key in ("base_train_exposures", "v2_train_unique_members", "added_v2_exposures", "total_train_exposures", "validation_images", "test_images")}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.base_root, args.v2_root, args.output_root, repeats=args.repeats, seed=args.seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
