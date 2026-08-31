"""Write local-only auxiliary dataset manifests without acquisition or image copying.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.fruitdet import import_fruitdet
from fruit_ssod.data.fruits360 import import_fruits360
from fruit_ssod.data.berremangra_orange import import_berremangra_orange
from fruit_ssod.data.deepnir import import_deepnir
from fruit_ssod.data.hf_crop_plant import import_hf_crop_plant
from fruit_ssod.data.kaggle_fruit_detection import import_kaggle_fruit_detection
from fruit_ssod.data.snacks_detection import import_snacks_detection
from fruit_ssod.data.strawberry_ds import import_strawberry_ds
from fruit_ssod.data.zenodo_strawberry import import_zenodo_strawberry
from fruit_ssod.data.schema import LicenseMetadata


def _add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    """Require provenance facts instead of guessing a source release or license."""
    parser.add_argument("--source-version", required=True, help="Local source release/version being indexed.")
    parser.add_argument("--source-page", required=True, help="Current source page URL checked by the operator.")
    parser.add_argument("--license-name", required=True, help="License name confirmed for this local source copy.")
    parser.add_argument("--license-url", help="Optional license URL confirmed by the operator.")
    parser.add_argument("--license-attribution", help="Optional attribution text to retain in the manifest.")


def _parser() -> argparse.ArgumentParser:
    """Build distinct local-only import commands for source-specific safety rules."""
    parser = argparse.ArgumentParser(description="Create auxiliary data manifests from local files only; no network calls are made.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fruits360 = subparsers.add_parser("fruits360", help="Index local Fruits-360 images as unlabeled train-pool records.")
    fruits360.add_argument("--images-root", required=True, type=Path)
    fruits360.add_argument("--output-dir", required=True, type=Path)
    fruits360.add_argument("--split", default="train_pool", choices=["train_pool"])
    _add_metadata_arguments(fruits360)
    fruitdet = subparsers.add_parser("fruitdet", help="Convert local FruitDet COCO annotations as external-test records only.")
    fruitdet.add_argument("--annotations", required=True, type=Path)
    fruitdet.add_argument("--images-root", required=True, type=Path)
    fruitdet.add_argument("--output-dir", required=True, type=Path)
    fruitdet.add_argument("--split", default="external_test", choices=["external_test"], help="Fixed to external_test to prevent primary-split contamination.")
    _add_metadata_arguments(fruitdet)
    snacks = subparsers.add_parser("snacks-detection", help="Import reviewed Snacks Detection boxes as supplementary labeled source records before a new project split.")
    snacks.add_argument("--images-root", required=True, type=Path)
    snacks.add_argument("--train-csv", required=True, type=Path)
    snacks.add_argument("--val-csv", required=True, type=Path)
    snacks.add_argument("--test-csv", required=True, type=Path)
    snacks.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(snacks)
    strawberry = subparsers.add_parser("strawberry-ds", help="Extract reviewed Strawberry-DS Parquet boxes as supplementary Strawberry records before a new project split.")
    strawberry.add_argument("--parquet", required=True, type=Path)
    strawberry.add_argument("--output-dir", required=True, type=Path, help="Fresh directory that will contain extracted images and manifest.json.")
    _add_metadata_arguments(strawberry)
    kaggle = subparsers.add_parser("kaggle-fruit-detection", help="Import a local reviewed Kaggle YOLO fruit-detection archive as a fresh labeled source before a new project split.")
    kaggle.add_argument("--dataset-root", required=True, type=Path)
    kaggle.add_argument("--data-yaml", required=True, type=Path)
    kaggle.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(kaggle)
    hf_crop = subparsers.add_parser("hf-crop-plant", help="Import the local reviewed Hugging Face crop/plant YOLO archive as a fresh labeled source before a new project split.")
    hf_crop.add_argument("--dataset-root", required=True, type=Path)
    hf_crop.add_argument("--data-yaml", required=True, type=Path)
    hf_crop.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(hf_crop)
    orange = subparsers.add_parser("berremangra-orange", help="Convert a local CC BY Berremangra Orange YOLO-segmentation release to auditable Orange detection boxes.")
    orange.add_argument("--dataset-root", required=True, type=Path)
    orange.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(orange)
    deepnir = subparsers.add_parser("deepnir", help="Import audited deepNIR Apple, Orange and Strawberry YOLO boxes using their reviewed source-directory categories.")
    deepnir.add_argument("--dataset-root", required=True, type=Path)
    deepnir.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(deepnir)
    zenodo_strawberry = subparsers.add_parser("zenodo-strawberry", help="Import Zenodo 6126677 ripe/unripe Strawberry YOLO boxes while recording peduncles as rejections.")
    zenodo_strawberry.add_argument("--dataset-root", required=True, type=Path)
    zenodo_strawberry.add_argument("--data-yaml", required=True, type=Path)
    zenodo_strawberry.add_argument("--output-dir", required=True, type=Path)
    _add_metadata_arguments(zenodo_strawberry)
    return parser


def _write_manifest(output_dir: Path, manifest: object) -> None:
    """Create only the caller-designated output directory and its one manifest file."""
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Problem: output path {output_dir} is not a directory. Likely cause: --output-dir points at a file. Remediation: provide a directory path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local importer and write its deterministic manifest at the explicit destination."""
    args = _parser().parse_args(argv)
    metadata = {
        "source_version": args.source_version,
        "source_page": args.source_page,
        "license_metadata": LicenseMetadata(name=args.license_name, url=args.license_url, attribution=args.license_attribution),
        "split": getattr(args, "split", "train_pool"),
    }
    if args.command == "fruits360":
        result = import_fruits360(args.images_root, **metadata)
    elif args.command == "fruitdet":
        result = import_fruitdet(args.annotations, args.images_root, **metadata)
    elif args.command == "strawberry-ds":
        result = import_strawberry_ds(args.parquet, args.output_dir, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    elif args.command == "kaggle-fruit-detection":
        result = import_kaggle_fruit_detection(args.dataset_root, args.data_yaml, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    elif args.command == "hf-crop-plant":
        result = import_hf_crop_plant(args.dataset_root, args.data_yaml, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    elif args.command == "berremangra-orange":
        result = import_berremangra_orange(args.dataset_root, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    elif args.command == "deepnir":
        result = import_deepnir(args.dataset_root, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    elif args.command == "zenodo-strawberry":
        result = import_zenodo_strawberry(args.dataset_root, args.data_yaml, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    else:
        result = import_snacks_detection(args.images_root, train_csv=args.train_csv, val_csv=args.val_csv, test_csv=args.test_csv, source_version=args.source_version, source_page=args.source_page, license_metadata=metadata["license_metadata"])
    _write_manifest(args.output_dir, result.manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "record_count": result.manifest["record_count"], "rejection_count": result.manifest["rejection_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
