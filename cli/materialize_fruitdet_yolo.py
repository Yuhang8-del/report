"""Materialize a local FruitDet YOLO checkout as external-test evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.fruitdet_yolo import FruitDetYoloError, materialize_fruitdet_yolo
from fruit_ssod.data.schema import LicenseMetadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize FruitDet YOLO test data as a sealed external-only canonical dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-page", required=True)
    parser.add_argument("--license-name", required=True)
    parser.add_argument("--license-url")
    args = parser.parse_args(argv)
    try:
        result = materialize_fruitdet_yolo(args.dataset_root, args.output_root, source_version=args.source_version, source_page=args.source_page, license_metadata=LicenseMetadata(name=args.license_name, url=args.license_url))
    except FruitDetYoloError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "dataset_yaml": str(result.dataset_yaml), "manifest": str(result.manifest), "image_count": result.image_count, "annotation_count": result.annotation_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
