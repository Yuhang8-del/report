"""Create a sealed image-only pseudo-label pool independent of a Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.independent_unlabeled import IndependentUnlabeledError, seal_independent_unlabeled_pool
from fruit_ssod.data.schema import LicenseMetadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal an image-only pseudo-label pool after checking that no image was used by the Teacher.")
    parser.add_argument("--base-split-manifest", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--relative-prefix", required=True)
    parser.add_argument("--teacher-dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", default="open_images_v7")
    parser.add_argument("--license-name", default="Open Images V7 image license")
    parser.add_argument("--license-url", default="https://creativecommons.org/licenses/by/2.0/")
    parser.add_argument("--license-attribution", default="Open Images V7")
    args = parser.parse_args(argv)
    try:
        result = seal_independent_unlabeled_pool(
            base_split_manifest=args.base_split_manifest,
            image_directory=args.image_directory,
            source_root=args.source_root,
            relative_prefix=args.relative_prefix,
            teacher_dataset_root=args.teacher_dataset_root,
            output_root=args.output_root,
            source=args.source,
            license_metadata=LicenseMetadata(name=args.license_name, url=args.license_url, attribution=args.license_attribution),
        )
    except IndependentUnlabeledError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "record_count": result.record_count, "split_fingerprint": result.split_fingerprint}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
