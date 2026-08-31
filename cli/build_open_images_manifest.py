"""Build canonical data-cleaning input from downloaded Open Images evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.open_images_manifest import OpenImagesManifestError, build_canonical_open_images_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical annotation rows from a completed Open Images conversion.")
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--selection-url-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_canonical_open_images_manifest(args.converted_root, args.selection_url_csv, args.output)
    except OpenImagesManifestError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "image_count": result.image_count, "annotation_count": result.annotation_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
