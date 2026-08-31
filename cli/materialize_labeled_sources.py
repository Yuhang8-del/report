"""CLI for a reproducible union of independently imported labeled sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.labeled_source_merge import LabeledSourceInput, LabeledSourceMergeError, materialize_labeled_sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize two or more canonical labeled sources under one fresh local image root.")
    parser.add_argument("--source", action="append", nargs=3, metavar=("NAME", "MANIFEST", "IMAGE_ROOT"), required=True, help="Repeat for each source: canonical name, manifest JSON, and image root for relative paths. Use quoted '*' only for a previously materialized union that contains several canonical sources.")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh output root for copied images and combined_annotations.json.")
    args = parser.parse_args(argv)
    try:
        inputs = [LabeledSourceInput(name, Path(manifest), Path(image_root)) for name, manifest, image_root in args.source]
        result = materialize_labeled_sources(inputs, args.output_root)
    except LabeledSourceMergeError as error:
        parser.error(str(error))
    print(json.dumps({"root": str(result.root), "manifest": str(result.manifest), "image_count": result.image_count, "record_count": result.record_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
