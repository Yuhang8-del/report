"""Select a bounded five-fruit Open Images subset from official local CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.open_images_selection import OpenImagesSelectionError, build_open_images_selection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create immutable Open Images fruit selection CSVs from official local metadata.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    caps = parser.add_mutually_exclusive_group(required=True)
    caps.add_argument("--per-class", type=int, help="Uniform unique source-image target for every canonical fruit class.")
    caps.add_argument("--class-caps", type=Path, help="JSON object mapping every canonical fruit class to its positive unique-image cap.")
    parser.add_argument("--image-split", choices=("train", "validation", "test"), default="train", help="Official Open Images source split used to construct download URLs.")
    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument("--exclude-image-ids", type=Path, help="Optional UTF-8 text file with one previously used Open Images ID per line.")
    exclusions.add_argument("--exclude-manifest", type=Path, help="Optional canonical manifest whose source_image_id values must not be selected again.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.class_caps is not None:
            payload = json.loads(args.class_caps.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise OpenImagesSelectionError("Problem: class-caps JSON is not an object. Likely cause: an incompatible caps file was supplied. Remediation: provide one positive integer value for every canonical fruit class.")
            per_class: int | dict[str, int] = payload
        else:
            per_class = args.per_class
        if args.exclude_manifest is not None:
            payload = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
            records = payload.get("records") if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                raise OpenImagesSelectionError("Problem: exclusion manifest has no record list. Likely cause: an incompatible manifest was supplied. Remediation: use a canonical cleaned annotation manifest.")
            excluded_ids = [str(record.get("source_image_id", "")) for record in records if isinstance(record, dict)]
        else:
            excluded_ids = () if args.exclude_image_ids is None else args.exclude_image_ids.read_text(encoding="utf-8").splitlines()
        result = build_open_images_selection(args.annotations, args.class_descriptions, args.image_metadata, args.output_dir, per_class=per_class, image_split=args.image_split, excluded_image_ids=excluded_ids)
    except OpenImagesSelectionError as error:
        _parser().error(str(error))
    print(json.dumps({"output_dir": str(args.output_dir), "selected_image_count": len(result.image_ids), "class_image_counts": dict(result.class_image_counts), "annotation_count": result.annotation_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
