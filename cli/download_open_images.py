"""CLI for local Open Images conversion and optional explicit acquisition.

Author: Fruit SSOD contributors
Date: 2026-07-31
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.open_images import convert_open_images, download_images


def _parser() -> argparse.ArgumentParser:
    """Build a CLI that never assumes a shared data location or source URL."""
    parser = argparse.ArgumentParser(description="Convert Open Images CSV annotations to YOLO labels.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--image-url-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--download", action="store_true", help="Explicitly fetch selected images from the URL map.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent image downloads when --download is set (default: 1).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only; do not write or access URLs.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path; the only dry-run write allowed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion, with downloading opt-in and disabled by dry-run."""
    args = _parser().parse_args(argv)
    result = convert_open_images(
        args.annotations,
        args.class_descriptions,
        args.image_url_map,
        args.output_root,
        max_images=args.max_images,
        dry_run=args.dry_run,
    )
    report = {
        "dry_run": args.dry_run,
        "selected_images": len(result.images),
        "source_image_ids": [item.source_image_id for item in result.images],
        "filtered_flagged_rows": result.filtered_flagged_rows,
        "rejected_invalid_boxes": result.rejected_invalid_boxes,
        "download_requested": args.download,
    }
    if args.download and not args.dry_run:
        downloaded = download_images({item.source_image_id: item.url for item in result.images}, args.output_root, workers=args.workers)
        report["downloaded_images"] = len(downloaded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
