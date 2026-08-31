"""Seal reviewed auxiliary image-only data into the pseudo-label pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.unlabeled_extension import UnlabeledExtensionError, extend_unlabeled_pool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extend Task 8's unlabeled pool with a reviewed auxiliary image-only manifest.")
    parser.add_argument("--base-unlabeled", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--auxiliary-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-prefix", required=True, help="Relative auxiliary image directory below the shared raw-data root.")
    args = parser.parse_args(argv)
    try:
        result = extend_unlabeled_pool(args.base_unlabeled, args.base_split, args.auxiliary_manifest, args.output_root, raw_prefix=args.raw_prefix)
    except UnlabeledExtensionError as error:
        parser.error(str(error))
    print(json.dumps({"output_root": str(result.root), "record_count": result.record_count, "auxiliary_record_count": result.auxiliary_record_count, "split_fingerprint": result.split_fingerprint}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
