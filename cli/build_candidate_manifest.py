"""Build image-level deterministic-split input from cleaned annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.data.candidate_manifest import CandidateManifestError, build_candidate_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate clean canonical object rows into the image-level create_splits input.")
    parser.add_argument("--cleaned-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        image_count = build_candidate_manifest(args.cleaned_manifest, args.output)
    except CandidateManifestError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "image_count": image_count}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
