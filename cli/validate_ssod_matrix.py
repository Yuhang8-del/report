"""Validate and print the Task 17 SSOD matrix without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.training.ssod_matrix import SsodMatrixError, matrix_queue, validate_ssod_matrix, verify_prepared_ssod_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the fixed Task 17 SSOD and ablation matrix before GPU work.")
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--queue", action="store_true", help="Emit all run/skip decisions after validation.")
    parser.add_argument("--artifact-root", type=Path, help="Artifact root used only for conservative resume checks.")
    parser.add_argument("--resume", action="store_true", help="Skip only complete runs with matching config and split fingerprints.")
    parser.add_argument("--verify-preparation", action="store_true", help="Require sealed Task 13→14→15 artifacts whose policy hash exactly matches every matrix config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        if args.resume and args.artifact_root is None:
            parser.error("--resume requires --artifact-root so completed records can be verified")
        if args.queue:
            payload = {"status": "valid", "queue": list(matrix_queue(args.config_directory, artifact_root=args.artifact_root, resume=args.resume, verify_preparation=args.verify_preparation))}
        else:
            validator = verify_prepared_ssod_artifacts if args.verify_preparation else validate_ssod_matrix
            payload = {"status": "valid", "configs": [str(item) for item in validator(args.config_directory)]}
    except SsodMatrixError as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
