"""Validate that Task 12 configs are untouched deterministic template renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.training.supervised_matrix import SupervisedMatrixError, validate_reference_configs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the canonical supervised reference matrix before GPU work.")
    parser.add_argument("--template", type=Path, required=True, help="Canonical supervised reference template YAML.")
    parser.add_argument("--config-directory", type=Path, required=True, help="Directory containing generated supervised matrix YAML files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        paths = validate_reference_configs(args.template, args.config_directory)
    except SupervisedMatrixError as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"status": "valid", "configs": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
