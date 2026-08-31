"""CLI for composing and training the sealed semi-supervised Student."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from fruit_ssod.training.semi_supervised import SemiSupervisedTrainingError, StudentTrainingRunner, load_student_experiment
from fruit_ssod.training.student_dataset import StudentDatasetError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose Task 8/14/15-sealed Student data and train a five-class SSOD Student.")
    parser.add_argument("--config", type=Path, required=True, help="Student SSOD experiment YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/compose the sealed Student snapshot without importing Ultralytics or using CUDA.")
    parser.add_argument("--epochs", type=int, help="Positive override, used for the approved one-epoch real smoke run.")
    parser.add_argument("--batch", type=int, help="Positive batch-size override.")
    parser.add_argument("--device", help="Ultralytics device selector, for example cuda:0.")
    parser.add_argument("--run-id", help="Optional unique safe run identifier.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        experiment = load_student_experiment(args.config)
        changes: dict[str, object] = {}
        if args.epochs is not None:
            changes["epochs"] = args.epochs
        if args.batch is not None:
            changes["batch"] = args.batch
        if args.device is not None:
            changes["device"] = args.device
        if any(isinstance(value, int) and value <= 0 for value in changes.values()):
            raise SemiSupervisedTrainingError("Problem: --epochs and --batch must be positive integers. Likely cause: a zero or negative override was supplied. Remediation: pass positive values.")
        if changes:
            experiment = replace(experiment, **changes)
        command = tuple(sys.argv if argv is None else (sys.executable, "-m", "fruit_ssod.cli.train_student", *argv))
        record, run_dir = StudentTrainingRunner().run(experiment, command=command, dry_run=args.dry_run, run_id=args.run_id)
    except (SemiSupervisedTrainingError, StudentDatasetError, ValueError, OSError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"run_id": record.run_id, "run_dir": str(run_dir), "status": record.status, "split_fingerprint": record.split_fingerprint}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
