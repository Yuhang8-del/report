"""CLI entry point for an auditable supervised YOLO reference run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from fruit_ssod.training.supervised import SupervisedTrainingError, SupervisedTrainingRunner, load_supervised_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or validate a five-class supervised fruit detector experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Supervised experiment YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and save provenance without loading Ultralytics or training.")
    parser.add_argument("--epochs", type=int, help="Override experiment epochs for a smoke run.")
    parser.add_argument("--batch", type=int, help="Override batch size; default is 4 from the model config.")
    parser.add_argument("--device", help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume from a prior weights/last.pt checkpoint; pass its path.")
    parser.add_argument("--run-id", help="Optional unique safe artifact ID; normally generated automatically.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        experiment = load_supervised_experiment(args.config)
        changes: dict[str, object] = {}
        if args.epochs is not None:
            changes["epochs"] = args.epochs
        if args.batch is not None:
            changes["batch"] = args.batch
        if args.device is not None:
            changes["device"] = args.device
        if changes:
            # Reuse the strict loader validators by checking simple CLI overrides here.
            if any(isinstance(value, int) and value <= 0 for value in changes.values()):
                raise SupervisedTrainingError("Problem: --epochs and --batch must be positive integers. Likely cause: a zero or negative smoke-run override was supplied. Remediation: pass a positive integer.")
            experiment = replace(experiment, **changes)
        resume: Path | None = None
        if args.resume:
            if args.resume == "auto":
                raise SupervisedTrainingError("Problem: --resume requires a checkpoint path. Likely cause: automatic checkpoint discovery could select the wrong experiment. Remediation: pass --resume path/to/weights/last.pt explicitly.")
            resume = Path(args.resume)
        command = tuple(sys.argv if argv is None else (sys.executable, "-m", "fruit_ssod.cli.train_supervised", *argv))
        record, run_dir = SupervisedTrainingRunner().run(experiment, command=command, dry_run=args.dry_run, resume=resume, run_id=args.run_id)
    except (SupervisedTrainingError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover
    print(json.dumps({"run_id": record.run_id, "run_dir": str(run_dir), "status": record.status, "split_fingerprint": record.split_fingerprint}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
