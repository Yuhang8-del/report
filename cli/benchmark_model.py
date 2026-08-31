"""CLI entry point for the sealed RTX 3080 inference benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fruit_ssod.evaluation.benchmark import BenchmarkConfig, BenchmarkError, benchmark_model, file_evidence, write_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a final five-class fruit detector on CUDA.")
    parser.add_argument("--weights", type=Path, required=True, help="Nonempty final five-class .pt checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="New immutable benchmark JSON path.")
    parser.add_argument("--warmup", type=int, default=20, help="CUDA warm-up inference iterations (default: 20).")
    parser.add_argument("--iterations", type=int, default=100, help="Measured synchronized inference iterations (default: 100).")
    parser.add_argument("--imgsz", type=int, default=640, help="Square synthetic input size (default: 640).")
    parser.add_argument("--device", default="cuda:0", help="CUDA device selector (default: cuda:0).")
    parser.add_argument("--dry-run", action="store_true", help="Validate weights and protocol without importing CUDA or running inference.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(warmup_iterations=args.warmup, measured_iterations=args.iterations, image_size=args.imgsz, device=args.device)
        evidence = file_evidence(args.weights)
        if args.output.exists():
            raise BenchmarkError(f"Problem: benchmark output already exists. Likely cause: {args.output} would be overwritten. Remediation: preserve the immutable benchmark or choose a new output path.")
        if args.dry_run:
            print(json.dumps({"authorized": True, "config": config.mapping(), "model": evidence, "output": str(args.output)}, sort_keys=True)); return 0
        summary = benchmark_model(weights=args.weights, config=config)
        output = write_benchmark(summary, args.output)
    except (BenchmarkError, OSError, ValueError) as error:
        parser.error(str(error)); return 2  # pragma: no cover
    print(json.dumps({"output": str(output), "benchmark": summary.mapping()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
