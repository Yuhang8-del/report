"""Command-line entry point for local deterministic split creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fruit_ssod.data.splitting import SplitError, split_records, write_split_outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create leakage-safe Fruit SSOD image-group splits without network access.")
    parser.add_argument("--input-manifest", required=True, help="Local JSON manifest containing an images array of image-level records.")
    parser.add_argument("--output-root", required=True, help="Explicit directory for generated split artifacts.")
    parser.add_argument("--source-root", help="Optional base directory for relative source image paths, used only for collision safety.")
    parser.add_argument("--seed", type=int, help="Set all three deterministic seeds to this integer.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--budget-seed", type=int, default=3407)
    parser.add_argument("--unlabeled-seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--pseudo-audit-fraction", type=float, default=0.05)
    parser.add_argument("--unlabeled-fraction", type=float, default=0.20)
    parser.add_argument("--budgets", default="10,20,40,100", help="Comma-separated nested labelled budgets, ending in 100 (default: 10,20,40,100).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report artifacts without writing files.")
    return parser


def _load_images(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"Problem: input manifest cannot be read as JSON. Likely cause: {error}. Remediation: provide a readable local JSON image manifest.") from error
    if isinstance(value, list):
        images = value
    elif isinstance(value, dict):
        images = value.get("images", value.get("records"))
    else:
        images = None
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        raise SplitError("Problem: input manifest has no image-level images array. Likely cause: object-label rows or malformed JSON were supplied. Remediation: provide {'images': [...]} where every item has image-level class_presence, labels, and duplicate_group_id fields.")
    return images


def _budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SplitError("Problem: --budgets is invalid. Likely cause: a budget is not an integer percentage. Remediation: use comma-separated values such as 10,20,40,100.") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = Path(args.input_manifest).resolve(strict=False)
        output_root = Path(args.output_root).resolve(strict=False)
        source_root = Path(args.source_root).resolve(strict=False) if args.source_root else None
        split_seed, budget_seed, unlabeled_seed = ((args.seed, args.seed, args.seed) if args.seed is not None else (args.split_seed, args.budget_seed, args.unlabeled_seed))
        result = split_records(_load_images(manifest), validation_fraction=args.validation_fraction, test_fraction=args.test_fraction, pseudo_audit_fraction=args.pseudo_audit_fraction, unlabeled_fraction=args.unlabeled_fraction, split_seed=split_seed, budget_seed=budget_seed, unlabeled_seed=unlabeled_seed, budgets=_budgets(args.budgets))
        written = write_split_outputs(result, output_root, input_manifest=manifest, source_root=source_root, dry_run=args.dry_run)
    except SplitError as error:
        parser.error(str(error))
        return 2  # pragma: no cover - argparse exits
    print(json.dumps({"output_root": str(output_root), "dry_run": args.dry_run, "written_count": len(written), "fingerprints": dict(result.fingerprints)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
