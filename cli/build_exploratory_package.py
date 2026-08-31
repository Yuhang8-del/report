"""Build an evidence-bound package for the first exploratory Student result.

This package is intentionally separate from the formal Task 18 report package.
It reads immutable run/evaluation evidence and never changes acceptance gates or
turns exploratory metrics into formal matrix results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExploratoryPackageError(ValueError):
    """Raised when exploratory evidence cannot be sealed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExploratoryPackageError(f"{description} cannot be read: {error}") from error
    if not isinstance(value, Mapping):
        raise ExploratoryPackageError(f"{description} is not a JSON object")
    return value


def _file_evidence(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExploratoryPackageError(f"{description} is missing: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _run_evidence(run_dir: Path) -> dict[str, Any]:
    record_path = run_dir / "run_record.json"
    test_path = run_dir / "evaluations" / "test.json"
    record = _load(record_path, "run record")
    evaluation = _load(test_path, "fixed-test evaluation")
    if record.get("status") != "complete":
        raise ExploratoryPackageError(f"run is not complete: {run_dir}")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ExploratoryPackageError(f"fixed-test metrics are missing: {test_path}")
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("relative_path") != "weights/best.pt":
        raise ExploratoryPackageError(f"fixed-test checkpoint evidence is missing: {test_path}")
    checkpoint_path = Path(str(checkpoint.get("path", run_dir / "weights" / "best.pt")))
    checkpoint_evidence = _file_evidence(checkpoint_path, "best checkpoint")
    if checkpoint_evidence["sha256"] != str(checkpoint.get("sha256", "")).lower():
        raise ExploratoryPackageError(f"best checkpoint SHA-256 differs from fixed-test evidence: {run_dir}")
    split_fingerprint = evaluation.get("split_fingerprint") or evaluation.get("membership", {}).get("split_fingerprint")
    if not isinstance(split_fingerprint, str) or not split_fingerprint:
        raise ExploratoryPackageError(f"fixed-test split fingerprint is missing: {test_path}")
    return {
        "run_id": evaluation.get("run_id") or record.get("run_id") or run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "seed": record.get("config_snapshot", {}).get("seed"),
        "label_budget_percent": record.get("config_snapshot", {}).get("label_budget_percent"),
        "validation": record.get("result"),
        "fixed_test": dict(metrics),
        "split_fingerprint": split_fingerprint,
        "exploratory": bool(evaluation.get("exploratory", True)),
        "checkpoint": checkpoint_evidence,
        "run_record": _file_evidence(record_path, "run record"),
        "fixed_test_json": _file_evidence(test_path, "fixed-test evaluation"),
    }


def _open_world_evidence(open_world_dir: Path) -> dict[str, Any]:
    """Seal the completed post-Student novel-fruit discovery artifacts."""

    results_path = open_world_dir / "discovery_results.json"
    results = _load(results_path, "open-world discovery results")
    if results.get("artifact_type") != "post_student_open_world_discovery":
        raise ExploratoryPackageError(f"open-world artifact type is invalid: {results_path}")
    required_files = {
        "results": results_path,
        "manifest": open_world_dir / "discovery_manifest.json",
        "protected_labels": open_world_dir / "protected_evaluation_labels.json",
        "cluster_assignments": open_world_dir / "cluster_assignments.jsonl",
        "self_supervised_checkpoint": open_world_dir / "self_supervised_encoder.pt",
    }
    evidence = {name: _file_evidence(path, f"open-world {name}") for name, path in required_files.items()}
    metrics = results.get("metrics")
    split = results.get("split")
    if not isinstance(metrics, Mapping) or not isinstance(split, Mapping):
        raise ExploratoryPackageError(f"open-world metrics/split evidence is missing: {results_path}")
    return {
        "directory": str(open_world_dir.resolve()),
        "results": dict(results),
        "files": evidence,
    }


def build_exploratory_package(
    *,
    run_dirs: Sequence[Path | str],
    gui_export: Path | str,
    output: Path | str,
    open_world_dir: Path | str | None = None,
) -> Path:
    destination = Path(output).resolve(strict=False)
    if destination.exists():
        raise ExploratoryPackageError(f"output already exists; refusing to overwrite: {destination}")
    runs = [_run_evidence(Path(item).resolve(strict=True)) for item in run_dirs]
    if not runs:
        raise ExploratoryPackageError("at least one completed exploratory run is required")
    fingerprints = {item["split_fingerprint"] for item in runs}
    if len(fingerprints) != 1:
        raise ExploratoryPackageError(f"exploratory runs use different test memberships: {sorted(fingerprints)}")
    gui = Path(gui_export).resolve(strict=True)
    metadata_path = gui / "v0_metadata.json"
    metadata = _load(metadata_path, "GUI candidate metadata")
    if metadata.get("camera_enabled") is not False or metadata.get("open_world_enabled") is not False:
        raise ExploratoryPackageError("GUI candidate metadata does not prove camera/open-world are disabled")
    selected = max(runs, key=lambda item: float(item["fixed_test"]["map50"]))
    open_world = _open_world_evidence(Path(open_world_dir).resolve(strict=True)) if open_world_dir is not None else None
    summary = {
        "schema_version": "1.0",
        "protocol": "fruit_ssod_exploratory_first_result_v1",
        "formal_matrix": False,
        "acceptance_claim": "none",
        "target_map50": 0.80,
        "test_split_fingerprint": next(iter(fingerprints)),
        "runs": sorted(runs, key=lambda item: (str(item["seed"]), item["run_id"])),
        "selected_demo_run_id": selected["run_id"],
        "selected_demo_checkpoint": selected["checkpoint"],
        "open_world": open_world,
        "gui_export": {
            "directory": str(gui),
            "metadata": _file_evidence(metadata_path, "GUI candidate metadata"),
            "manifest": _file_evidence(gui / "manifest.json", "GUI export manifest"),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        (temporary / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        selected_metric = float(selected["fixed_test"]["map50"])
        lines = [
            "# Exploratory first-result package",
            "",
            "This package is evidence-bound exploratory output. It is not the formal Task 17/18 acceptance package and does not claim the 0.80 target.",
            "",
            f"Selected demo run: `{selected['run_id']}`; fixed-test mAP@0.5 = `{selected_metric:.10f}`.",
            f"Protected test split fingerprint: `{summary['test_split_fingerprint']}`.",
            "",
            "## Completed runs",
            "",
            "| Run | Seed | Fixed-test mAP@0.5 | Precision | Recall | Checkpoint SHA-256 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for item in summary["runs"]:
            metrics = item["fixed_test"]
            lines.append(f"| {item['run_id']} | {item['seed']} | {float(metrics['map50']):.10f} | {float(metrics['precision']):.10f} | {float(metrics['recall']):.10f} | `{item['checkpoint']['sha256']}` |")
        if open_world is None:
            lines.extend(["", "The PySide6 export is offline image inference only; camera and open-world modes are disabled.", ""])
        else:
            open_results = open_world["results"]
            discovery = open_results["metrics"]["discovery"]
            holdout = open_results["metrics"]["holdout"]
            novelty = open_results["novelty"]
            known_test_count = int(novelty.get("known_fixed_test_count", 0))
            if known_test_count:
                known_fpr_text = f"{float(novelty.get('known_false_positive_rate', 0.0)):.6f} over {known_test_count} known-test images"
            else:
                known_fpr_text = "not measured (no known-test list supplied)"
            lines.extend(
                [
                    "",
                    "## Post-Student open-world discovery",
                    "",
                    f"Novel pool: `{open_results['split']['image_count']}` images across `{len(open_results['novel_categories_for_protected_evaluation'])}` protected evaluation categories.",
                    f"Discovery purity/NMI/ARI: `{float(discovery['purity']):.6f}` / `{float(discovery['nmi']):.6f}` / `{float(discovery['ari']):.6f}`.",
                    f"Holdout purity/NMI/ARI: `{float(holdout['purity']):.6f}` / `{float(holdout['nmi']):.6f}` / `{float(holdout['ari']):.6f}`.",
                    f"Unknown candidates at threshold `{float(novelty['threshold']):.2f}`: `{novelty['candidate_count']}` (`{float(novelty['candidate_rate']):.6f}`); known-test false-positive rate: `{known_fpr_text}`.",
                    "Cluster names are post-hoc evaluation mappings; runtime class IDs remain unchanged.",
                    "",
                ]
            )
        (temporary / "README.md").write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(destination)
        manifest = {
            "protocol": summary["protocol"],
            "summary": _file_evidence(destination / "summary.json", "summary"),
            "readme": _file_evidence(destination / "README.md", "README"),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--gui-export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--open-world-dir", default=None)
    args = parser.parse_args(argv)
    try:
        result = build_exploratory_package(
            run_dirs=args.run_dir,
            gui_export=args.gui_export,
            output=args.output,
            open_world_dir=args.open_world_dir,
        )
    except (ExploratoryPackageError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
