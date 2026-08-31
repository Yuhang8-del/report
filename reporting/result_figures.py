"""Evidence-led result figures; unavailable evaluator plots are never faked."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any, Mapping


def _svg(title: str, lines: list[str]) -> str:
    height = 100 + 28 * len(lines)
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="30" y="42" font-size="26" font-family="Arial">{escape(title)}</text>']
    text.extend(f'<text x="40" y="{80 + 28 * index}" font-size="18" font-family="Arial">{escape(line)}</text>' for index, line in enumerate(lines))
    return "\n".join(text + ["</svg>", ""])


def figure_payloads(aggregate: Mapping[str, Any]) -> dict[str, str]:
    """Render only summary figures that are directly determined by metrics."""
    groups = aggregate["summary"]["main_groups"]
    methods = ["supervised_20", "trust_main"]
    comparison = [f"{method}: mAP50 mean={groups[method]['metrics']['map50']['mean']}, std={groups[method]['metrics']['map50']['std']}" for method in methods]
    ap = groups["trust_main"]["per_class_ap50"]
    ap_lines = [f"class {class_id}: AP50 mean={item['mean']}, std={item['std']}" for class_id, item in ap.items()]
    ablations = [row for row in aggregate["rows"] if str(row["method"]).startswith("ablation_")]
    ablation_lines = [f"{row['method']}: {row['primary_test']['map50'] if isinstance(row.get('primary_test'), Mapping) else 'unavailable'}" for row in ablations] or ["No ablation result was supplied; this gap is intentionally visible."]
    return {
        "label_budget.svg": _svg("Label-budget results", [f"{row['method']} / budget {row['label_budget_percent']}: {row['primary_test']['map50'] if isinstance(row.get('primary_test'), Mapping) else 'unavailable'}" for row in aggregate["rows"] if str(row["method"]).startswith("supervised_")] or ["No supervised fixed-test evidence available."]),
        "method_comparison.svg": _svg("Method comparison on fixed primary test", comparison),
        "ablation.svg": _svg("Ablation results", ablation_lines),
        "per_class_ap50.svg": _svg("Trust Filter per-class AP50", ap_lines),
    }


def evaluator_figure_sources(aggregate: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Locate byte-verified raw PR/confusion exports or structured gap evidence.

    Raw exports are optional because framework versions do not always emit PR
    or confusion plots.  Their absence produces a machine-readable missing
    evidence record instead of a chart with invented points.
    """
    kinds = {"precision_recall": "precision_recall", "confusion": "confusion_matrix"}
    found: dict[str, Path] = {}
    missing: dict[str, dict[str, Any]] = {}
    designation = aggregate.get("summary", {}).get("final_trust_figure_source") if isinstance(aggregate.get("summary"), Mapping) else None
    if not isinstance(designation, Mapping) or set(designation) != {"method", "seed", "run_id"} or designation.get("method") != "trust_main" or designation.get("seed") != 42 or designation.get("run_id") != "ssod_trust_seed42":
        return {}, {name: {"status": "invalid", "artifact": artifact, "reason": "aggregate lacks the fixed final Trust run/seed designation"} for name, artifact in kinds.items()}
    selected = [row for row in aggregate["rows"] if isinstance(row, Mapping) and row.get("method") == designation["method"] and row.get("seed") == designation["seed"] and row.get("run_id") == designation["run_id"] and row.get("evaluation_status") == "complete"]
    for public_name, artifact_name in kinds.items():
        candidates: list[Mapping[str, Any]] = []
        for row in selected:
            protocol = row.get("primary_test_protocol") if isinstance(row, Mapping) else None
            raw = protocol.get("raw_evaluator_outputs") if isinstance(protocol, Mapping) else None
            item = raw.get(artifact_name) if isinstance(raw, Mapping) else None
            if isinstance(item, Mapping):
                candidates.append(item)
        if len(selected) != 1 or len(candidates) != 1:
            missing[public_name] = {"status": "missing", "artifact": artifact_name, "reason": "the designated final Trust run/seed has no unique sealed raw evaluator export", "designated_source": dict(designation), "selected_run_count": len(selected), "candidate_count": len(candidates)}
            continue
        item = candidates[0]
        value, digest, size = item.get("path"), item.get("sha256"), item.get("bytes")
        path = Path(value) if isinstance(value, str) else None
        if path is None or not path.is_file() or not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int) or size <= 0:
            missing[public_name] = {"status": "invalid", "artifact": artifact_name, "reason": "raw evaluator export metadata is malformed or file is unavailable"}
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest.lower() or path.stat().st_size != size:
            missing[public_name] = {"status": "invalid", "artifact": artifact_name, "reason": "raw evaluator export differs from sealed SHA-256/byte evidence"}
            continue
        found[public_name] = path
    return found, missing


def write_result_figures(aggregate: Mapping[str, Any], output_dir: Path | str) -> tuple[Path, ...]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, payload in figure_payloads(aggregate).items():
        path = destination / name
        if path.exists():
            raise FileExistsError(f"result figure already exists: {path}")
        path.write_text(payload, encoding="utf-8", newline="\n")
        paths.append(path)
    sources, missing = evaluator_figure_sources(aggregate)
    for name, source in sources.items():
        target = destination / f"{name}{source.suffix.lower() or '.bin'}"
        target.write_bytes(source.read_bytes())
        paths.append(target)
    for name, evidence in missing.items():
        target = destination / f"{name}.missing.json"
        target.write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths.append(target)
    return tuple(paths)
