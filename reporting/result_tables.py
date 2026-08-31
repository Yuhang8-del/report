"""Canonical result-table projections used by CSV, XLSX and later reports."""
from __future__ import annotations

import csv
import io
from typing import Any, Mapping, Sequence

from fruit_ssod.evaluation.aggregate import METRIC_NAMES, thaw


def primary_result_rows(aggregate: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """One unhidden row per submitted run, including failures and omissions."""
    rows: list[dict[str, Any]] = []
    for source in aggregate["rows"]:
        primary = source.get("primary_test") if isinstance(source, Mapping) else None
        row = {
            "run_id": source.get("run_id"), "method": source.get("method"), "status": source.get("status"), "evaluation_status": source.get("evaluation_status"),
            "seed": source.get("seed"), "label_budget_percent": source.get("label_budget_percent"),
            "split_fingerprint": source.get("split_fingerprint"), "run_dir": source.get("run_dir"),
            "issues": " | ".join(source.get("issues", ())),
        }
        for metric in METRIC_NAMES:
            row[metric] = primary.get(metric) if isinstance(primary, Mapping) else None
        if isinstance(primary, Mapping) and isinstance(primary.get("per_class_ap50"), Mapping):
            for class_id, ap in primary["per_class_ap50"].items():
                row[f"ap50_class_{class_id}"] = ap
        rows.append(row)
    return tuple(rows)


def main_summary_rows(aggregate: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    groups = aggregate["summary"]["main_groups"]
    rows: list[dict[str, Any]] = []
    for method in ("supervised_20", "trust_main"):
        source = groups[method]
        row: dict[str, Any] = {"method": method, "three_seed_complete": source["complete"], "observed_seeds": ",".join(str(seed) for seed in source["observed_seeds"])}
        for metric in METRIC_NAMES:
            stats = source["metrics"][metric]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_n"] = stats["n"]
        rows.append(row)
    return tuple(rows)


def fruitdet_rows(aggregate: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """FruitDet table deliberately includes only its declared mapped classes."""
    rows: list[dict[str, Any]] = []
    for source in aggregate["rows"]:
        fruitdet = source.get("fruitdet") if isinstance(source, Mapping) else None
        if not isinstance(fruitdet, Mapping):
            continue
        row = {"run_id": source.get("run_id"), "method": source.get("method"), "status": source.get("status"), "mapped_classes": ",".join(fruitdet["mapped_class_names"])}
        row["mapped_mean_ap50"] = fruitdet["mapped_mean_ap50"]
        for class_id, ap in fruitdet["per_class_ap50"].items():
            row[f"ap50_class_{class_id}"] = ap
        rows.append(row)
    return tuple(rows)


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Stable RFC-4180-style UTF-8 CSV with the union of all fields."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in keys})
    return output.getvalue()
