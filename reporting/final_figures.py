"""Evidence-bound final-report figure generation."""

from __future__ import annotations

import json
import hashlib
import re
from html import escape
from pathlib import Path
from typing import Any, Mapping

from fruit_ssod.reporting.result_figures import figure_payloads


class FinalFigureError(ValueError):
    """Raised when final figures cannot be derived from sealed report data."""


def _problem(problem: str, cause: str, remediation: str) -> FinalFigureError:
    return FinalFigureError(f"Problem: {problem}. Likely cause: {cause}. Remediation: {remediation}.")


def report_aggregate_view(report_data: Mapping[str, Any]) -> dict[str, Any]:
    """Project report data into the existing deterministic figure input shape."""
    if report_data.get("protocol") != "fruit_ssod_final_report_data_v1":
        raise _problem("report-data protocol is unsupported", repr(report_data.get("protocol")), "build assets only from immutable report_data.json")
    methods, metrics = report_data.get("methods"), report_data.get("metrics")
    if not isinstance(methods, Mapping) or not isinstance(metrics, Mapping):
        raise _problem("report-data methods or metrics are missing", "the report-data JSON is incomplete", "rebuild report_data.json from the verified result package")
    rows = metrics.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise _problem("report-data run rows are malformed", repr(rows), "rebuild report_data.json from immutable result rows")
    groups = methods.get("main_groups")
    if not isinstance(groups, Mapping) or not all(isinstance(groups.get(name), Mapping) for name in ("supervised_20", "trust_main")):
        raise _problem("report-data main method groups are missing", repr(groups), "rebuild report_data.json after Task 18 aggregation")
    return {"summary": {"main_groups": methods["main_groups"]}, "rows": rows}


def _svg(title: str, lines: list[str]) -> str:
    height = 100 + 28 * len(lines)
    contents = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="{height}" viewBox="0 0 1800 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="30" y="42" font-size="26" font-family="Arial">{escape(title)}</text>',
    ]
    contents.extend(f'<text x="40" y="{80 + 28 * index}" font-size="18" font-family="Arial">{escape(line)}</text>' for index, line in enumerate(lines))
    return "\n".join(contents + ["</svg>", ""])


def _mapping_lines(value: object, *, prefix: str, limit: int = 8) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}: unavailable from sealed report data."]
    lines = [f"{prefix}: {key} = {item}" for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:limit]]
    return lines or [f"{prefix}: no entries were recorded."]


def _workflow_figure() -> str:
    return _svg("Evidence-bound semi-supervised workflow", [
        "Public licensed data -> canonical five-class mapping -> audit and leakage-safe split",
        "Supervised Teacher -> offline pseudo-label candidates -> sealed Trust Filter audit",
        "Human labels plus accepted pseudo labels -> Student experiment matrix -> fixed primary test",
        "Result aggregation -> immutable report data -> figures, tables, report and delivery manifest",
        "Open-world discovery remains future work; no completed open-world capability is claimed.",
    ])


def _dataset_figure(report_data: Mapping[str, Any]) -> str:
    datasets = report_data.get("datasets")
    if not isinstance(datasets, Mapping):
        return _svg("Dataset composition and audit evidence", ["Dataset summary is unavailable from sealed report data."])
    lines = _mapping_lines(datasets.get("label_budget_image_counts"), prefix="Label-budget image count")
    source_summary = datasets.get("source_license_summary")
    if isinstance(source_summary, list):
        lines.extend(f"Source/ licence: {item}" for item in source_summary[:4])
    elif isinstance(source_summary, Mapping):
        lines.extend(_mapping_lines(source_summary, prefix="Source licence", limit=4))
    else:
        lines.append("Source/ licence summary: unavailable from sealed report data.")
    return _svg("Dataset composition and audit evidence", lines)


def _dataset_montage(report_data: Mapping[str, Any]) -> Path:
    datasets = report_data.get("datasets")
    montage = datasets.get("sample_annotation_montage") if isinstance(datasets, Mapping) else None
    path = Path(montage.get("path")) if isinstance(montage, Mapping) and isinstance(montage.get("path"), str) else None
    expected_bytes = montage.get("bytes") if isinstance(montage, Mapping) else None
    expected_sha = montage.get("sha256") if isinstance(montage, Mapping) else None
    if path is None or not path.is_file() or isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0 or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise _problem("dataset annotation montage evidence is malformed", repr(montage), "rebuild report_data.json from the dataset audit outputs")
    content = path.read_bytes()
    if len(content) != expected_bytes or hashlib.sha256(content).hexdigest() != expected_sha.lower():
        raise _problem("dataset annotation montage differs from its sealed evidence", str(path), "restore the audited montage or regenerate report_data.json")
    return path


def _write_publication_png(svg: str, destination: Path) -> None:
    """Rasterize the controlled text figures at 300 DPI without SVG tooling."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as error:
        raise _problem("Pillow is unavailable for report rasterization", error.name or "PIL", "install the locked report dependency in the fruit-ssod environment") from error
    lines = [re.sub(r"<[^>]+>", "", item) for item in re.findall(r"<text[^>]*>(.*?)</text>", svg)]
    if not lines:
        raise _problem("figure SVG has no text payload", destination.name, "regenerate the controlled final figures")
    try:
        title_font, body_font = ImageFont.truetype("arial.ttf", 48), ImageFont.truetype("arial.ttf", 32)
    except OSError:
        title_font, body_font = ImageFont.load_default(), ImageFont.load_default()
    height = max(360, 160 + 70 * len(lines))
    image = Image.new("RGB", (2400, height), "white")
    drawer = ImageDraw.Draw(image)
    drawer.text((60, 50), lines[0], fill="black", font=title_font)
    for index, line in enumerate(lines[1:]):
        drawer.text((80, 140 + 70 * index), line, fill="black", font=body_font)
    image.save(destination, format="PNG", dpi=(300, 300))


def _pseudo_quality_figure(report_data: Mapping[str, Any]) -> str:
    quality = report_data.get("pseudo_label_quality")
    if not isinstance(quality, Mapping):
        return _svg("Pseudo-label filtering evidence", ["Pseudo-label audit is unavailable from sealed report data."])
    lines = [f"Teacher run: {quality.get('teacher_run_id', 'unavailable')}" ]
    metrics = quality.get("metrics")
    after = metrics.get("after_filter") if isinstance(metrics, Mapping) else None
    overall = after.get("overall") if isinstance(after, Mapping) else None
    lines.extend(_mapping_lines(overall, prefix="Post-filter overall metric", limit=6))
    refresh = quality.get("pseudo_refresh")
    if isinstance(refresh, Mapping):
        lines.append(f"Pseudo refresh allowed: {refresh.get('allowed', 'unavailable')}; reason: {refresh.get('reason', 'unavailable')}")
    else:
        lines.append("Pseudo refresh gate: unavailable from sealed report data.")
    return _svg("Pseudo-label filtering evidence", lines)


def _failure_figure(rows: list[Mapping[str, Any]]) -> str:
    failed = [row for row in rows if row.get("status") != "complete" or row.get("evaluation_status") != "complete"]
    if not failed:
        return _svg("Failure and missing-evidence record", ["No failed or incomplete run rows were supplied; this is not evidence that failure cases do not exist."])
    lines = []
    for row in failed[:8]:
        issues = row.get("issues")
        reason = " | ".join(str(item) for item in issues) if isinstance(issues, list) and issues else "no structured issue text"
        lines.append(f"{row.get('run_id', 'unknown run')}: status={row.get('status')}, evaluation={row.get('evaluation_status')}; {reason}")
    return _svg("Failure and missing-evidence record", lines)


def _deployment_figure(report_data: Mapping[str, Any]) -> str:
    deployment = report_data.get("deployment")
    if not isinstance(deployment, Mapping):
        return _svg("RTX 3080 deployment evidence", ["Deployment benchmark is unavailable from sealed report data."])
    lines: list[str] = []
    for key in ("fps", "peak_allocated_mib"):
        lines.append(f"{key}: {deployment.get(key, 'unavailable')}")
    latency = deployment.get("latency_ms")
    lines.extend(_mapping_lines(latency, prefix="Latency (ms)", limit=5))
    environment = deployment.get("environment")
    if isinstance(environment, Mapping):
        lines.append(f"GPU: {environment.get('gpu_name', 'unavailable')}; CUDA: {environment.get('cuda_runtime', 'unavailable')}; PyTorch: {environment.get('torch_version', 'unavailable')}")
    else:
        lines.append("GPU environment: unavailable from sealed report data.")
    return _svg("RTX 3080 deployment evidence", lines)


def write_final_figures(report_data: Mapping[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Write at most ten evidence-bound figures, with missing values visible."""
    aggregate = report_aggregate_view(report_data)
    payloads = {
        "workflow.svg": _workflow_figure(),
        "dataset_composition.svg": _dataset_figure(report_data),
        "pseudo_label_quality.svg": _pseudo_quality_figure(report_data),
        **figure_payloads(aggregate),
        "failure_cases.svg": _failure_figure(aggregate["rows"]),
        "deployment.svg": _deployment_figure(report_data),
    }
    montage = _dataset_montage(report_data)
    if len(payloads) + 1 > 10:
        raise _problem("final figure set exceeds the report limit", str(len(payloads)), "reduce the controlled final figure set to at most ten")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, payload in payloads.items():
        destination = output_dir / f"{Path(name).stem}.png"
        _write_publication_png(payload, destination)
        paths.append(destination)
    montage_destination = output_dir / "dataset_annotation_examples.png"
    montage_destination.write_bytes(montage.read_bytes())
    paths.append(montage_destination)
    captions = {
        "workflow.png": "Figure: Evidence-bound workflow and scope boundary.",
        "dataset_composition.png": "Figure: Dataset composition, labels and licence evidence.",
        "dataset_annotation_examples.png": "Figure: Deterministic sample annotation montage emitted by the sealed dataset audit.",
        "pseudo_label_quality.png": "Figure: Sealed pseudo-label filtering evidence.",
        "label_budget.png": "Figure: Fixed-test performance across labeled-data budgets.",
        "method_comparison.png": "Figure: Main-method comparison on the fixed primary test.",
        "ablation.png": "Figure: Ablation evidence; unavailable results remain visible.",
        "per_class_ap50.png": "Figure: Per-class AP50 of the Trust Filter main result.",
        "failure_cases.png": "Figure: Failed, incomplete or unavailable evidence records.",
        "deployment.png": "Figure: RTX 3080 deployment benchmark evidence.",
    }
    (output_dir / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return tuple(paths)
