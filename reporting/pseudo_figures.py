"""Audit-only pseudo-label evidence figures.

The renderer is intentionally conservative: it never follows an absolute or
traversing image reference and always leaves a readable placeholder when a
sample image is unavailable.  Numeric results remain in the JSON audit record;
these figures are qualitative diagnostics, not a second source of metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from fruit_ssod.evaluation.pseudo_metrics import AuditBox, AuditImage, PseudoAuditError


def _safe_image_path(root: Path, relative_path: str) -> Path:
    root = root.resolve(strict=False)
    target = (root / relative_path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise PseudoAuditError("Problem: audit example path escapes image root. Likely cause: an image path contains a symbolic-link, absolute-path, or traversal escape. Remediation: use only images stored below the sealed pseudo-audit image root.") from error
    return target


def _placeholder(path: Path, title: str) -> None:
    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 20), title, fill="black")
    draw.text((16, 70), "No readable local pseudo-audit image was available for this diagnostic category.", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_box(draw: ImageDraw.ImageDraw, box: AuditBox, color: str, label: str) -> None:
    draw.rectangle(box.xyxy, outline=color, width=3)
    draw.text((box.xyxy[0] + 2, max(0, box.xyxy[1] - 14)), label, fill=color)


def render_pseudo_audit_examples(
    examples: Mapping[str, Sequence[tuple[AuditImage, AuditBox]]],
    output_dir: Path,
    *,
    image_root: Path | None = None,
    maximum_per_category: int = 4,
) -> Mapping[str, tuple[Path, ...]]:
    """Create kept/rejected/false-positive/missed annotated sample PNGs.

    ``examples`` contains only the boxes already selected by one-to-one audit
    matching.  The function writes a per-category placeholder even when zero
    examples exist, so a release folder cannot silently omit a required audit
    diagnostic.
    """
    required = ("kept", "rejected", "false_positive", "missed")
    if set(examples) != set(required):
        raise PseudoAuditError("Problem: pseudo example categories are incomplete. Likely cause: the audit caller omitted a required diagnostic. Remediation: provide exactly kept, rejected, false_positive, and missed categories.")
    if maximum_per_category <= 0:
        raise PseudoAuditError("Problem: maximum_per_category must be positive. Likely cause: audit examples were disabled. Remediation: retain at least one diagnostic slot per category.")
    root = output_dir.resolve(strict=False)
    if root.exists():
        raise PseudoAuditError(f"Problem: pseudo figure output already exists. Likely cause: {root} would be overwritten. Remediation: choose a new empty audit output directory.")
    published: dict[str, tuple[Path, ...]] = {}
    try:
        root.mkdir(parents=True, exist_ok=False)
        for category in required:
            results: list[Path] = []
            selected = tuple(examples[category])[:maximum_per_category]
            if not selected:
                destination = root / f"{category}_none.png"; _placeholder(destination, f"Pseudo audit: {category}"); results.append(destination)
            for index, (image_info, box) in enumerate(selected, start=1):
                destination = root / f"{category}_{index:02d}_{image_info.source_image_id}.png"
                if any(char in image_info.source_image_id for char in "\\/:"):
                    raise PseudoAuditError("Problem: pseudo-audit image ID cannot form a safe figure filename. Likely cause: an ID includes a path separator. Remediation: use Task 8 source_image_id values without separators.")
                if image_root is None:
                    _placeholder(destination, f"Pseudo audit: {category} ({image_info.source_image_id})"); results.append(destination); continue
                try:
                    with Image.open(_safe_image_path(image_root, image_info.file_path)) as opened:
                        image = opened.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    color = {"kept": "green", "rejected": "orange", "false_positive": "red", "missed": "blue"}[category]
                    _draw_box(draw, box, color, f"{category}: class {box.class_id}")
                    image.save(destination)
                except (OSError, ValueError):
                    _placeholder(destination, f"Pseudo audit: {category} ({image_info.source_image_id})")
                results.append(destination)
            published[category] = tuple(results)
        return published
    except OSError as error:
        raise PseudoAuditError(f"Problem: pseudo audit figures could not be written. Likely cause: {error}. Remediation: choose a new writable output directory.") from error
