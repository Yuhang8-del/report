"""Build the journal-style overview figure from real project photographs.

The drawing is deterministic and deliberately uses a restrained journal
language: white background, thin black rules, direct labels and only three
colorblind-safe accent colours.  Raster content is restricted to photographs
already present in the registered project datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KNOWN = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]
NOVEL = ["Avocado", "Blueberry", "Cherry", "Kiwi", "Mango", "Rockmelon"]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#202020"
MID = "#666666"
LIGHT = "#D8D8D8"


def _load_json(path: Path):
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _labelled_examples(dataset_yaml: Path):
    import yaml

    descriptor = yaml.safe_load(dataset_yaml.resolve(strict=True).read_text(encoding="utf-8"))
    root = dataset_yaml.resolve().parent
    train_list = root / descriptor.get("train", "train.txt")
    selected = {}
    for raw in train_list.read_text(encoding="utf-8").splitlines():
        image_path = Path(raw.strip())
        label_path = root / "labels" / "train" / f"{image_path.stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            continue
        labels = []
        for row in label_path.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if len(fields) >= 5:
                labels.append((int(fields[0]), *[float(v) for v in fields[1:5]]))
        for class_id, x, y, width, height in labels:
            area = width * height
            if class_id not in selected or area > selected[class_id][0]:
                selected[class_id] = (area, image_path, labels)
    if len(selected) < len(KNOWN):
        raise RuntimeError("Could not resolve one registered photograph per known class")
    return selected


def _unlabelled_examples(unlabelled_manifest: Path, limit=3):
    payload = _load_json(unlabelled_manifest)
    dataset_root = unlabelled_manifest.resolve().parents[2]
    paths = []
    for record in payload["records"]:
        relative = Path(record["file_path"])
        candidates = [
            unlabelled_manifest.resolve().parent / relative,
            dataset_root / "raw" / "open_images_v7" / "converted" / "formal_200_per_class" / relative,
        ]
        match = next((path for path in candidates if path.is_file()), None)
        if match is not None:
            paths.append(match)
        if len(paths) == limit:
            break
    if len(paths) < limit:
        raise RuntimeError("Could not resolve enough real unlabelled photographs")
    return paths


def _novel_examples(novel_manifest: Path):
    selected = {}
    for record in _load_json(novel_manifest)["records"]:
        path = Path(record["path"])
        lower_parts = {part.lower() for part in path.parts}
        for category in NOVEL:
            if category.lower() in lower_parts and category not in selected and path.is_file():
                selected[category] = path
        if len(selected) == len(NOVEL):
            break
    if len(selected) < len(NOVEL):
        raise RuntimeError("Could not resolve one real photograph per novel category")
    return selected


def build_figure(
    output_png: Path,
    output_pdf: Path,
    dataset_yaml: Path,
    unlabelled_manifest: Path,
    novel_manifest: Path,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from PIL import Image

    labelled = _labelled_examples(dataset_yaml)
    unlabelled = _unlabelled_examples(unlabelled_manifest)
    novel = _novel_examples(novel_manifest)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(13.6, 5.15))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.98, bottom=0.035)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5.15)
    ax.axis("off")

    def arrow(start, end, color=BLUE, width=1.25, style="-|>"):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle=style, color=color, lw=width, shrinkA=0, shrinkB=0, mutation_scale=10),
            zorder=8,
        )

    def text(x, y, value, **kwargs):
        defaults = dict(ha="center", va="center", color=DARK, fontsize=7.4)
        defaults.update(kwargs)
        ax.text(x, y, value, **defaults)

    def panel(x, width, letter, title):
        ax.add_patch(patches.Rectangle((x, 0.18), width, 4.76, fill=False, edgecolor=DARK, lw=0.85))
        text(x + 0.16, 4.77, f"({letter})", ha="left", va="top", fontsize=9.2, fontweight="bold")
        text(x + width / 2, 4.76, title, va="top", fontsize=9.2, fontweight="bold")

    def photo(path, x, y, width, height, label=None, boxes=None, box_color=BLUE, dashed=False):
        image = Image.open(path).convert("RGB")
        image_ratio = image.width / image.height
        frame_ratio = width / height
        if image_ratio >= frame_ratio:
            draw_width = width
            draw_height = width / image_ratio
        else:
            draw_height = height
            draw_width = height * image_ratio
        x0 = x + (width - draw_width) / 2
        y0 = y + (height - draw_height) / 2
        ax.add_patch(patches.Rectangle((x, y), width, height, facecolor="white", edgecolor="#777777", lw=0.55))
        ax.imshow(np.asarray(image), extent=(x0, x0 + draw_width, y0, y0 + draw_height), zorder=2, aspect="auto")
        if boxes:
            for class_id, cx, cy, bw, bh in boxes:
                left = x0 + (cx - bw / 2) * draw_width
                bottom = y0 + (1 - cy - bh / 2) * draw_height
                ax.add_patch(patches.Rectangle(
                    (left, bottom), bw * draw_width, bh * draw_height,
                    fill=False, edgecolor=box_color, lw=0.85,
                    linestyle=(0, (3, 2)) if dashed else "solid", zorder=5,
                ))
        if label:
            text(x + width / 2, y - 0.10, label, va="top", fontsize=6.5)

    def network(x, y, label, accent=BLUE):
        for index, (w, h, shade) in enumerate([(0.38, 0.75, "#E8E8E8"), (0.30, 0.61, "#D0D0D0"), (0.22, 0.47, "#B8B8B8")]):
            ax.add_patch(patches.Rectangle((x + index * 0.23, y + (0.75 - h) / 2), w, h, facecolor=shade, edgecolor=DARK, lw=0.65))
        ax.plot([x + 0.38, x + 0.47], [y + 0.375, y + 0.375], color=accent, lw=1.0)
        ax.plot([x + 0.70, x + 0.77], [y + 0.375, y + 0.375], color=accent, lw=1.0)
        text(x + 0.38, y - 0.15, label, va="top", fontsize=7.0, fontweight="bold")

    # (a) Supervised Teacher: real photographs, human boxes and detector.
    panel(0.08, 3.72, "a", "Supervised Teacher")
    text(1.94, 4.31, "Human-labelled public photographs (n = 542)", fontsize=7.2)
    known_ids = [0, 1, 4]
    for index, class_id in enumerate(known_ids):
        _, image_path, labels = labelled[class_id]
        own_boxes = [row for row in labels if row[0] == class_id]
        photo(image_path, 0.34 + index * 1.08, 3.22, 0.88, 0.73, KNOWN[class_id], own_boxes, box_color=BLUE)
    arrow((1.94, 3.08), (1.94, 2.72))
    text(2.14, 2.90, "box supervision", ha="left", fontsize=6.4, color=BLUE)
    network(1.47, 1.78, "YOLOv8m Teacher")
    arrow((1.94, 1.61), (1.94, 1.24))
    ax.add_patch(patches.Rectangle((0.74, 0.50), 2.40, 0.64, facecolor="white", edgecolor=DARK, lw=0.75))
    text(1.94, 0.86, "Five-class detector", fontsize=7.8, fontweight="bold")
    text(1.94, 0.64, "Apple | Banana | Orange | Strawberry | Pineapple", fontsize=5.9, color=MID)

    # (b) Teacher inference, filtering and Student construction.
    panel(3.94, 5.02, "b", "Semi-supervised Student")
    text(6.45, 4.31, "Real unlabelled photographs (n = 2,341)", fontsize=7.2)
    for index, image_path in enumerate(unlabelled):
        photo(image_path, 4.22 + index * 1.05, 3.35, 0.86, 0.65)
    network(7.62, 3.30, "Teacher inference")
    arrow((7.29, 3.63), (7.53, 3.63))
    arrow((6.60, 3.35), (6.60, 2.91))
    text(6.80, 3.10, "candidate boxes", ha="left", fontsize=6.3, color=BLUE)

    # Candidate thumbnail and the journal-style decision chain.
    photo(unlabelled[0], 4.28, 2.05, 1.20, 0.82, boxes=[(0, 0.52, 0.52, 0.55, 0.55)], box_color=ORANGE, dashed=True)
    text(4.88, 1.92, "Teacher candidates", va="top", fontsize=6.5)
    arrow((5.56, 2.47), (5.88, 2.47), color=ORANGE)
    ax.add_patch(patches.Rectangle((5.92, 1.84), 1.36, 1.23, facecolor="white", edgecolor=DARK, lw=0.75))
    text(6.60, 2.86, "Trust Filter", fontsize=7.5, fontweight="bold")
    for i, line in enumerate(["confidence", "paired-view IoU", "geometry / class"]):
        ax.plot([6.09, 6.20], [2.59 - i * 0.25, 2.59 - i * 0.25], color=GREEN, lw=1.6)
        text(6.26, 2.59 - i * 0.25, line, ha="left", fontsize=6.1)
    arrow((7.33, 2.47), (7.62, 2.47), color=GREEN)
    ax.add_patch(patches.Rectangle((7.67, 2.04), 0.96, 0.86, facecolor="white", edgecolor=DARK, lw=0.75))
    ax.plot([7.84, 8.46], [2.70, 2.70], color=GREEN, lw=1.0)
    ax.plot([7.84, 8.36], [2.53, 2.53], color=GREEN, lw=1.0)
    ax.plot([7.84, 8.27], [2.36, 2.36], color=GREEN, lw=1.0)
    text(8.15, 1.91, "2,036 pseudo boxes\n94.3% audit precision", va="top", fontsize=6.2, color=GREEN)

    # Human and pseudo supervision merge explicitly before Student fitting.
    text(4.58, 1.10, "verified human boxes", fontsize=6.2)
    text(6.20, 1.10, "accepted pseudo boxes", fontsize=6.2)
    ax.plot([4.58, 4.58, 6.78], [0.98, 0.76, 0.76], color=DARK, lw=0.8)
    ax.plot([6.20, 6.20, 6.78], [0.98, 0.76, 0.76], color=DARK, lw=0.8)
    arrow((6.78, 0.76), (7.32, 0.76), color=GREEN)
    network(7.43, 0.55, "YOLOv8m Student", accent=GREEN)

    # (c) Additional fruit discovery and customer-facing deployment.
    panel(9.10, 4.42, "c", "Extension and deployment")
    text(11.31, 4.31, "Six categories outside the detector registry", fontsize=7.2)
    for index, category in enumerate(NOVEL):
        row, col = divmod(index, 3)
        photo(novel[category], 9.39 + col * 0.83, 3.56 - row * 0.88, 0.67, 0.55, category)
    arrow((11.92, 3.19), (12.20, 3.19), color=ORANGE)
    ax.add_patch(patches.Circle((12.50, 3.19), 0.31, facecolor="white", edgecolor=DARK, lw=0.75))
    for angle, color in zip([0.5, 2.2, 3.8, 5.1], [BLUE, ORANGE, GREEN, "#CC79A7"]):
        ax.add_patch(patches.Circle((12.50 + 0.17 * np.cos(angle), 3.19 + 0.17 * np.sin(angle)), 0.035, color=color))
    text(12.50, 2.78, "self-supervised\nimage-level clusters", va="top", fontsize=6.2)

    ax.plot([9.34, 13.28], [2.32, 2.32], color=LIGHT, lw=0.8)
    text(11.31, 2.13, "Fixed-test evaluation and Windows demonstration", fontsize=7.2, fontweight="bold")
    ax.add_patch(patches.Rectangle((9.48, 0.72), 1.14, 0.86, facecolor="white", edgecolor=DARK, lw=0.75))
    for i in range(3):
        ax.plot([9.66, 10.45], [1.36 - i * 0.20, 1.36 - i * 0.20], color=MID, lw=0.75)
    text(10.05, 0.58, "frozen test", va="top", fontsize=6.3)
    arrow((10.72, 1.15), (11.10, 1.15))
    # Desktop, camera and export are represented as technical line icons.
    ax.add_patch(patches.Rectangle((11.14, 0.72), 1.18, 0.88, facecolor="white", edgecolor=DARK, lw=0.85))
    ax.add_patch(patches.Rectangle((11.25, 0.84), 0.96, 0.60, facecolor="#F4F4F4", edgecolor=MID, lw=0.55))
    ax.plot([11.73, 11.73], [0.58, 0.72], color=DARK, lw=0.75)
    ax.plot([11.46, 12.00], [0.58, 0.58], color=DARK, lw=0.75)
    ax.add_patch(patches.Circle((12.75, 1.17), 0.25, facecolor="white", edgecolor=DARK, lw=0.8))
    ax.add_patch(patches.Circle((12.75, 1.17), 0.09, facecolor="white", edgecolor=BLUE, lw=0.8))
    ax.add_patch(patches.Rectangle((12.58, 1.43), 0.34, 0.10, facecolor="white", edgecolor=DARK, lw=0.7))
    text(12.10, 0.37, "English GUI | images | video | external camera", fontsize=6.1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=360, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output_pdf, facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output_png, output_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--unlabelled-manifest", type=Path, required=True)
    parser.add_argument("--novel-manifest", type=Path, required=True)
    args = parser.parse_args()
    build_figure(
        args.output_png,
        args.output_pdf,
        args.dataset_yaml,
        args.unlabelled_manifest,
        args.novel_manifest,
    )


if __name__ == "__main__":
    main()
