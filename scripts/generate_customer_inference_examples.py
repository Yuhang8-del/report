"""Generate representative customer-facing inference examples from the fixed test set."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=1024)
    return parser.parse_args()


def xywhn_to_xyxy(row: list[float], width: int, height: int) -> np.ndarray:
    _, xc, yc, bw, bh = row
    return np.array(
        [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ],
        dtype=np.float32,
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = next(i for i, value in enumerate(parts) if value.lower() == "images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_ground_truth(image_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    labels = []
    path = label_path(image_path)
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) >= 5:
            labels.append((int(values[0]), xywhn_to_xyxy(values[:5], width, height)))
    return labels


def match_predictions(result, ground_truth: list[tuple[int, np.ndarray]]) -> tuple[int, int, int]:
    predictions = []
    if result.boxes is not None:
        for cls_id, confidence, box in zip(
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
            result.boxes.xyxy.cpu().numpy(),
        ):
            predictions.append((int(cls_id), float(confidence), box))

    matched_gt: set[int] = set()
    true_positives = 0
    for cls_id, _, pred_box in sorted(predictions, key=lambda item: item[1], reverse=True):
        candidates = [
            (box_iou(pred_box, gt_box), index)
            for index, (gt_cls, gt_box) in enumerate(ground_truth)
            if index not in matched_gt and gt_cls == cls_id
        ]
        if candidates:
            best_iou, best_index = max(candidates)
            if best_iou >= 0.5:
                matched_gt.add(best_index)
                true_positives += 1

    return true_positives, len(predictions) - true_positives, len(ground_truth) - true_positives


def make_contact_sheet(images: list[Path], output_path: Path) -> None:
    tiles = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        canvas = np.full((330, 440, 3), 245, dtype=np.uint8)
        scale = min(420 / image.shape[1], 285 / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        x = (440 - resized.shape[1]) // 2
        y = 8 + (285 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        name_parts = path.stem.split("_")
        caption = " ".join(name_parts[1:3]) if len(name_parts) >= 3 else path.stem[:48]
        cv2.putText(
            canvas,
            caption,
            (10, 318),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)
    if not tiles:
        return
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    blank = np.full_like(tiles[0], 245)
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
    encoded.tofile(str(output_path))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = [Path(line.strip()) for line in args.test_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=args.confidence,
        imgsz=args.image_size,
        device=device,
        batch=4,
        verbose=False,
    )

    candidates = []
    for image_path, result in zip(image_paths, results):
        height, width = result.orig_shape
        ground_truth = read_ground_truth(image_path, width, height)
        tp, fp, fn = match_predictions(result, ground_truth)
        confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
        gt_classes = sorted({cls_id for cls_id, _ in ground_truth})
        score = tp * 10.0 - fp * 3.0 - fn * 2.0 + sum(confidences) / max(1, len(confidences))
        candidates.append(
            {
                "path": image_path,
                "result": result,
                "classes": gt_classes,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "score": score,
            }
        )

    # Prefer clean true-positive examples, then ensure every class is represented.
    clean = [item for item in candidates if item["tp"] > 0 and item["fp"] == 0 and item["fn"] == 0]
    ranked = sorted(clean or candidates, key=lambda item: (item["score"], item["tp"]), reverse=True)
    selected = []
    class_counts: Counter[int] = Counter()
    per_class = max(2, args.count // max(1, len(model.names)))
    by_class: dict[int, list] = defaultdict(list)
    for item in ranked:
        for cls_id in item["classes"]:
            by_class[cls_id].append(item)
    for cls_id in sorted(model.names):
        for item in by_class.get(cls_id, []):
            if item not in selected:
                selected.append(item)
                for included_class in item["classes"]:
                    class_counts[included_class] += 1
            if class_counts[cls_id] >= per_class or len(selected) >= args.count:
                break
    for item in ranked:
        if len(selected) >= args.count:
            break
        if item not in selected:
            selected.append(item)

    written = []
    for index, item in enumerate(selected, start=1):
        class_text = "-".join(model.names[cls_id] for cls_id in item["classes"]) or "Fruit"
        destination = args.output / f"inference_{index:02d}_{class_text}_{item['path'].stem}.jpg"
        plotted = item["result"].plot(conf=True, labels=True, line_width=3, font_size=16)
        cv2.imwrite(str(destination), plotted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(destination)

    make_contact_sheet(written, args.output / "customer_inference_contact_sheet.jpg")
    print(f"device={device}")
    print(f"test_images={len(image_paths)}")
    print(f"selected={len(written)}")
    print("class_coverage=" + ", ".join(f"{model.names[k]}:{v}" for k, v in sorted(class_counts.items())))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
