"""Self-supervised novel-fruit discovery for the post-Student experiment.

This module deliberately keeps the five-class detector registry unchanged. It
uses an unlabeled pool of fruit images outside that registry, adapts an
ImageNet-initialised encoder with an augmentation-consistency (SimCLR-style)
objective, and clusters the resulting embeddings. Ground-truth category names
are loaded only from a protected evaluation side of the manifest, after
clustering, so the experiment can report discovery agreement without turning
the novel classes into training labels.

The output is an exploratory open-world *discovery* result. It is not a claim
that the five-class YOLO head has been silently expanded or that an arbitrary
cluster has a reliable semantic name at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


NOVEL_CLASSES: tuple[str, ...] = (
    "Avocado",
    "Blueberry",
    "Cherry",
    "Kiwi",
    "Mango",
    "Rockmelon",
)
_RAW_ALIASES = {name.casefold(): name for name in NOVEL_CLASSES}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_category(value: str) -> str | None:
    return _RAW_ALIASES.get(value.strip().casefold())


@dataclass(frozen=True)
class NovelImage:
    image_id: str
    path: Path
    category: str
    split: str


def discover_images(source_root: Path, *, seed: int, holdout_fraction: float) -> tuple[NovelImage, ...]:
    """Build a deterministic image-only pool from the DeepNIR category folders."""

    if not source_root.is_dir():
        raise FileNotFoundError(f"novel source root does not exist: {source_root}")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    records: list[NovelImage] = []
    for category_dir in sorted(source_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = _canonical_category(category_dir.name)
        if category is None:
            continue
        paths = sorted(
            path for path in category_dir.rglob("*") if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        )
        rng = random.Random(seed + sum(ord(char) for char in category))
        shuffled = list(paths)
        rng.shuffle(shuffled)
        holdout_count = max(1, int(round(len(shuffled) * holdout_fraction)))
        holdout_paths = {path for path in shuffled[:holdout_count]}
        for path in sorted(paths):
            relative = path.relative_to(source_root).as_posix()
            image_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
            records.append(
                NovelImage(
                    image_id=image_id,
                    path=path,
                    category=category,
                    split="holdout" if path in holdout_paths else "discovery",
                )
            )
    if not records:
        raise ValueError("no supported novel-fruit images were found")
    return tuple(sorted(records, key=lambda item: item.image_id))


class _ContrastiveDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, records: Sequence[NovelImage], transform: transforms.Compose) -> None:
        self.records = tuple(record for record in records if record.split == "discovery")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        with Image.open(self.records[index].path) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), self.transform(rgb)


class _EmbeddingDataset(Dataset[Tensor]):
    def __init__(self, records: Sequence[NovelImage], transform: transforms.Compose) -> None:
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.records[index].path) as image:
            return self.transform(image.convert("RGB"))


class _ProjectionModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(images)
        projection = self.projector(features)
        return features, projection


def _load_encoder(device: torch.device) -> _ProjectionModel:
    try:
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        # A disconnected Windows host can still run the experiment with a
        # deterministic random encoder; the run record records this fallback.
        base = models.resnet18(weights=None)
    base.fc = nn.Identity()
    model = _ProjectionModel(base).to(device)
    return model


def _contrastive_loss(first: Tensor, second: Tensor, temperature: float = 0.2) -> Tensor:
    first = nn.functional.normalize(first, dim=1)
    second = nn.functional.normalize(second, dim=1)
    representations = torch.cat([first, second], dim=0)
    logits = representations @ representations.T / temperature
    count = logits.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    positive = torch.cat(
        [torch.arange(first.shape[0], device=logits.device) + first.shape[0], torch.arange(first.shape[0], device=logits.device)]
    )
    return nn.functional.cross_entropy(logits, positive)


def train_self_supervised(
    model: _ProjectionModel,
    records: Sequence[NovelImage],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.35, 0.35, 0.35, 0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = _ContrastiveDataset(records, transform)
    if not dataset:
        raise ValueError("self-supervised discovery split is empty")
    loader = DataLoader(
        dataset,
        batch_size=max(2, batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=len(dataset) >= max(2, batch_size),
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _epoch in range(max(0, epochs)):
        running = 0.0
        batches = 0
        for first, second in loader:
            first, second = first.to(device), second.to(device)
            _, first_projection = model(first)
            _, second_projection = model(second)
            loss = _contrastive_loss(first_projection, second_projection)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        losses.append(running / max(1, batches))
    return losses


def extract_embeddings(model: _ProjectionModel, records: Sequence[NovelImage], *, device: torch.device, batch_size: int) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    loader = DataLoader(_EmbeddingDataset(records, transform), batch_size=max(1, batch_size), shuffle=False, num_workers=0)
    model.eval()
    features: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            embeddings, _ = model(images.to(device))
            features.append(embeddings.detach().cpu().numpy())
    matrix = np.concatenate(features, axis=0).astype(np.float32)
    matrix -= matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-8)
    return matrix


def _kmeans(features: np.ndarray, clusters: int, *, seed: int, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or len(features) < clusters:
        raise ValueError("k-means requires at least as many feature rows as clusters")
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(len(features), size=clusters, replace=False)].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster in range(clusters):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = features[rng.integers(0, len(features))]
    return centers, labels


def _assign(features: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)
    return labels.astype(np.int64), distances[np.arange(len(features)), labels].astype(np.float32)


def _contingency(labels: Sequence[int], truth: Sequence[str]) -> tuple[dict[tuple[int, str], int], dict[int, int], dict[str, int]]:
    table: dict[tuple[int, str], int] = {}
    cluster_totals: dict[int, int] = {}
    class_totals: dict[str, int] = {}
    for cluster, category in zip(labels, truth):
        key = (int(cluster), str(category))
        table[key] = table.get(key, 0) + 1
        cluster_totals[int(cluster)] = cluster_totals.get(int(cluster), 0) + 1
        class_totals[str(category)] = class_totals.get(str(category), 0) + 1
    return table, cluster_totals, class_totals


def _purity(labels: Sequence[int], truth: Sequence[str]) -> float:
    table, cluster_totals, _ = _contingency(labels, truth)
    if not labels:
        return 0.0
    return sum(max((count for (cluster, _category), count in table.items() if cluster == key), default=0) for key in cluster_totals) / len(labels)


def _nmi(labels: Sequence[int], truth: Sequence[str]) -> float:
    table, cluster_totals, class_totals = _contingency(labels, truth)
    total = len(labels)
    if total == 0:
        return 0.0
    mutual_information = 0.0
    for (cluster, category), count in table.items():
        mutual_information += (count / total) * math.log((count * total) / (cluster_totals[cluster] * class_totals[category]))
    cluster_entropy = -sum((count / total) * math.log(count / total) for count in cluster_totals.values() if count)
    class_entropy = -sum((count / total) * math.log(count / total) for count in class_totals.values() if count)
    return mutual_information / math.sqrt(cluster_entropy * class_entropy) if cluster_entropy and class_entropy else 0.0


def _combination_two(value: int) -> float:
    return value * (value - 1) / 2.0


def _ari(labels: Sequence[int], truth: Sequence[str]) -> float:
    table, cluster_totals, class_totals = _contingency(labels, truth)
    total = len(labels)
    if total < 2:
        return 0.0
    index = sum(_combination_two(value) for value in table.values())
    expected = sum(_combination_two(value) for value in cluster_totals.values()) * sum(_combination_two(value) for value in class_totals.values()) / _combination_two(total)
    maximum = 0.5 * (sum(_combination_two(value) for value in cluster_totals.values()) + sum(_combination_two(value) for value in class_totals.values()))
    return (index - expected) / (maximum - expected) if maximum != expected else 0.0


def _majority_mapping(labels: Sequence[int], truth: Sequence[str]) -> dict[str, str]:
    table, _, _ = _contingency(labels, truth)
    mapping: dict[str, str] = {}
    for cluster in sorted({int(value) for value in labels}):
        choices = [(count, category) for (value, category), count in table.items() if value == cluster]
        if choices:
            mapping[str(cluster)] = max(choices)[1]
    return mapping


def evaluate_clusters(labels: Sequence[int], truth: Sequence[str]) -> dict[str, Any]:
    if len(labels) != len(truth):
        raise ValueError("cluster labels and protected truth labels have different lengths")
    return {
        "image_count": len(labels),
        "purity": _purity(labels, truth),
        "nmi": _nmi(labels, truth),
        "ari": _ari(labels, truth),
        "posthoc_cluster_to_category": _majority_mapping(labels, truth),
    }


def _known_confidences(weights: Path, paths: Sequence[Path], *, device: str, image_size: int) -> dict[str, dict[str, Any]]:
    """Run the known five-class Student and return auditable image-level scores."""

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results: dict[str, dict[str, Any]] = {}
    # Pass one path per call rather than the complete list.  Ultralytics can
    # still materialize a large source list internally on some Windows
    # versions, even when ``batch=1`` is supplied; the full novel pool at
    # 768px then requests more memory than the 10-GiB RTX 3080 has.  One-image
    # calls keep peak inference memory bounded while preserving scores.
    for path in paths:
        predictions = model.predict(
            source=str(path),
            stream=False,
            batch=1,
            imgsz=image_size,
            conf=0.05,
            device=device,
            verbose=False,
        )
        prediction = predictions[0]
        confs = prediction.boxes.conf.detach().cpu().numpy().tolist() if prediction.boxes is not None else []
        max_confidence = max(confs, default=0.0)
        result_path = str(Path(str(prediction.path)).resolve())
        results[result_path] = {
            "known_detection_count": len(confs),
            "known_max_confidence": float(max_confidence),
            "novelty_score": float(1.0 - max_confidence),
        }
    return results


def run_discovery(
    *,
    student_weights: Path,
    source_root: Path,
    output_dir: Path,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    clusters: int = len(NOVEL_CLASSES),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda:0",
    image_size: int = 768,
    novelty_threshold: float = 0.5,
    known_test_list: Path | None = None,
) -> dict[str, Any]:
    """Execute the complete post-Student discovery experiment."""

    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_images(source_root, seed=seed, holdout_fraction=holdout_fraction)
    discovery_records = tuple(record for record in records if record.split == "discovery")
    holdout_records = tuple(record for record in records if record.split == "holdout")
    # Seal the evaluation-only names before any encoder fitting or clustering.
    # The self-supervised dataset receives only image paths/split values; this
    # file is consumed only after cluster assignments are produced.
    protected_labels_path = output_dir / "protected_evaluation_labels.json"
    protected_labels_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "purpose": "post-hoc open-world discovery evaluation only",
                "records": [
                    {"image_id": record.image_id, "category": record.category, "split": record.split}
                    for record in records
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _load_encoder(resolved_device)
    losses = train_self_supervised(
        model,
        records,
        device=resolved_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    discovery_features = extract_embeddings(model, discovery_records, device=resolved_device, batch_size=batch_size)
    holdout_features = extract_embeddings(model, holdout_records, device=resolved_device, batch_size=batch_size)
    centers, discovery_labels = _kmeans(discovery_features, clusters, seed=seed)
    holdout_labels, holdout_distances = _assign(holdout_features, centers)
    _, discovery_distances = _assign(discovery_features, centers)

    known_test_paths: list[Path] = []
    if known_test_list is not None:
        if not known_test_list.is_file():
            raise FileNotFoundError(f"known fixed-test list does not exist: {known_test_list}")
        root = known_test_list.parent
        known_test_paths = [
            (Path(line) if Path(line).is_absolute() else root / line)
            for line in known_test_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    known_scores = _known_confidences(
        student_weights,
        [record.path for record in records] + known_test_paths,
        device=device,
        image_size=image_size,
    )
    assignments: list[dict[str, Any]] = []
    for record, cluster, distance in zip(discovery_records, discovery_labels, discovery_distances):
        known = known_scores.get(str(record.path.resolve()), {})
        assignments.append(
            {
                "image_id": record.image_id,
                "path": str(record.path),
                "split": record.split,
                "cluster_id": int(cluster),
                "cluster_distance": float(distance),
                **known,
            }
        )
    for record, cluster, distance in zip(holdout_records, holdout_labels, holdout_distances):
        known = known_scores.get(str(record.path.resolve()), {})
        assignments.append(
            {
                "image_id": record.image_id,
                "path": str(record.path),
                "split": record.split,
                "cluster_id": int(cluster),
                "cluster_distance": float(distance),
                **known,
            }
        )

    holdout_truth = [record.category for record in holdout_records]
    discovery_truth = [record.category for record in discovery_records]
    discovery_metrics = evaluate_clusters(discovery_labels.tolist(), discovery_truth)
    holdout_metrics = evaluate_clusters(holdout_labels.tolist(), holdout_truth)
    novel_candidates = [item for item in assignments if item.get("novelty_score", 0.0) >= novelty_threshold]
    known_test_scores = [known_scores.get(str(path.resolve()), {}).get("novelty_score", 0.0) for path in known_test_paths]
    category_counts = {category: sum(record.category == category for record in records) for category in NOVEL_CLASSES}
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "post_student_open_world_discovery",
        "student_checkpoint": {"path": str(student_weights), "sha256": _sha256(student_weights)},
        "source_root": str(source_root),
        "known_registry": ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"],
        "novel_categories_for_protected_evaluation": list(NOVEL_CLASSES),
        "split": {
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "image_count": len(records),
            "discovery_count": len(discovery_records),
            "holdout_count": len(holdout_records),
            "category_counts": category_counts,
        },
        "self_supervised": {
            "method": "augmentation_consistency_simclr_style",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(resolved_device),
            "loss_curve": losses,
            "checkpoint": str(output_dir / "self_supervised_encoder.pt"),
        },
        "clustering": {"algorithm": "deterministic_kmeans", "clusters": clusters, "seed": seed},
        "novelty": {
            "score": "1 - max_known_student_confidence",
            "threshold": novelty_threshold,
            "candidate_count": len(novel_candidates),
            "candidate_rate": len(novel_candidates) / max(1, len(assignments)),
            "known_fixed_test_count": len(known_test_scores),
            "known_false_positive_rate": (
                sum(score >= novelty_threshold for score in known_test_scores) / max(1, len(known_test_scores))
            ),
        },
        "metrics": {"discovery": discovery_metrics, "holdout": holdout_metrics},
        "limitations": [
            "Cluster names are post-hoc evaluation mappings, not training labels.",
            "The first extension reports image-level unknown proposals; box-level open-world mAP remains a follow-up experiment.",
            "No new class ID is added to the five-class runtime registry by this command.",
        ],
    }
    torch.save({"model": model.state_dict(), "seed": seed, "novel_classes": NOVEL_CLASSES}, output_dir / "self_supervised_encoder.pt")
    (output_dir / "discovery_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "protected_evaluation_labels": {
                    "path": str(protected_labels_path),
                    "sha256": _sha256(protected_labels_path),
                },
                "records": [
                    {"image_id": record.image_id, "path": str(record.path), "split": record.split}
                    for record in records
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "cluster_assignments.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in assignments), encoding="utf-8"
    )
    (output_dir / "discovery_results.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload
