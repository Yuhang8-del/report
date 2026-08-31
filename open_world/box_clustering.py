"""Self-supervised feature clustering for localized unknown fruit boxes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from fruit_ssod.open_world.discovery import _ProjectionModel, _assign, _kmeans


@dataclass(frozen=True)
class BoxClusterInput:
    proposal_id: str
    image_id: str
    image_path: str
    xyxy: tuple[float, float, float, float]
    split: str
    novelty_score: float


@dataclass(frozen=True)
class BoxClusterAssignment:
    proposal_id: str
    image_id: str
    cluster_id: int
    cluster_distance: float
    novelty_score: float
    candidate_name: str | None = None


class _CropDataset(Dataset[torch.Tensor]):
    def __init__(self, records: Sequence[BoxClusterInput]) -> None:
        self.records = tuple(records)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            x1, y1, x2, y2 = record.xyxy
            left = max(0, min(image.width - 1, int(x1)))
            top = max(0, min(image.height - 1, int(y1)))
            right = max(left + 1, min(image.width, int(np.ceil(x2))))
            bottom = max(top + 1, min(image.height, int(np.ceil(y2))))
            crop = image.crop((left, top, right, bottom))
        return self.transform(crop)


def load_encoder(checkpoint: Path, device: torch.device) -> _ProjectionModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no self-supervised model state: {checkpoint}")
    base = models.resnet18(weights=None)
    base.fc = nn.Identity()
    model = _ProjectionModel(base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_crop_features(
    encoder: _ProjectionModel,
    records: Sequence[BoxClusterInput],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    if not records:
        return np.empty((0, 512), dtype=np.float32)
    loader = DataLoader(_CropDataset(records), batch_size=max(1, batch_size), shuffle=False, num_workers=0)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            features, _ = encoder(images.to(device))
            rows.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def normalize_features(features: np.ndarray, mean: np.ndarray) -> np.ndarray:
    normalized = features.astype(np.float32) - mean.astype(np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return normalized / np.maximum(norms, 1e-8)


def fit_box_cluster_model(
    records: Sequence[BoxClusterInput],
    *,
    encoder_checkpoint: Path,
    output_dir: Path,
    clusters: int = 6,
    seed: int = 42,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> tuple[BoxClusterAssignment, ...]:
    discovery = tuple(record for record in records if record.split == "discovery")
    if len(discovery) < clusters:
        raise ValueError("not enough localized discovery proposals to fit clusters")
    resolved_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    encoder = load_encoder(encoder_checkpoint, resolved_device)
    raw = extract_crop_features(encoder, discovery, device=resolved_device, batch_size=batch_size)
    feature_mean = raw.mean(axis=0, keepdims=True)
    features = normalize_features(raw, feature_mean)
    centers, labels = _kmeans(features, clusters, seed=seed)
    assigned_labels, distances = _assign(features, centers)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "box_cluster_model.npz",
        centers=centers.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        clusters=np.array([clusters], dtype=np.int64),
        seed=np.array([seed], dtype=np.int64),
    )
    assignments = tuple(
        BoxClusterAssignment(
            proposal_id=record.proposal_id,
            image_id=record.image_id,
            cluster_id=int(cluster),
            cluster_distance=float(distance),
            novelty_score=record.novelty_score,
        )
        for record, cluster, distance in zip(discovery, assigned_labels, distances)
    )
    (output_dir / "discovery_box_cluster_assignments.jsonl").write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in assignments),
        encoding="utf-8",
    )
    return assignments


class BoxClusterer:
    """Assign localized unknown boxes to frozen discovery clusters."""

    def __init__(
        self,
        *,
        encoder_checkpoint: Path,
        cluster_model: Path,
        candidate_names: dict[int, str] | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.encoder = load_encoder(encoder_checkpoint, self.device)
        payload = np.load(cluster_model)
        self.centers = payload["centers"].astype(np.float32)
        self.feature_mean = payload["feature_mean"].astype(np.float32)
        self.candidate_names = dict(candidate_names or {})

    def assign(self, records: Sequence[BoxClusterInput], *, batch_size: int = 32) -> tuple[BoxClusterAssignment, ...]:
        raw = extract_crop_features(self.encoder, records, device=self.device, batch_size=batch_size)
        features = normalize_features(raw, self.feature_mean)
        labels, distances = _assign(features, self.centers)
        return tuple(
            BoxClusterAssignment(
                proposal_id=record.proposal_id,
                image_id=record.image_id,
                cluster_id=int(cluster),
                cluster_distance=float(distance),
                novelty_score=record.novelty_score,
                candidate_name=self.candidate_names.get(int(cluster)),
            )
            for record, cluster, distance in zip(records, labels, distances)
        )
