"""Runtime composition of known detections, unknown boxes and box clusters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from fruit_ssod.detection.adapter import DetectorAdapter
from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.open_world.box_clustering import BoxClusterInput, BoxClusterer
from fruit_ssod.open_world.contracts import (
    KnownDetectionContractError,
    UnknownProposal,
    UnknownProposalProvider,
    UnknownProposalRequest,
)


@dataclass(frozen=True)
class ReviewedProposalRequest:
    """Request view for detections validated by the append-only V2 registry."""

    image_path: Path
    known_detections: tuple[object, ...]
    source_run_id: str
    evidence: Mapping[str, str]


@dataclass(frozen=True)
class OpenWorldDetection:
    kind: str
    xyxy: tuple[float, float, float, float]
    label: str
    score: float
    class_id: int | None = None
    cluster_id: int | None = None
    candidate_name: str | None = None
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"known", "unknown"}:
            raise ValueError("kind must be known or unknown")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")


@dataclass(frozen=True)
class OpenWorldInferenceResult:
    image_path: str
    image_id: str
    detections: tuple[OpenWorldDetection, ...]
    known_count: int
    unknown_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "image_id": self.image_id,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
            "detections": [asdict(item) for item in self.detections],
        }


class ProposalClusterer(Protocol):
    def assign(self, records: tuple[BoxClusterInput, ...], *, batch_size: int = 32): ...


class OpenWorldFruitPipeline:
    """Detect known fruits and independently retain unclaimed objectness boxes."""

    def __init__(
        self,
        *,
        known_detector: DetectorAdapter,
        proposal_provider: UnknownProposalProvider,
        clusterer: BoxClusterer | ProposalClusterer | None = None,
        known_confidence: float = 0.50,
        known_nms_iou: float = 0.60,
        source_run_id: str = "open-world-runtime-v1",
    ) -> None:
        self.known_detector = known_detector
        self.proposal_provider = proposal_provider
        self.clusterer = clusterer
        self.known_confidence = known_confidence
        self.known_nms_iou = known_nms_iou
        self.source_run_id = source_run_id

    def predict(self, image_path: str | Path) -> OpenWorldInferenceResult:
        path = Path(image_path).resolve(strict=True)
        image_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        known = self.known_detector.predict(
            path,
            confidence=self.known_confidence,
            nms_iou=self.known_nms_iou,
        )
        request_values = {
            "image_path": path,
            "known_detections": known,
            "source_run_id": self.source_run_id,
            "evidence": {"image_id": image_id, "runtime": "OpenWorldFruitPipeline"},
        }
        try:
            request = UnknownProposalRequest(**request_values)
        except KnownDetectionContractError:
            # The immutable V1 contract intentionally rejects IDs above four.
            # Incremental detections have already been validated against the
            # reviewed append-only registry by their dedicated adapter.
            if not all(
                hasattr(item, "xyxy") and hasattr(item, "class_id") and hasattr(item, "class_name")
                for item in known
            ):
                raise
            request = ReviewedProposalRequest(**request_values)
        proposals = self.proposal_provider.propose_unknowns(
            request  # type: ignore[arg-type]
        )
        cluster_by_proposal: dict[str, object] = {}
        if self.clusterer is not None and proposals:
            inputs = tuple(
                BoxClusterInput(
                    proposal_id=item.proposal_id,
                    image_id=image_id,
                    image_path=str(path),
                    xyxy=item.xyxy,
                    split="runtime",
                    novelty_score=item.novelty_score,
                )
                for item in proposals
            )
            cluster_by_proposal = {item.proposal_id: item for item in self.clusterer.assign(inputs)}
        detections = tuple(self._known(item) for item in known) + tuple(
            self._unknown(item, cluster_by_proposal.get(item.proposal_id)) for item in proposals
        )
        return OpenWorldInferenceResult(
            image_path=str(path),
            image_id=image_id,
            detections=detections,
            known_count=len(known),
            unknown_count=len(proposals),
        )

    @staticmethod
    def _known(record: DetectionRecord) -> OpenWorldDetection:
        return OpenWorldDetection(
            kind="known",
            xyxy=record.xyxy,
            label=record.class_name,
            score=record.confidence,
            class_id=record.class_id,
        )

    @staticmethod
    def _unknown(proposal: UnknownProposal, assignment: object | None) -> OpenWorldDetection:
        cluster_id = getattr(assignment, "cluster_id", None)
        candidate_name = getattr(assignment, "candidate_name", None)
        label = f"Unknown Cluster {cluster_id}" if cluster_id is not None else "Unknown fruit"
        if candidate_name:
            label += f" / {candidate_name}?"
        return OpenWorldDetection(
            kind="unknown",
            xyxy=proposal.xyxy,
            label=label,
            score=proposal.novelty_score,
            cluster_id=cluster_id,
            candidate_name=candidate_name,
            proposal_id=proposal.proposal_id,
        )
