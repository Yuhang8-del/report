from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.open_world.contracts import UnknownProposal
from fruit_ssod.open_world.pipeline import OpenWorldFruitPipeline
from fruit_ssod.open_world.incremental_adapter import ReviewedDetectionRecord


class _KnownDetector:
    def predict(self, _image: object, *, confidence: float | None = None, nms_iou: float | None = None):
        assert confidence == 0.5
        assert nms_iou == 0.6
        return (DetectionRecord(0, "Apple", 0.9, (1.0, 2.0, 10.0, 12.0), False, "student.pt"),)


class _ProposalProvider:
    def propose_unknowns(self, request):
        return (
            UnknownProposal(
                "proposal-1",
                request.evidence["image_id"],
                (20.0, 20.0, 40.0, 40.0),
                0.8,
                "objectness.pt",
                {"source": "fixture"},
            ),
        )


class _Clusterer:
    def assign(self, records, *, batch_size: int = 32):
        return (SimpleNamespace(proposal_id=records[0].proposal_id, cluster_id=4, candidate_name="Mango"),)


def test_pipeline_keeps_known_and_unknown_outputs_separate(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    pipeline = OpenWorldFruitPipeline(
        known_detector=_KnownDetector(), proposal_provider=_ProposalProvider(), clusterer=_Clusterer()
    )
    result = pipeline.predict(image)
    assert result.known_count == 1
    assert result.unknown_count == 1
    assert result.detections[0].kind == "known"
    assert result.detections[1].kind == "unknown"
    assert result.detections[1].cluster_id == 4
    assert result.detections[1].candidate_name == "Mango"


def test_pipeline_allows_reviewed_incremental_detections_in_unknown_exclusion(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")

    class IncrementalDetector:
        def predict(self, *_args, **_kwargs):
            return (
                ReviewedDetectionRecord(5, "Avocado", 0.95, (1.0, 2.0, 10.0, 12.0), False, "incremental.pt"),
            )

    class Provider:
        def propose_unknowns(self, request):
            assert request.known_detections[0].class_id == 5
            return ()

    result = OpenWorldFruitPipeline(known_detector=IncrementalDetector(), proposal_provider=Provider()).predict(image)
    assert result.known_count == 1
    assert result.detections[0].label == "Avocado"
