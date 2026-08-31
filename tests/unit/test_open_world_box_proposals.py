from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fruit_ssod.detection.types import DetectionRecord
from fruit_ssod.open_world.box_metrics import UnknownEvaluationSample, UnknownGroundTruth, evaluate_unknown_boxes
from fruit_ssod.open_world.box_proposals import UltralyticsObjectnessProposalProvider, box_iou
from fruit_ssod.open_world.contracts import UnknownProposalRequest


class _Value:
    def __init__(self, value: object) -> None:
        self._value = value

    def detach(self) -> "_Value":
        return self

    def cpu(self) -> "_Value":
        return self

    def tolist(self) -> object:
        return self._value


class _ObjectnessModel:
    def predict(self, **_kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            xyxy=_Value([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 160.0, 160.0]]),
            conf=_Value([0.9, 0.8]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_iou_and_known_exclusion(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    known = DetectionRecord(0, "Apple", 0.95, (8.0, 8.0, 52.0, 52.0), False, "student.pt")
    request = UnknownProposalRequest(
        image_path=image,
        known_detections=(known,),
        source_run_id="test-run",
        evidence={"image_id": "image-1"},
    )
    provider = UltralyticsObjectnessProposalProvider(model=_ObjectnessModel(), known_iou_threshold=0.35)
    proposals = provider.propose_unknowns(request)
    assert len(proposals) == 1
    assert proposals[0].xyxy == (100.0, 100.0, 160.0, 160.0)
    assert proposals[0].novelty_score == 0.8
    assert box_iou(proposals[0].xyxy, proposals[0].xyxy) == 1.0


def test_unknown_metrics_report_open_set_error(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    request = UnknownProposalRequest(
        image_path=image,
        known_detections=(),
        source_run_id="test-run",
        evidence={"image_id": "image-1"},
    )
    proposal = UltralyticsObjectnessProposalProvider(model=_ObjectnessModel()).propose_unknowns(request)[0]
    truth = UnknownGroundTruth("image-1", "Mango", proposal.xyxy)
    wrong_known = DetectionRecord(0, "Apple", 0.6, proposal.xyxy, False, "student.pt")
    metrics = evaluate_unknown_boxes(
        [UnknownEvaluationSample("image-1", (proposal,), (truth,), (wrong_known,))]
    )
    assert metrics["u_precision"] == 1.0
    assert metrics["u_recall"] == 1.0
    assert metrics["u_f1"] == 1.0
    assert metrics["u_ap50"] == 1.0
    assert metrics["a_ose"] == 1
