"""Tests for the deliberately disabled open-world extension boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fruit_ssod.detection import DetectionRecord
from fruit_ssod.open_world.contracts import (
    ClassRegistryUpdateProposal,
    KnownDetectionContractError,
    UnknownProposal,
    UnknownProposalRequest,
    assert_known_detection_results,
)


def _known_detection() -> DetectionRecord:
    return DetectionRecord(
        class_id=0,
        class_name="Apple",
        confidence=0.91,
        xyxy=(1.0, 2.0, 20.0, 30.0),
        is_unknown=False,
        source_model="fixture.pt",
    )


def test_known_detector_results_are_explicitly_not_unknown() -> None:
    """The current known-class output boundary is fixed at is_unknown=False."""
    result = assert_known_detection_results((_known_detection(),))

    assert result[0].is_unknown is False


def test_known_result_boundary_rejects_non_detection_records() -> None:
    """Future unknown proposal data cannot leak into the version-1 output path."""
    with pytest.raises(KnownDetectionContractError, match="not a DetectionRecord"):
        assert_known_detection_results((object(),))  # type: ignore[arg-type]


def test_reserved_proposals_are_auditable_but_cannot_mutate_the_registry() -> None:
    """Contracts carry evidence and omit any class-ID allocation operation."""
    proposal = UnknownProposal(
        proposal_id="unknown-0001",
        image_id="fruit-image-9",
        xyxy=(2.0, 3.0, 40.0, 50.0),
        novelty_score=0.88,
        source_model="future-novelty-model",
        evidence={"run_id": "future-run", "image_sha256": "abc"},
    )
    update = ClassRegistryUpdateProposal(
        base_registry_version="1.0.0",
        requested_class_name="Mango",
        supporting_proposal_ids=(proposal.proposal_id,),
        rationale="Review candidates before designing a new data protocol.",
    )

    assert proposal.novelty_score == 0.88
    assert update.supporting_proposal_ids == ("unknown-0001",)
    assert not hasattr(update, "class_id")
    assert not hasattr(update, "apply")


def test_future_proposals_and_registry_updates_are_deeply_immutable() -> None:
    """Future proposals preserve auditable evidence and cannot be reassigned."""
    proposal = UnknownProposal(
        proposal_id="unknown-0001",
        image_id="fruit-image-9",
        xyxy=(2.0, 3.0, 40.0, 50.0),
        novelty_score=0.88,
        source_model="future-novelty-model",
        evidence={"run_id": "future-run", "image_sha256": "abc"},
    )
    request = UnknownProposalRequest(
        image_path="fixtures/image-9.jpg",
        known_detections=(_known_detection(),),
        source_run_id="future-run",
        evidence={"image_sha256": "abc", "known_result_sha256": "def"},
    )
    update = ClassRegistryUpdateProposal(
        base_registry_version="1.0.0",
        requested_class_name="Mango",
        supporting_proposal_ids=(proposal.proposal_id,),
        rationale="Review candidates before designing a new data protocol.",
    )

    with pytest.raises(FrozenInstanceError):
        proposal.novelty_score = 0.1  # type: ignore[misc]
    with pytest.raises(TypeError):
        proposal.evidence["run_id"] = "replacement"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.source_run_id = "replacement"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.evidence["image_sha256"] = "replacement"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        update.requested_class_name = "Papaya"  # type: ignore[misc]
