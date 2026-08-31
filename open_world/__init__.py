"""Open-world contracts plus the post-Student discovery experiment.

The known detector remains fixed at five classes.  The separate discovery
module performs an offline, evidence-bound novel-fruit experiment and never
mutates the runtime registry or known detector output contract.
"""

from fruit_ssod.open_world.contracts import (
    ClassRegistryUpdateProposal,
    ClassRegistryUpdateProposer,
    KnownDetectionContractError,
    UnknownProposal,
    UnknownProposalProvider,
    UnknownProposalRequest,
    assert_known_detection_results,
)

__all__ = [
    "ClassRegistryUpdateProposal",
    "ClassRegistryUpdateProposer",
    "KnownDetectionContractError",
    "UnknownProposal",
    "UnknownProposalProvider",
    "UnknownProposalRequest",
    "assert_known_detection_results",
]
