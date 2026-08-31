# Open-world extension contract and discovery stage

Fruit SSOD version 1 is a fixed, five-class detector for Apple, Banana,
Orange, Strawberry, and Pineapple. The post-Student discovery command in
`discovery.py` now provides the first offline open-world experiment without
changing that detector registry. It discovers Avocado, Blueberry, Cherry,
Kiwi, Mango, and Rockmelon from an unlabeled pool using augmentation
consistency, embeddings and clustering.

`contracts.py` provides only two future-facing boundaries:

- `UnknownProposalProvider` may receive an immutable
  `UnknownProposalRequest` (image path, known detections, source run ID, and
  evidence) and produce reviewable unknown-region candidates as
  `UnknownProposal` records. The first discovery command writes image-level
  cluster proposals separately and does not route them through the known
  detector consumer path.
- `ClassRegistryUpdateProposer` may create a review-only
  `ClassRegistryUpdateProposal`. It cannot assign a class ID or mutate the
  canonical registry.

Known detector outputs remain `DetectionRecord` values with
`is_unknown=False`. The PySide6 application remains disabled for unknown
clusters until the post-Student artifacts have passed the evidence checks.
The discovery result uses `Unknown cluster <id>` and post-hoc evaluation names;
it does not silently create runtime class IDs.

Any future implementation must define an approved dataset protocol, human
review, a new registry version, retraining, evaluation, and acceptance criteria
before it is presented as a production capability. The first discovery output
is therefore an exploratory open-world experiment, not a claim of fully
supervised semantic naming or box-level open-world mAP.
