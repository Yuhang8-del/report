# Supervised reference experiments

Task 12 defines the fixed supervised reference matrix for the five canonical
fruit classes: Apple, Banana, Orange, Strawberry and Pineapple. It is a
reference for later semi-supervised comparisons, not evidence that a metric
target has already been reached.

## Fixed matrix

| Label budget | Seed | Experiment |
| --- | ---: | --- |
| 10% | 42 | `supervised_10_seed42` |
| 20% | 42 | `supervised_20_seed42` |
| 20% | 3407 | `supervised_20_seed3407` |
| 20% | 2026 | `supervised_20_seed2026` |
| 40% | 42 | `supervised_40_seed42` |
| 100% | 42 | `supervised_100_seed42` |

All six files are deterministic renderings of
`configs/experiments/supervised_reference_template.yaml`. The verification
tests reject an independently edited config. Every run must freeze the Task 8
split fingerprint and its effective configuration snapshot before training.

## Windows / Conda use

With the `fruit-ssod` Conda environment installed and both data-root variables
set, validate every configuration without starting training:

```powershell
.\scripts\run_supervised_matrix.ps1 -DryRun
```

Run the actual queue only after the data audit is clean:

```powershell
.\scripts\run_supervised_matrix.ps1
```

The launcher invokes Conda with an argument vector, not an interpolated shell
string. It trains one configuration at a time, evaluates its recorded fixed
test split, then calls `fruit_ssod.cli.aggregate_supervised_matrix`. It never
substitutes a different `--data` YAML for the fixed test protocol.

Dry-run records intentionally receive disposable UUID run IDs. They validate
the configuration without reserving the fixed actual IDs such as
`supervised_20_seed42`; a dry-run therefore cannot block the subsequent real
matrix execution or appear as its evidence.

If the default aggregate location is unsuitable, select a new, nonexistent
output explicitly:

```powershell
.\scripts\run_supervised_matrix.ps1 -AggregateOutput 'E:\fruit-artifacts\exports\supervised_matrix.json'
```

Existing aggregation JSON is intentionally not overwritten. Preserve original
run directories, `run_record.json`, validation results and `evaluations/test.json`.

## Evidence and decision gate

Aggregation includes every supplied run directory. Failed, dry-run, incomplete
and unreadable records are retained as rows with their diagnostic information;
they are never omitted to improve a summary.

Only a complete result with the required fixed run ID, the exact
`supervised_reference_v1` template identifier, matching budget/seed/name, split
fingerprint and frozen configuration provenance may influence the 100% upper
bound. A high score in any other visible row cannot mask a valid low 100%
reference result.

The 100% reference is screened using its fixed-primary-test `mAP@0.5`:

- If it is below `0.85`, the aggregate sets
  `data_quality_investigation_required: true`. Stop and investigate labels,
  duplicates, class mapping, splits and training setup before pseudo-label work.
- If its fixed-test metric is absent, the upper bound is **not** considered
  credible. Complete the fixed-test evaluation; do not use a validation metric
  as a replacement.
- The screen does not manufacture an accuracy claim. The project target
  remains a later machine-generated acceptance result.

## Current evidence and recovery decision

The original reference matrix is retained as the required final protocol, but
it is not yet complete. Two completed 100%-label diagnostic protocols failed
the credible-upper-reference screen:

| Protocol | Fixed-test mAP50 | Decision |
| --- | ---: | --- |
| v4 expanded Open Images V7 + Snacks Detection | 0.421867 | Do not begin Student SSOD. |
| v5 plus Strawberry-DS | 0.435585 | Do not begin Student SSOD. |
| v6 plus Berremangra Orange | 0.504752 | Do not begin Student SSOD. |
| v7 plus deepNIR | 0.536426 | Do not begin Student SSOD. |
| v8 plus Hugging Face crop/plant source | 0.529994 | Do not begin Student SSOD. |

The v5 result is an immutable formal evaluation from
`supervised_v5_strawberry_ds_100_seed42`; its split fingerprint is
`f20bd961cb89bd3f0e9deac2fd2ca40feb142ee4a44336fc56bf2eb8d9e391f6`.
The source-subset diagnostic found mAP50 values of 0.419390 for Open Images V7
and 0.481067 for Snacks Detection. Strawberry-DS produced 0.815707, but that
source contributes only 16 held-out images and is not evidence of five-class
acceptance.

This outcome is a data/protocol limitation, not a reason to relax the target
or report a pseudo-label gain. The next authorized experimental action is to
introduce a downloadable, licensed, five-class detection source, create a new
audited split fingerprint, and rerun the 100% upper-bound screen. The public
Kaggle candidate `lakshaytyagi01/fruit-detection` is CC0 with 8,479 YOLO
images. Its reviewed version-1 archive was subsequently acquired and SHA-256
verified (`b759435fc06e34cf900129dc3144535f38f1c6233247fe86faf862e1177545c5`),
alongside the CC BY 4.0 Zenodo Strawberry record `6126677` archive
(`d4664ccd6288e077934e175c281f6a8146da5bf194a1be0e96bca8336faf320c`). Both
archives were safely extracted and imported only after the v10 screen failed.
They form part of the separately audited v11 full-source protocol, not the v10
fixed split; v10 results remain immutable and directly comparable.

The v7 deepNIR recovery is now also complete. Its immutable fixed-test
evaluation, `run-159d43828278471b9895dee6d9aba5a5`, yielded mAP50 0.536426,
mAP50-95 0.367796, Precision 0.607982, Recall 0.505749 and F1 0.552173 on the
same five-class protocol. The evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v7\runs\run-159d43828278471b9895dee6d9aba5a5\evaluations\test.json`.
It improves the recovery diagnostic but remains below 0.85; this does not
authorize pseudo-label, Student or ablation work.

The v8 crop/plant recovery is also complete. Its immutable fixed-test
evaluation, `run-v8-hf-1b47ea31b5314f79a94a6395bca05e51`, yielded mAP50
0.529994, mAP50-95 0.366929, Precision 0.582775, Recall 0.509710 and F1
0.543800. The evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v8\runs\run-v8-hf-1b47ea31b5314f79a94a6395bca05e51\evaluations\test.json`.
It does not meet the 0.85 credible-upper-reference screen, so the planned
pseudo-label, Student and ablation experiments remain unauthorized.

The v10 source-ID-disjoint recovery is complete. Its fixed-test evidence at
`E:\fruit_ssod_runtime\artifacts_v10\runs\run-v10-open-images-fresh30-100-seed42\evaluations\test.json`
reports mAP50 0.546244, mAP50-95 0.384392, Precision 0.612699, Recall
0.515792 and F1 0.560084. It also fails the 0.85 screen, so no semi-supervised
run may use v10 as its Teacher.

The following v11 YOLOv8m expansion was stopped for validation futility after
87 epochs. Its best validation mAP50 was 0.629700 at epoch 70; because the run
was not a frozen candidate, no fixed-test evaluation was performed and it
cannot authorize semi-supervised work. Its retained `best.pt` is used only as
the declared initialization for the v12 high-resolution fine-tune.

The active v12 recovery is a separate fully labelled diagnostic upper-bound.
It restores 1,721 already-available hidden training labels to the 6,886-image
train pool, producing 8,607 labelled training images without changing the
1,148-image validation, 1,148-image fixed test, or 574-image pseudo-audit
memberships. A deterministic capped sampler changes class-image exposure from
`1999/1651/2583/1408/1084` to `2583/2583/2589/2583/2583` in canonical class
order. A separate training-only view adds 4,017 object-centric 512-pixel crops.
These are data/training views, not reported results. Candidate selection must
use validation only before a single fixed-test gate evaluation.

The initial v12 cache scan exposed 278 JPEG copies without a terminal EOI
marker; Ultralytics would repair and overwrite them. That launch was stopped
before epoch 1 and is a failed technical record, not an experiment result. The
replacement `v12n` materializer adds only the missing marker before sealing
image hashes. Its complete audit verified 10,903 images, 10,903 labels and
43,085 boxes with zero critical findings. A subsequent Ultralytics scan found
zero corrupt images, and the repeated post-scan hash audit also passed.
