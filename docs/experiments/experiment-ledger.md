# Fruit SSOD experiment ledger

This ledger is an evidence index, not a substitute for generated metrics.
Rows remain visible when an experiment fails, is running, or is deliberately
blocked by a protocol gate.

## Historical protocol: v4 expanded sources

| Run ID | Label budget | Fixed-test mAP50 | Status | Decision |
| --- | ---: | ---: | --- | --- |
| `supervised_v4_expanded_100_seed42` | 100% | 0.421867 | complete | Below the 0.85 credible-upper-reference screen; do not start the v4 Student. |

The complete v4 fixed-test evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v4\runs\supervised_v4_expanded_100_seed42\evaluations\test.json`.
The fixed v4 split fingerprint was
`869aba335c51fbec87d5eaa697aa588f29c135844d5c2e11181bf97fa10ac088`.

## Active protocol: v5 Strawberry-DS recovery

The v5 source union retains the same canonical five classes while adding the
audited public Strawberry-DS source. Its six source maturity annotations are
mapped only to canonical `Strawberry` (ID 3), with original category retained
in source provenance.

| Evidence | Location / value |
| --- | --- |
| Cleaned records | 7,266 boxes across 2,629 images |
| Sources | Open Images V7, Snacks Detection, Strawberry-DS |
| Data audit | zero critical findings |
| Fixed split fingerprint | `f20bd961cb89bd3f0e9deac2fd2ca40feb142ee4a44336fc56bf2eb8d9e391f6` |
| Audit evidence | `E:\fruit_ssod_runtime\artifacts_v5\data_audit\strawberry_ds_seed42\dataset_audit.json` |
| 100% run | `supervised_v5_strawberry_ds_100_seed42` – complete; fixed-test mAP50 0.435585 |

The v5 100% run completed with fixed-test mAP50 0.435585, mAP50-95 0.287611,
Precision 0.501724, Recall 0.447546 and F1 0.473089. This remains below the
0.85 credible-upper-reference screen and the 0.80 acceptance target. Therefore
the v5 Student, pseudo-label and ablation queue are not authorized under this
protocol: their outcomes could not credibly establish the requested acceptance
result. The fixed-test record is outside Git at
`E:\fruit_ssod_runtime\artifacts_v5\runs\supervised_v5_strawberry_ds_100_seed42\evaluations\test.json`.

An isolated, non-authoritative source breakdown reproduced the low aggregate
score (mAP50 0.435358) and localized it to Open Images V7 (0.419390) and
Snacks Detection (0.481067). Strawberry-DS reached 0.815707, but its held-out
subset contains only 16 images and cannot establish acceptance. The diagnostic
evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v5\diagnostics\v5_source_subset_metrics_isolated_v3\source_subset_metrics.json`.

## Active protocol: v6 Berremangra Orange recovery

The v6 protocol adds the audited CC BY 4.0 Berremangra Orange source. Its
YOLO polygons are converted to enclosing detection rectangles and then
re-cleaned and re-split with all prior sources.

| Evidence | Location / value |
| --- | --- |
| Cleaned records | 12,748 boxes across 3,198 images |
| Sources | Open Images V7, Snacks Detection, Strawberry-DS, Berremangra Orange |
| Data audit | zero critical findings |
| Fixed split fingerprint | `cc0dcac5426e53dde2cadf62068559e8203267dfd11c2ec9c36b966e524eae70` |
| Audit evidence | `E:\fruit_ssod_runtime\artifacts_v6\data_audit\berremangra_orange_seed42\dataset_audit.json` |
| 100% run | `run-933fa59deeb949ae9226915f65d985b3` — complete; fixed-test mAP50 0.504752 |

The v6 fixed test measured mAP50 0.504752, mAP50-95 0.340511, Precision
0.556167, Recall 0.521737 and F1 0.538402. Per-class AP50 was Apple 0.463613,
Banana 0.389405, Orange 0.829154, Strawberry 0.354621 and Pineapple 0.486967.
This is below the credible-upper-reference screen and the 0.80 acceptance
target, so no v6 pseudo-label, Student or ablation run is authorized. The
immutable fixed-test evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v6\runs\run-933fa59deeb949ae9226915f65d985b3\evaluations\test.json`.

## Completed protocol: v7 deepNIR recovery

The v7 protocol adds the audited deepNIR Zenodo record `6324489` (CC BY 4.0).
The source has malformed or placeholder YAML class names, so the importer only
uses its reviewed `apple`, `orange`, and `strawberry` directory names and
requires source class ID 0. It contributes 969 boxes on 161 images; it does
not infer missing Banana or Pineapple labels.

| Evidence | Location / value |
| --- | --- |
| Cleaned records | 13,717 boxes across 3,359 images |
| Sources | Open Images V7, Snacks Detection, Strawberry-DS, Berremangra Orange, deepNIR |
| Data audit | zero critical findings |
| Fixed split fingerprint | `63d05c1fd2a218775ccccf498520e803a159b20511297456fc04da90dec1600b` |
| Audit evidence | `E:\fruit_ssod_runtime\artifacts_v7\data_audit\deepnir_seed42\dataset_audit.json` |
| 100% configuration | `configs/experiments/v7_deepnir_100.yaml` |
| 100% run | `run-159d43828278471b9895dee6d9aba5a5` — complete; fixed-test mAP50 0.536426 |

The v7 fixed test measured mAP50 0.536426, mAP50-95 0.367796, Precision
0.607982, Recall 0.505749 and F1 0.552173. Per-class AP50 was Apple
0.468785, Banana 0.486339, Orange 0.817866, Strawberry 0.385027 and
Pineapple 0.524114. This is below the credible-upper-reference screen and the
0.80 acceptance target, so no v7 pseudo-label, Student or ablation run is
authorized. The immutable fixed-test evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v7\runs\run-159d43828278471b9895dee6d9aba5a5\evaluations\test.json`.

## Completed protocol: v8 Hugging Face crop/plant recovery

The v8 protocol adds the downloaded and locally indexed Hugging Face
`devshaheen/100_crops_plants_object_detection_25k_image_dataset` release.
Its dataset card lists MIT while its embedded upstream Roboflow YAML lists CC
BY 4.0; v8 records the more restrictive CC BY 4.0 terms. Only explicitly
reviewed `apple fruit`, `banana`, `orange`, `strawberry` and `pineapple`
labels are mapped to the canonical classes; all other source annotations are
recorded as rejections. The source's original train/validation/test layout is
discarded and never used as a project evaluation split.

| Evidence | Location / value |
| --- | --- |
| Downloaded archive | 1,526,663,271 bytes; SHA-256 `920e47bda4997e4af7aa773ed6088ceed0ebfeabfbaa58fbb5b61096e66df87b` |
| Imported source | 1,640 canonical boxes; 30,867 noncanonical annotations rejected |
| Source class coverage | Apple 326, Banana 331, Orange 327, Strawberry 327, Pineapple 329 boxes |
| Cleaned six-source union | 15,357 boxes across 4,608 images; quarantine count 0 |
| Duplicate review | 7 exact groups, 497 near groups, 499 split decisions |
| Data audit | zero critical findings |
| Fixed split fingerprint | `71f7443e35aa11e3cb968d4e870973393a962bf60235447f7eb7efca60f594fb` |
| Audit evidence | `E:\fruit_ssod_runtime\artifacts_v8\data_audit\hf_crop_plant_seed42\dataset_audit.json` |
| 100% configuration | `configs/experiments/v8_hf_crop_plant_100.yaml` |
| Configuration dry run | `run-0a7c973933fd4789b725f0fe21f32f67`; split fingerprint matches v8 |
| Technical launch record | `run-v8-hf-5c83b0fe123f4ec1a094c2ab70d35de9` — failed before training with the recorded `NoneType.write` launcher error; retained as a failed record |
| 100% run | `run-v8-hf-1b47ea31b5314f79a94a6395bca05e51` — complete; fixed-test mAP50 0.529994 |

The v8 fixed test measured mAP50 0.529994, mAP50-95 0.366929, Precision
0.582775, Recall 0.509710 and F1 0.543800. Per-class AP50 was Apple
0.483815, Banana 0.460367, Orange 0.810349, Strawberry 0.387484 and
Pineapple 0.507953. This is below both the credible-upper-reference screen
and the 0.80 acceptance target, so no v8 pseudo-label, Student or ablation
run is authorized. The immutable fixed-test evidence is outside Git at
`E:\fruit_ssod_runtime\artifacts_v8\runs\run-v8-hf-1b47ea31b5314f79a94a6395bca05e51\evaluations\test.json`.

## Rejected recovery attempt: duplicated Open Images V7 training snapshot

The project re-acquired the official Open Images training annotations and
metadata, then selected 200 source images per fruit class. The resulting
canonical manifest contained 903 images and 3,787 boxes. Before a split or
training run was created, its source-image IDs were compared with the existing
Open Images V7 contribution in v8: both sets contained 903 IDs, their
intersection was 903, and neither set had a unique ID. Therefore this snapshot
is a re-download of already used evidence, not a dataset expansion.

The attempted seven-source union is retained for audit only. Its 5,511 copied
files collapse to the same 4,608 source images as v8 when `(source,
source_image_id)` is made unique. No v9 split, supervised run, pseudo-label or
Student run is authorized from it. The canonical manifest is outside Git at
`E:\fruit_ssod_runtime\data\fruit_ssod\interim\open_images_v7_train_200_per_class_canonical_manifest.json`.

The selector now accepts an explicit previous-ID exclusion list.

## Completed protocol: v10 source-ID-disjoint Open Images recovery

Using that exclusion control, v10 selected 30 previously unused source images
per class from the same official Open Images training metadata. The candidate
contains 150 images and 542 boxes; its intersection with the 903 existing
Open Images source IDs is zero. All selected images downloaded successfully,
and the resulting seven-input union passed cleaning and audit before training.

| Evidence | Location / value |
| --- | --- |
| New Open Images contribution | 150 images, 542 boxes; 0 source-ID overlap with v8 |
| Cleaned union | 15,899 boxes across 4,758 images; quarantine count 0 |
| Duplicate review | 7 exact groups, 545 near groups, 547 split decisions |
| Data audit | zero critical findings |
| Fixed split protocol fingerprint | `83fb844b9a839ab3e3efab13dc55420c1708580f90e385f7756a1f702b287294` |
| 100% configuration | `configs/experiments/v10_open_images_fresh30_100.yaml` |
| Fixed-test mAP50 | `0.546244` |
| Fixed-test mAP50-95 | `0.384392` |
| Fixed-test precision / recall / F1 | `0.612699 / 0.515792 / 0.560084` |
| Fixed-test evidence | `E:\fruit_ssod_runtime\artifacts_v10\runs\run-v10-open-images-fresh30-100-seed42\evaluations\test.json` |

The completed v10 100% supervised upper-reference is below the 0.85
credible-upper-reference screen. Pseudo-label, Student and ablation work are
therefore not authorized from this protocol. The recovery proceeds only through
the separately audited v11 full-source supervised protocol.

## Stopped protocol: v11 full-source YOLOv8m upper-bound attempt

The v11 full-source run was stopped after 87 completed epochs because its
validation curve had plateaued. Its best validation mAP50 was `0.629700` at
epoch 70 and its final recorded epoch was `0.620320`. The run record uses the
status `failed` with problem `validation futility stop` so it cannot be
mistaken for a complete candidate. This is an intentional scientific stop,
not a launcher or hardware failure. No v11 fixed-test evaluation was run.

| Retained evidence | SHA-256 |
| --- | --- |
| `weights/best.pt` | `92a703861c0b7187e5b8dda82818e1c218523b64365810f75cc98b8a72e83267` |
| `weights/last.pt` | `c232b1c7c6e0c39d98f88d530283504e05f4af0bf50b8ad121c12d4774fd6a79` |
| `results.csv` | `58be3ebc2b2e3d00a8aca1f4adf7a0d9766eadb130086f556fac63c86bc5d6e7` |

The retained best checkpoint is authorized only as the declared v12
fine-tuning initialization. It does not satisfy the Teacher gate by itself.

## Active protocol: v12 full-label, balance and small-object recovery

The v12 supervised upper-bound restores the already-existing labels of the
1,721-image SSOD unlabeled pool only inside a separate diagnostic snapshot.
It does not alter the original label-budget artifacts. The snapshot contains
8,607 training images, while the protected validation (1,148), fixed test
(1,148), and pseudo-audit (574) memberships retain split fingerprint
`6f0728cc195e32fc7717c5bd1fc3a013c57175083f442a97c20d73716a9c5404`.

| v12 evidence | Value |
| --- | --- |
| Full-label membership SHA-256 | `29427d5455b34b5dcddfae24dea2ca43114ef60dec93ba2363dd367cb3b94565` |
| JPEG normalization | 278 copied images received only the missing standard EOI marker before their sealed digest was recorded |
| Pre/post framework audit | 10,903 image hashes and labels verified twice; 43,085 boxes; zero critical findings |
| Natural class-image exposure | Apple 1,999; Banana 1,651; Orange 2,583; Strawberry 1,408; Pineapple 1,084 |
| Balanced class-image exposure | Apple 2,583; Banana 2,583; Orange 2,589; Strawberry 2,583; Pineapple 2,583 |
| Balanced exposure count | 12,759; 3,737 unique images receive one or two capped repeats |
| Balanced membership SHA-256 | `eee3bcaa4efb990d00d9cbeee6d1010f7e39c7154d22179399db529c25d429fb` |
| Object-centric tile count | 4,017 train-only 512-pixel crops |
| Balanced plus tile exposure count | 16,776 |
| Tile membership SHA-256 | `659d4e09642caa63e9f25ff434ada51546381bd514977aab4a05f729dc295f07` |
| YOLO11m COCO initialization | Official `yolo11m.pt`, 41,876,504 bytes, SHA-256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`; local load verified |

The first v12 launch attempt, `run-v12-v8m-balanced-1024-ft40-seed42`,
was stopped during the dataset cache scan before epoch 1 because Ultralytics
repaired missing JPEG end markers in place. Its run record is terminal
`failed` with problem `sealed snapshot image mutation detected` and an
effective training epoch count of zero. It is retained as negative technical
evidence and cannot enter model comparison. The replacement `v12n` snapshot
adds missing EOI markers before hashing; an Ultralytics scan reported 12,759
loaded exposures, 8,607 unique images and zero corrupt records, after which the
full hash audit again passed unchanged.

Model screening remains validation-only. Exactly one frozen candidate may be
evaluated on the fixed test, and Task 13 remains blocked unless that result has
`mAP@0.5 >= 0.85`.

### Completed v12 balanced high-resolution candidate

`run-v12n-v8m-balanced-1024-ft40-seed42` completed as a validation-only v12
candidate. Its original process ended during epoch 10 without a terminal run
record; the retained `weights/last.pt` checkpoint represented nine completed
epochs (Ultralytics checkpoint epoch `8`). The run resumed from that exact
checkpoint using the unchanged configuration and split fingerprint; the resume
invocation is retained in `resume_history.jsonl` beside the run.

Ultralytics early stopping ended the run after 37 of 40 configured epochs
because no improvement was observed for ten epochs; its log identifies epoch
27 as the best checkpoint. The completed run record reports validation mAP50
`0.644387`, mAP50-95 `0.474313`, Precision `0.707121`, Recall `0.571896`, and
F1 `0.632360`. Per-class validation AP50 is Apple `0.582440`, Banana
`0.553037`, Orange `0.729382`, Strawberry `0.718333`, and Pineapple `0.638742`.

This is below the `0.85` validation screen used to protect the single fixed-test
evaluation. No fixed-test evaluation exists for this run, the protected test
membership remains sealed, and pseudo-label/Student work is not authorized
from this candidate.

### v12 validation-only source diagnosis

The independently materialized validation-source diagnostic is outside Git at
`E:\fruit_ssod_runtime\artifacts_v12\diagnostics\v12n_validation_sources_20260803\source_subset_metrics.json`.
It uses the completed v12 checkpoint and only the sealed validation membership;
it does not read or create a fixed-test evaluation artifact. This diagnostic is
not a formal acceptance metric.

The diagnostic localizes the dominant weakness to the more natural and mixed
scene sources: Open Images V7 (190 images, mAP50 `0.406022`) and Snacks
Detection (165 images, mAP50 `0.366554`). In contrast, the large controlled
Kaggle subset reaches `0.672470` on 612 validation images, while several
small single-domain sources are higher but too small to establish acceptance.
Within Open Images, Banana AP50 is `0.281169`; within Snacks, Orange AP50 is
`0.251360`. The next supervised recovery must therefore add and audit
source-disjoint natural/mixed-scene examples for these failure modes, rather
than merely repeating controlled images or starting pseudo-label training.

The v13 selection policy is therefore explicit rather than uniform: Apple
600, Banana 400, Orange 600, Strawberry 600 and Pineapple 5 requested
source-ID-disjoint Open Images training candidates. The lower Pineapple cap is
an observed availability constraint after excluding v11 IDs, not a removal of
Pineapple from the five-class protocol. The policy file is
`configs/experiments/v13_open_images_natural_class_caps.json`.

### Active v13 train-only natural-scene recovery

The official Open Images selection was downloaded through the reachable remote
server and transferred into the Windows-native runtime. Canonical conversion
produced 2,205 images and 10,637 annotations. Cleaning accepted all 10,637
annotations, with zero quarantine rows, zero exact groups and zero near groups
within the incremental source. During protected-split screening, 190 additions
were excluded as visual near-duplicates of the sealed v12 validation or test
members; the remaining 2,015 images (8,957 boxes) were appended only to the
balanced training list.

The resulting v13 view has 14,774 training exposures. Its audit reports zero
critical findings and preserves the 1,148-image validation list and
1,148-image fixed-test list by recorded SHA-256 values. Evidence is outside
Git at `E:\fruit_ssod_runtime\artifacts_v13\data_audits\v13_train_only_augmentation_seed42.json`;
its augmentation membership digest is
`6feb0bcbba92f1873b48c5fb2301d78403ebf7d4ee5a4551a1f69968a0aa6460`.

Validation-only training `run-v13-open-images-natural-1024-seed42` was then
started from the completed v12n best checkpoint on the Windows RTX 3080. It is
configured for 60 epochs with patience 15. No v13 fixed-test evaluation or
pseudo-label/Student experiment is authorized unless this candidate first
passes the `mAP@0.5 >= 0.85` validation screen.

The initial v13 process ended without writing a terminal run state after the
fifth completed epoch. The retained `weights/last.pt` was 155,552,040 bytes
and was resumed under the unchanged configuration and split fingerprint; the
recovery invocation is recorded in `resume_history.jsonl`. This preserves the
original run ID and avoids treating an interrupted attempt as a new candidate.

### Customer-authorized continuous v0 path: independent-pool Student

The expanded independent-pool Student run
`ssod-v0-opt-independent-pool-v2-teacher-seed42-r1` completed 60 epochs. Its
validation best row records mAP50 `0.381450` and mAP50-95 `0.232570`. The single
sealed fixed-test evaluation records mAP50 `0.2831098263`, mAP50-95
`0.1723647127`, Precision `0.4432110111`, Recall `0.2585118857`, and F1
`0.3265542988`.

The evaluation is bound to split fingerprint
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9`, 90 test
images, and checkpoint SHA-256
`4ef41e4c81e1fbdb2755d7acd425fd25fb1c58309dae7c2d78cc9540469beea7`. The
paired no-class-threshold audit reports post-filter Precision `0.65625` and
`refresh_allowed=false`; it is retained as exploratory evidence under the
customer-authorized continuous path.

The follow-up run
`ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r1` has been started
with a separate run ID. It must be compared using the same fixed-test protocol;
the 0.80/0.85 target screens remain reported criteria and do not block this
follow-up.

The first chained follow-up attempt (`...-r1`) terminated after its first
validation without writing a terminal state. It is retained and marked
`failed`; the cause is recorded as an interrupted detached process. The retry
(`...-r2`) uses explicit stdout/stderr files and a separate watcher so the
optimization remains observable and cannot overwrite the interrupted evidence.

### No-class-threshold combined Student retry r2 (completed 2026-08-05)

The retry `ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r2`
completed with early stopping after 31 recorded epochs. Its best validation
mAP50 was `0.41341` (epoch 14). The dedicated watcher then produced the sealed
fixed-test evaluation using the same 90-image test membership and split
fingerprint. The measured exploratory fixed-test metrics are mAP50
`0.2741157961`, mAP50-95 `0.1618724549`, Precision `0.3384724546`, Recall
`0.3613719988`, and F1 `0.3495475798`.

The checkpoint is
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r2/weights/best.pt`
with SHA-256
`12fbaac37ffa5674a6f4447df18fc62092fea19d1f0cbc3e3610684cf10e30a3`.
The evidence remains exploratory, does not pass the customer target, and is
retained as a separate comparable run rather than replacing the earlier r1
result. The run record, fixed-test JSON, dataset YAML hash, test-list hash and
raw confusion matrix are all present under the run directory.

### Natural-unresampled Student comparison (active)

Because r2's fixed-test mAP50 (`0.2741157961`) was below the earlier
independent-pool result (`0.2831098263`), the next exploratory comparison
keeps the same Teacher, expanded pseudo-label snapshot, split fingerprint and
fixed-test protocol while changing only Student exposure from `balanced_50_50`
to `natural_unresampled`. Configuration:
`configs/experiments/ssod_v0_opt_natural_unresampled_combined_v2_teacher_seed42.yaml`.
Run ID:
`ssod-v0-opt-natural-unresampled-combined-v2-teacher-seed42-r1`.
The run completed with early stopping after 42 recorded epochs. Its best
validation mAP50 was `0.44386` at epoch 32. The sealed fixed-test evaluation
measured mAP50 `0.2872889802`, mAP50-95 `0.1759981120`, Precision
`0.3459670219`, Recall `0.3586910823`, and F1 `0.3522141724`.

The checkpoint is
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-natural-unresampled-combined-v2-teacher-seed42-r1/weights/best.pt`
with SHA-256
`40873efeaf5dd2da18a2116bd6147e9b957c1f795bb4b3efeb00404e98c36f30`.
This is the strongest fixed-test Student result so far, improving on the
independent-pool result (`0.2831098263`) and r2 (`0.2741157961`) under the same
90-image test membership and split fingerprint. The result remains exploratory
and is now packaged as GUI candidate r3 without replacing earlier candidates.

The continuous queue has started the next independent run
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r1`, which retains the
same data and protocol while using learning rate `5e-5`, 80 epochs and patience
15. Its run directory and terminal evidence will be recorded separately.

The detached-chain attempt for this run stopped after its first validation
without a terminal state and is retained as interrupted evidence. It is not
used as a result. A fresh run
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2` is now running with
explicit stdout/stderr capture and a dedicated fixed-test watcher.

The continuous queue script is also active. After r2 has terminal training and
fixed-test evidence, it will start the seed-43 comparison
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed43-r1` with the same data,
pseudo-label snapshot and protocol. The queue uses separate logs and refuses
to overwrite an existing run directory.

### First-result package: natural low-learning-rate Student r2 (completed 2026-08-05)

The customer-authorized first-result run
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2` completed with early
stopping after 43 recorded epochs. Its best validation mAP50 was `0.45096` at
epoch 28. The fixed-test watcher produced a sealed evaluation on the same
90-image test membership and split fingerprint. The measured exploratory
fixed-test metrics are mAP50 `0.2985708107`, mAP50-95 `0.1723392126`, Precision
`0.3222840638`, Recall `0.3796603397`, and F1 `0.3486272603`.

The selected checkpoint is
`E:/fruit_ssod_runtime/artifacts_v15/runs/ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed42-r2/weights/best.pt`
with SHA-256
`f96cf93c4deeff9f93a89c467038ae3a9d79700b80b3196194efcd4098acbcb8`.
The evaluation records the dataset YAML hash, split-manifest hash, test-list
hash, checkpoint size and raw confusion matrix. This result is below the
customer target, but it is valid first-result evidence and does not block the
next optimization run.

The offline PySide6 export for this checkpoint is available at
`E:/fruit_ssod_runtime/artifacts_v15/exports/gui_v0_student_inference_r4`.
It contains the manifest, metadata, detections CSV, results JSON and three
annotated images; camera and open-world actions remain disabled as specified.

The serial queue has now started
`ssod-v0-opt-natural-low-lr-combined-v2-teacher-seed43-r1` with the same data,
pseudo-label snapshot and fixed-test protocol. Its training evidence is kept
under a separate run ID and log set; it does not overwrite the first-result
package.

### Customer-authorized v12 Teacher -> Student retry (completed 2026-08-05)

The non-overwriting retry
`ssod-v1-independent-openimages-v12teacher-seed42-r1` completed at epoch 22.
Its sealed fixed-test evaluation uses the independent-pool split fingerprint
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9` and the
90-image protected test membership. Exploratory fixed-test metrics are mAP50
`0.5477312199`, mAP50-95 `0.3756795960`, Precision `0.5923168542`, Recall
`0.5011102231`, and F1 `0.5429096043`.

The selected Student checkpoint is
`E:/fruit_ssod_runtime/artifacts_v17/runs/ssod-v1-independent-openimages-v12teacher-seed42-r1/weights/best.pt`
with SHA-256
`efca46b315ee4c3ee98281002412fe9f772d8117732d95950ad43a0468a9ed8a`.
The fixed-test JSON, dataset YAML hash, split-manifest hash, test-list hash and
raw confusion matrix are retained under the run directory. The offline PySide6
GUI candidate is exported at
`E:/fruit_ssod_runtime/artifacts_v17/exports/gui_v12teacher_student` and the
exploratory DOCX/PDF package is
`E:/fruit_ssod_runtime/artifacts_v17/exports/exploratory_v3_v12teacher_student_package`.
Camera and open-world actions remain disabled.

Completion released the expanded-data aggressive Teacher queue. Run
`supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1` emitted one
epoch with validation mAP50 `0.51448` and then exited before publishing a
checkpoint or terminal result. It is recorded as `failed` with its first-epoch
CSV and log retained as diagnostic evidence; it cannot overwrite the completed
Student evidence. The v2 fallback was then started separately on the same
8,738-image expansion.

### Aggressive Teacher best-checkpoint continuation queue (active)

The customer-authorized continuation is now automated by
`scripts/queue_student_after_best_aggressive_teacher.ps1`. It waits for
aggressive Teacher v1/v2 and the resumable v2-r1 recovery, compares their sealed fixed-test mAP50 values,
keeps the higher checkpoint, regenerates the independent-pool candidate and
Trust Filter artifacts, and starts the next Student without an historical
`mAP50 >= 0.80` or pseudo-audit precision stop. The protected pseudo-audit
report remains provenance evidence; `allow_below_precision_gate: true` is
explicitly bound in the exploratory Student config. If all aggressive v2
attempts exit before publishing a checkpoint, the queue explicitly falls back
to the completed, fixed-test-evaluated v12 checkpoint so delivery is not
blocked; this fallback is recorded as such and is not called a completed v2
run. The queue process is running in the Windows `fruit-ssod` Conda environment
and will publish the Student fixed-test JSON under a new non-overwriting run ID.

The first Student fallback invocation crashed in the NVIDIA driver during the
Ultralytics AMP check (`nvcuda64.dll`, Windows Application Error 1000,
exception `0xc0000409`). A second detached no-AMP attempt completed epoch 1
validation but was reaped before publishing weights. Both run records and logs
are retained as failed diagnostics. The current retry uses the same Student
config with `amp: false` in a foreground-managed process; it has published
`best.pt`, `last.pt` and `epoch0.pt` after epoch 1 and is continuing.

The foreground Student fixed-test hand-off is automated by
`scripts/evaluate_foreground_student_when_complete.ps1`. It waits for the
terminal Student record, evaluates the sealed 90-image fixed test, and writes
`evaluations/test.json`; the open-world and report watchers skip failed
diagnostic directories and consume this successful run only.

### Aggressive Teacher v2 recovery (historical diagnostic)

The original non-overwriting run
`supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2` emitted one
complete epoch and validation but exited before publishing a checkpoint. Its
record is now `failed`; the one-epoch log/CSV are retained and no metric from
this run is used as a Teacher checkpoint.

### Customer freeze: v2 directly to Student

The customer has frozen the Teacher search for this delivery cycle. Once v2
reaches a terminal record, it is evaluated and used directly to generate the
next pseudo labels and train Student. No v3 Teacher is queued before this
Student produces its fixed-test result; the v2 checkpoint and its Student form
the authoritative delivery chain for the current first-result request.

### Customer-authorized post-Student open-world discovery (queued)

After the v2-selected Student reaches a terminal record and its fixed-test
JSON is present, the independent watcher
`scripts/queue_open_world_after_student.ps1` will run the first open-world
experiment. It uses the locally available DeepNIR fruit folders outside the
known registry: Avocado, Blueberry, Cherry, Kiwi, Mango and Rockmelon. The
novel pool is split deterministically into an unlabeled discovery portion and
protected evaluation portion; Capsicum and Wheat are excluded from the fruit
claim.

The post-Student command performs augmentation-consistency self-supervised
adaptation, feature clustering, Student known-class confidence scoring and
protected post-hoc cluster metrics. It writes a separate immutable artifact
directory and does not change the five-class model registry. Runtime semantic
names are not claimed from clusters; the first result is reported as unknown
clusters with a protected evaluation mapping. The watcher is non-overwriting
and does not affect the running v2/Student queue.

### Aggressive Teacher v2 interruption and resumable recovery (active)

The original
`supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2` process
completed its first epoch and validation (validation mAP50 `0.53529`) but
exited before publishing `best.pt`, `last.pt` or a terminal run record. No
Python process or CUDA error remained. The record is marked `failed` with the
one-epoch CSV/log retained as diagnostic evidence; it is not used as a Teacher
checkpoint.

The non-overwriting recovery
`supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2-r1` repeated the
same one-epoch Windows exit: validation mAP50 `0.53529` was written, but no
`best.pt`/`last.pt` was published. It is marked `failed` with its CSV/log
retained as diagnostic evidence. The queue therefore selected the verified v12
fallback and is generating its pseudo labels now; the post-Student open-world
watcher remains attached to the eventual Student fixed-test evidence.

### First Student and open-world delivery chain (completed 2026-08-06)

The foreground-managed no-AMP Student recovery
`ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-noamp-fg-seed42`
completed by early stopping at epoch 16 and published a valid `weights/best.pt`.
Its sealed 90-image fixed test has mAP50 `0.5281147491`, mAP50-95
`0.36549631199`, Precision `0.6083260809`, Recall `0.4756262431`, and F1
`0.5338534584`. The checkpoint SHA-256 is
`f4f3553244416e370a5d1dff36d38e42c1b35d451d3153e4cb57774458b22c49` and the
test split fingerprint is
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9`.

The first background open-world attempt was retained as a failed diagnostic:
the complete image list caused an approximately 19.22-GiB Ultralytics
allocation on the 10-GiB RTX 3080. `_known_confidences` was corrected to run
one image per `predict` call with `batch=1`; the open-world contract tests
remain green. The foreground rerun
`post_student_ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-noamp-fg-seed42-foreground-r3`
completed successfully over 639 DeepNIR images (Avocado, Blueberry, Cherry,
Kiwi, Mango and Rockmelon). Protected discovery metrics are purity `0.733333`,
NMI `0.595215`, ARI `0.510885`; holdout metrics are purity `0.720930`, NMI
`0.582059`, ARI `0.499890`. It produced 384 unknown candidates at threshold
`0.50`, with known-test false-positive rate `0.233333`. This is an image-level
Unknown-cluster demonstration; it does not add runtime class IDs or claim
box-level novel-class mAP.

The evidence-bound GUI/report package including the open-world appendix is
`E:/fruit_ssod_runtime/artifacts_v17/exports/exploratory_ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-noamp-fg-seed42-openworld-r3_package`.

### Independent Student optimization and paired open-world evidence (completed 2026-08-06)

The non-overwriting run `ssod-v2-student-opt-lr5e5-patience20-seed42` used the
verified v12 Teacher checkpoint, the same sealed pseudo-label snapshot and
membership protocol as the first Student. It completed 24 epochs under early
stopping. Validation-best mAP50 was `0.57259` at epoch 4. Its sealed fixed-test
metrics are mAP50 `0.5745562991`, mAP50-95 `0.3918541158`, Precision
`0.6094863158`, Recall `0.5147319347`, and F1 `0.5581159537`. The checkpoint
SHA-256 is
`7ec352b2035f2bdd2f5e8d457289644cf1f8aee028eec6e74f8eba8a4aab663a`, with the
same protected split fingerprint
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9`.

The paired open-world run
`post_student_ssod-v2-student-opt-lr5e5-patience20-seed42-openworld-r1`
processed 639 DeepNIR images from Avocado, Blueberry, Cherry, Kiwi, Mango and
Rockmelon: 510 discovery and 129 holdout. Discovery purity/NMI/ARI were
`0.737255/0.598963/0.516306`; holdout purity/NMI/ARI were
`0.720930/0.573840/0.496637`. It produced 361 image-level Unknown candidates
at threshold `0.50` (candidate rate `0.564945`). No known-test list was passed,
so known-test false-positive rate is not measured for this run. The output is
still an image-level Unknown-cluster demonstration and does not add runtime
class IDs or claim box-level novel-class mAP.

The updated GUI/report package is
`E:/fruit_ssod_runtime/artifacts_v17/exports/exploratory_ssod-v2-student-opt-lr5e5-patience20-seed42-openworld-r1_package-r2`.

### Formal supervised reference matrix (running)

The six committed reference configurations passed dry-run validation with
split fingerprint
`7653d1f762053b90362803c8b2d25d287769de055fe11595565319f7fabe159c`. The
serial Windows queue is now running under the `fruit-ssod` Conda environment:
10%, 20% seeds 42/3407/2026, 40% and 100% references. The first row,
`supervised_10_seed42`, has started and is writing its run record. No formal
matrix aggregate or acceptance claim is made before all rows have terminal
records and fixed-test evidence.

The first row `supervised_10_seed42` has since completed all 100 epochs and
passed fixed-test evaluation. It measured mAP50 `0.1980433770`, mAP50-95
`0.1067535717`, Precision `0.2728901514`, Recall `0.2548331668` and F1
`0.2635527333`; its validation-best mAP50 was `0.22668`. The selected
checkpoint SHA-256 is
`09d5343705224b4b5821161c0e86edc7a5797ee9b4a7cf69fe50b3fbcfa1df48`.
The queue has started `supervised_20_seed42`; remaining rows and the formal
aggregate are still pending.

### Formal supervised reference matrix — second row complete

`supervised_20_seed42` completed 100 epochs and fixed-test evaluation. The
fixed-test mAP50 is `0.2127280641`; validation-best mAP50 is `0.29165`.
Evidence: `E:/fruit_ssod_runtime/artifacts_v17/runs/supervised_20_seed42/evaluations/test.json`.
The serial queue has advanced to `supervised_20_seed3407`; no aggregate or
acceptance claim is made until all six rows have terminal records and sealed
fixed-test evidence.

### Formal supervised reference matrix — third row complete

`supervised_20_seed3407` completed 100 epochs and fixed-test evaluation. The
fixed-test metrics are mAP50 `0.2120055147`, mAP50-95 `0.1204517277`,
Precision `0.2524731785`, Recall `0.3008121376` and F1 `0.2745310396`; its
validation-best mAP50 is `0.31669`. Evidence:
`E:/fruit_ssod_runtime/artifacts_v17/runs/supervised_20_seed3407/evaluations/test.json`.
The serial queue has advanced to `supervised_20_seed2026`; the aggregate is
still pending until all rows have terminal records and fixed-test evidence.

### Formal supervised reference matrix — fourth row complete

`supervised_20_seed2026` completed 100 epochs and fixed-test evaluation. The
fixed-test metrics are mAP50 `0.2031122730`, mAP50-95 `0.1162023378`,
Precision `0.3082580114`, Recall `0.2656225836` and F1 `0.2853565362`; its
validation-best mAP50 is `0.29495`. Evidence:
`E:/fruit_ssod_runtime/artifacts_v17/runs/supervised_20_seed2026/evaluations/test.json`.
The serial queue has advanced to `supervised_40_seed42`; the aggregate remains
pending until all six rows have terminal records and fixed-test evidence.

### Formal supervised reference matrix — fifth row complete

`supervised_40_seed42` completed 100 epochs and fixed-test evaluation. The
fixed-test metrics are mAP50 `0.2781973178`, mAP50-95 `0.1498394815`,
Precision `0.5180715397`, Recall `0.2761032301` and F1 `0.3602260635`; its
validation-best mAP50 is `0.36922`. Evidence:
`E:/fruit_ssod_runtime/artifacts_v17/runs/supervised_40_seed42/evaluations/test.json`.
The serial queue has advanced to the final `supervised_100_seed42` row;
aggregate and final report generation remain pending.

### Formal supervised reference matrix — complete

All six canonical rows have terminal `complete` records and sealed fixed-test
evidence. The published aggregate is
`E:/fruit_ssod_runtime/artifacts_v17/exports/supervised_matrix.json`; an
independent in-memory re-aggregation was byte-equivalent. The final
100%-label row `supervised_100_seed42` measured fixed-test mAP50
`0.3294846730`, mAP50-95 `0.1910059537`, Precision `0.4252348028`, Recall
`0.3759621510` and F1 `0.3990833724`. Matrix summary: 6 submitted, 6
complete, 0 failed. Its aggregate upper-bound diagnostic remains below the
historical 0.85 threshold, while the customer-authorized exploratory
Student/open-world path remains the delivery evidence. Final report assembly
and delivery QA are still pending.

### Evidence-bound delivery report sealed

The completed matrix, exploratory Student test and six-category open-world
evidence are packaged at
`E:/fruit_ssod_runtime/artifacts_v17/exports/final_delivery_report_v1`.
The package contains `summary.json`, `manifest.json`, `README.md`,
`delivery_evidence_report.docx` and `delivery_evidence_report.pdf`. DOCX
reopening, PDF header and all manifest SHA-256 checks passed. The report makes
no formal 0.80 acceptance claim and retains the upper-bound diagnostic as a
transparent limitation.

### Chinese GUI and requirements-aligned report release

The PySide6 demonstrator was refactored into a Chinese customer-facing
interface with file-based single-image, batch-folder and video workflows, an
experiment overview page, Chinese controls/status messages and a consistent
theme. The GUI regression suite passes 32 tests under Qt offscreen mode. The
requirements-aligned Word/PDF report is sealed at
`E:/fruit_ssod_runtime/artifacts_v17/exports/final_report_v2_r2`; it contains
2,112 narrative words, six figures, six tables and 21 hash-verified manifest
entries. It reports measured Teacher/Student/open-world evidence and keeps
the image-level novel-category discovery limitation explicit.

### v3-r3 Teacher, Student and post-Student open-category release

The domain-balanced Teacher run
`supervised-v3-domain-balanced-yolov8m-1024-seed42-r3` completed with fixed-test
mAP50 `0.6285094929` (mAP50-95 `0.4583980022`, Precision `0.6797439397`,
Recall `0.5687778888`, F1 `0.6193296971`). Its fixed-test evidence is under
`E:/fruit_ssod_runtime/artifacts_v17/runs/supervised-v3-domain-balanced-yolov8m-1024-seed42-r3/evaluations/test.json`.

The authorized Student run
`ssod-v3-teacher-r3-student-seed42` completed after 28 epochs under patience
20. Its best validation mAP50 was `0.578100` and fixed-test mAP50 was
`0.5322762374` (mAP50-95 `0.3748697875`, Precision `0.5631100209`, Recall
`0.5218187775`, F1 `0.5416786487`). The Student fixed-test evidence is under
`E:/fruit_ssod_runtime/artifacts_v17/runs/ssod-v3-teacher-r3-student-seed42/evaluations/test.json`.

The post-Student self-supervised open-category discovery completed over 639
independent images from Avocado, Blueberry, Cherry, Kiwi, Mango and Rockmelon.
Discovery purity/NMI/ARI were `0.7373/0.5952/0.5133`; protected holdout
purity/NMI/ARI were `0.7287/0.5828/0.5096`. The evidence is under
`E:/fruit_ssod_runtime/artifacts_v17/open_world/post_student_ssod-v3-teacher-r3-student-seed42/discovery_results.json`.
This is image-level discovery only; it does not add runtime class IDs or claim
box-level unknown-object mAP. The v3-r1 DOCX/PDF report and v3-r3 GUI export
are hash-validated under the paths recorded in the implementation plan.
