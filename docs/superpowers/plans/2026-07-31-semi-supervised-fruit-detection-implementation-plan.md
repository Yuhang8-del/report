# Semi-Supervised Fruit Detection Implementation Plan

> **Execution requirement:** Use `superpowers:executing-plans` to implement this plan task by task. Do not begin a later phase until the current phase verification gate passes.

**Goal:** Build a Windows-native research experiment and PySide6 demonstration prototype for five-class semi-supervised fruit object detection, then produce a real-results-based English Final Report in DOCX and PDF.

**Architecture:** A configuration-driven Python package prepares public datasets, trains a YOLOv8s supervised Teacher, generates auditable offline pseudo-labels, applies a reliability filter, trains a Student, evaluates all experiments, exposes a detector adapter to a PySide6 desktop GUI, and generates report-ready tables and figures from immutable run records.

**Primary stack:** Python 3.10, PyTorch with CUDA, Ultralytics YOLOv8, OpenCV, pandas, NumPy, PyYAML, Pillow, imagehash, PySide6, pytest, pytest-qt, python-docx, Matplotlib and Seaborn.

**Approved design:** `docs/superpowers/specs/2026-07-31-semi-supervised-fruit-detection-design.md`

## 1. Non-negotiable implementation rules

- Run natively on Windows; do not introduce WSL or Docker into the primary workflow.
- Use Apple, Banana, Orange, Strawberry and Pineapple with fixed IDs 0–4.
- Use Open Images V7 as the main experimental dataset.
- Use Fruits-360 only as auxiliary unlabeled data.
- Use FruitDet only as a separately reported external test.
- Keep validation, pseudo-audit and test labels inaccessible to training code.
- Treat `mAP@0.5 >= 0.80` and a three-point gain over the 20% supervised baseline as experiment targets, not fabricated guarantees.
- Do not implement or claim open-world detection in the first version.
- Do not add camera support.
- Do not create presentation slides, scripts or recordings.
- Generate report tables and figures from saved result files; never manually retype formal result values.
- Commit only project source and small metadata. Do not commit datasets, model weights, caches or generated videos.

## 2. Planned repository layout

```text
E:\bishe\fruit
├── configs
│   ├── project.yaml
│   ├── local.windows.example.yaml
│   ├── models
│   └── experiments
├── src\fruit_ssod
│   ├── cli
│   ├── config
│   ├── data
│   ├── detection
│   ├── pseudo
│   ├── training
│   ├── evaluation
│   ├── gui
│   ├── reporting
│   └── open_world
├── tests
│   ├── fixtures
│   ├── unit
│   ├── integration
│   └── gui
├── scripts
├── reports
│   └── final_report
├── docs
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

Runtime data and large artifacts live below the configured shared root:

```text
\\10.16.57.94\dataset2\lyg\detect_datasets\fruit_ssod
├── raw
├── interim
├── processed
├── manifests
├── pseudo_labels
├── runs
├── weights
└── exports
```

## Phase A — Environment and project foundation

### Task 1: Verify Windows, GPU and shared-storage prerequisites

**Files**

- Create: `scripts/preflight.ps1`
- Create: `docs/setup/windows-environment.md`
- Test: `tests/integration/test_environment_contract.py`

**Steps**

1. Add a failing environment-contract test for Python 3.10, a CUDA-capable PyTorch build, writable artifact storage and readable dataset storage.
2. Run:

   ```powershell
   python -m pytest tests/integration/test_environment_contract.py -v
   ```

   Expected before implementation: failure because the project environment and configuration do not exist.

3. Implement `preflight.ps1` to report:
   - selected Python executable and version;
   - `nvidia-smi` GPU name, driver and memory;
   - PyTorch version, CUDA runtime and `torch.cuda.is_available()`;
   - free space on local and configured shared roots;
   - UNC reachability and write test in a dedicated temporary folder;
   - actionable problem, likely cause and remediation for every failed check.
4. Document the approved shared paths:
   - Windows: `\\10.16.57.94\dataset2\lyg\detect_datasets`
   - server: `/mnt/dataset2/lyg/detect_datasets`
   - SSH reference: `linyugui@10.20.118.114:1022`
5. Run the preflight script without performing a dataset download.

**Verification gate**

- The RTX 3080 10GB is detected.
- Python 3.10 is selected explicitly instead of the installed Python 3.14 launcher default.
- The script either confirms shared-storage access or returns one clear blocking message explaining how to restore network/VPN/credential access.

**Commit**

```powershell
git add scripts/preflight.ps1 docs/setup/windows-environment.md tests/integration/test_environment_contract.py
git commit -m "chore: add Windows environment preflight"
```

### Task 2: Create the Python project and reproducible Conda environment

**Files**

- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/fruit_ssod/__init__.py`
- Create: `tests/unit/test_package_import.py`

**Steps**

1. Add a failing import test for `fruit_ssod`.
2. Create the recommended Conda environment with Python 3.10. A pre-existing Conda environment is also allowed only when its resolved package versions are captured in the project lock file:

   ```powershell
   conda create -n fruit-ssod python=3.10 -y
   conda run -n fruit-ssod python -m pip install --upgrade pip
   ```

3. Install a CUDA PyTorch wheel compatible with the current NVIDIA driver, then install the project requirements.
4. Pin the resolved package versions after the GPU smoke test passes.
5. Include at minimum:
   - `torch`, `torchvision`;
   - `ultralytics`;
   - `opencv-python`;
   - `numpy`, `pandas`, `pyyaml`;
   - `pillow`, `imagehash`;
   - `matplotlib`, `seaborn`;
   - `PySide6`;
   - `python-docx`;
   - `pytest`, `pytest-cov`, `pytest-qt`.
6. Ignore any optional local virtual environment, dataset directories, runs, weights, caches, generated reports and GUI exports.
7. Run:

   ```powershell
   conda run -n fruit-ssod python -m pytest tests/unit/test_package_import.py -v
   conda run -n fruit-ssod python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

**Verification gate**

- Package import test passes.
- CUDA reports `True`.
- GPU name reports `NVIDIA GeForce RTX 3080`.
- No original Word, PPTX, image or generated customer PDF is staged.

**Commit**

```powershell
git add pyproject.toml requirements.txt requirements-dev.txt .gitignore .env.example src/fruit_ssod/__init__.py tests/unit/test_package_import.py
git commit -m "build: bootstrap Python project"
```

### Task 3: Implement typed configuration and path validation

**Files**

- Create: `configs/project.yaml`
- Create: `configs/local.windows.example.yaml`
- Create: `src/fruit_ssod/config/models.py`
- Create: `src/fruit_ssod/config/loader.py`
- Create: `src/fruit_ssod/config/paths.py`
- Create: `tests/unit/test_config_loader.py`
- Create: `tests/unit/test_paths.py`

**Steps**

1. Write failing tests for:
   - environment-variable expansion;
   - UNC paths containing backslashes;
   - missing required keys;
   - non-existent dataset roots;
   - refusal to silently use the repository as a dataset root.
2. Define configuration fields for data root, artifact root, classes, image size, device, workers, seed and experiment name.
3. Support `FRUIT_SSOD_DATA_ROOT` and `FRUIT_SSOD_ARTIFACT_ROOT` overrides.
4. Return structured errors containing problem, cause and fix.
5. Run:

   ```powershell
   conda run -n fruit-ssod python -m pytest tests/unit/test_config_loader.py tests/unit/test_paths.py -v
   ```

**Commit**

```powershell
git add configs src/fruit_ssod/config tests/unit/test_config_loader.py tests/unit/test_paths.py
git commit -m "feat: add project configuration and path validation"
```

## Phase B — Dataset acquisition, cleaning and fixed protocols

### Task 4: Define the canonical class registry and annotation schema

**Files**

- Create: `configs/class_registry.json`
- Create: `src/fruit_ssod/data/schema.py`
- Create: `src/fruit_ssod/data/class_mapping.py`
- Create: `tests/unit/test_class_mapping.py`
- Create: `tests/fixtures/annotations/sample_annotations.json`

**Steps**

1. Write failing tests asserting fixed IDs 0–4 and rejecting unknown aliases unless explicitly mapped.
2. Define a canonical record with source, image ID, file path, width, height, class ID, `xyxy`, split, label status and license metadata.
3. Add source aliases without changing canonical names.
4. Ensure Pineapple remains in the main five-class registry even when an external dataset lacks it.
5. Run the unit tests.

**Commit**

```powershell
git add configs/class_registry.json src/fruit_ssod/data tests/unit/test_class_mapping.py tests/fixtures/annotations
git commit -m "feat: define fruit class and annotation schemas"
```

### Task 5: Implement Open Images V7 acquisition and conversion

**Files**

- Create: `src/fruit_ssod/data/open_images.py`
- Create: `src/fruit_ssod/cli/download_open_images.py`
- Create: `src/fruit_ssod/data/yolo_format.py`
- Create: `tests/unit/test_open_images_conversion.py`
- Create: `tests/integration/test_open_images_fixture_pipeline.py`
- Create: `docs/data/open-images.md`

**Steps**

1. Build a small synthetic Open Images CSV fixture containing valid boxes, out-of-range boxes and excluded flags.
2. Write failing conversion tests.
3. Implement resumable downloads, source-ID preservation and SHA-256 capture.
4. Filter `IsDepiction`, `IsInside` and `IsGroupOf`.
5. Convert normalized source boxes to validated YOLO labels.
6. Limit requested data by configured per-class caps, not by duplicating images.
7. Provide `--dry-run` and `--max-images` arguments.
8. Run the fixture pipeline first:

   ```powershell
   conda run -n fruit-ssod python -m pytest tests/unit/test_open_images_conversion.py tests/integration/test_open_images_fixture_pipeline.py -v
   ```

9. Run a real five-image-per-class smoke download only after shared storage passes Task 1.

**Commit**

```powershell
git add src/fruit_ssod/data/open_images.py src/fruit_ssod/data/yolo_format.py src/fruit_ssod/cli/download_open_images.py tests docs/data/open-images.md
git commit -m "feat: add Open Images acquisition pipeline"
```

### Task 6: Implement Fruits-360 and FruitDet importers

**Files**

- Create: `src/fruit_ssod/data/fruits360.py`
- Create: `src/fruit_ssod/data/fruitdet.py`
- Create: `src/fruit_ssod/cli/import_auxiliary_data.py`
- Create: `tests/unit/test_auxiliary_mapping.py`
- Create: `docs/data/auxiliary-datasets.md`

**Steps**

1. Write tests for source aliases, unsupported categories and FruitDet’s missing Pineapple category.
2. Import Fruits-360 images as unlabeled records only.
3. Import FruitDet boxes as external-test annotations only.
4. Save source pages, versions and license fields in the manifest.
5. Refuse to merge FruitDet into the primary test metric.
6. Run the importer tests and a two-image fixture smoke test.

**Commit**

```powershell
git add src/fruit_ssod/data/fruits360.py src/fruit_ssod/data/fruitdet.py src/fruit_ssod/cli/import_auxiliary_data.py tests/unit/test_auxiliary_mapping.py docs/data/auxiliary-datasets.md
git commit -m "feat: add auxiliary dataset importers"
```

### Task 7: Implement data cleaning, duplicate detection and quarantine

**Files**

- Create: `src/fruit_ssod/data/cleaning.py`
- Create: `src/fruit_ssod/data/deduplication.py`
- Create: `src/fruit_ssod/cli/clean_dataset.py`
- Create: `tests/unit/test_cleaning.py`
- Create: `tests/unit/test_deduplication.py`

**Steps**

1. Add fixtures for corrupt images, zero-area boxes, clipped boxes, exact duplicates and near duplicates.
2. Write failing tests for each case.
3. Implement decode checks, box correction policy, SHA-256 and perceptual hashes.
4. Create a quarantine manifest instead of silently deleting source data.
5. Apply split priority `test > validation > pseudo_audit > train`.
6. Run all cleaning and duplicate tests.

**Commit**

```powershell
git add src/fruit_ssod/data/cleaning.py src/fruit_ssod/data/deduplication.py src/fruit_ssod/cli/clean_dataset.py tests/unit/test_cleaning.py tests/unit/test_deduplication.py
git commit -m "feat: add dataset cleaning and deduplication"
```

### Task 8: Create deterministic splits and nested label budgets

**Files**

- Create: `src/fruit_ssod/data/splitting.py`
- Create: `src/fruit_ssod/cli/create_splits.py`
- Create: `tests/unit/test_splitting.py`
- Create: `tests/integration/test_no_split_leakage.py`

**Steps**

1. Write tests that prove:
   - no source ID or perceptual duplicate crosses splits;
   - 10% is contained in 20%, 20% in 40%, and 40% in 100%;
   - hidden labels cannot be opened through the unlabeled record interface;
   - seeds 42, 3407 and 2026 are deterministic.
2. Reserve approximately 5% of the original training pool for `pseudo_audit`.
3. Stratify budgets by available class presence while handling multi-label images deterministically.
4. Write split files and a fingerprint for each split.
5. Run:

   ```powershell
   conda run -n fruit-ssod python -m pytest tests/unit/test_splitting.py tests/integration/test_no_split_leakage.py -v
   ```

**Commit**

```powershell
git add src/fruit_ssod/data/splitting.py src/fruit_ssod/cli/create_splits.py tests/unit/test_splitting.py tests/integration/test_no_split_leakage.py
git commit -m "feat: add deterministic label-budget splits"
```

### Task 9: Build dataset audit and report-ready summaries

**Files**

- Create: `src/fruit_ssod/data/audit.py`
- Create: `src/fruit_ssod/cli/audit_dataset.py`
- Create: `src/fruit_ssod/reporting/dataset_figures.py`
- Create: `tests/integration/test_dataset_audit.py`

**Steps**

1. Add failing tests for missing classes, empty splits, duplicate hashes and illegal boxes.
2. Generate:
   - `data_manifest.csv`;
   - class and box counts by split and source;
   - label-budget membership summary;
   - source/license summary;
   - sample annotation montage;
   - machine-readable audit JSON.
3. Make any critical audit error return a non-zero process code.
4. Run the complete audit on fixture data, then on the real prepared dataset.

**Phase B verification gate**

- All five classes are present in the primary train, validation and test protocol.
- No duplicate crosses a split.
- Fixed test and pseudo-audit labels are sealed.
- Dataset audit has zero critical findings.

**Commit**

```powershell
git add src/fruit_ssod/data/audit.py src/fruit_ssod/cli/audit_dataset.py src/fruit_ssod/reporting/dataset_figures.py tests/integration/test_dataset_audit.py
git commit -m "feat: add dataset audit reporting"
```

## Phase C — Supervised detector baseline

### Task 10: Implement the detector adapter and Ultralytics backend

**Files**

- Create: `src/fruit_ssod/detection/types.py`
- Create: `src/fruit_ssod/detection/adapter.py`
- Create: `src/fruit_ssod/detection/ultralytics_backend.py`
- Create: `tests/unit/test_detection_types.py`
- Create: `tests/unit/test_detector_adapter.py`

**Steps**

1. Write failing tests for the unified detection record:
   - `class_id`;
   - `class_name`;
   - `confidence`;
   - `xyxy`;
   - `is_unknown`;
   - `source_model`.
2. Implement validation and JSON serialization.
3. Implement an Ultralytics adapter with dependency injection so tests use a fake model.
4. Set `is_unknown=False` for all first-version results.
5. Run unit tests without loading a real GPU model.

**Commit**

```powershell
git add src/fruit_ssod/detection tests/unit/test_detection_types.py tests/unit/test_detector_adapter.py
git commit -m "feat: add detector adapter contract"
```

### Task 11: Implement supervised training and evaluation runners

**Files**

- Create: `src/fruit_ssod/training/run_record.py`
- Create: `src/fruit_ssod/training/supervised.py`
- Create: `src/fruit_ssod/evaluation/detection_metrics.py`
- Create: `src/fruit_ssod/cli/train_supervised.py`
- Create: `src/fruit_ssod/cli/evaluate_model.py`
- Create: `configs/models/yolov8s_640.yaml`
- Create: `configs/experiments/supervised_20_seed42.yaml`
- Create: `tests/unit/test_run_record.py`
- Create: `tests/integration/test_training_dry_run.py`

**Steps**

1. Write tests for immutable run IDs, config snapshots, split fingerprints and result serialization.
2. Implement `--dry-run`, `--epochs`, `--batch`, `--device` and `--resume`.
3. Default to image size 640, AMP and batch 4.
4. Save `best.pt`, `last.pt`, training curves, raw validation output, environment details and the exact command.
5. Refuse to evaluate on test before a run is marked complete.
6. Execute a CPU/GPU fixture dry run, then a one-epoch real-data smoke run.

**Commit**

```powershell
git add src/fruit_ssod/training src/fruit_ssod/evaluation src/fruit_ssod/cli/train_supervised.py src/fruit_ssod/cli/evaluate_model.py configs/models configs/experiments tests
git commit -m "feat: add supervised training runner"
```

### Task 12: Run and validate supervised reference experiments

**Files**

- Create: `configs/experiments/supervised_10_seed42.yaml`
- Create: `configs/experiments/supervised_20_seed3407.yaml`
- Create: `configs/experiments/supervised_20_seed2026.yaml`
- Create: `configs/experiments/supervised_40_seed42.yaml`
- Create: `configs/experiments/supervised_100_seed42.yaml`
- Create: `scripts/run_supervised_matrix.ps1`
- Create: `docs/experiments/supervised-baselines.md`

**Steps**

1. Generate experiment configurations from one canonical template.
2. Validate all configs with `--dry-run`.
3. Run the 20% baseline for seeds 42, 3407 and 2026.
4. Run 10%, 40% and 100% references with seed 42.
5. Aggregate validation and fixed-test results without dropping failed runs.
6. Investigate data quality before continuing if the 100% upper bound has `mAP@0.5 < 0.85`.

#### Executed recovery addendum: v12 to v13 natural-scene expansion

The completed v12 balanced 1024-pixel YOLOv8m candidate reached validation
`mAP@0.5 = 0.644387` and early-stopped after 37 epochs; it did not qualify for
the fixed-test screen. A validation-only source diagnostic then identified the
lowest source subsets as Open Images V7 (`0.406022`) and Snacks Detection
(`0.366554`), rather than the small controlled-background sources. The fixed
validation, fixed test, and pseudo-audit memberships must therefore remain
unchanged during the recovery.

The v13 recovery is a train-only, source-ID-disjoint Open Images expansion.
Its selection policy is stored in
`configs/experiments/v13_open_images_natural_class_caps.json`: Apple 600,
Banana 400, Orange 600, Strawberry 600, and Pineapple 5 candidates. The lower
Pineapple count records post-exclusion availability and does not change the
five-class registry. Execute these gates in order:

1. Download and convert the selected official Open Images train images into a
   new immutable source root; retain download ledger and source metadata.
2. Build canonical annotations, clean them, and reject exact/near duplicates
   against all v12 protected members before any training list is created.
3. Materialize a fresh v13 training view that appends only accepted new train
   members while reusing byte-identical v12 validation and fixed-test lists.
4. Audit the new view, validate only on the retained validation set, and rank
   candidates without calling the fixed-test evaluator.
5. For the formal acceptance track, retain the documented
   `mAP@0.5 >= 0.85` validation/fixed-test screen. The customer-authorized
   first-result track may continue with the best reproducible checkpoint even
   when this screen is not passed; its metrics must be marked exploratory and
   must not be presented as a target-meeting result.

The reusable implementation of gate 3 is
`src/fruit_ssod/data/train_only_augmentation.py` with the
`materialize_train_only_augmentation` CLI. It accepts either the sealed v12
snapshot or its deterministic balanced training view, records hashes of the
reused validation/test lists, and fails closed if an addition reuses a sealed
source ID or exact/perceptual duplicate of a protected image.

**Execution status:** Open Images selection/download, canonical conversion and
cleaning are complete: 2,205 images and 10,637 annotations were received;
2,015 source-ID-disjoint, protected-duplicate-screened images were accepted
for the v13 train-only view. Its independent audit reports zero critical
findings while retaining the v12 protected lists. The validation-only v13
YOLOv8m 1024 candidate has been launched from the v12n best checkpoint; no
fixed-test evaluation has been requested or executed.

#### Customer-authorized exploratory SSOD continuation

The customer authorized progression to the next stage without requiring this
v13 recovery attempt to pass the `0.80`/`0.85` screening thresholds. This is
an execution exception only: it does not change the final acceptance targets,
does not qualify a model for the fixed-test screen, and must not be presented
as a target-meeting result.

- The v13 post-resume `best.pt` was independently revalidated and found
  unusable, so it is not used as a Teacher.
- The verified v12n `best.pt` is the exploratory Teacher. Because its
  full-label training snapshot contained the formerly hidden original SSOD
  unlabeled data, that original pool is explicitly prohibited from its pseudo
  label input.
- 2,205 copied Open Images files were sealed as a separate image-only pool.
  Every image SHA-256 was compared with the complete v12n snapshot; no overlap
  was admitted. Its input manifest contains no boxes or categories.
- Raw dual-view inference produced 202,491 candidates. The configuration-bound
  Trust Filter retained 2,888 candidates; protected pseudo-audit precision was
  `0.945455`, above the `0.90` refresh gate.
- A Student snapshot was dry-run sealed with 108 human images, 1,292 accepted
  pseudo-labeled images and 90 validation images. Test and pseudo-audit labels
  are absent. The first real launch exposed a relative-path publication defect,
  which is fixed and regression-tested. Its retry completed 11 epochs with
  curve-best validation mAP50 `0.553870`, but framework finalization rewrote
  the exported checkpoint; independent validation of that final file is 0.0,
  so it is failure evidence rather than a usable result.
- The active stabilization run retains the same Teacher/data split but sets
  `lr0=0.0001`, freezes 10 layers, uses max 20 epochs/patience 5 and captures
  every live curve-best checkpoint before Ultralytics finalization. The
  captured model is republished as canonical `weights/best.pt` before the
  required independent validation. It completed after 6 epochs under early
  stopping with validation mAP50 `0.607870`, mAP50-95 `0.421381`, precision
  `0.650913`, recall `0.557154`, and F1 `0.600395`. The published best and
  preserved curve-best files have identical SHA-256
  `bc8e5180e1be0c04dc623cda0fec41f3312e69df022b555c4752c8ffb211f3e2`.
  This is a valid exploratory negative result below the v12n Teacher's
  `0.644387` mAP50 and the customer `0.80` target, not a claimed pass. It
  still validates only the retained validation split.
- The next action is to freeze this checkpoint for the PySide6 qualitative
  demo and, if a causal SSOD comparison is required, run a validation-only
  human-label control. Do not repeat the same pseudo-label configuration
  without an explicit policy change.
- The checkpoint is now frozen for the PySide6 qualitative demo. In the
  Windows `fruit-ssod` Conda environment, the real `best.pt` loaded through
  `ModelManager` and produced one Pineapple detection on each of three shared
  Open Images samples. The GUI/contract suite passed 36 tests, and the
  published demo export contains annotated PNGs, CSV/JSON detections and a
  checksum manifest under
  `E:\\fruit_ssod_runtime\\artifacts_v12\\exports\\gui_demo_inference_20260804`.
  Its metadata records the checkpoint SHA-256 and explicitly keeps camera and
  open-world capabilities disabled. A validation-only human-label control is
  still optional if a causal SSOD comparison is required.
- The full Windows test suite passed `409 passed, 1 skipped` in the
  `fruit-ssod` Conda environment with Qt offscreen mode. GUI preflight passed
  with `-SkipDataRootReachability`; the approved UNC directory is separately
  reachable, but Windows cannot report capacity for this unmapped SMB root, so
  the launcher documents that diagnostic switch.
- The repository QA launcher passed the same test suite and source-file/
  staging checks. Remaining release gates are real experiment aggregation,
  report generation, manual GUI checks, and delivery-manifest verification.
- Formal Task 17 preparation was checked against the current `splits_v2`
  protocol. Previously stored pseudo-audit artifacts were bound to an older
  split and cannot be reused. A fresh Task 13 run produced 28,414 dual-view
  candidates; Task 14 retained 70 train-pool labels. The fresh protected Task
  15 audit achieved Precision `0.750000`, so `refresh_allowed=false` under the
  required `0.90` gate. This is recorded as a quality warning for the
  first-result track, not a reason to stop the customer-requested end-to-end
  run. A new Teacher or an explicitly changed filtering/data policy remains
  required for a formal refresh-allowed comparison.
- The customer-authorized v0 Teacher continuation on the current v2 split is
  complete after 60 epochs. Its independently validated validation metrics are
  mAP50 `0.422360`, mAP50-95 `0.255375`, Precision `0.478906`, Recall
  `0.433742`, F1 `0.455207`; the best epoch was 57 and the checkpoint SHA-256
  is `2913b82aa9ef3206cb4f6092b58026ce5d6bec881c2de61aa1c17072e6a094c5`.
  This is the frozen v0 Teacher, not a formal target pass.
- Using that frozen Teacher on the current v2 unlabeled pool produced 5,546
  raw dual-view candidates. The explicit v0 Trust Filter policy retained 146
  candidate occurrences. The protected current-v2 pseudo-audit measured
  Precision `0.608696` and `refresh_allowed=false` at the formal `0.90`
  threshold; the v0 Student configuration records an explicit exploratory
  below-gate override rather than changing or hiding this audit.
- The v0 Student dataset dry-run completed successfully with split fingerprint
  `7653d1f762053b90362803c8b2d25d287769de055fe11595565319f7fabe159c`, current
  human budget, accepted pseudo labels, validation-only evaluation data and
  no protected test/pseudo-audit images in the training snapshot. Real Student
  training then completed after early stopping. Its independently validated
  validation metrics are mAP50 `0.366312`, mAP50-95 `0.220509`, Precision
  `0.512921`, Recall `0.379380`, F1 `0.436158`. After checkpoint freeze, the
  v0 fixed-test evaluation measured mAP50 `0.263117`, mAP50-95 `0.154698`,
  Precision `0.300340`, Recall `0.318192`, F1 `0.309009`; the Teacher fixed-test
  reference measured mAP50 `0.317852`. These are complete exploratory v0
  results and are not formal target passes.
- `ssod_exploratory_v12n_independent_openimages_seed42.yaml` is deliberately
  excluded from the eight-entry comparable Task-17 matrix. It binds the
  completed Teacher run record, checkpoint hash and pseudo-label source model.
- The first optimization run `ssod-v0-opt-longer-supervised-v2-teacher-seed42-r1`
  has completed. It kept the same split fingerprint and pseudo-label snapshot,
  raised the training budget to 100 epochs with patience 15, and removed the
  initial ten-layer freeze. Its result was evaluated against the same fixed test
  set and compared with the sealed v0 baseline; it is exploratory and does not
  overwrite v0 artifacts.

The first optimization completed with early stopping at 17 epochs. Its
validation mAP50 is `0.342439` and its sealed fixed-test mAP50 is `0.266709`
(mAP50-95 `0.149349`, Precision `0.298437`, Recall `0.367040`, F1 `0.329202`).
This is only a small fixed-test improvement over v0 `0.263117`; the result is
recorded in `docs/experiments/v0-first-result-summary.md` and the next work
prioritizes pseudo-label quality and data coverage rather than simply adding
more epochs.

A second optimization completed as
`ssod-v0-opt-strict-pseudo-v2-teacher-seed42-r1`. Its strict Trust Filter
retained 52 full train-pool pseudo boxes, while the protected audit subset
retained 10 boxes at measured Precision `0.900000`. The same 90-image fixed
test evaluation measured mAP50 `0.246821`, below the v0 baseline `0.263117`;
the strict result is retained as a negative ablation and is not selected for
the demo. The next optimization therefore targets data coverage and class
balance rather than further threshold tightening.

The data-coverage optimization has begun by sealing 2,205 independent
Open Images image-only records at
`E:/fruit_ssod_runtime/data/fruit_ssod/processed/ssod_unlabeled_pool_openimages_v13_v2`.
The independence check produced zero image SHA-256 overlap with the current
v2 Teacher dataset and a new pool fingerprint
`30f24a47f7d0da16672fdb083935b3d153df383e40ed1b20418ad6d9fe00761d`.
Dual-view pseudo-label generation completed with 25,036 candidates; the
configuration-bound Trust Filter retained 3,499 full-pool boxes. The protected
current-v2 audit remains Precision `0.608696` and `refresh_allowed=false`, so
the explicit v0 override is recorded. The independent-pool Student run
`ssod-v0-opt-independent-pool-v2-teacher-seed42-r1` is now training against
this expanded snapshot and will use the same fixed test.

The next optimization candidate is a deterministic class-aware pseudo-label
exposure/sampling comparison. The current expanded filter snapshot is strongly
skewed toward Strawberry pseudo boxes and has very few Banana pseudo boxes, so
the follow-up will preserve the sealed validation/test members while comparing
class-balanced exposure with the current 50/50 human/pseudo policy. This is an
exploratory continuation and will not block first-result delivery on any
formal precision gate.

#### Customer-authorized first-result execution path (v0)

The customer explicitly requested an uninterrupted first complete result and
authorized proceeding without waiting for the `0.80` customer target or the
`0.85` formal Teacher gate. This path is separate from formal acceptance and
produces a traceable baseline that can be improved without restarting:

1. Let the current `supervised_v2_full_yolov8m_1024_teacher_seed42` run finish
   under early stopping, then freeze and independently validate its
   `weights/best.pt`. Record the run ID, split fingerprint, configuration and
   SHA-256.
2. Use that frozen checkpoint to generate Task 13 dual-view candidates from
   the current v2 unlabeled pool. Apply the current Trust Filter and retain
   the complete audit even if protected precision is below `0.90`;
   `refresh_allowed` is evidence, not a v0 stop condition.
3. Materialize a v0 Student dataset from the current split using the
   available human-label budget plus retained pseudo labels. Keep test and
   pseudo-audit images out of Student training and preserve provenance hashes.
4. Train one v0 Student model with early stopping, independently validate the
   selected checkpoint, then run the permitted fixed-test evaluation only
   after the checkpoint and evaluation configuration are frozen. Record low
   scores and failures rather than silently retrying until a target is met.
5. Package the first result with checkpoints, validation/fixed-test JSON,
   pseudo-label audit, GUI inference exports, reproduction commands and a
   concise v0 experiment summary. State clearly that it is exploratory and
   does not claim that the `0.80`/`0.85` gates passed.
6. After the v0 package exists, continue optimization separately: improve data
   balance/coverage, filtering policy, initialization and schedule, then run
   the formal multi-seed matrix and upgrade the report from immutable evidence.

This path prevents the project from being held at one quality gate while
preserving the formal gates needed for a defensible research comparison.

No fixed-test evaluation is executed before the v0 checkpoint/configuration
freeze described above.

**Phase C verification gate**

- Every run links to one split fingerprint and config snapshot.
- Three 20% baseline runs are complete.
- 10%, 40% and 100% reference results exist.
- The 100% upper bound is technically credible enough to proceed.

**Commit**

```powershell
git add configs/experiments scripts/run_supervised_matrix.ps1 docs/experiments/supervised-baselines.md
git commit -m "exp: define supervised experiment matrix"
```

Do not commit generated weights or run directories.

### Task 12R: Execute the v12 data-quality and small-object upper-bound recovery

**Why this recovery is required**

- The source-disjoint v10 fixed-test result is `mAP@0.5 = 0.546244`, below the
  `0.85` supervised credibility gate.
- The expanded v11 YOLOv8m run reached a validation best of `mAP@0.5 =
  0.629700` at epoch 70 and remained around `0.618`-`0.626` through epoch 85.
  Continuing the same configuration is therefore treated as an operational
  futility case, not as evidence that the project target is impossible.
- The v11 training pool contains 11,104 Orange boxes but only 1,400 Pineapple
  boxes, and 8,970 of its 23,546 boxes occupy less than one percent of their
  source image. The recovery must address class imbalance and small objects
  together instead of changing only the epoch count.

**Files**

- Create: `src/fruit_ssod/data/full_label_upper_bound.py`
- Create: `src/fruit_ssod/data/balanced_training.py`
- Create: `src/fruit_ssod/data/object_centric_tiles.py`
- Create: `src/fruit_ssod/data/sliced_detection.py`
- Create: `src/fruit_ssod/evaluation/validation_selection.py`
- Create: `src/fruit_ssod/cli/materialize_full_label_upper_bound.py`
- Create: `src/fruit_ssod/cli/materialize_object_centric_tiles.py`
- Create: `src/fruit_ssod/cli/select_validation_candidate.py`
- Create: `configs/models/yolo11m_1024.yaml`
- Create: `configs/experiments/v12_v8m_balanced_1024_finetune.yaml`
- Create: `configs/experiments/v12_yolo11m_balanced_1024.yaml`
- Create: `tests/unit/test_full_label_upper_bound.py`
- Create: `tests/unit/test_balanced_training.py`
- Create: `tests/unit/test_object_centric_tiles.py`
- Create: `tests/unit/test_validation_selection.py`
- Create: `tests/unit/test_sliced_detection.py`
- Update: `docs/experiments/supervised-baselines.md`
- Update: `docs/experiments/experiment-ledger.md`

**Steps**

1. Stop v11 for validation futility without deleting it. Preserve `best.pt`,
   `last.pt`, `results.csv`, logs, configuration and the reason for stopping.
   Do not run its fixed-test evaluation.
2. Keep the existing 1,148-image validation membership, 1,148-image fixed-test
   membership and 574-image pseudo-audit membership unchanged.
3. For the supervised upper-bound only, restore the labels of the 1,721 images
   currently hidden as the SSOD unlabeled pool. Materialize 8,607 labelled
   training images (`6,886 + 1,721`) from the already cleaned canonical source.
   This recovery snapshot is not a label-budget or Student comparison run.
4. Re-audit the materialized snapshot and fail closed on missing images,
   non-canonical classes, duplicate-group leakage or changed protected
   membership.
5. Create a deterministic balanced training view. Preserve every natural
   training image once, then oversample images containing Banana, Strawberry
   or Pineapple with a documented cap; never undersample the protected splits
   or silently remove Orange examples.
6. Create deterministic object-centric tiles for small objects. Retain full
   images, clip boxes only at tile boundaries, record parent image IDs and keep
   every tile in its parent image's split. Reject empty or non-positive boxes.
7. Run validation-only screening with the same protected split:
   - reuse the v11 `best.pt` for a 960/1024-pixel YOLOv8m fine-tune;
   - train COCO-pretrained YOLO11m on the same balanced view;
   - compare direct inference with sliced inference using validation only.
8. Select one pipeline using predeclared validation criteria: primary
   `mAP@0.5`, then per-class AP50 floor, then Recall. Do not inspect fixed-test
   labels or metrics during screening.
9. Freeze model, image size, tile size, overlap, confidence/NMS settings and
   checkpoint hash before one fixed-test evaluation.
10. Apply the gate without exception:
    - fixed-test `mAP@0.5 >= 0.85`: authorize Task 13 and later SSOD work;
    - `0.80 <= mAP@0.5 < 0.85`: record that the customer accuracy target is met
      but keep SSOD blocked pending a stronger Teacher;
    - `mAP@0.5 < 0.80`: keep SSOD blocked and perform targeted annotation/data
      remediation without changing the fixed-test membership.

**Phase C recovery verification gate**

- v11 futility evidence and retained checkpoint hashes are immutable.
- The v12 upper-bound contains exactly 8,607 labelled training images and the
  unchanged protected memberships.
- Balance and tile manifests are deterministic and contain parent-image
  provenance.
- Model selection uses validation evidence only.
- Exactly one frozen v12 candidate is evaluated on the fixed test.
- Formal Task 13 remains gated by fixed-test `mAP@0.5 >= 0.85`; the separate
  customer-authorized v0 path may proceed with a lower score and must label it
  exploratory.

**Execution state**

- [x] v11 stopped after 87 completed epochs; best validation mAP50 `0.629700`
  at epoch 70; retained checkpoint and curve hashes recorded; no fixed-test
  evaluation executed.
- [x] Full-label v12n snapshot materialized with 8,607 train, 1,148 validation
  and 1,148 test images; the 574-image pseudo-audit membership remains
  protected. The first unnormalized cache-scan attempt was stopped before
  epoch 1; the replacement normalized 278 missing JPEG end markers and passed
  complete image-hash/label audits both before and after Ultralytics loading.
- [x] Deterministic class-balanced view materialized with 12,759 training
  exposures and per-class image exposures of approximately 2,583.
- [x] Train-only 512-pixel object-centric view materialized with 4,017 tiles
  and 16,776 combined exposures.
- [ ] Complete validation-only YOLOv8m high-resolution fine-tuning.
- [ ] Complete the YOLO11m and sliced-inference validation comparisons if the
  earlier validation gate does not already justify freezing a candidate. The
  official local YOLO11m COCO initialization has been downloaded, SHA-256
  `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`, and
  successfully loaded against `yolo11m.yaml`; it is queued until the active
  v8m run releases the GPU.
- [ ] Freeze exactly one candidate and perform exactly one fixed-test
  evaluation before applying the `0.85` SSOD gate.
- [x] Validation-only selection protocol implemented: candidates that satisfy
  the five-class AP50 floor are preferred; then mAP50, Recall and stable ID
  rank the candidate. Runs with an existing fixed-test evaluation are rejected
  from selection; comparison requires identical sealed validation membership,
  not identical training-data YAML, so full-image and tiled training views can
  be compared fairly.

**Commit**

```powershell
git add src/fruit_ssod/data src/fruit_ssod/cli configs/models configs/experiments tests docs/experiments
git commit -m "exp: add v12 supervised upper-bound recovery"
```

Do not commit datasets, tiles, weights, caches or generated run directories.

## Phase D — Offline pseudo-label and Trust Filter pipeline

### Task 13: Implement dual-view pseudo-label generation

**Files**

- Create: `src/fruit_ssod/pseudo/candidates.py`
- Create: `src/fruit_ssod/pseudo/transforms.py`
- Create: `src/fruit_ssod/pseudo/generator.py`
- Create: `src/fruit_ssod/cli/generate_pseudo_labels.py`
- Create: `tests/unit/test_box_flip_mapping.py`
- Create: `tests/integration/test_pseudo_generation.py`

**Steps**

1. Write failing property tests for horizontal flip and reverse mapping.
2. Test candidate serialization using a fake detector.
3. Run original and horizontally flipped views.
4. Map all boxes to original coordinates and retain both raw prediction sets.
5. Save provenance: Teacher run ID, source image ID, view, confidence and box coordinates.
6. Verify the generator never loads hidden human labels.

**Commit**

```powershell
git add src/fruit_ssod/pseudo/candidates.py src/fruit_ssod/pseudo/transforms.py src/fruit_ssod/pseudo/generator.py src/fruit_ssod/cli/generate_pseudo_labels.py tests
git commit -m "feat: add dual-view pseudo-label generation"
```

### Task 14: Implement global-threshold baseline and Trust Filter

**Files**

- Create: `src/fruit_ssod/pseudo/thresholds.py`
- Create: `src/fruit_ssod/pseudo/trust_filter.py`
- Create: `src/fruit_ssod/cli/filter_pseudo_labels.py`
- Create: `tests/unit/test_threshold_selection.py`
- Create: `tests/unit/test_trust_filter.py`
- Create: `tests/fixtures/pseudo/candidates.json`

**Steps**

1. Write failing tests for:
   - global confidence filtering;
   - per-class threshold clamp 0.50–0.85;
   - same-class cross-view matching;
   - `IoU >= 0.60`;
   - minimum 16-pixel width and height at 640;
   - maximum 90% image area;
   - aspect ratio 0.1–10 and labeled-distribution bounds;
   - maximum 20 boxes per image;
   - deterministic NMS and tie-breaking.
2. Select per-class thresholds from validation PR data, targeting approximately 90% precision.
3. Record each rejected candidate with a reason code.
4. Export standard YOLO pseudo-label files plus an audit JSONL.
5. Run all Trust Filter tests.

**Commit**

```powershell
git add src/fruit_ssod/pseudo/thresholds.py src/fruit_ssod/pseudo/trust_filter.py src/fruit_ssod/cli/filter_pseudo_labels.py tests
git commit -m "feat: add reliable pseudo-label filtering"
```

### Task 15: Implement sealed pseudo-label audit

**Files**

- Create: `src/fruit_ssod/evaluation/pseudo_metrics.py`
- Create: `src/fruit_ssod/cli/audit_pseudo_labels.py`
- Create: `src/fruit_ssod/reporting/pseudo_figures.py`
- Create: `tests/unit/test_pseudo_metrics.py`
- Create: `tests/integration/test_audit_label_isolation.py`

**Steps**

1. Test one-to-one matching and TP/FP/FN calculation.
2. Ensure audit labels are accepted only by the audit command, never by training commands.
3. Calculate per-class and overall Precision, Recall and F1 before and after filtering.
4. Generate examples of kept, rejected, false-positive and missed boxes.
5. Stop pseudo-label refresh when audit Precision is below 90%.

**Commit**

```powershell
git add src/fruit_ssod/evaluation/pseudo_metrics.py src/fruit_ssod/cli/audit_pseudo_labels.py src/fruit_ssod/reporting/pseudo_figures.py tests
git commit -m "feat: add sealed pseudo-label audit"
```

### Task 16: Implement Student dataset composition and training

**Files**

- Create: `src/fruit_ssod/training/student_dataset.py`
- Create: `src/fruit_ssod/training/semi_supervised.py`
- Create: `src/fruit_ssod/cli/train_student.py`
- Create: `configs/experiments/ssod_global_seed42.yaml`
- Create: `configs/experiments/ssod_trust_seed42.yaml`
- Create: `tests/unit/test_student_dataset.py`
- Create: `tests/integration/test_student_training_dry_run.py`

**Steps**

1. Write failing tests for label precedence, human-label protection and approximately 50% human sample frequency.
2. Merge only human labels and accepted pseudo-labels.
3. Store pseudo-label source and reliability metadata separately from YOLO text labels.
4. Make initialization policy explicit in the run config and keep it identical across comparable SSOD runs.
5. Execute a one-epoch Student smoke run.
6. Verify test and pseudo-audit labels are not present in the training snapshot.

**Commit**

```powershell
git add src/fruit_ssod/training/student_dataset.py src/fruit_ssod/training/semi_supervised.py src/fruit_ssod/cli/train_student.py configs/experiments/ssod_global_seed42.yaml configs/experiments/ssod_trust_seed42.yaml tests
git commit -m "feat: add semi-supervised student training"
```

## Phase E — Experiment matrix, evaluation and acceptance

### Task 17: Define the SSOD and ablation experiment matrix

**Files**

- Create: `configs/experiments/ssod_trust_seed3407.yaml`
- Create: `configs/experiments/ssod_trust_seed2026.yaml`
- Create: `configs/experiments/ablation_no_class_threshold.yaml`
- Create: `configs/experiments/ablation_no_view_consistency.yaml`
- Create: `configs/experiments/ablation_no_size_filter.yaml`
- Create: `configs/experiments/ablation_no_human_resampling.yaml`
- Create: `scripts/run_ssod_matrix.ps1`
- Create: `tests/unit/test_experiment_matrix.py`

**Steps**

1. Test that the matrix contains:
   - global-threshold SSOD, one seed;
   - full Trust Filter SSOD, three seeds;
   - four one-seed ablations.
2. Test that comparison groups share the same base model, image size, split fingerprint and evaluation protocol.
3. Add resume support and skip only runs already marked complete with matching fingerprints.
4. Print the full queue before requesting GPU work.

**Commit**

```powershell
git add configs/experiments scripts/run_ssod_matrix.ps1 tests/unit/test_experiment_matrix.py
git commit -m "exp: define semi-supervised experiment matrix"
```

### Task 18: Implement immutable result aggregation and statistical summaries

**Files**

- Create: `src/fruit_ssod/evaluation/aggregate.py`
- Create: `src/fruit_ssod/evaluation/acceptance.py`
- Create: `src/fruit_ssod/reporting/result_tables.py`
- Create: `src/fruit_ssod/reporting/result_figures.py`
- Create: `src/fruit_ssod/cli/aggregate_results.py`
- Create: `tests/unit/test_result_aggregation.py`
- Create: `tests/unit/test_acceptance.py`

**Steps**

1. Write tests proving incomplete and failed runs remain visible.
2. Calculate mean and standard deviation for the two three-seed main groups.
3. Report mAP50, mAP50-95, Precision, Recall, F1 and per-class AP.
4. Keep FruitDet metrics in a separate table with only mapped classes.
5. Evaluate:
   - final mean `mAP@0.5 >= 0.80`;
   - mean improvement over 20% baseline `>= 0.03`.
6. Export canonical CSV/JSON plus XLSX convenience files.
7. Generate label-budget, method-comparison, ablation, PR, confusion and per-class AP figures.

**Commit**

```powershell
git add src/fruit_ssod/evaluation/aggregate.py src/fruit_ssod/evaluation/acceptance.py src/fruit_ssod/reporting src/fruit_ssod/cli/aggregate_results.py tests
git commit -m "feat: add result aggregation and acceptance checks"
```

### Task 19: Implement RTX 3080 deployment benchmark

**Files**

- Create: `src/fruit_ssod/evaluation/benchmark.py`
- Create: `src/fruit_ssod/cli/benchmark_model.py`
- Create: `tests/unit/test_benchmark_summary.py`
- Create: `docs/experiments/benchmark-protocol.md`

**Steps**

1. Define warm-up iterations, measured iterations, synchronization and image-size policy.
2. Measure latency, FPS, peak allocated memory and model size.
3. Record GPU, driver, PyTorch, CUDA and Ultralytics versions.
4. Test the summary logic with deterministic timing fixtures.
5. Run the final model benchmark with no other GPU process active.

**Phase E verification gate**

- The complete experiment matrix has results or an explicit failed-run record.
- Main results include three seeds and standard deviations.
- External test results are separate.
- Acceptance evaluation is machine-generated.
- Benchmark metadata is complete.

**Commit**

```powershell
git add src/fruit_ssod/evaluation/benchmark.py src/fruit_ssod/cli/benchmark_model.py tests/unit/test_benchmark_summary.py docs/experiments/benchmark-protocol.md
git commit -m "feat: add RTX 3080 inference benchmark"
```

## Phase F — PySide6 desktop demonstration

### Task 20: Build ModelManager and GUI application shell

**Files**

- Create: `src/fruit_ssod/gui/app.py`
- Create: `src/fruit_ssod/gui/main_window.py`
- Create: `src/fruit_ssod/gui/model_manager.py`
- Create: `src/fruit_ssod/gui/widgets/status_panel.py`
- Create: `tests/gui/test_app_startup.py`
- Create: `tests/gui/test_model_manager.py`

**Steps**

1. Write `pytest-qt` tests for startup without a model and invalid-weight errors.
2. Implement navigation for single image, batch images, video and logs.
3. Implement one-active-model GPU policy and model release.
4. Show actionable loading and compatibility errors.
5. Verify the application starts with:

   ```powershell
   conda run -n fruit-ssod python -m fruit_ssod.gui.app
   ```

**Commit**

```powershell
git add src/fruit_ssod/gui tests/gui
git commit -m "feat: add PySide6 application shell"
```

### Task 21: Implement single-image and batch-image inference

**Files**

- Create: `src/fruit_ssod/gui/workers/image_worker.py`
- Create: `src/fruit_ssod/gui/widgets/image_view.py`
- Create: `src/fruit_ssod/gui/result_exporter.py`
- Create: `tests/gui/test_image_inference.py`
- Create: `tests/gui/test_batch_cancellation.py`

**Steps**

1. Test that inference runs outside the main GUI thread.
2. Add file selection, folder selection, preview and previous/next navigation.
3. Add confidence and NMS controls.
4. Show boxes, class counts, confidence, progress and latency.
5. Add cancel support.
6. Export annotated images plus CSV/JSON results.

**Commit**

```powershell
git add src/fruit_ssod/gui/workers/image_worker.py src/fruit_ssod/gui/widgets/image_view.py src/fruit_ssod/gui/result_exporter.py tests/gui
git commit -m "feat: add image inference workflows"
```

### Task 22: Implement video-file inference

**Files**

- Create: `src/fruit_ssod/gui/workers/video_worker.py`
- Create: `src/fruit_ssod/gui/widgets/video_view.py`
- Create: `tests/gui/test_video_worker.py`
- Create: `tests/fixtures/video/tiny_video.mp4`

**Steps**

1. Test frame ordering, stop behavior, progress and export cleanup.
2. Add video open, play/pause processing, stop and save.
3. Display current FPS and total progress.
4. Keep capture code file-based; do not add a camera device path.
5. Verify that cancellation leaves no corrupt final export.

**Commit**

```powershell
git add src/fruit_ssod/gui/workers/video_worker.py src/fruit_ssod/gui/widgets/video_view.py tests/gui tests/fixtures/video
git commit -m "feat: add video file inference"
```

### Task 23: Add the disabled open-world extension contract

**Files**

- Create: `src/fruit_ssod/open_world/contracts.py`
- Create: `src/fruit_ssod/open_world/README.md`
- Create: `tests/unit/test_open_world_contract.py`

**Steps**

1. Test that known detector results always use `is_unknown=False`.
2. Define extension interfaces for unknown proposals and class registry updates without implementing either.
3. Ensure the GUI has no enabled open-world action.
4. Document that the interface is future work and not part of acceptance.

**Commit**

```powershell
git add src/fruit_ssod/open_world tests/unit/test_open_world_contract.py
git commit -m "docs: reserve open-world detector contract"
```

### Task 24: Add Windows launchers and user documentation

**Files**

- Create: `scripts/start_gui.ps1`
- Create: `scripts/run_pipeline.ps1`
- Create: `README.md`
- Create: `docs/user-guide.md`
- Create: `docs/troubleshooting.md`
- Test: `tests/integration/test_launcher_contract.py`

**Steps**

1. Make launchers resolve paths relative to the repository and run the configured Conda environment explicitly; permit a documented existing Conda environment override.
2. Add preflight checks before training and GUI startup.
3. Document setup, data preparation, smoke run, full experiments, GUI use and result locations.
4. Document common failures: no CUDA, OOM, inaccessible UNC path, missing weights, unsupported video and invalid config.
5. Keep PyInstaller packaging optional and outside core acceptance.

**Phase F verification gate**

- GUI starts and remains responsive.
- Single image, folder and video workflows pass.
- Cancellation works.
- No camera control exists.
- Open-world UI is absent or disabled.

**Commit**

```powershell
git add scripts README.md docs/user-guide.md docs/troubleshooting.md tests/integration/test_launcher_contract.py
git commit -m "docs: add Windows launch and user guides"
```

## Phase G — Full experiments and Final Report

### Task 25: Execute the complete experiment queue

**Files**

- Update: `docs/experiments/experiment-ledger.md`
- Generate outside Git: shared `runs`, `weights`, `pseudo_labels` and `exports`

**Steps**

1. Re-run the complete dataset audit.
2. Record the exact dataset fingerprint.
3. Execute the supervised matrix and Task 12R recovery when its gate is active.
4. For the formal matrix, generate raw pseudo-label candidates only from a
   Teacher whose frozen fixed-test `mAP@0.5` meets the `0.85` gate. The
   customer-authorized v0 path may use its frozen validation-best Teacher and
   must label the resulting Student exploratory.
5. Execute the global threshold baseline.
6. Select per-class thresholds from validation data.
7. Audit Trust Filter pseudo-labels.
8. Execute the three-seed Trust Filter matrix.
9. Execute four ablations.
10. Refresh pseudo-labels at most once and only if both approved conditions pass.
11. Evaluate the fixed primary test and FruitDet external test.
12. Run the RTX 3080 benchmark.
13. Aggregate all results and execute the acceptance checker.

**Verification gate**

- No experiment was silently omitted.
- All three primary seeds are reported.
- The final acceptance JSON states pass or fail with supporting values.
- Any failure to reach the target remains visible and is analyzed honestly.

**Commit**

```powershell
git add docs/experiments/experiment-ledger.md
git commit -m "docs: record completed experiment matrix"
```

### Task 26: Build the report-data contract

**Files**

- Create: `src/fruit_ssod/reporting/report_data.py`
- Create: `src/fruit_ssod/cli/build_report_data.py`
- Create: `reports/final_report/report_data.schema.json`
- Create: `tests/unit/test_report_data.py`

**Steps**

1. Define required fields for datasets, methods, metrics, pseudo-label quality, ablations, deployment performance and acceptance.
2. Reject missing runs, incompatible split fingerprints and manually overridden values.
3. Generate one immutable `report_data.json` from canonical result files.
4. Include provenance for every table cell and figure.
5. Run tests against complete and deliberately incomplete fixtures.

**Commit**

```powershell
git add src/fruit_ssod/reporting/report_data.py src/fruit_ssod/cli/build_report_data.py reports/final_report/report_data.schema.json tests/unit/test_report_data.py
git commit -m "feat: add report data contract"
```

### Task 27: Generate report figures and tables

**Files**

- Create: `src/fruit_ssod/reporting/final_figures.py`
- Create: `src/fruit_ssod/reporting/final_tables.py`
- Create: `src/fruit_ssod/cli/build_report_assets.py`
- Create: `tests/integration/test_report_assets.py`

**Steps**

1. Generate a controlled set of no more than 10 figures and 10 tables.
2. Use consistent English labels, units, class names, colors and captions.
3. Include method workflow, dataset examples, pseudo-label filtering, label-budget curve, method comparison, failure cases and deployment results.
4. Validate that all displayed values match `report_data.json`.
5. Render at publication-suitable resolution.

**Implementation status**

- [x] The asset builder produces exactly ten controlled 300-DPI PNG figures:
  workflow, dataset composition, sealed annotation examples, pseudo-label
  quality, label budget, method comparison, ablation, per-class AP50,
  failed/missing evidence and RTX 3080 deployment evidence. It also produces
  controlled CSV tables. The sample montage is copied only after its recorded
  byte count and SHA-256 match the immutable report data.
- [x] The report builder verifies each asset hash, embeds generated PNG figures
  and CSV tables into the DOCX, and rejects non-PNG report figures.

**Commit**

```powershell
git add src/fruit_ssod/reporting/final_figures.py src/fruit_ssod/reporting/final_tables.py src/fruit_ssod/cli/build_report_assets.py tests/integration/test_report_assets.py
git commit -m "feat: generate final report assets"
```

### Task 28: Write and generate the English Final Report

**Files**

- Create: `reports/final_report/outline.md`
- Create: `reports/final_report/references.bib`
- Create: `reports/final_report/impact_statement.md`
- Create: `reports/final_report/appendix.md`
- Create: `reports/final_report/build_report.py`
- Create: `tests/integration/test_final_report.py`
- Generate outside Git or in release output:
  - `Final_Report.docx`
  - `Final_Report.pdf`

**Steps**

1. Create the report only after `report_data.json` is complete.
2. Write:
   - Abstract and up to five Keywords;
   - Introduction;
   - Literature Review;
   - Methodology;
   - Experimental Setup;
   - Results;
   - Discussion;
   - Impact Statement;
   - Conclusions;
   - References;
   - Appendix.
3. Keep the main body at or below 5000 words.
4. Keep figures and tables at or below 10 each.
5. Follow the IMechE Part C manuscript-style requirements specified in the uploaded course guide.
6. Make the Impact Statement answer WHAT, WHO and HOW and cover relevant industrial, commercial, environmental and societal effects.
7. State that open-world discovery is future work.
8. Do not include fabricated metrics or unsupported conclusions.
9. Validate headings, word count, figure/table count, captions, cross-references and reference coverage.
10. Render DOCX and PDF, then visually inspect representative pages.

**Implementation status**

- [x] Evidence-bound report source, outline, impact statement, appendix,
  literature record and preflight test were added. The preflight rejects
  incomplete three-seed evidence, assets from a different `report_data.json`,
  altered asset hashes and more than ten figures or tables.
- [ ] No formal DOCX or PDF has been generated: the real experiment matrix,
  sealed `report_data.json`, publication-ready raster figures and formal
  acceptance evidence do not yet exist. Formal generation remains pending
  rather than inserting illustrative or manually entered metrics.
- [x] The customer-authorized v0 experiment summary is recorded in
  `docs/experiments/v0-first-result-summary.md`. It is clearly labeled
  exploratory and will not substitute for the formal report.

**Commit**

```powershell
git add reports/final_report tests/integration/test_final_report.py
git commit -m "docs: add reproducible final report source"
```

Generated DOCX/PDF may be included only in a designated release package, not mixed with source commits unless explicitly required.

## Phase H — Final quality and handoff

### Task 29: Run full automated and manual QA

**Files**

- Create: `docs/testing/final-qa-checklist.md`
- Create: `scripts/run_all_checks.ps1`
- Update: `README.md`

**Steps**

1. Run:

   ```powershell
   conda run -n fruit-ssod python -m pytest -v
   conda run -n fruit-ssod python -m pytest --cov=fruit_ssod --cov-report=term-missing
   ```

2. Run configuration, data audit and result-provenance checks.
3. Start the GUI through the documented launcher.
4. Manually verify single image, folder, video, cancellation, exports and invalid input errors.
5. Open the final DOCX and PDF and inspect title page, abstract, figures, tables, references and appendix.
6. Confirm that no camera, PPT or completed open-world claim appears.
7. Confirm that datasets, weights and large run outputs are not staged.

**Commit**

```powershell
git add docs/testing/final-qa-checklist.md scripts/run_all_checks.ps1 README.md
git commit -m "test: add final project QA workflow"
```

### Task 30: Assemble the delivery package and reproducibility handoff

**Files**

- Create: `docs/handoff/delivery-manifest.md`
- Create: `docs/handoff/reproduction.md`
- Create: `scripts/build_delivery_manifest.ps1`

**Steps**

1. List every delivered dataset manifest, configuration, script, checkpoint, metric file, GUI component and report file.
2. Record SHA-256 values for released model weights and formal reports.
3. Link every final result to its run ID and split fingerprint.
4. Include clean-machine setup and reproduction commands.
5. Verify that the delivery package excludes private credentials, temporary files and unlicensed source redistribution.
6. Tag the verified Git revision only after the delivery checklist passes.

**Implementation status**

- [x] Final QA checklist, source-check launcher, delivery-manifest contract,
  Windows reproduction guide and non-overwriting manifest builder are present.
- [ ] Formal final QA and delivery assembly remain blocked until the real
  experiment matrix, GUI release validation and final report evidence exist.

**Final completion gate**

- Windows environment and shared path instructions are reproducible.
- Dataset audit is clean.
- Required experiments and all three primary seeds are complete.
- Acceptance status is backed by generated evidence.
- PySide6 image, folder and video workflows pass.
- DOCX and PDF reports satisfy the uploaded course requirements.
- Delivery manifest and hashes match the actual files.

**Commit**

```powershell
git add docs/handoff scripts/build_delivery_manifest.ps1
git commit -m "docs: add delivery and reproduction handoff"
```

## 3. Execution checkpoints

During implementation, stop for review at these points:

1. **Environment gate:** CUDA works and the shared dataset root is accessible.
2. **Data gate:** cleaning, duplicate and split audits contain no critical findings.
3. **Supervised gate:** baseline matrix is complete and the 100% upper bound is credible.
4. **Pseudo-label gate:** for formal comparisons, audit Precision supports use
   of filtered pseudo-labels; for the customer-authorized v0 run, publish the
   measured audit and proceed with an explicit exploratory warning.
5. **Experiment gate:** all required seeds and ablations are accounted for.
6. **GUI gate:** image, folder and video workflows pass without camera code.
7. **Report gate:** all formal values come from immutable results.
8. **Delivery gate:** source, weights, results and reports are traceable and reproducible.

## 4. First execution action

Begin with Task 1 only. The previously observed network checks could not reach the shared data path, so full data work must not start until the Windows preflight either confirms access or produces a concrete network/VPN/credential remediation.

## 5. Execution update: first expanded-pool result and continuous optimization

The customer-authorized first-result path has now produced a complete,
traceable expanded-pool Student result. Run
`ssod-v0-opt-independent-pool-v2-teacher-seed42-r1` completed 60 epochs and
was evaluated once on the sealed 90-image fixed test: test mAP@0.5
`0.2831098263`, mAP@0.5:0.95 `0.1723647127`, Precision `0.4432110111`, and
Recall `0.2585118857`. The checkpoint SHA-256 is
`4ef41e4c81e1fbdb2755d7acd425fd25fb1c58309dae7c2d78cc9540469beea7` and the
split fingerprint is
`0653d942deab2f42d96066a2ad402c3c53618ddd5a4a03989e0f7880a9b173d9`.

The paired no-class-threshold audit was generated with post-filter Precision
`0.65625` and `refresh_allowed=false`; this is recorded as exploratory evidence
and does not block the next run. The next Student run
`ssod-v0-opt-no-class-threshold-combined-v2-teacher-seed42-r1` is running under
its own run ID. A PowerShell chain-watcher bug that kept the completed-state
wait loop alive was fixed, and the preparation command now passes explicit
candidate and image-root paths.

The `0.80` customer target and `0.85` formal credibility screen remain
reported acceptance criteria, not blockers for this continuous execution path.

### Recovery note

The first chained no-class-threshold attempt (`...-r1`) terminated after its
first validation without a terminal run state. Its run record is now marked
`failed` with the interruption cause, and no artifacts were overwritten. A new
`...-r2` run was started with explicit stdout/stderr capture and a dedicated
fixed-test watcher; it is the active optimization candidate.
