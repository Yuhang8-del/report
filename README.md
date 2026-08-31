# Fruit SSOD

Windows-native research and demonstration prototype for five-class
semi-supervised fruit detection: Apple, Banana, Orange, Strawberry, and
Pineapple. The delivered desktop interface is file based: single images,
folders, and video files. It contains no camera control. The first
customer-authorized open-world experiment is complete as a separate offline
artifact, producing reviewable Unknown clusters for Avocado, Blueberry,
Cherry, Kiwi, Mango and Rockmelon without mutating the five-class runtime
registry. The GUI does not silently expose those clusters as semantic runtime
classes.

## Current scope and evidence boundary

The repository provides data preparation, leakage-safe split, supervised/SSOD
matrix, result aggregation, RTX 3080 benchmark, and PySide6 demonstrator code.
A customer-authorized real-data v0 chain (Teacher → pseudo-label audit →
Student → fixed test → offline GUI export) is complete and documented in
`docs/experiments/v0-first-result-summary.md`. The formal multi-seed matrix,
final report release has not yet been completed, and an accuracy target of 0.80
is still pending; consult
the immutable run records before making any metric claim. The formal
multi-seed matrix remains a separate pending evidence track.

## Delivery status

The customer-facing GUI is Chinese, file based (single image, batch folder and
video file; no camera control), and contains an experiment overview page. The
final evidence-bound report follows the supplied Word/PPT requirements and is
available as Word/PDF at:

- `E:/fruit_ssod_runtime/artifacts_v17/exports/final_report_v2_r2/final_report.docx`
- `E:/fruit_ssod_runtime/artifacts_v17/exports/final_report_v2_r2/final_report.pdf`
- `E:/fruit_ssod_runtime/artifacts_v17/exports/final_report_v2_r2/requirements_alignment.md`

The report uses sealed fixed-test evidence and explicitly keeps image-level
novel-category discovery separate from the five-class runtime detector. It does
not claim that the historical 0.80 target has been met.

## Quick start on Windows

Use a named Conda environment (default: `fruit-ssod`). The active Conda
environment may be overridden per invocation with `-CondaEnvironment`, or by
setting `FRUIT_SSOD_CONDA_ENV`. Copy `.env.example` only as a local reminder;
PowerShell does not load it automatically.

```powershell
conda create -n fruit-ssod python=3.10
conda run -n fruit-ssod python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
conda run -n fruit-ssod python -m pip install -r requirements.txt
conda env config vars set -n fruit-ssod PYTHONNOUSERSITE=1

$env:FRUIT_SSOD_DATA_ROOT = '\\10.16.57.94\dataset2\lyg\detect_datasets'
$env:FRUIT_SSOD_ARTIFACT_ROOT = 'D:\fruit-ssod-artifacts'
New-Item -ItemType Directory -Force $env:FRUIT_SSOD_ARTIFACT_ROOT
```

The CUDA-enabled PyTorch command must run before `requirements.txt`; select a
compatible wheel for the installed NVIDIA driver if the documented CUDA 12.1
build is not appropriate. Then run a no-GUI diagnostic:

```powershell
.\scripts\start_gui.ps1 -PreflightOnly
```

Start the desktop program only after preflight passes:

```powershell
.\scripts\start_gui.ps1
```

For data preparation, smoke validation, controlled training, result locations,
and limitations, read the [Chinese customer user guide](docs/user-guide-chinese-reference.md)
or the [English reproduction guide](docs/user-guide.md). For common Windows,
CUDA, network-share, checkpoint, video, and configuration failures, read
[troubleshooting](docs/troubleshooting.md).

The v0 and subsequent exploratory reproduction paths are recorded in
`docs/handoff/reproduction.md`; the current customer-directed track records
the best available fixed-test result and labels it exploratory. The historical
formal gates remain documented only for a separate formal-claim matrix.

## Final QA and report release

The final English report is evidence bound: report figures, tables, DOCX and
PDF are generated from the completed immutable result package. Run the
automated source checks with:

```powershell
.\scripts\run_all_checks.ps1
```

Then complete the manual [final QA checklist](docs/testing/final-qa-checklist.md)
and follow the [reproduction handoff](docs/handoff/reproduction.md).

Packaging with PyInstaller is optional and outside the research prototype's
acceptance criteria.
