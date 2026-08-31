# Semi-Supervised Fruit Detection - Runnable Customer Package

This Windows delivery contains the runnable source code, trained checkpoints,
sample photographs, camera integration, experiment evidence and environment
launchers. Large historical training archives and intermediate checkpoints are
not required for normal inference and are not duplicated here.

See `CODE_FILE_FUNCTIONS.md` for a file-by-file code overview and
`CUSTOMER_DEPLOYMENT_AND_USAGE_GUIDE.md` for the detailed deployment procedure.
Metric evidence is stored under `evidence/` and described in
`evidence/README_EVIDENCE.md`.

## Included capabilities

- Five-class semi-supervised Student detector: Apple, Banana, Orange,
  Strawberry and Pineapple.
- Eleven-class extended detector: the five known fruits plus Avocado,
  Blueberry, Cherry, Kiwi, Mango and Rockmelon.
- Single-image, batch-image and video inference.
- Real-time USB or external-camera inference.
- Experimental open-category analysis interface.
- Automated checks for Python, PyTorch, CUDA, checkpoints, class registries and
  sample inference.

## First run

Requirements: Windows 10/11, Anaconda or Miniconda, and Python 3.10. An NVIDIA
GPU is recommended. CPU inference is possible for some static images but is
considerably slower.

1. Extract the complete ZIP file. Do not copy only a single Python file or
   checkpoint.
2. Run `01_install_environment.bat` once.
3. Run `02_run_self_check.bat`. A successful installation reports
   `"status": "passed"`.
4. Run `03_launch_fruit_detection_gui.bat` to open the English desktop GUI.
5. Run `04_launch_open_category_demo.bat` when the experimental open-category
   workflow is required.

Equivalent PowerShell commands are:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
powershell -ExecutionPolicy Bypass -File .\self_check.ps1
powershell -ExecutionPolicy Bypass -File .\run_gui.ps1
```

## Camera workflow

Open the `Live Camera` page, click `Refresh Devices`, select a camera and model,
and click `Start Detection`. The page can switch between the five-class Student
and eleven-class extended detector and displays multiple boxes, class names,
confidence, FPS and inference latency.

## Key locations

- `project/src/fruit_ssod/`: core Python package.
- `project/scripts/delivery_gui.py`: main GUI entry point.
- `models/student_best.pt`: five-class semi-supervised Student checkpoint.
- `models/incremental_11class_best.pt`: eleven-class extended checkpoint.
- `models/open_world_*`: open-category experiment resources.
- `samples/images/`: demonstration and self-check photographs.
- `outputs/`: self-check and inference outputs.

## Scope boundary

The five-class and eleven-class checkpoints are box-level object detectors. The
open-category workflow is an experimental aid for analysing additional fruit
categories and requires human review. It must not be treated as an unattended
grading, inventory or safety-critical decision system.

## Verified environment

The package was checked on Windows with Conda Python 3.10.20, PyTorch
2.5.1+cu121, Ultralytics 8.4.113 and an NVIDIA GeForce RTX 3080.

