# Code File Functions and Responsibilities

This catalogue explains the main launchers, source modules, models,
configuration files and verification assets in the runnable customer package.
Normal users need only the numbered launchers in the delivery root. The
`project/` tree is provided for deployment engineers, developers and research
reproduction.

## 1. Customer launchers

| File | Responsibility | Normal use |
|---|---|---|
| `01_install_environment.bat` | Creates the `fruit-ssod` Conda environment and installs pinned dependencies. | Run once on first installation. |
| `02_run_self_check.bat` | Checks Python, PyTorch, CUDA, checkpoints, open-category resources, camera profiles and sample inference. | Run after installation or package transfer. |
| `03_launch_fruit_detection_gui.bat` | Starts the English PySide6 application and preloads the five-class Student model. | Main daily entry point. |
| `04_launch_open_category_demo.bat` | Starts the experimental open-category interface. | Run when the extension demo is needed. |
| `setup_environment.ps1` | Forwards installation to the internal environment script. | PowerShell alternative to launcher 01. |
| `self_check.ps1` | Configures package paths and runs integrity and real-checkpoint inference tests. | PowerShell alternative to launcher 02. |
| `run_gui.ps1` | Selects the Conda environment and starts the main GUI. | PowerShell alternative to launcher 03. |
| `run_open_world_gui.ps1` | Configures package paths and starts the open-category GUI. | PowerShell alternative to launcher 04. |

## 2. Models and runtime resources

| File or directory | Responsibility |
|---|---|
| `models/student_best.pt` | Five-class semi-supervised Student checkpoint for Apple, Banana, Orange, Strawberry and Pineapple. |
| `models/teacher_best.pt` | Five-class supervised Teacher checkpoint and pseudo-label source. |
| `models/incremental_11class_best.pt` | Extended detector adding Avocado, Blueberry, Cherry, Kiwi, Mango and Rockmelon. |
| `models/class_registry_v2.json` | Ordered registry of the eleven detector classes and numeric IDs. |
| `models/open_world_objectness.pt` | Class-agnostic model for possible unknown fruit regions. |
| `models/open_world_encoder.pt` | Self-supervised encoder that converts candidate regions into cluster features. |
| `models/open_world_box_clusters.npz` | Stored clustering centres and parameters. |
| `models/open_world_cluster_names.json` | Display-name mapping for reviewed clusters. |
| `samples/images/` | Real photographs used by the self-check and demonstration. |
| `data/fruit_ssod/` | Split memberships, labels and runtime metadata. |
| `evidence/` | Fixed-test, pseudo-label audit and threshold-calibration evidence. |

## 3. GUI entry points and interface modules

| Code file | Responsibility |
|---|---|
| `project/scripts/delivery_gui.py` | Resolves the delivery root, locates the Student checkpoint, creates the Qt application and loads the model. |
| `project/scripts/open_world_demo.py` | Loads detector, proposal, encoder and cluster resources and assembles the open-category pipeline. |
| `project/src/fruit_ssod/gui/app.py` | Creates the Qt application and applies application-level settings. |
| `project/src/fruit_ssod/gui/main_window.py` | Implements the main window, page navigation and model lifecycle. |
| `project/src/fruit_ssod/gui/theme.py` | Defines colours, typography, cards, buttons and status styling. |
| `project/src/fruit_ssod/gui/model_manager.py` | Loads and releases checkpoints while preventing conflicting operations. |
| `project/src/fruit_ssod/gui/result_exporter.py` | Writes annotated images, detection metadata and exported results. |
| `project/src/fruit_ssod/gui/open_world_window.py` | Displays known detections, unknown proposals and reviewed extension results. |
| `project/src/fruit_ssod/gui/widgets/image_view.py` | Provides image selection, preview, inference controls and result saving. |
| `project/src/fruit_ssod/gui/widgets/video_view.py` | Provides video loading, frame inference, cancellation and export. |
| `project/src/fruit_ssod/gui/widgets/camera_view.py` | Provides device discovery, camera/model selection, live boxes, FPS, latency and snapshots. |
| `project/src/fruit_ssod/gui/workers/image_worker.py` | Runs image inference in a background thread. |
| `project/src/fruit_ssod/gui/workers/video_worker.py` | Runs cancellable frame-by-frame video inference. |
| `project/src/fruit_ssod/gui/workers/camera_worker.py` | Opens cameras, reads frames, executes inference and returns multi-object results. |

## 4. Detection and semi-supervised learning

| Code file | Responsibility |
|---|---|
| `project/src/fruit_ssod/detection/types.py` | Defines common box, class, confidence and image-result structures. |
| `project/src/fruit_ssod/detection/adapter.py` | Defines the framework-independent detector contract. |
| `project/src/fruit_ssod/detection/ultralytics_backend.py` | Adapts Ultralytics checkpoints and handles device selection and result conversion. |
| `project/src/fruit_ssod/training/supervised.py` | Runs supervised Teacher training and records checkpoints and metadata. |
| `project/src/fruit_ssod/training/semi_supervised.py` | Coordinates Teacher inference, pseudo-label inputs and Student training. |
| `project/src/fruit_ssod/training/student_dataset.py` | Combines human-labelled and accepted pseudo-labelled samples. |
| `project/src/fruit_ssod/training/run_record.py` | Stores reproducible run metadata, status, paths and hashes. |
| `project/src/fruit_ssod/pseudo/generator.py` | Converts Teacher detections into pseudo-label candidates. |
| `project/src/fruit_ssod/pseudo/trust_filter.py` | Applies confidence, class, geometry, size and consistency rules. |
| `project/src/fruit_ssod/pseudo/calibration.py` | Selects thresholds from protected validation evidence. |
| `project/src/fruit_ssod/pseudo/candidates.py` | Defines pseudo-label candidate schemas and serialisation. |

## 5. Open-category and incremental detection

| Code file | Responsibility |
|---|---|
| `project/src/fruit_ssod/open_world/box_proposals.py` | Produces class-agnostic candidates and removes regions explained by known classes. |
| `project/src/fruit_ssod/open_world/box_clustering.py` | Encodes candidate crops and assigns them to visual clusters. |
| `project/src/fruit_ssod/open_world/pipeline.py` | Coordinates known detection, unknown proposals and cluster assignment. |
| `project/src/fruit_ssod/open_world/box_metrics.py` | Computes localisation and unknown-candidate metrics. |
| `project/src/fruit_ssod/open_world/contracts.py` | Defines stable data contracts for extension components. |
| `project/src/fruit_ssod/open_world/incremental.py` | Supports reviewed class registration and incremental training preparation. |
| `project/src/fruit_ssod/open_world/incremental_adapter.py` | Adapts the reviewed eleven-class checkpoint to the standard detector interface. |

## 6. Data, evaluation and verification

| Directory or file | Responsibility |
|---|---|
| `project/src/fruit_ssod/data/` | Source import, class mapping, deduplication, cleaning, split construction and YOLO materialisation. |
| `project/src/fruit_ssod/evaluation/` | Detection metrics, pseudo-label audit, benchmarking, aggregation and checkpoint selection. |
| `project/src/fruit_ssod/reporting/` | Builds evidence tables and figures from stored JSON and manifests. |
| `project/src/fruit_ssod/cli/` | Command-line entry points for data, training, evaluation and extension workflows. |
| `project/configs/experiments/` | Reproducible experiment configurations and seeds. |
| `project/requirements-lock.txt` | Verified Python dependency versions. |
| `project/tests/unit/` | Unit tests for data, pseudo-label, evaluation and open-category components. |
| `project/tests/integration/` | Split isolation, launcher, dry-run and evidence-contract tests. |
| `project/tests/gui/` | Model loading, image/video/camera and GUI capability tests. |
| `outputs/self_check.json` | Latest machine-readable delivery self-check result. |
| `RUNNABLE_PACKAGE_MANIFEST.json` | Package inventory, verification summary and core-file hashes. |

## 7. Recommended workflow

```text
01_install_environment.bat
  -> 02_run_self_check.bat
  -> 03_launch_fruit_detection_gui.bat
  -> load an image, folder, video or camera
  -> save annotated results under outputs/
```

Experimental extension workflow:

```text
04_launch_open_category_demo.bat
  -> known-class detector
  -> class-agnostic proposals
  -> self-supervised feature encoder
  -> cluster assignment
  -> human review before class registration
```

