# User guide

## Scope

This is a Windows-native Conda project. Its implemented recognition classes are
Apple, Banana, Orange, Strawberry, and Pineapple, in IDs 0–4 respectively. The
desktop prototype accepts image files, a folder of images, and video files; it
does not use a camera. Open-world discovery is reserved for later work and must
not be presented as an implemented capability.

The code and fixtures are available now, but the approved real dataset and the
complete experimental evidence have not yet been executed. In particular, do
not state a real-data accuracy or `>= 0.80` outcome until the immutable
aggregation/acceptance package produced by the full protocol supports it.
The data preparation workflow below must be completed and audited before any
real GPU training.

## 1. Create a Conda environment

From a native Windows PowerShell prompt, create the default environment and
install project dependencies. CUDA-enabled PyTorch must be installed first for
the installed NVIDIA driver; `requirements.txt` deliberately does not select a
CPU-only wheel for you.

```powershell
conda create -n fruit-ssod python=3.10
conda run -n fruit-ssod python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
conda run -n fruit-ssod python -m pip install -r requirements.txt
conda env config vars set -n fruit-ssod PYTHONNOUSERSITE=1
conda run -n fruit-ssod python -m pip check
```

Close/reopen the shell after `conda env config vars set`, or use `conda run`.
An existing Conda environment is supported: use `-CondaEnvironment existing-env`
on each launcher, or set `$env:FRUIT_SSOD_CONDA_ENV = 'existing-env'`.

## 2. Configure paths and preflight

Set machine-local paths before any command. The data root should be the
approved UNC share, and the artifact root must be a writable directory outside
the repository.

```powershell
$env:FRUIT_SSOD_DATA_ROOT = '\\10.16.57.94\dataset2\lyg\detect_datasets'
$env:FRUIT_SSOD_ARTIFACT_ROOT = 'D:\fruit-ssod-artifacts'
New-Item -ItemType Directory -Force $env:FRUIT_SSOD_ARTIFACT_ROOT
.\scripts\start_gui.ps1 -PreflightOnly
```

The launchers resolve paths from the repository, so they may be invoked from a
different current directory. Command-line values override environment values:

```powershell
.\scripts\start_gui.ps1 -CondaEnvironment fruit-ssod -DataRoot '\\server\share\fruit-data' -ArtifactRoot D:\fruit-artifacts -PreflightOnly
```

`-SkipDataRootReachability` is a diagnostic-only escape hatch for an intentional
offline check; it is not permission to replace the approved data root. The
underlying preflight still checks the selected interpreter, CUDA visibility,
local worktree/artifact storage, and artifact write/delete permission. When the
configured data root is a UNC share, the switch also skips its free-space query
because that query requires the same unavailable share; a local data root still
has its free space checked.

## 3. Prepare and audit data

Follow the source-specific local import policies in
[Open Images](data/open-images.md) and [auxiliary datasets](data/auxiliary-datasets.md).
Do not assume that sources were downloaded simply because importer code exists.
Keep source licenses and source IDs in their manifests.

The canonical preparation sequence is:

1. Convert/import allowed local source data into a canonical image/annotation manifest.
2. Run `clean_dataset` and retain its deduplication evidence.
3. Run `create_splits` to produce the leakage-safe primary split, protected
   validation/test labels, nested label budgets, unlabeled pool, and pseudo audit.
4. Run `audit_dataset` against the protected split output. Fix critical findings
   before training.
5. Import FruitDet only as the limited external test set; it must not enter the
   primary training, validation, test, or unlabeled pool.

The exact command arguments depend on the prepared manifests. Use `--help` on
each CLI before a real-data run; commands reject missing or incompatible
artifacts rather than invent paths:

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.clean_dataset --help
conda run -n fruit-ssod python -m fruit_ssod.cli.create_splits --help
conda run -n fruit-ssod python -m fruit_ssod.cli.audit_dataset --help
```

## 4. Safe smoke checks

These checks do not start the GUI or GPU training. The first uses repository
fixtures; the other two validate controlled matrix configuration after real
data/preparation paths and pretrained weights exist.

```powershell
conda run -n fruit-ssod python -m pytest -q

$env:FRUIT_SSOD_PRETRAINED_WEIGHTS = 'D:\approved-weights\yolov8s.pt'
.\scripts\run_pipeline.ps1 -Stage Supervised -DryRun
.\scripts\run_pipeline.ps1 -Stage Ssod -DryRun
```

Dry-run still validates that committed matrix configurations reference the
prepared split, expected pseudo-label artifacts, and a readable shared
initialization checkpoint. It deliberately does not create a final result.

## 5. Full experiment queue

After audit is clean, set `FRUIT_SSOD_PRETRAINED_WEIGHTS` to the approved shared
initialization `.pt` file and retain it unchanged for comparable supervised and
student runs. Run the supervised matrix first:

```powershell
$env:FRUIT_SSOD_PRETRAINED_WEIGHTS = 'D:\approved-weights\yolov8s.pt'
.\scripts\run_pipeline.ps1 -Stage Supervised -Device cuda:0
```

Generate and audit pseudo-label candidates, select thresholds only from
protected validation data, then run the global baseline, three Trust Filter
seeds, and ablations:

```powershell
.\scripts\run_pipeline.ps1 -Stage Ssod -Device cuda:0
```

`-Stage All` simply runs these two controlled matrix launchers in order. It does
not silently manufacture the required pseudo-label preparation/calibration
artifacts. For each completed model, perform fixed primary-test and FruitDet
external-test evaluation, then run `benchmark_model` on the final checkpoint.
Finally, publish an immutable result package with `aggregate_results`; only its
acceptance JSON is the authoritative basis for the requested metric target.

### v12 recovery candidate selection

The v12 recovery has a separate validation-only decision before any new
fixed-test access. Prepare a JSON manifest with completed candidate run paths
and their declared inference settings, for example:

```json
{
  "candidates": [
    {
      "candidate_id": "v12_v8m_direct_1024",
      "run_dir": "E:\\fruit_ssod_runtime\\artifacts_v12\\runs\\run-id",
      "inference": {"mode": "direct", "image_size": 1024, "confidence": 0.001, "nms_iou": 0.7}
    }
  ]
}
```

Run the selector before exactly one fixed-test evaluation:

```powershell
.\scripts\select_v12_validation_candidate.ps1 `
  -CandidateManifest E:\fruit_ssod_runtime\artifacts_v12\v12_candidates.json `
  -Output E:\fruit_ssod_runtime\artifacts_v12\exports\v12_validation_selection.json
```

The selector rejects candidates with a fixed-test JSON already present, requires
the same sealed validation membership (but permits different training views,
such as full images versus training-only tiles), prefers the five-class AP50
floor, then ranks by validation mAP50 and Recall. It does not start training or
evaluate the fixed test.

## 6. Use the desktop demonstrator

Start it with `scripts\start_gui.ps1`. The GUI initially has no active model;
choose a compatible trained five-class `.pt` checkpoint using **Load model**.
Loading happens in a background worker and validates class IDs/names. Use the
single-image page, batch-image page, or video page. Controls remain responsive
while files are processed, and batch/video work can be cancelled. Release the
model or close the program when finished to release its GPU resources.

The video path is file based. Unsupported or corrupt video is reported without
falling back to a camera. It is normal for the GUI to work only after a valid
checkpoint exists; a missing final weight is not replaced by a synthetic model.

## 7. Result locations

All real generated outputs belong under `FRUIT_SSOD_ARTIFACT_ROOT`, not Git:

| Evidence | Location |
| --- | --- |
| Supervised/student run records and checkpoints | `runs/<experiment_name>/` |
| Pseudo candidates, filters, and audit | `pseudo/<experiment_name>/` |
| Threshold calibration evidence | `calibration/` |
| Matrix/result exports | `exports/` |
| Immutable aggregate package | caller-supplied `aggregate_results --output-dir` |
| GUI image/batch/video exports | GUI-selected export folder |

Preserve these files and their manifests. Do not overwrite an immutable run,
benchmark JSON, or aggregate package; choose a new output path instead.
