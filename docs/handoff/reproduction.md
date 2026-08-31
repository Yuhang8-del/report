# Windows Reproduction Handoff

## Environment

Use a Windows-native Conda environment. No WSL or Docker workflow is required.
Install CUDA-enabled PyTorch compatible with the RTX 3080 before installing the
locked project requirements:

```powershell
conda create -n fruit-ssod python=3.10
conda run -n fruit-ssod python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
conda run -n fruit-ssod python -m pip install -r requirements.txt
conda env config vars set -n fruit-ssod PYTHONNOUSERSITE=1
```

Set the data and artifact roots to authorised local or UNC locations; do not
commit these paths or credentials:

```powershell
$env:FRUIT_SSOD_DATA_ROOT = '\\10.16.57.94\dataset2\lyg\detect_datasets\fruit_ssod'
$env:FRUIT_SSOD_ARTIFACT_ROOT = 'E:\fruit_ssod_runtime'
```

## Reproduction order

1. Run `scripts/start_gui.ps1 -PreflightOnly` to verify the environment.
2. Materialize and audit the data snapshot using the committed configuration;
   retain the resulting manifest and split fingerprint.
3. Run validation-only candidate experiments. Select and freeze a candidate
   before performing its single fixed-test evaluation.
4. The customer-directed exploratory execution path does not block on the
   historical 0.80/0.85 accuracy or 0.90 pseudo-label gates. Promote a Teacher
   to candidate generation and Student training once its fixed-test `mAP@0.5`
   is at least 0.60 (record 0.50–0.60 as a lower-confidence exploratory seed).
   Keep `allow_below_precision_gate: true` in exploratory Student configs and
   preserve the measured audit as evidence; protected validation/test labels
   must still remain outside training. The historical formal matrix may still
   be run separately when a formal acceptance claim is required.
5. The current first-result path is sealed by the Student run and package
   recorded in `E:\fruit_ssod_runtime\artifacts_v17\exports`. Its fixed-test
   evidence is bound to the 90-image test membership and its checkpoint hash.
6. To reproduce the post-Student open-world experiment, run
   `src/fruit_ssod/cli/discover_novel_fruits.py` with the completed Student
   `weights/best.pt`, the DeepNIR source root, `--clusters 6`, `--epochs 10`,
   `--batch-size 32` and `--device cuda:0`. The implementation performs
   one-image-at-a-time Student confidence inference (`batch=1`) to keep peak
   memory within a 10-GiB RTX 3080. It writes protected labels, cluster
   assignments, novelty scores and self-supervised checkpoint evidence. The
   output is an image-level Unknown-cluster demonstration; it does not add
   semantic class IDs to the GUI registry.
7. Once all formal experiment rows are complete, aggregate results, run the
   acceptance check and build immutable `report_data.json`.
8. Generate report assets, benchmark the exact designated checkpoint on the
   RTX 3080, build the DOCX/PDF and complete final QA.
9. Build the release manifest from the prepared release directory:

```powershell
.\scripts\build_delivery_manifest.ps1 -ReleaseRoot 'E:\fruit_ssod_release' -Output 'E:\fruit_ssod_release_manifest.json'
```

The PySide6 application supports local image, folder and video input. Camera
control is not part of this project. Open-world discovery is delivered as a
separate evidence-bound offline experiment; semantic runtime registration and
box-level novel-class mAP remain future work.
