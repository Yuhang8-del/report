# Windows environment preflight

The tested Windows host uses `E:\anaconda\python.exe` (Python 3.10.9), PyTorch
2.5.1+cu121 with CUDA available, and an NVIDIA RTX 3080 with 10 GB of VRAM.
`nvidia-smi` is used during preflight to report the actual GPU name, driver, and
total memory on the current host.

## Storage locations

The approved shared dataset root is:

```text
\\10.16.57.94\dataset2\lyg\detect_datasets
```

Do not substitute a local data root when this share is unavailable. The
preflight does not download, copy, or alter data. Its default artifact location
is the repository `artifacts` directory; pass another writable artifact path
explicitly if the project convention changes.

## Run the preflight

From the repository root in native PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1
```

For a short, non-interactive diagnostic of the default UNC path, lower the
reachability timeout. To inspect only local prerequisites while intentionally
skipping data-root network checks, use the explicit skip switch; it does not
select a replacement data root. For a UNC data root it skips both the shared
directory reachability test and the data-root free-space query, because both
depend on contacting that share. It still checks local worktree/artifact free
space and the artifact write/delete probe. With a local data root, its local
free-space check still runs.

```powershell
.\scripts\preflight.ps1 -ReachabilityTimeoutSeconds 3
.\scripts\preflight.ps1 -SkipDataRootReachability
```

For a local smoke test, pass both roots explicitly:

```powershell
.\scripts\preflight.ps1 `
  -PythonExecutable E:\anaconda\python.exe `
  -DataRoot E:\temp\fruit-data `
  -ArtifactRoot E:\temp\fruit-artifacts `
  -ReachabilityTimeoutSeconds 2
```

The preflight reports selected Python/version, NVIDIA GPU/driver/VRAM,
PyTorch/CUDA status, and free space for the worktree and configured roots. It
uses a bounded shared-root check. Its only write is a create/delete probe inside
a uniquely named temporary subfolder of the configured artifact root. A required
failure exits non-zero and states the problem, likely cause, and remediation.

If `\\10.16.57.94\dataset2\lyg\detect_datasets` cannot be reached, connect to
the required LAN or VPN, authenticate to the file server, and verify that your
account has access to the share. Then rerun preflight; do not change the
configured data root as a workaround.

The server-side dataset mount can also be verified through the configured SSH
alias (rather than a direct host/port):

```powershell
ssh trx50-ai-top 'test -d /mnt/dataset2/lyg/detect_datasets && echo available'
```

This is a documented alternative check only. The PowerShell preflight does not
invoke SSH because aliases and authentication may be interactive.

## Eventual project virtual environment (Task 2)

Task 2 will create the project Python 3.10 environment. A Conda environment is
recommended, but the existing tested Conda interpreter is also supported. Until
then, use the tested interpreter above for diagnostics.

Recommended new Conda environment:

```powershell
conda create -n fruit-ssod python=3.10
conda activate fruit-ssod
python --version
```

Alternatively, use the existing Conda environment directly and pass its full
Python executable path to `-PythonExecutable`. A standard `.venv` is optional,
not required:

```powershell
E:\anaconda\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Dependencies will be installed only by Task 2. This preflight task creates no
environment, installs no packages, downloads no dataset, and makes no data
changes.
