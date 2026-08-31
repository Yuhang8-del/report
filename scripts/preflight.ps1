<#
.SYNOPSIS
Checks the Windows host prerequisites for the fruit SSOD project without changing data.

.DESCRIPTION
The script reports each check as PASS, FAIL, INFO, or SKIP. A failure includes
the problem, likely cause, and remediation, and makes the process exit non-zero.
#>
[CmdletBinding()]
param(
    [string]$PythonExecutable = 'E:\anaconda\python.exe',
    [string]$DataRoot = '\\10.16.57.94\dataset2\lyg\detect_datasets',
    [string]$ArtifactRoot = (Join-Path $PSScriptRoot '..\artifacts'),
    [ValidateRange(1, 30)]
    [int]$ReachabilityTimeoutSeconds = 5,
    [switch]$SkipDataRootReachability
)

$ErrorActionPreference = 'Stop'
$script:FailureCount = 0

function Write-Pass {
    param([string]$Check, [string]$Details)
    Write-Output ("[PASS] {0}: {1}" -f $Check, $Details)
}

function Write-Info {
    param([string]$Check, [string]$Details)
    Write-Output ("[INFO] {0}: {1}" -f $Check, $Details)
}

function Write-Skip {
    param([string]$Check, [string]$Details)
    Write-Output ("[SKIP] {0}: {1}" -f $Check, $Details)
}

function Write-Failure {
    param([string]$Check, [string]$Problem, [string]$LikelyCause, [string]$Remediation)
    $script:FailureCount++
    Write-Output ("[FAIL] {0} | Problem: {1} | Likely cause: {2} | Remediation: {3}" -f $Check, $Problem, $LikelyCause, $Remediation)
}

function Test-IsUncDataRoot {
    <#
    .SYNOPSIS
    Returns true only for a complete SMB UNC share path.

    Extended-length local paths begin with ``\\?\C:\`` and must remain local:
    treating every path beginning with two backslashes as a UNC path would let the
    offline skip switch conceal a missing local dataset.  Extended UNC paths use
    the distinct ``\\?\UNC\server\share`` form and are intentionally accepted.
    #>
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    $candidate = $Path.Trim()
    return $candidate -match '^(?:\\\\\?\\UNC\\[^\\/]+\\[^\\/]+(?:\\|$)|\\\\(?![?.\\])[^\\/]+\\[^\\/]+(?:\\|$))'
}

function Get-LocalDriveRoot {
    param([string]$Path)

    $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
    # DriveInfo rejects the extended-length local spelling (\\?\C:\), while
    # its ordinary drive-root equivalent refers to the same local volume.
    if ($root -match '^\\\\\?\\([A-Za-z]:\\)$') {
        return $Matches[1]
    }
    return $root
}

function Test-LocalFreeSpace {
    param([string]$Check, [string]$Path)
    try {
        if (Test-IsUncDataRoot $Path) {
            Write-Failure $Check "Could not determine free space for UNC path '$Path'." "Windows cannot report UNC share capacity through an unmapped drive root, or the share is unavailable." "Map the approved share to a drive with accessible capacity information, or restore network/VPN and share access before rerunning preflight."
            return
        }
        $root = Get-LocalDriveRoot $Path
        $drive = New-Object System.IO.DriveInfo($root)
        $freeGiB = [Math]::Round($drive.AvailableFreeSpace / 1GB, 2)
        $totalGiB = [Math]::Round($drive.TotalSize / 1GB, 2)
        Write-Pass $Check ("{0} GiB free of {1} GiB on {2}" -f $freeGiB, $totalGiB, $root)
    }
    catch {
        Write-Failure $Check "Could not determine free space for '$Path'." $_.Exception.Message "Use an accessible local drive or map the shared storage to a drive."
    }
}

function Test-ConfiguredDataRootFreeSpace {
    param([string]$Path, [bool]$SkipUncReachabilityChecks)
    $root = $null
    try {
        $root = Get-LocalDriveRoot $Path
    }
    catch {
        # Let the normal capacity check below emit its actionable failure for a
        # malformed local path.  A skip switch must not hide local path errors.
        Test-LocalFreeSpace 'Configured data root free space' $Path
        return
    }

    if ($SkipUncReachabilityChecks -and (Test-IsUncDataRoot $Path)) {
        Write-Skip 'Configured data root free space' 'Skipped by -SkipDataRootReachability because UNC share capacity depends on data-root reachability.'
        return
    }

    Test-LocalFreeSpace 'Configured data root free space' $Path
}

function Test-Directory {
    param([string]$Check, [string]$Path, [string]$Remediation)
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Write-Pass $Check $Path
        return $true
    }
    Write-Failure $Check "Directory does not exist: '$Path'." "The configured path is incorrect, unavailable, or not mounted." $Remediation
    return $false
}

function Test-DataRootReachability {
    param([string]$Path, [int]$TimeoutSeconds)
    $job = $null
    try {
        $job = Start-Job -ScriptBlock {
            param([string]$CandidatePath)
            Test-Path -LiteralPath $CandidatePath -PathType Container
        } -ArgumentList $Path
        $completedJob = Wait-Job -Job $job -Timeout $TimeoutSeconds
        if ($null -eq $completedJob) {
            Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
            Write-Failure 'Shared data root reachability' "Timed out after $TimeoutSeconds seconds while checking '$Path'." "The file server, VPN, SMB route, or credentials may be unavailable." "Connect to the required network/VPN, verify SMB access and credentials, then rerun preflight."
            return
        }
        $reachable = [bool](Receive-Job -Job $job -ErrorAction Stop)
        if ($reachable) {
            Write-Pass 'Shared data root reachability' $Path
        }
        else {
            Write-Failure 'Shared data root reachability' "Directory is not reachable: '$Path'." "The server/share is unavailable, the path is wrong, or access is denied." "Connect to the required network/VPN, authenticate to the server, and verify the configured UNC path."
        }
    }
    catch {
        Write-Failure 'Shared data root reachability' "Could not check '$Path': $($_.Exception.Message)" "The file server, VPN, SMB route, or credentials may be unavailable." "Connect to the required network/VPN, verify SMB access and credentials, then rerun preflight."
    }
    finally {
        if ($null -ne $job) {
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-ArtifactWriteDelete {
    param([string]$Path)
    $probeDirectory = Join-Path $Path ("preflight-probe-" + [Guid]::NewGuid().ToString('N'))
    try {
        New-Item -ItemType Directory -Path $probeDirectory -ErrorAction Stop | Out-Null
        $probeFile = Join-Path $probeDirectory 'write-delete-probe.txt'
        Set-Content -LiteralPath $probeFile -Value 'preflight probe' -NoNewline -ErrorAction Stop
        if (-not (Test-Path -LiteralPath $probeFile -PathType Leaf)) {
            throw 'The probe file was not created.'
        }
        Remove-Item -LiteralPath $probeDirectory -Recurse -Force -ErrorAction Stop
        Write-Pass 'Artifact write/delete probe' "Created and removed dedicated temporary subfolder '$probeDirectory'."
    }
    catch {
        Write-Failure 'Artifact write/delete probe' "Could not write and delete inside '$Path': $($_.Exception.Message)" "The artifact root may be read-only, full, or blocked by permissions." "Choose a writable artifact root with free space and update its permissions if necessary."
        if (Test-Path -LiteralPath $probeDirectory) {
            Remove-Item -LiteralPath $probeDirectory -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Output 'Fruit SSOD Windows environment preflight'
Write-Info 'Configured Python executable' $PythonExecutable
Write-Info 'Configured data root' $DataRoot
Write-Info 'Configured artifact root' $ArtifactRoot

if (Test-Path -LiteralPath $PythonExecutable -PathType Leaf) {
    try {
        $pythonVersion = (& $PythonExecutable -c 'import sys; print(sys.version.split()[0])').Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonVersion)) {
            throw 'Python did not return a version.'
        }
        Write-Pass 'Python executable' "$PythonExecutable (Python $pythonVersion)"
    }
    catch {
        Write-Failure 'Python executable' "Could not run '$PythonExecutable': $($_.Exception.Message)" "The selected interpreter is damaged or incompatible." "Install or select the approved Python 3.10 interpreter, then pass it with -PythonExecutable."
    }
}
else {
    Write-Failure 'Python executable' "File does not exist: '$PythonExecutable'." "The interpreter path is incorrect or Anaconda is not installed at the expected location." "Install the approved Python 3.10 interpreter or pass its full path with -PythonExecutable."
}

$nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) { $nvidiaSmi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue }
if ($null -ne $nvidiaSmi) {
    try {
        $gpuRows = & $nvidiaSmi.Source '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader'
        if ($LASTEXITCODE -ne 0 -or $null -eq $gpuRows) { throw 'nvidia-smi did not return GPU details.' }
        Write-Pass 'NVIDIA GPU' (($gpuRows | ForEach-Object { $_.Trim() }) -join '; ')
    }
    catch {
        Write-Failure 'NVIDIA GPU' "Could not query nvidia-smi: $($_.Exception.Message)" "The NVIDIA driver may be missing, unhealthy, or inaccessible." "Install/update the NVIDIA driver and confirm nvidia-smi works in a new terminal."
    }
}
else {
    Write-Failure 'NVIDIA GPU' 'nvidia-smi was not found on PATH.' "The NVIDIA driver is not installed or its utilities are unavailable." "Install/update the NVIDIA driver and add its utilities to PATH, then rerun preflight."
}

if (Test-Path -LiteralPath $PythonExecutable -PathType Leaf) {
    try {
        $torchDetails = (& $PythonExecutable -c 'import torch; print(chr(124).join(map(str, (torch.__version__, torch.version.cuda, torch.cuda.is_available()))))').Trim()
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch import failed.' }
        $torchParts = $torchDetails -split '\|', 3
        if ($torchParts.Count -ne 3) { throw "Unexpected PyTorch output: $torchDetails" }
        $cudaAvailable = $torchParts[2].ToLowerInvariant() -eq 'true'
        if ($cudaAvailable) {
            Write-Pass 'PyTorch CUDA' "torch $($torchParts[0]); CUDA runtime $($torchParts[1]); torch.cuda.is_available()=$($torchParts[2])"
        }
        else {
            Write-Failure 'PyTorch CUDA' "torch $($torchParts[0]); CUDA runtime $($torchParts[1]); torch.cuda.is_available()=$($torchParts[2])." "PyTorch cannot access an NVIDIA CUDA device." "Use a CUDA-enabled PyTorch build and a compatible NVIDIA driver, then rerun preflight."
        }
    }
    catch {
        Write-Failure 'PyTorch CUDA' "Could not import/query PyTorch: $($_.Exception.Message)" "PyTorch is missing or incompatible with the selected Python environment." "Install the project-required CUDA-enabled PyTorch build in the selected environment."
    }
}
else {
    Write-Failure 'PyTorch CUDA' 'Skipped because the selected Python executable is unavailable.' "The selected interpreter could not be checked." "Fix the Python executable path and rerun preflight."
}

$worktreeRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Test-LocalFreeSpace 'Local worktree free space' $worktreeRoot
Test-ConfiguredDataRootFreeSpace $DataRoot ([bool]$SkipDataRootReachability)
Test-LocalFreeSpace 'Configured artifact root free space' $ArtifactRoot

$artifactRootExists = Test-Directory 'Artifact root' $ArtifactRoot 'Create the configured artifact root or pass an existing writable directory with -ArtifactRoot.'
if ($SkipDataRootReachability -and (Test-IsUncDataRoot $DataRoot)) {
    Write-Skip 'Shared data root reachability' 'Skipped by -SkipDataRootReachability; no alternative data root was used.'
}
else {
    Test-DataRootReachability $DataRoot $ReachabilityTimeoutSeconds
}
if ($artifactRootExists) {
    Test-ArtifactWriteDelete $ArtifactRoot
}
else {
    Write-Failure 'Artifact write/delete probe' 'Skipped because the configured artifact root does not exist.' "A writable artifact root is required for the probe." "Create the configured artifact root or pass an existing writable directory with -ArtifactRoot."
}

if ($script:FailureCount -gt 0) {
    Write-Output "Overall result: FAIL ($script:FailureCount required check(s) failed)"
    exit 1
}
Write-Output 'Overall result: PASS'
exit 0
