[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Resume,
    [ValidateNotNullOrEmpty()]
    [string]$CondaEnvironment = 'fruit-ssod',
    [ValidateNotNullOrEmpty()]
    [string]$CondaExecutable = 'conda',
    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0'
)

# Native-Windows Task 17 launcher.  It prints the entire validated queue
# before a train_student invocation can touch CUDA.  No shell command string
# is composed: paths with spaces remain single argument-vector elements.
$ErrorActionPreference = 'Stop'
# Prevent user-site packages from changing the audited training runtime.
$env:PYTHONNOUSERSITE = '1'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ConfigDirectory = Join-Path $RepositoryRoot 'configs\experiments'
. (Join-Path $PSScriptRoot 'resolve_conda_command.ps1')
$CondaCommand = Resolve-CondaCommand -CondaExecutable $CondaExecutable

if ($Resume -and -not $env:FRUIT_SSOD_ARTIFACT_ROOT) {
    throw '-Resume requires FRUIT_SSOD_ARTIFACT_ROOT so existing run records can be compared with current fingerprints.'
}

$ValidationArguments = @(
    'run', '--no-capture-output', '--name', $CondaEnvironment,
    'python', '-m', 'fruit_ssod.cli.validate_ssod_matrix',
    '--config-directory', $ConfigDirectory, '--queue', '--verify-preparation'
)
if ($Resume) {
    $ValidationArguments += @('--resume', '--artifact-root', $env:FRUIT_SSOD_ARTIFACT_ROOT)
}
Write-Host ('SSOD matrix preflight: ' + ($CondaCommand, $ValidationArguments -join ' '))
$QueueOutput = & $CondaCommand @ValidationArguments 2>&1
if ($LASTEXITCODE -ne 0) {
    $QueueOutput | ForEach-Object { Write-Host $_ }
    throw 'SSOD matrix configuration preflight failed. Restore controlled configs before requesting GPU work.'
}
try {
    $QueuePayload = (($QueueOutput -join [Environment]::NewLine) | ConvertFrom-Json -ErrorAction Stop)
} catch {
    throw "SSOD matrix preflight did not return readable queue JSON: $($_.Exception.Message)"
}
if ($QueuePayload.status -ne 'valid' -or -not $QueuePayload.queue) {
    throw 'SSOD matrix preflight returned no valid queue.'
}

Write-Host 'Full SSOD queue (printed before any GPU request):'
foreach ($Entry in $QueuePayload.queue) {
    Write-Host ("  [{0}] {1} seed={2} role={3}: {4}" -f $Entry.action, $Entry.experiment_name, $Entry.seed, $Entry.role, $Entry.reason)
}

foreach ($Entry in $QueuePayload.queue) {
    if (-not $DryRun -and $Entry.action -eq 'skip') {
        Write-Host ("Resume skip: {0} ({1})" -f $Entry.experiment_name, $Entry.reason)
        continue
    }
    $Arguments = @(
        'run', '--no-capture-output', '--name', $CondaEnvironment,
        'python', '-m', 'fruit_ssod.cli.train_student', '--config', $Entry.config
    )
    if ($DryRun) {
        # train_student creates a disposable dry-run ID, intentionally leaving
        # the fixed real matrix directory available for a later GPU execution.
        $Arguments += '--dry-run'
    } else {
        $Arguments += @('--run-id', $Entry.experiment_name, '--device', $Device)
    }
    Write-Host ('Queue entry: ' + ($CondaCommand, $Arguments -join ' '))
    $OutputLines = & $CondaCommand @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    $OutputLines | ForEach-Object { Write-Host $_ }
    if ($ExitCode -ne 0) {
        if ($DryRun) {
            throw "SSOD configuration dry-run failed: $($Entry.experiment_name) (exit code $ExitCode). Correct it before GPU work."
        }
        # StudentTrainingRunner retains a terminal failed run record whenever
        # it reached artifact publication.  Continue to expose every later
        # matrix attempt; Task 18 aggregation will show absent/failed evidence.
        Write-Warning "SSOD matrix entry failed: $($Entry.experiment_name) (exit code $ExitCode). Continuing queue."
    }
}

if ($DryRun) {
    Write-Host 'All SSOD configurations completed dry-run validation. No GPU training or fixed-test evaluation was started.'
} else {
    Write-Host 'SSOD training queue finished. Run the fixed-test evaluation and immutable aggregation stages before interpreting results.'
}
