<#
.SYNOPSIS
Runs the controlled supervised and SSOD experiment launchers on Windows.

.DESCRIPTION
This wrapper performs host preflight before any training process.  It does not
create data, generate pseudo labels, or claim an acceptance result; those
artifacts must be prepared and audited as documented in docs/user-guide.md.
#>
[CmdletBinding()]
param(
    [ValidateSet('Supervised', 'Ssod', 'All')]
    [string]$Stage = 'All',
    [switch]$DryRun,
    [switch]$Resume,
    [ValidateNotNullOrEmpty()]
    [string]$Device = 'cuda:0',
    [string]$CondaEnvironment,
    [ValidateNotNullOrEmpty()]
    [string]$CondaExecutable = 'conda',
    [string]$DataRoot,
    [string]$ArtifactRoot,
    [string]$PretrainedWeights,
    [switch]$SkipDataRootReachability
)

$ErrorActionPreference = 'Stop'
# Keep Conda launches deterministic even if this PowerShell session was opened
# from a user Python installation with incompatible Qt/PySide bindings.
$env:PYTHONNOUSERSITE = '1'
$BoundSettings = $PSBoundParameters
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PreflightScript = Join-Path $PSScriptRoot 'preflight.ps1'
$SupervisedLauncher = Join-Path $PSScriptRoot 'run_supervised_matrix.ps1'
$SsodLauncher = Join-Path $PSScriptRoot 'run_ssod_matrix.ps1'
. (Join-Path $PSScriptRoot 'resolve_conda_command.ps1')

function Resolve-Setting {
    param([string]$BoundName, [string]$ParameterValue, [string]$EnvironmentName, [string]$Fallback)
    if ($BoundSettings.ContainsKey($BoundName) -and -not [string]::IsNullOrWhiteSpace($ParameterValue)) { return $ParameterValue }
    $environmentValue = [Environment]::GetEnvironmentVariable($EnvironmentName)
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) { return $environmentValue }
    return $Fallback
}

$EffectiveCondaEnvironment = Resolve-Setting 'CondaEnvironment' $CondaEnvironment 'FRUIT_SSOD_CONDA_ENV' 'fruit-ssod'
$EffectiveDataRoot = Resolve-Setting 'DataRoot' $DataRoot 'FRUIT_SSOD_DATA_ROOT' '\\10.16.57.94\dataset2\lyg\detect_datasets'
$EffectiveArtifactRoot = Resolve-Setting 'ArtifactRoot' $ArtifactRoot 'FRUIT_SSOD_ARTIFACT_ROOT' (Join-Path $RepositoryRoot 'artifacts')
$EffectivePretrainedWeights = Resolve-Setting 'PretrainedWeights' $PretrainedWeights 'FRUIT_SSOD_PRETRAINED_WEIGHTS' ''
$CondaCommand = Resolve-CondaCommand -CondaExecutable $CondaExecutable

function Resolve-CondaPython {
    $arguments = @('run', '--no-capture-output', '--name', $EffectiveCondaEnvironment, 'python', '-c', 'import sys; print(sys.executable)')
    $output = & $CondaCommand @arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Unable to run Python in Conda environment '$EffectiveCondaEnvironment'. Create it and install project dependencies, or pass -CondaEnvironment." }
    $python = ($output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ } | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Conda environment '$EffectiveCondaEnvironment' did not return a usable Python executable." }
    return $python
}

if ([string]::IsNullOrWhiteSpace($EffectivePretrainedWeights)) {
    throw 'FRUIT_SSOD_PRETRAINED_WEIGHTS (or -PretrainedWeights) is required. Point it to the approved shared pretrained checkpoint before matrix validation or training.'
}
if (-not (Test-Path -LiteralPath $EffectivePretrainedWeights -PathType Leaf)) {
    throw "Pretrained weights do not exist: $EffectivePretrainedWeights. Use the approved local .pt checkpoint and retry."
}

# Provide these only to child commands of this invocation, making path overrides
# explicit without editing checked-in YAML files.
$env:FRUIT_SSOD_DATA_ROOT = $EffectiveDataRoot
$env:FRUIT_SSOD_ARTIFACT_ROOT = $EffectiveArtifactRoot
$env:FRUIT_SSOD_PRETRAINED_WEIGHTS = $EffectivePretrainedWeights
$PythonExecutable = Resolve-CondaPython
$PreflightArguments = @{
    PythonExecutable = $PythonExecutable
    DataRoot = $EffectiveDataRoot
    ArtifactRoot = $EffectiveArtifactRoot
}
if ($SkipDataRootReachability) { $PreflightArguments.SkipDataRootReachability = $true }
Write-Host "Training preflight (Conda environment: $EffectiveCondaEnvironment)"
& $PreflightScript @PreflightArguments
if ($LASTEXITCODE -ne 0) { throw 'Training preflight failed. Correct the reported prerequisite before queueing any training process.' }

function Invoke-Supervised {
    $arguments = @('-CondaEnvironment', $EffectiveCondaEnvironment, '-CondaExecutable', $CondaCommand)
    if ($DryRun) { $arguments += '-DryRun' }
    & $SupervisedLauncher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Supervised matrix launcher failed with code $LASTEXITCODE." }
}

function Invoke-Ssod {
    $arguments = @('-CondaEnvironment', $EffectiveCondaEnvironment, '-CondaExecutable', $CondaCommand, '-Device', $Device)
    if ($DryRun) { $arguments += '-DryRun' }
    if ($Resume) { $arguments += '-Resume' }
    & $SsodLauncher @arguments
    if ($LASTEXITCODE -ne 0) { throw "SSOD matrix launcher failed with code $LASTEXITCODE." }
}

Push-Location $RepositoryRoot
try {
    switch ($Stage) {
        'Supervised' { Invoke-Supervised }
        'Ssod' { Invoke-Ssod }
        'All' {
            Invoke-Supervised
            Invoke-Ssod
            Write-Host 'Both matrices finished. Run fixed-test/FruitDet evaluation, benchmark, aggregation, and acceptance separately as documented; no final metric is implied by this launcher.'
        }
    }
}
finally { Pop-Location }
