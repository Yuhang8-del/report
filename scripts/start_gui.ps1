<#
.SYNOPSIS
Starts the file-based Fruit SSOD demonstrator from a named Conda environment.

.DESCRIPTION
The launcher resolves every repository file relative to this script.  It first
checks the selected Conda interpreter, storage roots, CUDA prerequisites, and
the package import before starting Qt.  It never opens a camera and does not
enable the deferred open-world extension.
#>
[CmdletBinding()]
param(
    [string]$CondaEnvironment,
    [ValidateNotNullOrEmpty()]
    [string]$CondaExecutable = 'conda',
    [string]$DataRoot,
    [string]$ArtifactRoot,
    [switch]$SkipDataRootReachability,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
# The selected Conda environment is the only supported dependency source.
$env:PYTHONNOUSERSITE = '1'
$BoundSettings = $PSBoundParameters
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PreflightScript = Join-Path $PSScriptRoot 'preflight.ps1'
. (Join-Path $PSScriptRoot 'resolve_conda_command.ps1')

function Resolve-Setting {
    param([string]$BoundName, [string]$ParameterValue, [string]$EnvironmentName, [string]$Fallback)
    if ($BoundSettings.ContainsKey($BoundName) -and -not [string]::IsNullOrWhiteSpace($ParameterValue)) { return $ParameterValue }
    $environmentValue = [Environment]::GetEnvironmentVariable($EnvironmentName)
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) { return $environmentValue }
    return $Fallback
}

# Keep these defaults aligned with .env.example.  A parameter is preferable for
# one invocation; an environment variable is convenient for a local Conda shell.
$EffectiveCondaEnvironment = Resolve-Setting 'CondaEnvironment' $CondaEnvironment 'FRUIT_SSOD_CONDA_ENV' 'fruit-ssod'
$EffectiveDataRoot = Resolve-Setting 'DataRoot' $DataRoot 'FRUIT_SSOD_DATA_ROOT' '\\10.16.57.94\dataset2\lyg\detect_datasets'
$EffectiveArtifactRoot = Resolve-Setting 'ArtifactRoot' $ArtifactRoot 'FRUIT_SSOD_ARTIFACT_ROOT' (Join-Path $RepositoryRoot 'artifacts')
$CondaCommand = Resolve-CondaCommand -CondaExecutable $CondaExecutable

function Resolve-CondaPython {
    $arguments = @('run', '--no-capture-output', '--name', $EffectiveCondaEnvironment, 'python', '-c', 'import sys; print(sys.executable)')
    $output = & $CondaCommand @arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Unable to run Python in Conda environment '$EffectiveCondaEnvironment'. Create it and install project dependencies, or pass -CondaEnvironment." }
    $python = ($output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ } | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Conda environment '$EffectiveCondaEnvironment' did not return a usable Python executable." }
    return $python
}

$PythonExecutable = Resolve-CondaPython
$PreflightArguments = @{
    PythonExecutable = $PythonExecutable
    DataRoot = $EffectiveDataRoot
    ArtifactRoot = $EffectiveArtifactRoot
}
if ($SkipDataRootReachability) { $PreflightArguments.SkipDataRootReachability = $true }
Write-Host "GUI preflight (Conda environment: $EffectiveCondaEnvironment)"
& $PreflightScript @PreflightArguments
if ($LASTEXITCODE -ne 0) { throw 'GUI preflight failed. Correct the reported prerequisite before starting the application.' }

$ImportArguments = @('run', '--no-capture-output', '--name', $EffectiveCondaEnvironment, 'python', '-c', 'import fruit_ssod.gui.app, PySide6; print(1)')
& $CondaCommand @ImportArguments
if ($LASTEXITCODE -ne 0) { throw "The selected Conda environment cannot import the GUI. Install requirements.txt in '$EffectiveCondaEnvironment'." }
Write-Host 'GUI import check: PASS'
if ($PreflightOnly) { Write-Host 'GUI preflight completed; application was not started (-PreflightOnly).'; exit 0 }

Push-Location $RepositoryRoot
try {
    $LaunchArguments = @('run', '--no-capture-output', '--name', $EffectiveCondaEnvironment, 'python', '-m', 'fruit_ssod.gui.app')
    Write-Host "Starting the file-based GUI from $RepositoryRoot"
    & $CondaCommand @LaunchArguments
    if ($LASTEXITCODE -ne 0) { throw "GUI exited with code $LASTEXITCODE." }
}
finally { Pop-Location }
