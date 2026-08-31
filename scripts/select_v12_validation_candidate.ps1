<#!
.SYNOPSIS
Publishes the validation-only frozen-candidate decision for v12 recovery.

.DESCRIPTION
Runs the Python selector through a Windows Conda environment. The selector
rejects incomplete runs and candidates that already have a fixed-test result;
it must run before exactly one final held-out test evaluation.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$CandidateManifest,
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Output,
    [ValidateRange(0.0, 1.0)]
    [double]$PerClassAp50Floor = 0.50,
    [ValidateNotNullOrEmpty()]
    [string]$CondaEnvironment = 'fruit-ssod',
    [ValidateNotNullOrEmpty()]
    [string]$CondaExecutable = 'conda'
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'resolve_conda_command.ps1')
$CondaCommand = Resolve-CondaCommand -CondaExecutable $CondaExecutable
$ManifestPath = (Resolve-Path -LiteralPath $CandidateManifest).Path
$OutputPath = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $OutputPath) {
    throw "Validation selection output already exists and will not be overwritten: $OutputPath"
}

$Arguments = @(
    'run', '--no-capture-output', '--name', $CondaEnvironment,
    'python', '-m', 'fruit_ssod.cli.select_validation_candidate',
    '--candidate-manifest', $ManifestPath,
    '--output', $OutputPath,
    '--per-class-ap50-floor', $PerClassAp50Floor.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
Push-Location $RepositoryRoot
try {
    Write-Host ('Validation-only candidate selection: ' + ($CondaCommand, $Arguments -join ' '))
    & $CondaCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Validation candidate selector failed with exit code $LASTEXITCODE. The fixed-test evaluation remains unauthorized."
    }
}
finally {
    Pop-Location
}
