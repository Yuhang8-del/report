param([string]$EnvName = "fruit-ssod")

$ErrorActionPreference = 'Stop'
$deliveryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Join-Path $deliveryRoot 'project'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:FRUIT_SSOD_DATA_ROOT = Join-Path $deliveryRoot 'data'
$env:FRUIT_SSOD_ARTIFACT_ROOT = Join-Path $deliveryRoot 'artifacts\v17'

$command = Get-Command conda -ErrorAction SilentlyContinue
$candidates = @(
    $(if ($command) { $command.Source }),
    "E:\anaconda\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $candidates) {
    throw "Conda was not found. Run setup_environment.ps1 first."
}
$conda = [string]($candidates | Select-Object -First 1)

Push-Location $projectRoot
try {
    & $conda run --no-capture-output -n $EnvName python (Join-Path $projectRoot 'scripts\open_world_demo.py')
    if ($LASTEXITCODE -ne 0) {
        throw "The open-world GUI failed to start. Exit code: $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
