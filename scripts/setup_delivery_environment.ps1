param(
    [string]$EnvName = "fruit-ssod"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Find-Conda {
    $command = Get-Command conda -ErrorAction SilentlyContinue
    $candidates = @(
        $(if ($command) { $command.Source }),
        "E:\anaconda\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $candidates) { throw "Conda was not found. Install Anaconda or Miniconda first." }
    return [string]($candidates | Select-Object -First 1)
}

$Conda = Find-Conda
$envInfo = & $Conda env list --json | ConvertFrom-Json
$exists = $envInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq $EnvName }
if (-not $exists) {
    Write-Host "Creating Conda environment $EnvName (Python 3.10)..."
    & $Conda create -y -n $EnvName python=3.10 pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Conda environment '$EnvName'." }
}

Write-Host "Installing locked dependencies. The first run may take a while..."
& $Conda run --no-capture-output -n $EnvName python -m pip install -r "$ProjectRoot\requirements-lock.txt"
if ($LASTEXITCODE -ne 0) { throw "Failed to install locked Python dependencies." }
& $Conda run --no-capture-output -n $EnvName python -m pip install -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install the fruit-ssod project package." }
& $Conda env config vars set -n $EnvName PYTHONNOUSERSITE=1
if ($LASTEXITCODE -ne 0) { throw "Failed to configure PYTHONNOUSERSITE for '$EnvName'." }
& $Conda run --no-capture-output -n $EnvName python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency consistency check failed." }
Write-Host "Environment setup completed. Run run_gui.ps1 from the delivery root."
