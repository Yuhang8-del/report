param(
    [string]$EnvName = "fruit-ssod"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DeliveryRoot = Split-Path -Parent $ProjectRoot
$env:FRUIT_SSOD_DATA_ROOT = "$DeliveryRoot\data"
$env:FRUIT_SSOD_ARTIFACT_ROOT = "$DeliveryRoot\artifacts\v17"
$env:FRUIT_SSOD_DEFAULT_MODEL = "$DeliveryRoot\models\student_best.pt"
$env:PYTHONPATH = "$ProjectRoot\src"
$env:PYTHONNOUSERSITE = "1"

$command = Get-Command conda -ErrorAction SilentlyContinue
$candidates = @(
    $(if ($command) { $command.Source }),
    "E:\anaconda\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $candidates) { throw "Conda was not found. Run setup_environment.ps1 first." }

$Conda = [string]($candidates | Select-Object -First 1)
& $Conda run --no-capture-output -n $EnvName python "$PSScriptRoot\delivery_gui.py"
if ($LASTEXITCODE -ne 0) { throw "Delivery GUI exited with code $LASTEXITCODE." }
