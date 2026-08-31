param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$ProjectRoot = 'D:\fruit_ssod_complete_project1\project',
    [string]$DeliveryRoot = 'D:\fruit_ssod_complete_project1',
    [string]$RuntimeRoot = 'E:\fruit_ssod_runtime',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$openWorldResult = Join-Path $RuntimeRoot 'artifacts_v18\open_world\box-evaluation-seed42\open_world_box_results.json'
$runRoot = Join-Path $RuntimeRoot 'artifacts_v18\open_world'
$runName = 'incremental-all6-student-seed42-r2'
$record = Join-Path $runRoot "$runName\incremental_run_record.json"
$log = Join-Path $runRoot 'incremental-all6-queue.log'

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

Write-Log 'waiting for box-level open-world evaluation'
while (-not (Test-Path -LiteralPath $openWorldResult)) {
    Start-Sleep -Seconds $PollSeconds
}
if (Test-Path -LiteralPath $record) {
    $existing = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    if ([string]$existing.status -in @('running', 'complete')) {
        Write-Log "incremental run already $($existing.status); preserving"
        exit 0
    }
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Push-Location $ProjectRoot
try {
    Write-Log 'starting reviewed 11-class incremental training'
    # Do not merge native stderr into the PowerShell success stream: warnings
    # from optional Polars imports are non-fatal and must not stop training.
    & $Python -u -m fruit_ssod.cli.train_incremental_open_world_detector `
        --base-weights (Join-Path $DeliveryRoot 'models\student_best.pt') `
        --dataset (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_incremental_all6_seed42\dataset.yaml') `
        --project $runRoot `
        --name $runName `
        --epochs 30 `
        --image-size 640 `
        --batch-size 4 `
        --workers 4 `
        --patience 8 `
        --seed 42 `
        --device 0 >> (Join-Path $runRoot 'incremental-all6.stdout.log') 2>> (Join-Path $runRoot 'incremental-all6.stderr.log')
    if ($LASTEXITCODE -ne 0) {
        throw "incremental training failed with exit code $LASTEXITCODE"
    }
    Copy-Item -LiteralPath (Join-Path $runRoot "$runName\weights\best.pt") -Destination (Join-Path $DeliveryRoot 'models\incremental_11class_best.pt') -Force
    Copy-Item -LiteralPath (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_incremental_all6_seed42\class_registry_v2.json') -Destination (Join-Path $DeliveryRoot 'models\class_registry_v2.json') -Force
    Write-Log 'incremental training complete and copied to delivery models'
} catch {
    Write-Log ("incremental training failed: " + $_.Exception.Message)
    throw
} finally {
    Pop-Location
}
