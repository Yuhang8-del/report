param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$ProjectRoot = 'D:\fruit_ssod_complete_project1\project',
    [string]$DeliveryRoot = 'D:\fruit_ssod_complete_project1',
    [string]$RuntimeRoot = 'E:\fruit_ssod_runtime',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$runRoot = Join-Path $RuntimeRoot 'artifacts_v18\open_world'
$training = Join-Path $runRoot 'incremental-all6-student-seed42-r2'
$record = Join-Path $training 'incremental_run_record.json'
$weights = Join-Path $training 'weights\best.pt'
$evaluation = Join-Path $runRoot 'incremental-all6-protected-evaluation'
$metrics = Join-Path $evaluation 'protected_novel_holdout_metrics.json'
$log = Join-Path $runRoot 'incremental-evaluation-queue.log'

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

Write-Log 'waiting for reviewed 11-class incremental training'
while ($true) {
    if (Test-Path -LiteralPath $record) {
        $state = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
        if ([string]$state.status -eq 'complete') { break }
        if ([string]$state.status -eq 'failed') {
            Write-Log 'incremental training failed; evaluation stopped'
            exit 1
        }
    }
    Start-Sleep -Seconds $PollSeconds
}

if (Test-Path -LiteralPath $metrics) {
    Write-Log 'protected evaluation already exists; preserving'
    exit 0
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Path (Join-Path $DeliveryRoot 'models') -Force | Out-Null
    Copy-Item -LiteralPath $weights -Destination (Join-Path $DeliveryRoot 'models\incremental_11class_best.pt') -Force
    Copy-Item -LiteralPath (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_incremental_all6_seed42\class_registry_v2.json') -Destination (Join-Path $DeliveryRoot 'models\class_registry_v2.json') -Force
    Write-Log 'starting protected novel-class evaluation'
    & $Python -u -m fruit_ssod.cli.evaluate_incremental_open_world `
        --weights $weights `
        --dataset (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_novel_holdout_eval_seed42\dataset.yaml') `
        --output $evaluation `
        --image-size 640 `
        --batch-size 4 `
        --device 0 >> (Join-Path $runRoot 'incremental-protected-evaluation.stdout.log') 2>> (Join-Path $runRoot 'incremental-protected-evaluation.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw "protected evaluation failed with exit code $LASTEXITCODE" }
    $exampleOutput = Join-Path $DeliveryRoot 'outputs\customer_incremental_11class_examples'
    Write-Log 'generating customer examples from protected holdout'
    & $Python -u (Join-Path $ProjectRoot 'scripts\generate_incremental_open_world_examples.py') `
        --weights $weights `
        --protected-truth (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\protected_novel_box_truth.json') `
        --output $exampleOutput `
        --confidence 0.25 `
        --image-size 640 `
        --device 0 `
        --per-class 2 >> (Join-Path $runRoot 'incremental-example-generation.stdout.log') 2>> (Join-Path $runRoot 'incremental-example-generation.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw "example generation failed with exit code $LASTEXITCODE" }
    Copy-Item -LiteralPath $metrics -Destination (Join-Path $DeliveryRoot 'outputs\incremental_11class_protected_metrics.json') -Force
    Write-Log 'incremental evaluation and customer examples complete'
} catch {
    Write-Log ("incremental evaluation failed: " + $_.Exception.Message)
    throw
} finally {
    Pop-Location
}
