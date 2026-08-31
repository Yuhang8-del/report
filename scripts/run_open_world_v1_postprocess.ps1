param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$ProjectRoot = 'D:\fruit_ssod_complete_project1\project',
    [string]$DeliveryRoot = 'D:\fruit_ssod_complete_project1',
    [string]$RuntimeRoot = 'E:\fruit_ssod_runtime',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$run = Join-Path $RuntimeRoot 'artifacts_v18\open_world\objectness-yolov8s-seed42'
$record = Join-Path $run 'open_world_run_record.json'
$best = Join-Path $run 'weights\best.pt'
$evaluation = Join-Path $RuntimeRoot 'artifacts_v18\open_world\box-evaluation-seed42'
$log = Join-Path $RuntimeRoot 'artifacts_v18\open_world\open-world-v1-postprocess.log'

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

Write-Log 'waiting for objectness training'
while ($true) {
    if (Test-Path -LiteralPath $record) {
        $state = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
        if ([string]$state.status -eq 'failed') {
            Write-Log 'objectness training failed; postprocess stopped'
            exit 2
        }
        if ([string]$state.status -eq 'complete') {
            break
        }
    }
    Start-Sleep -Seconds $PollSeconds
}

if (-not (Test-Path -LiteralPath $best)) {
    Write-Log "best checkpoint missing: $best"
    exit 3
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
New-Item -ItemType Directory -Force -Path $evaluation | Out-Null
Push-Location $ProjectRoot
try {
    Write-Log 'starting full box-level evaluation'
    & $Python -u -m fruit_ssod.cli.evaluate_open_world_boxes `
        --student-weights (Join-Path $DeliveryRoot 'models\student_best.pt') `
        --objectness-weights $best `
        --encoder-checkpoint (Join-Path $DeliveryRoot 'models\open_world_encoder.pt') `
        --public-manifest (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\novel_public_manifest.json') `
        --protected-truth (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\protected_novel_box_truth.json') `
        --output-dir $evaluation `
        --known-confidence 0.50 `
        --objectness-threshold 0.10 `
        --known-iou-threshold 0.35 `
        --image-size 768 `
        --device 0 *>> (Join-Path $evaluation 'evaluation.log')
    if ($LASTEXITCODE -ne 0) {
        throw "box-level evaluation failed with exit code $LASTEXITCODE"
    }

    $models = Join-Path $DeliveryRoot 'models'
    Copy-Item -LiteralPath $best -Destination (Join-Path $models 'open_world_objectness.pt') -Force
    Copy-Item -LiteralPath (Join-Path $evaluation 'box_clusters\box_cluster_model.npz') -Destination (Join-Path $models 'open_world_box_clusters.npz') -Force
    Copy-Item -LiteralPath (Join-Path $evaluation 'box_clusters\posthoc_cluster_names.json') -Destination (Join-Path $models 'open_world_cluster_names.json') -Force

    $examples = Join-Path $DeliveryRoot 'outputs\customer_open_world_box_examples'
    & $Python (Join-Path $ProjectRoot 'scripts\generate_open_world_box_examples.py') `
        --predictions (Join-Path $evaluation 'box_predictions.jsonl') `
        --assignments (Join-Path $evaluation 'box_clusters\holdout_box_cluster_assignments.jsonl') `
        --protected-truth (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\protected_novel_box_truth.json') `
        --output $examples `
        --per-class 2 *>> (Join-Path $evaluation 'example_generation.log')
    if ($LASTEXITCODE -ne 0) {
        throw "example generation failed with exit code $LASTEXITCODE"
    }
    Write-Log 'open-world V1 postprocess complete'
} catch {
    Write-Log ("postprocess failed: " + $_.Exception.Message)
    throw
} finally {
    Pop-Location
}
