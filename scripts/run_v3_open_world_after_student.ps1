param(
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17',
    [string]$DataRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$studentRunId = 'ssod-v3-teacher-r3-student-seed42'
$studentRun = Join-Path $ArtifactRoot "runs\$studentRunId"
$record = Join-Path $studentRun 'run_record.json'
$evaluation = Join-Path $studentRun 'evaluations\test.json'
$weights = Join-Path $studentRun 'weights\best.pt'
$sourceRoot = Join-Path $DataRoot 'raw\deepnir\extracted_sanitized\yolov5'
$knownTestList = Join-Path $DataRoot 'processed\yolo\supervised_v2_100_seed42\test.txt'
$outputDir = Join-Path $ArtifactRoot "open_world\post_student_$studentRunId"
$log = Join-Path $ArtifactRoot 'v3-open-world-after-student.log'

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
Log "waiting student=$studentRunId"
while ($true) {
    if (-not (Test-Path -LiteralPath $record)) {
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    $state = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    if ([string]$state.status -eq 'failed') {
        Log 'Student failed; open-world stage will not start.'
        exit 2
    }
    if ([string]$state.status -eq 'complete') {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

foreach ($required in @($evaluation, $weights, $sourceRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        Log "required post-Student input missing: $required"
        exit 3
    }
}
if (Test-Path -LiteralPath (Join-Path $outputDir 'discovery_results.json')) {
    Log "open-world result already exists; preserving: $outputDir"
    exit 0
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$env:PYTHONPATH = Join-Path $Worktree 'src'
Push-Location $Worktree
try {
    Log "starting source=$sourceRoot"
    & $Python -m fruit_ssod.cli.discover_novel_fruits `
        --student-weights $weights `
        --source-root $sourceRoot `
        --output-dir $outputDir `
        --seed 42 `
        --holdout-fraction 0.20 `
        --clusters 6 `
        --epochs 10 `
        --batch-size 32 `
        --learning-rate 0.0001 `
        --device cuda:0 `
        --image-size 768 `
        --known-test-list $knownTestList `
        --novelty-threshold 0.50 *>> (Join-Path $outputDir 'discovery.log')
    if ($LASTEXITCODE -ne 0) { throw "open-world discovery failed with exit code $LASTEXITCODE" }
    Log "completed output=$outputDir"
} finally {
    Pop-Location
}
