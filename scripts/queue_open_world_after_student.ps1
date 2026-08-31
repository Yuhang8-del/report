param(
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17',
    [string]$DataRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$log = Join-Path $ArtifactRoot 'open-world-after-student-queue.log'
$runPattern = 'ssod-v2-independent-openimages-aggressive-best-*-seed42'
$sourceRoot = Join-Path $DataRoot 'raw\deepnir\extracted_sanitized\yolov5'
$knownTestList = Join-Path $DataRoot 'processed\yolo\supervised_v2_100_seed42\test.txt'
$env:PYTHONPATH = Join-Path $Worktree 'src'

function Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
    Write-Output $line
}

New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
Log 'Waiting for the v2-selected Student terminal record and fixed-test evidence.'

$student = $null
while ($null -eq $student) {
    $runsRoot = Join-Path $ArtifactRoot 'runs'
    $candidates = @(Get-ChildItem -LiteralPath $runsRoot -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like $runPattern })
    foreach ($candidate in ($candidates | Sort-Object LastWriteTime -Descending)) {
        $recordPath = Join-Path $candidate.FullName 'run_record.json'
        if (-not (Test-Path -LiteralPath $recordPath)) { continue }
        try { $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json } catch { continue }
        if ([string]$record.status -eq 'complete') {
            $evaluation = Join-Path $candidate.FullName 'evaluations\test.json'
            $weights = Join-Path $candidate.FullName 'weights\best.pt'
            if ((Test-Path -LiteralPath $evaluation) -and (Test-Path -LiteralPath $weights)) {
                $student = $candidate
                break
            }
        }
        if ([string]$record.status -eq 'failed') {
            # Failed diagnostic attempts are expected during the Windows
            # recovery sequence.  Keep waiting for a later non-overwriting
            # Student run instead of allowing an old failure to block the
            # customer-authorized open-world stage.
            Log "Skipping failed Student diagnostic: $($candidate.Name)"
            continue
        }
    }
    if ($null -eq $student) { Start-Sleep -Seconds $PollSeconds }
}

$studentWeights = Join-Path $student.FullName 'weights\best.pt'
$outputDir = Join-Path $ArtifactRoot ("open_world\post_student_{0}" -f $student.Name)
if (Test-Path -LiteralPath (Join-Path $outputDir 'discovery_results.json')) {
    Log "Open-world result already exists; refusing to overwrite: $outputDir"
    exit 0
}
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Log "Novel source root is unavailable: $sourceRoot"
    exit 3
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Log "Starting post-Student novel-fruit discovery from $($student.Name)."
Push-Location $Worktree
try {
    & $Python -m fruit_ssod.cli.discover_novel_fruits `
        --student-weights $studentWeights `
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
    Log "Completed post-Student open-world discovery: $outputDir"
} finally {
    Pop-Location
}
