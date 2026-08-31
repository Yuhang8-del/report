param(
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17',
    [string]$DataRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$log = Join-Path $ArtifactRoot 'postprocess-best-aggressive-student.log'
$env:PYTHONPATH = Join-Path $Worktree 'src'

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

Log 'Waiting for a completed aggressive Teacher-derived Student fixed-test evaluation.'
$student = $null
while ($null -eq $student) {
    $runs = @(Get-ChildItem -LiteralPath (Join-Path $ArtifactRoot 'runs') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'ssod-v2-independent-openimages-aggressive-best-*-seed42' } | Sort-Object LastWriteTime -Descending)
    foreach ($candidate in $runs) {
        $recordPath = Join-Path $candidate.FullName 'run_record.json'
        $evaluationPath = Join-Path $candidate.FullName 'evaluations\test.json'
        if (-not (Test-Path -LiteralPath $recordPath)) { continue }
        try { $status = [string](Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
        if ($status -eq 'failed') {
            Log "Skipping failed Student diagnostic: $($candidate.Name)"
            continue
        }
        if ($status -eq 'complete' -and (Test-Path -LiteralPath $evaluationPath) -and (Test-Path -LiteralPath (Join-Path $candidate.FullName 'weights\best.pt'))) {
            $student = $candidate
            break
        }
    }
    if ($null -eq $student) { Start-Sleep -Seconds $PollSeconds }
}

$runId = $student.Name
$checkpoint = Join-Path $student.FullName 'weights\best.pt'
$testList = Join-Path $DataRoot 'processed\yolo\supervised_v2_100_seed42\test.txt'
$testImages = @(Get-Content -LiteralPath $testList | Where-Object { $_.Trim() } | Select-Object -First 3)
$gui = Join-Path $ArtifactRoot ("exports\gui_{0}" -f $runId)
$package = Join-Path $ArtifactRoot ("exports\exploratory_{0}_package" -f $runId)
Push-Location $Worktree
try {
    if (-not (Test-Path -LiteralPath $gui)) {
        Log "Exporting PySide6-compatible offline GUI evidence for $runId."
        & $Python scripts\export_gui_candidate.py --checkpoint $checkpoint --output $gui --images $testImages *>> (Join-Path $student.FullName 'postprocess.log')
        if ($LASTEXITCODE -ne 0) { throw "GUI export failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath $package)) {
        Log "Building evidence-bound exploratory package for $runId."
        & $Python -m fruit_ssod.cli.build_exploratory_package --run-dir $student.FullName --gui-export $gui --output $package *>> (Join-Path $student.FullName 'postprocess.log')
        if ($LASTEXITCODE -ne 0) { throw "exploratory package failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $package 'exploratory_best_result_report.pdf'))) {
        Log "Building DOCX/PDF exploratory report for $runId."
        & $Python scripts\build_exploratory_report.py --package $package *>> (Join-Path $student.FullName 'postprocess.log')
        if ($LASTEXITCODE -ne 0) { throw "exploratory report failed with exit code $LASTEXITCODE" }
    }
    Log "Completed Student GUI/report post-processing: $package"
} finally {
    Pop-Location
}
