param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17',
    [string]$RunId = 'ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-noamp-fg-seed42',
    [string]$FixedData = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\supervised_v2_100_seed42\dataset.yaml',
    [string]$StudentSplit = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\ssod_unlabeled_pool_fruits360_v2_openimages_v13b\split_manifest.json',
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
$runDir = Join-Path $ArtifactRoot ("runs\{0}" -f $runId)
$recordPath = Join-Path $runDir 'run_record.json'
$safeRunId = $RunId -replace '[^A-Za-z0-9_.-]', '_'
$log = Join-Path $ArtifactRoot ("foreground-student-fixed-test-{0}.log" -f $safeRunId)
$env:PYTHONPATH = Join-Path $Worktree 'src'

function Log([string]$Message) { Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8 }

Log "Waiting for foreground Student terminal record: $runId"
while (-not (Test-Path -LiteralPath $recordPath)) { Start-Sleep -Seconds $PollSeconds }
while ($true) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ([string]$record.status -eq 'failed') { Log 'Student failed; fixed-test evaluation will not run.'; exit 2 }
    if ([string]$record.status -eq 'complete') { break }
    Start-Sleep -Seconds $PollSeconds
}
$checkpoint = Join-Path $runDir 'weights\best.pt'
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Student completed without best.pt: $checkpoint" }
$evaluation = Join-Path $runDir 'evaluations\test.json'
if (Test-Path -LiteralPath $evaluation) { Log "Fixed-test evidence already exists: $evaluation"; exit 0 }
Push-Location $Worktree
try {
    Log "Running sealed fixed-test evaluation for $runId"
    & $Python -m fruit_ssod.cli.evaluate_student_test --run-dir $runDir --data $fixedData --split-manifest $studentSplit --device cuda:0 *>> $log
    $code = $LASTEXITCODE
    Log "Fixed-test evaluator exit=$code"
    exit $code
} finally { Pop-Location }
