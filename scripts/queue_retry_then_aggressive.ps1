param(
    [int]$PollSeconds = 30,
    [string]$RetryRun = 'E:\fruit_ssod_runtime\artifacts_v17\runs\ssod-v1-independent-openimages-v12teacher-seed42-r1'
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$retryRecord = Join-Path $RetryRun 'run_record.json'
$teacherRun = 'E:\fruit_ssod_runtime\artifacts_v17\runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1'
$teacherRecord = Join-Path $teacherRun 'run_record.json'
$teacherLog = 'E:\fruit_ssod_runtime\artifacts_v17\teacher-aggressive-after-v12-retry.log'

while (-not (Test-Path -LiteralPath $retryRecord)) { Start-Sleep -Seconds $PollSeconds }
do {
    try { $status = [string](Get-Content -LiteralPath $retryRecord -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
    if ($status -ne 'running') { break }
    Start-Sleep -Seconds $PollSeconds
} while ($true)

if (-not (Test-Path -LiteralPath $teacherRecord)) {
    Push-Location $work
    try {
        & $python -m fruit_ssod.cli.train_supervised --config configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v1.yaml --run-id supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1 *> $teacherLog
    } finally { Pop-Location }
}
