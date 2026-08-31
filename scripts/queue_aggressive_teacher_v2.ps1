param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$v1Run = 'E:\fruit_ssod_runtime\artifacts_v17\runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1'
$v1Record = Join-Path $v1Run 'run_record.json'
$v2Run = 'E:\fruit_ssod_runtime\artifacts_v17\runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2'
$v2Record = Join-Path $v2Run 'run_record.json'
$log = 'E:\fruit_ssod_runtime\artifacts_v17\teacher-aggressive-v2-queued.log'

while (-not (Test-Path -LiteralPath $v1Record)) { Start-Sleep -Seconds $PollSeconds }
do {
    try { $status = [string](Get-Content -LiteralPath $v1Record -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
    if ($status -ne 'running') { break }
    Start-Sleep -Seconds $PollSeconds
} while ($true)

if (-not (Test-Path -LiteralPath $v2Record)) {
    Push-Location $work
    try {
        & $python -m fruit_ssod.cli.train_supervised --config configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v2.yaml --run-id supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2 *> $log
    } finally { Pop-Location }
}

while (-not (Test-Path -LiteralPath $v2Record)) { Start-Sleep -Seconds $PollSeconds }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $work 'scripts\watch_completed_run_and_evaluate.ps1') -RunDir $v2Run -Python $python -WorkingDirectory $work -PollSeconds $PollSeconds
