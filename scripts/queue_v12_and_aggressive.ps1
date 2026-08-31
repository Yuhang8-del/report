$ErrorActionPreference = 'Stop'
$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$v15Record = 'E:\fruit_ssod_runtime\artifacts_v16\runs\ssod-v1-independent-openimages-v15teacher-seed42\run_record.json'
$v12Run = 'E:\fruit_ssod_runtime\artifacts_v17\runs\ssod-v1-independent-openimages-v12teacher-seed42'
$v12Record = Join-Path $v12Run 'run_record.json'
$v12Log = 'E:\fruit_ssod_runtime\artifacts_v17\ssod-v12-student-queued.log'
$aggressiveRun = 'E:\fruit_ssod_runtime\artifacts_v17\runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1'
$aggressiveRecord = Join-Path $aggressiveRun 'run_record.json'
$aggressiveLog = 'E:\fruit_ssod_runtime\artifacts_v17\teacher-aggressive-queued.log'

while (-not (Test-Path $v15Record)) { Start-Sleep -Seconds 30 }
do {
    try { $status = (Get-Content $v15Record -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
    if ($status -ne 'running') { break }
    Start-Sleep -Seconds 30
} while ($true)

if (-not (Test-Path $v12Record)) {
    Push-Location $work
    try {
        & $python -m fruit_ssod.cli.train_student --config configs\experiments\ssod_v1_independent_openimages_v12teacher_seed42.yaml --run-id ssod-v1-independent-openimages-v12teacher-seed42 *> $v12Log
    } finally { Pop-Location }
}

while (-not (Test-Path $v12Record)) { Start-Sleep -Seconds 30 }
do {
    try { $status = (Get-Content $v12Record -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
    if ($status -ne 'running') { break }
    Start-Sleep -Seconds 30
} while ($true)

if (-not (Test-Path $aggressiveRecord)) {
    Push-Location $work
    try {
        & $python -m fruit_ssod.cli.train_supervised --config configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v1.yaml --run-id supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1 *> $aggressiveLog
    } finally { Pop-Location }
}
