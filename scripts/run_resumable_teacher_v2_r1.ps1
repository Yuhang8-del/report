param(
    [int]$PollSeconds = 15
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }

$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$config = Join-Path $work 'configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v2_r1.yaml'
$runId = 'supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2-r1'
$runDir = Join-Path 'E:\fruit_ssod_runtime\artifacts_v17\runs' $runId
$recordPath = Join-Path $runDir 'run_record.json'
$lastPath = Join-Path $runDir 'weights\last.pt'
$log = 'E:\fruit_ssod_runtime\artifacts_v17\teacher-aggressive-v2-r1-resumable.log'
$env:PYTHONPATH = Join-Path $work 'src'

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

function Read-Status {
    if (-not (Test-Path -LiteralPath $recordPath)) { return 'missing' }
    try { return [string](Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json).status } catch { return 'running' }
}

if (Test-Path -LiteralPath $runDir) {
    $status = Read-Status
    if ($status -ne 'running') {
        throw "Recovery run directory already has terminal status ${status}: $runDir"
    }
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runDir) | Out-Null
}

Log "Starting resumable v2 Teacher controller for $runId. Each completed epoch is checkpointed with save_period=1."
Push-Location $work
try {
    $first = $true
    while ($true) {
        $status = Read-Status
        if ($status -eq 'complete') {
            Log 'Recovery run is complete; controller exits.'
            break
        }
        if ($status -eq 'failed') {
            Log 'Recovery run was marked failed by the trainer; controller stops without rewriting evidence.'
            exit 2
        }
        if ($first) {
            Log 'Launching initial v2-r1 training invocation.'
            & $python -m fruit_ssod.cli.train_supervised --config $config --run-id $runId *>> $log
            $first = $false
        } elseif (Test-Path -LiteralPath $lastPath) {
            Log "Launching resume invocation from $lastPath."
            & $python -m fruit_ssod.cli.train_supervised --config $config --resume $lastPath *>> $log
        } else {
            Log 'Trainer returned while run is still active but no last.pt exists; waiting before retry.'
            Start-Sleep -Seconds $PollSeconds
        }
        $status = Read-Status
        if ($status -eq 'complete') { continue }
        if ($status -eq 'failed') {
            Log 'Trainer emitted a terminal failure; controller stops.'
            exit 2
        }
        if (-not (Test-Path -LiteralPath $lastPath)) {
            Log 'No resumable last.pt was published after the invocation; retrying once after a short wait.'
            Start-Sleep -Seconds $PollSeconds
        }
    }
} finally {
    Pop-Location
}
