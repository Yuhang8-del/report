param(
    [string]$Config = 'configs\experiments\supervised_v3_domain_balanced_yolov8m_1024_seed42.yaml',
    [string]$RunId = 'supervised-v3-domain-balanced-yolov8m-1024-seed42',
    [int]$PollSeconds = 20
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }

$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$artifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17'
$runDir = Join-Path $artifactRoot ("runs\{0}" -f $RunId)
$recordPath = Join-Path $runDir 'run_record.json'
$lastPath = Join-Path $runDir 'weights\last.pt'
$log = Join-Path $artifactRoot ("{0}.log" -f $RunId)
$env:PYTHONPATH = Join-Path $work 'src'

function Read-Status {
    if (-not (Test-Path -LiteralPath $recordPath)) { return 'missing' }
    try { return [string](Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json).status } catch { return 'running' }
}

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

function Invoke-Trainer([string[]]$Arguments) {
    # Python libraries such as Polars may emit UserWarning messages on stderr
    # even when training succeeds.  Windows PowerShell converts native stderr
    # into ErrorRecords; with ErrorActionPreference=Stop that used to abort
    # this controller after a healthy epoch.  Keep the native exit code as the
    # authoritative result and allow stderr to remain in the run log.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $python @Arguments *>> $log
        return [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
}

if (Test-Path -LiteralPath $runDir) {
    $status = Read-Status
    if ($status -notin @('running', 'complete', 'failed')) { throw "Unexpected run status: $status" }
    if ($status -eq 'complete') { Log 'Run is already complete; no action taken.'; exit 0 }
    if ($status -eq 'failed') { throw "Refusing to resume a terminal failed run: $runDir" }
} else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runDir) | Out-Null
}

Push-Location $work
try {
    $first = -not (Test-Path -LiteralPath $recordPath)
    while ($true) {
        $status = Read-Status
        if ($status -eq 'complete') { Log 'Optimization Teacher completed.'; break }
        if ($status -eq 'failed') { throw 'Training recorded terminal failure; preserving evidence.' }
        if ($first) {
            Log "Launching initial optimization Teacher: $Config / $RunId"
            $exitCode = Invoke-Trainer @('-m', 'fruit_ssod.cli.train_supervised', '--config', $Config, '--run-id', $RunId)
            Log "Trainer process exited with code $exitCode"
            $first = $false
        } elseif (Test-Path -LiteralPath $lastPath) {
            Log "Trainer exited before terminal state; resuming from $lastPath"
            $exitCode = Invoke-Trainer @('-m', 'fruit_ssod.cli.train_supervised', '--config', $Config, '--run-id', $RunId, '--resume', $lastPath)
            Log "Resume trainer process exited with code $exitCode"
        } else {
            Log 'Run remains active without last.pt; waiting before retry.'
            Start-Sleep -Seconds $PollSeconds
        }
        if ((Read-Status) -eq 'complete') { continue }
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    Pop-Location
}
