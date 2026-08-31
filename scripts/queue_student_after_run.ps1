param(
    [Parameter(Mandatory = $true)][string]$CurrentRunDir,
    [Parameter(Mandatory = $true)][string]$CurrentEvaluation,
    [Parameter(Mandatory = $true)][string]$NextConfig,
    [Parameter(Mandatory = $true)][string]$NextRunId,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$FixedTestData,
    [Parameter(Mandatory = $true)][string]$StudentSplitManifest,
    [Parameter(Mandatory = $true)][string]$NextStdout,
    [Parameter(Mandatory = $true)][string]$NextStderr,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$resolvedCurrent = (Resolve-Path -LiteralPath $CurrentRunDir).Path
$resolvedWork = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$resolvedConfig = (Resolve-Path -LiteralPath $NextConfig).Path
$recordPath = Join-Path $resolvedCurrent 'run_record.json'
$evaluationPath = [System.IO.Path]::GetFullPath($CurrentEvaluation)
$logPath = Join-Path $resolvedCurrent 'queue_next_seed.log'
function Log([string]$Message) { Add-Content -LiteralPath $logPath -Value ("$(Get-Date -Format o) $Message") }

while ($true) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $status = [string]$record.status
    if ($status -eq 'complete') { break }
    if ($status -eq 'failed') { Log 'Current run failed; no follow-up was started.'; exit 2 }
    if ($status -ne 'running') { throw "Unexpected current status: $status" }
    Start-Sleep -Seconds $PollSeconds
}
while (!(Test-Path -LiteralPath $evaluationPath)) { Start-Sleep -Seconds $PollSeconds }

$nextRunDir = Join-Path 'E:\fruit_ssod_runtime\artifacts_v15\runs' $NextRunId
if (Test-Path -LiteralPath $nextRunDir) { Log "Refusing to overwrite existing next run: $nextRunDir"; exit 3 }

$stdoutParent = Split-Path -Parent $NextStdout
$stderrParent = Split-Path -Parent $NextStderr
New-Item -ItemType Directory -Force -Path $stdoutParent, $stderrParent | Out-Null
Log "Starting next Student run $NextRunId after fixed-test evidence appeared."
Start-Process -FilePath $Python `
    -ArgumentList @('-m','fruit_ssod.cli.train_student','--config',$resolvedConfig,'--run-id',$NextRunId) `
    -WorkingDirectory $resolvedWork `
    -RedirectStandardOutput $NextStdout `
    -RedirectStandardError $NextStderr `
    -WindowStyle Hidden | Out-Null

$watchScript = Join-Path $resolvedWork 'scripts\watch_completed_student_and_evaluate.ps1'
$watchLog = Join-Path (Split-Path -Parent $NextStdout) ("watch_{0}.log" -f $NextRunId)
$watchErr = $watchLog + '.err'
Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$watchScript,'-RunDir',$nextRunDir,'-Python',$Python,'-WorkingDirectory',$resolvedWork,'-Data',$FixedTestData,'-SplitManifest',$StudentSplitManifest,'-PollSeconds',$PollSeconds) `
    -WorkingDirectory $resolvedWork `
    -RedirectStandardOutput $watchLog `
    -RedirectStandardError $watchErr `
    -WindowStyle Hidden | Out-Null
Log "Started training and fixed-test watcher for $NextRunId."
