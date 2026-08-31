param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$SplitManifest,
    [int]$PollSeconds = 30
)
$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
while (-not (Test-Path -LiteralPath (Join-Path $RunDir 'run_record.json'))) { Start-Sleep -Seconds $PollSeconds }
& (Join-Path (Split-Path -Parent $PSCommandPath) 'watch_completed_student_and_evaluate.ps1') `
    -RunDir $RunDir -Python $Python -WorkingDirectory $WorkingDirectory `
    -Data $Data -SplitManifest $SplitManifest -PollSeconds $PollSeconds
exit $LASTEXITCODE
