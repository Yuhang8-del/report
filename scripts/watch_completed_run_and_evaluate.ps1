param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [Parameter(Mandatory = $true)]
    [string]$WorkingDirectory,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
Set-Location -LiteralPath $resolvedWorkingDirectory
$resolvedRun = (Resolve-Path -LiteralPath $RunDir).Path
$recordPath = Join-Path $resolvedRun 'run_record.json'
$evaluationPath = Join-Path $resolvedRun 'evaluations\test.json'
if (!(Test-Path -LiteralPath $recordPath)) { throw "Missing run record: $recordPath" }
if (Test-Path -LiteralPath $evaluationPath) {
    Write-Output 'Fixed-test evaluation already exists; watcher exits without overwriting evidence.'
    exit 0
}

while ($true) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    switch ([string]$record.status) {
        'complete' {
            & $Python -m fruit_ssod.cli.evaluate_model --run-dir $resolvedRun --split test --device 0
            exit $LASTEXITCODE
        }
        'failed' { throw "Training recorded status=failed; fixed-test evaluation is not authorized." }
        'running' { Start-Sleep -Seconds $PollSeconds }
        default { throw "Unexpected training status: $($record.status)" }
    }
}
