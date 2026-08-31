param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [Parameter(Mandatory = $true)]
    [string]$WorkingDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [Parameter(Mandatory = $true)]
    [string]$SplitManifest,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
Set-Location -LiteralPath $resolvedWorkingDirectory
$resolvedRun = (Resolve-Path -LiteralPath $RunDir).Path
$resolvedData = (Resolve-Path -LiteralPath $Data).Path
$resolvedManifest = (Resolve-Path -LiteralPath $SplitManifest).Path
$recordPath = Join-Path $resolvedRun 'run_record.json'
$evaluationPath = Join-Path $resolvedRun 'evaluations\test.json'
$logPath = Join-Path $resolvedRun 'student_fixed_test_watcher.log'
if (!(Test-Path -LiteralPath $recordPath)) { throw "Missing run record: $recordPath" }
if (Test-Path -LiteralPath $evaluationPath) {
    Add-Content -LiteralPath $logPath -Value 'Fixed-test evaluation already exists; watcher exits without overwriting evidence.'
    exit 0
}

while ($true) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    switch ([string]$record.status) {
        'complete' {
            Add-Content -LiteralPath $logPath -Value "Training complete; running sealed Student fixed-test evaluation $(Get-Date -Format o)."
            & $Python -m fruit_ssod.cli.evaluate_student_test --run-dir $resolvedRun --data $resolvedData --split-manifest $resolvedManifest --device cuda:0 *>> $logPath
            exit $LASTEXITCODE
        }
        'failed' {
            Add-Content -LiteralPath $logPath -Value 'Training recorded status=failed; fixed-test evaluation is not authorized.'
            exit 2
        }
        'running' { Start-Sleep -Seconds $PollSeconds }
        default { throw "Unexpected training status: $($record.status)" }
    }
}
