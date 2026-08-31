param(
    [Parameter(Mandatory = $true)]
    [string]$CurrentRunDir,
    [Parameter(Mandatory = $true)]
    [string]$CurrentEvaluation,
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [Parameter(Mandatory = $true)]
    [string]$WorkingDirectory,
    [Parameter(Mandatory = $true)]
    [string]$AuditLabels,
    [Parameter(Mandatory = $true)]
    [string]$AuditSplitManifest,
    [Parameter(Mandatory = $true)]
    [string]$TeacherWeights,
    [Parameter(Mandatory = $true)]
    [string]$TeacherRunId,
    [Parameter(Mandatory = $true)]
    [string]$TeacherCandidates,
    [Parameter(Mandatory = $true)]
    [string]$TeacherImageRoot,
    [Parameter(Mandatory = $true)]
    [string]$MatrixConfig,
    [Parameter(Mandatory = $true)]
    [string]$AuditOutput,
    [Parameter(Mandatory = $true)]
    [string]$AuditFilterOutput,
    [Parameter(Mandatory = $true)]
    [string]$StudentConfig,
    [Parameter(Mandatory = $true)]
    [string]$StudentRunId,
    [Parameter(Mandatory = $true)]
    [string]$FixedTestData,
    [Parameter(Mandatory = $true)]
    [string]$StudentSplitManifest,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
Set-Location -LiteralPath $resolvedWorkingDirectory
$resolvedCurrentRun = (Resolve-Path -LiteralPath $CurrentRunDir).Path
$resolvedCurrentEvaluation = [System.IO.Path]::GetFullPath($CurrentEvaluation)
$logPath = Join-Path $resolvedCurrentRun 'next_student_chain.log'
function Log([string]$Message) { Add-Content -LiteralPath $logPath -Value ("$(Get-Date -Format o) $Message") }
function Run-Checked([string[]]$Arguments) {
    & $Python @Arguments *>> $logPath
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')" }
}

$recordPath = Join-Path $resolvedCurrentRun 'run_record.json'
if (!(Test-Path -LiteralPath $recordPath)) { throw "Missing current run record: $recordPath" }
while ($true) {
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $status = [string]$record.status
    if ($status -eq 'complete') { break }
    if ($status -eq 'failed') {
        Log 'Current run failed; next experiment will not be started.'
        exit 2
    }
    if ($status -eq 'running') {
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    throw "Unexpected current status: $status"
}
while (!(Test-Path -LiteralPath $resolvedCurrentEvaluation)) { Start-Sleep -Seconds $PollSeconds }
Log 'Current Student fixed-test evidence exists; preparing paired no-class-threshold pseudo audit.'

if (!(Test-Path -LiteralPath $AuditOutput)) {
    Run-Checked @('-m','fruit_ssod.cli.audit_pseudo_labels','--audit-labels',$AuditLabels,'--split-manifest',$AuditSplitManifest,'--output',$AuditOutput,'--prepare-from-teacher','--weights',$TeacherWeights,'--teacher-run-id',$TeacherRunId,'--candidates',$TeacherCandidates,'--image-root',$TeacherImageRoot,'--matrix-config',$MatrixConfig,'--filter-output',$AuditFilterOutput,'--minimum-precision','0.90')
} else {
    Log 'Paired audit output already exists; preserving it.'
}

$nextRunDir = Join-Path 'E:\fruit_ssod_runtime\artifacts_v15\runs' $StudentRunId
if (!(Test-Path -LiteralPath $nextRunDir)) {
    Log 'Starting next exploratory Student training.'
    Run-Checked @('-m','fruit_ssod.cli.train_student','--config',$StudentConfig,'--run-id',$StudentRunId)
} else {
    Log 'Next Student run directory already exists; refusing to overwrite it.'
    exit 2
}

Log 'Next Student training complete; running sealed fixed-test evaluation.'
Run-Checked @('-m','fruit_ssod.cli.evaluate_student_test','--run-dir',$nextRunDir,'--data',$FixedTestData,'--split-manifest',$StudentSplitManifest,'--device','cuda:0')
Log 'Next Student fixed-test evaluation complete.'
