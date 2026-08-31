param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [int]$PollSeconds = 30
)
$ErrorActionPreference = 'Stop'
$resolvedWork = (Resolve-Path -LiteralPath $WorkingDirectory).Path
Set-Location -LiteralPath $resolvedWork
$record = Join-Path $RunDir 'run_record.json'
$evaluation = Join-Path $RunDir 'evaluations\test.json'
while (-not (Test-Path -LiteralPath $evaluation)) {
    if (Test-Path -LiteralPath $record) {
        try { $status = [string](Get-Content -LiteralPath $record -Raw | ConvertFrom-Json).status } catch { $status = 'running' }
        if ($status -eq 'failed') { throw "Student training recorded status=failed; post-processing cannot continue: $RunDir" }
    }
    Start-Sleep -Seconds $PollSeconds
}
$gui = 'E:\fruit_ssod_runtime\artifacts_v17\exports\gui_v12teacher_student'
$package = 'E:\fruit_ssod_runtime\artifacts_v17\exports\exploratory_v3_v12teacher_student_package'
$source = 'E:\fruit_ssod_runtime\artifacts_v16\exports\gui_optimized_independent_openimages\results.json'
if (-not (Test-Path -LiteralPath $gui)) {
    & $Python scripts\export_gui_candidate.py --checkpoint (Join-Path $RunDir 'weights\best.pt') --output $gui --source-results $source *>> (Join-Path $RunDir 'v12_postprocess.log')
    if ($LASTEXITCODE -ne 0) { throw "GUI export failed: $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath $package)) {
    & $Python -m fruit_ssod.cli.build_exploratory_package --run-dir $RunDir --gui-export $gui --output $package *>> (Join-Path $RunDir 'v12_postprocess.log')
    if ($LASTEXITCODE -ne 0) { throw "Exploratory package failed: $LASTEXITCODE" }
}
