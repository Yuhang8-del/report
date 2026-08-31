param([Parameter(Mandatory = $true)][string]$Package,[Parameter(Mandatory = $true)][string]$Python,[Parameter(Mandatory = $true)][string]$WorkingDirectory,[int]$PollSeconds = 30)
$ErrorActionPreference = 'Stop'
while (-not (Test-Path -LiteralPath (Join-Path $Package 'summary.json'))) { Start-Sleep -Seconds $PollSeconds }
Set-Location -LiteralPath (Resolve-Path -LiteralPath $WorkingDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $Package 'exploratory_best_result_report.docx'))) {
    & $Python scripts\build_exploratory_report.py --package $Package *>> (Join-Path $Package 'report_build.log')
    exit $LASTEXITCODE
}
