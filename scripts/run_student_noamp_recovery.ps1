param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17'
)

$ErrorActionPreference = 'Stop'
$sourceConfig = Join-Path $Worktree 'configs\experiments\ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-seed42.yaml'
$runId = 'ssod-v2-independent-openimages-aggressive-best-run-v12n-v8m-balanced-1024-ft40-seed42-noamp-seed42'
$configPath = Join-Path $Worktree ("configs\experiments\{0}.yaml" -f $runId)
$runDir = Join-Path $ArtifactRoot ("runs\{0}" -f $runId)
$log = Join-Path $ArtifactRoot 'student-noamp-recovery.log'

if (-not (Test-Path -LiteralPath $sourceConfig)) { throw "source Student config is missing: $sourceConfig" }
if (Test-Path -LiteralPath $runDir) { throw "recovery run already exists: $runDir" }
if (-not (Test-Path -LiteralPath $configPath)) {
    $yaml = Get-Content -LiteralPath $sourceConfig -Raw
    $yaml = $yaml.Replace('experiment_name: ssod_v2_independent_openimages_aggressive_best_run_v12n_v8m_balanced_1024_ft40_seed42_seed42', 'experiment_name: ssod_v2_independent_openimages_aggressive_best_run_v12n_v8m_balanced_1024_ft40_seed42_noamp_seed42')
    $yaml = $yaml.Replace('amp: true', 'amp: false')
    Set-Content -LiteralPath $configPath -Value $yaml -Encoding UTF8
}

$env:PYTHONPATH = Join-Path $Worktree 'src'
Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) Starting no-AMP Student recovery $runId; source=$sourceConfig") -Encoding UTF8
Push-Location $Worktree
try {
    & $Python -m fruit_ssod.cli.train_student --config $configPath --run-id $runId *>> $log
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) Student recovery process exited with code $exitCode") -Encoding UTF8
    exit $exitCode
} finally {
    Pop-Location
}
