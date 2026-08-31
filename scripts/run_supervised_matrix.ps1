[CmdletBinding()]
param(
    [switch]$DryRun,
    [ValidateNotNullOrEmpty()]
    [string]$CondaEnvironment = 'fruit-ssod',
    [ValidateNotNullOrEmpty()]
    [string]$CondaExecutable = 'conda',
    [ValidateNotNullOrEmpty()]
    [string]$AggregateOutput = ''
)

# Native Windows launcher for the fixed Task 12 reference matrix.  Keep
# arguments in arrays and invoke with `&` rather than composing a shell string:
# paths containing spaces stay safe and are not re-parsed by PowerShell.
$ErrorActionPreference = 'Stop'
# Prevent user-site packages from changing the audited training runtime.
$env:PYTHONNOUSERSITE = '1'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'resolve_conda_command.ps1')
$ConfigDirectory = Join-Path $RepositoryRoot 'configs\experiments'
$TemplatePath = Join-Path $ConfigDirectory 'supervised_reference_template.yaml'
$ConfigNames = @(
    'supervised_10_seed42.yaml',
    'supervised_20_seed42.yaml',
    'supervised_20_seed3407.yaml',
    'supervised_20_seed2026.yaml',
    'supervised_40_seed42.yaml',
    'supervised_100_seed42.yaml'
)

if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
    throw "Canonical supervised template is missing: $TemplatePath"
}
$CondaCommand = Resolve-CondaCommand -CondaExecutable $CondaExecutable
$RunDirectories = [System.Collections.Generic.List[string]]::new()
if (-not $DryRun -and -not $env:FRUIT_SSOD_ARTIFACT_ROOT) {
    throw 'FRUIT_SSOD_ARTIFACT_ROOT is required to retain and aggregate every attempted matrix run.'
}

# This must run before even configuration dry-runs.  It calls the same
# validate_reference_configs implementation covered by unit tests, rejecting a
# hand-edited epoch/path/budget config instead of queueing non-comparable GPU
# work.
$ValidationArguments = @(
    'run', '--no-capture-output', '--name', $CondaEnvironment,
    'python', '-m', 'fruit_ssod.cli.validate_supervised_matrix',
    '--template', $TemplatePath, '--config-directory', $ConfigDirectory
)
Write-Host ('Matrix configuration preflight: ' + ($CondaCommand, $ValidationArguments -join ' '))
& $CondaCommand @ValidationArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Supervised matrix configuration preflight failed. Restore deterministic template-generated YAML before queueing any run.'
}

foreach ($ConfigName in $ConfigNames) {
    $ConfigPath = Join-Path $ConfigDirectory $ConfigName
    $ExperimentName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigName)
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "Required supervised matrix config is missing: $ConfigPath"
    }

    # `--dry-run` validates paths/classes/fingerprints and writes only a dry
    # provenance record. It intentionally does not request a GPU or evaluate
    # the held-out test set.
    $Arguments = @(
        'run', '--no-capture-output', '--name', $CondaEnvironment,
        'python', '-m', 'fruit_ssod.cli.train_supervised', '--config', $ConfigPath
    )
    if ($DryRun) {
        # Deliberately omit --run-id: train_supervised creates a disposable
        # UUID record for validation, so a later real matrix run may safely
        # claim its fixed protocol run ID and artifact directory.
        $Arguments += '--dry-run'
    } else {
        $Arguments += @('--run-id', $ExperimentName)
    }

    Write-Host ('Queue: ' + ($CondaCommand, $Arguments -join ' '))
    $OutputLines = & $CondaCommand @Arguments 2>&1
    $ExitCode = $LASTEXITCODE
    $OutputLines | ForEach-Object { Write-Host $_ }
    if ($ExitCode -ne 0) {
        if ($DryRun) {
            throw "Configuration dry-run failed: $ConfigName (exit code $ExitCode). Correct the configuration before requesting GPU work."
        }
        # A matrix run can fail after its runner has emitted run_record.json.
        # Retain the deterministic expected directory even when the Python
        # process exits nonzero: the aggregator will show either the failure
        # record or an explicit unreadable row, never silently omit it.
        $FailedRunDirectory = Join-Path $env:FRUIT_SSOD_ARTIFACT_ROOT (Join-Path 'runs' $ExperimentName)
        $RunDirectories.Add($FailedRunDirectory)
        Write-Warning "Supervised matrix entry failed: $ConfigName (exit code $ExitCode). Continuing so aggregation retains this attempted run."
        continue
    }
    if ($DryRun) {
        continue
    }

    # The explicit run ID makes the artifact location deterministic, including
    # for later failed/evaluation rows. Do not infer it from noisy framework
    # stdout that may contain progress logging before the final JSON line.
    $RunDirectory = Join-Path $env:FRUIT_SSOD_ARTIFACT_ROOT (Join-Path 'runs' $ExperimentName)
    $RunDirectories.Add($RunDirectory)
    $EvaluationArguments = @(
        'run', '--no-capture-output', '--name', $CondaEnvironment,
        'python', '-m', 'fruit_ssod.cli.evaluate_model', '--run-dir', $RunDirectory, '--split', 'test'
    )
    Write-Host ('Fixed-test evaluation: ' + ($CondaCommand, $EvaluationArguments -join ' '))
    & $CondaCommand @EvaluationArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Fixed-test evaluation failed for $ConfigName. Continuing so aggregation retains the completed run with missing test evidence."
    }
}

if ($DryRun) {
    Write-Host 'All six configurations passed launcher dry-run validation. No training or held-out test evaluation was started.'
    exit 0
}

if ($AggregateOutput) {
    $AggregatePath = $AggregateOutput
} elseif ($env:FRUIT_SSOD_ARTIFACT_ROOT) {
    $AggregatePath = Join-Path $env:FRUIT_SSOD_ARTIFACT_ROOT 'exports\supervised_matrix.json'
} else {
    throw 'FRUIT_SSOD_ARTIFACT_ROOT is required for the default aggregate output. Set it or pass -AggregateOutput.'
}
$AggregateArguments = @(
    'run', '--no-capture-output', '--name', $CondaEnvironment,
    'python', '-m', 'fruit_ssod.cli.aggregate_supervised_matrix'
)
foreach ($RunDirectory in $RunDirectories) {
    $AggregateArguments += @('--run-dir', $RunDirectory)
}
$AggregateArguments += @('--output', $AggregatePath)
Write-Host ('Aggregate: ' + ($CondaCommand, $AggregateArguments -join ' '))
& $CondaCommand @AggregateArguments
if ($LASTEXITCODE -ne 0) {
    throw "Supervised matrix aggregation failed. The individual run records remain the authoritative evidence."
}

Write-Host "Matrix evidence written to: $AggregatePath"
