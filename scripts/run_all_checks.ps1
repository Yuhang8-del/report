[CmdletBinding()]
param(
    [string]$CondaEnvironment = $(if ($env:FRUIT_SSOD_CONDA_ENV) { $env:FRUIT_SSOD_CONDA_ENV } else { 'fruit-ssod' }),
    [switch]$SkipGuiTests
)

$ErrorActionPreference = 'Stop'
# Do not permit packages from the invoking user's site directory to influence
# the reproducibility check.
$env:PYTHONNOUSERSITE = '1'
$repository = Split-Path -Parent $PSScriptRoot
Push-Location $repository
try {
    $pythonCheck = @('-m', 'pytest', '-q')
    if ($SkipGuiTests) {
        $pythonCheck += @('--ignore', 'tests/gui')
    }
    & conda run -n $CondaEnvironment python @pythonCheck
    if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }

    $required = @(
        'reports/final_report/build_report.py',
        'reports/final_report/outline.md',
        'reports/final_report/references.bib',
        'docs/testing/final-qa-checklist.md',
        'docs/handoff/delivery-manifest.md',
        'docs/handoff/reproduction.md'
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) { throw "Required release source files are missing: $($missing -join ', ')" }

    $stagedLarge = @(git diff --cached --numstat | ForEach-Object {
        $parts = $_ -split "`t"
        if ($parts.Count -ge 3 -and $parts[0] -match '^\d+$' -and [int64]$parts[0] -gt 52428800) { $parts[2] }
    })
    if ($stagedLarge.Count -gt 0) { throw "Large generated files are staged: $($stagedLarge -join ', ')" }
    Write-Output 'Automated QA source checks passed. Complete the manual checklist before release.'
}
finally {
    Pop-Location
}
