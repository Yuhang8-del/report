<#
.SYNOPSIS
Builds and audits the v13 Open Images train-only recovery dataset on Windows.

.DESCRIPTION
This command deliberately does not train a model. It verifies the completed
v13 image conversion, builds the canonical and cleaned manifests, appends only
accepted images to the v12 balanced training view, and writes a pre-training
audit. The v12 validation and test lists are reused by hash, never recreated.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$PythonExecutable,
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$DataRoot,
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$ArtifactRoot,
    [ValidateRange(1, 100000)]
    [int]$ExpectedImageCount = 2205,
    [ValidateRange(0, 64)]
    [int]$NearHashThreshold = 4
)

$ErrorActionPreference = 'Stop'
$env:PYTHONNOUSERSITE = '1'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = Join-Path $RepositoryRoot 'src'
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw "Python executable does not exist: $PythonExecutable" }

$ConvertedRoot = Join-Path $DataRoot 'fruit_ssod\processed\open_images_v7_train_v13_targeted_caps'
$SelectionRoot = Join-Path $DataRoot 'fruit_ssod\raw\open_images_v7\selections\v13_natural_targeted_caps_excluding_v11'
$ManifestRoot = Join-Path $DataRoot 'fruit_ssod\manifests'
$BaseTrainingRoot = Join-Path $DataRoot 'fruit_ssod\processed\yolo\supervised_v12n_full_label_balanced_seed42'
$OutputRoot = Join-Path $DataRoot 'fruit_ssod\processed\yolo\supervised_v13_open_images_train_only_seed42'
$CanonicalManifest = Join-Path $ManifestRoot 'v13_open_images_natural_targeted_caps_canonical.json'
$CleanedManifest = Join-Path $ManifestRoot 'v13_open_images_natural_targeted_caps_cleaned.json'
$QuarantineManifest = Join-Path $ManifestRoot 'v13_open_images_natural_targeted_caps_quarantine.jsonl'
$CandidateManifest = Join-Path $ManifestRoot 'v13_open_images_natural_targeted_caps_candidates.json'
$AuditOutput = Join-Path $ArtifactRoot 'data_audits\v13_train_only_augmentation_seed42.json'

foreach ($required in @(
    (Join-Path $ConvertedRoot 'manifest.jsonl'),
    (Join-Path $ConvertedRoot 'images'),
    (Join-Path $ConvertedRoot 'labels'),
    (Join-Path $SelectionRoot 'image-urls.csv'),
    (Join-Path $BaseTrainingRoot 'membership.json')
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required v13 input is missing: $required" }
}
$imageCount = @(Get-ChildItem -LiteralPath (Join-Path $ConvertedRoot 'images') -File -Filter '*.jpg').Count
$labelCount = @(Get-ChildItem -LiteralPath (Join-Path $ConvertedRoot 'labels') -File -Filter '*.txt').Count
$manifestCount = @(Get-Content -LiteralPath (Join-Path $ConvertedRoot 'manifest.jsonl')).Count
if ($imageCount -ne $ExpectedImageCount -or $labelCount -ne $ExpectedImageCount -or $manifestCount -ne $ExpectedImageCount) {
    throw "Incomplete v13 conversion: images=$imageCount, labels=$labelCount, manifestRows=$manifestCount, expected=$ExpectedImageCount. Do not curate or train until the completed remote transfer is reconciled."
}
foreach ($output in @($CanonicalManifest, $CleanedManifest, $QuarantineManifest, $CandidateManifest, $OutputRoot, $AuditOutput)) {
    if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite immutable v13 evidence: $output" }
}

function Invoke-ProjectPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Project Python command failed: $PythonExecutable $($Arguments -join ' ')" }
}

Invoke-ProjectPython @('-m', 'fruit_ssod.cli.build_open_images_manifest', '--converted-root', $ConvertedRoot, '--selection-url-csv', (Join-Path $SelectionRoot 'image-urls.csv'), '--output', $CanonicalManifest)
Invoke-ProjectPython @('-m', 'fruit_ssod.cli.clean_dataset', '--input-manifest', $CanonicalManifest, '--output-manifest', $CleanedManifest, '--quarantine-manifest', $QuarantineManifest, '--image-root', $ConvertedRoot, '--near-hash-threshold', $NearHashThreshold)
Invoke-ProjectPython @('-m', 'fruit_ssod.cli.build_candidate_manifest', '--cleaned-manifest', $CleanedManifest, '--output', $CandidateManifest)
Invoke-ProjectPython @('-m', 'fruit_ssod.cli.materialize_train_only_augmentation', '--base-training-root', $BaseTrainingRoot, '--added-candidate-manifest', $CandidateManifest, '--added-source-root', $ConvertedRoot, '--output-root', $OutputRoot, '--protected-near-hash-threshold', $NearHashThreshold)
Invoke-ProjectPython @('-m', 'fruit_ssod.cli.audit_train_only_augmentation', '--augmentation-root', $OutputRoot, '--output', $AuditOutput)

Write-Host "v13 data recovery is complete: $OutputRoot"
Write-Host "Pre-training audit: $AuditOutput"
