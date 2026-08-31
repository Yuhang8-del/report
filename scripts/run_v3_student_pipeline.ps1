param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$Worktree = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation',
    [string]$ArtifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17'
)

$ErrorActionPreference = 'Stop'
$teacherRunId = 'supervised-v3-domain-balanced-yolov8m-1024-seed42-r3'
$studentRunId = 'ssod-v3-teacher-r3-student-seed42'
$teacherWeights = Join-Path $ArtifactRoot "runs\$teacherRunId\weights\best.pt"
$teacherRecord = Join-Path $ArtifactRoot "runs\$teacherRunId\run_record.json"
$studentConfig = Join-Path $Worktree 'configs\experiments\ssod_v3_teacher_r3_student_seed42.yaml'
$unlabeledManifest = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\ssod_unlabeled_pool_fruits360_v2_openimages_v13b\unlabeled.json'
$studentSplitManifest = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\ssod_unlabeled_pool_fruits360_v2_openimages_v13b\split_manifest.json'
$imageRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\ssod_combined_image_root_v2'
$pseudoRoot = Join-Path $ArtifactRoot 'pseudo\v3_teacher_r3_seed42'
$candidates = Join-Path $pseudoRoot 'candidates.json'
$filterRoot = Join-Path $pseudoRoot 'filter'
$filterAudit = Join-Path $filterRoot 'audit.jsonl'
$filterDecision = Join-Path $filterRoot 'decision_manifest.json'
$auditLabels = 'E:\fruit_ssod_runtime\data\fruit_ssod\manifests\splits_v2\protected_splits\pseudo_audit_labels.json'
$auditRoot = Join-Path $pseudoRoot 'audit_report'
$auditReport = Join-Path $auditRoot 'pseudo_audit.json'
# Keep audit-only inputs beside (not below) the atomic report directory:
# write_pseudo_candidates creates its parent, while the audit publisher
# requires the report directory itself to be absent before publication.
$auditCandidates = Join-Path $pseudoRoot 'audit_candidates.json'
$auditFilterRoot = Join-Path $pseudoRoot 'audit_filter'
$studentRunDir = Join-Path $ArtifactRoot "runs\$studentRunId"
$fixedTestData = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\supervised_v2_100_seed42\dataset.yaml'
$log = Join-Path $ArtifactRoot 'v3-student-pipeline.log'

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message") -Encoding UTF8
}

function Invoke-Python([string[]]$Arguments) {
    # Native Python warnings are written to stderr on Windows.  They must be
    # retained in the log but cannot terminate this multi-stage controller.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments *>> $log
        $code = [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    Log "python exit_code=$code args=$($Arguments -join ' ')"
    if ($code -ne 0) { throw "Python stage failed with exit code ${code}: $($Arguments -join ' ')" }
}

foreach ($required in @($teacherWeights, $teacherRecord, $studentConfig, $unlabeledManifest, $studentSplitManifest, $imageRoot, $auditLabels, $fixedTestData)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "required pipeline input is missing: $required" }
}

$env:PYTHONPATH = Join-Path $Worktree 'src'
Push-Location $Worktree
try {
    $teacherStatus = [string](Get-Content -LiteralPath $teacherRecord -Raw | ConvertFrom-Json).status
    if ($teacherStatus -ne 'complete') { throw "Teacher run is not complete: $teacherRunId status=$teacherStatus" }

    if (-not (Test-Path -LiteralPath $candidates)) {
        Log "stage=generate_pseudo_labels teacher=$teacherRunId"
        Invoke-Python @('-m','fruit_ssod.cli.generate_pseudo_labels','--unlabeled-manifest',$unlabeledManifest,'--split-manifest',$studentSplitManifest,'--weights',$teacherWeights,'--teacher-run-id',$teacherRunId,'--output',$candidates,'--image-root',$imageRoot,'--confidence','0.01')
    } else {
        Log "stage=generate_pseudo_labels skipped existing=$candidates"
    }

    if (-not (Test-Path -LiteralPath $filterAudit) -or -not (Test-Path -LiteralPath $filterDecision)) {
        if (Test-Path -LiteralPath $filterRoot) { throw "filter output exists but is incomplete: $filterRoot" }
        Log "stage=filter_pseudo_labels policy=trust_filter_v1"
        Invoke-Python @('-m','fruit_ssod.cli.filter_pseudo_labels','--candidates',$candidates,'--unlabeled-manifest',$unlabeledManifest,'--split-manifest',$studentSplitManifest,'--output',$filterRoot,'--mode','trust','--validation-pr','E:\fruit_ssod_runtime\artifacts_v15\calibration\supervised_v2_teacher_seed42\fixed_validation_pr.json','--aspect-ratio-bounds','E:\fruit_ssod_runtime\artifacts_v15\calibration\supervised_v2_teacher_seed42\fixed_aspect_ratio_bounds.json','--target-precision','0.70','--policy-id','trust_filter_v1','--matrix-config',$studentConfig)
    } else {
        Log "stage=filter_pseudo_labels skipped existing=$filterRoot"
    }

    if (-not (Test-Path -LiteralPath $auditReport)) {
        if (Test-Path -LiteralPath $auditRoot) { throw "pseudo audit output exists but is incomplete: $auditRoot" }
        Log "stage=audit_pseudo_labels minimum_precision=0.90 (non-blocking override is in Student config)"
        # The audit CLI accepts only the sealed pseudo_audit partition.  The
        # full unlabeled candidate envelope above is intentionally not valid
        # input here, so prepare a separate audit-only candidate/filter pair
        # with the same Teacher and the exact Student matrix policy.  The
        # resulting report is evidence only; the Student config explicitly
        # allows training below the historical precision gate.
        Invoke-Python @('-m','fruit_ssod.cli.audit_pseudo_labels','--audit-labels',$auditLabels,'--split-manifest',$studentSplitManifest,'--output',$auditRoot,'--prepare-from-teacher','--weights',$teacherWeights,'--teacher-run-id',$teacherRunId,'--candidates',$auditCandidates,'--image-root',$imageRoot,'--matrix-config',$studentConfig,'--filter-output',$auditFilterRoot,'--minimum-precision','0.90')
    } else {
        Log "stage=audit_pseudo_labels skipped existing=$auditRoot"
    }

    if (-not (Test-Path -LiteralPath $studentRunDir)) {
        Log "stage=train_student run_id=$studentRunId"
        Invoke-Python @('-m','fruit_ssod.cli.train_student','--config',$studentConfig,'--run-id',$studentRunId)
    } else {
        $studentRecord = Join-Path $studentRunDir 'run_record.json'
        if (-not (Test-Path -LiteralPath $studentRecord)) { throw "Student run directory exists without run_record.json: $studentRunDir" }
        $studentStatus = [string](Get-Content -LiteralPath $studentRecord -Raw | ConvertFrom-Json).status
        if ($studentStatus -eq 'complete') { Log "stage=train_student skipped completed=$studentRunId" }
        else { throw "Student run directory already exists with status=${studentStatus}: $studentRunDir" }
    }

    $studentEvaluation = Join-Path $studentRunDir 'evaluations\test.json'
    if (-not (Test-Path -LiteralPath $studentEvaluation)) {
        Log "stage=evaluate_student_test run_id=$studentRunId"
        Invoke-Python @('-m','fruit_ssod.cli.evaluate_student_test','--run-dir',$studentRunDir,'--data',$fixedTestData,'--split-manifest',$studentSplitManifest,'--device','cuda:0')
    } else {
        Log "stage=evaluate_student_test skipped existing=$studentEvaluation"
    }
    Log "pipeline=complete teacher=$teacherRunId student=$studentRunId"
} finally {
    Pop-Location
}
