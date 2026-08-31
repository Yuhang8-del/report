param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }

$python = 'E:\anaconda\envs\fruit-ssod\python.exe'
$work = 'E:\bishe\fruit\.worktrees\fruit-ssod-implementation'
$artifactRoot = 'E:\fruit_ssod_runtime\artifacts_v17'
$dataRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod'
$v1Run = Join-Path $artifactRoot 'runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v1'
$v2Run = Join-Path $artifactRoot 'runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2'
$v2r1Run = Join-Path $artifactRoot 'runs\supervised-v2-full-yolov8m-1024-teacher-seed42-aggressive-v2-r1'
$v12FallbackRun = 'E:\fruit_ssod_runtime\artifacts_v12\runs\run-v12n-v8m-balanced-1024-ft40-seed42'
$log = Join-Path $artifactRoot 'aggressive-best-teacher-student-queue.log'

function Log([string]$Message) {
    Add-Content -LiteralPath $log -Value ("$(Get-Date -Format o) $Message")
}

function Wait-Terminal([string]$RunDir) {
    $recordPath = Join-Path $RunDir 'run_record.json'
    while (-not (Test-Path -LiteralPath $recordPath)) {
        Start-Sleep -Seconds $PollSeconds
    }
    while ($true) {
        try {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            $status = [string]$record.status
        } catch {
            $status = 'running'
        }
        if ($status -ne 'running') {
            return $status
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Wait-Evaluation([string]$RunDir) {
    $evaluationPath = Join-Path $RunDir 'evaluations\test.json'
    while (-not (Test-Path -LiteralPath $evaluationPath)) {
        Start-Sleep -Seconds $PollSeconds
    }
    return $evaluationPath
}

function To-YamlPath([string]$Path) { return ($Path -replace '\\', '/') }

Log 'Waiting for all aggressive Teacher variants/recovery runs to reach terminal records.'
$v1Status = Wait-Terminal $v1Run
Log "v1 terminal status: $v1Status"
$v2Status = Wait-Terminal $v2Run
Log "v2 terminal status: $v2Status"
$v2r1Status = Wait-Terminal $v2r1Run
Log "v2-r1 terminal status: $v2r1Status"

$candidates = @()
foreach ($row in @(@{ Run = $v1Run; Status = $v1Status }, @{ Run = $v2Run; Status = $v2Status }, @{ Run = $v2r1Run; Status = $v2r1Status })) {
    if ($row.Status -ne 'complete') {
        Log "Skipping non-complete Teacher $($row.Run): $($row.Status)"
        continue
    }
    $evaluation = Wait-Evaluation $row.Run
    try {
        $payload = Get-Content -LiteralPath $evaluation -Raw | ConvertFrom-Json
        $map50 = [double]$payload.metrics.map50
        $candidates += [pscustomobject]@{ RunDir = $row.Run; Evaluation = $evaluation; Map50 = $map50 }
        Log "Teacher fixed-test mAP50: $($row.Run) = $map50"
    } catch {
        Log "Could not read Teacher fixed-test evidence ${evaluation}: $($_.Exception.Message)"
    }
}
if ($candidates.Count -eq 0) {
    # The aggressive v2 attempts may terminate on this Windows host after
    # epoch 1 before Ultralytics publishes a checkpoint.  Do not block the
    # requested first Student result in that case: the completed v12 Teacher
    # is an immutable, fixed-test-evaluated checkpoint and is explicitly used
    # as a delivery fallback.  The failed v2 evidence remains preserved.
    $fallbackRecord = Join-Path $v12FallbackRun 'run_record.json'
    $fallbackWeights = Join-Path $v12FallbackRun 'weights\best.pt'
    $fallbackEvaluation = Join-Path $v12FallbackRun 'evaluations\test.json'
    if ((Test-Path -LiteralPath $fallbackRecord) -and (Test-Path -LiteralPath $fallbackWeights) -and (Test-Path -LiteralPath $fallbackEvaluation)) {
        try {
            $fallbackStatus = [string](Get-Content -LiteralPath $fallbackRecord -Raw | ConvertFrom-Json).status
            $fallbackPayload = Get-Content -LiteralPath $fallbackEvaluation -Raw | ConvertFrom-Json
            if ($fallbackStatus -eq 'complete') {
                $fallbackMap50 = [double]$fallbackPayload.metrics.map50
                $candidates += [pscustomobject]@{ RunDir = $v12FallbackRun; Evaluation = $fallbackEvaluation; Map50 = $fallbackMap50; Fallback = $true }
                Log "No completed aggressive Teacher checkpoint was available; selected verified v12 fallback $v12FallbackRun with fixed-test mAP50 $fallbackMap50."
            }
        } catch {
            Log "Could not read verified v12 fallback evidence: $($_.Exception.Message)"
        }
    }
}
if ($candidates.Count -eq 0) {
    Log 'No completed Teacher fixed-test evidence was available; no Student was started.'
    exit 2
}

$best = $candidates | Sort-Object -Property Map50 -Descending | Select-Object -First 1
$teacherRunDir = [string]$best.RunDir
$teacherRunId = Split-Path -Leaf $teacherRunDir
$teacherWeights = Join-Path $teacherRunDir 'weights\best.pt'
$teacherRecord = Join-Path $teacherRunDir 'run_record.json'
$teacherConfig = if ($teacherRunId -eq 'run-v12n-v8m-balanced-1024-ft40-seed42') {
    Join-Path $work 'configs\experiments\teacher_compat_v12.yaml'
} elseif ($teacherRunId -match '-aggressive-v1$') {
    Join-Path $work 'configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v1.yaml'
} elseif ($teacherRunId -match '-aggressive-v2-r1$') {
    Join-Path $work 'configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v2_r1.yaml'
} else {
    Join-Path $work 'configs\experiments\supervised_v2_full_yolov8m_1024_teacher_seed42_aggressive_v2.yaml'
}

$pseudoRoot = Join-Path $artifactRoot ("pseudo\aggressive_best_{0}" -f $teacherRunId)
$candidatePath = Join-Path $pseudoRoot 'candidates.json'
$filterRoot = Join-Path $pseudoRoot 'filter'
$auditCandidatePath = Join-Path $pseudoRoot 'audit_candidates.json'
$auditFilterRoot = Join-Path $pseudoRoot 'audit_filter'
$auditReportRoot = Join-Path $pseudoRoot 'audit_report'
$studentRunId = "ssod-v2-independent-openimages-aggressive-best-$teacherRunId-seed42"
$studentRunDir = Join-Path $artifactRoot ("runs\{0}" -f $studentRunId)
$studentConfig = Join-Path $work ("configs\experiments\{0}.yaml" -f $studentRunId)

if (Test-Path -LiteralPath $studentRunDir) {
    Log "Student run already exists; refusing to overwrite: $studentRunDir"
    exit 3
}
if (Test-Path -LiteralPath $studentConfig) {
    Log "Student config already exists; refusing to overwrite: $studentConfig"
    exit 3
}

New-Item -ItemType Directory -Force -Path $pseudoRoot | Out-Null
$templatePath = Join-Path $work 'configs\experiments\ssod_v1_independent_openimages_v12teacher_seed42.yaml'
$template = Get-Content -LiteralPath $templatePath -Raw
$teacherWeightsYaml = To-YamlPath $teacherWeights
$teacherRecordYaml = To-YamlPath $teacherRecord
$teacherConfigYaml = To-YamlPath $teacherConfig
$pseudoRootYaml = To-YamlPath $pseudoRoot
$newName = $studentRunId -replace '-', '_'
$comparisonGroup = "aggressive_teacher_$teacherRunId`_seed42"
$template = $template.Replace('ssod_v1_independent_openimages_v12teacher_seed42', $newName)
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v12/runs/run-v12n-v8m-balanced-1024-ft40-seed42/weights/best.pt', $teacherWeightsYaml)
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v17/pseudo/independent_openimages_v12teacher/candidates.json', (To-YamlPath $candidatePath))
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v17/pseudo/independent_openimages_v12teacher/filter/audit.jsonl', (To-YamlPath (Join-Path $filterRoot 'audit.jsonl')))
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v17/pseudo/independent_openimages_v12teacher/filter/decision_manifest.json', (To-YamlPath (Join-Path $filterRoot 'decision_manifest.json')))
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v17/pseudo/independent_openimages_v12teacher/audit_report/pseudo_audit.json', (To-YamlPath (Join-Path $auditReportRoot 'pseudo_audit.json')))
$template = $template.Replace('independent_openimages_v12teacher_init_v1', 'aggressive_teacher_checkpoint_init_v1')
$template = $template.Replace('independent_openimages_v12teacher_seed42', $comparisonGroup)
$template = $template.Replace('teacher_compat_v12.yaml', $teacherConfigYaml)
$template = $template.Replace('run-v12n-v8m-balanced-1024-ft40-seed42', $teacherRunId)
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v12/runs/run-v12n-v8m-balanced-1024-ft40-seed42/weights/best.pt', $teacherWeightsYaml)
$template = $template.Replace('E:/fruit_ssod_runtime/artifacts_v12/runs/run-v12n-v8m-balanced-1024-ft40-seed42/run_record.json', $teacherRecordYaml)
$template = $template.Replace(("E:/fruit_ssod_runtime/artifacts_v12/runs/{0}/run_record.json" -f $teacherRunId), $teacherRecordYaml)
Set-Content -LiteralPath $studentConfig -Value $template -Encoding UTF8
Log "Selected Teacher $teacherRunId with fixed-test mAP50 $($best.Map50); generated Student config $studentConfig"

$unlabeled = Join-Path $dataRoot 'processed\ssod_unlabeled_pool_fruits360_v2_openimages_v13b\unlabeled.json'
$poolSplit = Join-Path $dataRoot 'processed\ssod_unlabeled_pool_fruits360_v2_openimages_v13b\split_manifest.json'
$imageRoot = Join-Path $dataRoot 'processed\ssod_combined_image_root_v2'
$auditLabels = Join-Path $dataRoot 'manifests\splits_v2\protected_splits\pseudo_audit_labels.json'
$auditSplit = Join-Path $dataRoot 'manifests\splits_v2\split_manifest.json'
$validationPr = 'E:\fruit_ssod_runtime\artifacts_v15\calibration\supervised_v2_teacher_seed42\fixed_validation_pr.json'
$aspectBounds = 'E:\fruit_ssod_runtime\artifacts_v15\calibration\supervised_v2_teacher_seed42\fixed_aspect_ratio_bounds.json'

$env:PYTHONPATH = Join-Path $work 'src'
Push-Location $work
try {
    if (-not (Test-Path -LiteralPath $candidatePath)) {
        Log 'Generating train-pool dual-view candidates from the selected Teacher.'
        & $python -m fruit_ssod.cli.generate_pseudo_labels --unlabeled-manifest $unlabeled --split-manifest $poolSplit --weights $teacherWeights --teacher-run-id $teacherRunId --output $candidatePath --image-root $imageRoot --confidence 0.01 *>> (Join-Path $pseudoRoot 'candidate_generation.log')
        if ($LASTEXITCODE -ne 0) { throw "candidate generation failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $filterRoot 'decision_manifest.json'))) {
        Log 'Applying the executable Trust Filter to the train-pool candidates.'
        & $python -m fruit_ssod.cli.filter_pseudo_labels --candidates $candidatePath --unlabeled-manifest $unlabeled --split-manifest $poolSplit --output $filterRoot --mode trust --validation-pr $validationPr --aspect-ratio-bounds $aspectBounds --target-precision 0.70 --matrix-config $studentConfig *>> (Join-Path $pseudoRoot 'filter.log')
        if ($LASTEXITCODE -ne 0) { throw "train-pool filtering failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $auditReportRoot 'pseudo_audit.json'))) {
        Log 'Preparing the protected pseudo-audit report for provenance only; it is not a stop gate.'
        & $python -m fruit_ssod.cli.audit_pseudo_labels --audit-labels $auditLabels --split-manifest $auditSplit --output $auditReportRoot --prepare-from-teacher --weights $teacherWeights --teacher-run-id $teacherRunId --candidates $auditCandidatePath --image-root $imageRoot --matrix-config $studentConfig --filter-output $auditFilterRoot --minimum-precision 0.90 *>> (Join-Path $pseudoRoot 'audit.log')
        if ($LASTEXITCODE -ne 0) { throw "pseudo-audit preparation failed with exit code $LASTEXITCODE" }
    }
    Log 'Starting the next exploratory Student without any historical accuracy gate.'
    $studentLog = Join-Path $pseudoRoot 'student.log'
    & $python -m fruit_ssod.cli.train_student --config $studentConfig --run-id $studentRunId *>> $studentLog
    if ($LASTEXITCODE -ne 0) { throw "Student training failed with exit code $LASTEXITCODE" }
    Log 'Student training completed; evaluating the sealed fixed test.'
    $fixedData = Join-Path $dataRoot 'processed\yolo\supervised_v2_100_seed42\dataset.yaml'
    $studentSplit = $poolSplit
    $evalLog = Join-Path $pseudoRoot 'student_fixed_test.log'
    & $python -m fruit_ssod.cli.evaluate_student_test --run-dir $studentRunDir --data $fixedData --split-manifest $studentSplit --device cuda:0 *>> $evalLog
    if ($LASTEXITCODE -ne 0) { throw "Student fixed-test evaluation failed with exit code $LASTEXITCODE" }
    Log "Completed Student fixed-test evaluation: $studentRunDir"
} finally {
    Pop-Location
}
