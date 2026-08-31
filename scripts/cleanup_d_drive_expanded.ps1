$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath "D:\fruit_ssod_complete_project1").Path
$AllowedNames = @(
    "_expanded_full_data_pending_delete",
    "_expanded_full_artifacts_pending_delete",
    "_slim_stage"
)
$Targets = @(
    "D:\fruit_ssod_complete_project1\_expanded_full_data_pending_delete",
    "D:\fruit_ssod_complete_project1\_expanded_full_artifacts_pending_delete",
    "D:\fruit_ssod_complete_project1\_slim_stage"
)

function Remove-LongPathTree {
    param([Parameter(Mandatory = $true)][string]$Path)
    $LongPath = "\\?\$Path"
    if (-not [System.IO.Directory]::Exists($LongPath)) {
        return
    }
    foreach ($File in [System.IO.Directory]::EnumerateFiles(
        $LongPath,
        "*",
        [System.IO.SearchOption]::AllDirectories
    )) {
        try {
            [System.IO.File]::SetAttributes($File, [System.IO.FileAttributes]::Normal)
        }
        catch {
            Write-Output "ATTRIBUTE_WARNING=$File"
        }
    }
    [System.IO.File]::SetAttributes($LongPath, [System.IO.FileAttributes]::Normal)
    [System.IO.Directory]::Delete($LongPath, $true)
}

foreach ($Target in $Targets) {
    if (-not (Test-Path -LiteralPath $Target)) {
        continue
    }
    $Resolved = (Resolve-Path -LiteralPath $Target).Path
    if ((Split-Path -Parent $Resolved) -ne $Root) {
        throw "Unsafe parent for delete target: $Resolved"
    }
    if ((Split-Path -Leaf $Resolved) -notin $AllowedNames) {
        throw "Unexpected delete target: $Resolved"
    }
    Write-Output "VERIFIED_DELETE_TARGET=$Resolved"
    try {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    }
    catch {
        Write-Output "REMOVE_ITEM_FALLBACK=$Resolved"
        Remove-LongPathTree -Path $Resolved
    }
    if (Test-Path -LiteralPath $Resolved) {
        Remove-LongPathTree -Path $Resolved
    }
    Write-Output "DELETED=$Resolved"
}

$ToolTest = "D:\fruit_ssod_complete_project1\outputs\archive_tool_test.tar.zst"
if (Test-Path -LiteralPath $ToolTest) {
    Remove-Item -LiteralPath $ToolTest -Force
    Write-Output "DELETED=$ToolTest"
}
