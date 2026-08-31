param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$ProjectRoot = 'D:\fruit_ssod_complete_project1\project',
    [string]$DeliveryRoot = 'D:\fruit_ssod_complete_project1',
    [string]$RuntimeRoot = 'E:\fruit_ssod_runtime',
    [int]$Limit = 12,
    [int]$StartIndex = 0,
    [int]$PauseSeconds = 2
)

$ErrorActionPreference = 'Stop'
$output = Join-Path $DeliveryRoot 'outputs\customer_incremental_11class_gui_screenshots'
$manifestPath = Join-Path $DeliveryRoot 'outputs\customer_incremental_11class_examples\example_manifest.json'
$examples = (Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json).images
$captureHelper = 'C:\Users\linyugui\.codex\skills\screenshot\scripts\take_screenshot.ps1'
$progressLog = Join-Path $output '_capture_batch.log'
New-Item -ItemType Directory -Path $output -Force | Out-Null

$windowCode = @'
using System;
using System.Runtime.InteropServices;
public static class GuiCaptureWindow {
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr handle, int command);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr handle);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr handle);
    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr handle, IntPtr insertAfter, int x, int y, int width, int height, uint flags);
}
'@
Add-Type $windowCode -ErrorAction SilentlyContinue
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue

$captureCount = [Math]::Min($examples.Count - $StartIndex, $Limit)
$endIndex = $StartIndex + $captureCount
for ($index = $StartIndex; $index -lt $endIndex; $index++) {
    $number = $index + 1
    $stem = [IO.Path]::GetFileNameWithoutExtension([string]$examples[$index])
    $destination = Join-Path $output ($stem + '_GUI.png')
    $stdout = Join-Path $output ("_window_{0:D2}.stdout.log" -f $index)
    $stderr = Join-Path $output ("_window_{0:D2}.stderr.log" -f $index)
    Remove-Item -LiteralPath $stdout,$stderr -ErrorAction SilentlyContinue
    $arguments = @(
        '-u', 'scripts\generate_gui_inference_screenshots.py',
        '--detector', (Join-Path $DeliveryRoot 'models\incremental_11class_best.pt'),
        '--registry', (Join-Path $DeliveryRoot 'models\class_registry_v2.json'),
        '--objectness', (Join-Path $DeliveryRoot 'models\open_world_objectness.pt'),
        '--encoder', (Join-Path $DeliveryRoot 'models\open_world_encoder.pt'),
        '--clusters', (Join-Path $DeliveryRoot 'models\open_world_box_clusters.npz'),
        '--names', (Join-Path $DeliveryRoot 'models\open_world_cluster_names.json'),
        '--protected-truth', (Join-Path $RuntimeRoot 'data\fruit_ssod\processed\yolo\open_world_v1_seed42\protocol\protected_novel_box_truth.json'),
        '--example-manifest', $manifestPath,
        '--output', $output,
        '--device', '0',
        '--show-index', [string]$index
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    try {
        $deadline = (Get-Date).AddSeconds(45)
        $match = $null
        while ((Get-Date) -lt $deadline) {
            if ($process.HasExited) {
                $errorText = if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Raw } else { '' }
                throw "GUI process exited before exposing a window handle: $errorText"
            }
            if (Test-Path $stdout) {
                $text = Get-Content -LiteralPath $stdout -Raw
                if ($text) {
                    $match = [regex]::Match($text, 'GUI_HANDLE=(\d+);CATEGORY=([^;]+);IMAGE_ID=([^\r\n]+)')
                    if ($match.Success) { break }
                }
            }
            Start-Sleep -Milliseconds 300
        }
        if ($null -eq $match -or -not $match.Success) { throw "timed out waiting for GUI window at index $index" }
        $handleValue = [int64]$match.Groups[1].Value
        $handle = [IntPtr]$handleValue
        [GuiCaptureWindow]::ShowWindow($handle, 9) | Out-Null
        # Keep the target above Codex while the screenshot helper starts in a
        # separate process. Flags: NOMOVE | NOSIZE | SHOWWINDOW.
        [GuiCaptureWindow]::SetWindowPos($handle, [IntPtr](-1), 0, 0, 0, 0, 0x43) | Out-Null
        [GuiCaptureWindow]::SetForegroundWindow($handle) | Out-Null
        Start-Sleep -Milliseconds 1800
        if (-not [GuiCaptureWindow]::IsWindowVisible($handle)) { throw "GUI window is not visible before capture" }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $captureHelper -WindowHandle $handleValue -Path $destination | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $destination)) {
            throw "window capture failed for index $index"
        }
        Add-Content -LiteralPath $progressLog -Value ("{0}/{1} {2} {3}" -f $number,$examples.Count,$match.Groups[2].Value,$destination) -Encoding UTF8
    } finally {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $stdout,$stderr -ErrorAction SilentlyContinue
        if ($PauseSeconds -gt 0) { Start-Sleep -Seconds $PauseSeconds }
    }
}

Add-Content -LiteralPath $progressLog -Value ("complete screenshots={0}" -f $captureCount) -Encoding UTF8
