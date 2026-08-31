param(
    [string]$Python = 'E:\anaconda\envs\fruit-ssod\python.exe',
    [string]$ProjectRoot = 'D:\fruit_ssod_complete_project1\project',
    [string]$DeliveryRoot = 'D:\fruit_ssod_complete_project1',
    [string]$TestImageRoot = 'E:\fruit_ssod_runtime\data\fruit_ssod\processed\yolo\supervised_v2_100_seed42\images\test'
)

$ErrorActionPreference = 'Stop'
$output = Join-Path $DeliveryRoot 'outputs\customer_inference_gui_images'
$model = Join-Path $DeliveryRoot 'models\student_best.pt'
$captureHelper = 'C:\Users\linyugui\.codex\skills\screenshot\scripts\take_screenshot.ps1'
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('fruit_ssod_gui_capture_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

$examples = @(
    @{ Label = 'Apple';      Id = 'cd5d140218805739' },
    @{ Label = 'Apple';      Id = 'cc48c73b8406f7c8' },
    @{ Label = 'Banana';     Id = 'f255f1c229310fb8' },
    @{ Label = 'Banana';     Id = 'c077585b82a30324' },
    @{ Label = 'Orange';     Id = 'd026f29e6f0fe84c' },
    @{ Label = 'Orange';     Id = 'd3bad7a54afc6eef' },
    @{ Label = 'Orange';     Id = 'f88c11ff7d959391' },
    @{ Label = 'Strawberry'; Id = 'ceadeca51ad3554b' },
    @{ Label = 'Strawberry'; Id = 'fdb470561d52c8a8' },
    @{ Label = 'Strawberry'; Id = 'ef980578956d7605' },
    @{ Label = 'Pineapple';  Id = 'f390b1a50ef67591' },
    @{ Label = 'Pineapple';  Id = '91ab0883e18b264b' }
)

$windowCode = @(
    'using System;',
    'using System.Runtime.InteropServices;',
    'public static class StudentGuiCaptureWindow {',
    '    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr handle, int command);',
    '    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr handle);',
    '    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr handle);',
    '    [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr handle, IntPtr insertAfter, int x, int y, int width, int height, uint flags);',
    '}'
) -join [Environment]::NewLine
Add-Type $windowCode -ErrorAction SilentlyContinue
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue

try {
    for ($index = 0; $index -lt $examples.Count; $index++) {
        $example = $examples[$index]
        $image = Join-Path $TestImageRoot ($example.Id + '.jpg')
        if (-not (Test-Path -LiteralPath $image)) { throw "missing test image: $image" }
        $number = $index + 1
        $destination = Join-Path $output ("SSOD_{0:D2}_{1}_{2}_GUI.png" -f $number,$example.Label,$example.Id)
        $stdout = Join-Path $tempRoot ("window_{0:D2}.stdout.log" -f $number)
        $stderr = Join-Path $tempRoot ("window_{0:D2}.stderr.log" -f $number)
        $arguments = @(
            '-u', 'scripts\show_student_gui_inference.py',
            '--model', $model,
            '--image', $image,
            '--label', $example.Label
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        try {
            $deadline = (Get-Date).AddSeconds(75)
            $match = $null
            while ((Get-Date) -lt $deadline) {
                if ($process.HasExited) {
                    $errorText = if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Raw } else { '' }
                    throw "Student GUI exited before capture: $errorText"
                }
                if (Test-Path $stdout) {
                    $text = Get-Content -LiteralPath $stdout -Raw
                    if ($text) {
                        $match = [regex]::Match($text, 'GUI_HANDLE=(\d+);LABEL=([^;]+);IMAGE_ID=([^;]+);DETECTIONS=(\d+);COUNTS=([^\r\n]+)')
                        if ($match.Success) { break }
                    }
                }
                Start-Sleep -Milliseconds 300
            }
            if ($null -eq $match -or -not $match.Success) { throw "timed out waiting for Student GUI index $number" }
            $handleValue = [int64]$match.Groups[1].Value
            $handle = [IntPtr]$handleValue
            [StudentGuiCaptureWindow]::ShowWindow($handle, 9) | Out-Null
            [StudentGuiCaptureWindow]::SetWindowPos($handle, [IntPtr](-1), 0, 0, 0, 0, 0x43) | Out-Null
            [StudentGuiCaptureWindow]::SetForegroundWindow($handle) | Out-Null
            Start-Sleep -Milliseconds 1500
            if (-not [StudentGuiCaptureWindow]::IsWindowVisible($handle)) { throw 'Student GUI window is not visible' }
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $captureHelper -WindowHandle $handleValue -Path $destination | Out-Null
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $destination)) {
                throw "window capture failed for index $number"
            }
            Write-Output ("{0}/{1} {2} detections={3} counts={4}" -f $number,$examples.Count,$destination,$match.Groups[4].Value,$match.Groups[5].Value)
        }
        finally {
            if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Milliseconds 800
        }
    }
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
