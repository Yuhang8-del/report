$ErrorActionPreference = 'Stop'
$deliveryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Join-Path $deliveryRoot 'project'
$python = 'E:\anaconda\envs\fruit-ssod\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 fruit-ssod Conda 环境：$python"
}

$env:PYTHONPATH = Join-Path $projectRoot 'src'
Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'scripts\open_world_demo.py')
    if ($LASTEXITCODE -ne 0) {
        throw "开放世界 GUI 启动失败，退出码：$LASTEXITCODE"
    }
} finally {
    Pop-Location
}
