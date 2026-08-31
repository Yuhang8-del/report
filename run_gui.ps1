param([string]$EnvName = "fruit-ssod")
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\project\scripts\run_delivery_gui.ps1" -EnvName $EnvName
