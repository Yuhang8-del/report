param([string]$EnvName = "fruit-ssod")
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\project\scripts\run_delivery_self_check.ps1" -EnvName $EnvName
