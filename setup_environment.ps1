param([string]$EnvName = "fruit-ssod")
$ErrorActionPreference = "Stop"
& "$PSScriptRoot\project\scripts\setup_delivery_environment.ps1" -EnvName $EnvName
