$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $RepositoryRoot 'scripts\resolve_conda_command.ps1')

$TestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("fruit-ssod-conda-resolve-" + [guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
    $BatchPath = Join-Path $TestRoot 'conda.bat'
    $ExecutablePath = Join-Path $TestRoot 'conda.exe'
    $CmdPath = Join-Path $TestRoot 'conda.cmd'
    $ScriptPath = Join-Path $TestRoot 'conda.ps1'
    $TextPath = Join-Path $TestRoot 'conda.txt'
    $OtherExecutablePath = Join-Path $TestRoot 'not-conda.exe'
    Set-Content -LiteralPath $BatchPath -Value '@echo off' -NoNewline
    Set-Content -LiteralPath $ExecutablePath -Value '' -NoNewline
    Set-Content -LiteralPath $CmdPath -Value '@echo off' -NoNewline
    Set-Content -LiteralPath $ScriptPath -Value 'Write-Output nope' -NoNewline
    Set-Content -LiteralPath $TextPath -Value 'not an executable' -NoNewline
    Set-Content -LiteralPath $OtherExecutablePath -Value '' -NoNewline

    $MultipleMatches = @(
        [pscustomobject]@{ Path = $BatchPath; Source = $BatchPath },
        [pscustomobject]@{ Path = $ExecutablePath; Source = $ExecutablePath }
    )
    $Resolved = Resolve-CondaCommand -CondaExecutable 'conda' -CommandResolver {
        param([string]$Name)
        $MultipleMatches
    }
    if ($Resolved -is [array]) { throw 'Resolved Conda command must be scalar.' }
    if ($Resolved -ne (Resolve-Path -LiteralPath $ExecutablePath).ProviderPath) {
        throw "Expected deterministic conda.exe choice; got '$Resolved'."
    }

    $Explicit = Resolve-CondaCommand -CondaExecutable $BatchPath -CommandResolver {
        throw 'A full explicit path must not be looked up on PATH.'
    }
    if ($Explicit -ne (Resolve-Path -LiteralPath $BatchPath).ProviderPath) {
        throw "Explicit Conda executable was not respected; got '$Explicit'."
    }

    $CmdExplicit = Resolve-CondaCommand -CondaExecutable $CmdPath
    if ($CmdExplicit -ne (Resolve-Path -LiteralPath $CmdPath).ProviderPath) {
        throw "Explicit conda.cmd was not respected; got '$CmdExplicit'."
    }

    foreach ($InvalidPath in @($ScriptPath, $TextPath, $OtherExecutablePath)) {
        try {
            Resolve-CondaCommand -CondaExecutable $InvalidPath | Out-Null
            throw "Expected explicit non-Conda path '$InvalidPath' to be rejected."
        }
        catch {
            if ($_.Exception.Message -like "Expected explicit non-Conda path*") { throw }
            if ($_.Exception.Message -notlike '*conda.exe, conda.bat, or conda.cmd*') {
                throw "Unexpected rejection for '$InvalidPath': $($_.Exception.Message)"
            }
        }
    }

    Set-Item -Path Function:\conda.exe -Value { Write-Output fake }
    try {
        Resolve-CondaCommand -CondaExecutable 'Function:\conda.exe' | Out-Null
        throw 'Expected non-FileSystem explicit Conda path to be rejected.'
    }
    catch {
        if ($_.Exception.Message -like 'Expected non-FileSystem*') { throw }
        if ($_.Exception.Message -notlike 'The explicit Conda executable does not exist:*') {
            throw "Unexpected non-FileSystem rejection: $($_.Exception.Message)"
        }
    }
    finally {
        Remove-Item -Path Function:\conda.exe -ErrorAction SilentlyContinue
    }
    Write-Host 'Resolve-CondaCommand multiple-match contract: PASS'
}
finally {
    if (Test-Path -LiteralPath $TestRoot) { Remove-Item -LiteralPath $TestRoot -Recurse -Force }
}
