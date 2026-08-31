<#
.SYNOPSIS
Resolves one executable Conda command for the Windows launchers.

.DESCRIPTION
`Get-Command -CommandType Application` may return more than one executable
when several Conda installations are on PATH.  Native invocation (`&`) needs a
single scalar path, so this helper always returns exactly one path.  An
explicit filesystem path supplied through `-CondaExecutable` takes precedence.
#>

function Resolve-CondaCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$CondaExecutable,

        # This seam is used only by the PowerShell contract test.  Production
        # callers use the default, which queries application commands on PATH.
        [scriptblock]$CommandResolver
    )

    $hasDirectoryComponent = [System.IO.Path]::IsPathRooted($CondaExecutable) -or $CondaExecutable.IndexOfAny([char[]]'\\/') -ge 0
    if ($hasDirectoryComponent) {
        $item = Get-Item -LiteralPath $CondaExecutable -Force -ErrorAction SilentlyContinue
        if ($null -eq $item -or $item.PSProvider.Name -ne 'FileSystem' -or -not ($item -is [System.IO.FileInfo])) {
            throw "The explicit Conda executable does not exist: '$CondaExecutable'."
        }
        $allowedNames = @('conda.exe', 'conda.bat', 'conda.cmd')
        if ($item.Name.ToLowerInvariant() -notin $allowedNames) {
            throw "The explicit Conda executable must be a FileSystem conda.exe, conda.bat, or conda.cmd file: '$CondaExecutable'."
        }
        return [string]$item.FullName
    }

    if ($null -eq $CommandResolver) {
        $CommandResolver = {
            param([string]$Name)
            Get-Command -Name $Name -CommandType Application -ErrorAction Stop
        }
    }

    $candidates = [System.Collections.Generic.List[string]]::new()
    foreach ($command in @(& $CommandResolver $CondaExecutable)) {
        if ($null -eq $command) { continue }
        $path = ''
        if ($command.PSObject.Properties.Match('Path').Count -gt 0) { $path = [string]$command.Path }
        if ([string]::IsNullOrWhiteSpace($path) -and $command.PSObject.Properties.Match('Source').Count -gt 0) { $path = [string]$command.Source }
        if ([string]::IsNullOrWhiteSpace($path)) { continue }
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $resolved = [string](Resolve-Path -LiteralPath $path -ErrorAction Stop).ProviderPath
            if (-not $candidates.Contains($resolved)) { $candidates.Add($resolved) }
        }
    }

    if ($candidates.Count -eq 0) {
        throw "Unable to resolve an executable Conda command named '$CondaExecutable'. Add Conda to PATH or pass its full path with -CondaExecutable."
    }

    # Prefer the standard native executable, then the batch launcher.  The
    # final lexical tie-break makes a multi-install PATH deterministic.
    $selected = $candidates | Sort-Object `
        @{ Expression = {
            switch ([System.IO.Path]::GetFileName($_).ToLowerInvariant()) {
                'conda.exe' { 0; break }
                'conda.bat' { 1; break }
                'conda.cmd' { 2; break }
                default { 3; break }
            }
        } }, `
        @{ Expression = { $_.ToLowerInvariant() } } | Select-Object -First 1
    return [string]$selected
}
