[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ReleaseRoot,
    [Parameter(Mandatory)]
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ReleaseRoot -ErrorAction Stop).Path
$destination = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $destination) { throw "Delivery manifest already exists: $destination" }
$rootWithSlash = $root.TrimEnd('\') + '\'
if ($destination.StartsWith($rootWithSlash, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Write the delivery manifest outside the immutable release root.' }
$forbiddenNames = @('.env', 'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519', 'credentials.json', 'secrets.json')
$files = @(Get-ChildItem -LiteralPath $root -File -Recurse -Force)
$records = foreach ($file in $files) {
    if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Symbolic links are not permitted in the release package: $($file.FullName)" }
    if ($forbiddenNames -contains $file.Name.ToLowerInvariant()) { throw "Potential credential file is not permitted: $($file.FullName)" }
    # Compatible with Windows PowerShell 5.1 as well as newer PowerShell.
    $rootUri = [System.Uri]($rootWithSlash)
    $fileUri = [System.Uri]$file.FullName
    $relative = [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($fileUri).ToString()).Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{ relative_path = $relative; bytes = [int64]$file.Length; sha256 = $hash }
}
if ($records.Count -eq 0) { throw "Release root contains no files: $root" }
$payload = [ordered]@{
    schema_version = '1.0'
    protocol = 'fruit_ssod_delivery_manifest_v1'
    release_root = $root
    file_count = $records.Count
    files = @($records | Sort-Object relative_path)
}
$parent = Split-Path -Parent $destination
New-Item -ItemType Directory -Force -Path $parent | Out-Null
[System.IO.File]::WriteAllText($destination, ($payload | ConvertTo-Json -Depth 5) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Output "Wrote immutable delivery manifest: $destination"
