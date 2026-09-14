[CmdletBinding()]
param(
    [string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) 'cannjudge-submit-plaintext.zip')
)

$ErrorActionPreference = 'Stop'
$packageName = Split-Path -Leaf $PSScriptRoot
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cannjudge-package-" + [guid]::NewGuid())
$stagingPackage = Join-Path $stagingRoot $packageName
$excludedNames = @('.env', 'private.pem', 'public.pem', 'project.zip', '__pycache__')

try {
    New-Item -ItemType Directory -Path $stagingPackage | Out-Null

    Get-ChildItem -LiteralPath $PSScriptRoot -Force | ForEach-Object {
        if ($_.Name -in $excludedNames -or $_.Extension -in @('.pyc', '.zip')) {
            return
        }
        Copy-Item -LiteralPath $_.FullName -Destination $stagingPackage -Recurse -Force
    }

    $destination = [System.IO.Path]::GetFullPath($Output)
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    if (Test-Path -LiteralPath $destination) {
        Remove-Item -LiteralPath $destination -Force
    }

    Compress-Archive -LiteralPath $stagingPackage -DestinationPath $destination -CompressionLevel Optimal

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($destination)
    try {
        $unsafe = @($archive.Entries | Where-Object {
            $parts = $_.FullName -split '[/\\]'
            $parts -contains '.env' -or
            $parts -contains '__pycache__' -or
            $_.Name -in @('private.pem', 'public.pem', 'project.zip') -or
            $_.Name.EndsWith('.pyc', [System.StringComparison]::OrdinalIgnoreCase)
        })
        if ($unsafe.Count -gt 0) {
            throw "Unsafe files found in package: $($unsafe.FullName -join ', ')"
        }
    } finally {
        $archive.Dispose()
    }

    Write-Output $destination
} finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
