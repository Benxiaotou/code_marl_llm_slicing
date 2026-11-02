<###############################################################################
P2-B val5601 sequential launcher (PowerShell 5.1+).
Default: READ-ONLY preflight. Use -Execute to run uncompleted jobs serially.
No original experiment source/data/old run directory is overwritten.
###############################################################################>
param(
    [switch]$Execute,
    [ValidateSet('all', 'knn', 'mappo', 'rerank', 'ir_icl')]
    [string]$Scope = 'all',
    [switch]$ExtendedIR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $project 'p1c_runner.py') -PathType Leaf)) {
    throw "Place both P2-B queue scripts at your experiment project root: $project"
}

Push-Location $project
try {
    $cmdArgs = @('-u', '.\p2b_run_val5601_queue.py', '--scope', $Scope)
    if ($Execute) { $cmdArgs += '--execute' }
    if ($ExtendedIR) { $cmdArgs += '--extended-ir' }
    & python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "P2-B queue stopped with exit code $LASTEXITCODE. Read the STOPPED message; no results were overwritten."
    }
} finally {
    Pop-Location
}
