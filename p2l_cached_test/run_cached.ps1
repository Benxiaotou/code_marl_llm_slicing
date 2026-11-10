param()
$ErrorActionPreference = 'Stop'
$releasePython = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { 'python' }
$env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($name in @('LLM_BASE_URL', 'LLM_API_KEY', 'P1A_MODEL', 'GPT61_API_KEY', 'GPT61_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_BASE_URL')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$taskTemp = Join-Path $PSScriptRoot ('temp_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    & $releasePython -B .\p2l_cached_test\run_cached.py
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
