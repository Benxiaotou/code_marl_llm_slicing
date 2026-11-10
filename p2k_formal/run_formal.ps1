param()
$ErrorActionPreference = 'Stop'
$releasePython = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { 'python' }
foreach ($name in @('LLM_BASE_URL', 'LLM_API_KEY')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if ([string]::IsNullOrWhiteSpace($value)) { throw ($name + ' is not configured.') }
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
foreach ($name in @('GPT61_API_KEY', 'GPT61_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_BASE_URL')) {
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$env:P1A_MODEL = 'qwen3-next-80b-a3b-instruct'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskTemp = Join-Path $PSScriptRoot ('temp_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTemp | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
try {
    & $releasePython -B .\p2k_formal\run_formal.py
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
