$ErrorActionPreference = 'Stop'
$releasePython = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { 'python' }
foreach ($name in @('LLM_BASE_URL', 'LLM_API_KEY', 'P1A_MODEL')) {
    $value = [Environment]::GetEnvironmentVariable($name, 'User')
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
if ([string]::IsNullOrWhiteSpace($env:LLM_BASE_URL) -or [string]::IsNullOrWhiteSpace($env:LLM_API_KEY)) {
    throw 'Qwen environment is incomplete; no API call started.'
}
$env:P1A_MODEL = 'qwen3-next-80b-a3b-instruct'
& $releasePython -B "$PSScriptRoot\qwen_link_pilot.py"
exit $LASTEXITCODE
