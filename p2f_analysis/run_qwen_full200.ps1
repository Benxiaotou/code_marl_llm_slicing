param([ValidateSet('check','run')][string]$Mode = 'check')
$ErrorActionPreference = 'Stop'
$releasePython = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { 'python' }
if ($Mode -eq 'run') {
    foreach ($name in @('LLM_BASE_URL', 'LLM_API_KEY')) {
        $value = [Environment]::GetEnvironmentVariable($name, 'User')
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
    if ([string]::IsNullOrWhiteSpace($env:LLM_BASE_URL) -or [string]::IsNullOrWhiteSpace($env:LLM_API_KEY)) {
        throw 'Internal Qwen configuration missing; no requests started.'
    }
    $env:P1A_MODEL = 'qwen3-next-80b-a3b-instruct'
}
& $releasePython -B "$PSScriptRoot\qwen_full200.py" --mode $Mode
exit $LASTEXITCODE
