param([ValidateSet('preflight', 'run')][string]$Mode = 'preflight')
$ErrorActionPreference = 'Stop'
$releasePython = if ($env:MARL_PYTHON) { $env:MARL_PYTHON } else { 'python' }
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
if ($Mode -eq 'run') {
    foreach ($configName in @('LLM_BASE_URL', 'LLM_API_KEY', 'GPT61_BASE_URL', 'GPT61_API_KEY')) {
        $savedConfig = [Environment]::GetEnvironmentVariable($configName, 'User')
        if ([string]::IsNullOrWhiteSpace($savedConfig)) { throw "Missing saved User configuration: $configName" }
        [Environment]::SetEnvironmentVariable($configName, $savedConfig.Trim(), 'Process')
    }
    Remove-Variable savedConfig
    $env:P1A_MODEL = 'qwen3-next-80b-a3b-instruct'
}
& $releasePython -B -u "$PSScriptRoot\run_dev20.py" --mode $Mode
exit $LASTEXITCODE
