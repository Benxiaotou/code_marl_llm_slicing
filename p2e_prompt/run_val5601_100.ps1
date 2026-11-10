param(
    [ValidateSet('check', 'run', 'compare')][string]$Mode = 'check',
    [ValidateSet('all', 'mappo', 'semantic_balance', 'joint_feasibility')][string]$Method = 'all'
)
$ErrorActionPreference = 'Stop'
if ($Mode -eq 'run' -and $Method -ne 'mappo') {
    foreach ($name in @('GPT61_API_KEY', 'GPT61_BASE_URL')) {
        $savedValue = [Environment]::GetEnvironmentVariable($name, 'User')
        if ([string]::IsNullOrWhiteSpace($savedValue)) { throw "$name not saved in User environment" }
        [Environment]::SetEnvironmentVariable($name, $savedValue.Trim(), 'Process')
    }
    Remove-Variable savedValue
}
& conda run --no-capture-output -n marl python -B -u (Join-Path $PSScriptRoot 'val5601_100.py') --mode $Mode --method $Method
exit $LASTEXITCODE
