param(
    [ValidateSet('check', 'run', 'compare')][string]$Mode = 'check',
    [ValidateSet('semantic_balance', 'teacher_scale', 'joint_feasibility')][string]$Variant,
    [double]$BudgetUsd = 2,
    [string]$ReportDir
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if ($Mode -eq 'run') {
    if (-not $Variant) { throw 'Run requires -Variant' }
    foreach ($name in @('GPT61_API_KEY', 'GPT61_BASE_URL')) {
        $savedValue = [Environment]::GetEnvironmentVariable($name, 'User')
        if ([string]::IsNullOrWhiteSpace($savedValue)) { throw "$name is not saved in User environment" }
        [Environment]::SetEnvironmentVariable($name, $savedValue.Trim(), 'Process')
    }
    Remove-Variable savedValue
}
$runArgs = @('run', '--no-capture-output', '-n', 'marl', 'python', '-B', '-u',
             (Join-Path $PSScriptRoot 'p2e_prompt_3way.py'),
             '--mode', $Mode, '--project', $projectRoot)
if ($Variant) { $runArgs += @('--variant', $Variant) }
if ($Mode -eq 'run') { $runArgs += @('--budget-usd', $BudgetUsd.ToString([Globalization.CultureInfo]::InvariantCulture)) }
if ($ReportDir) { $runArgs += @('--report-dir', $ReportDir) }
& conda @runArgs
exit $LASTEXITCODE
