param(
    [ValidateSet('plan','prepare','probe','run','costs')][string]$Mode = 'plan',
    [ValidateRange(1,20)][int]$Scenes = 20,
    [ValidateRange(1,20)][int]$ChunkSize = 5,
    [ValidateSet('chat','responses')][string]$UpstreamApi = 'chat',
    [string]$UpstreamModel = 'gpt-6.1-sol',
    [ValidateSet('low','medium','high','xhigh','max')][string]$Effort = 'medium',
    [double]$BudgetUsd = 8.0,
    [int]$MaxOutputTokens = 8192,
    [double]$InputUsdPerM = 2.0,
    [double]$CachedInputUsdPerM = 0.1,
    [double]$OutputUsdPerM = 10.0,
    [switch]$AllowHttpUpstream
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$argv = @('.\gpt61_sol_dev.py', '--mode', $Mode, '--scenes', $Scenes,
          '--chunk-size', $ChunkSize, '--upstream-api', $UpstreamApi,
          '--upstream-model', $UpstreamModel, '--effort', $Effort,
          '--budget-usd', $BudgetUsd.ToString([cultureinfo]::InvariantCulture),
          '--max-output-tokens', $MaxOutputTokens,
          '--input-usd-per-m', $InputUsdPerM.ToString([cultureinfo]::InvariantCulture),
          '--cached-input-usd-per-m', $CachedInputUsdPerM.ToString([cultureinfo]::InvariantCulture),
          '--output-usd-per-m', $OutputUsdPerM.ToString([cultureinfo]::InvariantCulture))
if ($AllowHttpUpstream) { $argv += '--allow-http-upstream' }
& python @argv
if ($LASTEXITCODE -ne 0) { throw "GPT-6.1 Sol adapter stopped with exit $LASTEXITCODE. Inspect output and preserve existing files." }
