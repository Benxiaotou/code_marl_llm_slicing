param(
    [ValidateSet('plan','all','prepare','execute','aggregate')]
    [string]$Mode = 'plan'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "P2-C Qwen8 serial runner - mode: $Mode" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
& python -u .\p2c_qwen8_queue.py --mode $Mode
if ($LASTEXITCODE -ne 0) {
    Write-Error "P2-C queue stopped with exit code $LASTEXITCODE. Inspect logs and preserve partial results."
    exit $LASTEXITCODE
}
Write-Host "P2-C mode '$Mode' finished successfully." -ForegroundColor Green
