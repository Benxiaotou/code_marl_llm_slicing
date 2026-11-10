# P2-B: produce an independent teacher-experience candidate pool.
# Run from path/to/code_marl_llm_slicing; use conda env "marl".
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('generate','probe','collect','select','audit')]
    [string]$Stage,
    [int]$Seed=2602,
    [int]$Count=1600
)
$ErrorActionPreference = 'Stop'
$datasetId = "p2b_train_seed$Seed"
$datasetManifest = ".\p1b_data\$datasetId.json"
$probeDir = ".\p2b_runs\${datasetId}_probe3"
$collectionDir = ".\p2b_runs\${datasetId}_all$Count"
$libraryDir = ".\p2b_libraries\${datasetId}_candidate_v1"
$devManifest = '.\p1b_data\p1c_val_seed4601.json'

foreach ($name in @('p1b_generate.py','p1b_collect.py','p1b_select.py','p1b_audit.py')) {
    if (-not (Test-Path -LiteralPath $name)) { throw "Missing expected source file: $name. Run in original project directory." }
}
function Invoke-CheckedPython([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python exit code $LASTEXITCODE" }
}
function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file not found: $Path" }
}
function Refuse-Existing([string]$Path) {
    if (Test-Path -LiteralPath $Path) { throw "Output already exists; refusing overwrite: $Path" }
}

switch ($Stage) {
    'generate' {
        Refuse-Existing $datasetManifest
        Invoke-CheckedPython -Arguments @('p1b_generate.py','--name',$datasetId,'--split','train','--seed',"$Seed",'--count',"$Count")
        Require-File $datasetManifest
    }
    'probe' {
        Require-File $datasetManifest
        Refuse-Existing $probeDir
        Invoke-CheckedPython -Arguments @('p1b_collect.py','--dataset','custom','--dataset-manifest',$datasetManifest,'--teacher','live','--start','0','--count','3','--seed','1','--output-dir',$probeDir)
    }
    'collect' {
        Require-File $datasetManifest
        Refuse-Existing $collectionDir
        Invoke-CheckedPython -Arguments @('p1b_collect.py','--dataset','custom','--dataset-manifest',$datasetManifest,'--teacher','live','--start','0','--count',"$Count",'--seed','1','--output-dir',$collectionDir)
    }
    'select' {
        if (-not (Test-Path -LiteralPath $collectionDir -PathType Container)) { throw "Collection is missing: $collectionDir" }
        Refuse-Existing $libraryDir
        Invoke-CheckedPython -Arguments @('p1b_select.py','--collection',$collectionDir,'--output-dir',$libraryDir,'--max-keep',"$Count",'--cost-quantile','0.95')
    }
    'audit' {
        if (-not (Test-Path -LiteralPath $libraryDir -PathType Container)) { throw "Candidate library is missing: $libraryDir" }
        Require-File $devManifest
        Invoke-CheckedPython -Arguments @('p1b_audit.py','--library',$libraryDir,'--holdout-manifest',$devManifest)
    }
}
Write-Host "P2-B Stage $Stage succeeded. No existing P2-A baseline was modified."
