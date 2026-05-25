param(
    [string]$Python = "S:\pycharm\Anaconda\envs\torch_env\python.exe",
    [string]$OutputBase = "results\paper_experiments_perf_compare",
    [string]$BeforeRoot = "results\paper_experiments_perf_compare\all6_medium_serial",
    [string]$AfterRoot = "results\paper_experiments_perf_compare\all6_medium_optimized",
    [string]$CompareRoot = "results\paper_experiments_perf_compare\all6_medium_compare"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Output "[Driver] $Name started $(Get-Date -Format o)"
    & $Python @Arguments
    $exitCode = $LASTEXITCODE
    Write-Output "[Driver] $Name exit=$exitCode $(Get-Date -Format o)"
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

New-Item -ItemType Directory -Force -Path $OutputBase | Out-Null

Invoke-Step -Name "serial/control arm" -Arguments @(
    "-m", "src.experiments.paper_figures.run_paper_figures",
    "--figs", "all",
    "--scope", "main",
    "--seeds", "1000",
    "--device", "cuda",
    "--benchmark-profile", "medium",
    "--jobs", "1",
    "--seed-jobs", "1",
    "--parallel-axis", "auto",
    "--cpu-workers", "1",
    "--cpu-threads-per-worker", "1",
    "--max-build-workers", "1",
    "--fig1-delay-decode-backend", "sklearn_linear_svc",
    "--check-only-build",
    "--output-root", $BeforeRoot,
    "--force",
    "--no-progress"
)

Invoke-Step -Name "optimized arm" -Arguments @(
    "-m", "src.experiments.paper_figures.run_paper_figures",
    "--figs", "all",
    "--scope", "main",
    "--seeds", "1000",
    "--device", "cuda",
    "--benchmark-profile", "medium",
    "--jobs", "1",
    "--seed-jobs", "1",
    "--parallel-axis", "subtask",
    "--cpu-workers", "4",
    "--cpu-threads-per-worker", "1",
    "--max-build-workers", "2",
    "--enable-gpu-batching",
    "--enable-gpu-metrics",
    "--fig1-delay-decode-backend", "torch_linear_probe",
    "--experiment-batch-size", "16",
    "--fig1-dms-batch-size", "16",
    "--fig4-l3-region-batch-size", "16",
    "--check-only-build",
    "--output-root", $AfterRoot,
    "--force",
    "--no-progress"
)

Invoke-Step -Name "runtime summary" -Arguments @(
    "scripts\summarize_paper_figures_runtime_compare.py",
    "--before-root", $BeforeRoot,
    "--after-root", $AfterRoot,
    "--before-label", "serial",
    "--after-label", "optimized",
    "--output-dir", $CompareRoot
)

Invoke-Step -Name "validation summary" -Arguments @(
    "scripts\validate_paper_figures_runtime_compare.py",
    "--before-root", $BeforeRoot,
    "--after-root", $AfterRoot,
    "--output-dir", $CompareRoot,
    "--atol", "1e-5",
    "--rtol", "1e-5",
    "--python", $Python
)

Write-Output "[Driver] complete $(Get-Date -Format o)"
