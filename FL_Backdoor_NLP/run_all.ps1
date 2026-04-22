# run_all.ps1
# Runs all FL experiments in sequence (Phases 1-6)
# Usage: cd FL_Backdoor_NLP; .\run_all.ps1
# Defense experiments (Phase 7) are excluded — need code change first.

$ErrorActionPreference = "Continue"

$configs = @(
    # ── Phase 1: 10-Client Baseline (remaining datasets) ──────────────
    "configs/fedavg_mnist.yaml",
    "configs/fedavg_cifar100.yaml",
    "configs/fedavg_shakespeare.yaml",
    "configs/fedavg_sent140.yaml",

    # ── Phase 2: Alpha Sweep — MNIST ──────────────────────────────────
    "configs/fedavg_mnist.yaml --alpha 0.01",
    "configs/fedavg_mnist.yaml --alpha 0.1",
    "configs/fedavg_mnist.yaml --alpha 1.0",
    "configs/fedavg_mnist.yaml --alpha iid",

    # ── Phase 3: Alpha Sweep — CIFAR-10 ───────────────────────────────
    "configs/fedavg_cifar10.yaml --alpha 0.01",
    "configs/fedavg_cifar10.yaml --alpha 0.1",
    "configs/fedavg_cifar10.yaml --alpha 1.0",
    "configs/fedavg_cifar10.yaml --alpha iid",

    # ── Phase 4: Backdoor Attack — 10 Clients ─────────────────────────
    "configs/backdoor_mnist.yaml",
    "configs/backdoor_cifar10.yaml",

    # ── Phase 5: 50-Client Experiments ────────────────────────────────
    "configs/fedavg_mnist_50c.yaml",
    "configs/fedavg_fmnist_50c.yaml",
    "configs/fedavg_cifar10_50c.yaml",
    "configs/fedavg_cifar100_50c.yaml",
    "configs/fedavg_shakespeare_50c.yaml",
    "configs/fedavg_sent140_50c.yaml",
    "configs/backdoor_mnist_50c.yaml",
    "configs/backdoor_cifar10_50c.yaml",

    # ── Phase 6: 100-Client Experiments ───────────────────────────────
    "configs/fedavg_mnist_100c.yaml",
    "configs/fedavg_fmnist_100c.yaml",
    "configs/fedavg_cifar10_100c.yaml",
    "configs/fedavg_cifar100_100c.yaml",
    "configs/fedavg_shakespeare_100c.yaml",
    "configs/fedavg_sent140_100c.yaml",
    "configs/backdoor_mnist_100c.yaml",
    "configs/backdoor_cifar10_100c.yaml"
)

$total = $configs.Count
$completed = 0
$failed = @()
$startTime = Get-Date

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  FL Experiment Runner — $total experiments queued" -ForegroundColor Yellow
Write-Host "  Started: $startTime" -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Yellow

foreach ($cfg in $configs) {
    $completed++
    $elapsed = (Get-Date) - $startTime
    Write-Host "[$completed/$total] $(Get-Date -Format 'HH:mm:ss') | Elapsed: $([math]::Round($elapsed.TotalMinutes,1)) min" -ForegroundColor Cyan
    Write-Host "  -> python run_experiment.py --config $cfg`n" -ForegroundColor White

    $cmd = "python run_experiment.py --config $cfg"
    Invoke-Expression $cmd
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "  [FAILED] Exit code: $exitCode" -ForegroundColor Red
        $failed += $cfg
    } else {
        Write-Host "  [OK]" -ForegroundColor Green
    }
    Write-Host ""
}

# ── Generate all plots ─────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "  Generating plots and results table..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
python results/plot_results.py --results_dir results/

# ── Summary ────────────────────────────────────────────────────────────
$totalTime = (Get-Date) - $startTime
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  DONE in $([math]::Round($totalTime.TotalMinutes,1)) minutes" -ForegroundColor Green
Write-Host "  Successful: $($total - $failed.Count) / $total" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "  Failed:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
}
Write-Host "  Results -> results/" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
