# Code for "(Mis)Understanding Benign Overfitting in Equity Return Prediction"

This repository contains the MATLAB code required to reproduce the empirical results in the paper.

## Abstract

Highly overparameterized models often predict well despite interpolating training data in complex domains, challenging the classical bias–variance tradeoff. We investigate whether this "benign overfitting" phenomenon extends to equity return prediction. Consistent with recent statistical theory, we document two key phenomena: first, a double descent pattern in the ridgeless model's prediction risk; and second, that while the optimal ridge model consistently outperforms its ridgeless counterpart, this performance gap becomes negligible at large parameter-to-observation ratios. Ultimately, however, both models fail to outperform a simple historical average. This empirical evidence aligns with our asymptotic results under the null hypothesis of zero slope coefficients, suggesting that standard equity predictors lack true forecasting power — even within highly flexible, nonlinear machine learning architectures. These findings reconcile modern and classical machine learning in asset pricing: in the absence of a true signal, they asymptotically collapse to the historical average benchmark.

## Repository structure

```
.
├── raw/          # Raw data estimate pipeline: predictions, double descent, tables
└── bootstrap/     # Block-bootstrap null-distribution pipeline (CV model)
```

## 1. `raw/` — Predictions, double descent, and main tables

Point-estimate pipeline: generates monthly return forecasts for the RFF model (fixed-z grid and CV-selected z), the raw-feature OLS/ridge benchmark, and the resulting summary tables and figures.

### Data

- **`GYdata.mat`** — Predictor variables (`X`), equity returns (`Y`), from the Goyal & Welch (2008) dataset.

### Scripts

- **`run_rff.m`** — Managed portfolio return prediction on RFF features, with fixed penality `z`.
- **`run_rff_cv.m`** — Managed portfolio return prediction on RFF features, with cross-validation penality `z`.
- **`run_simple.m`** — Managed portfolio return prediction on 14 Goyal & Welch (2008) features, with fixed penality `z`.

### Tables and figures

- **`table_fix.m`** — Builds the main results table for the fixed-z RFF and Simple models: Sharpe ratio, OOS R², correlation with the historical-average benchmark, and three CAPM alphas with Newey-West t-statistics (Alpha1: vs. Market; Alpha2: vs. Market + HistAvg), reported across the three demean settings, windows, and sub-periods.
- **`table_cv.m`** — Analogous consolidated table for the CV-selected-z RFF model.
- **`ave_z_cv.m`** — Summary statistics of CV-selected regularization values and effective degrees of freedom across all RFF generations.
- **`double_decent.m`** — Plots the double descent curve: out-of-sample test risk against `gamma = P/n` (log scale).
- **`kernel.m`** — Implements six market-timing strategies (History, TsMom, VolMom, DW, PV, EWPV) and computes Sharpe ratio, correlation with the historical-average strategy, and alpha t-statistics.
- **`plot_alpha2.m`** — Bar chart of Alpha2 t-statistics for the RFF model across the z-grid and the CV estimate.

## 2. `bootstrap/` — Null-distribution bootstrap (CV model)

Block-bootstrap pipeline used to assess the statistical significance of the CV-selected-z RFF model's Sharpe ratio, OOS R², and alpha estimates against a resampled null distribution.

### Scripts

- **`rff_cv_boot.m`** — Bootstrap version of `rff_cv.m`: produces CV-selected-z RFF forecasts from resampled feature blocks.
- **`run_rff_cv_boot_par.m`** — Runs `rff_cv_boot3.m` across training windows, bootstrap replicates, and simulations in parallel.
- **`table_boot_cv.m`** — Summarizes the null sampling distribution of Sharpe ratio, OOS R², and alpha t-statistics per window.

## Requirements

- MATLAB with the Parallel Computing Toolbox (`parfor`, `parpool`) and Statistics and Machine Learning Toolbox.
- Recommended: a multi-core machine or HPC cluster — the fixed-z, CV, and bootstrap driver scripts are parallelized across simulation draws and (for the bootstrap) across 100+ replicates per window.

## Key references

- Goyal A, Welch I (2008) A comprehensive look at the empirical performance of equity premium prediction. The Review of Financial Studies 21(4):1455–1508
- Goyal A, Welch I, Zafirov A (2024) A comprehensive 2022 look at the empirical performance of equity premium prediction. The Review of Financial Studies 37(11):3490–3557.
- Hastie T, Montanari A, Rosset S, Tibshirani RJ (2022) Surprises in high-dimensional ridgeless least squares interpolation. The Annals of Statistics 50(2):949–986.
- Kelly B, Malamud S, Zhou K (2024) The virtue of complexity in return prediction. The Journal of Finance 79(1):459–503.
