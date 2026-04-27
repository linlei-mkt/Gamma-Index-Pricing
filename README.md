# Gamma-Index-Pricing
# Replication Package for "Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand"

This repository contains all code and instructions to reproduce every empirical table and figure in the paper. Every script is self-contained, reads raw data from a user-specified folder, and writes outputs alongside the script.

---

## 1\. Data

You need the three CSVs from the **2020 MSOM Data-Driven Research Challenge** (Shen, Tang, Wu, Yuan, Zhou, *MSOM* 2020), available at [https://connect.informs.org/msom/events/datadriven-call](https://connect.informs.org/msom/events/datadriven-call):

| File | Size | Used? |
| :---- | :---- | :---- |
| `JD_order_data.csv` | \~57 MB, 549,990 transactions | ✓ all empirical scripts |
| `JD_sku_data.csv` | \~1 MB, 31,868 SKUs | ✓ for `type` and `brand_ID` |
| `JD_user_data.csv` | \~18 MB, user demographics | ✗ not used; demographic-conditional extension noted as future work in §15 |

Put all three in one folder and set:

```shell
export JD_DATA_DIR=/absolute/path/to/JD_csvs
```

If the variable is unset, scripts fall back to a hard-coded path you can edit at the top of each file.

---

## 2\. Setup

Python 3.9+ recommended.

```shell
# Required for all scripts
pip install numpy pandas scipy statsmodels matplotlib

# Required for HB MCMC scripts (jd_hierarchical_bayes.py and downstream)
pip install pymc arviz nutpie

# Required for the optional PyBLP comparison
pip install pyblp
```

Tested on Python 3.10.12 with: pandas 2.1.4, numpy 1.26.3, scipy 1.11.4, statsmodels 0.14.1, matplotlib 3.8.2, pymc 5.10, arviz 0.16, nutpie 0.9, pyblp 1.1.0.

---

## 3\. Paper result → script mapping

Every numbered Table and Figure in the paper, with the script that produces it and the exact output filename. Boldfaced entries are **headline results**; others are robustness checks.

### Tables

| \# | Section | Content | Script | Output |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §2.2 | Sign conventions for $\\gamma^\\star$ | (analytical, no script) | — |
| 2 | §7.1 | Seven pricing-rule benchmarks | (analytical, no script) | — |
| 3 | §8.2 | HKMR (1995) calibrated illustration | `empirical_gamma.py` | console output |
| **4** | **§9.3** | **JD four-method comparison ($\\gamma^\\star=0$)** | `jd_hierarchical_bayes.py` | `jd_hb_pricing_comparison.csv` |
| **5** | **§10.2** | **JD GMV-constrained pricing (15% floor)** | `jd_gmv_constrained.py` | `jd_gmv_pricing_comparison.csv` |
| **6** | **§10.3** | **GMV-floor sensitivity ($\\phi \\in {1.05,\\ldots,1.25}$)** | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_summary.csv` |
| 7 | §11.1 | Hausman-IV cross-DC estimates | `jd_hausman_iv.py` | `jd_iv_estimates.csv` |
| 8 | §11.2 | Wall-clock scalability ($n \\in {500,\\ldots,50{,}000}$) | `scalability_demo.py` | `scalability_results.csv` |
| 9 | §11.3 | $M\_{\\mathrm{mult}} \\times c/\\bar p$ sensitivity grid | `jd_hb_sensitivity.py` | `jd_hb_sensitivity_results.csv` |
| **10** | **§11.7** | **Top-$N$ catalog-truncation robustness** | `jd_topn_sensitivity.py` | `jd_topn_summary.csv` |

### Figures

| \# | Section | Content | Script | Output |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §9.2 | $\\bar e$ distribution across 31 daily markets | `jd_experiment.py` | `jd_ebar_distribution.png` |
| 2 (left) | §9.3 | Profit gap vs $\\bar e$ on JD data | `jd_hierarchical_bayes.py` | `jd_hb_profit_gap_vs_ebar.png` |
| 2 (right) | §9.3 | Per-iteration convergence on a representative day | `jd_experiment.py` | `jd_convergence.png` |
| 3 | §10.2 | GMV-constrained: profit vs revenue \+ tuned $\\gamma^\\star$ distribution | `jd_gmv_constrained.py` | `jd_gmv_profit_vs_revenue.png`, `jd_gmv_gamma_star_distribution.png` |
| 4 | §10.3 | GMV-floor sensitivity (gap vs $\\phi$, $\\gamma^\\star$ vs $\\phi$) | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_sensitivity.png` |
| 5 | §11.2 | Scalability log-log curves | `scalability_demo.py` | `scalability_plot.png` |
| 6 | §11.5 | Empirical operator norm $\\rho\_t$ vs $\\bar e\_t$ | `jd_cbar_diagnostic.py` | `jd_cbar_scatter.png` |
| 7 | §11.7 | Top-$N$ sensitivity (median $\\bar e$, gaps vs $N$) | `jd_topn_sensitivity.py` | `jd_topn_sensitivity.png` |

---

## 4\. Minimum reproduction sequence

Run these in order. Total runtime: \~2-3 hours on a laptop, dominated by the HB MCMC fits in steps 2 and 7\. On Colab with `nutpie`, the MCMC steps drop to \~10–15 minutes each.

### Step 1 — JD bucket-level baseline (Figures 1 left, 2 right; supplementary)

```shell
python3 jd_experiment.py
```

Runtime: \~30 sec.

Outputs:

- `jd_pricing_comparison.csv` — per-day bucket-level results (sanity check)  
- `jd_elasticities.csv` — 10 price-decile $\\hat\\beta\_b$ estimates  
- `jd_ebar_distribution.png` — **Figure 1**  
- `jd_profit_gap_vs_ebar.png`, `jd_convergence.png` — supplementary visuals  
- `jd_wallclock_comparison.png` — supplementary

### Step 2 — Hierarchical-Bayes MCI demand (Table 4, Figure 2 left; main result)

```shell
python3 jd_hierarchical_bayes.py
```

Runtime: \~5–15 min with `nutpie`, \~30–60 min without.

Outputs:

- `jd_hb_posterior_summary.csv` — posterior mean $\\hat\\beta\_i$ and $\\sigma$ per SKU **(required by every downstream script)**  
- `jd_hb_bucket_means.csv` — posterior $\\hat\\mu\_b$ and $\\hat\\tau$  
- `jd_hb_pricing_comparison.csv` — **Table 4** numbers  
- `jd_hb_profit_gap_vs_ebar.png` — **Figure 2 (left)**  
- `jd_hb_trace_plots.png`, `jd_hb_shrinkage.png` — MCMC diagnostics

### Step 3 — GMV-constrained pricing (Table 5, Figure 3\)

```shell
python3 jd_gmv_constrained.py
```

**Requires `jd_hb_posterior_summary.csv` from Step 2\.** Runtime: \~1 min.

Outputs:

- `jd_gmv_pricing_comparison.csv` — **Table 5**  
- `jd_gmv_profit_vs_revenue.png` — **Figure 3 (left)**  
- `jd_gmv_gamma_star_distribution.png` — **Figure 3 (right)**

### Step 4 — GMV-floor sensitivity (Table 6, Figure 4\)

```shell
python3 jd_gmv_floor_sensitivity.py
```

**Requires Step 2 and the `jd_gmv_constrained.py` module (imports helpers from it).** Runtime: \~5–10 min.

Sweeps the floor multiplier $\\phi \\in {1.05, 1.10, 1.15, 1.20, 1.25}$ on the same 31 daily markets.

Outputs:

- `jd_gmv_floor_sensitivity.csv` — per-$(\\phi, \\mathrm{day})$ rows (5 × 31 \= 155\)  
- `jd_gmv_floor_summary.csv` — **Table 6**  
- `jd_gmv_floor_sensitivity.png` — **Figure 4**

### Step 5 — Hausman IV (Table 7\)

```shell
python3 jd_hausman_iv.py
```

Runtime: \~30 sec. Reads JD orders directly (no HB dependency).

Outputs:

- `jd_iv_estimates.csv` — **Table 7** (OLS / 2SLS coefficients, F-stat, n\_cells)  
- `jd_iv_first_stage.csv` — first-stage residual diagnostics  
- `jd_iv_comparison.png` — supplementary visual

### Step 6 — Scalability demo (Table 8, Figure 5\)

```shell
python3 scalability_demo.py
```

Runtime: \~3 min. Synthetic MCI, no JD data needed.

Outputs:

- `scalability_results.csv` — **Table 8** (wall-clock per $n$ per method)  
- `scalability_plot.png` — **Figure 5**

### Step 7 — $M \\times c$ sensitivity (Table 9\)

```shell
python3 jd_hb_sensitivity.py
```

**Requires Step 2\.** Runtime: \~10 min with `nutpie` (5 HB re-fits at different $M\_{\\mathrm{mult}}$, then pricing).

Outputs:

- `jd_hb_sensitivity_results.csv` — **Table 9**  
- `jd_hb_sensitivity_heatmap_gamma.png`, `jd_hb_sensitivity_heatmap_ebar.png` — heatmap visuals  
- `jd_hb_sensitivity_heatmap_uniform.png`, `jd_hb_sensitivity_heatmap_speedup.png` — supplementary

### Step 8 — Mixed-logit Monte Carlo (§11.4 narrative)

```shell
python3 mixed_logit_robustness.py
```

Runtime: \~30 sec. Synthetic, no JD data needed.

Outputs:

- `mixed_logit_results.csv` — 79 synthetic markets  
- `mixed_logit_profit_gap.png` — $\\bar e^2$ fit diagnostic  
- `mixed_logit_convergence.png`, `mixed_logit_compare_mci.png` — supplementary

### Step 9 — Empirical contraction diagnostic (Figure 6\)

```shell
python3 jd_cbar_diagnostic.py
```

**Requires Step 2\.** Runtime: \~30 sec.

Outputs:

- `jd_cbar_results.csv` — per-market $(\\bar e, C^{\\mathrm{theory}}, \\rho)$  
- `jd_cbar_scatter.png` — **Figure 6**  
- `jd_cbar_histogram.png` — supplementary

### Step 10 — Top-$N$ robustness (Table 10, Figure 7\)

```shell
python3 jd_topn_sensitivity.py
```

**Imports from `jd_hierarchical_bayes.py`.** Runtime: \~40–60 min (re-fits HB MCMC at $N \\in {200, 500, 1000, 2000}$). Per-$N$ HB posteriors are cached, so re-runs of this script skip already-done $N$.

Outputs:

- `jd_topn_sensitivity.csv` — per-$(N, \\mathrm{day})$ rows (4 × 31 \= 124\)  
- `jd_topn_summary.csv` — **Table 10**  
- `jd_topn_sensitivity.png` — **Figure 7**  
- `jd_hb_posterior_summary_N{200,500,1000,2000}.csv` — per-$N$ HB posterior caches

### Step 11 — HKMR calibrated illustration (Table 3\)

```shell
python3 empirical_gamma.py
```

Runtime: \~5 sec. Synthetic, no JD data needed.

Outputs: console output only — read off the **Table 3** numbers from the printed comparison block.

---

## 5\. Optional / supplementary scripts (not required for paper)

| Script | Purpose |
| :---- | :---- |
| `jd_pyblp.py` | PyBLP mixed-logit on JD with Hausman \+ rival-price IVs and Gandhi–Houde differentiation IVs. Mentioned in §11.4 framing; SKU attributes 1–2 enter only through differentiation IVs (not $X\_1$) because SKU fixed effects absorb time-invariant SKU characteristics. Runtime: 10–60 min. |
| `jd_brand_experiment.py` | Brand-level elasticity alternative (top-15 brands \+ "other"); robustness only. |
| `jd_sensitivity.py` | Same $M \\times c$ grid as `jd_hb_sensitivity.py` but with bucket-level $\\beta$ (no HB). Faster (\~30 sec) but less refined; use if PyMC unavailable. |
| `jd_hb_validation.py` | Posterior-predictive check \+ holdout fit \+ $\\gamma$-gap posterior propagation. **Runs but results are catalogued as scope limitations in §11.6** (see paper). Runtime: \~30–60 min on Colab. |
| `mixed_logit_gmv_constrained.py` | Synthetic mixed-logit GMV-floor experiment. In replication package per §11 footnote, not in main text. |
| `gamma_simulation.py` | Single-product $\\gamma$-index promotion-allocation illustration (companion working paper). Not used in the main paper. |
| `gamma_worked_example.py` | Tiny 3-product worked example of the generalized Lerner rule. Pedagogical only. |
| `identification_mc.py` | BLP-bias identification Monte Carlo. **Used only by the companion paper on $\\gamma$-corrected supply-side inversion**; the main paper merely sketches the bias formula in §12. |
| `jd_mixed_logit_real.py` | **Deprecated** custom-PyMC mixed-logit on JD; the aggregate likelihood has a known share-aggregation bug. Use `jd_pyblp.py` for the structural mixed-logit claim. |

---

## 6\. Google Colab workflow

The Drive FUSE mount sometimes hides files from Python's import machinery even when `os.path.exists` says they're there. Workaround: copy the .py files to local Colab storage `/content/` once at the start.

```py
# Cell 1: mount Drive and copy scripts to local /content
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil
DRIVE = '/content/drive/MyDrive/JD_gamma'   # ← edit to your folder
LOCAL = '/content/JD_gamma_local'
os.makedirs(LOCAL, exist_ok=True)
for f in os.listdir(DRIVE):
    if f.endswith('.py') or f == 'jd_hb_posterior_summary.csv':
        shutil.copy(os.path.join(DRIVE, f), os.path.join(LOCAL, f))

if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)

os.environ['JD_DATA_DIR'] = DRIVE   # JD CSVs stay on Drive
```

```py
# Cell 2: install dependencies
!pip install -q pandas numpy scipy statsmodels matplotlib
!pip install -q pymc arviz nutpie       # for HB MCMC scripts
!pip install -q pyblp                    # optional, for jd_pyblp.py
```

```py
# Cell 3: run scripts
%run /content/JD_gamma_local/jd_hierarchical_bayes.py
%run /content/JD_gamma_local/jd_gmv_constrained.py
%run /content/JD_gamma_local/jd_gmv_floor_sensitivity.py
%run /content/JD_gamma_local/jd_topn_sensitivity.py
# ... etc.
```

```py
# Cell 4: copy outputs (csv + png) back to Drive
for f in os.listdir(LOCAL):
    if f.endswith('.csv') or f.endswith('.png'):
        shutil.copy(os.path.join(LOCAL, f), os.path.join(DRIVE, f))
```

---

## 7\. Reproducibility notes

All scripts use `SEED = 2026` for Monte Carlo draws. Bayesian scripts set `random_seed=SEED` in `pm.sample()`. Bit-for-bit reproducibility holds on the same Python / numpy / scipy / pymc versions. Across major version bumps (e.g., numpy 2.x vs 1.x) results may drift in the 3rd–4th decimal due to internal RNG reorganization; all qualitative claims in the paper are robust to this drift.

For PyMC scripts, exact numerical reproducibility also requires the same `pytensor` linear-algebra backend (`numpy`\-backed vs `jax`\-backed sampling). Default Colab installs use `numpy`.

---

## 8\. Common failure modes

In decreasing order of probability:

1. **`FileNotFoundError` on JD CSVs** — `$JD_DATA_DIR` is wrong or the files are renamed. Scripts expect exactly `JD_order_data.csv`, `JD_sku_data.csv`, `JD_user_data.csv` (the third is required to be present in the bundle but not actually read).  
     
2. **`ModuleNotFoundError: jd_hierarchical_bayes` (or similar) on Colab** — Drive FUSE issue. Use the `/content/JD_gamma_local` workaround in §6.  
     
3. **`ModuleNotFoundError: pymc`** — `pip install pymc arviz nutpie`.  
     
4. **MCMC very slow** — install `nutpie` for \~3–5× speedup. If still slow, drop `DRAWS` and `TUNE` at the top of the offending script from 1000 to 500\.  
     
5. **MCMC divergences** — raise `TARGET_ACCEPT` from 0.85 to 0.95.  
     
6. **Out-of-memory at $N=500$ MCMC** — drop `N_TOP_SKU` from 500 to 300 at the top of `jd_hierarchical_bayes.py`. (For the top-$N$ sweep at $N=2000$, OOM is more likely; lower `N_VALUES` in `jd_topn_sensitivity.py`.)  
     
7. **`jd_gmv_floor_sensitivity.py` ImportError on `jd_gmv_constrained`** — the floor-sensitivity script imports helpers from `jd_gmv_constrained.py`; they must be in the same folder, and that folder must be on `sys.path` (handled by the Colab cell above).  
     
8. **`jd_topn_sensitivity.py` ImportError on `jd_hierarchical_bayes`** — same fix.

---

## 9\. License and citation

Code: MIT License. The JD MSOM 2020 dataset is distributed under the MSOM challenge terms; see the challenge website for data-use conditions.

Cite the paper as:

