# Replication Guide

Step-by-step protocol for replicating every empirical Table (3–10) and Figure (1–9) in *Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand*.

The replication needs only Python 3.9+ and the JD MSOM 2020 dataset (included in this repository under `JD_MSOM/`). Total runtime: \~15 minutes for the headline numbers, \~3 hours for every Table and Figure.

---

## Quick start

```shell
git clone <repo-url>
cd gamma-equalization-replication
pip install -r requirements.txt

cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM

python3 jd_hierarchical_bayes.py     # Table 3, Figure 2 (left)
python3 jd_gmv_constrained.py         # Table 4, Figure 3
```

If those two scripts complete and the median profit gaps in their output CSVs match the headline numbers in §4 below, the empirical core of the paper has been reproduced.

---

## Repository layout

```
gamma-equalization-replication/
├── README.md
├── REPLICATION.md                  this file
├── requirements.txt
├── MainCodes/                      all replication scripts (.py)
├── JD_MSOM/                        bundled MSOM 2020 CSVs
├── figures/                        pre-generated reference PNGs
└── reference_output_results/       pre-generated reference CSVs
```

All scripts live in `MainCodes/`. Every script writes its outputs (CSV / PNG) into `MainCodes/` itself. Compare against the matching files in `figures/` and `reference_output_results/` to verify reproducibility.

---

## 1\. Environment

Python 3.9 or later. Tested on Python 3.10.12 with:

```
pandas      2.1.4
numpy       1.26.3
scipy       1.11.4
statsmodels 0.14.1
matplotlib  3.8.2
pymc        5.10
arviz       0.16
nutpie      0.9
pyblp       1.1.0   (only needed for the optional jd_pyblp.py)
```

Install via `pip install -r requirements.txt`, or manually:

```shell
pip install numpy pandas scipy statsmodels matplotlib
pip install pymc arviz nutpie       # required for HB MCMC scripts
pip install pyblp                    # optional
```

---

## 2\. Data

The MSOM 2020 Data-Driven Research Challenge (Shen, Tang, Wu, Yuan, Zhou, *MSOM* 2020\) is bundled in `JD_MSOM/`:

| File | Size | Used by paper? |
| :---- | :---- | :---- |
| `JD_order_data.csv` | \~57 MB, 549,990 transactions | Yes — every empirical script |
| `JD_sku_data.csv` | \~1 MB, 31,868 SKUs | Yes — for `type` and `brand_ID` |
| `JD_user_data.csv` | \~18 MB, user demographics | Not used; demographic-conditional extension is noted as future work in §13 |

Set the data path:

```shell
cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM
```

If `JD_DATA_DIR` is unset, scripts fall back to a hard-coded default at the top of each file that you can edit.

---

## 3\. Paper structure → script mapping

The paper proceeds: theory (§1–§7) → JD calibrated illustration (§8) → constrained pricing (§9) → robustness summary (§10) → managerial implications (§11) → discussion (§12) → conclusion (§13). Appendices A–H contain proofs (Theorem 1, Propositions 2, 14, etc., Theorem 3, Lipschitz bound, Morrow–Skerlos equivalence). Appendix **I** contains the seven-test robustness package referenced in §10. Appendix **J** contains the calibrated Monte Carlo and HKMR (1995) illustration. Appendix **K** discusses BLP supply-side identification implications under revenue-constrained pricing.

Tables 1, 2 are analytical (no script). The empirical Tables 3–10 and Figures 1–9 are listed below with the producing script.

### Tables

| \# | Location | Content | Script | Primary output |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §2.2 | Sign conventions for $\\gamma^\\star$ | (analytical, no script) | — |
| 2 | §7.1 | Multi-product pricing rules ranked by information intensity | (analytical, no script) | — |
| **3** | **§8.3** | **JD four-method comparison ($\\gamma^\\star=0$)** | `jd_hierarchical_bayes.py` | `jd_hb_pricing_comparison.csv` |
| **4** | **§9.2** | **JD GMV-constrained pricing (15% floor)** | `jd_gmv_constrained.py` | `jd_gmv_pricing_comparison.csv` |
| **5** | **§9.3** | **GMV-floor sensitivity ($\\phi \\in {1.05,\\ldots,1.25}$)** | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_summary.csv` |
| 6 | App I.1 | Pooled price elasticity, OLS vs Hausman-IV | `jd_hausman_iv.py` | `jd_iv_estimates.csv` |
| 7 | App I.2 | Wall-clock scalability ($n \\in {500,\\ldots,50{,}000}$) | `scalability_demo.py` | `scalability_results.csv` |
| 8 | App I.3 | $M\_{\\mathrm{mult}} \\times c/\\bar p$ sensitivity grid | `jd_hb_sensitivity.py` | `jd_hb_sensitivity_results.csv` |
| **9** | **App I.7** | **Top-$N$ catalog-truncation robustness** | `jd_topn_sensitivity.py` | `jd_topn_summary.csv` |
| 10 | App J | HKMR (1995) calibrated illustration | `empirical_gamma.py` | console output |

### Figures

| \# | Location | Content | Script | Output filename |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §8.2 | $\\bar e$ distribution across 31 daily markets | `jd_experiment.py` | `jd_ebar_distribution.png` |
| 2 (left) | §8.3 | Profit gap vs $\\bar e$ on JD data | `jd_hierarchical_bayes.py` | `jd_hb_profit_gap_vs_ebar.png` |
| 2 (right) | §8.3 | Per-iteration convergence on a median-$\\bar e$ day | `jd_experiment.py` | `jd_convergence.png` |
| 3 (left) | §9.2 | Profit vs revenue with floor | `jd_gmv_constrained.py` | `jd_gmv_profit_vs_revenue.png` |
| 3 (right) | §9.2 | Distribution of tuned $\\hat\\gamma^\\star$ | `jd_gmv_constrained.py` | `jd_gmv_gamma_star_distribution.png` |
| 4 | §9.3 | Floor sensitivity (gap and tuned $\\hat\\gamma^\\star$ vs $\\phi$) | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_sensitivity.png` |
| 5 | App I.2 | Scalability log-log curves at $n \\in {500,\\ldots,50{,}000}$ | `scalability_demo.py` | `scalability_plot.png` |
| 6 | App I.5 | Empirical operator norm $\\rho\_t$ vs share-only $\\bar e\_t$ | `jd_cbar_diagnostic.py` | `jd_cbar_scatter.png` |
| **7** | **App I.6** | **Posterior predictive residual histogram** | `jd_hb_validation.py` | `jd_ppc_histogram.png` |
| **8** | **App I.6** | **Per-day posterior $\\gamma$-gap with 90% credible bands** | `jd_hb_validation.py` | `jd_uncertainty_interval.png` |
| **9** | **App I.7** | **Top-$N$ sensitivity (median $\\bar e$ and median gap vs $N$)** | `jd_topn_sensitivity.py` | `jd_topn_sensitivity.png` |

---

## 4\. Headline reproduction (\~15 minutes)

For reviewers who want to verify the paper's main empirical claims quickly. Run from inside `MainCodes/`:

### Step 1 — Estimate hierarchical-Bayes MCI demand

```shell
python3 jd_hierarchical_bayes.py
```

**Runtime:** 5–15 minutes with `nutpie`, 30–60 minutes without.

**Outputs:**

- `jd_hb_posterior_summary.csv` — posterior $\\hat\\beta\_i$ per SKU (required by Steps 2 onward)  
- `jd_hb_pricing_comparison.csv` — **Table 3**  
- `jd_hb_profit_gap_vs_ebar.png` — **Figure 2 (left)**  
- `jd_hb_bucket_means.csv`, `jd_hb_trace_plots.png`, `jd_hb_shrinkage.png` — MCMC diagnostics

**Headline number to check:** Median value of column `gap_gamma` in `jd_hb_pricing_comparison.csv` should be approximately **1.11%**.

### Step 2 — GMV-constrained pricing

```shell
python3 jd_gmv_constrained.py
```

Requires `jd_hb_posterior_summary.csv` from Step 1\. **Runtime:** \~1 minute.

**Outputs:**

- `jd_gmv_pricing_comparison.csv` — **Table 4**  
- `jd_gmv_profit_vs_revenue.png` — **Figure 3 (left)**  
- `jd_gmv_gamma_star_distribution.png` — **Figure 3 (right)**

**Headline numbers to check:** In `jd_gmv_pricing_comparison.csv`, median `gap_gamma_pct` ≈ **5.57%** and median `gap_unif_pct` ≈ **15.93%**.

If both replicate within ±0.05 pp, the paper's empirical core is verified.

---

## 5\. Full reproduction (\~3 hours)

After Steps 1 and 2 above, run the remaining scripts to reproduce every Table and Figure. Most are independent and can be parallelized across terminals if desired.

### Independent scripts (any order, no HB dependency)

```shell
python3 empirical_gamma.py            # Table 10 (Appendix J), console output
python3 jd_experiment.py              # Figures 1, 2 (right)
python3 jd_hausman_iv.py              # Table 6 (Appendix I.1)
python3 scalability_demo.py           # Table 7, Figure 5 (Appendix I.2)
python3 mixed_logit_robustness.py     # Appendix I.4 narrative
```

| Script | Runtime | Notes |
| :---- | :---- | :---- |
| `empirical_gamma.py` | \~5 sec | Synthetic; reads HKMR (1995) elasticities. Output is **console only** — read the printed comparison block to recover Table 10 (Appendix J). |
| `jd_experiment.py` | \~30 sec | Bucket-level MCI baseline (no MCMC). Produces Figure 1 ($\\bar e$ distribution) and Figure 2 right (per-iteration convergence on median-$\\bar e$ day). |
| `jd_hausman_iv.py` | \~30 sec | Three-way fixed-effects (SKU, day, DC) IV using leave-one-out cross-DC mean log-price. Produces Table 6\. |
| `scalability_demo.py` | \~3 min | Synthetic; no JD data needed. Reproduces Table 7 wall-clock numbers and Figure 5\. |
| `mixed_logit_robustness.py` | \~30 sec | 79 synthetic mixed-logit markets; supports the Appendix I.4 claim that the $O(\\bar e^2)$ profit bound holds beyond MCI. No numbered table or figure; the headline `gap_γ ≈ 0.69·ē²` and `98.7%` of markets below `ē²` are reported in I.4 prose. |

### Scripts depending on `jd_hb_posterior_summary.csv` (Step 1 output)

```shell
python3 jd_gmv_floor_sensitivity.py   # Table 5, Figure 4
python3 jd_hb_sensitivity.py          # Table 8 (Appendix I.3)
python3 jd_cbar_diagnostic.py         # Figure 6 (Appendix I.5)
python3 jd_hb_validation.py           # Figures 7–8 (Appendix I.6)
```

| Script | Runtime | Notes |
| :---- | :---- | :---- |
| `jd_gmv_floor_sensitivity.py` | 5–10 min | Sweeps $\\phi \\in {1.05, 1.10, 1.15, 1.20, 1.25}$. Imports helpers from `jd_gmv_constrained.py`. |
| `jd_hb_sensitivity.py` | \~10 min with `nutpie` | Re-fits HB MCMC at five $M\_{\\mathrm{mult}}$ values, then runs the pricing comparison on the $5 \\times 5$ grid of $(M\_{\\mathrm{mult}}, c/\\bar p)$. |
| `jd_cbar_diagnostic.py` | \~30 sec | Computes empirical local operator norm $\\rho\_t$ on 31 daily markets via finite differences on the linearized $\\gamma$-update at observed prices. |
| `jd_hb_validation.py` | 30–50 min | PPC \+ temporal holdout \+ posterior propagation; re-fits HB on 25-day training window for the holdout exercise. Produces Figures 7 (PPC residuals) and 8 (posterior γ-gap intervals). |

### Heaviest single script

```shell
python3 jd_topn_sensitivity.py        # Table 9, Figure 9 (Appendix I.7)
```

**Runtime:** 40–60 minutes. Re-fits HB MCMC at $N \\in {200, 500, 1000, 2000}$. Per-$N$ HB posteriors are cached to `jd_hb_posterior_summary_N{n}.csv`, so a crashed run can be resumed by re-invoking the script.

**Note on the $0.84%$ vs $1.11%$ figures.** The $N \= 500$ median $\\gamma$-gap reported in Table 9 ($0.84%$) is consistent with, but not numerically identical to, the $1.11%$ headline of §8. Both use the same hierarchical-Bayes posterior at $N \= 500$ (identical seed, prior, sampler), but the top-$N$ sweep applies a common methodology across all four $N$ levels for internal consistency: a uniform $25$-SKU per-day filter (vs $50$ in §8, since $50$ cannot be applied at $N \= 200$), and signed profit gaps without zero-clipping (§8 clips negative gaps to zero on the small subset of days where Newton-BN is marginally exceeded by $\\gamma$ under numerical noise). Both conventions agree on the qualitative $O(\\bar e^2)$ conclusion; the headline $1.11%$ is the conservative upward report. This reconciliation is also stated in Appendix I.7.

### Detailed: HB validation (Figures 7–8, Appendix I.6)

```shell
python3 jd_hb_validation.py
```

Requires `jd_hb_posterior_summary.csv` from Step 1\. **Runtime:** \~30–50 minutes.

This script runs three exercises corresponding to Appendix I.6:

1. **Posterior predictive check (PPC).** Draw $K \= 100$ posterior samples of $\\hat\\beta\_i$, compute MCI-predicted log-share-ratios on every (SKU, day) cell (10,408 cells), report RMSE / MAE / 90% posterior-predictive interval coverage.  
2. **Temporal holdout.** Re-fit the HB-MCI model on the first 25 days of March 2018 (8,300 observations), predict log-share-ratios on the last 6 days (2,085 observations), report out-of-sample $R^2$.  
3. **Uncertainty propagation.** For each posterior draw of $\\hat\\beta\_i$, run the $\\gamma$-iteration \+ Newton-BN comparison on all 31 daily markets and record the resulting profit gap; report posterior median and 90% credible interval per day.

**Outputs:**

- `jd_ppc_diagnostics.txt` — PPC RMSE / MAE / 90% PPI coverage  
- `jd_ppc_histogram.png` — **Figure 7**  
- `jd_ppc_results.csv` — per-observation residuals  
- `jd_holdout_metrics.txt` — out-of-sample $R^2$ and RMSE  
- `jd_holdout_results.csv` — per-observation predictions on the holdout window  
- `jd_uncertainty_profit.csv` — per-day posterior median, q05, q95 of $\\gamma$-gap  
- `jd_uncertainty_interval.png` — **Figure 8**

**Headline numbers reported in Appendix I.6:**

- PPC: RMSE \= 0.717, MAE \= 0.515, 90% PPI coverage \= 36.7%  
- Holdout: out-of-sample $R^2$ \= 0.4364, RMSE \= 0.986  
- Uncertainty propagation: median(across days) of posterior median $\\gamma$-gap \= 1.85%; median width of 90% credible interval \= 0.38 pp; days 1 ($S=0.33$) and 8 ($S=0.22$) carry the widest intervals ($\[7.6%, 9.7%\]$ and $\[5.0%, 6.2%\]$); on the remaining 29 days the 90% interval is contained within $\[0.4%, 3.5%\]$.

The under-coverage in PPC (36.7% vs nominal 90%) reflects single-$\\sigma$ homoskedasticity in the likelihood; this does not affect the point estimate of $\\beta\_i$ or the downstream pricing comparison, but is acknowledged in Appendix I.6 as a model limitation motivating a heteroskedastic extension.

---

## 6\. Verifying outputs

After each script completes, compare your output files against the shipped reference outputs:

| Your fresh output (in `MainCodes/`) | Compare against |
| :---- | :---- |
| `jd_hb_pricing_comparison.csv` | `reference_output_results/jd_hb_pricing_comparison.csv` |
| `jd_gmv_pricing_comparison.csv` | `reference_output_results/jd_gmv_pricing_comparison.csv` |
| `jd_gmv_floor_summary.csv` | `reference_output_results/jd_gmv_floor_summary.csv` |
| `jd_topn_summary.csv` | `reference_output_results/jd_topn_summary.csv` |
| `jd_iv_estimates.csv` | `reference_output_results/jd_iv_estimates.csv` |
| `jd_hb_sensitivity_results.csv` | `reference_output_results/jd_hb_sensitivity_results.csv` |
| `scalability_results.csv` | `reference_output_results/scalability_results.csv` |
| `jd_uncertainty_profit.csv` | `reference_output_results/jd_uncertainty_profit.csv` |
| `jd_ppc_results.csv` | `reference_output_results/jd_ppc_results.csv` |
| ...all CSVs and PNGs | matching files in `reference_output_results/` and `figures/` |

Numerical agreement should be exact under the same software versions listed in §1, and to the 3rd decimal under reasonable version drift (e.g., numpy 1.x vs 2.x).

---

## 7\. Optional scripts in `MainCodes/`

The following scripts are also in `MainCodes/` but are **not required** to reproduce any numbered Table or Figure in the paper. They are included for completeness and reviewer follow-up:

| Script | Purpose |
| :---- | :---- |
| `jd_hb_prior_sensitivity.py` | Prior sensitivity check on $\\tau$: re-fits HB-MCI under $\\tau \\sim \\text{HalfNormal}(0.5)$ (baseline), $\\text{HalfNormal}(1.0)$, and $\\text{HalfCauchy}(0.5)$. Reports posterior $\\hat\\tau$ and per-SKU $\\hat\\beta\_i$ correlation across specs. Provided in case a reviewer asks whether the §8 partial-pooling conclusion is prior-driven. \~15–25 min. |
| `jd_pyblp.py` | PyBLP mixed-logit on JD with Hausman \+ rival-price \+ Gandhi–Houde IVs. Mentioned in Appendix I.4 framing. SKU `attribute1`/`attribute2` enter only through differentiation IVs (not $X\_1$); SKU fixed effects absorb time-invariant SKU characteristics. Requires `pip install pyblp`. |
| `jd_brand_experiment.py` | Brand-level elasticity alternative (top-15 brands \+ "other" pool). |
| `jd_sensitivity.py` | Bucket-level $\\beta$ version of the $M\\times c$ grid (faster, no PyMC). |
| `mixed_logit_gmv_constrained.py` | Synthetic mixed-logit GMV-floor experiment. In package per Appendix I footnote, not in main text. |
| `gamma_simulation.py` | Single-product $\\gamma$-index promotion-allocation illustration. Companion-paper material. |
| `gamma_worked_example.py` | Tiny 3-product worked example of the generalized Lerner rule. Pedagogical. |
| `identification_mc.py` | BLP-bias identification Monte Carlo. Used by the **companion paper** on $\\gamma$-corrected supply-side inversion; the main paper sketches the bias formula in Appendix K. |

To run any of these, invoke them the same way as the headline scripts: `python3 <script>.py` from inside `MainCodes/`.

---

## 8\. Google Colab workflow

If running on Colab with the data in Google Drive, the FUSE mount sometimes hides files from Python's import machinery. The robust workaround is to copy `MainCodes/` to local Colab storage at the start of the session:

```py
# Cell 1: mount Drive and copy scripts to local /content
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil
DRIVE = '/content/drive/MyDrive/gamma-equalization-replication'
LOCAL = '/content/gamma_local'
os.makedirs(LOCAL, exist_ok=True)

src = os.path.join(DRIVE, 'MainCodes')
for f in os.listdir(src):
    s = os.path.join(src, f)
    if os.path.isfile(s) and (f.endswith('.py') or f.endswith('.csv')):
        shutil.copy(s, os.path.join(LOCAL, f))

if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)

os.environ['JD_DATA_DIR'] = os.path.join(DRIVE, 'JD_MSOM')
print('Setup complete.')
```

```py
# Cell 2: install dependencies
!pip install -q pandas numpy scipy statsmodels matplotlib
!pip install -q pymc arviz nutpie
```

```py
# Cell 3: run scripts from /content/gamma_local using %run
%run /content/gamma_local/jd_hierarchical_bayes.py
%run /content/gamma_local/jd_gmv_constrained.py
%run /content/gamma_local/jd_hb_validation.py
%run /content/gamma_local/jd_topn_sensitivity.py
# ... etc.
```

```py
# Cell 4: copy outputs back to Drive
out_dir = os.path.join(DRIVE, 'MainCodes')
for f in os.listdir(LOCAL):
    if f.endswith('.csv') or f.endswith('.png') or f.endswith('.txt') or f.endswith('.nc'):
        shutil.copy(os.path.join(LOCAL, f), os.path.join(out_dir, f))
```

Use `%run` instead of pasting script content into a cell; `%run` sets `__file__` correctly so the scripts' path-resolution logic points to `/content/gamma_local/` for outputs.

---

## 9\. Troubleshooting

| Symptom | Fix |
| :---- | :---- |
| `FileNotFoundError: JD_*_data.csv` | The three CSVs are not at `$JD_DATA_DIR`. Re-check §2. |
| `ModuleNotFoundError: jd_hierarchical_bayes` (or `jd_gmv_constrained`) | You are not running from inside `MainCodes/`. `cd MainCodes` and retry. |
| `ModuleNotFoundError: pymc` | `pip install pymc arviz nutpie`. |
| `ModuleNotFoundError: pyblp` | Only needed for `jd_pyblp.py`. `pip install pyblp`. |
| `NameError: name 'OUT' is not defined` (Colab) | You pasted the script content into a cell instead of using `%run`. Use `%run /content/gamma_local/<script>.py` so `__file__` is correctly set. |
| `SyntaxWarning: invalid escape sequence` (Python 3.12+) | Cosmetic only; safe to ignore. Affects docstrings, not execution. |
| MCMC takes \>30 minutes | Install `nutpie` for \~3-5× speedup. If still slow, edit `DRAWS = 500` and `TUNE = 500` near the top of the offending script. |
| MCMC reports divergences | Edit `TARGET_ACCEPT = 0.95` (default 0.85). |
| Out-of-memory during MCMC | Edit `N_TOP_SKU = 300` (default 500\) at top of `jd_hierarchical_bayes.py`. For `jd_topn_sensitivity.py` at $N \= 2000$, reduce `N_VALUES` to drop the largest size. |
| `jd_topn_sensitivity.py` crashes partway | Per-$N$ HB results are cached; re-invoke and it will resume. |
| `jd_hb_validation.py` PPC coverage looks low (\~37%) | Expected — Appendix I.6 acknowledges this is from single-$\\sigma$ homoskedastic likelihood; a heteroskedastic refinement is flagged as future work. |
| Top-$N$ Table 9 reports $0.84%$ at $N \= 500$ but §8 reports $1.11%$ | Expected — see §5 above (and Appendix I.7) for the full reconciliation. Both numbers are correct under their respective reporting conventions. |

For other issues, please open a GitHub issue with the full traceback and your Python / numpy / pymc versions.

---

## 10\. Reproducibility notes

- All scripts use `SEED = 2026` for Monte Carlo draws; Bayesian scripts set `random_seed = SEED` in `pm.sample()`.  
- Bit-for-bit reproducibility holds on the same Python / numpy / scipy / pymc versions. Across major version bumps (e.g., numpy 2.x vs 1.x), results may drift in the 3rd–4th decimal due to internal RNG reorganization. All qualitative claims in the paper are robust to this drift.  
- For PyMC scripts, exact reproducibility also requires the same `pytensor` linear-algebra backend (`numpy`\-backed vs `jax`\-backed). Default Colab installs use `numpy`.  
- The PPC coverage in `jd_hb_validation.py` (≈ 36.7%) is determined by the single-$\\sigma$ likelihood and reproduces exactly across re-runs with the same seed.

---

## 11\. License and citation

Code: MIT License. The JD MSOM 2020 dataset is distributed under the MSOM challenge terms; see the [challenge page](https://connect.informs.org/msom/events/datadriven-call) for data-use conditions.

Cite the paper as:

\[\]. ** , \.  
