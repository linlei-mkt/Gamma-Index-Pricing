# Replication Guide

Step-by-step protocol for replicating every empirical Table (3–10) and every Figure (1–7) in *Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand*.

The replication needs only Python 3.9+ and the JD MSOM 2020 dataset (included in this repository under `JD_MSOM/`). Total runtime: \~15 minutes for the headline numbers, \~2–3 hours for every Table and Figure.

---

## Quick start

```shell
git clone <repo-url>
cd gamma-equalization-replication
pip install -r requirements.txt

cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM

python3 jd_hierarchical_bayes.py     # Table 4, Figure 2
python3 jd_gmv_constrained.py         # Table 5, Figure 3
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
| `JD_user_data.csv` | \~18 MB, user demographics | Not used; demographic-conditional extension is noted as future work in §15 |

Set the data path:

```shell
cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM
```

If `JD_DATA_DIR` is unset, scripts fall back to a hard-coded default at the top of each file that you can edit.

---

## 3\. Paper element → script mapping

Every numbered Table and Figure with the script that produces it and the exact output filename. **Run all scripts from inside `MainCodes/`.**

### Tables

| \# | Section | Content | Script | Primary output |
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

| \# | Section | Content | Script | Output filename |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §9.2 | $\\bar e$ distribution across 31 daily markets | `jd_experiment.py` | `jd_ebar_distribution.png` |
| 2 (left) | §9.3 | Profit gap vs $\\bar e$ on JD data | `jd_hierarchical_bayes.py` | `jd_hb_profit_gap_vs_ebar.png` |
| 2 (right) | §9.3 | Per-iteration convergence | `jd_experiment.py` | `jd_convergence.png` |
| 3 (left) | §10.2 | Profit vs revenue with floor | `jd_gmv_constrained.py` | `jd_gmv_profit_vs_revenue.png` |
| 3 (right) | §10.2 | Distribution of tuned $\\gamma^\\star$ | `jd_gmv_constrained.py` | `jd_gmv_gamma_star_distribution.png` |
| 4 | §10.3 | Floor sensitivity (gap and $\\gamma^\\star$ vs $\\phi$) | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_sensitivity.png` |
| 5 | §11.2 | Scalability log-log curves | `scalability_demo.py` | `scalability_plot.png` |
| 6 | §11.5 | Empirical operator norm $\\rho\_t$ vs $\\bar e\_t$ | `jd_cbar_diagnostic.py` | `jd_cbar_scatter.png` |
| 7 | §11.7 | Top-$N$ sensitivity | `jd_topn_sensitivity.py` | `jd_topn_sensitivity.png` |

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
- `jd_hb_pricing_comparison.csv` — **Table 4**  
- `jd_hb_profit_gap_vs_ebar.png` — **Figure 2 (left)**  
- `jd_hb_bucket_means.csv`, `jd_hb_trace_plots.png`, `jd_hb_shrinkage.png` — MCMC diagnostics

**Headline number to check:** Median value of column `gap_gamma` in `jd_hb_pricing_comparison.csv` should be approximately **1.11%**.

### Step 2 — GMV-constrained pricing

```shell
python3 jd_gmv_constrained.py
```

Requires `jd_hb_posterior_summary.csv` from Step 1\. **Runtime:** \~1 minute.

**Outputs:**

- `jd_gmv_pricing_comparison.csv` — **Table 5**  
- `jd_gmv_profit_vs_revenue.png` — **Figure 3 (left)**  
- `jd_gmv_gamma_star_distribution.png` — **Figure 3 (right)**

**Headline numbers to check:** In `jd_gmv_pricing_comparison.csv`, median `gap_gamma_pct` ≈ **5.57%** and median `gap_unif_pct` ≈ **15.93%**.

If both replicate within ±0.05 pp, the paper's empirical core is verified.

---

## 5\. Full reproduction (\~2–3 hours)

After Steps 1 and 2 above, run the remaining scripts to reproduce every Table and Figure. Most are independent and can be parallelized across terminals if desired.

### Independent scripts (any order, no HB dependency)

```shell
python3 empirical_gamma.py            # Table 3 (console output)
python3 jd_experiment.py              # Figure 1, Figure 2 right
python3 jd_hausman_iv.py              # Table 7
python3 scalability_demo.py           # Table 8, Figure 5
python3 mixed_logit_robustness.py     # §11.4 narrative
```

| Script | Runtime | Notes |
| :---- | :---- | :---- |
| `empirical_gamma.py` | \~5 sec | Synthetic; reads HKMR (1995) elasticities. Output is **console only** — read the printed comparison block to recover Table 3\. |
| `jd_experiment.py` | \~30 sec | Bucket-level MCI baseline (no MCMC). |
| `jd_hausman_iv.py` | \~30 sec | Reads JD orders directly. |
| `scalability_demo.py` | \~3 min | Synthetic; no JD data needed. |
| `mixed_logit_robustness.py` | \~30 sec | 79 synthetic mixed-logit markets. |

### Scripts depending on `jd_hb_posterior_summary.csv` (Step 1 output)

```shell
python3 jd_gmv_floor_sensitivity.py   # Table 6, Figure 4
python3 jd_hb_sensitivity.py          # Table 9
python3 jd_cbar_diagnostic.py         # Figure 6
```

| Script | Runtime | Notes |
| :---- | :---- | :---- |
| `jd_gmv_floor_sensitivity.py` | 5–10 min | Sweeps $\\phi \\in {1.05, 1.10, 1.15, 1.20, 1.25}$. Imports helpers from `jd_gmv_constrained.py`. |
| `jd_hb_sensitivity.py` | \~10 min with `nutpie` | Re-fits HB MCMC at five $M\_{\\mathrm{mult}}$ values. |
| `jd_cbar_diagnostic.py` | \~30 sec | Computes empirical operator norm $\\rho\_t$ on 31 daily markets. |

### Heaviest single script

```shell
python3 jd_topn_sensitivity.py        # Table 10, Figure 7
```

**Runtime:** 40–60 minutes. Re-fits HB MCMC at $N \\in {200, 500, 1000, 2000}$. Per-$N$ HB posteriors are cached to `jd_hb_posterior_summary_N{n}.csv`, so a crashed run can be resumed by re-invoking the script.

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
| ...all CSVs and PNGs | matching files in `reference_output_results/` and `figures/` |

Numerical agreement should be exact under the same software versions listed in §1, and to the 3rd decimal under reasonable version drift (e.g., numpy 1.x vs 2.x).

---

## 7\. Optional scripts in `MainCodes/`

The following scripts are also in `MainCodes/` but are **not required** to reproduce any numbered Table or Figure in the paper. They are included for completeness:

| Script | Purpose |
| :---- | :---- |
| `jd_pyblp.py` | PyBLP mixed-logit on JD with Hausman \+ rival-price \+ Gandhi–Houde IVs. Mentioned in §11.4 framing. SKU `attribute1`/`attribute2` enter only through differentiation IVs (not $X\_1$); SKU fixed effects absorb time-invariant SKU characteristics. Requires `pip install pyblp`. |
| `jd_brand_experiment.py` | Brand-level elasticity alternative (top-15 brands \+ "other" pool). |
| `jd_sensitivity.py` | Bucket-level $\\beta$ version of the $M\\times c$ grid (faster, no PyMC). |
| `jd_hb_validation.py` | PPC \+ holdout \+ posterior propagation. Results catalogued as scope limitations in §11.6, not in main results. |
| `mixed_logit_gmv_constrained.py` | Synthetic mixed-logit GMV-floor experiment. In package per §11 footnote, not in main text. |
| `gamma_simulation.py` | Single-product $\\gamma$-index promotion-allocation illustration. Companion-paper material. |
| `gamma_worked_example.py` | Tiny 3-product worked example of the generalized Lerner rule. Pedagogical. |
| `identification_mc.py` | BLP-bias identification Monte Carlo. Used by the **companion paper** on $\\gamma$-corrected supply-side inversion; the main paper only sketches the bias formula in §12. |

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
# Cell 3: run scripts from /content/gamma_local
%run /content/gamma_local/jd_hierarchical_bayes.py
%run /content/gamma_local/jd_gmv_constrained.py
# ... etc.
```

```py
# Cell 4: copy outputs back to Drive
out_dir = os.path.join(DRIVE, 'MainCodes')
for f in os.listdir(LOCAL):
    if f.endswith('.csv') or f.endswith('.png'):
        shutil.copy(os.path.join(LOCAL, f), os.path.join(out_dir, f))
```

---

## 9\. Troubleshooting

| Symptom | Fix |
| :---- | :---- |
| `FileNotFoundError: JD_*_data.csv` | The three CSVs are not at `$JD_DATA_DIR`. Re-check §2. |
| `ModuleNotFoundError: jd_hierarchical_bayes` (or `jd_gmv_constrained`) | You are not running from inside `MainCodes/`. `cd MainCodes` and retry. |
| `ModuleNotFoundError: pymc` | `pip install pymc arviz nutpie`. |
| `ModuleNotFoundError: pyblp` | Only needed for `jd_pyblp.py`. `pip install pyblp`. |
| MCMC takes \>30 minutes | Install `nutpie` for \~3-5× speedup. If still slow, edit `DRAWS = 500` and `TUNE = 500` near the top of `jd_hierarchical_bayes.py`. |
| MCMC reports divergences | Edit `TARGET_ACCEPT = 0.95` (default 0.85). |
| Out-of-memory during MCMC | Edit `N_TOP_SKU = 300` (default 500\) at top of `jd_hierarchical_bayes.py`. For `jd_topn_sensitivity.py` at $N \= 2000$, reduce `N_VALUES` to drop the largest size. |
| `jd_topn_sensitivity.py` crashes partway | Per-$N$ HB results are cached; just re-invoke the script and it will resume. |

For other issues, please open a GitHub issue with the full traceback and your Python / numpy / pymc versions.

---

## 10\. Reproducibility notes

- All scripts use `SEED = 2026` for Monte Carlo draws; Bayesian scripts set `random_seed = SEED` in `pm.sample()`.  
- Bit-for-bit reproducibility holds on the same Python / numpy / scipy / pymc versions. Across major version bumps (e.g., numpy 2.x vs 1.x), results may drift in the 3rd–4th decimal due to internal RNG reorganization. All qualitative claims in the paper are robust to this drift.  
- For PyMC scripts, exact reproducibility also requires the same `pytensor` linear-algebra backend (`numpy`\-backed vs `jax`\-backed). Default Colab installs use `numpy`.

---

## 11\. License and citation

Code: MIT License. The JD MSOM 2020 dataset is distributed under the MSOM challenge terms; see the [challenge page](https://connect.informs.org/msom/events/datadriven-call) for data-use conditions.

Cite the paper as:

\[Authors blinded for review\]. *Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand.* Working paper, 2026\.  
