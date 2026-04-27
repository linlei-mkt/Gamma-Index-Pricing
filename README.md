# Replication Package for "Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand"

This package contains the code, data placeholder, and instructions to reproduce **every numbered Table (Tables 3–10) and every Figure (Figures 1–7) in the paper**, plus a set of supplementary scripts for robustness checks beyond the main text.

---

## 0\. Repository layout

After unzipping, the directory should look like this:

```
gamma-equalization-replication/
├── README.md                     ← this file
├── Gamma.pdf                     ← compiled paper (Marketing Science submission)
├── Gamma_index.pdf               ← companion working paper (single-product γ-index)
├── MainCodes/                    ← scripts that produce paper Tables 3–10 and Figures 1–7
│   └── supplementary/            ← optional / companion-paper / deprecated scripts
├── JD_MSOM/                      ← place the three MSOM CSVs here (empty by default)
├── figures/                      ← pre-generated reference PNGs (one per paper figure)
└── reference_output_results/     ← pre-generated reference CSVs (one per paper table CSV)
```

### One-time reorganization (run this first)

The shipped zip places several paper-required scripts in `MainCodes/supplementary/`. Before running anything, move them up so that `MainCodes/` contains exactly the scripts needed to reproduce the paper:

```shell
cd gamma-equalization-replication/MainCodes
mv supplementary/empirical_gamma.py            .
mv supplementary/jd_gmv_floor_sensitivity.py   .
mv supplementary/jd_hausman_iv.py              .
mv supplementary/scalability_demo.py           .
mv supplementary/jd_cbar_diagnostic.py         .
mv supplementary/jd_topn_sensitivity.py        .
mv mixed_logit_gmv_constrained.py supplementary/
```

Verify the result:

```shell
ls MainCodes/         # should list 11 .py files plus supplementary/
ls MainCodes/supplementary/   # should list 8 .py files
```

After reorganization, `MainCodes/` contains:

| Script | Reproduces |
| :---- | :---- |
| `empirical_gamma.py` | Table 3 (HKMR illustration) |
| `jd_experiment.py` | Figure 1, Figure 2 (right) |
| `jd_hierarchical_bayes.py` | Table 4, Figure 2 (left) |
| `jd_gmv_constrained.py` | Table 5, Figure 3 |
| `jd_gmv_floor_sensitivity.py` | Table 6, Figure 4 |
| `jd_hausman_iv.py` | Table 7 |
| `scalability_demo.py` | Table 8, Figure 5 |
| `jd_hb_sensitivity.py` | Table 9 |
| `mixed_logit_robustness.py` | §11.4 narrative |
| `jd_cbar_diagnostic.py` | Figure 6 |
| `jd_topn_sensitivity.py` | Table 10, Figure 7 |

`MainCodes/supplementary/` then contains optional / companion / alternate scripts (see §6 below).

---

## 1\. Data

You need the three CSVs from the **2020 MSOM Data-Driven Research Challenge** (Shen, Tang, Wu, Yuan, Zhou, *MSOM* 2020), available at [https://connect.informs.org/msom/events/datadriven-call](https://connect.informs.org/msom/events/datadriven-call):

| File | Size | Used by paper? |
| :---- | :---- | :---- |
| `JD_order_data.csv` | \~57 MB, 549,990 transactions | ✓ all empirical scripts |
| `JD_sku_data.csv` | \~1 MB, 31,868 SKUs | ✓ for `type` and `brand_ID` |
| `JD_user_data.csv` | \~18 MB, user demographics | ✗ not used; demographic-conditional extension noted as future work in §15 |

Place all three CSVs into the empty `JD_MSOM/` folder included in this bundle, then export the path:

```shell
cd gamma-equalization-replication
export JD_DATA_DIR=$(pwd)/JD_MSOM
```

(Or place them anywhere and point `JD_DATA_DIR` at that absolute path. If `JD_DATA_DIR` is unset, scripts fall back to a hard-coded path you can edit at the top of each file.)

---

## 2\. Setup

Python 3.9+ recommended. From the repository root:

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

## 3\. Paper result → script mapping (exact)

Every numbered Table and Figure in the paper, with the script that produces it and the exact output filename. **Run all scripts from inside `MainCodes/`** (i.e., `cd MainCodes` first) so that `Path(__file__).parent` resolves correctly and cross-script imports work. All output CSVs and PNGs are written into `MainCodes/`.

### Tables

| \# | § | Content | Script (in `MainCodes/`) | Primary output |
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

| \# | § | Content | Script | Output filename |
| :---- | :---- | :---- | :---- | :---- |
| 1 | §9.2 | $\\bar e$ distribution across 31 daily markets | `jd_experiment.py` | `jd_ebar_distribution.png` |
| 2 (left) | §9.3 | Profit gap vs $\\bar e$ on JD data | `jd_hierarchical_bayes.py` | `jd_hb_profit_gap_vs_ebar.png` |
| 2 (right) | §9.3 | Per-iteration convergence | `jd_experiment.py` | `jd_convergence.png` |
| 3 (left) | §10.2 | Profit vs revenue with floor | `jd_gmv_constrained.py` | `jd_gmv_profit_vs_revenue.png` |
| 3 (right) | §10.2 | Distribution of tuned $\\gamma^\\star$ | `jd_gmv_constrained.py` | `jd_gmv_gamma_star_distribution.png` |
| 4 | §10.3 | Floor sensitivity (gap and $\\gamma^\\star$ vs $\\phi$) | `jd_gmv_floor_sensitivity.py` | `jd_gmv_floor_sensitivity.png` |
| 5 | §11.2 | Scalability log-log curves | `scalability_demo.py` | `scalability_plot.png` |
| 6 | §11.5 | Empirical operator norm $\\rho\_t$ vs $\\bar e\_t$ | `jd_cbar_diagnostic.py` | `jd_cbar_scatter.png` |
| 7 | §11.7 | Top-$N$ sensitivity (median $\\bar e$, gaps vs $N$) | `jd_topn_sensitivity.py` | `jd_topn_sensitivity.png` |

### Reference outputs (for verification)

We ship pre-generated reference outputs:

- `figures/` — every paper figure as a PNG. Compare your fresh runs against these.  
- `reference_output_results/` — every paper-table CSV plus the per-day detail CSVs. Compare numerical agreement to the 3rd decimal (see §7 for tolerances).

---

## 4\. Minimum reproduction sequence

Run these in order from `MainCodes/`. Total runtime: \~2–3 hours on a laptop, dominated by the HB MCMC fits in steps 2 and 10\. On Colab with `nutpie`, MCMC steps drop to \~10–15 minutes each.

```shell
cd gamma-equalization-replication/MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM
```

### Step 1 — Bucket-level baseline (Figure 1, Figure 2 right)

```shell
python3 jd_experiment.py
```

Runtime: \~30 sec. Outputs:

- `jd_ebar_distribution.png` — **Figure 1**  
- `jd_convergence.png` — **Figure 2 (right)**  
- `jd_pricing_comparison.csv` — bucket-level per-day results (sanity)  
- `jd_elasticities.csv`, `jd_profit_gap_vs_ebar.png`, `jd_wallclock_comparison.png` — supplementary

### Step 2 — Hierarchical-Bayes MCI demand (Table 4, Figure 2 left)

```shell
python3 jd_hierarchical_bayes.py
```

Runtime: \~5–15 min with `nutpie`, \~30–60 min without. Outputs:

- `jd_hb_posterior_summary.csv` — posterior $\\hat\\beta\_i$ per SKU **(required by Steps 3, 4, 7, 9, 10\)**  
- `jd_hb_pricing_comparison.csv` — **Table 4**  
- `jd_hb_profit_gap_vs_ebar.png` — **Figure 2 (left)**  
- `jd_hb_bucket_means.csv`, `jd_hb_trace_plots.png`, `jd_hb_shrinkage.png` — diagnostics

### Step 3 — GMV-constrained pricing (Table 5, Figure 3\)

```shell
python3 jd_gmv_constrained.py
```

Requires `jd_hb_posterior_summary.csv` from Step 2\. Runtime: \~1 min. Outputs:

- `jd_gmv_pricing_comparison.csv` — **Table 5**  
- `jd_gmv_profit_vs_revenue.png` — **Figure 3 (left)**  
- `jd_gmv_gamma_star_distribution.png` — **Figure 3 (right)**

### Step 4 — GMV-floor sensitivity (Table 6, Figure 4\)

```shell
python3 jd_gmv_floor_sensitivity.py
```

Requires Step 2 plus `jd_gmv_constrained.py` in the same folder (it imports helpers). Runtime: \~5–10 min. Outputs:

- `jd_gmv_floor_summary.csv` — **Table 6**  
- `jd_gmv_floor_sensitivity.png` — **Figure 4**  
- `jd_gmv_floor_sensitivity.csv` — per-$(\\phi, \\mathrm{day})$ rows

### Step 5 — Hausman IV (Table 7\)

```shell
python3 jd_hausman_iv.py
```

Reads JD orders directly (no HB dependency). Runtime: \~30 sec. Outputs:

- `jd_iv_estimates.csv` — **Table 7**  
- `jd_iv_first_stage.csv`, `jd_iv_comparison.png` — diagnostics

### Step 6 — Scalability demo (Table 8, Figure 5\)

```shell
python3 scalability_demo.py
```

Synthetic, no JD data. Runtime: \~3 min. Outputs:

- `scalability_results.csv` — **Table 8**  
- `scalability_plot.png` — **Figure 5**

### Step 7 — $M \\times c$ sensitivity (Table 9\)

```shell
python3 jd_hb_sensitivity.py
```

Requires Step 2\. Runtime: \~10 min with `nutpie`. Outputs:

- `jd_hb_sensitivity_results.csv` — **Table 9**  
- `jd_hb_sensitivity_heatmap_gamma.png`, `jd_hb_sensitivity_heatmap_ebar.png`, `jd_hb_sensitivity_heatmap_uniform.png`, `jd_hb_sensitivity_heatmap_speedup.png` — heatmap visuals

### Step 8 — Mixed-logit Monte Carlo (§11.4 narrative)

```shell
python3 mixed_logit_robustness.py
```

Synthetic, no JD data. Runtime: \~30 sec. Outputs:

- `mixed_logit_results.csv` — 79 synthetic markets  
- `mixed_logit_profit_gap.png` — $\\bar e^2$ fit diagnostic  
- `mixed_logit_convergence.png`, `mixed_logit_compare_mci.png` — supplementary

### Step 9 — Empirical contraction diagnostic (Figure 6\)

```shell
python3 jd_cbar_diagnostic.py
```

Requires Step 2\. Runtime: \~30 sec. Outputs:

- `jd_cbar_scatter.png` — **Figure 6**  
- `jd_cbar_results.csv`, `jd_cbar_histogram.png` — supplementary

### Step 10 — Top-$N$ robustness (Table 10, Figure 7\)

```shell
python3 jd_topn_sensitivity.py
```

Imports from `jd_hierarchical_bayes.py` and re-fits HB MCMC at four $N$ values. Runtime: \~40–60 min. Per-$N$ HB posteriors are cached, so re-runs skip already-done $N$. Outputs:

- `jd_topn_summary.csv` — **Table 10**  
- `jd_topn_sensitivity.png` — **Figure 7**  
- `jd_topn_sensitivity.csv` — per-$(N, \\mathrm{day})$ rows  
- `jd_hb_posterior_summary_N{200,500,1000,2000}.csv` — per-$N$ HB caches

### Step 11 — HKMR calibrated illustration (Table 3\)

```shell
python3 empirical_gamma.py
```

Synthetic, no JD data. Runtime: \~5 sec. Output: console only — read **Table 3** numbers directly from the printed comparison block. The script does not write a CSV; the printed format is the table.

---

## 5\. Google Colab workflow

Google Drive's FUSE mount sometimes hides files from Python's import machinery even when `os.path.exists` confirms they are present. The robust workaround is to copy the `MainCodes/` scripts into local Colab storage `/content/` once at the start of a session.

```py
# Cell 1: mount Drive and copy MainCodes/ scripts to local /content
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil
DRIVE   = '/content/drive/MyDrive/gamma-equalization-replication'
LOCAL   = '/content/gamma_local'
os.makedirs(LOCAL, exist_ok=True)

# Copy all MainCodes/ scripts and the HB cache (if it exists)
src_dir = os.path.join(DRIVE, 'MainCodes')
for f in os.listdir(src_dir):
    s = os.path.join(src_dir, f)
    if os.path.isfile(s) and (f.endswith('.py') or f.endswith('.csv')):
        shutil.copy(s, os.path.join(LOCAL, f))

if LOCAL not in sys.path:
    sys.path.insert(0, LOCAL)

# JD CSVs stay on Drive
os.environ['JD_DATA_DIR'] = os.path.join(DRIVE, 'JD_MSOM')
print('Setup complete.')
```

```py
# Cell 2: install dependencies
!pip install -q pandas numpy scipy statsmodels matplotlib
!pip install -q pymc arviz nutpie       # for HB MCMC scripts
!pip install -q pyblp                    # optional, for jd_pyblp.py
```

```py
# Cell 3: run scripts (they read from /content/gamma_local and write outputs there)
%run /content/gamma_local/jd_hierarchical_bayes.py
%run /content/gamma_local/jd_gmv_constrained.py
%run /content/gamma_local/jd_gmv_floor_sensitivity.py
%run /content/gamma_local/jd_topn_sensitivity.py
# ... etc.
```

```py
# Cell 4: copy outputs back to Drive once done
out_dir = os.path.join(DRIVE, 'MainCodes')
for f in os.listdir(LOCAL):
    if f.endswith('.csv') or f.endswith('.png'):
        shutil.copy(os.path.join(LOCAL, f), os.path.join(out_dir, f))
print('Outputs synced back to Drive.')
```

---

## 6\. Supplementary scripts (not required for paper)

`MainCodes/supplementary/` contains scripts that are not used in the main text but are included for completeness:

| Script | Purpose |
| :---- | :---- |
| `jd_pyblp.py` | PyBLP mixed-logit on JD with Hausman \+ rival-price IVs and Gandhi–Houde differentiation IVs. Mentioned in §11.4 framing. SKU attributes 1–2 enter only through differentiation IVs (not $X\_1$) because SKU fixed effects absorb time-invariant SKU characteristics. Runtime: 10–60 min. |
| `jd_brand_experiment.py` | Brand-level elasticity alternative (top-15 brands \+ "other"). Robustness only. |
| `jd_sensitivity.py` | Bucket-level $\\beta$ version of the $M \\times c$ grid (faster than `jd_hb_sensitivity.py`, no PyMC). |
| `jd_hb_validation.py` | Posterior predictive check \+ holdout fit \+ posterior propagation. Runs but **results are catalogued as scope limitations in §11.6** (see paper); not in main results. Runtime: \~30–60 min on Colab. |
| `mixed_logit_gmv_constrained.py` | Synthetic mixed-logit GMV-floor experiment. In package per §11 footnote, not in main text. |
| `gamma_simulation.py` | Single-product $\\gamma$-index promotion-allocation illustration. Companion working paper material. |
| `gamma_worked_example.py` | Tiny 3-product worked example of the generalized Lerner rule. Pedagogical only. |
| `identification_mc.py` | BLP-bias identification Monte Carlo. Used by the **companion paper** on $\\gamma$-corrected supply-side inversion; the main paper only sketches the bias formula in §12. |

To run a supplementary script, `cd MainCodes/supplementary` and invoke it directly. None of these are required to reproduce any numbered Table or Figure in the paper.

---

## 7\. Reproducibility notes

- All scripts use `SEED = 2026` for Monte Carlo draws. Bayesian scripts set `random_seed=SEED` in `pm.sample()`.  
- Bit-for-bit reproducibility holds on the same Python / numpy / scipy / pymc versions. Across major version bumps (e.g., numpy 2.x vs 1.x), results may drift in the 3rd–4th decimal due to internal RNG reorganization. All qualitative claims in the paper are robust to this drift.  
- For PyMC scripts, exact reproducibility also requires the same `pytensor` linear-algebra backend (`numpy`\-backed vs `jax`\-backed sampling). Default Colab installs use `numpy`.  
- `figures/` and `reference_output_results/` are the reference outputs generated on the testing environment listed in §2. Use them to verify your fresh runs.

---

## 8\. Common failure modes

In decreasing order of probability:

1. **`FileNotFoundError` on JD CSVs** — `$JD_DATA_DIR` is unset or wrong. Make sure the three CSVs are inside the folder pointed to by `JD_DATA_DIR`. The scripts expect exactly the filenames `JD_order_data.csv`, `JD_sku_data.csv`, `JD_user_data.csv`.  
     
2. **`ModuleNotFoundError: jd_hierarchical_bayes` (or similar)** — Either you forgot to do the one-time reorganization (see §0), or you are running the script from outside `MainCodes/`. Either:  
     
   - `cd MainCodes` and re-run, or  
   - on Colab, use the `/content/gamma_local` workaround from §5.

   

3. **`ModuleNotFoundError: pymc`** — `pip install pymc arviz nutpie`.  
     
4. **`ModuleNotFoundError: pyblp`** — Only needed for the supplementary `jd_pyblp.py`. `pip install pyblp`.  
     
5. **MCMC very slow** — Install `nutpie` for \~3–5× speedup. If still slow, drop `DRAWS` and `TUNE` at the top of the offending script from 1000 to 500\.  
     
6. **MCMC divergences** — Raise `TARGET_ACCEPT` from 0.85 to 0.95.  
     
7. **Out-of-memory at $N=500$ MCMC** — Drop `N_TOP_SKU` from 500 to 300 at the top of `jd_hierarchical_bayes.py`. For the top-$N$ sweep at $N=2000$, OOM is more likely; lower `N_VALUES` in `jd_topn_sensitivity.py`.  
     
8. **`jd_topn_sensitivity.py` crashes at HB step at $N \\neq 500$** — The script monkey-patches `hb.N_TOP_SKU` and re-fits HB MCMC. The per-$N$ posterior is cached to `jd_hb_posterior_summary_N{n}.csv` after success, so a crashed run can be resumed by re-invoking the script (it skips cached $N$).

---

## 9\. License and citation

Code: MIT License. The JD MSOM 2020 dataset is distributed under the MSOM challenge terms; see the MSOM challenge website for data-use conditions.

Cite the paper as:
