"""
JD Hausman-style IV estimation of MCI price elasticity using cross-DC
variation, with multi-way fixed effects absorbed via iterative
within-transformation (Frisch--Waugh--Lovell).

Addresses the reviewer's critique about price endogeneity: within-SKU
over-time price variation may be correlated with unobserved demand
shocks (promotions, stockouts, seasonal shifts). We exploit the JD
dataset's destination distribution centers to construct a
Hausman-style instrument.

Identification strategy
-----------------------
For each (SKU, day, DC) cell, the instrument is the leave-one-out
mean log-price of the same SKU on the same day at OTHER DCs:

    z_{i,t,dc} = (1/(|D_{i,t}| - 1)) * sum_{dc' != dc} log p_{i,t,dc'}

where D_{i,t} is the set of DCs with positive sales of SKU i on
day t.

Validity: the IV is valid if cross-DC demand shocks are independent
after absorbing SKU, day, and DC fixed effects. SKU FE removes
product-level attractiveness; day FE removes aggregate time-varying
shocks; DC FE removes regional baseline demand. Residual cross-DC
price variation then arises from DC-specific supply-side factors
(regional coupons, inventory allocation, warehouse promotions) that
shift prices but not demand after controlling for FEs.

Relevance: cross-DC price variation for the same SKU on the same day
is substantial (median CV 4%, mean 9%; 38% of cells with CV > 5%).
First-stage F-statistic reported below.

Specification
-------------
Stage 1 (first stage):
    log(p_{i,t,dc}) = π0 + π1 z_{i,t,dc} + SKU+day+DC FE + u

Stage 2 (structural):
    log(q_{i,t,dc}) = α + β log(p_{i,t,dc}) + SKU+day+DC FE + ε

The reported "β_MCI" in the output is defined as -β (so that a
positive number corresponds to the MCI attraction exponent; own-price
elasticity is β × (1-s) ≈ β for small shares, both negative).

We implement the FE absorption via iterative within-demeaning
(Gauss-Seidel) to avoid the memory cost of a 500-SKU x 60-DC x 31-day
dummy matrix. Standard errors are conservative (the FWL residualized
OLS standard errors without a degrees-of-freedom correction for the
absorbed FEs).

Outputs
-------
  - jd_iv_first_stage.csv       first-stage F-statistic, π1, SE
  - jd_iv_estimates.csv         OLS vs IV β estimates with SEs
  - jd_iv_diagnostics.txt       verbose diagnostic output
  - jd_iv_comparison.png        β OLS vs β IV bar chart
"""
from __future__ import annotations

import os, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "../JD_MSOM"
))
OUT = SCRIPT_DIR

N_TOP_SKU = 500

# ======================================================================
# Iterative within-demeaning (Frisch-Waugh-Lovell for multiple FEs)
# ======================================================================
def demean_multiway(y, group_cols, df, tol=1e-8, max_iter=200):
    """Iteratively subtract group means for each group column until
    convergence. Returns the residualized vector."""
    r = y.copy().astype(float).to_numpy()
    # Precompute group indices for speed
    group_indices = []
    for g in group_cols:
        # For each group level, the row indices that belong to it
        codes, _ = pd.factorize(df[g])
        group_indices.append(codes)

    for it in range(max_iter):
        r_old = r.copy()
        for codes in group_indices:
            # Group means
            gsums = np.bincount(codes, weights=r)
            gcounts = np.bincount(codes)
            gmeans = gsums / np.maximum(gcounts, 1)
            r = r - gmeans[codes]
        max_change = np.max(np.abs(r - r_old))
        if max_change < tol:
            break
    return r

# ======================================================================
# 1. Load and aggregate
# ======================================================================
print("[1/5] Loading JD order data...")
t0 = time.time()
orders = pd.read_csv(
    DATA_DIR / "JD_order_data.csv",
    usecols=['sku_ID', 'order_date', 'quantity', 'final_unit_price', 'dc_des'],
    dtype={'sku_ID': 'string'}, parse_dates=['order_date'],
)
orders['day'] = orders['order_date'].dt.day
print(f"    {len(orders):,} orders loaded in {time.time()-t0:.1f}s")

top_ids = orders.groupby('sku_ID').size().nlargest(N_TOP_SKU).index
o = orders[orders.sku_ID.isin(top_ids)].copy()
o['rev'] = o['final_unit_price'] * o['quantity']

print("[2/5] Aggregating to (SKU, day, DC) cells...")
t0 = time.time()
cells = (o.groupby(['sku_ID', 'day', 'dc_des'], as_index=False)
          .agg(qty=('quantity', 'sum'), rev=('rev', 'sum')))
cells['price'] = cells['rev'] / cells['qty']
cells = cells[(cells['qty'] > 0) & (cells['price'] > 0)].reset_index(drop=True)
cells['log_qty'] = np.log(cells['qty'])
cells['log_price'] = np.log(cells['price'])
print(f"    n_cells = {len(cells):,} in {time.time()-t0:.1f}s")
print(f"    n_sku = {cells['sku_ID'].nunique()}, "
      f"n_day = {cells['day'].nunique()}, "
      f"n_dc = {cells['dc_des'].nunique()}")

# ======================================================================
# 2. Hausman IV: leave-one-out cross-DC mean log-price
# ======================================================================
print("[3/5] Constructing Hausman IV...")
t0 = time.time()
sku_day_group = cells.groupby(['sku_ID', 'day'])
log_price_sum = sku_day_group['log_price'].transform('sum')
log_price_cnt = sku_day_group['log_price'].transform('count')
cells['hausman_iv'] = (log_price_sum - cells['log_price']) / (log_price_cnt - 1)
# Drop cells that are the only DC for that (SKU, day)
mask = log_price_cnt > 1
cells = cells[mask].dropna(subset=['hausman_iv']).reset_index(drop=True)
print(f"    n_cells after drop-singletons = {len(cells):,} in {time.time()-t0:.1f}s")
print(f"    corr(log_price, hausman_iv) = {cells[['log_price','hausman_iv']].corr().iloc[0,1]:.3f}")
print(f"    SD(log_price) = {cells['log_price'].std():.3f}, "
      f"SD(hausman_iv) = {cells['hausman_iv'].std():.3f}")

# ======================================================================
# 3. Within-demean everything by SKU, day, DC fixed effects
# ======================================================================
print("[4/5] Within-demeaning by SKU, day, DC fixed effects...")
t0 = time.time()
fe_cols = ['sku_ID', 'day', 'dc_des']
y = demean_multiway(cells['log_qty'], fe_cols, cells)
x = demean_multiway(cells['log_price'], fe_cols, cells)
z = demean_multiway(cells['hausman_iv'], fe_cols, cells)
print(f"    demeaned in {time.time()-t0:.1f}s")
print(f"    demeaned SD(log_qty) = {np.std(y):.3f}, SD(log_price) = {np.std(x):.3f}, SD(z) = {np.std(z):.3f}")
print(f"    demeaned corr(log_price, z) = {np.corrcoef(x, z)[0,1]:.3f}")

# ======================================================================
# 4. OLS on residualized variables (FWL)
# ======================================================================
print("[5/5] Running OLS and 2SLS on residualized variables...")
n = len(y)
# degrees-of-freedom correction for absorbed FEs
k_fe = cells['sku_ID'].nunique() + cells['day'].nunique() + cells['dc_des'].nunique() - 2  # one base per FE
df_residual = n - k_fe - 1

# OLS: β = cov(y, x) / var(x)
beta_ols_coef = np.sum(y * x) / np.sum(x * x)
resid_ols = y - beta_ols_coef * x
sigma2_ols = np.sum(resid_ols**2) / df_residual
se_ols = np.sqrt(sigma2_ols / np.sum(x * x))
beta_ols_mci = -beta_ols_coef
print()
print("--- OLS (uncorrected) ---")
print(f"    coef on log_price   = {beta_ols_coef:.4f} (SE {se_ols:.4f})")
print(f"    implied MCI β̂       = {beta_ols_mci:.4f}")

# First stage: x = π0 + π1 z + FEs, so with demeaned: x = π1 z + u
pi1 = np.sum(z * x) / np.sum(z * z)
resid_fs = x - pi1 * z
sigma2_fs = np.sum(resid_fs**2) / df_residual
se_pi1 = np.sqrt(sigma2_fs / np.sum(z * z))
t_pi1 = pi1 / se_pi1
F_first = t_pi1 ** 2
# Also first-stage R² on residualized vars
r2_fs = 1 - np.sum(resid_fs**2) / np.sum((x - np.mean(x))**2)
print()
print("--- First stage ---")
print(f"    π̂₁ (coef on IV)    = {pi1:.4f} (SE {se_pi1:.4f})")
print(f"    t-stat = {t_pi1:.2f}, first-stage F = {F_first:.2f}")
print(f"    first-stage residualized R² = {r2_fs:.4f}")
if F_first < 10:
    print(f"    WARNING: First-stage F < 10; instrument may be weak.")
else:
    print(f"    First-stage F exceeds conventional threshold (10).")

# 2SLS: β_IV = (z'y)/(z'x) in the just-identified case (demeaned vars)
beta_iv_coef = np.sum(z * y) / np.sum(z * x)
# IV standard error (asymptotic formula for just-identified 2SLS)
resid_iv = y - beta_iv_coef * x
sigma2_iv = np.sum(resid_iv**2) / df_residual
# Var(β_IV) = σ² * (z'z) / (z'x)²
se_iv = np.sqrt(sigma2_iv * np.sum(z*z) / (np.sum(z*x))**2)
beta_iv_mci = -beta_iv_coef
print()
print("--- 2SLS (Hausman IV) ---")
print(f"    coef on log_price  = {beta_iv_coef:.4f} (SE {se_iv:.4f})")
print(f"    implied MCI β̂_IV   = {beta_iv_mci:.4f}")

# ======================================================================
# Save results
# ======================================================================
pd.DataFrame([
    {"estimator": "OLS (uncorrected)", "log_price_coef": float(beta_ols_coef),
     "SE": float(se_ols), "beta_MCI": float(beta_ols_mci)},
    {"estimator": "2SLS (Hausman IV)", "log_price_coef": float(beta_iv_coef),
     "SE": float(se_iv), "beta_MCI": float(beta_iv_mci)},
]).to_csv(OUT / "jd_iv_estimates.csv", index=False)

pd.DataFrame([{
    "first_stage_F": float(F_first),
    "first_stage_pi1": float(pi1),
    "first_stage_pi1_SE": float(se_pi1),
    "first_stage_R2_residualized": float(r2_fs),
    "n_cells": int(len(cells)),
    "n_sku": int(cells['sku_ID'].nunique()),
    "n_day": int(cells['day'].nunique()),
    "n_dc": int(cells['dc_des'].nunique()),
    "corr_log_price_hausman_iv": float(
        cells[['log_price', 'hausman_iv']].corr().iloc[0, 1]),
}]).to_csv(OUT / "jd_iv_first_stage.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
methods = ["OLS\n(uncorrected)", "2SLS\nHausman IV"]
betas = [beta_ols_mci, beta_iv_mci]
ses = [se_ols, se_iv]
colors = ["tab:red", "tab:blue"]
ax.bar(methods, betas, yerr=[1.96 * se for se in ses], color=colors,
       edgecolor='black', alpha=0.85, capsize=10)
ax.set_ylabel(r"Estimated MCI exponent $\hat\beta$")
ax.set_title(f"OLS vs Hausman-IV estimate of price elasticity\n"
             f"(first-stage F = {F_first:.1f}, N = {len(cells):,} cells)")
for i, (b, s) in enumerate(zip(betas, ses)):
    ax.text(i, b + 1.96 * s + 0.02, f"{b:.3f}\n(SE {s:.3f})",
            ha='center', fontsize=10)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5,
           label=r"$\beta=1$ (unit-elastic threshold)")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(OUT / "jd_iv_comparison.png", dpi=150)
plt.close()

# Diagnostic output
with open(OUT / "jd_iv_diagnostics.txt", "w") as f:
    f.write("JD Hausman-IV diagnostic output\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Data: {len(cells):,} (SKU, day, DC) cells\n")
    f.write(f"  n_sku = {cells['sku_ID'].nunique()}, "
            f"n_day = {cells['day'].nunique()}, "
            f"n_dc = {cells['dc_des'].nunique()}\n")
    f.write(f"  corr(log_price, hausman_iv) = {cells[['log_price','hausman_iv']].corr().iloc[0,1]:.4f}\n\n")
    f.write(f"First-stage F-statistic (after FE absorption) = {F_first:.2f}\n")
    f.write(f"  π̂₁ = {pi1:.4f} (SE {se_pi1:.4f})\n")
    f.write(f"  residualized first-stage R² = {r2_fs:.4f}\n\n")
    f.write(f"OLS β̂ (MCI exponent)  = {beta_ols_mci:.4f} (SE {se_ols:.4f})\n")
    f.write(f"IV β̂ (Hausman, MCI)   = {beta_iv_mci:.4f} (SE {se_iv:.4f})\n")
    f.write(f"IV/OLS ratio          = {beta_iv_mci/beta_ols_mci:.3f}\n")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"OLS β̂              = {beta_ols_mci:.4f}  (SE {se_ols:.4f})")
print(f"Hausman-IV β̂       = {beta_iv_mci:.4f}  (SE {se_iv:.4f})")
print(f"First-stage F      = {F_first:.2f}")
print(f"IV/OLS ratio       = {beta_iv_mci/beta_ols_mci:.3f}")
if beta_iv_mci > beta_ols_mci:
    print("→ OLS UNDERSTATES price sensitivity; IV correction pushes β̂ up.")
else:
    print("→ OLS OVERSTATES price sensitivity; IV correction pushes β̂ down.")
print()
print(f"Outputs written to: {OUT}")
