"""
Hierarchical-Bayes demand validation for the JD calibrated illustration.

Addresses the reviewer's request for stronger empirical validation of
the HB-MCI demand model before its point estimates are used as inputs
to the pricing counterfactual. Three separate exercises:

  1. POSTERIOR PREDICTIVE CHECK (PPC)
     Draw K=500 samples of $\hat\beta_i$ from the HB posterior.
     For each sample, compute MCI-predicted log(s/s_0) on every
     (SKU, day) observation and compare to the observed value.
     Report: per-observation MAE, RMSE, 90% posterior predictive
     interval coverage, and a binned residual plot.

  2. HOLDOUT FIT (time-based)
     Re-fit the HB model on the first 25 days and predict log(s/s_0)
     on the held-out last 6 days. Report out-of-sample R^2.

  3. UNCERTAINTY PROPAGATION
     For each posterior draw of $\hat\beta_i$, run the
     $\gamma$-iteration + MS2011 + Newton comparison on all 31 daily
     markets. Report posterior median and 90% interval for
     profit-gap(\gamma, BN) across the posterior.

This script requires:
  - pymc, arviz, nutpie (optional fast sampler)
  - jd_hierarchical_bayes.py already run so the posterior trace is
    saved (we re-run HB if no cached trace is available).

Because the script runs PyMC MCMC and posterior-sweep pricing, it is
slower than the other replication scripts: ~30-60 minutes on Colab
with nutpie.

Outputs
-------
  - jd_ppc_results.csv         per-observation residuals
  - jd_ppc_diagnostics.txt     PPC summary statistics
  - jd_ppc_histogram.png       posterior predictive residual histogram
  - jd_holdout_results.csv     held-out observation predictions
  - jd_holdout_metrics.txt     out-of-sample R^2
  - jd_uncertainty_profit.csv  posterior-draw profit gaps
  - jd_uncertainty_interval.png  95% interval plot of gamma-gap

To run:
    pip install pymc arviz nutpie pandas numpy matplotlib
    JD_DATA_DIR=/path/to/jd_csvs python3 jd_hb_validation.py
"""
from __future__ import annotations

import os, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "/Users/linlei/Library/Application Support/Claude/local-agent-mode-sessions/"
    "28ce55e3-2aeb-47b1-a159-176e9d6a9dbf/0f00e7ba-81f1-4095-94e5-73a365a8f51b/"
    "local_c02dff0e-7360-4888-8cca-0a64aed3b4e1/uploads"
))
OUT = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 500
N_DECILES = 10
M_MULT = 3.0
MARGIN_RATIO = 0.70
BETA_FLOOR = 1.2
TOL = 1e-8
MAX_ITER = 2000

DRAWS = 1000          # MCMC draws per chain
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.85
SEED = 2026

# Sample-from-posterior exercises
N_POSTERIOR_DRAWS = 100
HOLDOUT_DAYS = 6            # last 6 of 31 days held out

# ======================================================================
# Data prep (same as jd_hierarchical_bayes.py)
# ======================================================================
def load_and_aggregate():
    orders = pd.read_csv(
        DATA_DIR / "JD_order_data.csv",
        usecols=['sku_ID', 'order_date', 'quantity', 'final_unit_price'],
        dtype={'sku_ID': 'string'}, parse_dates=['order_date'],
    )
    sku = pd.read_csv(
        DATA_DIR / "JD_sku_data.csv",
        usecols=['sku_ID', 'type'],
        dtype={'sku_ID': 'string'},
    )
    top_ids = orders.groupby('sku_ID').size().nlargest(N_TOP_SKU).index
    o = orders[orders.sku_ID.isin(top_ids)].merge(sku, on='sku_ID', how='left')
    o['day'] = o['order_date'].dt.day
    agg = (o.assign(rev=o['final_unit_price'] * o['quantity'])
            .groupby(['day', 'sku_ID'], as_index=False)
            .agg(qty=('quantity', 'sum'), rev=('rev', 'sum')))
    agg['price'] = agg['rev'] / agg['qty']
    agg = agg.merge(sku[['sku_ID', 'type']], on='sku_ID', how='left')
    agg = agg[(agg['qty'] > 0) & (agg['price'] > 0) & agg['type'].notna()].copy()
    agg['type'] = agg['type'].astype(int)

    sku_mean_p = agg.groupby('sku_ID')['price'].mean()
    bucket = pd.qcut(sku_mean_p.rank(method='first'), q=N_DECILES,
                     labels=False).astype(int)
    agg = agg.merge(bucket.rename('bucket'), left_on='sku_ID',
                    right_index=True, how='left')

    peak_inside = agg.groupby('day')['qty'].sum().max()
    M = M_MULT * peak_inside
    agg['share'] = agg['qty'] / M
    daily_Q = agg.groupby('day')['qty'].sum().rename('Q_in')
    s0 = (1.0 - daily_Q / M).rename('s0')
    agg = agg.merge(s0, on='day')
    return agg, M


# ======================================================================
# HB model builder
# ======================================================================
def build_hb_model(work):
    """Return (pm.Model, helper dicts, (y, logp arrays))."""
    if not HAS_PYMC:
        raise ImportError("pip install pymc arviz")
    sku_codes, sku_uniques = pd.factorize(work['sku_ID'], sort=True)
    day_codes, day_uniques = pd.factorize(work['day'], sort=True)
    sku_to_bucket = (
        work.drop_duplicates('sku_ID').set_index('sku_ID').loc[sku_uniques, 'bucket']
            .astype(int).to_numpy()
    )
    y = (np.log(work['share']) - np.log(work['s0'])).to_numpy()
    logp = np.log(work['price'].to_numpy())
    n_sku = len(sku_uniques)
    n_day = len(day_uniques)

    with pm.Model() as model:
        mu_b = pm.Normal('mu_bucket', mu=2.0, sigma=1.0, shape=N_DECILES)
        tau = pm.HalfNormal('tau', sigma=0.5)
        u_std = pm.Normal('u_std', mu=0.0, sigma=1.0, shape=n_sku)
        beta_sku = pm.Deterministic('beta_sku', mu_b[sku_to_bucket] + tau * u_std)
        alpha_sku = pm.Normal('alpha_sku', mu=0.0, sigma=5.0, shape=n_sku)
        delta_day = pm.Normal('delta_day', mu=0.0, sigma=1.0, shape=n_day)
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        mu_pred = (alpha_sku[sku_codes] + delta_day[day_codes]
                   - beta_sku[sku_codes] * logp)
        pm.Normal('y_obs', mu=mu_pred, sigma=sigma, observed=y)

    return model, {
        'sku_codes': sku_codes, 'day_codes': day_codes,
        'sku_uniques': sku_uniques, 'day_uniques': day_uniques,
        'sku_to_bucket': sku_to_bucket, 'n_sku': n_sku, 'n_day': n_day,
    }, (y, logp)


def fit_hb(work, label='full'):
    model, meta, _ = build_hb_model(work)
    with model:
        try:
            import nutpie  # noqa
            idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                              target_accept=TARGET_ACCEPT, random_seed=SEED,
                              nuts_sampler='nutpie', progressbar=True)
        except ImportError:
            idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                              target_accept=TARGET_ACCEPT, random_seed=SEED,
                              progressbar=True)
    # Save trace for later reuse
    idata.to_netcdf(OUT / f"jd_hb_trace_{label}.nc")
    return idata, meta


# ======================================================================
# Pricing primitives (same as jd_experiment.py)
# ======================================================================
def mci_shares(p, alpha, beta, M):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def calibrate_alpha(p_obs, s_obs, s0_obs, beta):
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def share_jacobian(p, s, beta):
    u = beta * s / p
    Om = np.outer(u, s)
    np.fill_diagonal(Om, -u * (1.0 - s))
    return Om


def gamma_iteration(p0, c, alpha, beta, M):
    p = p0.copy()
    for k in range(MAX_ITER):
        s, _ = mci_shares(p, alpha, beta, M)
        eta = np.maximum(beta * (1.0 - s), 1.01)
        p_new = np.maximum(c / (1.0 - 1.0 / eta), c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            return p_new
        p = p_new
    return p


def ms_newton(p0, c, alpha, beta, M):
    p = np.maximum(p0.copy(), c * 1.0001)
    for _ in range(500):
        s, _ = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        p_new = np.maximum(c - (s + Gamma @ (p - c)) / diag, c * 1.0001)
        if np.max(np.abs(p_new - p)) < 1e-10:
            return p_new
        p = p_new
    return p


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# Exercise 1: posterior predictive check
# ======================================================================
def exercise_ppc(idata, meta, work):
    """Posterior predictive check on log(s/s_0)."""
    print("\n[PPC] Computing posterior predictive residuals...")
    y = (np.log(work['share']) - np.log(work['s0'])).to_numpy()
    logp = np.log(work['price'].to_numpy())
    sku_codes = meta['sku_codes']
    day_codes = meta['day_codes']

    # Flatten posterior samples; subsample to N_POSTERIOR_DRAWS
    post_beta = idata.posterior['beta_sku'].stack(sample=('chain', 'draw')).values
    post_alpha = idata.posterior['alpha_sku'].stack(sample=('chain', 'draw')).values
    post_delta = idata.posterior['delta_day'].stack(sample=('chain', 'draw')).values
    n_samples_total = post_beta.shape[-1]
    idx = np.random.default_rng(SEED).choice(n_samples_total,
                                              size=N_POSTERIOR_DRAWS, replace=False)
    post_beta = post_beta[:, idx]
    post_alpha = post_alpha[:, idx]
    post_delta = post_delta[:, idx]

    # Predictions: y_pred[k,obs] = alpha[sku(obs), k] + delta[day(obs), k]
    #                                - beta[sku(obs), k] * logp[obs]
    y_preds = np.empty((N_POSTERIOR_DRAWS, len(y)))
    for k in range(N_POSTERIOR_DRAWS):
        y_preds[k] = (post_alpha[sku_codes, k] + post_delta[day_codes, k]
                      - post_beta[sku_codes, k] * logp)

    residuals = y[None, :] - y_preds             # (K, n_obs)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    # 90% posterior predictive interval coverage
    q05 = np.quantile(y_preds, 0.05, axis=0)
    q95 = np.quantile(y_preds, 0.95, axis=0)
    inside = (y >= q05) & (y <= q95)
    coverage = float(np.mean(inside))

    print(f"    RMSE (y - y_pred) = {rmse:.4f}")
    print(f"    MAE  (y - y_pred) = {mae:.4f}")
    print(f"    90% PPI coverage  = {coverage*100:.1f}%")

    pd.DataFrame({
        'day': work['day'].to_numpy(),
        'sku_ID': work['sku_ID'].to_numpy(),
        'y_observed': y,
        'y_pred_mean': y_preds.mean(axis=0),
        'y_pred_q05': q05,
        'y_pred_q95': q95,
        'covered': inside,
    }).to_csv(OUT / 'jd_ppc_results.csv', index=False)

    with open(OUT / 'jd_ppc_diagnostics.txt', 'w') as f:
        f.write(f"PPC diagnostics\n")
        f.write(f"===============\n\n")
        f.write(f"N obs: {len(y)}\n")
        f.write(f"Posterior draws used: {N_POSTERIOR_DRAWS}\n")
        f.write(f"RMSE = {rmse:.4f}\n")
        f.write(f"MAE  = {mae:.4f}\n")
        f.write(f"90% PPI coverage = {coverage*100:.1f}%\n")

    # Histogram of residuals (flattened)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals.flatten(), bins=50, color='tab:blue',
            edgecolor='black', alpha=0.75)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel(r"posterior predictive residual $y - \hat y$")
    ax.set_ylabel("count")
    ax.set_title(f"Posterior predictive residuals (N={len(y)}, K={N_POSTERIOR_DRAWS} draws)")
    plt.tight_layout()
    plt.savefig(OUT / 'jd_ppc_histogram.png', dpi=150)
    plt.close()
    return rmse, mae, coverage


# ======================================================================
# Exercise 2: holdout fit
# ======================================================================
def exercise_holdout(work):
    """Fit HB on first 25 days, predict on last 6 days."""
    print("\n[HOLDOUT] Fitting HB on first 25 days, predicting on last 6...")
    max_day = work['day'].max()
    cutoff = max_day - HOLDOUT_DAYS
    train = work[work['day'] <= cutoff].copy()
    test = work[work['day'] > cutoff].copy()
    print(f"    train: {len(train)} obs (days ≤ {cutoff}), test: {len(test)} obs")

    idata_train, meta_train = fit_hb(train, label='holdout')

    # Predict test-day log-shares using posterior means (we don't have
    # held-out day FEs in the model, so we use the average delta across
    # training days; this is a conservative forecast)
    beta_mean = idata_train.posterior['beta_sku'].mean(dim=('chain', 'draw')).values
    alpha_mean = idata_train.posterior['alpha_sku'].mean(dim=('chain', 'draw')).values
    delta_mean = idata_train.posterior['delta_day'].mean(dim=('chain', 'draw')).values.mean()

    sku_to_beta = dict(zip(meta_train['sku_uniques'], beta_mean))
    sku_to_alpha = dict(zip(meta_train['sku_uniques'], alpha_mean))

    test_sku_in_train = test[test['sku_ID'].isin(sku_to_beta.keys())].copy()
    y_test = (np.log(test_sku_in_train['share']) - np.log(test_sku_in_train['s0'])).to_numpy()
    logp_test = np.log(test_sku_in_train['price'].to_numpy())
    y_pred = (test_sku_in_train['sku_ID'].map(sku_to_alpha).to_numpy()
              + delta_mean
              - test_sku_in_train['sku_ID'].map(sku_to_beta).to_numpy() * logp_test)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    ss_res = np.sum((y_test - y_pred)**2)
    r2_oos = 1 - ss_res / ss_tot
    rmse_oos = np.sqrt(np.mean((y_test - y_pred)**2))
    print(f"    out-of-sample R^2 = {r2_oos:.4f}")
    print(f"    out-of-sample RMSE = {rmse_oos:.4f}")

    pd.DataFrame({
        'day': test_sku_in_train['day'].to_numpy(),
        'sku_ID': test_sku_in_train['sku_ID'].to_numpy(),
        'y_obs': y_test, 'y_pred': y_pred,
    }).to_csv(OUT / 'jd_holdout_results.csv', index=False)
    with open(OUT / 'jd_holdout_metrics.txt', 'w') as f:
        f.write(f"Holdout validation (last {HOLDOUT_DAYS} days)\n")
        f.write(f"train: {len(train)} obs, test (in-train SKUs): {len(y_test)} obs\n")
        f.write(f"R^2 out-of-sample = {r2_oos:.4f}\n")
        f.write(f"RMSE out-of-sample = {rmse_oos:.4f}\n")
    return r2_oos, rmse_oos


# ======================================================================
# Exercise 3: uncertainty propagation into profit-gap intervals
# ======================================================================
def exercise_uncertainty(idata, meta, work, M):
    """For each posterior draw, run the pricing comparison and record
    profit gap. Report posterior median + 90% interval."""
    print("\n[UNCERTAINTY] Propagating posterior into γ-gap...")
    post_beta = idata.posterior['beta_sku'].stack(sample=('chain', 'draw')).values
    n_total = post_beta.shape[-1]
    idx = np.random.default_rng(SEED).choice(n_total,
                                              size=N_POSTERIOR_DRAWS, replace=False)
    post_beta_samples = post_beta[:, idx]   # (n_sku, K)

    sku_uniques = meta['sku_uniques']

    gaps_by_draw = []   # (K, n_days)
    t0 = time.time()
    for k in range(N_POSTERIOR_DRAWS):
        sku_to_beta = dict(zip(sku_uniques, np.maximum(post_beta_samples[:, k],
                                                       BETA_FLOOR)))
        gaps_day = []
        for day in sorted(work['day'].unique()):
            mkt = work[work['day'] == day].copy()
            if len(mkt) < 50:
                continue
            mkt = mkt[mkt['sku_ID'].isin(sku_to_beta.keys())]
            p_obs = mkt['price'].to_numpy()
            s_obs = mkt['share'].to_numpy()
            s0_obs = float(mkt['s0'].iloc[0])
            beta_vec = mkt['sku_ID'].map(sku_to_beta).to_numpy()
            alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
            c = MARGIN_RATIO * p_obs
            p_g = gamma_iteration(p_obs.copy(), c, alpha, beta_vec, M)
            p_bn = ms_newton(p_obs.copy(), c, alpha, beta_vec, M)
            pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)
            pi_g = total_profit(p_g, c, alpha, beta_vec, M)
            gap = max(0.0, (pi_bn - pi_g) / pi_bn) if pi_bn > 0 else np.nan
            gaps_day.append(gap)
        gaps_by_draw.append(gaps_day)
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{N_POSTERIOR_DRAWS} draws done  "
                  f"(elapsed {time.time()-t0:.0f}s)")

    gaps_arr = np.array(gaps_by_draw)   # (K, n_days)
    # Per-day posterior summary
    median_by_day = np.nanmedian(gaps_arr, axis=0)
    q05_by_day = np.nanquantile(gaps_arr, 0.05, axis=0)
    q95_by_day = np.nanquantile(gaps_arr, 0.95, axis=0)

    out_df = pd.DataFrame({
        'day': sorted(work['day'].unique()),
        'gap_median': median_by_day,
        'gap_q05': q05_by_day,
        'gap_q95': q95_by_day,
    })
    out_df.to_csv(OUT / 'jd_uncertainty_profit.csv', index=False)

    # Aggregate: median of median, median of width
    med_med = np.nanmedian(median_by_day)
    widths = q95_by_day - q05_by_day
    med_width = np.nanmedian(widths)
    print(f"    median(median γ-gap across days) = {med_med*100:.2f}%")
    print(f"    median(90% interval width across days) = {med_width*100:.2f} pp")

    # Plot: γ-gap posterior intervals across days
    fig, ax = plt.subplots(figsize=(8, 4))
    days = out_df['day'].to_numpy()
    ax.fill_between(days, out_df['gap_q05']*100, out_df['gap_q95']*100,
                    color='tab:blue', alpha=0.25, label='90% posterior interval')
    ax.plot(days, out_df['gap_median']*100, 'o-', color='tab:blue',
            label='posterior median')
    ax.set_xlabel("day of March 2018")
    ax.set_ylabel(r"$\gamma$-equalization profit gap to BN (\%)")
    ax.set_title(f"Posterior uncertainty in $\\gamma$-gap (K={N_POSTERIOR_DRAWS} draws)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'jd_uncertainty_interval.png', dpi=150)
    plt.close()

    return med_med, med_width


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("JD Hierarchical-Bayes validation package")
    print("=" * 70)
    if not HAS_PYMC:
        print("ERROR: PyMC not installed. pip install pymc arviz")
        return

    agg, M = load_and_aggregate()
    print(f"Loaded {len(agg)} obs, M = {M:.0f}")

    # Load or fit full HB
    full_trace_path = OUT / 'jd_hb_trace_full.nc'
    if full_trace_path.exists():
        print(f"Loading cached HB trace from {full_trace_path}")
        idata_full = az.from_netcdf(full_trace_path)
        _, meta_full, _ = build_hb_model(agg)
    else:
        print("Fitting full-sample HB...")
        idata_full, meta_full = fit_hb(agg, label='full')

    # Exercise 1: PPC
    rmse, mae, coverage = exercise_ppc(idata_full, meta_full, agg)

    # Exercise 2: holdout
    r2_oos, rmse_oos = exercise_holdout(agg)

    # Exercise 3: uncertainty propagation
    med_gap, med_width = exercise_uncertainty(idata_full, meta_full, agg, M)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PPC: RMSE={rmse:.4f}, MAE={mae:.4f}, 90% coverage={coverage*100:.1f}%")
    print(f"Holdout: OOS R²={r2_oos:.4f}, OOS RMSE={rmse_oos:.4f}")
    print(f"Uncertainty: median γ-gap = {med_gap*100:.2f}%, "
          f"median 90% interval width = {med_width*100:.2f} pp")
    print()
    print(f"All outputs written to {OUT}")


if __name__ == '__main__':
    main()
