"""
JD hierarchical-Bayes SENSITIVITY SWEEP (M × margin grid).

Purpose
-------
The sensitivity sweep in jd_sensitivity.py re-estimated 10 bucket-level
β's at each (M_mult, margin) combination.  This version re-estimates
the HIERARCHICAL BAYESIAN per-SKU β̂_i at each M_mult, so the sensitivity
analysis is done on the paper's primary empirical specification rather
than the exposition baseline.

Because β̂_i depends on shares, which depend on M_mult, HB must be
re-run for each value of M_mult.  margin only affects cost c = margin·p,
not demand, so one HB fit per M_mult suffices — we then sweep margin
within each.

Grid:
    M_mult ∈ {1.5, 2, 3, 5, 7}   (5 HB fits)
    margin ∈ {0.4, 0.5, 0.6, 0.7, 0.8}   (5 per HB fit)
    Total: 25 pricing comparisons, 5 MCMC runs.

Runtime: each HB fit ~1.5 minutes on Colab with nutpie, plus ~1 second
per pricing run. Total ~10 minutes on Colab.

Outputs (in MainCodes/):
  - jd_hb_sensitivity_results.csv       one row per (M_mult, margin)
  - jd_hb_sensitivity_heatmap_gamma.png γ median profit gap (%)
  - jd_hb_sensitivity_heatmap_uniform.png uniform profit gap (%)
  - jd_hb_sensitivity_heatmap_ebar.png median ebar (%)
  - jd_hb_sensitivity_heatmap_speedup.png γ-vs-MS wall-clock speedup

Required packages:
    pip install pymc arviz pandas numpy scipy statsmodels matplotlib
    pip install nutpie   # optional, ~3-5× faster sampling

Usage:
    JD_DATA_DIR=/path/to/csvs python3 jd_hb_sensitivity.py

If you want a quick sanity check (one or two M_mult values), edit
M_MULTS and MARGINS near the top. If MCMC sampling is too slow
reduce DRAWS and TUNE below.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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
    "../JD_MSOM"
))
OUT_DIR = SCRIPT_DIR

# ==================  GRID  ==================
M_MULTS = [1.5, 2.0, 3.0, 5.0, 7.0]
MARGINS = [0.4, 0.5, 0.6, 0.7, 0.8]

# ==================  Fixed config  ==================
N_TOP_SKU = 500
N_DECILES = 10
BETA_FLOOR = 1.2
TOL = 1e-8
MAX_ITER = 2000
SEED = 2026

# MCMC config  (trim to speed up if needed)
DRAWS = 700
TUNE = 700
CHAINS = 2
TARGET_ACCEPT = 0.85


# ======================================================================
# Data and price-decile assignment (same as jd_sensitivity.py)
# ======================================================================
def load_data():
    orders = pd.read_csv(
        DATA_DIR / "JD_order_data.csv",
        usecols=["sku_ID", "order_date", "quantity", "final_unit_price"],
        dtype={"sku_ID": "string"}, parse_dates=["order_date"],
    )
    sku = pd.read_csv(
        DATA_DIR / "JD_sku_data.csv",
        usecols=["sku_ID", "type", "brand_ID"],
        dtype={"sku_ID": "string", "brand_ID": "string"},
    )
    top_ids = orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index.tolist()
    o = orders[orders["sku_ID"].isin(top_ids)].merge(sku, on="sku_ID", how="left")
    o["day"] = o["order_date"].dt.day
    agg = (
        o.assign(rev=o["final_unit_price"] * o["quantity"])
         .groupby(["day", "sku_ID"], as_index=False)
         .agg(qty=("quantity", "sum"), rev=("rev", "sum"))
    )
    agg["price"] = agg["rev"] / agg["qty"]
    agg = agg.merge(sku[["sku_ID", "type"]], on="sku_ID", how="left")
    agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
    # Price decile by each SKU's mean observed price
    sku_mean_p = agg.groupby("sku_ID")["price"].mean()
    bucket = pd.qcut(sku_mean_p.rank(method="first"), q=N_DECILES,
                     labels=False).astype(int)
    agg = agg.merge(bucket.rename("bucket"), left_on="sku_ID",
                    right_index=True, how="left")
    return agg


def prepare_for_m(agg, M_mult):
    """Compute shares and s0 for this value of M_mult."""
    work = agg.copy()
    peak = work.groupby("day")["qty"].sum().max()
    M = M_mult * peak
    work["share"] = work["qty"] / M
    daily_Q = work.groupby("day")["qty"].sum().rename("Q_in")
    s0 = (1.0 - daily_Q / M).rename("s0")
    work = work.merge(s0, on="day")
    return work, M


# ======================================================================
# Hierarchical Bayes estimation at one M_mult
# ======================================================================
def fit_hb_one_m(work):
    if not HAS_PYMC:
        raise ImportError("pip install pymc arviz")
    sku_codes, sku_uniques = pd.factorize(work["sku_ID"], sort=True)
    day_codes, day_uniques = pd.factorize(work["day"], sort=True)
    sku_to_bucket = (
        work.drop_duplicates("sku_ID").set_index("sku_ID")
            .loc[sku_uniques, "bucket"].astype(int).to_numpy()
    )
    n_sku = len(sku_uniques)
    n_day = len(day_uniques)
    y = (np.log(work["share"]) - np.log(work["s0"])).to_numpy()
    logp = np.log(work["price"].to_numpy())

    with pm.Model():
        mu_b = pm.Normal("mu_bucket", mu=2.0, sigma=1.0, shape=N_DECILES)
        tau = pm.HalfNormal("tau", sigma=0.5)
        u_std = pm.Normal("u_std", mu=0.0, sigma=1.0, shape=n_sku)
        beta_sku = pm.Deterministic(
            "beta_sku", mu_b[sku_to_bucket] + tau * u_std
        )
        alpha_sku = pm.Normal("alpha_sku", mu=0.0, sigma=5.0, shape=n_sku)
        delta_day = pm.Normal("delta_day", mu=0.0, sigma=1.0, shape=n_day)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu = (alpha_sku[sku_codes] + delta_day[day_codes]
              - beta_sku[sku_codes] * logp)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        try:
            import nutpie  # noqa
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                nuts_sampler="nutpie", progressbar=False,
            )
        except ImportError:
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                progressbar=False,
            )

    beta_mean = idata.posterior["beta_sku"].mean(dim=("chain", "draw")).values
    beta_mean = np.maximum(beta_mean, BETA_FLOOR)
    return dict(zip(sku_uniques, beta_mean))


# ======================================================================
# MCI pricing primitives (identical to jd_sensitivity.py)
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


def ebar_from_shares(s):
    S = s.sum()
    return float(np.max((S - s) / (1.0 - s)))


def gamma_iteration(p0, c, alpha, beta, M):
    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(MAX_ITER):
        s, _ = mci_shares(p, alpha, beta, M)
        eta = np.maximum(beta * (1.0 - s), 1.01)
        p_new = np.maximum(c / (1.0 - 1.0 / eta), c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new; break
        p = p_new
    return p, k + 1, time.perf_counter() - t0


def ms_iteration(p0, c, alpha, beta, M):
    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(MAX_ITER):
        s, _ = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        p_new = np.maximum(c - (s + Gamma @ (p - c)) / diag, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new; break
        p = p_new
    return p, k + 1, time.perf_counter() - t0


def newton_bn(p0, c, alpha, beta, M):
    """Robust MS-based near-BN solver."""
    p, _, _ = ms_iteration(p0, c, alpha, beta, M)
    return p


def uniform_pricing(c, alpha, beta, M):
    def neg_profit(m):
        if not (0.0 < m < 0.999):
            return 1e18
        p = c / (1.0 - m)
        s, _ = mci_shares(p, alpha, beta, M)
        return -np.sum((p - c) * s) * M
    res = minimize_scalar(neg_profit, bounds=(0.001, 0.999), method="bounded",
                          options={"xatol": 1e-8})
    return c / (1.0 - res.x)


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# Sensitivity sweep with HB at each M_mult
# ======================================================================
def run_one_margin(work, M, sku_to_beta, margin):
    """Run γ/MS/uniform/Newton across all days at this (M, margin)."""
    results = []
    work = work[work["sku_ID"].isin(sku_to_beta.keys())].copy()
    work["beta_hat"] = work["sku_ID"].map(sku_to_beta)
    for day in sorted(work["day"].unique()):
        mkt = work[work["day"] == day].copy().reset_index(drop=True)
        if len(mkt) < 50:
            continue
        p_obs = mkt["price"].to_numpy()
        s_obs = mkt["share"].to_numpy()
        s0_obs = float(mkt["s0"].iloc[0])
        beta_vec = mkt["beta_hat"].to_numpy()
        alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
        c = margin * p_obs
        p0 = p_obs.copy()
        ebar = ebar_from_shares(s_obs)

        p_g, it_g, t_g = gamma_iteration(p0, c, alpha, beta_vec, M)
        p_m, it_m, t_m = ms_iteration(p0, c, alpha, beta_vec, M)
        p_u = uniform_pricing(c, alpha, beta_vec, M)
        p_bn = newton_bn(p_m.copy(), c, alpha, beta_vec, M)
        pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)
        if pi_bn <= 0:
            continue
        gap = lambda p: max(0.0, (pi_bn - total_profit(p, c, alpha, beta_vec, M)) / pi_bn)
        results.append({
            "day": int(day),
            "ebar": ebar,
            "gap_gamma": gap(p_g),
            "gap_MS": gap(p_m),
            "gap_uniform": gap(p_u),
            "time_gamma": t_g,
            "time_MS": t_m,
            "iter_gamma": it_g,
            "iter_MS": it_m,
        })
    if not results:
        return None
    df = pd.DataFrame(results)
    return {
        "n_days": len(df),
        "ebar_median": df["ebar"].median(),
        "ebar_max": df["ebar"].max(),
        "gap_gamma_mean": df["gap_gamma"].mean(),
        "gap_gamma_median": df["gap_gamma"].median(),
        "gap_gamma_max": df["gap_gamma"].max(),
        "gap_uniform_mean": df["gap_uniform"].mean(),
        "gap_uniform_median": df["gap_uniform"].median(),
        "gamma_beats_uniform_pct": (df["gap_gamma"] < df["gap_uniform"]).mean() * 100,
        "gamma_gap_lt_ebar2_pct": (df["gap_gamma"] < df["ebar"] ** 2).mean() * 100,
        "time_gamma_mean": df["time_gamma"].mean(),
        "time_MS_mean": df["time_MS"].mean(),
        "speedup_gamma_vs_MS": df["time_MS"].mean() / df["time_gamma"].mean(),
        "iter_gamma_mean": df["iter_gamma"].mean(),
        "iter_MS_mean": df["iter_MS"].mean(),
    }


def main():
    print("=" * 70)
    print("JD HIERARCHICAL-BAYES SENSITIVITY SWEEP")
    print("=" * 70)
    if not HAS_PYMC:
        print("pip install pymc arviz")
        return
    print(f"M_MULTS:  {M_MULTS}")
    print(f"MARGINS:  {MARGINS}")
    print(f"Total HB fits: {len(M_MULTS)},  pricing runs: {len(M_MULTS)*len(MARGINS)}")
    print()

    t_global = time.time()
    print("[1/3] Loading data...")
    agg = load_data()

    print("[2/3] HB estimation and pricing across the grid:")
    rows = []
    for M_mult in M_MULTS:
        work, M = prepare_for_m(agg, M_mult)
        print(f"  ── M_mult = {M_mult} ── fitting HB (n_obs={len(work)})...")
        t0 = time.time()
        sku_to_beta = fit_hb_one_m(work)
        t_hb = time.time() - t0
        print(f"     HB fit done in {t_hb/60:.1f} min; median β̂ = "
              f"{np.median(list(sku_to_beta.values())):.3f}")
        for margin in MARGINS:
            t1 = time.time()
            res = run_one_margin(work, M, sku_to_beta, margin)
            t2 = time.time() - t1
            if res is None:
                print(f"     margin={margin}: FAILED ({t2:.1f}s)")
                continue
            res["M_mult"] = M_mult
            res["margin"] = margin
            res["M"] = M
            rows.append(res)
            print(f"     margin={margin}: γ-gap {res['gap_gamma_median']*100:5.2f}%  "
                  f"uniform {res['gap_uniform_median']*100:5.2f}%  "
                  f"speedup {res['speedup_gamma_vs_MS']:.0f}×  ({t2:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "jd_hb_sensitivity_results.csv", index=False)

    pivots = {}
    for col in ["gap_gamma_median", "gap_uniform_median", "ebar_median",
                "speedup_gamma_vs_MS"]:
        pivots[col] = df.pivot(index="M_mult", columns="margin", values=col)

    def heatmap(mat, title, fname, cmap, fmt):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
        ax.set_xlabel("margin (c / p)")
        ax.set_ylabel("M_mult")
        ax.set_title(title + " (HB β̂)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=150)
        plt.close()

    print("[3/3] Generating heatmaps...")
    heatmap(pivots["gap_gamma_median"] * 100,
            "γ-equalization median profit gap (%)",
            "jd_hb_sensitivity_heatmap_gamma.png", "viridis", "{:.2f}")
    heatmap(pivots["gap_uniform_median"] * 100,
            "Uniform-markup median profit gap (%)",
            "jd_hb_sensitivity_heatmap_uniform.png", "magma", "{:.2f}")
    heatmap(pivots["ebar_median"],
            "Median ebar at observed prices",
            "jd_hb_sensitivity_heatmap_ebar.png", "coolwarm", "{:.3f}")
    heatmap(pivots["speedup_gamma_vs_MS"],
            "γ vs MS2011 wall-clock speedup (×)",
            "jd_hb_sensitivity_heatmap_speedup.png", "viridis", "{:.0f}")

    print()
    print("=" * 70)
    print(f"Total runtime: {(time.time() - t_global)/60:.1f} min")
    print()
    print("γ-equalization median profit gap (%), HB β̂:")
    print((pivots["gap_gamma_median"] * 100).round(2).to_string())
    print()
    print("uniform median profit gap (%), HB β̂:")
    print((pivots["gap_uniform_median"] * 100).round(2).to_string())
    print()
    print("median ebar:")
    print(pivots["ebar_median"].round(3).to_string())
    print()
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
