"""
JD top-N robustness sweep for §11 of the paper.

Re-fits hierarchical-Bayes MCI demand at four catalog-truncation levels
N ∈ {200, 500, 1000, 2000} (top-N SKUs by order count) and runs the
four-method pricing comparison (γ-iteration, MS2011, uniform, Newton)
at each. Reports how (a) inside share, (b) median ē, (c) median
γ-equalization profit gap to BN, and (d) γ-vs-uniform dominance count
move with N.

This addresses the reviewer concern that the §9 main result is pinned
to the top-500 truncation. Expected pattern: as N grows, inside share
grows (more SKUs are inside the market), so median ē mechanically
grows, and the γ-equalization profit gap tracks ē² consistent with
the theory. The qualitative conclusion (γ ≈ BN at small ē, γ beats
uniform under elasticity heterogeneity) should be invariant.

Strategy
--------
For each N we re-fit HB MCMC (because the SKU set, the price-decile
buckets, and the observed shares all change with N). We cache the
posterior to jd_hb_posterior_summary_N{n}.csv so re-runs of this
script skip already-fitted N values.

Outputs
-------
  - jd_topn_sensitivity.csv          per-(N, day) results
  - jd_topn_summary.csv              4-row summary across N levels
  - jd_topn_sensitivity.png          summary plot

Usage
-----
  JD_DATA_DIR=/path/to/csvs python3 jd_topn_sensitivity.py

Runtime: ~40-60 minutes on a laptop with nutpie (4 HB fits ×
~10 min + per-N pricing). Cached HB posteriors skip MCMC on re-run.

Memory note: at N = 2000, HB MCMC trace can use ~1-2 GB. If you have
< 8 GB RAM, drop the largest N from N_VALUES below.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root

warnings.simplefilter("ignore", category=FutureWarning)

# Make sure the directory containing this script (or the standard
# Colab Drive folder) is on sys.path so we can import the local
# jd_hierarchical_bayes module regardless of how the script is
# invoked (notebook %run, !python3, or content pasted into a cell).
import sys as _sys
for _candidate in (
    "/content/drive/MyDrive/JD_gamma",
    str(Path(__file__).resolve().parent) if "__file__" in dir() else None,
    str(Path.cwd()),
):
    if _candidate and _candidate not in _sys.path:
        _sys.path.insert(0, _candidate)

# Reuse the verified HB pipeline & pricing primitives
import jd_hierarchical_bayes as hb

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "../JD_MSOM"
))
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
N_VALUES = [200, 500, 1000, 2000]
SEED = 2026
TOL = 1e-8
MAX_ITER = 2000
MARGIN_RATIO = 0.70


# ======================================================================
# HB cache layer: fit once per N, cache to disk
# ======================================================================
def get_hb_posterior_for_N(N):
    """Returns a dict sku_ID -> beta_hat from an HB fit at top-N.

    If the cached posterior CSV exists, load it. Otherwise re-fit HB
    at top-N (this is the slow path, ~10 min).
    """
    cache_path = OUT_DIR / f"jd_hb_posterior_summary_N{N}.csv"
    if cache_path.exists():
        print(f"    [cache hit] {cache_path.name}")
        df = pd.read_csv(cache_path)
        df["beta_used"] = df["beta_posterior_mean"].clip(lower=hb.BETA_FLOOR)
        return dict(zip(df["sku_ID"], df["beta_used"]))

    print(f"    [cache miss] re-fitting HB at N = {N}...")
    if not hb.HAS_PYMC:
        raise RuntimeError(
            "PyMC not installed; cannot fit HB. "
            "Run: pip install pymc arviz nutpie"
        )

    # Monkey-patch N_TOP_SKU before calling load_and_aggregate
    original_N = hb.N_TOP_SKU
    hb.N_TOP_SKU = N
    try:
        agg_n, M_n = hb.load_and_aggregate()
        idata, sku_uniques, _, sku_to_bucket = hb.fit_hierarchical_bayes(agg_n)

        # Extract posterior means (same as jd_hierarchical_bayes.main)
        post_beta = idata.posterior["beta_sku"].stack(sample=("chain", "draw"))
        beta_mean = post_beta.mean(dim="sample").values
        beta_sd = post_beta.std(dim="sample").values

        out_df = pd.DataFrame({
            "sku_ID": sku_uniques,
            "beta_posterior_mean": beta_mean,
            "beta_posterior_sd": beta_sd,
            "bucket": sku_to_bucket,
        })
        out_df.to_csv(cache_path, index=False)
        print(f"    [cache write] {cache_path.name}")
        out_df["beta_used"] = out_df["beta_posterior_mean"].clip(lower=hb.BETA_FLOOR)
        return dict(zip(out_df["sku_ID"], out_df["beta_used"]))
    finally:
        hb.N_TOP_SKU = original_N


# ======================================================================
# Per-N data prep using the existing HB load_and_aggregate at top-N
# ======================================================================
def load_and_aggregate_for_N(N):
    """Subset orders to top-N SKUs, aggregate to (day, SKU) cells,
    compute shares with M = M_MULT × peak-day inside quantity at this N."""
    original_N = hb.N_TOP_SKU
    hb.N_TOP_SKU = N
    try:
        agg, M = hb.load_and_aggregate()
        return agg, M
    finally:
        hb.N_TOP_SKU = original_N


# ======================================================================
# Pricing primitives (re-use HB module's implementations)
# ======================================================================
mci_shares = hb.mci_shares
calibrate_alpha = hb.calibrate_alpha_mci
share_jacobian = hb.share_jacobian
ebar_from_shares = hb.ebar_from_shares
gamma_iteration = hb.gamma_iteration
ms_iteration = hb.ms_iteration
newton_bn = hb.newton_bn
uniform_pricing = hb.uniform_pricing
total_profit = hb.total_profit


def run_pricing_at_N(agg, M, sku_to_beta, N):
    """For a given truncation N, run four-method pricing across all
    daily markets. Returns a per-day DataFrame."""
    # Restrict aggregation to SKUs that have HB posterior at this N
    agg_n = agg[agg["sku_ID"].isin(sku_to_beta.keys())].copy()
    agg_n["beta_hat"] = agg_n["sku_ID"].map(sku_to_beta)
    rows = []
    days = sorted(agg_n["day"].unique())
    for d in days:
        mkt = agg_n[agg_n["day"] == d].copy().reset_index(drop=True)
        if len(mkt) < 25:
            continue
        p_obs = mkt["price"].to_numpy()
        s_obs = mkt["share"].to_numpy()
        s0_obs = float(mkt["s0"].iloc[0])
        beta_vec = mkt["beta_hat"].to_numpy()
        alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
        c = MARGIN_RATIO * p_obs
        p0 = p_obs.copy()

        # Compute ebar at observed shares (share-only, MCI form)
        ebar_obs = ebar_from_shares(s_obs)

        # Run four methods
        try:
            p_g, it_g, t_g = gamma_iteration(p0, c, alpha, beta_vec, M)
            pi_g = total_profit(p_g, c, alpha, beta_vec, M)
        except Exception:
            pi_g = np.nan; it_g = -1; t_g = np.nan
        try:
            p_ms, it_ms, t_ms = ms_iteration(p0, c, alpha, beta_vec, M)
            pi_ms = total_profit(p_ms, c, alpha, beta_vec, M)
        except Exception:
            pi_ms = np.nan; it_ms = -1; t_ms = np.nan
        try:
            p_unif = uniform_pricing(c, alpha, beta_vec, M)
            pi_unif = total_profit(p_unif, c, alpha, beta_vec, M)
        except Exception:
            pi_unif = np.nan
        try:
            p_bn, it_bn, t_bn = newton_bn(p0, c, alpha, beta_vec, M)
            pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)
        except Exception:
            pi_bn = np.nan; it_bn = -1; t_bn = np.nan

        # Profit gaps relative to Newton-BN
        gap = lambda pi: (
            100.0 * (pi_bn - pi) / max(pi_bn, 1e-12)
            if not np.isnan(pi) and not np.isnan(pi_bn) else np.nan
        )

        rows.append({
            "N": N,
            "day": int(d),
            "n_skus": len(mkt),
            "S_inside": float(s_obs.sum()),
            "ebar": ebar_obs,
            "pi_BN": pi_bn,
            "pi_gamma": pi_g,
            "pi_MS": pi_ms,
            "pi_uniform": pi_unif,
            "gap_gamma": gap(pi_g),
            "gap_MS": gap(pi_ms),
            "gap_uniform": gap(pi_unif),
            "iter_gamma": it_g,
            "iter_MS": it_ms,
            "iter_BN": it_bn,
            "time_gamma": t_g,
            "time_MS": t_ms,
            "time_BN": t_bn,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("JD TOP-N ROBUSTNESS SWEEP")
    print("=" * 70)
    print(f"N values: {N_VALUES}")
    print()

    all_rows = []
    for N in N_VALUES:
        print(f"--- N = {N} ---")
        # Step 1: get HB posterior (cached if available)
        t0 = time.time()
        sku_to_beta = get_hb_posterior_for_N(N)
        t_hb = time.time() - t0
        print(f"    HB posterior ready ({len(sku_to_beta)} SKUs, {t_hb:.1f}s)")

        # Step 2: load/aggregate market data at this N
        agg_n, M_n = load_and_aggregate_for_N(N)
        print(f"    market data: {len(agg_n)} obs, M = {M_n:.0f}")

        # Step 3: pricing comparison
        t0 = time.time()
        df = run_pricing_at_N(agg_n, M_n, sku_to_beta, N)
        t_pr = time.time() - t0
        print(
            f"    pricing done ({len(df)} days, {t_pr:.1f}s) | "
            f"median ē={df['ebar'].median():.3f} | "
            f"median γ-gap={df['gap_gamma'].median():.2f}% | "
            f"γ beats uniform on {(df['gap_gamma'] < df['gap_uniform']).sum()}/{len(df)} days"
        )
        all_rows.append(df)
        print()

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(OUT_DIR / "jd_topn_sensitivity.csv", index=False)
    print(f"wrote jd_topn_sensitivity.csv ({len(full_df)} rows)")

    # ==================================================================
    # Summary table
    # ==================================================================
    summary = (
        full_df.groupby("N")
        .agg(
            n_days=("day", "count"),
            mean_S=("S_inside", "mean"),
            median_ebar=("ebar", "median"),
            mean_ebar=("ebar", "mean"),
            n_ebar_above_03=("ebar", lambda x: int((x > 0.3).sum())),
            median_gap_gamma=("gap_gamma", "median"),
            mean_gap_gamma=("gap_gamma", "mean"),
            median_gap_uniform=("gap_uniform", "median"),
            mean_gap_uniform=("gap_uniform", "mean"),
            median_iter_gamma=("iter_gamma", "median"),
            median_iter_MS=("iter_MS", "median"),
        )
        .reset_index()
    )
    win_counts = (
        full_df.groupby("N")
        .apply(lambda g: int((g["gap_gamma"] < g["gap_uniform"]).sum()))
        .rename("gamma_dominates_count")
    )
    summary = summary.merge(win_counts, on="N")
    summary.to_csv(OUT_DIR / "jd_topn_summary.csv", index=False)
    print()
    print(summary.round(3).to_string(index=False))
    print(f"\nwrote jd_topn_summary.csv")

    # ==================================================================
    # Plot: ebar and gamma-gap as functions of N
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(summary["N"], summary["median_ebar"], "o-",
            color="tab:blue", linewidth=2, markersize=8, label="Median ē")
    ax.set_xlabel("Top-N SKU truncation")
    ax.set_ylabel(r"Median $\bar e$ across days")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Diagonal-dominance slack vs catalog size")

    ax = axes[1]
    ax.plot(summary["N"], summary["median_gap_gamma"], "o-",
            color="tab:blue", linewidth=2, markersize=8,
            label="γ-equalization")
    ax.plot(summary["N"], summary["median_gap_uniform"], "s-",
            color="tab:orange", linewidth=2, markersize=8,
            label="Uniform markup")
    ax.set_xlabel("Top-N SKU truncation")
    ax.set_ylabel("Median profit gap to BN (%)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Profit gap vs catalog size")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_topn_sensitivity.png", dpi=150)
    plt.close()
    print("wrote jd_topn_sensitivity.png")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
