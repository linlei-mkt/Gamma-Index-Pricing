"""
JD GMV-floor sensitivity sweep for §10 of the paper.

Sweeps the GMV floor multiplier across {1.05, 1.10, 1.15, 1.20, 1.25}
on the same hierarchical-Bayes MCI demand and the same 31 daily JD
markets used in jd_gmv_constrained.py. For each floor level we report:

  - median tuned γ⋆ across days (and IQR)
  - median profit gap of γ-tuned vs. constrained BN
  - median profit gap of uniform-tuned vs. constrained BN
  - dominance count: γ-tuned beats uniform-tuned on how many days
  - feasibility count: how many days admit a feasible solution

This addresses the reviewer concern that the §10 main result is pinned
to a single 15% floor. The expected pattern: as the floor tightens
(higher multiplier), tuned γ⋆ becomes more negative, both methods'
profit gaps grow, but the γ-vs-uniform gap should also grow because
elasticity heterogeneity is more valuable under tighter constraints.

Outputs
-------
  - jd_gmv_floor_sensitivity.csv     per-(floor, day) results
  - jd_gmv_floor_summary.csv         5×k summary across floor levels
  - jd_gmv_floor_sensitivity.png     summary plot

Usage
-----
  JD_DATA_DIR=/path/to/csvs python3 jd_gmv_floor_sensitivity.py

Requires jd_hb_posterior_summary.csv (run jd_hierarchical_bayes.py first).

Runtime: ~5-10 minutes on a laptop (5 floor levels × 31 days × 4 methods).
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# Make sure the directory containing this script is on sys.path so
# we can import the local jd_gmv_constrained module regardless of
# how the script is invoked.
import sys as _sys
for _candidate in (
    "/content/drive/MyDrive/JD_gamma",
    str(Path(__file__).resolve().parent) if "__file__" in dir() else None,
    str(Path.cwd()),
):
    if _candidate and _candidate not in _sys.path:
        _sys.path.insert(0, _candidate)

# Reuse the verified primitives from the baseline GMV-constrained script
from jd_gmv_constrained import (
    load_hb_posterior, load_and_aggregate, calibrate_alpha,
    newton_bn, constrained_bn_floor, gamma_iteration,
    uniform_tuned, tune_gamma_star,
    total_profit, total_revenue,
    MARGIN_RATIO,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
FLOOR_MULTS = [1.05, 1.10, 1.15, 1.20, 1.25]
SEED = 2026


def run_one_floor(agg, M, sku_to_beta, floor_mult):
    """Run the four-method comparison at a single floor multiplier."""
    rows = []
    days = sorted(agg["day"].unique())
    for d in days:
        mkt = agg[agg["day"] == d].copy().reset_index(drop=True)
        if len(mkt) < 50:
            continue
        p_obs = mkt["price"].to_numpy()
        s_obs = mkt["share"].to_numpy()
        s0_obs = float(mkt["s0"].iloc[0])
        beta_vec = mkt["beta_hat"].to_numpy()
        alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
        c = MARGIN_RATIO * p_obs
        p0 = p_obs.copy()
        R_observed = float(np.sum(p_obs * s_obs) * M)
        R_target = floor_mult * R_observed

        # (a) Constrained BN (ground truth)
        try:
            p_cbn, mu_star = constrained_bn_floor(
                p0, c, alpha, beta_vec, M, R_target
            )
            pi_cbn = total_profit(p_cbn, c, alpha, beta_vec, M)
            R_cbn = total_revenue(p_cbn, alpha, beta_vec, M)
            cbn_meets = R_cbn >= R_target * 0.999
        except Exception:
            pi_cbn = np.nan
            mu_star = np.nan
            cbn_meets = False

        # (b) γ-iteration with γ⋆ tuned to floor
        try:
            gamma_star, iters_g = tune_gamma_star(
                p0, c, alpha, beta_vec, M, R_target
            )
            p_g, _ = gamma_iteration(p0, c, alpha, beta_vec, M, gamma_star)
            pi_g = total_profit(p_g, c, alpha, beta_vec, M)
            R_g = total_revenue(p_g, alpha, beta_vec, M)
            g_meets = R_g >= R_target * 0.999
        except Exception:
            pi_g = np.nan
            gamma_star = np.nan
            R_g = np.nan
            g_meets = False

        # (c) Uniform markup tuned to floor
        try:
            p_unif, m_star, ok = uniform_tuned(
                c, alpha, beta_vec, M, R_target
            )
            pi_unif = total_profit(p_unif, c, alpha, beta_vec, M)
            R_unif = total_revenue(p_unif, alpha, beta_vec, M)
            unif_meets = ok and (R_unif >= R_target * 0.999)
        except Exception:
            pi_unif = np.nan
            m_star = np.nan
            R_unif = np.nan
            unif_meets = False

        # Profit gaps to constrained BN
        gap_g = (
            100.0 * (pi_cbn - pi_g) / max(pi_cbn, 1e-12)
            if not np.isnan(pi_g) and not np.isnan(pi_cbn) else np.nan
        )
        gap_unif = (
            100.0 * (pi_cbn - pi_unif) / max(pi_cbn, 1e-12)
            if not np.isnan(pi_unif) and not np.isnan(pi_cbn) else np.nan
        )

        rows.append({
            "floor_mult": floor_mult,
            "day": int(d),
            "R_observed": R_observed,
            "R_target": R_target,
            "mu_star_cbn": mu_star,
            "gamma_star_tuned": gamma_star,
            "m_star_unif": m_star,
            "pi_cbn": pi_cbn,
            "pi_gamma_tuned": pi_g,
            "pi_unif_tuned": pi_unif,
            "gap_gamma_pct": gap_g,
            "gap_unif_pct": gap_unif,
            "cbn_meets_floor": cbn_meets,
            "gamma_meets_floor": g_meets,
            "unif_meets_floor": unif_meets,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("JD GMV-FLOOR SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"Floor multipliers: {FLOOR_MULTS}")
    print()

    print("[1/3] Loading data and HB posterior...")
    agg, M = load_and_aggregate()
    sku_to_beta = load_hb_posterior()
    agg = agg[agg["sku_ID"].isin(sku_to_beta.keys())].copy()
    agg["beta_hat"] = agg["sku_ID"].map(sku_to_beta)
    print(f"    {len(agg)} obs, M = {M:.0f}, {len(sku_to_beta)} SKUs with HB β̂")
    print(f"    days: {sorted(agg['day'].unique())[:5]}... ({agg['day'].nunique()} total)")

    print()
    print("[2/3] Running per-floor sweep...")
    all_rows = []
    for fm in FLOOR_MULTS:
        print(f"  --- floor_mult = {fm:.2f} ---")
        t0 = time.time()
        df = run_one_floor(agg, M, sku_to_beta, fm)
        elapsed = time.time() - t0
        n_days = len(df)
        # Per-floor diagnostics
        med_gamma_star = df["gamma_star_tuned"].median()
        med_gap_g = df["gap_gamma_pct"].median()
        med_gap_u = df["gap_unif_pct"].median()
        n_g_dom = int((df["gap_gamma_pct"] < df["gap_unif_pct"]).sum())
        n_cbn_feas = int(df["cbn_meets_floor"].sum())
        print(
            f"    {n_days} days, t={elapsed:.1f}s | "
            f"median γ⋆={med_gamma_star:+.3f} | "
            f"gap_γ={med_gap_g:.2f}%, gap_unif={med_gap_u:.2f}% | "
            f"γ wins {n_g_dom}/{n_days} | cBN feasible {n_cbn_feas}/{n_days}"
        )
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(OUT_DIR / "jd_gmv_floor_sensitivity.csv", index=False)
    print(f"\n    wrote jd_gmv_floor_sensitivity.csv ({len(full_df)} rows)")

    # ==================================================================
    # Summary table: one row per floor level
    # ==================================================================
    print()
    print("[3/3] Building summary table and plot...")

    summary = (
        full_df.groupby("floor_mult")
        .agg(
            n_days=("day", "count"),
            cbn_feasible=("cbn_meets_floor", "sum"),
            gamma_feasible=("gamma_meets_floor", "sum"),
            unif_feasible=("unif_meets_floor", "sum"),
            median_gamma_star=("gamma_star_tuned", "median"),
            iqr_low_gamma_star=("gamma_star_tuned", lambda x: x.quantile(0.25)),
            iqr_hi_gamma_star=("gamma_star_tuned", lambda x: x.quantile(0.75)),
            mean_gap_gamma=("gap_gamma_pct", "mean"),
            median_gap_gamma=("gap_gamma_pct", "median"),
            mean_gap_unif=("gap_unif_pct", "mean"),
            median_gap_unif=("gap_unif_pct", "median"),
        )
        .reset_index()
    )
    # γ wins (vs uniform) per floor
    win_counts = (
        full_df.groupby("floor_mult")
        .apply(lambda g: int((g["gap_gamma_pct"] < g["gap_unif_pct"]).sum()))
        .rename("gamma_dominates_count")
    )
    summary = summary.merge(win_counts, on="floor_mult")
    # Average gap difference (uniform minus gamma)
    summary["mean_gap_advantage_pp"] = (
        summary["mean_gap_unif"] - summary["mean_gap_gamma"]
    )
    summary["median_gap_advantage_pp"] = (
        summary["median_gap_unif"] - summary["median_gap_gamma"]
    )

    summary.to_csv(OUT_DIR / "jd_gmv_floor_summary.csv", index=False)
    print()
    print(summary.round(3).to_string(index=False))
    print(f"\n    wrote jd_gmv_floor_summary.csv")

    # ==================================================================
    # Plot: profit gap vs floor multiplier, both methods
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(summary["floor_mult"], summary["median_gap_gamma"], "o-",
            color="tab:blue", linewidth=2, markersize=8, label="γ tuned")
    ax.plot(summary["floor_mult"], summary["median_gap_unif"], "s-",
            color="tab:orange", linewidth=2, markersize=8, label="Uniform tuned")
    ax.set_xlabel("GMV floor multiplier (× observed revenue)")
    ax.set_ylabel("Median profit gap to constrained BN (%)")
    ax.set_title("Profit gap by floor level")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(summary["floor_mult"], summary["median_gamma_star"], "o-",
            color="tab:green", linewidth=2, markersize=8)
    ax.fill_between(
        summary["floor_mult"],
        summary["iqr_low_gamma_star"],
        summary["iqr_hi_gamma_star"],
        color="tab:green", alpha=0.2,
        label="IQR across days",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("GMV floor multiplier (× observed revenue)")
    ax.set_ylabel(r"Tuned $\gamma^\star$ (median across days)")
    ax.set_title("Shadow price by floor level")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_gmv_floor_sensitivity.png", dpi=150)
    plt.close()
    print("    wrote jd_gmv_floor_sensitivity.png")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
