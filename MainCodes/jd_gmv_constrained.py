"""
JD GMV-constrained pricing demo using hierarchical-Bayes β̂_i.

Purpose
-------
At γ* = 0 the γ-equalization rule reduces to classical Dorfman-Steiner
monopoly pricing. The distinctive contribution of γ-equalization is the
γ* ≠ 0 case: when the retailer faces a binding revenue (GMV) constraint,
γ* is the shadow price of that constraint, and γ-equalization with a
tuned γ* implements constrained BN pricing via a diagonal-only iteration.

This script demonstrates that on real JD data:
    1. Impose a GMV floor: R(p) ≥ R_target on each day's market, where
       R_target is a multiple of observed day-t revenue (default 1.15).
    2. Solve exact constrained BN by tuning the Lagrange multiplier μ so
       that the effective-cost BN solution c_eff = c/(1+μ) satisfies the
       floor with equality. (This is the textbook Lagrangian transform:
       revenue-floor constrained profit-max ≡ unconstrained profit-max
       at scaled costs.)
    3. Solve γ-iteration at γ* = 0 (classical Lerner — ignores constraint).
    4. Solve γ-iteration with γ* tuned by binary search to match the
       same GMV floor.
    5. Report profit, revenue attained, final γ*, solver time.

Expected result: (2) and (4) give nearly identical profits and both
satisfy the GMV floor; (3) attains higher profit but VIOLATES the
floor; (1) [reported separately as 'no-constraint benchmark'] is the
theoretical max profit ignoring revenue commitments.

Uses HB-posterior β̂_i from jd_hb_posterior_summary.csv (must be run
first via jd_hierarchical_bayes.py).

Outputs
-------
  - jd_gmv_pricing_comparison.csv   per-day results
  - jd_gmv_profit_vs_revenue.png    scatter of profit vs revenue achieved
  - jd_gmv_gamma_star_distribution.png   distribution of tuned γ* across days

Usage
-----
  JD_DATA_DIR=/path/to/csvs python3 jd_gmv_constrained.py

Runtime: ~60 seconds on a laptop.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root, brentq

warnings.simplefilter("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "/Users/linlei/Library/Application Support/Claude/local-agent-mode-sessions/"
    "28ce55e3-2aeb-47b1-a159-176e9d6a9dbf/0f00e7ba-81f1-4095-94e5-73a365a8f51b/"
    "local_c02dff0e-7360-4888-8cca-0a64aed3b4e1/uploads"
))
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 500
M_MULT = 3.0
MARGIN_RATIO = 0.70
GMV_TARGET_MULT = 1.15      # GMV floor = 1.15 × observed day-t revenue
BETA_FLOOR = 1.2             # avoid degenerate γ-iteration updates
TOL = 1e-8
MAX_ITER = 2000


# ======================================================================
# Load data + HB elasticities
# ======================================================================
def load_hb_posterior():
    """Load per-SKU β̂_i from jd_hb_posterior_summary.csv."""
    path = OUT_DIR / "jd_hb_posterior_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run jd_hierarchical_bayes.py first."
        )
    df = pd.read_csv(path)
    # Clip betas below the γ-iteration floor (otherwise denominator blows up)
    df["beta_used"] = df["beta_posterior_mean"].clip(lower=BETA_FLOOR)
    return dict(zip(df["sku_ID"], df["beta_used"]))


def load_and_aggregate():
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
    peak = agg.groupby("day")["qty"].sum().max()
    M = M_MULT * peak
    agg["share"] = agg["qty"] / M
    daily_Q = agg.groupby("day")["qty"].sum().rename("Q_in")
    s0 = (1.0 - daily_Q / M).rename("s0")
    agg = agg.merge(s0, on="day")
    return agg, M


# ======================================================================
# MCI primitives
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


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


def total_revenue(p, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum(p * s) * M)


# ======================================================================
# Unconstrained BN (Newton)
# ======================================================================
def newton_bn(p0, c, alpha, beta, M):
    """Unconstrained BN via MS-ζ iteration (robust) + optional Newton polish.
    The MS iteration is a Banach contraction under MCI so it always
    converges from a sensible start; Newton-Krylov sometimes fails on
    awkward share configurations."""
    p = np.maximum(p0.copy(), c * 1.0001)
    # MS iteration until near-converged
    for _ in range(500):
        s, _ = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        p_new = np.maximum(c - (s + Gamma @ (p - c)) / diag, c * 1.0001)
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    return p


# ======================================================================
# Constrained BN: revenue floor via Lagrangian transform
# ======================================================================
def constrained_bn_floor(p0, c, alpha, beta, M, R_target):
    """Constrained BN: max profit s.t. revenue ≥ R_target.

    Lagrangian FOC:  ∂π/∂p + μ ∂R/∂p = 0
                     s + Ω(p - c) + μ(s + Ωp) = 0
                     (1 + μ)s + Ω[(1 + μ)p - c] = 0
                     s + Ω(p - c/(1 + μ)) = 0     [divide by (1 + μ)]

    So the constrained optimum is the unconstrained BN at scaled
    costs c/(1 + μ). Binary-search μ ≥ 0 to hit R = R_target with equality.
    """
    # First check: does unconstrained BN already satisfy the floor?
    p_unc = newton_bn(p0, c, alpha, beta, M)
    R_unc = total_revenue(p_unc, alpha, beta, M)
    if R_unc >= R_target:
        return p_unc, 0.0  # constraint slack

    def residual(mu):
        c_eff = c / (1.0 + mu)
        p_mu = newton_bn(p0, c_eff, alpha, beta, M)
        return total_revenue(p_mu, alpha, beta, M) - R_target

    # Find μ > 0 where residual = 0
    try:
        mu_star = brentq(residual, 1e-6, 100.0, xtol=1e-6, maxiter=50)
    except ValueError:
        # If still can't hit target even at huge μ, return best effort
        mu_star = 100.0
    c_eff = c / (1.0 + mu_star)
    p_bn = newton_bn(p0, c_eff, alpha, beta, M)
    return p_bn, mu_star


# ======================================================================
# γ-iteration at any γ*
# ======================================================================
def gamma_iteration(p0, c, alpha, beta, M, gamma_star):
    """p_i ← c_i / (1 - γ* - (1 - γ*)/|η_i(p)|)."""
    p = p0.copy()
    for k in range(MAX_ITER):
        s, _ = mci_shares(p, alpha, beta, M)
        eta = np.maximum(beta * (1.0 - s), 1.01)
        denom = 1.0 - gamma_star - (1.0 - gamma_star) / eta
        # Guard: denom must be positive for p > 0
        denom = np.maximum(denom, 0.01)
        p_new = np.maximum(c / denom, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new
            break
        p = p_new
    return p, k + 1


def uniform_tuned(c, alpha, beta, M, R_target):
    """Single scalar markup m ∈ (0,1) tuned by binary search so that
    total revenue hits R_target. p_i = c_i / (1 - m) for all products."""
    def residual(m):
        p = c / (1.0 - m)
        return total_revenue(p, alpha, beta, M) - R_target
    try:
        r_lo = residual(0.01)
        r_hi = residual(0.99)
        if r_lo * r_hi > 0:
            return c / (1.0 - 0.5), 0.5, False
        m_star = brentq(residual, 0.01, 0.99, xtol=1e-6, maxiter=50)
    except Exception:
        return c / (1.0 - 0.5), 0.5, False
    return c / (1.0 - m_star), m_star, True


def tune_gamma_star(p0, c, alpha, beta, M, R_target, gamma_bounds=(-5.0, 0.99)):
    """Binary-search γ* so γ-iteration FP hits revenue R_target.
    Lower γ* → lower markups → lower prices → higher volume → higher R.
    """
    def residual(gs):
        p_gs, _ = gamma_iteration(p0, c, alpha, beta, M, gs)
        return total_revenue(p_gs, alpha, beta, M) - R_target

    lo, hi = gamma_bounds
    try:
        r_lo = residual(lo)
        r_hi = residual(hi)
        if r_lo * r_hi > 0:
            # Can't bracket — constraint infeasible or slack
            # Pick whichever endpoint is closer to zero residual
            if abs(r_lo) < abs(r_hi):
                return lo, float("nan")
            return hi, float("nan")
        gamma_star = brentq(residual, lo, hi, xtol=1e-4, maxiter=50)
    except Exception:
        return 0.0, float("nan")
    p_g, iters = gamma_iteration(p0, c, alpha, beta, M, gamma_star)
    return gamma_star, iters


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("JD GMV-constrained pricing demo with hierarchical-Bayes β̂")
    print("=" * 70)
    print(f"GMV floor: R(p) ≥ {GMV_TARGET_MULT}× observed daily revenue")
    print()

    print("[1/3] Loading data and HB posterior...")
    agg, M = load_and_aggregate()
    sku_to_beta = load_hb_posterior()
    # Restrict to SKUs present in HB posterior
    agg = agg[agg["sku_ID"].isin(sku_to_beta.keys())].copy()
    agg["beta_hat"] = agg["sku_ID"].map(sku_to_beta)
    print(f"    {len(agg)} obs, M = {M:.0f}, {len(sku_to_beta)} SKUs with HB β̂")

    print("[2/3] Running per-day comparison...")
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
        # Observed revenue on this day, and target above it
        R_observed = float(np.sum(p_obs * s_obs) * M)
        R_target = GMV_TARGET_MULT * R_observed

        # (a) Unconstrained BN
        t0 = time.perf_counter()
        p_unc = newton_bn(p0, c, alpha, beta_vec, M)
        t_unc = time.perf_counter() - t0

        # (b) Constrained BN (ground truth)
        t0 = time.perf_counter()
        p_cbn, mu_star = constrained_bn_floor(
            p_unc.copy(), c, alpha, beta_vec, M, R_target
        )
        t_cbn = time.perf_counter() - t0

        # (c) γ-iteration at γ* = 0 (classical Lerner, ignores constraint)
        t0 = time.perf_counter()
        p_g0, it_g0 = gamma_iteration(p0, c, alpha, beta_vec, M, 0.0)
        t_g0 = time.perf_counter() - t0

        # (d) γ-iteration with γ* tuned to match GMV target
        t0 = time.perf_counter()
        gamma_star_tuned, _ = tune_gamma_star(
            p0, c, alpha, beta_vec, M, R_target
        )
        p_gtuned, it_gtuned = gamma_iteration(
            p0, c, alpha, beta_vec, M, gamma_star_tuned
        )
        t_gtuned = time.perf_counter() - t0

        # (e) Uniform markup tuned to match GMV target
        t0 = time.perf_counter()
        p_utuned, m_star, u_ok = uniform_tuned(c, alpha, beta_vec, M, R_target)
        t_utuned = time.perf_counter() - t0

        # Profits and revenues
        pi_unc  = total_profit(p_unc,  c, alpha, beta_vec, M)
        pi_cbn  = total_profit(p_cbn,  c, alpha, beta_vec, M)
        pi_g0   = total_profit(p_g0,   c, alpha, beta_vec, M)
        pi_gtuned = total_profit(p_gtuned, c, alpha, beta_vec, M)
        pi_utuned = total_profit(p_utuned, c, alpha, beta_vec, M)
        R_unc   = total_revenue(p_unc,  alpha, beta_vec, M)
        R_cbn   = total_revenue(p_cbn,  alpha, beta_vec, M)
        R_g0    = total_revenue(p_g0,   alpha, beta_vec, M)
        R_gtuned = total_revenue(p_gtuned, alpha, beta_vec, M)
        R_utuned = total_revenue(p_utuned, alpha, beta_vec, M)

        # Gap (vs constrained BN, which is the honest benchmark under the floor)
        gap = lambda pi: max(0.0, (pi_cbn - pi) / pi_cbn) if pi_cbn > 0 else np.nan

        rows.append({
            "day": int(d),
            "n_products": len(mkt),
            "R_observed": R_observed,
            "R_target": R_target,
            "mu_lagrangian": mu_star,
            "gamma_star_tuned": gamma_star_tuned,
            "uniform_m_tuned": m_star,
            "pi_unc_BN": pi_unc,       "R_unc_BN": R_unc,
            "pi_cbn": pi_cbn,          "R_cbn": R_cbn,
            "pi_gamma0": pi_g0,        "R_gamma0": R_g0,
            "pi_gamma_tuned": pi_gtuned, "R_gamma_tuned": R_gtuned,
            "pi_uniform_tuned": pi_utuned, "R_uniform_tuned": R_utuned,
            "gap_gamma0_vs_cbn": gap(pi_g0),
            "gap_gamma_tuned_vs_cbn": gap(pi_gtuned),
            "gap_uniform_tuned_vs_cbn": gap(pi_utuned),
            "gamma_beats_uniform_constrained": pi_gtuned > pi_utuned,
            "gamma0_violates_floor": R_g0 < R_target,
            "gamma_tuned_meets_floor": R_gtuned >= R_target * 0.999,
            "uniform_tuned_meets_floor": R_utuned >= R_target * 0.999,
            "time_cbn": t_cbn,
            "time_gamma_tuned": t_gtuned,
            "time_uniform_tuned": t_utuned,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "jd_gmv_pricing_comparison.csv", index=False)

    # ==================================================================
    # Figures
    # ==================================================================
    print("[3/3] Generating figures...")

    # Profit vs revenue scatter, with R_target line
    fig, ax = plt.subplots(figsize=(7, 5))
    for col_r, col_p, label, marker, color in [
        ("R_unc_BN", "pi_unc_BN", "unconstrained BN (ignores floor)", "x", "grey"),
        ("R_cbn", "pi_cbn", "constrained BN (ground truth)", "o", "tab:red"),
        ("R_gamma0", "pi_gamma0", "γ at γ*=0 (classical Lerner)", "s", "tab:green"),
        ("R_uniform_tuned", "pi_uniform_tuned", "uniform markup (tuned)", "^", "tab:orange"),
        ("R_gamma_tuned", "pi_gamma_tuned", "γ with γ* tuned", "D", "tab:blue"),
    ]:
        ax.scatter(df[col_r], df[col_p], label=label, marker=marker,
                   color=color, s=30, alpha=0.8)
    # R_target lines per day (connecting the dots would be busy, use median)
    ax.axvline(df["R_target"].median(), color="black", linestyle="--", alpha=0.4,
               label=f"median R_target")
    ax.set_xlabel("revenue R(p)")
    ax.set_ylabel("profit π(p)")
    ax.set_title(f"GMV-constrained pricing (floor = {GMV_TARGET_MULT}× observed R)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_gmv_profit_vs_revenue.png", dpi=150)
    plt.close()

    # γ* distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = df["gamma_star_tuned"].dropna()
    ax.hist(valid, bins=20, color="tab:blue", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="red", linestyle="--", label="γ* = 0 (Lerner)")
    ax.axvline(valid.median(), color="black", linestyle=":",
               label=f"median γ* = {valid.median():.3f}")
    ax.set_xlabel("tuned γ* (shadow price of GMV floor)")
    ax.set_ylabel("number of days")
    ax.set_title(f"Distribution of γ* across days, GMV floor = {GMV_TARGET_MULT}× observed R")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_gmv_gamma_star_distribution.png", dpi=150)
    plt.close()

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 70)
    print(f"GMV floor = {GMV_TARGET_MULT} × observed daily revenue")
    print(f"Number of days: {len(df)}")
    print()
    print("Profit and revenue results (mean across days, rounded):")
    print(df[["pi_unc_BN", "pi_cbn", "pi_gamma0", "pi_gamma_tuned",
             "R_unc_BN", "R_cbn", "R_gamma0", "R_gamma_tuned",
             "R_target"]].mean().round(1).to_string())
    print()
    print("Tuned γ* (shadow price of GMV floor):")
    print(df["gamma_star_tuned"].describe().round(4).to_string())
    print()
    print("Days where γ*=0 (classical Lerner) VIOLATES the GMV floor: "
          f"{df['gamma0_violates_floor'].sum()}/{len(df)}")
    print("Days where γ* tuned correctly meets the floor: "
          f"{df['gamma_tuned_meets_floor'].sum()}/{len(df)}")
    print()
    print("Profit gap to constrained BN (ground truth under the floor):")
    print(f"  γ at γ*=0 (classical Lerner):   "
          f"mean {df['gap_gamma0_vs_cbn'].mean()*100:6.3f}%  median "
          f"{df['gap_gamma0_vs_cbn'].median()*100:6.3f}%  (VIOLATES floor)")
    print(f"  uniform markup (tuned):         "
          f"mean {df['gap_uniform_tuned_vs_cbn'].mean()*100:6.3f}%  median "
          f"{df['gap_uniform_tuned_vs_cbn'].median()*100:6.3f}%  (meets floor, no elasticity info)")
    print(f"  γ with γ* tuned:                "
          f"mean {df['gap_gamma_tuned_vs_cbn'].mean()*100:6.3f}%  median "
          f"{df['gap_gamma_tuned_vs_cbn'].median()*100:6.3f}%  (meets floor, own-elasticity only)")
    print()
    print(f"γ-tuned beats uniform-tuned: "
          f"{df['gamma_beats_uniform_constrained'].sum()}/{len(df)} days "
          f"({df['gamma_beats_uniform_constrained'].mean()*100:.1f}%)")
    print()
    print("Wall-clock per solve:")
    print(f"  constrained BN (Newton/MS): mean {df['time_cbn'].mean()*1000:6.1f} ms")
    print(f"  γ with γ* tuning:           mean {df['time_gamma_tuned'].mean()*1000:6.1f} ms")
    print(f"  uniform markup tuning:      mean {df['time_uniform_tuned'].mean()*1000:6.1f} ms")
    print()
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
