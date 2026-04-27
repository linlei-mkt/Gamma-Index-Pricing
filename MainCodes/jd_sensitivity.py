"""
JD real-data SENSITIVITY SWEEP.

Runs the full pricing experiment across a grid of two modeling choices
that are NOT pinned down by the data:

    M_mult  ∈ {1.5, 2, 3, 5, 7}    potential market size = M_mult × peak daily inside qty
    margin  ∈ {0.4, 0.5, 0.6, 0.7, 0.8}  marginal cost c = margin × observed price

so 5 × 5 = 25 calibrations.  For each, we re-estimate MCI demand (10
price-decile buckets), recompute ebar, and solve γ / MS2011 / uniform /
Newton BN on every day.  Reports how the headline numbers move across
the grid, so a reviewer can see the main conclusions are robust to
these modeling choices (or precisely where they break).

Outputs (in /Users/linlei/Downloads/Gamma/):
  - jd_sensitivity_results.csv         one row per (M_mult, margin, metric)
  - jd_sensitivity_summary.csv         pivot table: γ-gap across M × margin
  - jd_sensitivity_heatmap_gamma.png   γ median profit gap (%) by (M, margin)
  - jd_sensitivity_heatmap_uniform.png uniform profit gap (%) by (M, margin)
  - jd_sensitivity_heatmap_ebar.png    median ebar by (M, margin)
  - jd_sensitivity_heatmap_speedup.png γ-vs-MS wall-clock speedup by (M, margin)

Runtime: ~5 minutes on a laptop (~8-12 sec per calibration).

To run:
  pip install pandas numpy scipy statsmodels matplotlib
  JD_DATA_DIR=/path/to/csvs python3 jd_sensitivity.py

If you want a finer grid, edit M_MULTS and MARGINS below.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize_scalar, root

warnings.simplefilter("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    ""
))
OUT_DIR = SCRIPT_DIR

# ==================  GRID  ==================
M_MULTS = [1.5, 2.0, 3.0, 5.0, 7.0]
MARGINS = [0.4, 0.5, 0.6, 0.7, 0.8]
# Change to e.g. [2.0, 3.0, 5.0] / [0.5, 0.7] for a quicker sanity check.

# ==================  Fixed config  ==================
N_TOP_SKU = 500
N_DECILES = 10
BETA_FLOOR = 1.2
TOL = 1e-8
MAX_ITER = 2000
SEED = 2026
np.random.seed(SEED)

# ======================================================================
# Shared functions (mirror jd_experiment.py, kept here for
# self-contained replication of this script)
# ======================================================================
def load_orders_and_sku():
    orders = pd.read_csv(
        DATA_DIR / "JD_order_data.csv",
        usecols=["sku_ID", "order_date", "quantity", "final_unit_price"],
        dtype={"sku_ID": "string"},
        parse_dates=["order_date"],
    )
    sku = pd.read_csv(
        DATA_DIR / "JD_sku_data.csv",
        usecols=["sku_ID", "type", "brand_ID"],
        dtype={"sku_ID": "string", "brand_ID": "string"},
    )
    return orders, sku


def aggregate_daily(orders, sku):
    """Top-N SKUs, aggregated to (day, sku_ID) with quantity and qty-weighted price."""
    top_ids = orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index.tolist()
    o = orders[orders["sku_ID"].isin(top_ids)].merge(sku, on="sku_ID", how="left")
    o["day"] = o["order_date"].dt.day
    agg = (
        o.assign(rev=o["final_unit_price"] * o["quantity"])
         .groupby(["day", "sku_ID"], as_index=False)
         .agg(qty=("quantity", "sum"), rev=("rev", "sum"))
    )
    agg["price"] = agg["rev"] / agg["qty"]
    agg = agg.merge(sku[["sku_ID", "type", "brand_ID"]], on="sku_ID", how="left")
    agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
    agg["type"] = agg["type"].astype(int)
    return agg


def assign_price_deciles(agg):
    sku_mean_p = agg.groupby("sku_ID")["price"].mean()
    bucket = pd.qcut(sku_mean_p.rank(method="first"), q=N_DECILES, labels=False).astype(int)
    return agg.merge(bucket.rename("bucket"), left_on="sku_ID", right_index=True, how="left")


def estimate_mci_by_decile(agg, M):
    """Return (dict bucket -> β) given a specific potential market size M."""
    work = agg.copy()
    work["share"] = work["qty"] / M
    daily_inside = work.groupby("day")["qty"].sum().rename("Q_in")
    s0 = 1.0 - daily_inside / M
    work = work.merge(s0.rename("s0"), on="day")
    work["y"] = np.log(work["share"]) - np.log(work["s0"])
    work["log_price"] = np.log(work["price"])
    X = pd.get_dummies(
        work[["sku_ID", "day"]].astype(str),
        columns=["sku_ID", "day"], drop_first=True, dtype=float,
    )
    for b in range(N_DECILES):
        X[f"logp_b{b}"] = work["log_price"] * (work["bucket"] == b).astype(float)
    X = sm.add_constant(X)
    y = work["y"].astype(float)
    mod = sm.OLS(y, X).fit()
    betas = {}
    for b in range(N_DECILES):
        col = f"logp_b{b}"
        if col in mod.params.index:
            betas[b] = -mod.params[col]
    for b in list(betas.keys()):
        if betas[b] < BETA_FLOOR:
            betas[b] = BETA_FLOOR
    return betas, mod.rsquared


# ==================  MCI pricing primitives  ==================
def mci_shares(p, alpha, beta, M):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def calibrate_alpha(p_obs, s_obs, s0_obs, beta):
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def share_jacobian(p, s, beta):
    """Ω_{ij} = ∂s_j/∂p_i under MCI."""
    u = beta * s / p
    Om = np.outer(u, s)
    np.fill_diagonal(Om, -u * (1.0 - s))
    return Om


def ebar_from_shares(s):
    S = s.sum()
    return float(np.max((S - s) / (1.0 - s)))


def gamma_iteration(p0, c, alpha, beta, M, gamma_star=0.0):
    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(MAX_ITER):
        s, _ = mci_shares(p, alpha, beta, M)
        eta = beta * (1.0 - s)
        eta_safe = np.maximum(eta, 1.01)
        p_new = c / (1.0 - gamma_star - (1.0 - gamma_star) / eta_safe)
        p_new = np.maximum(p_new, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new
            break
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
        p_new = c - (s + Gamma @ (p - c)) / diag
        p_new = np.maximum(p_new, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new
            break
        p = p_new
    return p, k + 1, time.perf_counter() - t0


def newton_bn(p0, c, alpha, beta, M):
    def F(p):
        p_pos = np.maximum(p, c * 1.0001)
        s, _ = mci_shares(p_pos, alpha, beta, M)
        Om = share_jacobian(p_pos, s, beta)
        return s + Om @ (p_pos - c)
    t0 = time.perf_counter()
    sol = root(F, p0, method="krylov", tol=1e-10, options={"maxiter": 500})
    return np.maximum(sol.x, c * 1.0001), sol.nit or -1, time.perf_counter() - t0


def uniform_pricing(c, alpha, beta, M):
    def neg_profit(m):
        if not (0.0 < m < 0.999):
            return 1e18
        p = c / (1.0 - m)
        s, _ = mci_shares(p, alpha, beta, M)
        return -np.sum((p - c) * s) * M
    t0 = time.perf_counter()
    res = minimize_scalar(neg_profit, bounds=(0.001, 0.999), method="bounded",
                          options={"xatol": 1e-8})
    return c / (1.0 - res.x), 50, time.perf_counter() - t0


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# Sensitivity sweep
# ======================================================================
def run_one_calibration(agg, M_mult, margin):
    """Run the full pricing experiment for one (M_mult, margin) pair.
    Returns a dict with aggregate statistics across days."""
    peak_inside = agg.groupby("day")["qty"].sum().max()
    M = M_mult * peak_inside
    # Re-estimate demand with this M
    betas, R2 = estimate_mci_by_decile(agg, M)
    # Compute per-day shares with this M and run pricing
    results = []
    for day in sorted(agg["day"].unique()):
        mkt = agg[agg["day"] == day].copy()
        if len(mkt) < 50:
            continue
        p_obs = mkt["price"].to_numpy()
        q_obs = mkt["qty"].to_numpy()
        s_obs = q_obs / M
        s0_obs = 1.0 - s_obs.sum()
        if s0_obs <= 0:
            continue
        beta_vec = mkt["bucket"].map(betas).to_numpy()
        alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
        c = margin * p_obs
        p0 = p_obs.copy()
        ebar_obs = ebar_from_shares(s_obs)

        p_g, it_g, t_g = gamma_iteration(p0, c, alpha, beta_vec, M)
        p_m, it_m, t_m = ms_iteration(p0, c, alpha, beta_vec, M)
        p_u, it_u, t_u = uniform_pricing(c, alpha, beta_vec, M)
        try:
            p_bn, it_bn, t_bn = newton_bn(p_m.copy(), c, alpha, beta_vec, M)
        except Exception:
            p_bn, it_bn, t_bn = p_m.copy(), 0, 0.0

        pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)
        if pi_bn <= 0:
            continue
        gap = lambda p: max(0.0, (pi_bn - total_profit(p, c, alpha, beta_vec, M)) / pi_bn)
        results.append({
            "day": int(day),
            "ebar": ebar_obs,
            "gap_gamma": gap(p_g),
            "gap_MS": gap(p_m),
            "gap_uniform": gap(p_u),
            "time_gamma": t_g, "time_MS": t_m,
            "iter_gamma": it_g, "iter_MS": it_m,
        })
    if not results:
        return None
    df = pd.DataFrame(results)
    return {
        "M_mult": M_mult,
        "margin": margin,
        "M": M,
        "R2": R2,
        "n_days": len(df),
        "ebar_median": df["ebar"].median(),
        "ebar_max": df["ebar"].max(),
        "gap_gamma_mean": df["gap_gamma"].mean(),
        "gap_gamma_median": df["gap_gamma"].median(),
        "gap_gamma_max": df["gap_gamma"].max(),
        "gap_MS_mean": df["gap_MS"].mean(),
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
    print("JD SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"M_MULTS:  {M_MULTS}")
    print(f"MARGINS:  {MARGINS}")
    print(f"Total calibrations: {len(M_MULTS) * len(MARGINS)}")
    print()

    t_global = time.time()
    print("[1/3] Loading data...")
    orders, sku = load_orders_and_sku()
    print("[2/3] Aggregating and assigning price deciles...")
    agg = aggregate_daily(orders, sku)
    agg = assign_price_deciles(agg)

    print("[3/3] Running calibrations:")
    rows = []
    total = len(M_MULTS) * len(MARGINS)
    k = 0
    for M_mult in M_MULTS:
        for margin in MARGINS:
            k += 1
            t0 = time.time()
            res = run_one_calibration(agg, M_mult, margin)
            t = time.time() - t0
            if res is None:
                print(f"    ({k}/{total}) M_mult={M_mult}, margin={margin}: FAILED ({t:.1f}s)")
                continue
            rows.append(res)
            print(f"    ({k}/{total}) M_mult={M_mult}, margin={margin}:  "
                  f"γ-gap {res['gap_gamma_median']*100:.2f}%  "
                  f"uniform-gap {res['gap_uniform_median']*100:.2f}%  "
                  f"speedup {res['speedup_gamma_vs_MS']:.0f}×  "
                  f"({t:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "jd_sensitivity_results.csv", index=False)

    # Pivot tables
    pivots = {}
    for col in ["gap_gamma_median", "gap_uniform_median", "ebar_median",
                "speedup_gamma_vs_MS", "gamma_beats_uniform_pct"]:
        pivots[col] = df.pivot(index="M_mult", columns="margin", values=col)
    pivots["gap_gamma_median"].to_csv(OUT_DIR / "jd_sensitivity_summary.csv")

    # Heatmaps
    def heatmap(mat, title, fname, cmap, fmt):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                       origin="lower")
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
        ax.set_xlabel("margin (c / p)")
        ax.set_ylabel("M_mult (potential market multiplier)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(OUT_DIR / fname, dpi=150)
        plt.close()

    heatmap(pivots["gap_gamma_median"] * 100,
            "γ-equalization median profit gap (%)",
            "jd_sensitivity_heatmap_gamma.png", "viridis", "{:.2f}")
    heatmap(pivots["gap_uniform_median"] * 100,
            "Uniform-markup median profit gap (%)",
            "jd_sensitivity_heatmap_uniform.png", "magma", "{:.2f}")
    heatmap(pivots["ebar_median"],
            "Median ebar at observed prices",
            "jd_sensitivity_heatmap_ebar.png", "coolwarm", "{:.3f}")
    heatmap(pivots["speedup_gamma_vs_MS"],
            "γ vs MS2011 wall-clock speedup (×)",
            "jd_sensitivity_heatmap_speedup.png", "viridis", "{:.0f}")

    print()
    print("=" * 70)
    print(f"Total runtime: {(time.time() - t_global)/60:.1f} min")
    print()
    print("γ-equalization median profit gap (%) across M × margin:")
    print((pivots["gap_gamma_median"] * 100).round(2).to_string())
    print()
    print("uniform median profit gap (%) across M × margin:")
    print((pivots["gap_uniform_median"] * 100).round(2).to_string())
    print()
    print("median ebar across M × margin:")
    print(pivots["ebar_median"].round(3).to_string())
    print()
    print("γ vs MS2011 speedup across M × margin:")
    print(pivots["speedup_gamma_vs_MS"].round(0).to_string())
    print()
    print(f"outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
