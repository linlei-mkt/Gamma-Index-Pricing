"""
JD γ-robustness experiments (responds to two referee questions):

Experiment A — γ-misspecification sweep.
  For each of the 31 daily markets, find the bisection-tuned γ̂ that
  meets the GMV floor R_target = 1.15 × observed GMV. Then perturb
  γ = γ̂ + Δγ for Δγ ∈ {±0.02, ±0.05, ±0.10, ±0.20} and, at each γ:
    - run the inner price iteration to convergence,
    - record: converged?, #iterations, GMV deviation from target (%),
      raw profit change vs the γ̂ solution (%), and the Lagrangian loss
      at the day's true multiplier μ* (from constrained BN).
  Expected: convergence everywhere in the admissible range (contraction
  does not depend on γ being "correct"); GMV miss and raw profit change
  first-order in Δγ; Lagrangian loss ≈ quadratic in Δγ (envelope).

Experiment B — outcome-based (model-free) feedback calibration.
  The outer loop is only allowed to observe REALIZED total GMV
  (optionally with multiplicative noise), not the fitted demand model.
  Rule: GMV below target → move γ down (more negative); above → move
  γ up toward 0 (damped bisection). Report the trajectory γ_t and
  GMV_t/target across rounds, median across days, with/without noise.

Uses the same pipeline as jd_gmv_constrained.py (HB posterior β̂,
MCI calibration, M = 3 × peak-day quantity, c = 0.7·p̄).

Outputs (in OUT_DIR):
  - jd_gamma_robustness.csv         Experiment A per-(day, Δγ) results
  - jd_gamma_robustness.png         two-panel: GMV miss / Lagrangian loss
  - jd_gamma_feedback.csv           Experiment B per-(day, round) results
  - jd_gamma_feedback.png           trajectory of γ_t and GMV_t/target

Usage:
  JD_DATA_DIR=/path/to/csvs HB_CSV=/path/to/jd_hb_posterior_summary.csv \
      python3 jd_gamma_robustness.py
Runtime: ~2 minutes.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

warnings.simplefilter("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("JD_DATA_DIR", "../JD_MSOM"))
HB_CSV = Path(os.environ.get("HB_CSV", str(SCRIPT_DIR / "jd_hb_posterior_summary.csv")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(SCRIPT_DIR)))

N_TOP_SKU = 500
M_MULT = 3.0
MARGIN_RATIO = 0.70
GMV_TARGET_MULT = 1.15
BETA_FLOOR = 1.2
TOL = 1e-8
MAX_ITER = 2000
DELTA_GRID = [-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20]
FEEDBACK_NOISE_SD = 0.02      # 2% multiplicative noise on observed GMV
FEEDBACK_ROUNDS = 12
SEED = 2026

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# Data + MCI primitives (identical to jd_gmv_constrained.py)
# ----------------------------------------------------------------------
def load_hb_posterior():
    df = pd.read_csv(HB_CSV)
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
        usecols=["sku_ID", "type"], dtype={"sku_ID": "string"},
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
    agg = agg[(agg["qty"] > 0) & (agg["price"] > 0)].copy()
    peak = agg.groupby("day")["qty"].sum().max()
    M = M_MULT * peak
    agg["share"] = agg["qty"] / M
    daily_Q = agg.groupby("day")["qty"].sum().rename("Q_in")
    s0 = (1.0 - daily_Q / M).rename("s0")
    agg = agg.merge(s0, on="day")
    return agg, M


def mci_shares(p, alpha, beta):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D


def calibrate_alpha(p_obs, s_obs, s0_obs, beta):
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def share_jacobian(p, s, beta):
    u = beta * s / p
    Om = np.outer(u, s)
    np.fill_diagonal(Om, -u * (1.0 - s))
    return Om


def total_profit(p, c, alpha, beta, M):
    s = mci_shares(p, alpha, beta)
    return float(np.sum((p - c) * s) * M)


def total_revenue(p, alpha, beta, M):
    s = mci_shares(p, alpha, beta)
    return float(np.sum(p * s) * M)


def newton_bn(p0, c, alpha, beta, M):
    p = np.maximum(p0.copy(), c * 1.0001)
    for _ in range(500):
        s = mci_shares(p, alpha, beta)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        p_new = np.maximum(c - (s + Gamma @ (p - c)) / diag, c * 1.0001)
        if np.max(np.abs(p_new - p)) < 1e-10:
            return p_new
        p = p_new
    return p


def constrained_bn_floor(p0, c, alpha, beta, M, R_target):
    p_unc = newton_bn(p0, c, alpha, beta, M)
    if total_revenue(p_unc, alpha, beta, M) >= R_target:
        return p_unc, 0.0

    def residual(mu):
        p_mu = newton_bn(p0, c / (1.0 + mu), alpha, beta, M)
        return total_revenue(p_mu, alpha, beta, M) - R_target

    try:
        mu_star = brentq(residual, 1e-6, 100.0, xtol=1e-6, maxiter=50)
    except ValueError:
        mu_star = 100.0
    return newton_bn(p0, c / (1.0 + mu_star), alpha, beta, M), mu_star


def gamma_iteration(p0, c, alpha, beta, M, gamma_star):
    """Inner loop. Returns (p, iterations, converged_flag).
    NOTE: no denominator clamp here — we want to OBSERVE failure at
    extreme γ rather than mask it. Admissible iff denominator > 0."""
    p = p0.copy()
    for k in range(MAX_ITER):
        s = mci_shares(p, alpha, beta)
        eta = np.maximum(beta * (1.0 - s), 1.01)
        denom = 1.0 - gamma_star - (1.0 - gamma_star) / eta
        if np.any(denom <= 1e-6):
            return p, k + 1, False        # denominator hit → inadmissible γ
        p_new = np.maximum(c / denom, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            return p_new, k + 1, True
        p = p_new
    return p, MAX_ITER, False


def tune_gamma_star(p0, c, alpha, beta, M, R_target, bounds=(-5.0, 0.99)):
    def residual(gs):
        p_gs, _, _ = gamma_iteration(p0, c, alpha, beta, M, gs)
        return total_revenue(p_gs, alpha, beta, M) - R_target
    lo, hi = bounds
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo * r_hi > 0:
        return (lo if abs(r_lo) < abs(r_hi) else hi)
    return brentq(residual, lo, hi, xtol=1e-6, maxiter=60)


# ----------------------------------------------------------------------
# Load once, build per-day market structs
# ----------------------------------------------------------------------
print("[1/4] Loading data and HB posterior...")
agg, M = load_and_aggregate()
sku_to_beta = load_hb_posterior()
agg = agg[agg["sku_ID"].isin(sku_to_beta.keys())].copy()
agg["beta_hat"] = agg["sku_ID"].map(sku_to_beta)

markets = []
for d in sorted(agg["day"].unique()):
    mkt = agg[agg["day"] == d].reset_index(drop=True)
    if len(mkt) < 50:
        continue
    p_obs = mkt["price"].to_numpy()
    s_obs = mkt["share"].to_numpy()
    beta = mkt["beta_hat"].to_numpy()
    alpha = calibrate_alpha(p_obs, s_obs, float(mkt["s0"].iloc[0]), beta)
    c = MARGIN_RATIO * p_obs
    R_obs = float(np.sum(p_obs * s_obs) * M)
    markets.append(dict(day=int(d), p0=p_obs, c=c, alpha=alpha, beta=beta,
                        R_target=GMV_TARGET_MULT * R_obs))
print(f"    {len(markets)} daily markets, M = {M:.0f}")

# ----------------------------------------------------------------------
# Experiment A: γ-misspecification sweep
# ----------------------------------------------------------------------
print("[2/4] Experiment A: γ-misspecification sweep...")
rows_a = []
for mkt in markets:
    p0, c, alpha, beta = mkt["p0"], mkt["c"], mkt["alpha"], mkt["beta"]
    R_target = mkt["R_target"]

    g_hat = tune_gamma_star(p0, c, alpha, beta, M, R_target)
    p_hat, it_hat, conv_hat = gamma_iteration(p0, c, alpha, beta, M, g_hat)
    pi_hat = total_profit(p_hat, c, alpha, beta, M)

    # true multiplier from constrained BN (theoretical comparison object)
    p_cbn, mu_star = constrained_bn_floor(p0, c, alpha, beta, M, R_target)
    L = lambda p: (total_profit(p, c, alpha, beta, M)
                   + mu_star * total_revenue(p, alpha, beta, M))
    L_cbn = L(p_cbn)

    for dg in DELTA_GRID:
        g = g_hat + dg
        p_g, iters, conv = gamma_iteration(p0, c, alpha, beta, M, g)
        R_g = total_revenue(p_g, alpha, beta, M)
        pi_g = total_profit(p_g, c, alpha, beta, M)
        rows_a.append(dict(
            day=mkt["day"], gamma_hat=g_hat, delta_gamma=dg, gamma=g,
            mu_star=mu_star, converged=conv, iterations=iters,
            gmv_miss_pct=100.0 * (R_g - R_target) / R_target,
            profit_change_pct=100.0 * (pi_g - pi_hat) / pi_hat,
            lagrangian_loss_pct=100.0 * max(L_cbn - L(p_g), 0.0) / L_cbn,
            lagrangian_loss_at0_pct=100.0 * max(L_cbn - L(p_hat), 0.0) / L_cbn,
        ))

df_a = pd.DataFrame(rows_a)
df_a.to_csv(OUT_DIR / "jd_gamma_robustness.csv", index=False)

# ----------------------------------------------------------------------
# Experiment B: outcome-based feedback calibration
# ----------------------------------------------------------------------
print("[3/4] Experiment B: outcome-based feedback (model-free outer loop)...")
rows_b = []
for noise_sd, tag in [(0.0, "noiseless"), (FEEDBACK_NOISE_SD, "noisy")]:
    for mkt in markets:
        p0, c, alpha, beta = mkt["p0"], mkt["c"], mkt["alpha"], mkt["beta"]
        R_target = mkt["R_target"]
        g_hat = tune_gamma_star(p0, c, alpha, beta, M, R_target)

        lo, hi = -5.0, 0.0            # bracket; direction rule shrinks it
        g_t = 0.0                      # start at classical Lerner
        for t in range(1, FEEDBACK_ROUNDS + 1):
            p_t, _, conv = gamma_iteration(p0, c, alpha, beta, M, g_t)
            R_true = total_revenue(p_t, alpha, beta, M)
            R_seen = R_true * (1.0 + rng.normal(0.0, noise_sd))
            rows_b.append(dict(
                regime=tag, day=mkt["day"], round=t, gamma_t=g_t,
                gamma_hat=g_hat, gmv_ratio=R_seen / R_target,
                gamma_error=abs(g_t - g_hat), converged=conv,
            ))
            # Direction rule using ONLY the observed aggregate GMV:
            if R_seen < R_target:
                hi = g_t              # too little GMV → γ must fall
            else:
                lo = g_t              # enough GMV → γ can rise toward 0
            g_t = 0.5 * (lo + hi)

df_b = pd.DataFrame(rows_b)
df_b.to_csv(OUT_DIR / "jd_gamma_feedback.csv", index=False)

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
print("[4/4] Figures...")

# --- Figure A: two-panel robustness ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

med = df_a.groupby("delta_gamma").agg(
    gmv=("gmv_miss_pct", "median"),
    gmv_lo=("gmv_miss_pct", lambda x: x.quantile(0.25)),
    gmv_hi=("gmv_miss_pct", lambda x: x.quantile(0.75)),
    lag=("lagrangian_loss_pct", "median"),
    lag_lo=("lagrangian_loss_pct", lambda x: x.quantile(0.25)),
    lag_hi=("lagrangian_loss_pct", lambda x: x.quantile(0.75)),
    prof=("profit_change_pct", "median"),
    conv=("converged", "mean"),
    iters=("iterations", "median"),
).reset_index()

ax = axes[0]
ax.errorbar(med["delta_gamma"], med["gmv"],
            yerr=[med["gmv"] - med["gmv_lo"], med["gmv_hi"] - med["gmv"]],
            fmt="o-", color="tab:blue", capsize=3, label="GMV miss (median, IQR)")
# linear fit through origin
k_lin = np.polyfit(med["delta_gamma"], med["gmv"], 1)
xs = np.linspace(min(DELTA_GRID), max(DELTA_GRID), 100)
ax.plot(xs, np.polyval(k_lin, xs), "k--", alpha=0.6,
        label=f"linear fit (slope {k_lin[0]:.1f})")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel(r"$\Delta\gamma = \gamma - \hat\gamma^\star$")
ax.set_ylabel("GMV deviation from target (%)")
ax.set_title("(a) Feasibility error: first-order in $\\Delta\\gamma$")
ax.legend(fontsize=8)

ax = axes[1]
# Excess Lagrangian loss over each day's Δγ=0 floor (the O(ē²) baseline)
df_a["excess_loss"] = df_a["lagrangian_loss_pct"] - df_a["lagrangian_loss_at0_pct"]
med_ex = df_a.groupby("delta_gamma").agg(
    ex=("excess_loss", "median"),
    ex_lo=("excess_loss", lambda x: x.quantile(0.25)),
    ex_hi=("excess_loss", lambda x: x.quantile(0.75)),
).reset_index()
floor_med = df_a["lagrangian_loss_at0_pct"].median()
ax.errorbar(med_ex["delta_gamma"], med_ex["ex"],
            yerr=[med_ex["ex"] - med_ex["ex_lo"],
                  med_ex["ex_hi"] - med_ex["ex"]],
            fmt="s-", color="tab:red", capsize=3,
            label="excess Lagrangian loss over $\\Delta\\gamma=0$ floor")
# fit a·|Δγ| + b·Δγ² on the excess
X = np.column_stack([np.abs(med_ex["delta_gamma"]), med_ex["delta_gamma"] ** 2])
coef, *_ = np.linalg.lstsq(X, med_ex["ex"], rcond=None)
ax.plot(xs, coef[0] * np.abs(xs) + coef[1] * xs ** 2, "k--", alpha=0.6,
        label=f"fit ${coef[0]:.2f}\\,|\\Delta\\gamma| + {coef[1]:.1f}\\,\\Delta\\gamma^2$")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel(r"$\Delta\gamma = \gamma - \hat\gamma^\star$")
ax.set_ylabel("excess loss (% of constrained-BN Lagrangian)")
ax.set_title(f"(b) Excess loss $\\approx$ quadratic "
             f"(floor: {floor_med:.2f}%)", fontsize=11)
ax.legend(fontsize=8, loc="upper center")

plt.tight_layout()
plt.savefig(OUT_DIR / "jd_gamma_robustness.png", dpi=150)
plt.close()

# --- Figure B: feedback trajectories ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for tag, color in [("noiseless", "tab:blue"), ("noisy", "tab:orange")]:
    sub = df_b[df_b["regime"] == tag]
    tr = sub.groupby("round").agg(
        err=("gamma_error", "median"),
        err_lo=("gamma_error", lambda x: x.quantile(0.25)),
        err_hi=("gamma_error", lambda x: x.quantile(0.75)),
        ratio=("gmv_ratio", "median"),
        r_lo=("gmv_ratio", lambda x: x.quantile(0.25)),
        r_hi=("gmv_ratio", lambda x: x.quantile(0.75)),
    ).reset_index()
    axes[0].plot(tr["round"], tr["err"], "o-", color=color, label=tag)
    axes[0].fill_between(tr["round"], tr["err_lo"], tr["err_hi"],
                         color=color, alpha=0.15)
    axes[1].plot(tr["round"], tr["ratio"], "o-", color=color, label=tag)
    axes[1].fill_between(tr["round"], tr["r_lo"], tr["r_hi"],
                         color=color, alpha=0.15)

axes[0].set_yscale("log")
axes[0].set_xlabel("feedback round $t$")
axes[0].set_ylabel(r"$|\gamma_t - \hat\gamma^\star|$ (log scale)")
axes[0].set_title("(a) Outcome-based calibration converges")
axes[0].legend()
axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.6,
                label="GMV = target")
axes[1].set_xlabel("feedback round $t$")
axes[1].set_ylabel("realized GMV / target")
axes[1].set_title("(b) GMV homes in on the target")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_gamma_feedback.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------
print()
print("=" * 72)
print("EXPERIMENT A: γ-misspecification sweep  (31 days × Δγ grid)")
print("=" * 72)
print(med.round(3).to_string(index=False))
print()
conv_tab = df_a.groupby("delta_gamma")["converged"].mean()
print("Convergence rate by Δγ:", dict(conv_tab.round(3)))
print(f"Δγ=0 Lagrangian floor (median): {floor_med:.3f}%  [the O(ē²) approximation floor]")
print("Excess loss by Δγ (median):")
print(med_ex.round(3).to_string(index=False))
print(f"Excess-loss fit: {coef[0]:.3f}·|Δγ| + {coef[1]:.2f}·Δγ²")
print()
print("=" * 72)
print("EXPERIMENT B: outcome-based feedback")
print("=" * 72)
for tag in ["noiseless", "noisy"]:
    fin = df_b[(df_b["regime"] == tag) & (df_b["round"] == FEEDBACK_ROUNDS)]
    print(f"{tag:10s}: median |γ_T − γ̂| = {fin['gamma_error'].median():.4f}, "
          f"median |GMV/target − 1| = {abs(fin['gmv_ratio'] - 1).median()*100:.2f}%")
print()
print("Outputs in:", OUT_DIR)
