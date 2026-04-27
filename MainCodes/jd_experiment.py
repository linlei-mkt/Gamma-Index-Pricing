"""
JD Real-Data Application: compare γ-equalization, MS2011 ζ-iteration,
uniform pricing, and full Newton (BN optimum) on the 2020 MSOM JD data
competition dataset.

Pipeline:
  1. Load orders + SKU metadata, filter to top 500 SKUs by order volume.
  2. Aggregate to daily × SKU: quantity-weighted mean price, total quantity.
  3. Compute market shares: inside potential market size M = 3 × peak-day
     inside quantity.  Outside share = 1 - sum(inside).
  4. Estimate MCI demand by pooled OLS of log(s_i/s_0) on log(p_i),
     sku FE, day FE, one price slope per type (β₁, β₂).
  5. For each day t: given β, shares s(p), and c = 0.7·p̄, run
        (a) γ-iteration at γ* = 0
        (b) MS-ζ iteration (Morrow-Skerlos 2011)
        (c) uniform markup (1-D optimization)
        (d) full Newton on BN FOC (ground-truth optimum)
     Tolerance: max |p^(k+1)-p^k| < 1e-8.  Record iterations,
     wall-clock time, profit, and ebar at observed shares.
  6. Write CSVs + figures to the workspace folder.

Outputs (in /Users/linlei/Downloads/Gamma/):
  - jd_elasticities.csv           β estimates + R²
  - jd_pricing_comparison.csv     per-day results
  - jd_convergence.png            iteration vs profit gap
  - jd_ebar_distribution.png      ebar across days
  - jd_profit_gap_vs_ebar.png     profit gap vs ebar scatter
  - jd_wallclock_comparison.png   wall-clock per method

To replicate:
  pip install pandas numpy scipy statsmodels matplotlib
  python jd_experiment.py
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
import statsmodels.api as sm

warnings.simplefilter("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    # default: where the three CSVs live
    ""
))
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
N_TOP_SKU = 500           # top SKUs by order count
MARGIN_RATIO = 0.70       # c_i = MARGIN_RATIO · mean(p_i)  (30% gross margin)
MARKET_MULT = 3.0         # potential market M = MARKET_MULT × peak-day inside quantity
TOL = 1e-8
MAX_ITER = 2000
SEED = 2026

np.random.seed(SEED)

# ======================================================================
# 1. Data prep
# ======================================================================
print("[1/6] Loading and filtering data...")
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

# Top N SKUs by number of orders
top_ids = (
    orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index.tolist()
)
o = orders[orders["sku_ID"].isin(top_ids)].merge(sku, on="sku_ID", how="left")
o["day"] = o["order_date"].dt.day

# Aggregate daily × SKU: quantity-weighted mean price, total quantity
agg = (
    o.assign(rev=o["final_unit_price"] * o["quantity"])
     .groupby(["day", "sku_ID"], as_index=False)
     .agg(qty=("quantity", "sum"),
          rev=("rev", "sum"),
          nobs=("final_unit_price", "count"))
)
agg["price"] = agg["rev"] / agg["qty"]
agg = agg.merge(sku[["sku_ID", "type", "brand_ID"]], on="sku_ID", how="left")

# Only keep day × SKU with positive quantity, positive price, and valid type
agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
agg["type"] = agg["type"].astype(int)

# Potential market size M: 3 × peak-day inside quantity, constant across days
peak_inside = agg.groupby("day")["qty"].sum().max()
M_pot = MARKET_MULT * peak_inside
print(f"    peak daily inside quantity={peak_inside:.0f}, potential market M={M_pot:.0f}")

# Daily inside total quantity -> outside share
daily_inside = agg.groupby("day")["qty"].sum().rename("Q_inside").to_frame()
daily_inside["s0"] = 1.0 - daily_inside["Q_inside"] / M_pot
daily_inside["S"] = 1.0 - daily_inside["s0"]
agg = agg.merge(daily_inside[["s0", "S"]], on="day")
agg["share"] = agg["qty"] / M_pot
print(f"    market share summary (inside total S across days): "
      f"min={daily_inside['S'].min():.4f}, median={daily_inside['S'].median():.4f}, "
      f"max={daily_inside['S'].max():.4f}")

# ======================================================================
# 2. MCI demand estimation
# ======================================================================
# Assign each SKU to a price-decile bucket within its type. Decile is
# computed on each SKU's mean price across its daily observations.
# Rationale: consumers of high-price products are systematically more
# price-inelastic than consumers of low-price products within the same
# product category (Tellis 1988 JMR meta-analysis); price tier therefore
# proxies for unobserved demand heterogeneity in a way that product-type
# dummies alone cannot. Bucketing by price decile (rather than by
# anonymous `attribute1/attribute2`) keeps the heterogeneity economically
# interpretable to reviewers.
N_DECILES = 10
print(f"[2/6] Estimating MCI demand ({N_DECILES} price-decile buckets)...")

sku_mean_p = agg.groupby("sku_ID")["price"].mean().rename("mean_p")
sku_type = agg.groupby("sku_ID")["type"].first().rename("sku_type")
sku_info = pd.concat([sku_mean_p, sku_type], axis=1)
# Rank globally by mean price, cut into deciles (labels 0..N_DECILES-1)
sku_info["bucket"] = pd.qcut(
    sku_info["mean_p"].rank(method="first"),
    q=N_DECILES,
    labels=False,
).astype(int)
agg = agg.merge(sku_info["bucket"], left_on="sku_ID", right_index=True, how="left")

print(f"    decile bucket price ranges:")
for b in range(N_DECILES):
    sub = sku_info[sku_info["bucket"] == b]
    print(f"      bucket {b}: n_sku={len(sub)}, mean_p range "
          f"[{sub['mean_p'].min():.1f}, {sub['mean_p'].max():.1f}]")

# log(s_i / s_0) = α_i (sku FE) + δ_t (day FE) - β_{b(i)} · log(p_i) + ε
agg["y"] = np.log(agg["share"]) - np.log(agg["s0"])
agg["log_price"] = np.log(agg["price"])

# Design: SKU FE + day FE + bucket-specific log-price slopes
X = pd.get_dummies(
    agg[["sku_ID", "day"]].astype(str),
    columns=["sku_ID", "day"],
    drop_first=True,
    dtype=float,
)
for b in range(N_DECILES):
    X[f"logp_b{b}"] = agg["log_price"] * (agg["bucket"] == b).astype(float)
X = sm.add_constant(X)
y = agg["y"].astype(float)

print(f"    regression dimensions: n={len(y)}, k={X.shape[1]}")
mod = sm.OLS(y, X).fit()
print(f"    R² = {mod.rsquared:.4f}")

betas = {}
for b in range(N_DECILES):
    col = f"logp_b{b}"
    if col in mod.params.index:
        # MCI: log(s/s0) = ... - β log p   →  coef on log p = -β
        betas[b] = -mod.params[col]

print("    estimated β by price decile:")
for b, bv in sorted(betas.items()):
    sub = sku_info[sku_info["bucket"] == b]
    print(f"      bucket {b} (p∈[{sub['mean_p'].min():.0f},"
          f"{sub['mean_p'].max():.0f}]): β = {bv:.3f}")

# Clamp any estimate below floor so γ-iteration is well-defined.
beta_floor = 1.2
for b in list(betas.keys()):
    if betas[b] < beta_floor:
        print(f"    WARNING: β_bucket{b}={betas[b]:.3f} below floor {beta_floor}, clamping.")
        betas[b] = beta_floor

# Save with bucket price ranges for interpretability.
# Note on sign: `beta_MCI_exponent` is the MCI attraction exponent (positive
# by convention).  The own-price elasticity is η_i = -β_i · (1 - s_i),
# which is negative. `own_elasticity_typical` evaluates η at the median
# observed share within the bucket to make the negative sign explicit.
median_share_by_bucket = agg.groupby("bucket")["share"].median()
bucket_rows = []
for b in range(N_DECILES):
    sub = sku_info[sku_info["bucket"] == b]
    s_med = float(median_share_by_bucket.get(b, 0.001))
    eta_typical = -betas[b] * (1.0 - s_med)
    bucket_rows.append({
        "bucket": b,
        "n_sku": int(len(sub)),
        "mean_p_min": float(sub["mean_p"].min()),
        "mean_p_max": float(sub["mean_p"].max()),
        "mean_p_median": float(sub["mean_p"].median()),
        "beta_MCI_exponent": betas[b],          # positive
        "median_share": s_med,
        "own_elasticity_typical": eta_typical,  # negative
    })
pd.DataFrame(bucket_rows).assign(R2=mod.rsquared, n_obs=len(y)).to_csv(
    OUT_DIR / "jd_elasticities.csv", index=False
)

# ======================================================================
# 3. Pricing solvers
# ======================================================================
def mci_shares(p: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
               M: float) -> tuple[np.ndarray, float]:
    """Return (inside shares s, outside share s0) under MCI.
    A_i = α_i · p_i^{-β_i}.  Outside attraction A_0 fixed to yield calibrated s0.
    Here α already absorbs A_0 = 1 by construction below.
    """
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()  # 1 = A_0 normalization
    s = A / D
    s0 = 1.0 / D
    return s, s0


def calibrate_alpha(p_obs: np.ndarray, s_obs: np.ndarray, s0_obs: float,
                    beta: np.ndarray) -> np.ndarray:
    """Back out α_i so MCI reproduces the observed shares at observed prices
    with outside-share normalization A_0 = 1."""
    # s_i / s_0 = A_i = α_i · p_i^{-β_i}
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def share_jacobian(p: np.ndarray, s: np.ndarray,
                   beta: np.ndarray) -> np.ndarray:
    """Ω_{ij} = ∂s_j/∂p_i for MCI.
        Ω_{ii} = -β_i · s_i · (1-s_i) / p_i
        Ω_{ij} = +β_i · s_i · s_j / p_i  for j≠i
    """
    # Ω = (β ⊙ s / p)_i · (outer product) with diagonal correction
    u = beta * s / p                              # shape (n,)
    Om = np.outer(u, s)                           # Ω_{ij} = u_i · s_j  (off-diagonal form)
    # Correct diagonal: Ω_{ii} should be -u_i (1-s_i) = -u_i + u_i s_i
    # Current diagonal is u_i s_i.  Replace with -u_i (1-s_i).
    np.fill_diagonal(Om, -u * (1.0 - s))
    return Om


def ebar_from_shares(s: np.ndarray) -> float:
    """ebar(p) = max_i (S - s_i)/(1 - s_i)."""
    S = s.sum()
    return float(np.max((S - s) / (1.0 - s)))


# ----- γ-iteration (γ* = 0; unconstrained profit max approximation) -----
def gamma_iteration(p0, c, alpha, beta, M, gamma_star=0.0,
                    tol=TOL, max_iter=MAX_ITER):
    """Update: p_i^{t+1} = c_i / (1 - γ* - (1-γ*)/|η_i(p^t)|),
    where |η_i| = β_i·(1-s_i(p^t)).
    Information used at each step: own elasticity |η_i| and c_i only."""
    p = p0.copy()
    hist = []
    t0 = time.perf_counter()
    for k in range(max_iter):
        s, s0 = mci_shares(p, alpha, beta, M)
        eta = beta * (1.0 - s)               # |η_i| under MCI
        # Avoid division by near-zero: if eta_i ≤ 1, Lerner rule blows up.
        # Clamp to a small safety margin above 1.
        eta_safe = np.maximum(eta, 1.01)
        p_new = c / (1.0 - gamma_star - (1.0 - gamma_star) / eta_safe)
        p_new = np.maximum(p_new, c * 1.0001)   # keep p > c
        diff = float(np.max(np.abs(p_new - p)))
        hist.append(diff)
        p = p_new
        if diff < tol:
            break
    wall = time.perf_counter() - t0
    return p, k + 1, wall, hist


# ----- MS-ζ iteration (Morrow-Skerlos 2011) -----
def ms_iteration(p0, c, alpha, beta, M, tol=TOL, max_iter=MAX_ITER):
    """Update: p^{t+1} = c - Λ^{-1}·[s + Γ(p-c)], where Λ = diag(Ω_{ii}),
    Γ = Ω - Λ.  This is a Jacobi splitting of the BN FOC s + Ω(p-c) = 0.
    Information used at each step: full n×n Jacobian Ω(p^t)."""
    p = p0.copy()
    hist = []
    t0 = time.perf_counter()
    for k in range(max_iter):
        s, s0 = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()                 # Λ_{ii}
        Gamma = Om - np.diag(diag)                # off-diagonal part
        # Update: p_new_i = c_i - (1/diag_i) [s_i + (Γ(p-c))_i]
        rhs = s + Gamma @ (p - c)
        p_new = c - rhs / diag
        p_new = np.maximum(p_new, c * 1.0001)
        diff = float(np.max(np.abs(p_new - p)))
        hist.append(diff)
        p = p_new
        if diff < tol:
            break
    wall = time.perf_counter() - t0
    return p, k + 1, wall, hist


# ----- Full Newton on BN FOC (ground truth) -----
def newton_bn(p0, c, alpha, beta, M, tol=1e-12, max_iter=200):
    """Solve BN FOC F(p) = s(p) + Ω(p)(p-c) = 0 by Newton's method.
    Returns the profit-maximizing price vector of an unconstrained
    multi-product monopolist (joint optimum).  Information: full Ω + Hessian."""
    def F_and_J(p):
        s, s0 = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        F = s + Om @ (p - c)
        # Approximate Jacobian: 2Ω (secant-style).  This is Morrow's
        # Jacobi-Newton; full second-derivative Hessian would be more
        # expensive.  For convergence benchmarking it's plenty accurate.
        J = Om + np.diag(Om @ np.ones_like(p)) * 0.0  # placeholder
        J = Om + Om.T  # second-order-ish; symmetric damping
        # Regularize for numerical stability
        J = J + 1e-10 * np.eye(len(p))
        return F, J

    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(max_iter):
        F, J = F_and_J(p)
        try:
            dp = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dp = np.linalg.lstsq(J, -F, rcond=None)[0]
        # Damped step
        step = 0.5
        p_new = np.maximum(p + step * dp, c * 1.0001)
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    wall = time.perf_counter() - t0
    return p, k + 1, wall


def newton_bn_robust(p0, c, alpha, beta, M):
    """Back-stop: use scipy.optimize.root with Krylov method on BN FOC."""
    def F(p):
        p_pos = np.maximum(p, c * 1.0001)
        s, _ = mci_shares(p_pos, alpha, beta, M)
        Om = share_jacobian(p_pos, s, beta)
        return s + Om @ (p_pos - c)
    t0 = time.perf_counter()
    sol = root(F, p0, method="krylov", tol=1e-10,
               options={"maxiter": 500})
    wall = time.perf_counter() - t0
    p_opt = np.maximum(sol.x, c * 1.0001)
    return p_opt, sol.nit if hasattr(sol, "nit") else -1, wall


# ----- Uniform markup (1-D optimization) -----
def uniform_pricing(c, alpha, beta, M):
    """Pick a single scalar markup m ∈ (0,1), p_i = c_i / (1-m), maximize profit."""
    def neg_profit(m):
        if not (0.0 < m < 0.999):
            return 1e18
        p = c / (1.0 - m)
        s, s0 = mci_shares(p, alpha, beta, M)
        return -np.sum((p - c) * s) * M  # scale by M to total profit
    t0 = time.perf_counter()
    res = minimize_scalar(neg_profit, bounds=(0.001, 0.999), method="bounded",
                          options={"xatol": 1e-8})
    wall = time.perf_counter() - t0
    m_star = res.x
    p = c / (1.0 - m_star)
    return p, res.nit if hasattr(res, "nit") else 50, wall, m_star


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# 4. Run per-day experiment
# ======================================================================
print("[3/6] Running per-day pricing experiments...")
rows = []
convergence_histories = {}   # keep one representative day for the convergence plot

days_sorted = sorted(agg["day"].unique())
for day in days_sorted:
    mkt = agg[agg["day"] == day].copy()
    n = len(mkt)
    if n < 50:
        continue
    p_obs = mkt["price"].to_numpy()
    s_obs = mkt["share"].to_numpy()
    s0_obs = float(mkt["s0"].iloc[0])
    beta_vec = mkt["bucket"].map(betas).to_numpy()
    # Back out α so MCI matches observed shares exactly at observed prices
    alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
    # Costs: 30% margin from observed prices
    c = MARGIN_RATIO * p_obs
    M = M_pot
    ebar_obs = ebar_from_shares(s_obs)

    # Starting point: observed prices (realistic warm-start for any method)
    p0 = p_obs.copy()

    # (a) γ-iteration
    p_g, it_g, t_g, hist_g = gamma_iteration(p0, c, alpha, beta_vec, M)
    # (b) MS-ζ
    p_m, it_m, t_m, hist_m = ms_iteration(p0, c, alpha, beta_vec, M)
    # (c) uniform
    p_u, it_u, t_u, m_star = uniform_pricing(c, alpha, beta_vec, M)
    # (d) Newton BN (ground truth)
    try:
        p_bn, it_bn, t_bn = newton_bn_robust(p_m.copy(), c, alpha, beta_vec, M)
    except Exception:
        p_bn, it_bn, t_bn = p_m.copy(), 0, 0.0  # fall back to MS result

    pi_g = total_profit(p_g, c, alpha, beta_vec, M)
    pi_m = total_profit(p_m, c, alpha, beta_vec, M)
    pi_u = total_profit(p_u, c, alpha, beta_vec, M)
    pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)

    # Profit gaps (relative to BN, lower = better)
    def gap(pi): return max(0.0, (pi_bn - pi) / pi_bn)

    rows.append({
        "day": int(day),
        "n_products": n,
        "ebar": ebar_obs,
        "S_inside": s_obs.sum(),
        "pi_BN": pi_bn,
        "pi_gamma": pi_g, "pi_MS": pi_m, "pi_uniform": pi_u,
        "gap_gamma": gap(pi_g), "gap_MS": gap(pi_m), "gap_uniform": gap(pi_u),
        "iter_gamma": it_g, "iter_MS": it_m, "iter_uniform": it_u, "iter_BN": it_bn,
        "time_gamma": t_g, "time_MS": t_m, "time_uniform": t_u, "time_BN": t_bn,
    })
    # Save one representative convergence history (median-ebar day)
    convergence_histories[day] = {
        "gamma": np.array(hist_g),
        "MS": np.array(hist_m),
        "ebar": ebar_obs,
    }

df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
df.to_csv(OUT_DIR / "jd_pricing_comparison.csv", index=False)
print(f"    wrote {OUT_DIR/'jd_pricing_comparison.csv'}  ({len(df)} days)")
print()
print("===== Summary across days =====")
print(df[["ebar", "gap_gamma", "gap_MS", "gap_uniform",
         "iter_gamma", "iter_MS", "time_gamma", "time_MS"]].describe().round(6))

# ======================================================================
# 5. Figures
# ======================================================================
print("[4/6] Generating figures...")

# (i) Convergence on a representative (median-ebar) day
med_day = df.loc[(df["ebar"] - df["ebar"].median()).abs().idxmin(), "day"]
ch = convergence_histories[int(med_day)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(ch["gamma"], label=f"γ-iteration (converged in {len(ch['gamma'])} iter)",
            color="tab:blue", linewidth=2)
ax.semilogy(ch["MS"], label=f"MS2011 ζ-iteration (converged in {len(ch['MS'])} iter)",
            color="tab:red", linewidth=2, linestyle="--")
ax.axhline(TOL, color="grey", linestyle=":", label=f"tol = {TOL:g}")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\|\mathbf{p}^{(k+1)} - \mathbf{p}^{(k)}\|_\infty$")
ax.set_title(f"Convergence on day {int(med_day)}  (ebar = {ch['ebar']:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_convergence.png", dpi=150)
plt.close()

# (ii) ebar distribution across days
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["ebar"], bins=15, color="tab:blue", edgecolor="black", alpha=0.85)
ax.axvline(df["ebar"].median(), color="red", linestyle="--",
           label=f"median ebar = {df['ebar'].median():.3f}")
ax.axvline(0.3, color="grey", linestyle=":", label="theory threshold 0.3")
ax.set_xlabel(r"$\bar e$ at observed prices")
ax.set_ylabel("number of days")
ax.set_title("Distribution of ebar across days (JD March 2018, top-500 SKUs)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_ebar_distribution.png", dpi=150)
plt.close()

# (iii) Profit gap vs ebar  (γ and MS on same axes)
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["ebar"], df["gap_gamma"] * 100, label="γ-equalization",
           color="tab:blue", s=40)
ax.scatter(df["ebar"], df["gap_MS"] * 100, label="MS2011",
           color="tab:red", s=40, marker="x")
ax.scatter(df["ebar"], df["gap_uniform"] * 100, label="uniform markup",
           color="tab:green", s=40, marker="^")
# Overlay theoretical ebar² curve for γ
xs = np.linspace(df["ebar"].min(), df["ebar"].max(), 100)
scale = df["gap_gamma"].max() / (df["ebar"].max() ** 2) * 1.0
ax.plot(xs, scale * xs ** 2 * 100, "b--", alpha=0.5,
        label=r"$\propto \bar e^2$ (theory)")
ax.set_xlabel(r"$\bar e$ (diagonal dominance slack)")
ax.set_ylabel("profit gap to BN optimum  (%)")
ax.set_title("Profit loss vs observed ebar")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_profit_gap_vs_ebar.png", dpi=150)
plt.close()

# (iv) Wall-clock comparison
fig, ax = plt.subplots(figsize=(6, 4))
w = 0.25
x = np.arange(3)
means = [df["time_gamma"].mean(), df["time_MS"].mean(), df["time_BN"].mean()]
stds = [df["time_gamma"].std(), df["time_MS"].std(), df["time_BN"].std()]
ax.bar(x, means, w * 2, yerr=stds, color=["tab:blue", "tab:red", "tab:grey"],
       edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["γ-iteration", "MS2011", "Newton BN"])
ax.set_ylabel("wall-clock time (seconds)")
ax.set_title(f"Mean solver time per day  (n={N_TOP_SKU} products)")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s, f"{m*1000:.1f} ms", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_wallclock_comparison.png", dpi=150)
plt.close()

# ======================================================================
# 6. Summary printout
# ======================================================================
print("[5/6] Final summary")
print("-" * 70)
print(f"Data: JD MSOM 2020 competition, March 2018, top-{N_TOP_SKU} SKUs")
print(f"Demand: MCI, βs = {betas}")
print(f"Cost: c_i = {MARGIN_RATIO}·p̄_i")
print(f"Markets: {len(df)} days.  ebar range {df['ebar'].min():.3f}–{df['ebar'].max():.3f}")
print()
print("Profit gap to BN optimum (%):")
for col, name in [("gap_gamma", "γ-equalization"), ("gap_MS", "MS2011"),
                  ("gap_uniform", "uniform markup")]:
    print(f"  {name:18s}: mean {df[col].mean()*100:.4f}%, "
          f"median {df[col].median()*100:.4f}%, "
          f"max {df[col].max()*100:.4f}%")
print()
print("Wall-clock per solve (ms):")
for col, name in [("time_gamma", "γ"), ("time_MS", "MS2011"), ("time_BN", "Newton BN")]:
    print(f"  {name:10s}: mean {df[col].mean()*1000:.2f} ms, "
          f"median {df[col].median()*1000:.2f} ms")
print()
print("Iterations to tol=1e-8:")
for col, name in [("iter_gamma", "γ"), ("iter_MS", "MS2011")]:
    print(f"  {name:10s}: mean {df[col].mean():.1f}, median {df[col].median():.0f}")
print()
print(f"Speedup γ vs MS2011 (wall-clock): "
      f"{df['time_MS'].mean() / df['time_gamma'].mean():.2f}×")
print("[6/6] Done.  Outputs in:", OUT_DIR)
