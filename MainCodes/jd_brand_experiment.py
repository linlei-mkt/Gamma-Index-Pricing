"""
JD Real-Data Application: BRAND-LEVEL ELASTICITY ROBUSTNESS.

Companion to jd_experiment.py.  Same pipeline, same four pricing
methods (γ-equalization, MS2011 ζ-iteration, uniform markup, Newton
BN), but with elasticities estimated by brand rather than by
price-decile.  Brand is exogenous to the pricing process — a product
is born with a brand — so this specification sidesteps the concern
that price-decile bucketing conflates true elasticity heterogeneity
with selection into price tier.

Specification:
    Brands with ≥ MIN_SKU_PER_BRAND SKUs in the top-N set get their
    own price-slope β_b.  The remaining "tail" brands are pooled into
    one "other" bucket.  All other regression structure (SKU FE + day
    FE, pooled across 31 days) matches the price-decile script.

Outputs (in /Users/linlei/Downloads/Gamma/):
  - jd_brand_elasticities.csv      β per brand bucket + brand SKU counts
  - jd_brand_pricing_comparison.csv per-day γ/MS/uniform/BN results
  - jd_brand_ebar_distribution.png
  - jd_brand_profit_gap_vs_ebar.png
  - jd_brand_convergence.png
  - jd_brand_wallclock_comparison.png

To replicate:
  pip install pandas numpy scipy statsmodels matplotlib
  JD_DATA_DIR=/path/to/csvs python3 jd_brand_experiment.py
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
    ""
))
OUT_DIR = SCRIPT_DIR

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
N_TOP_SKU = 500
MIN_SKU_PER_BRAND = 7       # brands with ≥7 SKUs become own bucket (~20 brands)
USE_WLS = True              # weight regression by quantity to reduce heteroscedasticity
MARGIN_RATIO = 0.70
MARKET_MULT = 3.0
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

top_ids = (
    orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index.tolist()
)
o = orders[orders["sku_ID"].isin(top_ids)].merge(sku, on="sku_ID", how="left")
o["day"] = o["order_date"].dt.day

agg = (
    o.assign(rev=o["final_unit_price"] * o["quantity"])
     .groupby(["day", "sku_ID"], as_index=False)
     .agg(qty=("quantity", "sum"),
          rev=("rev", "sum"),
          nobs=("final_unit_price", "count"))
)
agg["price"] = agg["rev"] / agg["qty"]
agg = agg.merge(sku[["sku_ID", "type", "brand_ID"]], on="sku_ID", how="left")
agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["brand_ID"].notna()].copy()

peak_inside = agg.groupby("day")["qty"].sum().max()
M_pot = MARKET_MULT * peak_inside
print(f"    peak daily inside quantity={peak_inside:.0f}, potential market M={M_pot:.0f}")

daily_inside = agg.groupby("day")["qty"].sum().rename("Q_inside").to_frame()
daily_inside["s0"] = 1.0 - daily_inside["Q_inside"] / M_pot
daily_inside["S"] = 1.0 - daily_inside["s0"]
agg = agg.merge(daily_inside[["s0", "S"]], on="day")
agg["share"] = agg["qty"] / M_pot
print(f"    S across days: min={daily_inside['S'].min():.4f}, "
      f"median={daily_inside['S'].median():.4f}, "
      f"max={daily_inside['S'].max():.4f}")

# ======================================================================
# 2. Brand-bucket assignment
# ======================================================================
print(f"[2/6] Assigning brand buckets (MIN_SKU_PER_BRAND={MIN_SKU_PER_BRAND})...")
# Count SKUs per brand within the top-N set
brand_sku_counts = (
    agg.groupby("brand_ID")["sku_ID"].nunique().sort_values(ascending=False)
)
major_brands = brand_sku_counts[brand_sku_counts >= MIN_SKU_PER_BRAND].index.tolist()
n_major = len(major_brands)
print(f"    {n_major} major brands (≥{MIN_SKU_PER_BRAND} SKUs each), "
      f"{len(brand_sku_counts) - n_major} pooled into 'other'")

# Map brand_ID → bucket label.  Major brands get 0..n_major-1; others get n_major ('other').
brand_to_bucket = {b: i for i, b in enumerate(major_brands)}
OTHER_BUCKET = n_major
agg["bucket"] = agg["brand_ID"].map(brand_to_bucket).fillna(OTHER_BUCKET).astype(int)
n_buckets = n_major + 1
print(f"    total brand buckets (including 'other'): {n_buckets}")
print(f"    SKUs per bucket:")
bucket_sku = agg.groupby("bucket")["sku_ID"].nunique()
for b in range(n_buckets):
    label = major_brands[b][:10] + "…" if b < n_major else "other"
    print(f"      bucket {b} ({label}): n_sku={bucket_sku.get(b, 0)}")

# ======================================================================
# 3. MCI demand estimation with brand-bucket elasticity
# ======================================================================
print("[3/6] Estimating MCI demand (brand-bucket β)...")
agg["y"] = np.log(agg["share"]) - np.log(agg["s0"])
agg["log_price"] = np.log(agg["price"])

X = pd.get_dummies(
    agg[["sku_ID", "day"]].astype(str),
    columns=["sku_ID", "day"],
    drop_first=True,
    dtype=float,
)
for b in range(n_buckets):
    X[f"logp_b{b}"] = agg["log_price"] * (agg["bucket"] == b).astype(float)
X = sm.add_constant(X)
y = agg["y"].astype(float)

print(f"    regression dimensions: n={len(y)}, k={X.shape[1]}")
if USE_WLS:
    # Weight each observation by sqrt(qty) -- equivalently, quantity-weighted
    # log-share regression.  Reduces influence of noisy low-volume cells.
    weights = agg["qty"].astype(float).to_numpy()
    mod = sm.WLS(y, X, weights=weights).fit()
    print(f"    (quantity-weighted least squares) R² = {mod.rsquared:.4f}")
else:
    mod = sm.OLS(y, X).fit()
    print(f"    R² = {mod.rsquared:.4f}")

betas = {}
for b in range(n_buckets):
    col = f"logp_b{b}"
    if col in mod.params.index:
        betas[b] = -mod.params[col]

print("    estimated β by brand bucket:")
for b in range(n_buckets):
    label = major_brands[b] if b < n_major else "other"
    count = bucket_sku.get(b, 0)
    print(f"      bucket {b} ({label[:10]}, n_sku={count}): β = {betas[b]:.3f}")

# Clamp any estimate below floor
beta_floor = 1.2
clamped = []
for b in list(betas.keys()):
    if betas[b] < beta_floor:
        clamped.append((b, betas[b]))
        betas[b] = beta_floor
if clamped:
    print(f"    clamped {len(clamped)} bucket(s) to floor {beta_floor}: {clamped}")

# Save elasticity table.  Note: `beta` is the MCI attraction exponent
# (positive by convention).  The own-price elasticity under MCI is
# η_i = -β_i · (1 - s_i), which is negative.  The `own_elasticity_typical`
# column evaluates η at the median observed share in each bucket to make
# the negative sign explicit.
median_share_by_bucket = agg.groupby("bucket")["share"].median()
brand_rows = []
for b in range(n_buckets):
    label = major_brands[b] if b < n_major else "other"
    n_sku_b = int(bucket_sku.get(b, 0))
    s_med = float(median_share_by_bucket.get(b, 0.001))
    # Typical own elasticity (note the negative sign)
    eta_typical = -betas[b] * (1.0 - s_med)
    brand_rows.append({
        "bucket": b,
        "brand_label": label,
        "is_pooled_other": b == OTHER_BUCKET,
        "n_sku": n_sku_b,
        "beta_MCI_exponent": betas[b],        # positive by convention
        "median_share": s_med,
        "own_elasticity_typical": eta_typical,  # -β(1-s), negative
    })
pd.DataFrame(brand_rows).assign(R2=mod.rsquared, n_obs=len(y)).to_csv(
    OUT_DIR / "jd_brand_elasticities.csv", index=False
)

# ======================================================================
# 4. Pricing solvers  (identical to jd_experiment.py)
# ======================================================================
def mci_shares(p, alpha, beta, M):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def calibrate_alpha(p_obs, s_obs, s0_obs, beta):
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def share_jacobian(p, s, beta):
    """Ω_{ij} = ∂s_j/∂p_i  (MCI)."""
    u = beta * s / p
    Om = np.outer(u, s)
    np.fill_diagonal(Om, -u * (1.0 - s))
    return Om


def ebar_from_shares(s):
    S = s.sum()
    return float(np.max((S - s) / (1.0 - s)))


def gamma_iteration(p0, c, alpha, beta, M, gamma_star=0.0,
                    tol=TOL, max_iter=MAX_ITER):
    p = p0.copy()
    hist = []
    t0 = time.perf_counter()
    for k in range(max_iter):
        s, s0 = mci_shares(p, alpha, beta, M)
        eta = beta * (1.0 - s)
        eta_safe = np.maximum(eta, 1.01)
        p_new = c / (1.0 - gamma_star - (1.0 - gamma_star) / eta_safe)
        p_new = np.maximum(p_new, c * 1.0001)
        diff = float(np.max(np.abs(p_new - p)))
        hist.append(diff)
        p = p_new
        if diff < tol:
            break
    wall = time.perf_counter() - t0
    return p, k + 1, wall, hist


def ms_iteration(p0, c, alpha, beta, M, tol=TOL, max_iter=MAX_ITER):
    p = p0.copy()
    hist = []
    t0 = time.perf_counter()
    for k in range(max_iter):
        s, s0 = mci_shares(p, alpha, beta, M)
        Om = share_jacobian(p, s, beta)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
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


def newton_bn_robust(p0, c, alpha, beta, M):
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
    wall = time.perf_counter() - t0
    m_star = res.x
    p = c / (1.0 - m_star)
    return p, res.nit if hasattr(res, "nit") else 50, wall, m_star


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# 5. Per-day experiment
# ======================================================================
print("[4/6] Running per-day pricing experiments...")
rows = []
convergence_histories = {}
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
    alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
    c = MARGIN_RATIO * p_obs
    M = M_pot
    ebar_obs = ebar_from_shares(s_obs)
    p0 = p_obs.copy()

    p_g, it_g, t_g, hist_g = gamma_iteration(p0, c, alpha, beta_vec, M)
    p_m, it_m, t_m, hist_m = ms_iteration(p0, c, alpha, beta_vec, M)
    p_u, it_u, t_u, m_star = uniform_pricing(c, alpha, beta_vec, M)
    try:
        p_bn, it_bn, t_bn = newton_bn_robust(p_m.copy(), c, alpha, beta_vec, M)
    except Exception:
        p_bn, it_bn, t_bn = p_m.copy(), 0, 0.0

    pi_g = total_profit(p_g, c, alpha, beta_vec, M)
    pi_m = total_profit(p_m, c, alpha, beta_vec, M)
    pi_u = total_profit(p_u, c, alpha, beta_vec, M)
    pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)

    def gap(pi):
        return max(0.0, (pi_bn - pi) / pi_bn)

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
    convergence_histories[day] = {
        "gamma": np.array(hist_g),
        "MS": np.array(hist_m),
        "ebar": ebar_obs,
    }

df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
df.to_csv(OUT_DIR / "jd_brand_pricing_comparison.csv", index=False)
print(f"    wrote {OUT_DIR/'jd_brand_pricing_comparison.csv'}  ({len(df)} days)")
print()
print("===== Summary across days (brand-level β) =====")
print(df[["ebar", "gap_gamma", "gap_MS", "gap_uniform",
         "iter_gamma", "iter_MS", "time_gamma", "time_MS"]].describe().round(6))

# ======================================================================
# 6. Figures
# ======================================================================
print("[5/6] Generating figures...")
med_day = df.loc[(df["ebar"] - df["ebar"].median()).abs().idxmin(), "day"]
ch = convergence_histories[int(med_day)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(ch["gamma"], label=f"γ-iteration ({len(ch['gamma'])} iter)",
            color="tab:blue", linewidth=2)
ax.semilogy(ch["MS"], label=f"MS2011 ({len(ch['MS'])} iter)",
            color="tab:red", linewidth=2, linestyle="--")
ax.axhline(TOL, color="grey", linestyle=":", label=f"tol = {TOL:g}")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\|p^{(k+1)}-p^{(k)}\|_\infty$")
ax.set_title(f"Convergence on day {int(med_day)}  (ebar={ch['ebar']:.3f}, brand β)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_brand_convergence.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["ebar"], bins=15, color="tab:blue", edgecolor="black", alpha=0.85)
ax.axvline(df["ebar"].median(), color="red", linestyle="--",
           label=f"median ebar={df['ebar'].median():.3f}")
ax.axvline(0.3, color="grey", linestyle=":", label="theory threshold 0.3")
ax.set_xlabel(r"$\bar e$ at observed prices")
ax.set_ylabel("number of days")
ax.set_title("ebar distribution across days (brand-level β specification)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_brand_ebar_distribution.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["ebar"], df["gap_gamma"] * 100, label="γ-equalization",
           color="tab:blue", s=40)
ax.scatter(df["ebar"], df["gap_MS"] * 100, label="MS2011",
           color="tab:red", s=40, marker="x")
ax.scatter(df["ebar"], df["gap_uniform"] * 100, label="uniform markup",
           color="tab:green", s=40, marker="^")
xs = np.linspace(df["ebar"].min(), df["ebar"].max(), 100)
scale = df["gap_gamma"].max() / (df["ebar"].max() ** 2)
ax.plot(xs, scale * xs ** 2 * 100, "b--", alpha=0.5,
        label=r"$\propto \bar e^2$ (theory)")
ax.set_xlabel(r"$\bar e$")
ax.set_ylabel("profit gap to BN optimum (%)")
ax.set_title("Profit loss vs ebar (brand-level β)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_brand_profit_gap_vs_ebar.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(3)
means = [df["time_gamma"].mean(), df["time_MS"].mean(), df["time_BN"].mean()]
stds = [df["time_gamma"].std(), df["time_MS"].std(), df["time_BN"].std()]
ax.bar(x, means, 0.5, yerr=stds, color=["tab:blue", "tab:red", "tab:grey"],
       edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["γ-iteration", "MS2011", "Newton BN"])
ax.set_ylabel("wall-clock time (seconds)")
ax.set_title(f"Mean solver time per day  (brand-level β, n={N_TOP_SKU})")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s, f"{m*1000:.1f} ms", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "jd_brand_wallclock_comparison.png", dpi=150)
plt.close()

# ======================================================================
# 7. Summary
# ======================================================================
print("[6/6] Final summary (brand-level β)")
print("-" * 70)
print(f"Data: JD MSOM 2020, March 2018, top-{N_TOP_SKU} SKUs")
print(f"Demand: MCI, {n_buckets} brand buckets "
      f"({n_major} major + 1 pooled 'other')")
print(f"Cost: c_i = {MARGIN_RATIO}·p̄_i")
print(f"Markets: {len(df)} days.  ebar range "
      f"{df['ebar'].min():.3f}–{df['ebar'].max():.3f}")
print()
print("Profit gap to BN optimum (%):")
for col, name in [("gap_gamma", "γ-equalization"), ("gap_MS", "MS2011"),
                  ("gap_uniform", "uniform markup")]:
    print(f"  {name:18s}: mean {df[col].mean()*100:.4f}%, "
          f"median {df[col].median()*100:.4f}%, "
          f"max {df[col].max()*100:.4f}%")
print()
print("Wall-clock per solve (ms):")
for col, name in [("time_gamma", "γ"), ("time_MS", "MS2011"),
                  ("time_BN", "Newton BN")]:
    print(f"  {name:10s}: mean {df[col].mean()*1000:.2f} ms, "
          f"median {df[col].median()*1000:.2f} ms")
print()
print("Iterations to tol=1e-8:")
for col, name in [("iter_gamma", "γ"), ("iter_MS", "MS2011")]:
    print(f"  {name:10s}: mean {df[col].mean():.1f}, "
          f"median {df[col].median():.0f}")
print()
print(f"γ always beats uniform? "
      f"{(df['gap_gamma'] < df['gap_uniform']).all()}")
print(f"γ gap < ebar² in fraction of markets: "
      f"{(df['gap_gamma'] < df['ebar']**2).mean():.1%}")
print(f"Speedup γ vs MS2011 (wall-clock): "
      f"{df['time_MS'].mean() / df['time_gamma'].mean():.2f}×")
print("Done.  Outputs in:", OUT_DIR)
