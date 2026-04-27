"""
JD real-data HIERARCHICAL BAYESIAN elasticity estimation.

Estimates a per-product MCI exponent β_i with partial pooling toward a
price-decile bucket mean.  This addresses the concern that 10 bucket-
level elasticities mask meaningful within-bucket heterogeneity, while
avoiding the variance explosion of 500 independent frequentist slopes.

Model
-----
    log(s_{it} / s_{0t}) = α_i + δ_t - β_i · log(p_{it}) + ε_{it}

        α_i  ~ Normal(0, 5)                       SKU fixed effect
        δ_t  ~ Normal(0, 1)                        day fixed effect
        β_i  = μ_{b(i)} + u_i                      per-product slope
        μ_b  ~ Normal(2.0, 1.0)                    bucket-mean prior
        u_i  ~ Normal(0, τ)                        within-bucket deviation
        τ    ~ HalfNormal(0.5)                     shrinkage strength
        ε_it ~ Normal(0, σ),  σ ~ HalfNormal(1)

The posterior mean of each β_i borrows strength from other products in
the same price-decile bucket: small-sample products are shrunk toward
μ_bucket, large-sample products can deviate freely.  This is the
standard hierarchical / partial-pooling remedy for "many products, thin
data per product."

After sampling, we plug the posterior-mean β̂_i into the four pricing
solvers (γ / MS2011 / uniform / Newton BN) and redo the per-day
comparison.  If the headline results (γ-gap scales as ebar², γ beats
uniform, γ is ~100-200× faster than MS) survive partial pooling at
the product level, that closes the "are you over-smoothing?" concern.

Outputs (in /Users/linlei/Downloads/Gamma/):
  - jd_hb_posterior_summary.csv     β_i posterior mean/sd for each SKU
  - jd_hb_bucket_means.csv          μ_b posterior (bucket-level slope means)
  - jd_hb_trace_plots.png           MCMC trace plots (diagnostic)
  - jd_hb_shrinkage.png             OLS β_i vs HB posterior β_i scatter
  - jd_hb_pricing_comparison.csv    per-day pricing results under HB β̂
  - jd_hb_profit_gap_vs_ebar.png    γ/MS/uniform profit-gap scatter

Required packages:
    pip install pymc arviz pandas numpy scipy matplotlib statsmodels
    # Optional (faster sampling):
    pip install nutpie

Runtime: 10-30 minutes on a laptop depending on cores and samplers.
To speed up, set `target_accept=0.8` (default 0.9) and `draws=500` below.

To run:
    JD_DATA_DIR=/path/to/csvs python3 jd_hierarchical_bayes.py
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

# PyMC is optional at import time so the user gets a helpful error if missing
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    ""
))
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 500
N_DECILES = 10
M_MULT = 3.0
MARGIN_RATIO = 0.70
BETA_FLOOR = 1.2
TOL = 1e-8
MAX_ITER = 2000
SEED = 2026

# Sampler config (trade-off speed vs. mixing)
DRAWS = 1000
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.85


# ======================================================================
# Data prep
# ======================================================================
def load_and_aggregate():
    """Return (agg dataframe, bucket map, M)."""
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
    agg = agg.merge(sku[["sku_ID", "type", "brand_ID"]], on="sku_ID", how="left")
    agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
    agg["type"] = agg["type"].astype(int)

    # Bucket by price decile
    sku_mean_p = agg.groupby("sku_ID")["price"].mean()
    bucket = pd.qcut(
        sku_mean_p.rank(method="first"),
        q=N_DECILES, labels=False,
    ).astype(int)
    agg = agg.merge(bucket.rename("bucket"), left_on="sku_ID",
                    right_index=True, how="left")

    # Potential market size
    peak_inside = agg.groupby("day")["qty"].sum().max()
    M = M_MULT * peak_inside
    agg["share"] = agg["qty"] / M
    daily_Q = agg.groupby("day")["qty"].sum().rename("Q_in")
    s0 = (1.0 - daily_Q / M).rename("s0")
    agg = agg.merge(s0, on="day")
    return agg, M


# ======================================================================
# Hierarchical model
# ======================================================================
def fit_hierarchical_bayes(agg):
    """PyMC hierarchical MCI elasticity model. Returns InferenceData."""
    if not HAS_PYMC:
        raise ImportError(
            "PyMC not installed. Run:  pip install pymc arviz"
        )

    # Integer codes for FE / hierarchy
    sku_codes, sku_uniques = pd.factorize(agg["sku_ID"], sort=True)
    day_codes, day_uniques = pd.factorize(agg["day"], sort=True)
    bucket_codes = agg["bucket"].astype(int).to_numpy()
    # SKU -> bucket mapping (for hierarchy)
    sku_to_bucket = (
        agg.drop_duplicates("sku_ID").set_index("sku_ID").loc[sku_uniques, "bucket"]
           .astype(int).to_numpy()
    )

    y = (np.log(agg["share"]) - np.log(agg["s0"])).to_numpy()
    logp = np.log(agg["price"].to_numpy())

    n_sku = len(sku_uniques)
    n_day = len(day_uniques)
    n_buckets = N_DECILES

    print(f"    model dims: n_obs={len(y)}, n_sku={n_sku}, n_day={n_day}, "
          f"n_buckets={n_buckets}")

    with pm.Model() as model:
        # Hyperpriors on bucket-level slope means
        mu_b = pm.Normal("mu_bucket", mu=2.0, sigma=1.0, shape=n_buckets)
        # Shared within-bucket shrinkage scale
        tau = pm.HalfNormal("tau", sigma=0.5)
        # Per-SKU slope deviations (non-centered parameterization for mixing)
        u_std = pm.Normal("u_std", mu=0.0, sigma=1.0, shape=n_sku)
        # Per-SKU slope: μ_bucket + τ·u_std
        beta_sku = pm.Deterministic(
            "beta_sku",
            mu_b[sku_to_bucket] + tau * u_std,
        )
        # SKU and day fixed effects
        alpha_sku = pm.Normal("alpha_sku", mu=0.0, sigma=5.0, shape=n_sku)
        delta_day = pm.Normal("delta_day", mu=0.0, sigma=1.0, shape=n_day)
        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Linear predictor: log(s/s0) = α_i + δ_t − β_i · log(p)
        mu = (
            alpha_sku[sku_codes]
            + delta_day[day_codes]
            - beta_sku[sku_codes] * logp
        )
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        print(f"    starting MCMC: draws={DRAWS}, tune={TUNE}, chains={CHAINS}")
        t0 = time.time()
        # Prefer nutpie if available (faster), else PyMC's NUTS
        try:
            import nutpie  # noqa
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                nuts_sampler="nutpie", progressbar=True,
            )
            print("    (used nutpie sampler)")
        except ImportError:
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                progressbar=True,
            )
        print(f"    MCMC done in {(time.time()-t0)/60:.1f} min")

    return idata, sku_uniques, day_uniques, sku_to_bucket


# ======================================================================
# Pricing solvers (same as jd_experiment.py)
# ======================================================================
def mci_shares(p, alpha, beta, M):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def calibrate_alpha_mci(p_obs, s_obs, s0_obs, beta):
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
    res = minimize_scalar(neg_profit, bounds=(0.001, 0.999), method="bounded",
                          options={"xatol": 1e-8})
    return c / (1.0 - res.x)


def total_profit(p, c, alpha, beta, M):
    s, _ = mci_shares(p, alpha, beta, M)
    return float(np.sum((p - c) * s) * M)


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("JD HIERARCHICAL BAYES elasticity estimation")
    print("=" * 70)
    if not HAS_PYMC:
        print("ERROR: PyMC not installed.")
        print("  pip install pymc arviz nutpie")
        return

    print("[1/4] Loading data...")
    agg, M = load_and_aggregate()
    print(f"    {len(agg)} observations, M = {M:.0f}")

    print("[2/4] Fitting hierarchical Bayesian model (this may take a while)...")
    idata, sku_uniques, day_uniques, sku_to_bucket = fit_hierarchical_bayes(agg)

    # ---- Posterior summary ----
    print("[3/4] Extracting posterior means...")
    post_beta = idata.posterior["beta_sku"].stack(sample=("chain", "draw"))
    beta_mean = post_beta.mean(dim="sample").values
    beta_sd = post_beta.std(dim="sample").values

    post_mu = idata.posterior["mu_bucket"].stack(sample=("chain", "draw"))
    mu_mean = post_mu.mean(dim="sample").values
    mu_sd = post_mu.std(dim="sample").values
    tau_mean = float(idata.posterior["tau"].mean().values)

    print(f"    bucket means μ_b: {np.array2string(mu_mean, precision=2)}")
    print(f"    within-bucket shrinkage τ = {tau_mean:.3f}")

    # Save posterior summaries
    pd.DataFrame({
        "sku_ID": sku_uniques,
        "bucket": sku_to_bucket,
        "beta_posterior_mean": beta_mean,
        "beta_posterior_sd": beta_sd,
    }).to_csv(OUT_DIR / "jd_hb_posterior_summary.csv", index=False)

    pd.DataFrame({
        "bucket": np.arange(N_DECILES),
        "mu_posterior_mean": mu_mean,
        "mu_posterior_sd": mu_sd,
        "tau_within_bucket": tau_mean,
    }).to_csv(OUT_DIR / "jd_hb_bucket_means.csv", index=False)

    # ---- Diagnostic plots ----
    try:
        axes = az.plot_trace(
            idata, var_names=["mu_bucket", "tau", "sigma"],
            compact=True, combined=True,
        )
        fig = axes.ravel()[0].figure
        fig.tight_layout()
        fig.savefig(OUT_DIR / "jd_hb_trace_plots.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"    (trace plot failed: {e})")

    # ---- Shrinkage visualization: OLS β_i vs HB posterior β_i ----
    # Quick OLS per-SKU (demeaned, within-SKU variation only)
    import statsmodels.api as sm
    ols_beta = np.full(len(sku_uniques), np.nan)
    for i, sku_id in enumerate(sku_uniques):
        sub = agg[agg["sku_ID"] == sku_id]
        if len(sub) < 5 or sub["price"].std() < 0.01:
            continue
        y_i = np.log(sub["share"]) - np.log(sub["s0"])
        x_i = np.log(sub["price"])
        # within-SKU: demean y and x
        y_d = y_i - y_i.mean()
        x_d = x_i - x_i.mean()
        if x_d.var() < 1e-10:
            continue
        b = -(x_d * y_d).sum() / (x_d ** 2).sum()
        ols_beta[i] = b
    mask = np.isfinite(ols_beta)
    if mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ols_beta[mask], beta_mean[mask], alpha=0.5, s=20)
        lo, hi = min(ols_beta[mask].min(), beta_mean.min()), \
                 max(ols_beta[mask].max(), beta_mean.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        # Shrinkage toward bucket means
        for b in range(N_DECILES):
            ax.axhline(mu_mean[b], color="red", alpha=0.1, linestyle=":")
        ax.set_xlabel(r"OLS $\hat\beta_i$ (per-product, within-SKU)")
        ax.set_ylabel(r"Hierarchical Bayes posterior mean $\hat\beta_i$")
        ax.set_title("Shrinkage: noisy OLS β_i pulled toward bucket means")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "jd_hb_shrinkage.png", dpi=150)
        plt.close()

    # ---- Pricing comparison under HB β̂ ----
    print("[4/4] Running per-day pricing comparison with HB β̂...")
    sku_to_beta = dict(zip(sku_uniques, np.maximum(beta_mean, BETA_FLOOR)))
    rows = []
    for day in sorted(agg["day"].unique()):
        mkt = agg[agg["day"] == day].copy()
        if len(mkt) < 50:
            continue
        p_obs = mkt["price"].to_numpy()
        s_obs = mkt["share"].to_numpy()
        s0_obs = float(mkt["s0"].iloc[0])
        beta_vec = mkt["sku_ID"].map(sku_to_beta).to_numpy()
        alpha = calibrate_alpha_mci(p_obs, s_obs, s0_obs, beta_vec)
        c = MARGIN_RATIO * p_obs
        p0 = p_obs.copy()
        ebar = ebar_from_shares(s_obs)

        p_g, it_g, t_g = gamma_iteration(p0, c, alpha, beta_vec, M)
        p_m, it_m, t_m = ms_iteration(p0, c, alpha, beta_vec, M)
        p_u = uniform_pricing(c, alpha, beta_vec, M)
        try:
            p_bn, it_bn, t_bn = newton_bn(p_m.copy(), c, alpha, beta_vec, M)
        except Exception:
            p_bn, it_bn, t_bn = p_m.copy(), 0, 0.0
        pi_bn = total_profit(p_bn, c, alpha, beta_vec, M)
        if pi_bn <= 0:
            continue
        gap = lambda p: max(0.0, (pi_bn - total_profit(p, c, alpha, beta_vec, M)) / pi_bn)
        rows.append({
            "day": int(day), "ebar": ebar,
            "gap_gamma": gap(p_g), "gap_MS": gap(p_m), "gap_uniform": gap(p_u),
            "iter_gamma": it_g, "iter_MS": it_m,
            "time_gamma": t_g, "time_MS": t_m,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "jd_hb_pricing_comparison.csv", index=False)

    # Scatter: gap vs ebar
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df["ebar"], df["gap_gamma"] * 100, label="γ (HB β̂)",
               color="tab:blue", s=30)
    ax.scatter(df["ebar"], df["gap_MS"] * 100, label="MS2011",
               color="tab:red", s=30, marker="x")
    ax.scatter(df["ebar"], df["gap_uniform"] * 100, label="uniform",
               color="tab:green", s=30, marker="^")
    xs = np.linspace(df["ebar"].min(), df["ebar"].max(), 100)
    mask = df["ebar"] > 0
    if mask.any():
        c_fit = (df.loc[mask, "gap_gamma"] * df.loc[mask, "ebar"] ** 2).sum() \
                / (df.loc[mask, "ebar"] ** 4).sum()
        ax.plot(xs, c_fit * xs ** 2 * 100, "b--", alpha=0.5,
                label=fr"γ gap ≈ {c_fit:.2f}·ebar²")
    ax.set_xlabel(r"$\bar e$")
    ax.set_ylabel("profit gap to BN (%)")
    ax.set_title("Pricing comparison under hierarchical-Bayes β̂")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_hb_profit_gap_vs_ebar.png", dpi=150)
    plt.close()

    # Summary
    print()
    print("=" * 70)
    print("Summary under HB β̂:")
    print(f"  γ-gap   : mean {df['gap_gamma'].mean()*100:.3f}%, "
          f"median {df['gap_gamma'].median()*100:.3f}%, "
          f"max {df['gap_gamma'].max()*100:.3f}%")
    print(f"  MS-gap  : {df['gap_MS'].mean()*100:.4f}%")
    print(f"  unif-gap: mean {df['gap_uniform'].mean()*100:.3f}%, "
          f"median {df['gap_uniform'].median()*100:.3f}%")
    print(f"  γ beats uniform in {(df['gap_gamma'] < df['gap_uniform']).mean()*100:.1f}% of days")
    print(f"  γ-gap < ebar² in {(df['gap_gamma'] < df['ebar']**2).mean()*100:.1f}% of days")
    print(f"  speedup γ vs MS: {df['time_MS'].mean() / df['time_gamma'].mean():.1f}×")
    print()
    print("All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
