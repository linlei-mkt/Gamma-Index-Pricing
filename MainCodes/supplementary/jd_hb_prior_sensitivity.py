"""
JD HB-MCI prior sensitivity check.

Re-fits the hierarchical Bayesian MCI elasticity model of
`jd_hierarchical_bayes.py` under three priors on the within-bucket
shrinkage scale τ, and compares the resulting posterior τ and per-SKU
β_i estimates.

Specifications
--------------
    Baseline  : τ ~ HalfNormal(0.5)        (the paper's main spec)
    Wider     : τ ~ HalfNormal(1.0)        (looser shrinkage)
    Heavy-tail: τ ~ HalfCauchy(0.5)        (allows large τ if data demands)

The other priors (μ_bucket, α_sku, δ_day, σ) are held fixed at the
baseline values so the comparison is clean.

Why this matters
----------------
HalfNormal(0.5) is a mild shrinkage prior that is appropriate when the
posterior τ is well-identified. HalfCauchy and the wider HalfNormal are
the two standard alternatives: HalfCauchy lets τ go large if the data
support it (heavier right tail), and HalfNormal(1.0) is the same family
with a less informative scale. If the posterior τ and β_i estimates
move materially across these three priors, the partial-pooling result
of the paper is prior-driven; if they barely move, the baseline is
robust.

Outputs (in script directory):
  - jd_hb_prior_sensitivity_summary.csv   τ posterior summary, all 3 specs
  - jd_hb_prior_sensitivity_betas.csv     per-SKU posterior β̂_i, all 3 specs
  - jd_hb_prior_sensitivity_tau_density.png
  - jd_hb_prior_sensitivity_beta_scatter.png

Required packages: same as jd_hierarchical_bayes.py
    pip install pymc arviz pandas numpy scipy matplotlib

Runtime: 15-25 minutes total (3 specs × 5-8 min each).
To run faster, set DRAWS=300, TUNE=300 below.

To run:
    JD_DATA_DIR=/path/to/csvs python3 jd_hb_prior_sensitivity.py
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

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("JD_DATA_DIR", SCRIPT_DIR.parent / "JD_MSOM"))
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 500
N_DECILES = 10
M_MULT = 3.0
SEED = 2026

# Sampler config — shorter chains than baseline since this is a
# sensitivity check, not the headline estimation
DRAWS = 500
TUNE = 500
CHAINS = 2
TARGET_ACCEPT = 0.85

# Three prior specifications on τ
PRIOR_SPECS = [
    ("baseline_HN05",   "HalfNormal(0.5)",  "HalfNormal", 0.5),
    ("wider_HN10",      "HalfNormal(1.0)",  "HalfNormal", 1.0),
    ("heavy_HC05",      "HalfCauchy(0.5)",  "HalfCauchy", 0.5),
]


# ======================================================================
# Data prep (replicates jd_hierarchical_bayes.py exactly)
# ======================================================================
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
    agg = agg.merge(sku[["sku_ID", "type", "brand_ID"]], on="sku_ID", how="left")
    agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
    agg["type"] = agg["type"].astype(int)

    sku_mean_p = agg.groupby("sku_ID")["price"].mean()
    bucket = pd.qcut(
        sku_mean_p.rank(method="first"),
        q=N_DECILES, labels=False,
    ).astype(int)
    agg = agg.merge(bucket.rename("bucket"), left_on="sku_ID",
                    right_index=True, how="left")

    peak_inside = agg.groupby("day")["qty"].sum().max()
    M = M_MULT * peak_inside
    agg["share"] = agg["qty"] / M
    daily_Q = agg.groupby("day")["qty"].sum().rename("Q_in")
    s0 = (1.0 - daily_Q / M).rename("s0")
    agg = agg.merge(s0, on="day")
    return agg, M


# ======================================================================
# Hierarchical model with switchable τ-prior
# ======================================================================
def fit_hb_with_tau_prior(agg, prior_family: str, prior_scale: float, label: str):
    """Fit HB-MCI under a chosen prior on τ. Returns InferenceData + book-keeping."""
    sku_codes, sku_uniques = pd.factorize(agg["sku_ID"], sort=True)
    day_codes, day_uniques = pd.factorize(agg["day"], sort=True)
    sku_to_bucket = (
        agg.drop_duplicates("sku_ID").set_index("sku_ID").loc[sku_uniques, "bucket"]
           .astype(int).to_numpy()
    )

    y = (np.log(agg["share"]) - np.log(agg["s0"])).to_numpy()
    logp = np.log(agg["price"].to_numpy())

    n_sku = len(sku_uniques)
    n_day = len(day_uniques)
    n_buckets = N_DECILES

    print(f"  [{label}] dims: n_obs={len(y)}, n_sku={n_sku}, "
          f"n_day={n_day}, n_buckets={n_buckets}")

    with pm.Model() as model:
        mu_b = pm.Normal("mu_bucket", mu=2.0, sigma=1.0, shape=n_buckets)

        # ---- the only line that varies across specs ----
        if prior_family == "HalfNormal":
            tau = pm.HalfNormal("tau", sigma=prior_scale)
        elif prior_family == "HalfCauchy":
            tau = pm.HalfCauchy("tau", beta=prior_scale)
        else:
            raise ValueError(f"Unknown prior family: {prior_family}")

        u_std = pm.Normal("u_std", mu=0.0, sigma=1.0, shape=n_sku)
        beta_sku = pm.Deterministic(
            "beta_sku",
            mu_b[sku_to_bucket] + tau * u_std,
        )
        alpha_sku = pm.Normal("alpha_sku", mu=0.0, sigma=5.0, shape=n_sku)
        delta_day = pm.Normal("delta_day", mu=0.0, sigma=1.0, shape=n_day)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = (
            alpha_sku[sku_codes]
            + delta_day[day_codes]
            - beta_sku[sku_codes] * logp
        )
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        t0 = time.time()
        try:
            import nutpie  # noqa
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                nuts_sampler="nutpie", progressbar=True,
            )
        except ImportError:
            idata = pm.sample(
                draws=DRAWS, tune=TUNE, chains=CHAINS,
                target_accept=TARGET_ACCEPT, random_seed=SEED,
                progressbar=True,
            )
        print(f"  [{label}] MCMC done in {(time.time()-t0)/60:.1f} min")

    return idata, sku_uniques, sku_to_bucket


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("JD HB-MCI prior sensitivity check (τ prior)")
    print("=" * 70)
    if not HAS_PYMC:
        print("ERROR: PyMC not installed. Run:  pip install pymc arviz nutpie")
        return

    print("[1] Loading data...")
    agg, M = load_and_aggregate()
    print(f"    {len(agg)} observations, M = {M:.0f}")

    summary_rows = []
    beta_table = None  # will hold per-SKU β̂ across specs

    for spec_id, spec_label, fam, scale in PRIOR_SPECS:
        print()
        print(f"[2] Fitting spec: {spec_label}  (id={spec_id})")
        idata, sku_uniques, sku_to_bucket = fit_hb_with_tau_prior(
            agg, fam, scale, spec_label,
        )

        # --- τ posterior ---
        tau_post = idata.posterior["tau"].stack(sample=("chain", "draw")).values
        tau_mean = float(tau_post.mean())
        tau_sd = float(tau_post.std())
        tau_q05 = float(np.quantile(tau_post, 0.05))
        tau_q50 = float(np.quantile(tau_post, 0.50))
        tau_q95 = float(np.quantile(tau_post, 0.95))

        # --- β_i posterior means ---
        post_beta = idata.posterior["beta_sku"].stack(sample=("chain", "draw"))
        beta_mean = post_beta.mean(dim="sample").values
        beta_sd = post_beta.std(dim="sample").values

        # --- σ and bucket-mean posteriors (for sanity check) ---
        sigma_mean = float(idata.posterior["sigma"].mean().values)
        mu_b_post = idata.posterior["mu_bucket"].stack(sample=("chain", "draw"))
        mu_b_mean = mu_b_post.mean(dim="sample").values

        # --- R-hat for τ as MCMC-quality check ---
        try:
            rhat_tau = float(az.rhat(idata, var_names=["tau"]).tau.values)
        except Exception:
            rhat_tau = np.nan

        summary_rows.append({
            "spec_id": spec_id,
            "tau_prior": spec_label,
            "tau_post_mean": tau_mean,
            "tau_post_sd": tau_sd,
            "tau_post_q05": tau_q05,
            "tau_post_q50": tau_q50,
            "tau_post_q95": tau_q95,
            "tau_rhat": rhat_tau,
            "sigma_post_mean": sigma_mean,
            "beta_post_mean_avg": float(beta_mean.mean()),
            "beta_post_mean_sd": float(beta_mean.std()),
            "n_sku": len(sku_uniques),
        })

        # save per-SKU β̂ in a wide table
        df_b = pd.DataFrame({
            "sku_ID": sku_uniques,
            "bucket": sku_to_bucket,
            f"beta_{spec_id}_mean": beta_mean,
            f"beta_{spec_id}_sd": beta_sd,
        })
        if beta_table is None:
            beta_table = df_b
        else:
            beta_table = beta_table.merge(
                df_b.drop(columns=["bucket"]), on="sku_ID", how="outer",
            )

        # save raw τ draws for the density plot
        np.save(OUT_DIR / f"_tau_draws_{spec_id}.npy", tau_post)

    # ---- Save summary CSVs ----
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "jd_hb_prior_sensitivity_summary.csv", index=False,
    )
    beta_table.to_csv(
        OUT_DIR / "jd_hb_prior_sensitivity_betas.csv", index=False,
    )

    # ---- Quantitative check: pairwise β̂_i correlation across specs ----
    print()
    print("=" * 70)
    print("τ posterior summary across priors:")
    print("-" * 70)
    print(f"{'spec':<24} {'mean':>8} {'sd':>8} {'5%':>8} {'50%':>8} {'95%':>8} {'rhat':>8}")
    for r in summary_rows:
        print(f"{r['tau_prior']:<24} {r['tau_post_mean']:>8.3f} {r['tau_post_sd']:>8.3f} "
              f"{r['tau_post_q05']:>8.3f} {r['tau_post_q50']:>8.3f} "
              f"{r['tau_post_q95']:>8.3f} {r['tau_rhat']:>8.3f}")

    print()
    print("Pairwise per-SKU β̂_i correlation across specs:")
    print("-" * 70)
    bcols = [f"beta_{s[0]}_mean" for s in PRIOR_SPECS]
    bdf = beta_table[bcols].dropna()
    for i, ci in enumerate(bcols):
        for cj in bcols[i+1:]:
            r = float(bdf[[ci, cj]].corr().iloc[0, 1])
            md = float((bdf[ci] - bdf[cj]).abs().median())
            print(f"  {ci} vs {cj}:  Pearson r = {r:.4f},  "
                  f"median |Δβ̂| = {md:.4f}")

    # ---- Plot 1: τ posterior densities ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (spec_id, label, _, _), col in zip(PRIOR_SPECS, colors):
        draws = np.load(OUT_DIR / f"_tau_draws_{spec_id}.npy")
        ax.hist(draws, bins=60, density=True, alpha=0.4, color=col, label=label)
    ax.set_xlabel(r"$\tau$ posterior")
    ax.set_ylabel("posterior density")
    ax.set_title("τ posterior under three priors")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_hb_prior_sensitivity_tau_density.png", dpi=150)
    plt.close()

    # ---- Plot 2: per-SKU β̂ scatter, baseline vs each robustness ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    base_col = f"beta_{PRIOR_SPECS[0][0]}_mean"
    for ax, (spec_id, label, _, _) in zip(axes, PRIOR_SPECS[1:]):
        col = f"beta_{spec_id}_mean"
        sub = beta_table[[base_col, col]].dropna()
        ax.scatter(sub[base_col], sub[col], alpha=0.4, s=15)
        lo = float(min(sub[base_col].min(), sub[col].min()))
        hi = float(max(sub[base_col].max(), sub[col].max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        r = float(sub.corr().iloc[0, 1])
        ax.set_xlabel(rf"$\hat\beta_i$ baseline ({PRIOR_SPECS[0][1]})")
        ax.set_ylabel(rf"$\hat\beta_i$ ({label})")
        ax.set_title(f"r = {r:.4f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_hb_prior_sensitivity_beta_scatter.png", dpi=150)
    plt.close()

    # ---- Cleanup tmp files ----
    for spec_id, _, _, _ in PRIOR_SPECS:
        f = OUT_DIR / f"_tau_draws_{spec_id}.npy"
        if f.exists():
            f.unlink()

    print()
    print("=" * 70)
    print("Outputs:")
    print(f"  {OUT_DIR / 'jd_hb_prior_sensitivity_summary.csv'}")
    print(f"  {OUT_DIR / 'jd_hb_prior_sensitivity_betas.csv'}")
    print(f"  {OUT_DIR / 'jd_hb_prior_sensitivity_tau_density.png'}")
    print(f"  {OUT_DIR / 'jd_hb_prior_sensitivity_beta_scatter.png'}")


if __name__ == "__main__":
    main()
