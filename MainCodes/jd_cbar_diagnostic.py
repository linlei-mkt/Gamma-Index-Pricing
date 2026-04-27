"""
Empirical $C \bar e$ diagnostic for the $\gamma$-iteration contraction
condition.

Addresses the reviewer's concern that reporting $\bar e$ alone is
insufficient: the practical contraction condition for the
$\gamma$-update map is $C\bar e < 1$ where the constant $C$ depends
on Lerner-rule denominators $D_i = 1 - \gamma^\star - (1-\gamma^\star)/|\eta_i|$,
elasticities, and markups (Appendix D of the paper). When elasticities
are close to unit elasticity, $C$ can blow up. This script computes
$C$ and $C\bar e$ empirically on the 31 JD daily markets using
hierarchical-Bayes $\hat\beta_i$, observed shares, and the $\gamma^\star=0$
setting.

Two quantities are reported per market:

1. Theoretical upper bound $C_t^{\text{theory}}$ from Appendix D:
     $C_t = \bar L \cdot |1 - \gamma^\star| / (\eta_{\min,t}^2 D_{\min,t}^2)$,
   where $\eta_{\min,t}$ and $D_{\min,t}$ are the min own-elasticity
   magnitude and min Lerner-denominator across SKUs in market $t$.

2. Empirical operator norm $\rho_t$ of the linearized $\gamma$-update
   Jacobian at observed prices, computed by finite differences.

The meaningful "safe regime" is $C\bar e < 1$ or $\rho \cdot \bar e < 1$.
We report both the median and interquartile range across the 31
markets, plus a scatter of $\rho$ vs $\bar e$ to visualize whether
the $\bar e \approx 0.3$ threshold has any reliable interpretation.

Outputs
-------
  - jd_cbar_results.csv       per-market $\bar e$, $C^{\text{theory}}$, $\rho$, $C\bar e$, $\rho\bar e$
  - jd_cbar_histogram.png     histograms
  - jd_cbar_scatter.png       $\rho$ vs $\bar e$ scatter

To replicate:
    pip install pandas numpy scipy matplotlib
    JD_DATA_DIR=/path/to/jd_csvs python3 jd_cbar_diagnostic.py
Requires jd_hb_posterior_summary.csv from jd_hierarchical_bayes.py.
"""
from __future__ import annotations

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "../JD_MSOM"
))
OUT = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 500
M_MULT = 3.0
MARGIN_RATIO = 0.70
GAMMA_STAR = 0.0       # unconstrained case (most benign for γ-iteration)
BETA_FLOOR = 1.2
EPS_FD = 1e-4          # finite-difference step (relative)


def load_data_and_hb():
    orders = pd.read_csv(
        DATA_DIR / "JD_order_data.csv",
        usecols=['sku_ID', 'order_date', 'quantity', 'final_unit_price'],
        dtype={'sku_ID': 'string'}, parse_dates=['order_date'],
    )
    orders['day'] = orders['order_date'].dt.day
    top_ids = orders.groupby('sku_ID').size().nlargest(N_TOP_SKU).index
    o = orders[orders.sku_ID.isin(top_ids)].copy()
    o['rev'] = o['final_unit_price'] * o['quantity']
    agg = (o.groupby(['day', 'sku_ID'], as_index=False)
            .agg(qty=('quantity', 'sum'), rev=('rev', 'sum')))
    agg['price'] = agg['rev'] / agg['qty']
    agg = agg[(agg['qty'] > 0) & (agg['price'] > 0)].reset_index(drop=True)
    peak = agg.groupby('day')['qty'].sum().max()
    M = M_MULT * peak
    agg['share'] = agg['qty'] / M
    daily_Q = agg.groupby('day')['qty'].sum().rename('Q_in')
    s0 = (1.0 - daily_Q / M).rename('s0')
    agg = agg.merge(s0, on='day')

    # HB posterior
    hb_path = SCRIPT_DIR / "jd_hb_posterior_summary.csv"
    if not hb_path.exists():
        raise FileNotFoundError(f"{hb_path} — run jd_hierarchical_bayes.py first")
    hb = pd.read_csv(hb_path)
    sku_to_beta = dict(zip(hb['sku_ID'], hb['beta_posterior_mean'].clip(lower=BETA_FLOOR)))
    agg = agg[agg.sku_ID.isin(sku_to_beta.keys())].copy()
    agg['beta_hat'] = agg['sku_ID'].map(sku_to_beta)
    return agg, M


def mci_shares(p, alpha, beta, M):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def calibrate_alpha(p_obs, s_obs, s0_obs, beta):
    return (s_obs / s0_obs) * np.power(p_obs, beta)


def gamma_update(p, c, alpha, beta, M, gamma_star=GAMMA_STAR):
    """One step of the γ-iteration update map."""
    s, _ = mci_shares(p, alpha, beta, M)
    eta = np.maximum(beta * (1.0 - s), 1.01)
    denom = 1.0 - gamma_star - (1.0 - gamma_star) / eta
    denom = np.maximum(denom, 0.01)
    return np.maximum(c / denom, c * 1.0001)


def operator_norm_Tgamma(p, c, alpha, beta, M, gamma_star=GAMMA_STAR, eps=EPS_FD):
    """Infinity-operator norm of the Jacobian ∂T_γ/∂p at p, by finite
    differences. Returns max over rows of row-sum absolute values."""
    n = len(p)
    scale = np.maximum(np.abs(p), 1e-6)
    Tp = gamma_update(p, c, alpha, beta, M, gamma_star)
    norm_row_sums = np.zeros(n)
    for j in range(n):
        p_plus = p.copy()
        p_plus[j] += eps * scale[j]
        Tpj = gamma_update(p_plus, c, alpha, beta, M, gamma_star)
        col_j = (Tpj - Tp) / (eps * scale[j])
        norm_row_sums += np.abs(col_j)
    # row-sum norm ≥ infinity-operator norm; an exact column sum gives
    # the 1-norm. For square matrix symmetric in scale, both are close
    # to the spectral radius bound.
    return float(np.max(norm_row_sums))


def ebar_from_shares(s):
    S = s.sum()
    return float(np.max((S - s) / (1.0 - s)))


def theoretical_C(p, c, alpha, beta, M, gamma_star=GAMMA_STAR):
    """Compute the theoretical constant C from Appendix D's bound."""
    s, _ = mci_shares(p, alpha, beta, M)
    eta = beta * (1.0 - s)
    eta_min = np.min(np.abs(eta))
    denom = 1.0 - gamma_star - (1.0 - gamma_star) / eta
    denom_min = np.min(np.abs(denom))
    L = (p - c) / p
    L_max = np.max(L)
    C = abs(1.0 - gamma_star) * L_max / (eta_min ** 2 * denom_min ** 2)
    return float(C), float(eta_min), float(denom_min)


def main():
    print("=" * 70)
    print("Empirical $C \\bar e$ diagnostic on JD daily markets")
    print("=" * 70)
    print(f"γ* = {GAMMA_STAR} (unconstrained case)")
    print()

    agg, M = load_data_and_hb()
    print(f"Loaded {len(agg)} obs, M = {M:.0f}, "
          f"{agg['sku_ID'].nunique()} SKUs with HB β̂")
    print()

    rows = []
    for day in sorted(agg['day'].unique()):
        mkt = agg[agg['day'] == day].copy().reset_index(drop=True)
        if len(mkt) < 50:
            continue
        p_obs = mkt['price'].to_numpy()
        s_obs = mkt['share'].to_numpy()
        s0_obs = float(mkt['s0'].iloc[0])
        beta_vec = mkt['beta_hat'].to_numpy()
        alpha = calibrate_alpha(p_obs, s_obs, s0_obs, beta_vec)
        c = MARGIN_RATIO * p_obs

        ebar = ebar_from_shares(s_obs)
        C_theory, eta_min, D_min = theoretical_C(p_obs, c, alpha, beta_vec, M)
        rho_emp = operator_norm_Tgamma(p_obs, c, alpha, beta_vec, M)

        rows.append({
            "day": int(day),
            "n_products": len(mkt),
            "ebar": ebar,
            "eta_min": eta_min,
            "D_min": D_min,
            "C_theory": C_theory,
            "rho_empirical": rho_emp,
            "C_theory_times_ebar": C_theory * ebar,
            "rho_times_ebar": rho_emp * ebar,
            "rho_contraction": rho_emp < 1.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "jd_cbar_results.csv", index=False)
    print("Per-day summary (showing head and tail):")
    print(df[['day', 'ebar', 'eta_min', 'D_min', 'C_theory',
              'rho_empirical', 'rho_times_ebar', 'rho_contraction']]
          .head().round(3).to_string(index=False))
    print("...")
    print(df[['day', 'ebar', 'eta_min', 'D_min', 'C_theory',
              'rho_empirical', 'rho_times_ebar', 'rho_contraction']]
          .tail().round(3).to_string(index=False))

    print()
    print("Cross-market statistics:")
    print(df[['ebar', 'eta_min', 'D_min', 'C_theory', 'rho_empirical',
              'rho_times_ebar']].describe().round(3).to_string())
    print()
    print(f"Fraction of markets with ρ < 1 (empirical contraction): "
          f"{df['rho_contraction'].mean()*100:.1f}%")
    print(f"Median ρ·ebar across markets: {df['rho_times_ebar'].median():.3f}")
    print(f"Max ρ·ebar across markets: {df['rho_times_ebar'].max():.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['ebar'], bins=15, color='tab:blue', edgecolor='black',
                 alpha=0.75, label=f"$\\bar e$ (median {df['ebar'].median():.3f})")
    axes[0].hist(df['rho_empirical'], bins=15, color='tab:red', edgecolor='black',
                 alpha=0.5, label=f"$\\rho$ empirical (median {df['rho_empirical'].median():.3f})")
    axes[0].axvline(1.0, color='black', linestyle='--', label='contraction threshold')
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("number of daily markets")
    axes[0].set_title(r"Distribution of $\bar e$ vs empirical $\rho$")
    axes[0].legend()

    axes[1].scatter(df['ebar'], df['rho_empirical'], s=40, alpha=0.7,
                    color='tab:blue', edgecolor='black')
    axes[1].plot([0, max(df['ebar'].max(), df['rho_empirical'].max())],
                 [0, max(df['ebar'].max(), df['rho_empirical'].max())],
                 'k--', alpha=0.3, label=r"$\rho = \bar e$")
    axes[1].axhline(1.0, color='red', linestyle=':', alpha=0.5,
                    label='contraction threshold')
    axes[1].set_xlabel(r"$\bar e$ (share-only diagnostic)")
    axes[1].set_ylabel(r"$\rho$ (empirical Jacobian operator norm)")
    axes[1].set_title(r"Empirical contraction rate vs $\bar e$")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "jd_cbar_histogram.png", dpi=150)
    plt.close()

    # Second scatter: C_theory*ebar vs rho*ebar
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df['ebar'], df['rho_times_ebar'], s=40, alpha=0.8,
               color='tab:blue', label=r"$\rho \cdot \bar e$ (empirical)")
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5,
               label=r"contraction threshold $\rho\bar e = 1$")
    ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5,
               label=r"$\bar e = 0.3$ old threshold")
    ax.set_xlabel(r"$\bar e$ at observed prices")
    ax.set_ylabel(r"$\rho \cdot \bar e$ (effective contraction rate)")
    ax.set_title(r"Effective contraction rate $\rho \cdot \bar e$ across 31 daily markets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "jd_cbar_scatter.png", dpi=150)
    plt.close()

    print()
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
