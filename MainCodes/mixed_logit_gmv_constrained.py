"""
Mixed-logit GMV-constrained pricing test.

Purpose
-------
Tests whether γ-equalization's distinctive γ* ≠ 0 contribution
(shadow-price interpretation under binding GMV constraints) survives
under mixed-logit demand — i.e., beyond the MCI closed-form setting.

For each synthetic market drawn from a mixed-logit DGP, we:
  1. Solve unconstrained BN (Newton on the full mixed-logit Jacobian).
  2. Impose a GMV floor R ≥ R_target, with R_target a multiple of the
     unconstrained-BN revenue (default 1.25× — i.e., the constraint is
     always binding because BN-unconstrained revenue is already at a
     local max for the profit objective).
  3. Solve exact constrained BN via Lagrangian transform:
         max π  s.t.  R ≥ R_target  ≡  max at c_eff = c/(1+μ)
     binary-search μ ≥ 0 to hit the constraint with equality.
  4. Solve γ-iteration at γ* = 0 (classical Lerner - ignores floor).
  5. Solve γ-iteration with γ* tuned by binary search to match floor.
  6. Compare profit and whether the constraint is satisfied.

Outputs (in /Users/linlei/Downloads/Gamma/):
  - mixed_logit_gmv_results.csv        per-market rows
  - mixed_logit_gmv_profit_gap.png     γ-tuned gap vs constrained BN
  - mixed_logit_gmv_gamma_star.png     tuned γ* distribution

Runtime: ~2-5 minutes for 80 Monte Carlo markets.

Usage:
    python3 mixed_logit_gmv_constrained.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

# ==================  Config  ==================
N_MARKETS = 80
R_CONSUMERS = 100
GMV_TARGET_MULT = 1.25        # GMV floor = 1.25× unconstrained-BN revenue
TOL = 1e-8
MAX_ITER = 2000
SEED = 2026
rng = np.random.default_rng(SEED)


# ==================  Mixed-logit primitives (stable form)  ==================
def draw_alpha(mu_log_alpha, sigma_log_alpha, R, rng_local):
    z = rng_local.standard_normal(R)
    return np.exp(mu_log_alpha + sigma_log_alpha * z)


def ml_shares(p, delta, alpha_r):
    u = delta[None, :] - np.outer(alpha_r, p)
    shift = np.maximum(u.max(axis=1, keepdims=True), 0.0)
    e = np.exp(u - shift)
    e0 = np.exp(-shift[:, 0])
    D = e0 + e.sum(axis=1)
    s_rj = e / D[:, None]
    return s_rj.mean(axis=0), (e0 / D).mean(), s_rj


def ml_jacobian(p, delta, alpha_r):
    _, _, s_rj = ml_shares(p, delta, alpha_r)
    R = len(alpha_r)
    cross = (alpha_r[:, None] * s_rj).T @ s_rj / R    # (n, n), Σ α_r s_ri s_rj
    # Sign convention: with u_{rj} = δ - α_r p_j (i.e., positive α enters
    # negatively), ∂s_{rj}/∂p_i = -α_r s_{rj}(δ_{ij} - s_{ri}). So:
    #   Ω_{ii} = -(1/R) Σ α_r s_{ri}(1-s_{ri})     (negative)
    #   Ω_{ij} = +(1/R) Σ α_r s_{rj} s_{ri}  i≠j   (positive)
    diag_own = -(alpha_r[:, None] * s_rj * (1.0 - s_rj)).mean(axis=0)
    Om = cross.copy()
    np.fill_diagonal(Om, diag_own)
    return Om


def total_profit(p, c, delta, alpha_r):
    s, _, _ = ml_shares(p, delta, alpha_r)
    return float(np.sum((p - c) * s))


def total_revenue(p, delta, alpha_r):
    s, _, _ = ml_shares(p, delta, alpha_r)
    return float(np.sum(p * s))


# ==================  Solvers  ==================
def newton_bn(p0, c, delta, alpha_r):
    """Solve BN FOC via MS-style Jacobi splitting (robust) under mixed logit."""
    p = np.maximum(p0.copy(), c * 1.0001)
    for _ in range(500):
        s, _, _ = ml_shares(p, delta, alpha_r)
        Om = ml_jacobian(p, delta, alpha_r)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        # Guard against tiny diagonal (which would blow up p)
        diag_safe = np.where(np.abs(diag) > 1e-12, diag, -1e-12)
        p_new = c - (s + Gamma @ (p - c)) / diag_safe
        p_new = np.maximum(p_new, c * 1.0001)
        p_new = np.minimum(p_new, c * 100)  # cap runaway
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new; break
        p = p_new
    return p


def constrained_bn_floor(p0, c, delta, alpha_r, R_target):
    """max profit s.t. R ≥ R_target, via Lagrangian c -> c/(1+μ)."""
    p_unc = newton_bn(p0, c, delta, alpha_r)
    R_unc = total_revenue(p_unc, delta, alpha_r)
    if R_unc >= R_target:
        return p_unc, 0.0
    def residual(mu):
        c_eff = c / (1.0 + mu)
        p_mu = newton_bn(p0, c_eff, delta, alpha_r)
        return total_revenue(p_mu, delta, alpha_r) - R_target
    try:
        mu_star = brentq(residual, 1e-6, 100.0, xtol=1e-6, maxiter=60)
    except ValueError:
        mu_star = 100.0
    p_bn = newton_bn(p0, c / (1.0 + mu_star), delta, alpha_r)
    return p_bn, mu_star


def gamma_iteration_ml(p0, c, delta, alpha_r, gamma_star):
    """γ-iteration under mixed logit at any γ*."""
    p = p0.copy()
    for k in range(MAX_ITER):
        s, _, s_rj = ml_shares(p, delta, alpha_r)
        # Aggregate own-elasticity magnitude
        diag_own = (alpha_r[:, None] * s_rj * (1.0 - s_rj)).mean(axis=0)
        eta_abs = np.abs(p * diag_own / np.maximum(s, 1e-12))
        eta_safe = np.maximum(eta_abs, 1.01)
        denom = 1.0 - gamma_star - (1.0 - gamma_star) / eta_safe
        denom = np.maximum(denom, 0.01)
        p_new = np.maximum(c / denom, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new; break
        p = p_new
    return p, k + 1


def uniform_tuned_ml(c, delta, alpha_r, R_target):
    """Scalar markup m binary-searched to hit R_target under mixed logit."""
    def residual(m):
        p = c / (1.0 - m)
        return total_revenue(p, delta, alpha_r) - R_target
    try:
        r_lo, r_hi = residual(0.01), residual(0.99)
        if r_lo * r_hi > 0:
            return c / (1.0 - 0.5), 0.5
        m_star = brentq(residual, 0.01, 0.99, xtol=1e-6, maxiter=50)
    except Exception:
        return c / (1.0 - 0.5), 0.5
    return c / (1.0 - m_star), m_star


def tune_gamma_star_ml(p0, c, delta, alpha_r, R_target,
                       bounds=(-5.0, 0.95)):
    def residual(gs):
        p_gs, _ = gamma_iteration_ml(p0, c, delta, alpha_r, gs)
        return total_revenue(p_gs, delta, alpha_r) - R_target
    lo, hi = bounds
    try:
        r_lo = residual(lo)
        r_hi = residual(hi)
        if r_lo * r_hi > 0:
            return lo if abs(r_lo) < abs(r_hi) else hi, float("nan")
        gamma_star = brentq(residual, lo, hi, xtol=1e-4, maxiter=50)
        return gamma_star, None
    except Exception:
        return 0.0, float("nan")


# ==================  Monte Carlo market  ==================
def one_market(m_id, rng_local):
    # Same DGP regime as mixed_logit_robustness.py — yields ebar in JD range
    N = 60
    mu_log_alpha = np.log(rng_local.uniform(2.5, 4.0))
    sigma_log_alpha = rng_local.uniform(0.05, 0.3)
    outside_shift = rng_local.uniform(-4.0, -2.0)
    delta = rng_local.standard_normal(N) * 0.6 + outside_shift
    c = rng_local.uniform(0.4, 0.8, size=N)
    alpha_r = draw_alpha(mu_log_alpha, sigma_log_alpha, R_CONSUMERS, rng_local)

    p_start = c * 1.5
    try:
        p_unc = newton_bn(p_start, c, delta, alpha_r)
    except Exception:
        return None
    s_unc, s0_unc, _ = ml_shares(p_unc, delta, alpha_r)
    if np.any(s_unc < 0) or s0_unc < 0:
        return None
    R_unc = total_revenue(p_unc, delta, alpha_r)
    if R_unc <= 0:
        return None
    R_target = GMV_TARGET_MULT * R_unc

    # Constrained BN (ground truth under floor)
    t0 = time.perf_counter()
    p_cbn, mu_star = constrained_bn_floor(p_unc.copy(), c, delta, alpha_r,
                                          R_target)
    t_cbn = time.perf_counter() - t0

    # γ at γ*=0 (classical Lerner, no constraint awareness)
    t0 = time.perf_counter()
    p_g0, _ = gamma_iteration_ml(p_unc.copy(), c, delta, alpha_r, 0.0)
    t_g0 = time.perf_counter() - t0

    # γ with γ* tuned to floor
    t0 = time.perf_counter()
    gamma_star, _ = tune_gamma_star_ml(p_unc.copy(), c, delta, alpha_r,
                                       R_target)
    p_gtuned, _ = gamma_iteration_ml(p_unc.copy(), c, delta, alpha_r, gamma_star)
    t_gtuned = time.perf_counter() - t0

    # Uniform markup tuned to floor
    t0 = time.perf_counter()
    p_utuned, m_star = uniform_tuned_ml(c, delta, alpha_r, R_target)
    t_utuned = time.perf_counter() - t0

    pi_unc  = total_profit(p_unc,  c, delta, alpha_r)
    pi_cbn  = total_profit(p_cbn,  c, delta, alpha_r)
    pi_g0   = total_profit(p_g0,   c, delta, alpha_r)
    pi_gtuned = total_profit(p_gtuned, c, delta, alpha_r)
    pi_utuned = total_profit(p_utuned, c, delta, alpha_r)
    R_cbn   = total_revenue(p_cbn,  delta, alpha_r)
    R_g0    = total_revenue(p_g0,   delta, alpha_r)
    R_gtuned = total_revenue(p_gtuned, delta, alpha_r)
    R_utuned = total_revenue(p_utuned, delta, alpha_r)

    gap = lambda pi: max(0.0, (pi_cbn - pi) / pi_cbn) if pi_cbn > 0 else np.nan

    return {
        "market": m_id,
        "mu_log_alpha": mu_log_alpha,
        "sigma_log_alpha": sigma_log_alpha,
        "R_target": R_target,
        "R_unc": R_unc,
        "mu_lagrangian": mu_star,
        "gamma_star_tuned": gamma_star,
        "uniform_m_tuned": m_star,
        "pi_unc": pi_unc, "pi_cbn": pi_cbn,
        "pi_gamma0": pi_g0, "pi_gamma_tuned": pi_gtuned,
        "pi_uniform_tuned": pi_utuned,
        "R_cbn": R_cbn, "R_gamma0": R_g0, "R_gamma_tuned": R_gtuned,
        "R_uniform_tuned": R_utuned,
        "gap_gamma0_vs_cbn": gap(pi_g0),
        "gap_gamma_tuned_vs_cbn": gap(pi_gtuned),
        "gap_uniform_tuned_vs_cbn": gap(pi_utuned),
        "gamma_beats_uniform_constrained": pi_gtuned > pi_utuned,
        "gamma0_violates_floor": R_g0 < R_target,
        "gamma_tuned_meets_floor": R_gtuned >= R_target * 0.99,
        "uniform_tuned_meets_floor": R_utuned >= R_target * 0.99,
        "time_cbn": t_cbn,
        "time_gamma_tuned": t_gtuned,
        "time_uniform_tuned": t_utuned,
    }


# ==================  Main  ==================
def main():
    print("=" * 70)
    print("MIXED-LOGIT GMV-constrained pricing test")
    print("=" * 70)
    print(f"GMV floor = {GMV_TARGET_MULT}× unconstrained-BN revenue")
    print(f"Monte Carlo markets: {N_MARKETS}")
    print()

    t0 = time.time()
    rows = []
    for m in range(N_MARKETS):
        r = one_market(m, rng)
        if r is not None:
            rows.append(r)
        if (m + 1) % 20 == 0:
            print(f"  {m + 1}/{N_MARKETS}  elapsed {time.time()-t0:.1f}s  "
                  f"({len(rows)} succeeded)")
    print(f"  {len(rows)}/{N_MARKETS} markets succeeded in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mixed_logit_gmv_results.csv", index=False)

    # Plot 1: profit gap vs tuned γ*
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df["gamma_star_tuned"], df["gap_gamma_tuned_vs_cbn"] * 100,
               color="tab:blue", s=25, label="γ with γ* tuned")
    ax.scatter(df["gamma_star_tuned"], df["gap_gamma0_vs_cbn"] * 100,
               color="tab:red", s=25, marker="x", alpha=0.6,
               label="γ at γ*=0 (violates floor)")
    ax.set_xlabel("tuned γ* (shadow price of GMV floor)")
    ax.set_ylabel("profit gap to constrained BN (%)")
    ax.set_title(f"Mixed-logit GMV-constrained results (floor = "
                 f"{GMV_TARGET_MULT}× BN revenue)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mixed_logit_gmv_profit_gap.png", dpi=150)
    plt.close()

    # Plot 2: γ* distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["gamma_star_tuned"].dropna(), bins=20, color="tab:blue",
            edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="red", linestyle="--", label="γ* = 0 (Lerner)")
    ax.axvline(df["gamma_star_tuned"].median(), color="black", linestyle=":",
               label=f"median = {df['gamma_star_tuned'].median():.3f}")
    ax.set_xlabel("tuned γ*")
    ax.set_ylabel("number of markets")
    ax.set_title(f"Distribution of γ* under mixed-logit GMV floor "
                 f"({GMV_TARGET_MULT}× BN revenue)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mixed_logit_gmv_gamma_star.png", dpi=150)
    plt.close()

    print()
    print("=" * 70)
    print(f"Markets: {len(df)}")
    print()
    print("Tuned γ* (shadow price of GMV floor):")
    print(df["gamma_star_tuned"].describe().round(4).to_string())
    print()
    print(f"Days where γ*=0 violates floor:         "
          f"{df['gamma0_violates_floor'].sum()}/{len(df)}")
    print(f"Days where γ* tuned correctly meets it: "
          f"{df['gamma_tuned_meets_floor'].sum()}/{len(df)}")
    print()
    print("Profit gap to constrained BN (ground truth under floor):")
    print(f"  γ at γ*=0 (classical Lerner):   "
          f"mean {df['gap_gamma0_vs_cbn'].mean()*100:6.2f}%  "
          f"median {df['gap_gamma0_vs_cbn'].median()*100:6.2f}%  "
          f"(VIOLATES floor)")
    print(f"  uniform markup (tuned):         "
          f"mean {df['gap_uniform_tuned_vs_cbn'].mean()*100:6.2f}%  "
          f"median {df['gap_uniform_tuned_vs_cbn'].median()*100:6.2f}%  "
          f"(meets floor, no elasticity info)")
    print(f"  γ with γ* tuned:                "
          f"mean {df['gap_gamma_tuned_vs_cbn'].mean()*100:6.2f}%  "
          f"median {df['gap_gamma_tuned_vs_cbn'].median()*100:6.2f}%  "
          f"(meets floor, own-elasticity only)")
    print()
    print(f"γ-tuned beats uniform-tuned: "
          f"{df['gamma_beats_uniform_constrained'].sum()}/{len(df)} markets")
    print()
    print("Wall-clock per solve (ms):")
    print(f"  constrained BN:     mean {df['time_cbn'].mean()*1000:6.1f} ms")
    print(f"  γ with γ* tuning:   mean {df['time_gamma_tuned'].mean()*1000:6.1f} ms")
    speedup = df["time_cbn"].mean() / df["time_gamma_tuned"].mean()
    print(f"  Speedup γ-tuned vs constrained BN: {speedup:.2f}×")
    print()
    print("NOTE on speed: γ-tuning calls brentq binary search over γ*,")
    print("  which runs γ-iteration many times.  At n=60 (this simulation's")
    print("  market size) the per-call cost of MS-based constrained BN is")
    print("  small, and γ-tuning's repeated calls can be slower overall.")
    print("  γ-tuning's O(n) per-iteration advantage over constrained BN's")
    print("  O(n^2) becomes dominant only at n >= 200, as on the JD real")
    print("  data at n=500 where γ was ~180x faster than MS2011.")
    print()
    print("The purpose of this simulation is NOT speed — it is to confirm")
    print("that γ-equalization with tuned γ* reproduces constrained BN")
    print("pricing under a NON-MCI demand system (mixed logit).")
    print()
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
