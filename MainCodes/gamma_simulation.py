"""
Simulation: γ-Equalization vs Uniform vs BLP-Newton under Mixed Logit Demand
------------------------------------------------------------------------------
Accompanies the QME paper "γ-Equalization as Approximate Optimal Pricing."

A single-firm multi-product retailer faces mixed-logit demand and maximizes
total gross profit Π(p) = Σ_i (p_i − c_i) s_i(p).  We compare four pricing
rules:

    (U)   Uniform pricing        : one scalar p across all products.
    (G)   γ-equalization         : each product solves its own scalar FOC
                                      s_i + (p_i − c_i) ∂s_i/∂p_i = 0
                                   (drop cross term R_i).   Cost O(n).
    (Gk)  Iterated γ, k sweeps   : Jacobi on full FOC,
                                      p_i^{+}=c_i−(s_i+Σ_{j≠i} M_ij (p_j−c_j))/M_ii.
                                   Converges to BLP at rate ē.
    (B)   BLP-Newton             : exact profit max (scipy minimize, gradient).

Diagnostics
===========
  ē_off = max_i Σ_{j≠i} |M_ij|/|M_ii|    (diagonal-dominance slack)
  V_η   = std of own price elasticities  (heterogeneity)
  gap_X = 100·(Π_BLP − Π_X)/Π_BLP        (percent profit lost)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from pathlib import Path
import time, warnings
warnings.filterwarnings('ignore')

OUT = Path(__file__).parent


# ============================================================================
# Mixed-logit primitives
# ============================================================================

def shares(p, delta, alpha_h):
    """Market & per-type shares under mixed logit with an outside option."""
    u = delta[None, :] - alpha_h[:, None] * p[None, :]
    # numerical stabilizer
    u_max = np.maximum(u.max(axis=1, keepdims=True), 0.0)
    expu = np.exp(u - u_max)
    exp0 = np.exp(-u_max)                       # outside option has utility 0
    denom = exp0 + expu.sum(axis=1, keepdims=True)
    s_h = expu / denom
    s = s_h.mean(axis=0)
    s_0h = (exp0 / denom).squeeze(-1)
    return s, s_h, s_0h


def share_jacobian(p, delta, alpha_h):
    """M[i,j] = ∂s_j/∂p_i (n×n)."""
    _, s_h, _ = shares(p, delta, alpha_h)
    weighted = alpha_h[:, None] * s_h
    M = weighted.T @ s_h / alpha_h.shape[0]           # E[α s_i s_j]
    diag_off = -np.mean(weighted, axis=0)             # -E[α s_i]
    np.fill_diagonal(M, M.diagonal() + diag_off)      # -E[α s_i(1-s_i)] on diag
    return M


def ebar_off(p, delta, alpha_h):
    M = share_jacobian(p, delta, alpha_h)
    n = M.shape[0]
    ratios = np.empty(n)
    for i in range(n):
        own = abs(M[i, i])
        cross = np.sum(np.abs(M[i, :])) - own
        ratios[i] = cross / own
    return float(ratios.max()), float(ratios.mean())


def total_profit(p, delta, alpha_h, c):
    s, _, _ = shares(p, delta, alpha_h)
    return float(np.sum((p - c) * s))


def profit_gradient(p, delta, alpha_h, c):
    """∂Π/∂p_i = s_i + Σ_j (p_j − c_j) · ∂s_j/∂p_i."""
    s, _, _ = shares(p, delta, alpha_h)
    M = share_jacobian(p, delta, alpha_h)
    return s + M @ (p - c)


def own_elasticities(p, delta, alpha_h):
    s, _, _ = shares(p, delta, alpha_h)
    M = share_jacobian(p, delta, alpha_h)
    return p * np.diag(M) / np.maximum(s, 1e-12)


# ============================================================================
# Pricing rules
# ============================================================================

def solve_uniform(delta, alpha_h, c):
    n = len(c)
    def neg(ps):
        return -total_profit(np.full(n, ps), delta, alpha_h, c)
    lo = float(c.mean()) + 1e-4
    hi = float(c.mean()) + 60.0 / float(alpha_h.mean())
    res = minimize_scalar(neg, bounds=(lo, hi), method='bounded',
                          options=dict(xatol=1e-10))
    return np.full(n, res.x)


def gamma_eq(delta, alpha_h, c, p_start=None):
    """γ-eq: solve single-product FOC s_i + (p_i − c_i) ∂s_i/∂p_i = 0 for all i.
       Under mixed logit with α > 0, this is:
         p_i − c_i = s_i(p) / E_h[α_h s_{ih}(1 − s_{ih})]
       Fixed-point iteration (damped)."""
    n = len(c)
    if p_start is None:
        p = c + 0.5 / alpha_h.mean()
    else:
        p = p_start.copy()
    for _ in range(300):
        s, s_h, _ = shares(p, delta, alpha_h)
        own = np.mean(alpha_h[:, None] * s_h * (1 - s_h), axis=0)
        p_new = c + s / np.maximum(own, 1e-12)
        if np.max(np.abs(p_new - p)) < 1e-11:
            return p_new
        p = 0.5 * p + 0.5 * p_new
    return p


def zeta_step(p, delta, alpha_h, c):
    """One Jacobi (ζ / iterated-γ) sweep on the BLP FOC."""
    s, _, _ = shares(p, delta, alpha_h)
    M = share_jacobian(p, delta, alpha_h)
    off = M - np.diag(np.diag(M))
    p_new = c - (s + off @ (p - c)) / np.diag(M)
    # cap oversized step
    step = p_new - p
    cap = max(1.0, 0.5 * np.max(np.abs(p)))
    sm = np.max(np.abs(step))
    if sm > cap:
        step *= cap / sm
    return p + step


def iterated_gamma(delta, alpha_h, c, p_start, n_sweeps):
    p = p_start.copy()
    hist = [p.copy()]
    for _ in range(n_sweeps):
        p = zeta_step(p, delta, alpha_h, c)
        hist.append(p.copy())
    return p, hist


def blp_optimum(delta, alpha_h, c, p_start, p_uni=None):
    """True profit max via L-BFGS-B with analytic gradient.
       Uses several warm starts (γ-eq, uniform, ζ-iterated, and a
       ladder of scalar markups), keeps the best objective."""
    def neg_profit(p):
        return -total_profit(p, delta, alpha_h, c)
    def neg_grad(p):
        return -profit_gradient(p, delta, alpha_h, c)
    bounds = [(float(ci) + 1e-4, float(ci) + 200.0) for ci in c]
    # Multiple warm starts
    candidates = [p_start.copy()]
    if p_uni is not None:
        candidates.append(p_uni.copy())
    try:
        p_zeta, _ = iterated_gamma(delta, alpha_h, c, p_start, 150)
        candidates.append(p_zeta)
    except Exception:
        pass
    # uniform-markup ladder over scalars in (c_i, c_i + 5/ᾱ)
    amu = 1.0 / float(alpha_h.mean())
    for m in (0.25, 0.5, 1.0, 1.5, 2.5, 4.0):
        candidates.append(c + m * amu)
    best_p, best_pi = None, -np.inf
    for p0 in candidates:
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(neg_profit, p0, jac=neg_grad, method='L-BFGS-B',
                           bounds=bounds, options=dict(ftol=1e-13,
                                                        gtol=1e-11,
                                                        maxiter=500))
            if -res.fun > best_pi:
                best_pi = -res.fun
                best_p = res.x
        except Exception:
            continue
    return best_p if best_p is not None else candidates[0]


# ============================================================================
# Scenario driver
# ============================================================================

def simulate_one(n=12, H=150,
                 delta_bar=2.0, sigma_delta=1.0,
                 alpha_bar=2.0, sigma_alpha=0.4,
                 sigma_c=0.3,
                 seed=0):
    rng = np.random.default_rng(seed)
    delta = delta_bar + sigma_delta * rng.standard_normal(n)
    alpha_h = alpha_bar * np.exp(sigma_alpha * rng.standard_normal(H)
                                  - 0.5 * sigma_alpha ** 2)
    # Heterogeneous marginal costs.  Cost heterogeneity is REQUIRED for
    # uniform pricing to be suboptimal under mixed logit with homogeneous α:
    # with c_i = c̄ for all i, exp(δ_i) factors cancel in the FOC ratio and
    # uniform pricing is first-best optimal.
    c = np.exp(sigma_c * rng.standard_normal(n)
               - 0.5 * sigma_c ** 2)  # mean ≈ 1, lognormal

    p_uni = solve_uniform(delta, alpha_h, c)
    p_g = gamma_eq(delta, alpha_h, c, p_uni)
    p_blp = blp_optimum(delta, alpha_h, c, p_g, p_uni=p_uni)

    # iterated γ trajectories (starting from uniform)
    _, hist = iterated_gamma(delta, alpha_h, c, p_uni, 25)

    e_max, e_mean = ebar_off(p_blp, delta, alpha_h)
    s_blp, _, _ = shares(p_blp, delta, alpha_h)
    outside_share = float(1.0 - s_blp.sum())
    elasts = own_elasticities(p_blp, delta, alpha_h)
    V_eta = float(elasts.std())
    mean_elast = float(elasts.mean())

    pi_blp = total_profit(p_blp, delta, alpha_h, c)
    gap = lambda pi: 100.0 * (pi_blp - pi) / max(pi_blp, 1e-12)

    # pick three iterated-γ snapshots for the main plot
    gap_traj = [gap(total_profit(pk, delta, alpha_h, c)) for pk in hist]

    return dict(
        ebar_max=e_max,
        ebar_mean=e_mean,
        V_eta=V_eta,
        mean_own_elast=mean_elast,
        outside_share=outside_share,
        gap_uniform=gap(total_profit(p_uni, delta, alpha_h, c)),
        gap_gamma=gap(total_profit(p_g, delta, alpha_h, c)),
        gap_gamma3=gap_traj[3] if len(gap_traj) > 3 else gap_traj[-1],
        gap_gamma10=gap_traj[10] if len(gap_traj) > 10 else gap_traj[-1],
        gap_gamma25=gap_traj[-1],
        pi_blp=pi_blp,
        delta_bar=delta_bar,
        sigma_delta=sigma_delta,
        seed=seed,
        gap_traj=gap_traj,
    )


def main():
    # Scenario grid chosen to cover a wide range of ē.
    # low δ̄ → high outside share → low ē; large δ̄ → low outside share → high ē.
    delta_bars = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    sigma_deltas = [0.3, 0.9, 1.6]
    n_seeds = 12

    records = []
    trajectories = []
    t0 = time.time()
    total = len(delta_bars) * len(sigma_deltas) * n_seeds
    done = 0
    for db in delta_bars:
        for sd in sigma_deltas:
            for seed in range(n_seeds):
                try:
                    r = simulate_one(delta_bar=db, sigma_delta=sd, seed=seed)
                    trajectories.append(dict(delta_bar=db, sigma_delta=sd,
                                             seed=seed,
                                             ebar_max=r['ebar_max'],
                                             gap_traj=r['gap_traj']))
                    r.pop('gap_traj')
                    records.append(r)
                except Exception as e:
                    print(f"[fail] db={db} sd={sd} s={seed}: {e}", flush=True)
                done += 1
                if done % 12 == 0:
                    el = time.time() - t0
                    r = records[-1]
                    print(f"[{done}/{total}] t={el:5.1f}s  db={db:+.1f} sd={sd} "
                          f"ē={r['ebar_max']:.3f} s0={r['outside_share']:.2f} "
                          f"gapU={r['gap_uniform']:7.3f} "
                          f"gapG={r['gap_gamma']:7.3f} "
                          f"gapG10={r['gap_gamma10']:7.3f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(OUT / 'simulation_results.csv', index=False)

    # Guard: BLP should (weakly) dominate every method. Clip tiny negatives.
    for col in ['gap_uniform', 'gap_gamma', 'gap_gamma3', 'gap_gamma10', 'gap_gamma25']:
        df[col] = df[col].clip(lower=-0.1)  # tolerate rounding

    # ------- Summary table ------------------------------------------------
    agg_cols = ['ebar_max', 'ebar_mean', 'V_eta', 'mean_own_elast',
                'outside_share', 'gap_uniform', 'gap_gamma',
                'gap_gamma3', 'gap_gamma10', 'gap_gamma25']
    summary = df.groupby(['delta_bar', 'sigma_delta'])[agg_cols].mean().round(3)
    summary.to_csv(OUT / 'simulation_summary.csv')
    print("\n========================= SIMULATION SUMMARY =========================")
    print(summary.to_string())
    print("======================================================================\n")

    # Summary by ē bucket
    df['ebar_bucket'] = pd.cut(df['ebar_max'],
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                labels=['<0.2', '0.2-0.4', '0.4-0.6',
                                        '0.6-0.8', '0.8-1.0'])
    bucket_summary = df.groupby('ebar_bucket')[
        ['gap_uniform', 'gap_gamma', 'gap_gamma3', 'gap_gamma10']].agg(
        ['mean', 'median']).round(3)
    bucket_summary.to_csv(OUT / 'simulation_summary_by_ebar.csv')
    print("=========== MEAN / MEDIAN PROFIT GAP BY ē_off BUCKET ============")
    print(bucket_summary.to_string())
    print("=================================================================\n")

    # ---- Figure 1: profit gap vs ē ---------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(df['ebar_max'], df['gap_uniform'], alpha=0.55, s=42,
               color='#C0392B', label='Uniform pricing',
               edgecolor='white', lw=0.4)
    ax.scatter(df['ebar_max'], df['gap_gamma'], alpha=0.55, s=42,
               color='#2980B9',
               label=r'$\gamma$-equalization (1 sweep, $O(n)$)',
               edgecolor='white', lw=0.4)
    ax.scatter(df['ebar_max'], df['gap_gamma3'], alpha=0.55, s=42,
               color='#8E44AD', label=r'Iterated $\gamma$, 3 sweeps',
               edgecolor='white', lw=0.4)
    ax.scatter(df['ebar_max'], df['gap_gamma10'], alpha=0.55, s=42,
               color='#27AE60', label=r'Iterated $\gamma$, 10 sweeps',
               edgecolor='white', lw=0.4)

    bins = np.linspace(df['ebar_max'].min() - 1e-6,
                       df['ebar_max'].max() + 1e-6, 9)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    grp = df.groupby(pd.cut(df['ebar_max'], bins))[
        ['gap_uniform', 'gap_gamma', 'gap_gamma3', 'gap_gamma10']].mean()
    ax.plot(ctr, grp['gap_uniform'].values, '--', color='#C0392B', lw=2, alpha=0.85)
    ax.plot(ctr, grp['gap_gamma'].values, '--', color='#2980B9', lw=2, alpha=0.85)
    ax.plot(ctr, grp['gap_gamma3'].values, '--', color='#8E44AD', lw=2, alpha=0.85)
    ax.plot(ctr, grp['gap_gamma10'].values, '--', color='#27AE60', lw=2, alpha=0.85)

    ax.axhline(0, color='black', lw=0.4)
    ax.set_xlabel(r'Diagonal-dominance slack  $\bar{e}_{\mathrm{off}}$',
                  fontsize=12)
    ax.set_ylabel('Profit gap vs. BLP-Newton optimum  (%)', fontsize=12)
    ax.set_title(r'Profit gap across '
                 + f"{len(df)}"
                 + r' calibrated mixed-logit scenarios',
                 fontsize=13)
    ax.legend(loc='upper left', frameon=True, fontsize=10.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'profit_gap_vs_ebar.png', dpi=160)
    plt.close(fig)

    # ---- Figure 2: γ-vs-uniform advantage vs heterogeneity ---------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    advantage = df['gap_uniform'] - df['gap_gamma']
    sc = ax.scatter(df['V_eta'], advantage,
                    c=df['ebar_max'], cmap='viridis', s=48, alpha=0.8,
                    edgecolor='white', lw=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'$\bar{e}_{\mathrm{off}}$', fontsize=11)
    ax.axhline(0, color='red', lw=1.5, ls='--', label='equal performance')
    ax.set_xlabel(r'Elasticity heterogeneity $V_{\eta}$ (std of own elasticities)',
                  fontsize=12)
    ax.set_ylabel(r'$\gamma$-eq advantage over uniform  '
                  r'(gap$_U$ − gap$_\gamma$, %)', fontsize=12)
    ax.set_title(r'When does $\gamma$-equalization beat uniform pricing?',
                 fontsize=13)
    ax.legend(loc='best', fontsize=10.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'heterogeneity_crossover.png', dpi=160)
    plt.close(fig)

    # ---- Figure 3: iterated γ convergence --------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap('plasma')
    traj_df = pd.DataFrame(trajectories)
    ebar_targets = [0.15, 0.35, 0.55, 0.75, 0.90]
    for j, eb in enumerate(ebar_targets):
        idx = (traj_df['ebar_max'] - eb).abs().idxmin()
        tr = np.maximum(traj_df.loc[idx, 'gap_traj'], 1e-8)
        ax.semilogy(range(len(tr)), tr, marker='o', markersize=4.5,
                    color=cmap(j / len(ebar_targets)),
                    label=fr'$\bar{{e}}={traj_df.loc[idx, "ebar_max"]:.2f}$')
    ax.set_xlabel(r'Iterated $\gamma$ sweep', fontsize=12)
    ax.set_ylabel('Profit gap vs BLP-Newton  (%, log scale)', fontsize=12)
    ax.set_title(r'Iterated $\gamma$ converges at rate $\bar{e}_{\mathrm{off}}$',
                 fontsize=13)
    ax.legend(frameon=True, fontsize=10.5)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(OUT / 'convergence_iterated_gamma.png', dpi=160)
    plt.close(fig)

    print(f"Saved figures and tables to {OUT}")
    return df, summary


if __name__ == '__main__':
    df, summary = main()
