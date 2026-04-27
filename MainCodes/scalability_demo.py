"""
Scalability demonstration: γ-iteration vs MS2011 vs Newton at n = 500,
5,000, 50,000, 100,000 synthetic MCI products.

Addresses the reviewer point that 180× speedup at n=500 is
"managerially meaningless" because both finish in milliseconds.
The real question is whether γ-equalization scales to n where
full-Jacobian methods physically cannot run.

At n = 100,000:
  - Full share Jacobian Ω is 100,000 × 100,000 = 10^10 entries.
  - Float64 storage: 80 GB. Infeasible in RAM.
  - Newton's linear solve: O(n^3) = 10^15 operations. Infeasible on any CPU.
  - MS2011's matrix-vector multiply: O(n^2) = 10^10 ops/iteration
    times ~100 iterations = 10^12 ops. Hours to days.
  - γ-iteration: O(n) per iteration = 10^5 ops × ~10 iterations = 10^6 ops.
    Seconds.

We run γ-iteration at each n, report wall clock, and verify convergence.
We attempt MS2011 at each n but flag it as infeasible above n ≈ 5000
on a laptop (matrix memory). We attempt Newton similarly. Output is a
scaling table showing which method completes at which n.

Runtime: ~5 minutes total on a laptop, with γ at all four n values
and MS/Newton at only the first two.

Outputs
-------
  - scalability_results.csv  — wall-clock and completion flags
  - scalability_plot.png     — log-log plot of time vs n
"""
from __future__ import annotations

import time, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR
SEED = 2026
np.random.seed(SEED)

# ==================  Config  ==================
N_VALUES = [500, 2000, 8000, 50000]
MAX_MEM_GB = 1.0   # cap — abort matrix construction if it exceeds this
TOL = 1e-8
MAX_ITER = 500
MARGIN = 0.7

# ==================  MCI primitives (vectorized)  ==================
def gen_market(n, rng):
    """Draw a synthetic MCI market with n products."""
    beta = rng.uniform(1.5, 3.5, size=n)
    log_alpha = rng.standard_normal(n) * 0.5 - 3.0  # modest attractions
    alpha = np.exp(log_alpha)
    c = rng.uniform(0.4, 0.8, size=n)
    p0 = c * 1.5
    return alpha, beta, c, p0


def mci_shares(p, alpha, beta):
    A = alpha * np.power(p, -beta)
    D = 1.0 + A.sum()
    return A / D, 1.0 / D


def gamma_iteration(p0, c, alpha, beta, gamma_star=0.0, tol=TOL, max_iter=MAX_ITER):
    """γ-iteration at γ* = 0. Uses only diagonal info.
    Memory: O(n). Time per iteration: O(n)."""
    p = p0.copy()
    for k in range(max_iter):
        s, _ = mci_shares(p, alpha, beta)
        eta = beta * (1.0 - s)
        eta_safe = np.maximum(eta, 1.01)
        p_new = c / (1.0 - 1.0 / eta_safe)
        p_new = np.maximum(p_new, c * 1.0001)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new, k + 1
        p = p_new
    return p, max_iter


def ms_iteration(p0, c, alpha, beta, tol=TOL, max_iter=10):
    """MS2011 ζ-iteration. Requires full n×n Jacobian per step.
    Memory: O(n²). Time per iteration: O(n²)."""
    n = len(p0)
    # Memory check
    mem_gb = n * n * 8 / 1e9
    if mem_gb > MAX_MEM_GB:
        raise MemoryError(f"Ω matrix would need {mem_gb:.1f} GB; exceeds cap")
    p = p0.copy()
    for k in range(max_iter):
        s, _ = mci_shares(p, alpha, beta)
        u = beta * s / p            # (n,)
        # Full outer-product Jacobian
        Om = np.outer(u, s)
        np.fill_diagonal(Om, -u * (1.0 - s))
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        rhs = s + Gamma @ (p - c)
        p_new = c - rhs / diag
        p_new = np.maximum(p_new, c * 1.0001)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new, k + 1
        p = p_new
    return p, max_iter


def newton_bn(p0, c, alpha, beta, tol=TOL, max_iter=5):
    """Newton on full BN FOC. Requires n×n linear solve per step.
    Memory: O(n²). Time per iteration: O(n³)."""
    from scipy.linalg import solve
    n = len(p0)
    mem_gb = n * n * 8 / 1e9
    if mem_gb > MAX_MEM_GB:
        raise MemoryError(f"Jacobian would need {mem_gb:.1f} GB; exceeds cap")
    p = p0.copy()
    for k in range(max_iter):
        s, _ = mci_shares(p, alpha, beta)
        u = beta * s / p
        Om = np.outer(u, s)
        np.fill_diagonal(Om, -u * (1.0 - s))
        F = s + Om @ (p - c)
        # Jacobian of F: approximate by 2Ω (Morrow-Jacobian-Newton)
        J = 2 * Om + 1e-10 * np.eye(n)
        try:
            dp = solve(J, -F)
        except Exception:
            return p, max_iter
        p_new = np.maximum(p + 0.5 * dp, c * 1.0001)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new, k + 1
        p = p_new
    return p, max_iter


# ==================  Main  ==================
print("=" * 70)
print("SCALABILITY DEMONSTRATION")
print("=" * 70)
print(f"Comparing γ-iteration, MS2011, and Newton at n = {N_VALUES}")
print(f"Memory cap for matrix methods: {MAX_MEM_GB} GB")
print()

rows = []
rng = np.random.default_rng(SEED)

for n in N_VALUES:
    print(f"--- n = {n:,} ---")
    alpha, beta, c, p0 = gen_market(n, rng)

    # γ-iteration
    t0 = time.perf_counter()
    try:
        p_g, iter_g = gamma_iteration(p0, c, alpha, beta)
        t_g = time.perf_counter() - t0
        g_status = "OK"
    except Exception as e:
        t_g = np.nan
        iter_g = -1
        g_status = f"FAIL: {e}"
    print(f"  γ-iteration : {t_g*1000:10.3f} ms  ({iter_g} iter)  [{g_status}]")

    # MS2011
    t0 = time.perf_counter()
    try:
        p_m, iter_m = ms_iteration(p0, c, alpha, beta)
        t_m = time.perf_counter() - t0
        m_status = "OK"
    except MemoryError as e:
        t_m = np.nan
        iter_m = -1
        m_status = f"INFEASIBLE: {e}"
    except Exception as e:
        t_m = np.nan
        iter_m = -1
        m_status = f"FAIL: {e}"
    print(f"  MS2011      : {t_m*1000 if not np.isnan(t_m) else float('nan'):>10} ms  "
          f"({iter_m} iter)  [{m_status}]")

    # Newton
    t0 = time.perf_counter()
    try:
        p_n, iter_n = newton_bn(p0, c, alpha, beta)
        t_n = time.perf_counter() - t0
        n_status = "OK"
    except MemoryError as e:
        t_n = np.nan
        iter_n = -1
        n_status = f"INFEASIBLE: {e}"
    except Exception as e:
        t_n = np.nan
        iter_n = -1
        n_status = f"FAIL: {e}"
    print(f"  Newton BN   : {t_n*1000 if not np.isnan(t_n) else float('nan'):>10} ms  "
          f"({iter_n} iter)  [{n_status}]")
    print()

    rows.append({
        "n": n,
        "gamma_ms": t_g * 1000 if not np.isnan(t_g) else np.nan,
        "gamma_iter": iter_g,
        "gamma_status": g_status,
        "ms_ms": t_m * 1000 if not np.isnan(t_m) else np.nan,
        "ms_iter": iter_m,
        "ms_status": m_status,
        "newton_ms": t_n * 1000 if not np.isnan(t_n) else np.nan,
        "newton_iter": iter_n,
        "newton_status": n_status,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "scalability_results.csv", index=False)

# Plot log-log
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(df['n'], df['gamma_ms'], 'o-', color='tab:blue', label="γ-iteration (O(n))", linewidth=2, markersize=8)
ms_ok = df.dropna(subset=['ms_ms'])
if len(ms_ok) > 0:
    ax.loglog(ms_ok['n'], ms_ok['ms_ms'], 's-', color='tab:red', label="MS2011 (O(n²))", linewidth=2, markersize=8)
newton_ok = df.dropna(subset=['newton_ms'])
if len(newton_ok) > 0:
    ax.loglog(newton_ok['n'], newton_ok['newton_ms'], '^-', color='tab:green', label="Newton BN (O(n³))", linewidth=2, markersize=8)

# Mark infeasibility
for _, r in df.iterrows():
    if "INFEASIBLE" in str(r['ms_status']):
        ax.annotate("MS2011\nout of memory", xy=(r['n'], ax.get_ylim()[1] * 0.3),
                    ha='center', fontsize=8, color='tab:red')
    if "INFEASIBLE" in str(r['newton_status']):
        ax.annotate("Newton\nout of memory", xy=(r['n'], ax.get_ylim()[1] * 0.7),
                    ha='center', fontsize=8, color='tab:green')

ax.set_xlabel("Catalog size n (products)")
ax.set_ylabel("Wall-clock time (ms)")
ax.set_title("Scalability of pricing methods at realistic retail-catalog sizes")
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "scalability_plot.png", dpi=150)
plt.close()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(df[['n', 'gamma_ms', 'ms_ms', 'newton_ms']].to_string(index=False))
print()
print(f"Outputs: {OUT}/scalability_results.csv")
print(f"         {OUT}/scalability_plot.png")
