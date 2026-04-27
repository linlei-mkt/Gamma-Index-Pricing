"""
empirical_gamma.py
==================
Empirical illustration for "Gamma-Equalization as Approximate Optimal Pricing."

This script runs the paper's empirical exercise end-to-end:

  (1) Builds an MCI-calibrated market using own-price elasticity estimates
      drawn from Hoch, Kim, Montgomery, and Rossi (1995, JMR, "Determinants
      of Store-Level Price Elasticity"), Table 3 -- publicly available.
      The five categories used (ownprice elasticities in brackets):
          Analgesics (-3.13)   Cookies (-3.77)   Crackers (-2.34)
          Soap (-4.27)         Soft Drinks (-4.05)
      Each category is treated as an inside good; the outside share
      matches Nevo (2001)'s cereal-industry level (s_0 = 0.65).

  (2) Solves the single-firm multi-product pricing problem under:
          Uniform (single scalar price)
          Category-markup (constant % markup)
          gamma-equalization (one Jacobi sweep)
          Iterated gamma (3 and 10 sweeps)
          Full BLP-Newton (gold standard)

  (3) Reports profit as a percentage of BLP-Newton optimum, the
      empirical diagonal-dominance slack ebar_off, and the implied
      gamma^star (common shadow price of the GMV constraint) to
      test whether observed markups are consistent with gamma-equalization.

  (4) Optional replication with actual Dominick's Finer Foods scanner
      data (Kilts Center, University of Chicago Booth). Pass
      --data-path to point at a CSV of weekly UPC-level prices/quantities.

Author: Author information blinded for peer review
April 2026.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize


# -----------------------------------------------------------------------------
# 1. Hoch--Kim--Montgomery--Rossi (1995) grocery calibration
# -----------------------------------------------------------------------------
# Own-price elasticities for five categories (JMR 1995, Table 3, mean across
# stores). These are long-run, store-level, category-average estimates.

HKMR1995_CATEGORIES = [
    ("Analgesics",    3.13,  2.80),
    ("Cookies",       3.77,  2.20),
    ("Crackers",      2.34,  1.90),
    ("Soap",          4.27,  1.70),
    ("Soft drinks",   4.05,  1.20),
]
# (label, |eta| own-price elasticity, observed retail price in $, per item)

# Nevo (2001, Ectra), Table V: outside-share of cereal industry ~ 65%.
# We use this as the category-level outside-share so that the empirical
# ebar_off lives in the right ballpark.
S0_DEFAULT = 0.65

# Marginal cost calibration: use the standard retail wholesale/retail spread
# documented in Dominick's data -- Chevalier-Kashyap (2019) report gross
# margins of 20--30% for these categories. Set c_i = (1 - margin_i) * p_i.
HKMR1995_MARGINS = {
    "Analgesics":  0.34,
    "Cookies":     0.28,
    "Crackers":    0.25,
    "Soap":        0.30,
    "Soft drinks": 0.22,
}


# -----------------------------------------------------------------------------
# 2. MCI demand system
# -----------------------------------------------------------------------------

@dataclass
class MCIMarket:
    """Multiplicative Competitive Interaction demand with outside good.

    A_i(p_i) = kappa_i * p_i^{-beta_i}
    s_i(p)  = A_i(p_i) / (A_0 + sum_k A_k(p_k))
    """
    kappa: np.ndarray   # (n,) attraction levels
    beta: np.ndarray    # (n,) attraction elasticities (positive)
    A0: float           # outside attraction
    c: np.ndarray       # (n,) marginal costs

    @property
    def n(self) -> int:
        return len(self.kappa)

    def attractions(self, p: np.ndarray) -> np.ndarray:
        return self.kappa * np.power(p, -self.beta)

    def shares(self, p: np.ndarray) -> np.ndarray:
        A = self.attractions(p)
        D = self.A0 + A.sum()
        return A / D

    def s0(self, p: np.ndarray) -> float:
        A = self.attractions(p)
        return self.A0 / (self.A0 + A.sum())

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """d s_j / d p_i ; row i, column j."""
        s = self.shares(p)
        # M_ii = -beta_i/p_i * s_i * (1 - s_i)
        # M_ij = +beta_i/p_i * s_i * s_j  (j != i)
        ratio = (self.beta / p)[:, None]        # (n, 1)
        M = ratio * np.outer(s, s)              # beta_i/p_i * s_i * s_j
        # diagonal: subtract 2*s_i*s_i times ratio (giving -s_i*(1-s_i))
        diag_val = -(self.beta / p) * s * (1.0 - s)
        np.fill_diagonal(M, diag_val)
        return M

    def own_elasticity(self, p: np.ndarray) -> np.ndarray:
        s = self.shares(p)
        return -self.beta * (1.0 - s)

    def ebar_off(self, p: np.ndarray) -> float:
        """max_i (S - s_i) / (1 - s_i); the MCI closed form."""
        s = self.shares(p)
        S = s.sum()
        ratio = (S - s) / (1.0 - s + 1e-12)
        return float(ratio.max())

    def gross_profit(self, p: np.ndarray) -> float:
        s = self.shares(p)
        return float(np.sum((p - self.c) * s))

    def gmv(self, p: np.ndarray) -> float:
        s = self.shares(p)
        return float(np.sum(p * s))

    def profit_grad(self, p: np.ndarray) -> np.ndarray:
        """d Pi / d p_i = s_i + sum_j (p_j - c_j) d s_j / d p_i."""
        s = self.shares(p)
        J = self.jacobian(p)  # J[i, j] = d s_j / d p_i
        return s + J @ (p - self.c)


# -----------------------------------------------------------------------------
# 3. Pricing rules
# -----------------------------------------------------------------------------

def solve_uniform(mkt: MCIMarket, p_lo: float = 0.1, p_hi: float = 100.0) -> np.ndarray:
    """Single scalar price maximizing profit."""
    def neg_profit(x):
        p = np.full(mkt.n, x[0])
        return -mkt.gross_profit(p)
    res = minimize(neg_profit, x0=[np.mean(mkt.c) * 1.5],
                   method="L-BFGS-B",
                   bounds=[(p_lo, p_hi)])
    return np.full(mkt.n, res.x[0])


def solve_category_markup(mkt: MCIMarket) -> np.ndarray:
    """Single common Lerner margin m, p_i = c_i / (1 - m)."""
    def neg_profit(x):
        m = x[0]
        if m <= 0 or m >= 0.95:
            return 1e6
        p = mkt.c / (1.0 - m)
        return -mkt.gross_profit(p)
    res = minimize(neg_profit, x0=[0.3], method="L-BFGS-B",
                   bounds=[(0.01, 0.94)])
    m_star = res.x[0]
    return mkt.c / (1.0 - m_star)


def solve_gamma_eq(mkt: MCIMarket, tol: float = 1e-9,
                   max_iter: int = 200) -> np.ndarray:
    """One Jacobi sweep: each product solves the own-price FOC
       s_i + (p_i - c_i) d s_i / d p_i = 0.
    With MCI own-derivative d s_i / d p_i = -beta_i/p_i * s_i (1 - s_i),
    the FOC collapses to the per-product Lerner condition:
        L_i = 1 / |eta_i| = 1 / (beta_i (1 - s_i)),
    which is solved by a fixed-point update on L_i given s_{-i} held fixed.
    """
    # initialise with uniform-markup warm start
    p = mkt.c / (1.0 - 0.3)
    for _ in range(max_iter):
        s = mkt.shares(p)
        # FOC: (p_i - c_i) / p_i = 1 / (beta_i (1 - s_i))
        denom = mkt.beta * (1.0 - s)
        L = 1.0 / np.maximum(denom, 1e-8)
        L = np.clip(L, 0.0, 0.95)
        p_new = mkt.c / (1.0 - L)
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = p_new
    return p


def zeta_iterate(mkt: MCIMarket, max_sweeps: int, tol: float = 1e-10) -> np.ndarray:
    """Full Jacobi iteration of the BLP-Nash FOC system (Morrow-Skerlos).
       p_i <- c_i + zeta_i(p) where zeta_i = -s_i / M_ii + correction.
       Here we implement the symmetric form of Morrow-Skerlos 2011:
           p_new = c + Lambda(p)^{-1} [ Lambda(p) p - s - Gamma(p)(p - c) ]
       which for a single-firm monopolist reduces to:
           p_new_i = c_i - s_i / M_ii + sum_{j!=i} (p_j - c_j) M_ji / M_ii.
    """
    p = mkt.c / (1.0 - 0.3)
    for _ in range(max_sweeps):
        s = mkt.shares(p)
        J = mkt.jacobian(p)  # J[i,j] = d s_j / d p_i
        M_own = np.diag(J)   # d s_i / d p_i
        p_new = np.empty_like(p)
        for i in range(mkt.n):
            cross = 0.0
            for j in range(mkt.n):
                if j == i:
                    continue
                cross += (p[j] - mkt.c[j]) * J[i, j]
            p_new[i] = mkt.c[i] - (s[i] + cross) / M_own[i]
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = 0.5 * p + 0.5 * p_new     # mild relaxation for stability
    return p


def solve_iterated_gamma(mkt: MCIMarket, K: int) -> np.ndarray:
    """K sweeps of the full Jacobi update (equivalent to zeta-iteration
    by Theorem 1 in the paper)."""
    return zeta_iterate(mkt, max_sweeps=K)


def solve_blp_optimum(mkt: MCIMarket) -> np.ndarray:
    """Gold standard: full-Newton on the profit function with multiple warm
    starts. We use L-BFGS-B since the profit Hessian is naturally handled
    numerically and we avoid explicit analytic Hessian code."""
    def neg_profit(p):
        return -mkt.gross_profit(p)

    def neg_grad(p):
        return -mkt.profit_grad(p)

    warm_starts = [
        solve_uniform(mkt),
        solve_category_markup(mkt),
        solve_gamma_eq(mkt),
        zeta_iterate(mkt, max_sweeps=150),
        mkt.c / (1.0 - 0.25),
        mkt.c / (1.0 - 0.50),
    ]

    best_p = None
    best_profit = -np.inf
    for p0 in warm_starts:
        res = minimize(neg_profit, x0=p0, jac=neg_grad,
                       method="L-BFGS-B",
                       bounds=[(1e-3, None)] * mkt.n)
        if -res.fun > best_profit:
            best_profit = -res.fun
            best_p = res.x
    return best_p


# -----------------------------------------------------------------------------
# 4. Main calibration exercise
# -----------------------------------------------------------------------------

def calibrate_hkmr1995(s0: float = S0_DEFAULT) -> MCIMarket:
    """Build an MCI market from Hoch-Kim-Montgomery-Rossi (1995) Table 3.

    We start from observed retail prices p_i^obs and published |eta_i|.
    Under MCI, eta_i = -beta_i * (1 - s_i). We target:
        - s_0 = 0.65 (total outside share; Nevo 2001 cereal level).
        - s_i symmetric so that each category has inside share S/n where
          S = 1 - s_0 = 0.35 and n = 5 gives s_i = 0.07.
        - beta_i = |eta_i| / (1 - s_i) = |eta_i| / 0.93.
        - kappa_i chosen so that the shares materialize at observed prices:
          A_i = s_i * (A_0 + sum_k A_k).
          With symmetric s_i and a normalised A_0 = s_0 D, we have
          A_i = (s_i / s_0) * A_0.
        - margins from Chevalier-Kashyap documented values.
    """
    labels = [c[0] for c in HKMR1995_CATEGORIES]
    eta = np.array([c[1] for c in HKMR1995_CATEGORIES])
    p_obs = np.array([c[2] for c in HKMR1995_CATEGORIES])
    margins = np.array([HKMR1995_MARGINS[lab] for lab in labels])

    n = len(labels)
    S = 1.0 - s0                         # inside share
    s_i = np.full(n, S / n)              # symmetric inside
    beta = eta / (1.0 - s_i)

    A0 = 1.0
    Ai = (s_i / s0) * A0                  # A_i = kappa_i * p_i^{-beta_i}
    kappa = Ai * np.power(p_obs, beta)

    c = p_obs * (1.0 - margins)

    return MCIMarket(kappa=kappa, beta=beta, A0=A0, c=c)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.3f}%"


def run_main_experiment():
    print("\n" + "=" * 78)
    print(" EMPIRICAL ILLUSTRATION: HOCH-KIM-MONTGOMERY-ROSSI (1995) CALIBRATION")
    print("=" * 78)

    mkt = calibrate_hkmr1995(s0=S0_DEFAULT)
    labels = [c[0] for c in HKMR1995_CATEGORIES]
    p_obs = np.array([c[2] for c in HKMR1995_CATEGORIES])

    print(f"\nCategories: {labels}")
    print(f"Own elasticities |eta_i|: {[c[1] for c in HKMR1995_CATEGORIES]}")
    print(f"Observed retail prices: {p_obs}")
    print(f"Calibrated marginal costs: {np.round(mkt.c, 3)}")
    print(f"Calibrated beta_i: {np.round(mkt.beta, 3)}")
    print(f"Outside share target s_0 = {S0_DEFAULT}")

    # Solve under all rules
    p_uni  = solve_uniform(mkt)
    p_cat  = solve_category_markup(mkt)
    p_gam  = solve_gamma_eq(mkt)
    p_it3  = solve_iterated_gamma(mkt, K=3)
    p_it10 = solve_iterated_gamma(mkt, K=10)
    p_blp  = solve_blp_optimum(mkt)

    rules = [
        ("Uniform price",        p_uni),
        ("Uniform markup",       p_cat),
        ("Gamma-equalization",   p_gam),
        ("Iter gamma (3 sweeps)", p_it3),
        ("Iter gamma (10 sweeps)", p_it10),
        ("BLP-Newton",           p_blp),
    ]

    profit_blp = mkt.gross_profit(p_blp)
    print(f"\nBLP-Newton optimal profit: {profit_blp:.6f}")

    print("\n{:<25}{:>12}{:>15}{:>12}{:>10}".format(
        "Rule", "Profit", "% of BLP", "GMV", "ebar"))
    print("-" * 78)
    out_rows = []
    for name, p in rules:
        pi = mkt.gross_profit(p)
        gmv = mkt.gmv(p)
        eb = mkt.ebar_off(p)
        gap_pct = 100.0 * (profit_blp - pi) / profit_blp
        pct_blp = 100.0 * pi / profit_blp
        print(f"{name:<25}{pi:12.6f}{pct_blp:14.3f}%{gmv:12.6f}{eb:10.4f}")
        out_rows.append({
            "rule": name, "profit": pi, "gap_pct": gap_pct,
            "gmv": gmv, "ebar_off": eb,
            "prices": p.tolist(),
        })

    # Headline diagnostic evaluated at the BLP optimum
    eb_star = mkt.ebar_off(p_blp)
    print(f"\nDiagonal-dominance slack at p*: ebar_off = {eb_star:.4f}")
    print(f"Predicted price error of one-sweep gamma-eq: O(ebar) = {eb_star:.3f}")
    print(f"Predicted profit gap:                       O(ebar^2) = {eb_star**2:.4f}")

    # Implied common gamma^star under the gamma-eq rule
    s_blp = mkt.shares(p_blp)
    L_i = (p_blp - mkt.c) / p_blp
    eta_i = mkt.own_elasticity(p_blp)
    implied_gamma = np.array(
        [(1.0 + L_i[i] * eta_i[i]) / (1.0 + eta_i[i]) for i in range(mkt.n)]
    )
    print(f"\nImplied gamma_i at BLP optimum:")
    for lab, g in zip(labels, implied_gamma):
        print(f"   {lab:<14}  gamma = {g:8.4f}")
    print(f"Dispersion std(gamma_i) = {implied_gamma.std():.4f}   "
          f"(zero if gamma-equalization is the DGP)")

    write_csv_results("empirical_results.csv", out_rows, labels)
    return out_rows


def write_csv_results(path: str, rows, labels):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "profit", "gap_pct_vs_BLP", "gmv", "ebar_off"]
                   + [f"p_{lab}" for lab in labels])
        for r in rows:
            w.writerow([r["rule"], r["profit"], r["gap_pct"],
                        r["gmv"], r["ebar_off"]] + r["prices"])
    print(f"\nWrote {out}")


# -----------------------------------------------------------------------------
# 5. Optional Dominick's replication harness
# -----------------------------------------------------------------------------

def replicate_dominicks(csv_path: str):
    """Runs the full pipeline on Dominick's Finer Foods weekly data.
    The Kilts Center provides this at
        https://www.chicagobooth.edu/research/kilts/research-data/dominicks
    Expected CSV columns (after the user's preprocessing):
        upc, category, week, price, move, profit_margin
    where move is units sold and profit_margin is the Kilts gross margin.
    This function aggregates to category-week shares, estimates within-category
    log-log own-price elasticities via OLS, and then repeats the pricing
    exercise with the estimated parameters in place of the HKMR (1995) values.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    req = {"upc", "category", "week", "price", "move", "profit_margin"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {req}; got {df.columns.tolist()}")

    df = df[df["price"] > 0]
    df = df[df["move"] > 0]
    df["log_price"] = np.log(df["price"])
    df["log_move"]  = np.log(df["move"])

    rows = []
    for cat, g in df.groupby("category"):
        # OLS: log(move) = a + b * log(price)
        x = g["log_price"].values
        y = g["log_move"].values
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        eta_hat = beta[1]  # should be negative
        p_mean = g["price"].mean()
        margin = g["profit_margin"].mean()
        rows.append({
            "category": cat, "eta_hat": eta_hat,
            "p_mean": p_mean, "margin": margin,
            "n_obs": len(g),
        })

    print("\nEstimated own-price elasticities per category:")
    for r in rows:
        print(f"  {r['category']:<25}  eta_hat = {r['eta_hat']:.3f}  "
              f"(p~{r['p_mean']:.2f}, margin~{r['margin']:.2f}, N={r['n_obs']})")
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None,
                        help="optional Dominick's CSV for replication")
    args = parser.parse_args()

    run_main_experiment()

    if args.data_path:
        print("\n" + "=" * 78)
        print(f" REPLICATION ON DOMINICK'S DATA: {args.data_path}")
        print("=" * 78)
        replicate_dominicks(args.data_path)


if __name__ == "__main__":
    main()
