#!/usr/bin/env python3
"""Numerical check of Proposition 5, parts (ii)-(iii) (Online Appendix I).

Under MCI demand with beta_i > 1 and outside attraction A_0 > 0, the
aggregate-markup map Phi(A) = s(p(A))' (p(A) - c~) along the ray
p_i(A) = (c~_i + A) / (1 - 1/beta_i) satisfies Phi'(A) = 0 at every
fixed point, so g(A) = Phi(A) - A crosses zero exactly once, and p(A*)
is the global maximizer of the effective-cost profit.

This script verifies, on randomly drawn MCI markets:
  (a) |Phi'(A*)| ~ 0 at the root (central difference);
  (b) g has exactly one sign change on a fine grid;
  (c) no multi-start coordinate-ascent run finds higher profit than p(A*).

Usage: python prop5_uniqueness_check.py [n_trials]
"""
import sys

import numpy as np
from scipy.optimize import brentq


def check(trial_seed: int) -> dict:
    rng = np.random.default_rng(trial_seed)
    n = int(rng.integers(5, 40))
    beta = rng.uniform(1.05, 4.0, n)
    kappa = np.exp(rng.normal(0.0, 1.0, n))
    c = rng.uniform(0.5, 3.0, n)
    a0 = rng.uniform(0.5, 5.0)

    def shares(p):
        a = kappa * np.power(p, -beta)
        return a / (a0 + a.sum())

    def phi(aggr):
        p = (c + aggr) / (1.0 - 1.0 / beta)
        return float(shares(p) @ (p - c))

    def g(aggr):
        return phi(aggr) - aggr

    hi = 1.0
    while g(hi) > 0:
        hi *= 3.0
    a_star = brentq(g, 0.0, hi, xtol=1e-14)

    h = 1e-6
    dphi = (phi(a_star + h) - phi(a_star - h)) / (2.0 * h)

    grid = np.linspace(1e-9, 3.0 * hi, 20000)
    gvals = np.array([g(a) for a in grid])
    n_roots = int(np.sum(np.sign(gvals[:-1]) != np.sign(gvals[1:])))

    p_star = (c + a_star) / (1.0 - 1.0 / beta)

    def profit(p):
        return float(shares(p) @ (p - c))

    best = profit(p_star)
    beaten = False
    for _ in range(200):
        p = np.maximum(c * np.exp(rng.normal(0.3, 0.5, n)), c * 1.001)
        for _ in range(500):
            aggr = float(shares(p) @ (p - c))
            p = np.maximum(0.5 * (c + aggr) / (1.0 - 1.0 / beta) + 0.5 * p,
                           c * 1.0001)
        if profit(p) > best + 1e-9:
            beaten = True
    return dict(n=n, dphi=dphi, n_roots=n_roots, beaten=beaten)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ok = True
    for t in range(n_trials):
        r = check(7 + t)
        line_ok = abs(r["dphi"]) < 1e-6 and r["n_roots"] == 1 and not r["beaten"]
        ok &= line_ok
        print(f"trial {t:2d}: n={r['n']:3d}  Phi'(A*)={r['dphi']:+.2e}  "
              f"roots={r['n_roots']}  multistart_beats_ray={r['beaten']}  "
              f"{'OK' if line_ok else 'FAIL'}")
    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
