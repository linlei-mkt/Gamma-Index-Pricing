#!/usr/bin/env python3
"""GMV capacity of single-scalar pricing families (referee suggestion 1).

For each JD daily market, compute:
  R_max      revenue-maximizing prices over ALL positive price vectors.
             Under MCI the revenue FOC is s + Omega p = 0, the mu -> infinity
             limit of the FI Lagrangian family, i.e. the Prop-5 ray with
             effective costs zero: p_i = A beta_i/(beta_i - 1).
  Rbar_gamma sup of R along the gamma-equalization path (grid + refinement).
  Rbar_unif  sup of R along the uniform-markup path p = c/(1-m).

Reported: capacity ratios Rbar/R_max, the maximum feasible floor multiplier
phi_max = Rbar/R_obs for each family, and the correlation with ebar.

Output: reference_output_results/gamma_capacity.csv + figures/gamma_capacity.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

G = Path(__file__).resolve().parents[2]
N_TOP, M_MULT, MR, BF = 500, 3.0, 0.70, 1.2

orders = pd.read_csv(G / "JD_MSOM/JD_order_data.csv",
                     usecols=["sku_ID", "order_date", "quantity", "final_unit_price"],
                     dtype={"sku_ID": "string"}, parse_dates=["order_date"])
sku = pd.read_csv(G / "JD_MSOM/JD_sku_data.csv", usecols=["sku_ID", "type"],
                  dtype={"sku_ID": "string"})
top = orders.groupby("sku_ID").size().nlargest(N_TOP).index
o = orders[orders["sku_ID"].isin(top)].copy()
o["day"] = o["order_date"].dt.day
agg = (o.assign(rev=o["final_unit_price"] * o["quantity"])
       .groupby(["day", "sku_ID"], as_index=False)
       .agg(qty=("quantity", "sum"), rev=("rev", "sum")))
agg["price"] = agg["rev"] / agg["qty"]
agg = agg.merge(sku, on="sku_ID", how="left")
agg = agg[(agg["qty"] > 0) & (agg["price"] > 0) & agg["type"].notna()].copy()
M = M_MULT * agg.groupby("day")["qty"].sum().max()
agg["share"] = agg["qty"] / M
post = pd.read_csv(G / "MainCodes/jd_hb_posterior_summary.csv")
raw_map = dict(zip(post["sku_ID"], post["beta_posterior_mean"]))
agg = agg[agg["sku_ID"].isin(raw_map)].copy()


def shares(p, a, b):
    A = a * np.power(p, -b)
    return A / (1.0 + A.sum())


def rev(p, a, b):
    return float(shares(p, a, b) @ p)


def g_it(p0, a, b, c, gs, sweeps=600, tol=1e-9):
    for omega in (1.0, 0.5, 0.25):
        p = p0.copy()
        for _ in range(sweeps):
            s = shares(p, a, b)
            eta = np.maximum(b * (1.0 - s), 1.01)
            pn = (1.0 - omega) * p + omega * c / ((1.0 - gs) * (1.0 - 1.0 / eta))
            if np.max(np.abs(pn - p) / p) < tol:
                return pn, True
            p = pn
    return p, False


def r_max_ray(a, b):
    """Revenue max: Prop-5 ray at zero effective cost, p_i = A b_i/(b_i-1)."""
    def gA(A):
        p = A * b / (b - 1.0)
        return float(shares(p, a, b) @ p) - A
    lo = 1e-9
    hi = 1.0
    while gA(hi) > 0:
        hi *= 3.0
    A = brentq(gA, lo, hi, xtol=1e-12)
    p = A * b / (b - 1.0)
    return rev(p, a, b), p


rows = []
for day, g in agg.groupby("day"):
    p_obs = g["price"].to_numpy()
    b = np.maximum(np.array([raw_map[x] for x in g["sku_ID"]]), BF)
    s_obs = g["share"].to_numpy()
    a = (s_obs / (1.0 - s_obs.sum())) * np.power(p_obs, b)
    c = MR * p_obs
    R_obs = rev(p_obs, a, b)
    S_obs = float(s_obs.sum())
    ebar = max((S_obs - s_obs) / (1.0 - s_obs))

    # (1) unrestricted revenue max (also verify > coordinate refinements)
    R_max, p_rmax = r_max_ray(a, b)

    # (2) gamma-path capacity: grid then golden refinement
    gs_grid = np.arange(0.0, -10.0001, -0.02)
    best_R, best_g, conv_all = -1.0, 0.0, True
    Rs = []
    for gsv in gs_grid:
        p, conv = g_it(p_obs, a, b, c, gsv)
        conv_all &= conv
        r = rev(p, a, b)
        Rs.append(r)
        if r > best_R:
            best_R, best_g = r, gsv
        if len(Rs) > 25 and r < 0.5 * best_R:
            break
    lo_g, hi_g = best_g - 0.02, best_g + 0.02
    res = minimize_scalar(lambda gv: -rev(g_it(p_obs, a, b, c, gv)[0], a, b),
                          bounds=(lo_g, min(hi_g, 0.0)), method="bounded",
                          options={"xatol": 1e-6})
    Rbar_g = max(best_R, -res.fun)
    g_at_peak = float(res.x) if -res.fun >= best_R else best_g

    # (3) uniform-path capacity
    res_u = minimize_scalar(lambda m: -rev(c / (1.0 - m), a, b),
                            bounds=(-20.0, 0.999), method="bounded",
                            options={"xatol": 1e-8})
    Rbar_u = -res_u.fun

    rows.append(dict(day=int(day), ebar=ebar, S=S_obs,
                     R_obs=R_obs, R_max=R_max,
                     cap_gamma=Rbar_g / R_max, cap_unif=Rbar_u / R_max,
                     phi_max_gamma=Rbar_g / R_obs, phi_max_unif=Rbar_u / R_obs,
                     phi_max_fi=R_max / R_obs, g_peak=g_at_peak,
                     conv_all=conv_all))

d = pd.DataFrame(rows).sort_values("ebar", ascending=False)
d.to_csv(G / "reference_output_results/gamma_capacity.csv", index=False)

print("=== 按 ebar 降序（前8天）===")
print(d.head(8)[["day", "ebar", "phi_max_gamma", "phi_max_unif", "phi_max_fi",
                 "cap_gamma", "cap_unif", "g_peak"]].round(3).to_string(index=False))
print("\n=== 汇总 ===")
print(f"cap_gamma = Rbar_g/R_max: min {d.cap_gamma.min():.3f}, "
      f"median {d.cap_gamma.median():.3f}, max {d.cap_gamma.max():.3f}")
print(f"cap_unif: min {d.cap_unif.min():.3f}, median {d.cap_unif.median():.3f}")
print(f"phi_max_gamma: min {d.phi_max_gamma.min():.3f}, median {d.phi_max_gamma.median():.3f}")
print(f"phi_max_fi:    min {d.phi_max_fi.min():.3f}, median {d.phi_max_fi.median():.3f}")
print(f"corr(ebar, cap_gamma) = {d.ebar.corr(d.cap_gamma):.3f}")
print(f"corr(ebar, phi_max_gamma) = {d.ebar.corr(d.phi_max_gamma):.3f}")
print(f"days with phi_max_gamma < 1.15: {(d.phi_max_gamma < 1.15).sum()}"
      f" | < 1.20: {(d.phi_max_gamma < 1.20).sum()}"
      f" | < 1.25: {(d.phi_max_gamma < 1.25).sum()}")
print(f"all grid solves converged: {bool(d.conv_all.all())}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
a1.scatter(d.ebar, d.phi_max_gamma, c="tab:blue", label=r"$\gamma$ family")
a1.scatter(d.ebar, d.phi_max_unif, c="tab:orange", marker="^", label="uniform family")
a1.scatter(d.ebar, d.phi_max_fi, c="tab:red", marker="o", facecolors="none",
           label="FI (unrestricted)")
for phi in [1.15, 1.25]:
    a1.axhline(phi, color="grey", lw=0.8, ls="--")
a1.set_xlabel(r"$\bar e$ at observed prices")
a1.set_ylabel(r"maximum attainable floor $\bar R / R^{\rm obs}$")
a1.legend(fontsize=8)
a1.set_title("GMV capacity by pricing family (31 days)", fontsize=10)
a2.scatter(d.ebar, 100 * (1 - d.cap_gamma), c="tab:blue")
a2.set_xlabel(r"$\bar e$ at observed prices")
a2.set_ylabel(r"capacity shortfall $100(1-\bar R_\gamma/R_{\max})$ (%)")
a2.set_title("Single-scalar capacity shortfall vs. exposure", fontsize=10)
plt.tight_layout()
plt.savefig(G / "figures/gamma_capacity.png", dpi=150)
print("figure written")
