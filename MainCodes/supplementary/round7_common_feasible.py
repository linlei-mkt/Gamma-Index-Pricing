#!/usr/bin/env python3
"""Round-7 rerun: common feasible set for the constrained comparison.

The gamma-iteration previously safeguarded prices at p >= 1.0001c while the
scalar-FI benchmark allowed below-cost prices. Under MCI the raw update is
p = c / [(1-gamma*)(1-1/eta)], which is strictly positive whenever eta > 1,
so the cost safeguard can be dropped and both methods share the feasible
set R^n_++ (no price boxes).

Outputs (reference_output_results/):
  round7_constrained_daily.csv   per-day phi=1.15 comparison (unclamped gamma)
  round7_floor_summary.csv       floor grid phi in {1.05..1.25}
  round7_beta_floor_sens.csv     beta-floor grid {1.05,1.1,1.2,1.3,1.5}
Figures (figures/):
  jd_gmv_profit_vs_GMV.png, jd_gmv_gamma_star_distribution.png
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

G = Path(__file__).resolve().parents[2]
OUT = G / "reference_output_results"
FIG = G / "figures"
N_TOP, M_MULT, MR, BF = 500, 3.0, 0.70, 1.2

# ---------------- shared JD pipeline ----------------
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


def markets(beta_floor):
    out = []
    for day, g in agg.groupby("day"):
        p_obs = g["price"].to_numpy()
        s_obs = g["share"].to_numpy()
        b = np.maximum(np.array([raw_map[x] for x in g["sku_ID"]]), beta_floor)
        c = MR * p_obs
        s0 = 1.0 - s_obs.sum()
        a = (s_obs / s0) * np.power(p_obs, b)
        out.append((int(day), p_obs, b, a, c))
    return out


def shares(p, a, b):
    A = a * np.power(p, -b)
    return A / (1.0 + A.sum())


def prof(p, a, b, c):
    return float(shares(p, a, b) @ (p - c))


def rev(p, a, b):
    return float(shares(p, a, b) @ p)


def scalar_fi(a, b, c, ceff=None):
    cc = c if ceff is None else ceff
    def gA(A):
        p = (cc + A) / (1.0 - 1.0 / b)
        return float(shares(p, a, b) @ (p - cc)) - A
    hi = 1.0
    while gA(hi) > 0:
        hi *= 3.0
    A = brentq(gA, 0.0, hi, xtol=1e-12)
    return (cc + A) / (1.0 - 1.0 / b)


def constrained_fi(a, b, c, Rt):
    p = scalar_fi(a, b, c)
    if rev(p, a, b) >= Rt:
        return p, 0.0
    f = lambda mu: rev(scalar_fi(a, b, c, ceff=c / (1.0 + mu)), a, b) - Rt
    mu = brentq(f, 1e-8, 100.0, xtol=1e-8)
    return scalar_fi(a, b, c, ceff=c / (1.0 + mu)), mu


def g_it(p0, a, b, c, gs, sweeps=400, tol=1e-8):
    """Unclamped gamma-iteration: p = c/[(1-gs)(1-1/eta)] > 0 whenever eta > 1.
    Falls back to 0.5-damped updates if the undamped map oscillates."""
    for omega in (1.0, 0.5):
        p = p0.copy()
        for _ in range(sweeps):
            s = shares(p, a, b)
            eta = np.maximum(b * (1.0 - s), 1.01)
            den = (1.0 - gs) * (1.0 - 1.0 / eta)
            pn = (1.0 - omega) * p + omega * c / den
            if np.max(np.abs(pn - p) / p) < tol:
                return pn, True
            p = pn
    return p, False


def tune_gamma(p0, a, b, c, Rt, step=0.05, g_min=-8.0, iters=60):
    """Find the least-negative gamma* with R(p(gamma*)) = Rt.

    Revenue along the gamma path is hump-shaped once below-cost prices are
    allowed (shares saturate while prices fall), so bracket by stepping down
    from 0 and declare infeasibility only after the revenue peak is passed."""
    g_prev, r_prev = 0.0, rev(g_it(p0, a, b, c, 0.0)[0], a, b)
    if r_prev >= Rt:
        return 0.0, g_it(p0, a, b, c, 0.0)[0]
    declines = 0
    g = -step
    while g >= g_min:
        r = rev(g_it(p0, a, b, c, g)[0], a, b)
        if r >= Rt:
            lo, hi = g, g_prev            # bracket: R(lo) >= Rt > R(hi)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if rev(g_it(p0, a, b, c, mid)[0], a, b) < Rt:
                    hi = mid
                else:
                    lo = mid
            gs = 0.5 * (lo + hi)
            return gs, g_it(p0, a, b, c, gs)[0]
        declines = declines + 1 if r < r_prev else 0
        if declines >= 2:
            return None, None             # past the revenue peak, still short
        g_prev, r_prev = g, r
        g -= step
    return None, None


def tune_uniform(a, b, c, Rt):
    def gap_at(m):
        p = c / (1.0 - m)
        return rev(p, a, b) - Rt
    lo, hi = 1e-4, 0.999
    if gap_at(lo) < 0:
        return None, None
    m = brentq(lambda m: gap_at(m), lo, hi, xtol=1e-10) if gap_at(hi) < 0 else None
    if m is None:  # revenue still above target at hi? pick profit-max m meeting floor
        r = minimize_scalar(lambda m: -prof(c / (1.0 - m), a, b, c),
                            bounds=(lo, hi), method="bounded")
        m = float(r.x)
    p = c / (1.0 - m)
    if rev(p, a, b) < Rt - 1e-6 * Rt:
        return None, None
    return m, p


# ---------------- floor grid ----------------
rows, daily = [], []
for phi in [1.05, 1.10, 1.15, 1.20, 1.25]:
    per = []
    for day, p_obs, b, a, c in markets(BF):
        Rt = phi * rev(p_obs, a, b)
        t0 = time.perf_counter()
        pf, mu = constrained_fi(a, b, c, Rt)
        t_fi = 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        gs, pg = tune_gamma(p_obs, a, b, c, Rt)
        t_g = 1e3 * (time.perf_counter() - t0)
        m, pu = tune_uniform(a, b, c, Rt)
        pif = prof(pf, a, b, c)
        rec = dict(phi=phi, day=day, mu=mu, gamma_star=gs,
                   t_fi_ms=t_fi, t_gamma_ms=t_g,
                   fi_below_c=int((pf < c).sum()),
                   gamma_feasible=gs is not None,
                   unif_feasible=m is not None, n=len(b))
        if gs is not None:
            rec.update(gap_gamma=100 * (pif - prof(pg, a, b, c)) / pif,
                       gamma_below_c=int((pg < c).sum()),
                       min_eta_g=float((b * (1 - shares(pg, a, b))).min()))
        if m is not None:
            rec.update(gap_unif=100 * (pif - prof(pu, a, b, c)) / pif, m_star=m)
        if phi == 1.15:
            rec.update(pi_fi=pif,
                       pi_g=prof(pg, a, b, c) if gs is not None else np.nan,
                       pi_u=prof(pu, a, b, c) if m is not None else np.nan,
                       R_fi=rev(pf, a, b) / Rt,
                       R_g=rev(pg, a, b) / Rt if gs is not None else np.nan,
                       R_u=rev(pu, a, b) / Rt if m is not None else np.nan,
                       p_unc=prof(scalar_fi(a, b, c), a, b, c),
                       R_unc=rev(scalar_fi(a, b, c), a, b) / Rt)
            pl, _ = g_it(p_obs, a, b, c, 0.0)
            rec.update(p_lerner=prof(pl, a, b, c), R_lerner=rev(pl, a, b) / Rt)
            daily.append(rec)
        per.append(rec)
    d = pd.DataFrame(per)
    feas = d[d.gamma_feasible & d.unif_feasible]
    rows.append(dict(
        phi=phi, gamma_feasible=int(d.gamma_feasible.sum()),
        unif_feasible=int(d.unif_feasible.sum()),
        median_gamma_star=feas.gamma_star.median(),
        median_gap_gamma=feas.gap_gamma.median(),
        mean_gap_gamma=feas.gap_gamma.mean(),
        median_gap_unif=feas.gap_unif.median(),
        mean_gap_unif=feas.gap_unif.mean(),
        gamma_wins=int((feas.gap_gamma < feas.gap_unif).sum()),
        mean_adv_pp=(feas.gap_unif - feas.gap_gamma).mean(),
        median_adv_pp=(feas.gap_unif - feas.gap_gamma).median(),
        fi_below_c_total=int(d.fi_below_c.sum()),
        gamma_below_c_total=int(d.gamma_below_c.fillna(0).sum()),
    ))
    print(f"phi={phi}: done")

floor_sum = pd.DataFrame(rows)
floor_sum.to_csv(OUT / "round7_floor_summary.csv", index=False)
dd = pd.DataFrame(daily)
dd.to_csv(OUT / "round7_constrained_daily.csv", index=False)

h = dd  # phi = 1.15
print("\n===== phi=1.15 headline (unclamped gamma) =====")
print(f"gamma gap: median {h.gap_gamma.median():.2f}%  mean {h.gap_gamma.mean():.2f}%")
print(f"unif  gap: median {h.gap_unif.median():.2f}%  mean {h.gap_unif.mean():.2f}%")
print(f"advantage: mean {(h.gap_unif - h.gap_gamma).mean():.1f} pp  wins {(h.gap_gamma < h.gap_unif).sum()}/31")
print(f"gamma_star: [{h.gamma_star.min():.2f},{h.gamma_star.max():.2f}] median {h.gamma_star.median():.2f}")
print(f"mu_star: [{h.mu.min():.2f},{h.mu.max():.2f}] median {h.mu.median():.2f}")
print(f"below-cost pairs: FI {h.fi_below_c.sum()} | gamma {int(h.gamma_below_c.sum())} of {h.n.sum()}")
print(f"min eta at gamma solutions: {h.min_eta_g.min():.2f}")
print(f"wall clock ms: gamma median {h.t_gamma_ms.median():.1f} | FI median {h.t_fi_ms.median():.1f}")
print("\n===== floor grid =====")
print(floor_sum.round(2).to_string(index=False))

# ---------------- beta-floor sensitivity ----------------
bs = []
for bf in [1.05, 1.10, 1.20, 1.30, 1.50]:
    unc, con = [], []
    for day, p_obs, b, a, c in markets(bf):
        pf_u = scalar_fi(a, b, c)
        pl, _ = g_it(p_obs, a, b, c, 0.0)
        unc.append(100 * (prof(pf_u, a, b, c) - prof(pl, a, b, c)) / prof(pf_u, a, b, c))
        Rt = 1.15 * rev(p_obs, a, b)
        pf, mu = constrained_fi(a, b, c, Rt)
        gs, pg = tune_gamma(p_obs, a, b, c, Rt)
        if gs is not None:
            con.append(100 * (prof(pf, a, b, c) - prof(pg, a, b, c)) / prof(pf, a, b, c))
    nfl = int((post.beta_posterior_mean < bf).sum())
    bs.append(dict(beta_floor=bf, n_floored=nfl,
                   median_gap_unconstrained=np.median(unc),
                   median_gap_constrained=np.median(con)))
    print(f"beta floor {bf}: floored {nfl}/415, unc {np.median(unc):.2f}%, con {np.median(con):.2f}%")
bsd = pd.DataFrame(bs)
bsd.to_csv(OUT / "round7_beta_floor_sens.csv", index=False)

# ---------------- figures (phi = 1.15) ----------------
fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.scatter(h.R_unc, h.p_unc / 1e5, marker="x", c="grey", label="Unconstrained FI")
ax.scatter(h.R_lerner, h.p_lerner / 1e5, marker="s", facecolors="none",
           edgecolors="tab:green", label=r"$\gamma$ at $\gamma^\star=0$")
ax.scatter(h.R_fi, h.pi_fi / 1e5, marker="o", facecolors="none",
           edgecolors="tab:red", label="Constrained FI")
ax.scatter(h.R_u, h.pi_u / 1e5, marker="^", c="tab:orange", alpha=0.85,
           label="Uniform tuned")
ax.scatter(h.R_g, h.pi_g / 1e5, marker="D", c="tab:blue", alpha=0.85,
           label=r"$\gamma$-tuned")
ax.axvline(1.0, color="k", linestyle="--", lw=1, label=r"$R/R_{\rm target}=1$")
ax.set_xlabel(r"$R(\mathbf{p})\,/\,R_{\rm target}$ (per day)")
ax.set_ylabel(r"Profit $\Pi$ ($\times 10^5$)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Profit vs. revenue relative to target (31 days)", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "jd_gmv_profit_vs_GMV.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(h.gamma_star, bins=20, color="tab:blue", edgecolor="black", alpha=0.85)
ax.axvline(0.0, color="red", linestyle="--", label=r"$\gamma^\star=0$ (Lerner)")
ax.axvline(h.gamma_star.median(), color="k", linestyle=":",
           label=fr"median $\hat\gamma^\star$ = {h.gamma_star.median():.3f}")
ax.set_xlabel(r"tuned $\hat\gamma^\star$")
ax.set_ylabel("days")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "jd_gmv_gamma_star_distribution.png", dpi=150)
plt.close()
print("\nfigures written")
