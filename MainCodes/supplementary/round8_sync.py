#!/usr/bin/env python3
"""Round-8 sync: put every constrained exercise on the same feasible set.

(A) Uniform markup is re-tuned over m < 1 without an implicit cost floor
    (m < 0 gives below-cost prices), matching the unclamped gamma rule.
(B) Appendix M.9 (tuning-error sweep, outcome-based feedback) is re-run on
    the days that are feasible for the unclamped gamma family, and both
    figures are regenerated.

Outputs (reference_output_results/): round8_floor_summary.csv,
round8_constrained_daily.csv, round8_gamma_sweep.csv, round8_feedback.csv
Figures: jd_gmv_profit_vs_GMV.png, jd_gmv_gamma_star_distribution.png,
jd_gmv_floor_sensitivity.png, jd_gamma_robustness.png, jd_gamma_feedback.png
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

G = Path(__file__).resolve().parents[2]
OUT, FIG = G / "reference_output_results", G / "figures"
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

MKTS = []
for day, g in agg.groupby("day"):
    p_obs = g["price"].to_numpy()
    b = np.maximum(np.array([raw_map[x] for x in g["sku_ID"]]), BF)
    s_obs = g["share"].to_numpy()
    a = (s_obs / (1.0 - s_obs.sum())) * np.power(p_obs, b)
    MKTS.append((int(day), p_obs, b, a, MR * p_obs))


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
    return (cc + brentq(gA, 0.0, hi, xtol=1e-12)) / (1.0 - 1.0 / b)


def constrained_fi(a, b, c, Rt):
    p = scalar_fi(a, b, c)
    if rev(p, a, b) >= Rt:
        return p, 0.0
    mu = brentq(lambda m: rev(scalar_fi(a, b, c, ceff=c / (1.0 + m)), a, b) - Rt,
                1e-8, 100.0, xtol=1e-8)
    return scalar_fi(a, b, c, ceff=c / (1.0 + mu)), mu


def g_it(p0, a, b, c, gs, sweeps=400, tol=1e-8):
    for omega in (1.0, 0.5):
        p = p0.copy()
        for k in range(sweeps):
            s = shares(p, a, b)
            eta = np.maximum(b * (1.0 - s), 1.01)
            pn = (1.0 - omega) * p + omega * c / ((1.0 - gs) * (1.0 - 1.0 / eta))
            if np.max(np.abs(pn - p) / p) < tol:
                return pn, True, k + 1
            p = pn
    return p, False, sweeps


def _rightmost_crossing(r_of_x, x_hi, Rt, step, x_min, iters=60):
    """Largest x <= x_hi with r(x) >= Rt; None if the hump peaks below Rt."""
    x_prev, r_prev = x_hi, r_of_x(x_hi)
    if r_prev >= Rt:
        return x_hi
    declines, x = 0, x_hi - step
    while x >= x_min:
        r = r_of_x(x)
        if r >= Rt:
            lo, hi = x, x_prev
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if r_of_x(mid) < Rt:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        declines = declines + 1 if r < r_prev else 0
        if declines >= 2:
            return None
        x_prev, r_prev, x = x, r, x - step
    return None


def tune_gamma(p0, a, b, c, Rt):
    gs = _rightmost_crossing(lambda g: rev(g_it(p0, a, b, c, g)[0], a, b),
                             0.0, Rt, 0.05, -8.0)
    return (None, None) if gs is None else (gs, g_it(p0, a, b, c, gs)[0])


def tune_uniform(a, b, c, Rt):
    """No cost floor: m may be negative, giving p_i = c_i/(1-m) < c_i."""
    r_of_m = lambda m: rev(c / (1.0 - m), a, b)
    if r_of_m(0.999) >= Rt:
        m = 0.999
    else:
        m = _rightmost_crossing(r_of_m, 0.999, Rt, 0.02, -8.0)
    return (None, None) if m is None else (m, c / (1.0 - m))


# ---------------- (A) floor grid with unclamped uniform ----------------
rows, daily = [], []
for phi in [1.05, 1.10, 1.15, 1.20, 1.25]:
    per = []
    for day, p_obs, b, a, c in MKTS:
        Rt = phi * rev(p_obs, a, b)
        t0 = time.perf_counter(); pf, mu = constrained_fi(a, b, c, Rt)
        t_fi = 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter(); gs, pg = tune_gamma(p_obs, a, b, c, Rt)
        t_g = 1e3 * (time.perf_counter() - t0)
        m, pu = tune_uniform(a, b, c, Rt)
        pif = prof(pf, a, b, c)
        rec = dict(phi=phi, day=day, mu=mu, gamma_star=gs, m_star=m,
                   t_fi_ms=t_fi, t_gamma_ms=t_g, n=len(b),
                   fi_below_c=int((pf < c).sum()),
                   gamma_feasible=gs is not None, unif_feasible=m is not None)
        if gs is not None:
            rec.update(gap_gamma=100 * (pif - prof(pg, a, b, c)) / pif,
                       gamma_below_c=int((pg < c).sum()),
                       min_eta_g=float((b * (1 - shares(pg, a, b))).min()))
        if m is not None:
            rec.update(gap_unif=100 * (pif - prof(pu, a, b, c)) / pif,
                       unif_below_c=int((pu < c).sum()))
        if phi == 1.15:
            pl = g_it(p_obs, a, b, c, 0.0)[0]
            pu_ = scalar_fi(a, b, c)
            rec.update(pi_fi=pif, R_fi=rev(pf, a, b) / Rt,
                       pi_g=prof(pg, a, b, c) if gs is not None else np.nan,
                       R_g=rev(pg, a, b) / Rt if gs is not None else np.nan,
                       pi_u=prof(pu, a, b, c) if m is not None else np.nan,
                       R_u=rev(pu, a, b) / Rt if m is not None else np.nan,
                       p_unc=prof(pu_, a, b, c), R_unc=rev(pu_, a, b) / Rt,
                       p_lerner=prof(pl, a, b, c), R_lerner=rev(pl, a, b) / Rt)
            daily.append(rec)
        per.append(rec)
    d = pd.DataFrame(per)
    f = d[d.gamma_feasible & d.unif_feasible]
    rows.append(dict(phi=phi, gamma_feasible=int(d.gamma_feasible.sum()),
                     unif_feasible=int(d.unif_feasible.sum()), n_joint=len(f),
                     median_gamma_star=f.gamma_star.median(),
                     median_m_star=f.m_star.median(),
                     median_gap_gamma=f.gap_gamma.median(),
                     mean_gap_gamma=f.gap_gamma.mean(),
                     median_gap_unif=f.gap_unif.median(),
                     mean_gap_unif=f.gap_unif.mean(),
                     gamma_wins=int((f.gap_gamma < f.gap_unif).sum()),
                     median_adv_pp=(f.gap_unif - f.gap_gamma).median(),
                     mean_adv_pp=(f.gap_unif - f.gap_gamma).mean(),
                     unif_below_c=int(d.unif_below_c.fillna(0).sum())))
    print(f"phi={phi} done")

fs = pd.DataFrame(rows); fs.to_csv(OUT / "round8_floor_summary.csv", index=False)
h = pd.DataFrame(daily); h.to_csv(OUT / "round8_constrained_daily.csv", index=False)
hj = h[h.gamma_feasible & h.unif_feasible]
print("\n===== phi=1.15 =====")
print(f"gamma: median {hj.gap_gamma.median():.2f}% mean {hj.gap_gamma.mean():.2f}% "
      f"| feasible {int(h.gamma_feasible.sum())}/31")
print(f"unif : median {hj.gap_unif.median():.2f}% mean {hj.gap_unif.mean():.2f}% "
      f"| feasible {int(h.unif_feasible.sum())}/31 | m* in "
      f"[{h.m_star.min():.3f},{h.m_star.max():.3f}]")
print(f"wins {int((hj.gap_gamma < hj.gap_unif).sum())}/{len(hj)} "
      f"| mean adv {(hj.gap_unif - hj.gap_gamma).mean():.1f} pp")
print(f"gamma* [{h.gamma_star.min():.2f},{h.gamma_star.max():.2f}] "
      f"median {h.gamma_star.median():.2f}")
print("\n===== floor grid =====")
print(fs.round(2).to_string(index=False))

# figures 1-2 (phi = 1.15)
fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.scatter(h.R_unc, h.p_unc / 1e5, marker="x", c="grey", label="Unconstrained FI")
ax.scatter(h.R_lerner, h.p_lerner / 1e5, marker="s", facecolors="none",
           edgecolors="tab:green", label=r"$\gamma$ at $\gamma^\star=0$")
ax.scatter(h.R_fi, h.pi_fi / 1e5, marker="o", facecolors="none",
           edgecolors="tab:red", label="Constrained FI")
ax.scatter(h.R_u, h.pi_u / 1e5, marker="^", c="tab:orange", alpha=.85,
           label="Uniform tuned")
ax.scatter(h.R_g, h.pi_g / 1e5, marker="D", c="tab:blue", alpha=.85,
           label=r"$\gamma$-tuned")
ax.axvline(1.0, color="k", ls="--", lw=1, label=r"$R/R_{\rm target}=1$")
ax.set_xlabel(r"$R(\mathbf{p})\,/\,R_{\rm target}$ (per day)")
ax.set_ylabel(r"Profit $\Pi$ ($\times 10^5$)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Profit vs. revenue relative to target", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "jd_gmv_profit_vs_GMV.png", dpi=150); plt.close()

gsv = h.gamma_star.dropna()
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(gsv, bins=20, color="tab:blue", edgecolor="black", alpha=.85)
ax.axvline(0.0, color="red", ls="--", label=r"$\gamma^\star=0$ (Lerner)")
ax.axvline(gsv.median(), color="k", ls=":",
           label=fr"median $\hat\gamma^\star$ = {gsv.median():.3f}")
ax.set_xlabel(r"tuned $\hat\gamma^\star$ ({} feasible days)".format(len(gsv)))
ax.set_ylabel("days"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "jd_gmv_gamma_star_distribution.png", dpi=150); plt.close()

# floor-sensitivity figure
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(fs.phi, fs.median_gap_gamma, "o-", color="tab:blue", label=r"$\gamma$-tuned")
a1.plot(fs.phi, fs.median_gap_unif, "^-", color="tab:orange", label="uniform-tuned")
a1.set_xlabel(r"floor multiplier $\phi$")
a1.set_ylabel("median profit gap to constrained FI (%)")
a1.legend(); a1.set_title("Median gap by floor level", fontsize=10)
q = h.groupby("phi") if False else None
gq = (pd.read_csv(OUT / "round8_floor_summary.csv"))
a2.plot(fs.phi, fs.median_gamma_star, "o-", color="tab:blue",
        label=r"median $\hat\gamma^\star$")
a2.set_xlabel(r"floor multiplier $\phi$"); a2.set_ylabel(r"tuned $\hat\gamma^\star$")
a2.legend(); a2.set_title("Tuned scalar by floor level", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "jd_gmv_floor_sensitivity.png", dpi=150); plt.close()

# ---------------- (B) M.9 on the feasible days ----------------
feas = [(day, p_obs, b, a, c) for (day, p_obs, b, a, c) in MKTS
        if bool(h.set_index("day").loc[day, "gamma_feasible"])]
print(f"\nM.9 on {len(feas)} feasible days")

sweep = []
for day, p_obs, b, a, c in feas:
    Rt = 1.15 * rev(p_obs, a, b)
    gs, pg = tune_gamma(p_obs, a, b, c, Rt)
    _, mu = constrained_fi(a, b, c, Rt)
    lag = lambda p: prof(p, a, b, c) + mu * (rev(p, a, b) - Rt)
    pf, _ = constrained_fi(a, b, c, Rt)
    base = lag(pf)          # Lagrangian at the FI optimum, the correct benchmark
    for dg in [0.0, .02, -.02, .05, -.05, .10, -.10, .20, -.20]:
        p, conv, it = g_it(p_obs, a, b, c, gs + dg)
        sweep.append(dict(day=day, delta_gamma=dg, converged=conv, iters=it,
                          gmv=100 * (rev(p, a, b) - Rt) / Rt,
                          lag_loss=100 * (base - lag(p)) / abs(base)))
sw = pd.DataFrame(sweep); sw.to_csv(OUT / "round8_gamma_sweep.csv", index=False)
med = sw.groupby("delta_gamma").agg(gmv=("gmv", "median"),
                                    lag=("lag_loss", "median"),
                                    it=("iters", "median")).reset_index()
kl = np.polyfit(med.delta_gamma, med.gmv, 1)
nz = med[med.delta_gamma != 0]
base_med = float(med.loc[med.delta_gamma == 0, "lag"].iloc[0])
A_ = np.column_stack([np.abs(nz.delta_gamma), nz.delta_gamma ** 2])
coef, *_ = np.linalg.lstsq(A_, nz.lag - base_med, rcond=None)
print(f"convergence: {int(sw.converged.sum())}/{len(sw)} pairs, median {med.it.median():.0f} sweeps")
print(f"GMV slope {kl[0]:.1f}%/unit | loss = {base_med:.2f}% + "
      f"{coef[0]:.2f}|dg| + {coef[1]:.1f}dg^2")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(sw.delta_gamma, sw.gmv, s=12, alpha=.35, color="tab:blue")
ax.plot(med.delta_gamma, med.gmv, "o-", color="k", label="median")
xs = np.linspace(-.22, .22, 50)
ax.plot(xs, np.polyval(kl, xs), "k--", alpha=.6, label=f"linear fit (slope {kl[0]:.1f})")
ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
ax.set_xlabel(r"$\Delta\gamma = \gamma - \hat\gamma^\star$")
ax.set_ylabel("GMV deviation from target (%)")
ax.set_title(f"GMV miss is first order in $\\Delta\\gamma$ ({len(feas)} days)", fontsize=10)
ax.legend(fontsize=8)
ax = axes[1]
ax.scatter(sw.delta_gamma, sw.lag_loss - base_med, s=12, alpha=.35, color="tab:red")
ax.plot(med.delta_gamma, med.lag - base_med, "o-", color="k", label="median")
ax.plot(xs, coef[0] * np.abs(xs) + coef[1] * xs ** 2, "k--", alpha=.6,
        label=fr"${coef[0]:.2f}|\Delta\gamma| + {coef[1]:.1f}\Delta\gamma^2$")
ax.axvline(0, color="grey", lw=.8)
ax.set_xlabel(r"$\Delta\gamma = \gamma - \hat\gamma^\star$")
ax.set_ylabel("excess Lagrangian loss over baseline (pp)")
ax.set_title("Excess loss above the $\\Delta\\gamma=0$ baseline", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "jd_gamma_robustness.png", dpi=150); plt.close()

fb = []
for tag, noise in [("no noise", 0.0), ("2% GMV noise", 0.02)]:
    for day, p_obs, b, a, c in feas:
        Rt = 1.15 * rev(p_obs, a, b)
        gs_true, _ = tune_gamma(p_obs, a, b, c, Rt)
        rng = np.random.default_rng(2026 + day)
        lo, hi, g = -3.0, 0.0, 0.0
        for rd in range(1, 13):
            p, _, _ = g_it(p_obs, a, b, c, g)
            R = rev(p, a, b) * (1.0 + (rng.normal(0, noise) if noise else 0.0))
            fb.append(dict(tag=tag, day=day, round=rd, gamma=g,
                           err=abs(g - gs_true), ratio=R / Rt))
            if R < Rt:
                hi = g
            else:
                lo = g
            g = 0.5 * (lo + hi)
fbd = pd.DataFrame(fb); fbd.to_csv(OUT / "round8_feedback.csv", index=False)
med_fb = fbd.groupby(["tag", "round"]).agg(err=("err", "median"),
                                           ratio=("ratio", "median")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for tag, col in [("no noise", "tab:blue"), ("2% GMV noise", "tab:orange")]:
    t = med_fb[med_fb.tag == tag]
    axes[0].semilogy(t["round"], t["err"], "o-", color=col, label=tag)
    axes[1].plot(t["round"], t["ratio"], "o-", color=col, label=tag)
axes[0].set_xlabel("feedback round"); axes[0].set_ylabel(r"median $|\gamma_t-\hat\gamma^\star|$")
axes[0].set_title(f"Outcome-based calibration ({len(feas)} feasible days)", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].axhline(1.0, color="k", ls="--", lw=1, label="target")
axes[1].set_xlabel("feedback round"); axes[1].set_ylabel(r"realized $R_t/R_{\rm target}$")
axes[1].set_title("Realized GMV reaches the target", fontsize=10)
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "jd_gamma_feedback.png", dpi=150); plt.close()

for tag in ["no noise", "2% GMV noise"]:
    t = med_fb[med_fb.tag == tag]
    r12 = t[t["round"] == 12].iloc[0]
    print(f"{tag}: round-12 |gamma err| = {r12.err:.4f}, GMV ratio = {r12.ratio:.4f}")
r1 = med_fb[(med_fb.tag == 'no noise') & (med_fb['round'] == 1)].iloc[0]
print(f"round 1 (gamma=0) attains {100*r1.ratio:.0f}% of target")
print("figures written")
