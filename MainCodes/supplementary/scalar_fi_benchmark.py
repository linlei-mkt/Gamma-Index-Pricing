"""Decisive benchmark: exact FI via the MCI scalar reduction.
Under MCI, FOC => p_i = (c_i + A)/(1 - 1/beta_i) with ONE scalar
A = s(p(A))'(p(A)-c). Solve A by 1-D root find (each eval O(n)).
Compare vs gamma-iteration and dense MS on all 31 JD days."""
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq, minimize_scalar
warnings.simplefilter("ignore")
G = Path("/sessions/epic-brave-gauss/mnt/Gamma/repl_check/gamma-equalization-replication")
DATA = G/"JD_MSOM"; N_TOP, M_MULT, MR, BF, TOL, MAXIT = 500, 3.0, 0.70, 1.2, 1e-8, 2000

orders = pd.read_csv(DATA/"JD_order_data.csv", usecols=["sku_ID","order_date","quantity","final_unit_price"], dtype={"sku_ID":"string"}, parse_dates=["order_date"])
sku = pd.read_csv(DATA/"JD_sku_data.csv", usecols=["sku_ID","type","brand_ID"], dtype={"sku_ID":"string","brand_ID":"string"})
top = orders.groupby("sku_ID").size().nlargest(N_TOP).index
o = orders[orders["sku_ID"].isin(top)].copy(); o["day"]=o["order_date"].dt.day
agg = o.assign(rev=o["final_unit_price"]*o["quantity"]).groupby(["day","sku_ID"],as_index=False).agg(qty=("quantity","sum"),rev=("rev","sum"))
agg["price"]=agg["rev"]/agg["qty"]
agg = agg.merge(sku[["sku_ID","type","brand_ID"]],on="sku_ID",how="left")
agg = agg[(agg["qty"]>0)&(agg["price"]>0)&agg["type"].notna()].copy()
M = M_MULT*agg.groupby("day")["qty"].sum().max(); agg["share"]=agg["qty"]/M
s0=(1.0-agg.groupby("day")["qty"].sum()/M).rename("s0"); agg=agg.merge(s0,on="day")
post = pd.read_csv(G/"MainCodes/jd_hb_posterior_summary.csv")
s2b = dict(zip(post["sku_ID"], np.maximum(post["beta_posterior_mean"], BF)))
agg = agg[agg["sku_ID"].isin(s2b)].copy()

def shares(p,a,b): A=a*np.power(p,-b); return A/(1.0+A.sum())
def prof(p,c,a,b): return float(np.sum((p-c)*shares(p,a,b))*M)
def g_it(p0,c,a,b):
    p=p0.copy(); t0=time.perf_counter()
    for k in range(MAXIT):
        s=shares(p,a,b); eta=np.maximum(b*(1.0-s),1.01)
        pn=np.maximum(c/(1.0-1.0/eta), c*1.0001)
        if np.max(np.abs(pn-p))<TOL: p=pn; break
        p=pn
    return p, time.perf_counter()-t0, k+1
def ms_dense(p0,c,a,b):
    p=p0.copy(); t0=time.perf_counter()
    for k in range(MAXIT):
        s=shares(p,a,b); u=b*s/p; Om=np.outer(u,s); np.fill_diagonal(Om,-u*(1.0-s))
        d=np.diag(Om).copy(); Ga=Om-np.diag(d)
        pn=np.maximum(c-(s+Ga@(p-c))/d, c*1.0001)
        if np.max(np.abs(pn-p))<TOL: p=pn; break
        p=pn
    return p, time.perf_counter()-t0, k+1
def scalar_fi(c,a,b):
    """exact FI: p(A) = (c+A)/(1-1/b); solve g(A)=s'(p-c)-A=0."""
    t0=time.perf_counter(); n_ev=[0]
    def g(A):
        n_ev[0]+=1
        p=(c+A)/(1.0-1.0/b); s=shares(p,a,b)
        return float(s@(p-c)) - A
    hi=1.0
    while g(hi)>0 and hi<1e7: hi*=4.0
    A=brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-14, maxiter=200)
    p=(c+A)/(1.0-1.0/b)
    return p, time.perf_counter()-t0, n_ev[0]

rows=[]
for day in sorted(agg["day"].unique()):
    mkt=agg[agg["day"]==day]
    if len(mkt)<50: continue
    p_obs=mkt["price"].to_numpy(); s_obs=mkt["share"].to_numpy(); b=mkt["sku_ID"].map(s2b).to_numpy()
    a=(s_obs/float(mkt["s0"].iloc[0]))*np.power(p_obs,b); c=MR*p_obs
    pg,tg,kg = g_it(p_obs,c,a,b)
    pm,tm,km = ms_dense(p_obs,c,a,b)
    pf,tf,kf = scalar_fi(c,a,b)
    pi_f=prof(pf,c,a,b)
    rows.append(dict(day=int(day),
        t_gamma=tg, t_ms=tm, t_scalarfi=tf, ev_scalarfi=kf,
        gap_gamma=(pi_f-prof(pg,c,a,b))/pi_f,
        gap_ms=(pi_f-prof(pm,c,a,b))/pi_f,
        agree_ms_vs_fi=float(np.max(np.abs(pm-pf)/pf))))
d=pd.DataFrame(rows); m=d.median(numeric_only=True)
print("days:",len(d))
print("scalar-FI vs dense-MS price agreement (max rel diff, median): %.2e" % m.agree_ms_vs_fi)
print("gap to scalar-FI: gamma %.2f%%  dense-MS %.4f%%" % (100*m.gap_gamma, 100*m.gap_ms))
print("wall-clock median ms: gamma %.3f | scalar-FI %.3f | dense-MS %.2f" % (1000*m.t_gamma, 1000*m.t_scalarfi, 1000*m.t_ms))
print("scalar-FI function evals (median): %d" % m.ev_scalarfi)
print("ratios: denseMS/gamma = %.0fx ; scalarFI/gamma = %.1fx ; denseMS/scalarFI = %.0fx" % (m.t_ms/m.t_gamma, m.t_scalarfi/m.t_gamma, m.t_ms/m.t_scalarfi))
d.to_csv("/sessions/epic-brave-gauss/mnt/outputs/scalar_fi_benchmark.csv", index=False)
