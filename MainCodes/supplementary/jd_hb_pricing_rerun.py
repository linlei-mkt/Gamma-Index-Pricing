"""Pricing-only rerun of the HB four-method comparison (Table 3).
Uses cached posterior (jd_hb_posterior_summary.csv); mirrors
jd_hierarchical_bayes.py solvers verbatim. Reports SIGNED gaps
(no clipping) + full iteration/wall-clock columns + ebar at
observed / FI / gamma prices."""
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, root
warnings.simplefilter("ignore")

G = Path("/sessions/epic-brave-gauss/mnt/Gamma")
N_TOP_SKU, M_MULT, MARGIN_RATIO, BETA_FLOOR, TOL, MAX_ITER = 500, 3.0, 0.70, 1.2, 1e-8, 2000

orders = pd.read_csv(G/"JD_order_data.csv", usecols=["sku_ID","order_date","quantity","final_unit_price"], dtype={"sku_ID":"string"}, parse_dates=["order_date"])
sku = pd.read_csv(G/"JD_sku_data.csv", usecols=["sku_ID","type","brand_ID"], dtype={"sku_ID":"string","brand_ID":"string"})
top = orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index
o = orders[orders["sku_ID"].isin(top)].copy(); o["day"] = o["order_date"].dt.day
agg = o.assign(rev=o["final_unit_price"]*o["quantity"]).groupby(["day","sku_ID"], as_index=False).agg(qty=("quantity","sum"), rev=("rev","sum"))
agg["price"] = agg["rev"]/agg["qty"]
agg = agg.merge(sku[["sku_ID","type","brand_ID"]], on="sku_ID", how="left")
agg = agg[(agg["qty"]>0)&(agg["price"]>0)&agg["type"].notna()].copy()
M = M_MULT*agg.groupby("day")["qty"].sum().max()
agg["share"] = agg["qty"]/M
s0 = (1.0-agg.groupby("day")["qty"].sum()/M).rename("s0"); agg = agg.merge(s0, on="day")
post = pd.read_csv(G/"jd_hb_posterior_summary.csv")
s2b = dict(zip(post["sku_ID"], np.maximum(post["beta_posterior_mean"], BETA_FLOOR)))
agg = agg[agg["sku_ID"].isin(s2b)].copy()

def mci_shares(p,a,b,M):
    A=a*np.power(p,-b); D=1.0+A.sum(); return A/D, 1.0/D
def cal_alpha(p,s,s0,b): return (s/s0)*np.power(p,b)
def jac(p,s,b):
    u=b*s/p; Om=np.outer(u,s); np.fill_diagonal(Om,-u*(1.0-s)); return Om
def ebar(s):
    S=s.sum(); return float(np.max((S-s)/(1.0-s)))
def prof(p,c,a,b,M):
    s,_=mci_shares(p,a,b,M); return float(np.sum((p-c)*s)*M)
def g_it(p0,c,a,b,M):
    p=p0.copy(); t0=time.perf_counter()
    for k in range(MAX_ITER):
        s,_=mci_shares(p,a,b,M); eta=np.maximum(b*(1.0-s),1.01)
        pn=np.maximum(c/(1.0-1.0/eta), c*1.0001)
        if np.max(np.abs(pn-p))<TOL: p=pn; break
        p=pn
    return p,k+1,time.perf_counter()-t0
def ms_it(p0,c,a,b,M):
    p=p0.copy(); t0=time.perf_counter()
    for k in range(MAX_ITER):
        s,_=mci_shares(p,a,b,M); Om=jac(p,s,b); d=np.diag(Om).copy(); Ga=Om-np.diag(d)
        pn=np.maximum(c-(s+Ga@(p-c))/d, c*1.0001)
        if np.max(np.abs(pn-p))<TOL: p=pn; break
        p=pn
    return p,k+1,time.perf_counter()-t0
def newt(p0,c,a,b,M):
    def F(p):
        pp=np.maximum(p,c*1.0001); s,_=mci_shares(pp,a,b,M); return s+jac(pp,s,b)@(pp-c)
    t0=time.perf_counter(); sol=root(F,p0,method="krylov",tol=1e-10,options={"maxiter":500})
    return np.maximum(sol.x,c*1.0001), (sol.nit or -1), time.perf_counter()-t0, sol.success
def unif(c,a,b,M):
    n_ev=[0]
    def negp(m):
        n_ev[0]+=1
        if not (0.0<m<0.999): return 1e18
        p=c/(1.0-m); s,_=mci_shares(p,a,b,M); return -np.sum((p-c)*s)*M
    t0=time.perf_counter()
    r=minimize_scalar(negp,bounds=(0.001,0.999),method="bounded",options={"xatol":1e-8})
    return c/(1.0-r.x), n_ev[0], time.perf_counter()-t0

rows=[]
for day in sorted(agg["day"].unique()):
    mkt=agg[agg["day"]==day].copy()
    if len(mkt)<50: continue
    p_obs=mkt["price"].to_numpy(); s_obs=mkt["share"].to_numpy(); s0_=float(mkt["s0"].iloc[0])
    b=mkt["sku_ID"].map(s2b).to_numpy(); a=cal_alpha(p_obs,s_obs,s0_,b); c=MARGIN_RATIO*p_obs
    p_g,it_g,t_g=g_it(p_obs,c,a,b,M); p_m,it_m,t_m=ms_it(p_obs,c,a,b,M)
    p_u,it_u,t_u=unif(c,a,b,M); p_bn,it_bn,t_bn,ok=newt(p_m.copy(),c,a,b,M)
    pi=prof(p_bn,c,a,b,M)
    if pi<=0: continue
    sg=lambda p:(pi-prof(p,c,a,b,M))/pi
    sb,_=mci_shares(p_bn,a,b,M); sgm,_=mci_shares(p_g,a,b,M)
    rows.append(dict(day=int(day), n=len(mkt), newton_ok=ok,
        gap_gamma=sg(p_g), gap_MS=sg(p_m), gap_uniform=sg(p_u),
        iter_gamma=it_g, iter_MS=it_m, iter_unif=it_u, iter_newton=it_bn,
        t_gamma=t_g, t_MS=t_m, t_unif=t_u, t_newton=t_bn,
        ebar_obs=ebar(s_obs), ebar_FI=ebar(sb), ebar_gamma=ebar(sgm)))
df=pd.DataFrame(rows)
df.to_csv("/sessions/epic-brave-gauss/mnt/outputs/jd_hb_pricing_full.csv", index=False)
m=df.median(numeric_only=True)
print("days:", len(df), " newton_ok:", int(df.newton_ok.sum()))
print("MEDIANS  gap%%: gamma %.2f  MS %.4f  uniform %.2f" % (100*m.gap_gamma,100*m.gap_MS,100*m.gap_uniform))
print("  signed min gamma gap %.3f%%  (負值天数 %d)" % (100*df.gap_gamma.min(), (df.gap_gamma<0).sum()))
print("ITERS   : gamma %d  MS %d  unif(nfev) %d  newton %d" % (m.iter_gamma,m.iter_MS,m.iter_unif,m.iter_newton))
print("TIME ms : gamma %.3f  MS %.2f  unif %.2f  newton %.1f" % (1000*m.t_gamma,1000*m.t_MS,1000*m.t_unif,1000*m.t_newton))
print("RATIOS  : MS/gamma time %.0fx  iter %.0fx" % (m.t_MS/m.t_gamma, m.iter_MS/m.iter_gamma))
print("EBAR med: obs %.3f  FI %.3f  gamma %.3f ; max FI %.3f" % (m.ebar_obs,m.ebar_FI,m.ebar_gamma,df.ebar_FI.max()))
