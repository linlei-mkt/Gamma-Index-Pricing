"""Round-3 computations:
A) Constrained scalar-FI on JD: outer mu-brentq + inner scalar-A; wall clock.
B) Scalar-FI wall clock at synthetic n in {500,2000,8000,50000}.
C) Noisy own-response oracle: gamma-iteration with per-sweep noise on eta.
D) Posterior propagation (normal approx) into the constrained 5.57%.
"""
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import brentq
warnings.simplefilter("ignore")
G = Path("/sessions/epic-brave-gauss/mnt/Gamma/repl_check/gamma-equalization-replication")
rng = np.random.default_rng(2026)
N_TOP, M_MULT, MR, BF = 500, 3.0, 0.70, 1.2

# ---------- shared JD pipeline ----------
orders = pd.read_csv(G/"JD_MSOM/JD_order_data.csv", usecols=["sku_ID","order_date","quantity","final_unit_price"], dtype={"sku_ID":"string"}, parse_dates=["order_date"])
sku = pd.read_csv(G/"JD_MSOM/JD_sku_data.csv", usecols=["sku_ID","type"], dtype={"sku_ID":"string"})
top = orders.groupby("sku_ID").size().nlargest(N_TOP).index
o = orders[orders["sku_ID"].isin(top)].copy(); o["day"]=o["order_date"].dt.day
agg = o.assign(rev=o["final_unit_price"]*o["quantity"]).groupby(["day","sku_ID"],as_index=False).agg(qty=("quantity","sum"),rev=("rev","sum"))
agg["price"]=agg["rev"]/agg["qty"]; agg=agg.merge(sku,on="sku_ID",how="left")
agg=agg[(agg["qty"]>0)&(agg["price"]>0)&agg["type"].notna()].copy()
M=M_MULT*agg.groupby("day")["qty"].sum().max(); agg["share"]=agg["qty"]/M
s0=(1.0-agg.groupby("day")["qty"].sum()/M).rename("s0"); agg=agg.merge(s0,on="day")
post=pd.read_csv(G/"MainCodes/jd_hb_posterior_summary.csv")
mu_map=dict(zip(post["sku_ID"], np.maximum(post["beta_posterior_mean"],BF)))
sd_map=dict(zip(post["sku_ID"], post["beta_posterior_sd"]))
agg=agg[agg["sku_ID"].isin(mu_map)].copy()

def shares(p,a,b): A=a*np.power(p,-b); return A/(1.0+A.sum())
def prof(p,c,a,b): return float(np.sum((p-c)*shares(p,a,b))*M)
def rev(p,a,b): return float(np.sum(p*shares(p,a,b))*M)
def scalar_fi(c,a,b,ceff=None):
    cc = c if ceff is None else ceff
    def g(A):
        p=(cc+A)/(1.0-1.0/b); s=shares(p,a,b); return float(s@(p-cc))-A
    hi=1.0
    while g(hi)>0 and hi<1e7: hi*=4.0
    A=brentq(g,0.0,hi,xtol=1e-12,maxiter=300); return (cc+A)/(1.0-1.0/b)
def g_it(p0,c,a,b,gs=0.0,noise=0.0,rloc=None):
    p=p0.copy()
    for k in range(2000):
        s=shares(p,a,b); eta=np.maximum(b*(1.0-s),1.01)
        if noise>0: eta=np.maximum(eta*(1.0+rloc.normal(0,noise,len(eta))),1.01)
        den=np.maximum(1.0-gs-(1.0-gs)/eta,0.01)
        pn=np.maximum(c/den,c*1.0001)
        if np.max(np.abs(pn-p))<1e-8 and noise==0: return pn,k+1
        if noise>0 and k>=60: return pn,k+1
        p=pn
    return p,k+1
def tune_gamma(p0,c,a,b,Rt):
    def res(gs): p,_=g_it(p0,c,a,b,gs); return rev(p,a,b)-Rt
    return brentq(res,-5.0,0.99,xtol=1e-6,maxiter=60)
def constrained_scalar_fi(c,a,b,Rt):
    """outer mu search + inner scalar-A (Prop 5). Returns (p, wallclock)."""
    t0=time.perf_counter()
    p_un=scalar_fi(c,a,b)
    if rev(p_un,a,b)>=Rt: return p_un, time.perf_counter()-t0
    def res(mu):
        p=scalar_fi(c,a,b,ceff=c/(1.0+mu)); return rev(p,a,b)-Rt
    mu=brentq(res,1e-6,100.0,xtol=1e-6,maxiter=60)
    p=scalar_fi(c,a,b,ceff=c/(1.0+mu))
    return p, time.perf_counter()-t0

mkts=[]
for day in sorted(agg["day"].unique()):
    m=agg[agg["day"]==day]
    if len(m)<50: continue
    p_obs=m["price"].to_numpy(); s_obs=m["share"].to_numpy()
    b=m["sku_ID"].map(mu_map).to_numpy(); bsd=m["sku_ID"].map(sd_map).to_numpy()
    a=(s_obs/float(m["s0"].iloc[0]))*np.power(p_obs,b); c=MR*p_obs
    mkts.append((int(day),p_obs,s_obs,b,bsd,a,c))

# ---- A) constrained scalar-FI wall clock + gamma-tuned wall clock ----
tA_fi=[]; tA_g=[]
for day,p_obs,s_obs,b,bsd,a,c in mkts:
    Rt=1.15*rev(p_obs,a,b)
    t0=time.perf_counter(); gs=tune_gamma(p_obs,c,a,b,Rt); pg,_=g_it(p_obs,c,a,b,gs); tA_g.append(time.perf_counter()-t0)
    pf,tf=constrained_scalar_fi(c,a,b,Rt); tA_fi.append(tf)
print("A) constrained wall clock median ms: gamma-tuned %.2f | scalar-FI-tuned %.2f" % (1000*np.median(tA_g),1000*np.median(tA_fi)))

# ---- B) scalar-FI timing at synthetic scale ----
print("B) scalar-FI at synthetic n:")
for n in [500,2000,8000,50000]:
    bb=rng.uniform(1.5,3.5,n); pp=np.exp(rng.normal(3,0.5,n)); ss=rng.dirichlet(np.ones(n))*0.25
    aa=(ss/0.75)*np.power(pp,bb); cc=0.7*pp
    t0=time.perf_counter(); _=scalar_fi(cc,aa,bb); dt=time.perf_counter()-t0
    print("   n=%6d: %.2f ms" % (n,1000*dt))

# ---- C) noisy own-response oracle ----
print("C) noisy own-response oracle (unconstrained, median gap %% to FI):")
base={}
for day,p_obs,s_obs,b,bsd,a,c in mkts:
    base[day]=prof(scalar_fi(c,a,b),c,a,b)
for tau in [0.0,0.05,0.10,0.20]:
    gaps=[]
    for day,p_obs,s_obs,b,bsd,a,c in mkts:
        rl=np.random.default_rng(1000+day)
        p,_=g_it(p_obs,c,a,b,0.0,noise=tau,rloc=rl)
        gaps.append(100*(base[day]-prof(p,c,a,b))/base[day])
    print("   tau=%3.0f%%: median gap %.2f%%" % (100*tau,np.median(gaps)))

# ---- D) posterior propagation into constrained gap (normal approx, K=100) ----
print("D) constrained-gap posterior propagation (normal-approx draws, K=100):")
K=100; med_by_day=[]
for day,p_obs,s_obs,b,bsd,a,c in mkts:
    Rt_base=None; draws=[]
    for k in range(K):
        rk=np.random.default_rng(day*1000+k)
        bk=np.maximum(b+rk.normal(0,1,len(b))*np.nan_to_num(bsd,nan=0.0),BF)
        ak=(s_obs/ (1.0-s_obs.sum()) )*np.power(p_obs,bk)*( (1.0-s_obs.sum()) / (1.0-s_obs.sum()) )  # recalibrate alpha with bk
        ak=(s_obs/(1.0-float(s_obs.sum())))*0+ (s_obs/ float(1.0-s_obs.sum()) )*np.power(p_obs,bk)
        Rt=1.15*rev(p_obs,ak,bk)
        try:
            gs=tune_gamma(p_obs,c,ak,bk,Rt); pg,_=g_it(p_obs,c,ak,bk,gs)
            pf,_=constrained_scalar_fi(c,ak,bk,Rt)
            pif=prof(pf,c,ak,bk)
            if pif>0: draws.append(100*(pif-prof(pg,c,ak,bk))/pif)
        except Exception: pass
    if draws: med_by_day.append((day,np.median(draws),np.percentile(draws,5),np.percentile(draws,95)))
mm=pd.DataFrame(med_by_day,columns=["day","med","lo","hi"])
print("   median across days of per-day median: %.2f%% ; median CI width %.2f pp" % (mm["med"].median(), (mm["hi"]-mm["lo"]).median()))
mm.to_csv("/sessions/epic-brave-gauss/mnt/outputs/constrained_posterior_prop.csv",index=False)
