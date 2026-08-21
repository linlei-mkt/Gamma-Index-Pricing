import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
warnings.simplefilter("ignore")
rng=np.random.default_rng(2026)
def shares(p,a,b): A=a*np.power(p,-b); return A/(1.0+A.sum())
def scalar_fi(c,a,b):
    def g(A):
        p=(c+A)/(1.0-1.0/b); s=shares(p,a,b); return float(s@(p-c))-A
    hi=1.0
    while g(hi)>0 and hi<1e7: hi*=4.0
    A=brentq(g,0.0,hi,xtol=1e-12,maxiter=300); return (c+A)/(1.0-1.0/b)
def g_it(p0,c,a,b):
    p=p0.copy()
    for k in range(2000):
        s=shares(p,a,b); eta=np.maximum(b*(1.0-s),1.01)
        pn=np.maximum(c/(1.0-1.0/eta),c*1.0001)
        if np.max(np.abs(pn-p))<1e-8: return pn,k+1
        p=pn
    return p,k+1
def ms_dense(p0,c,a,b):
    p=p0.copy()
    for k in range(2000):
        s=shares(p,a,b); u=b*s/p; Om=np.outer(u,s); np.fill_diagonal(Om,-u*(1.0-s))
        d=np.diag(Om).copy(); Ga=Om-np.diag(d)
        pn=np.maximum(c-(s+Ga@(p-c))/d,c*1.0001)
        if np.max(np.abs(pn-p))<1e-8: return pn,k+1
        p=pn
    return p,k+1
def newton_dense(p0,c,a,b):
    p=p0.copy()
    for k in range(200):
        s=shares(p,a,b); u=b*s/p; Om=np.outer(u,s); np.fill_diagonal(Om,-u*(1.0-s))
        F=s+Om@(p-c)
        if np.max(np.abs(F))<1e-10: return p,k
        # J approx = Om (Gauss-Newton style step used in dense benchmarks)
        step=np.linalg.solve(Om+1e-12*np.eye(len(p)), -F)
        p=np.maximum(p+step, c*1.0001)
    return p,k
rows=[]
for n in [500,2000,8000,50000]:
    bb=rng.uniform(1.5,3.5,n); pp=np.exp(rng.normal(3,0.5,n)); ss=rng.dirichlet(np.ones(n))*0.25
    aa=(ss/0.75)*np.power(pp,bb); cc=0.7*pp; p0=pp.copy()
    t0=time.perf_counter(); g_it(p0,cc,aa,bb); tg=time.perf_counter()-t0
    t0=time.perf_counter(); scalar_fi(cc,aa,bb); tf=time.perf_counter()-t0
    tm=np.nan; tn=np.nan
    if n<=8000:
        t0=time.perf_counter(); ms_dense(p0,cc,aa,bb); tm=time.perf_counter()-t0
    if n<=2000:
        t0=time.perf_counter(); newton_dense(p0,cc,aa,bb); tn=time.perf_counter()-t0
    rows.append(dict(n=n,gamma_ms=1000*tg,scalarfi_ms=1000*tf,ms_ms=1000*tm,newton_ms=1000*tn))
    print(rows[-1])
d=pd.DataFrame(rows)
d.to_csv("/sessions/epic-brave-gauss/mnt/outputs/scalability_results.csv",index=False)
fig,ax=plt.subplots(figsize=(7.2,5))
ax.loglog(d.n,d.gamma_ms,'o-',color='tab:blue',label="γ-iteration (O(n))",lw=2,ms=8)
ax.loglog(d.n,d.scalarfi_ms,'d-',color='tab:purple',label="scalar FI, Prop. 5 (O(n))",lw=2,ms=8)
ok=d.dropna(subset=['ms_ms']); ax.loglog(ok.n,ok.ms_ms,'s-',color='tab:red',label="dense MS2011 (O(n²))",lw=2,ms=8)
ok2=d.dropna(subset=['newton_ms']); ax.loglog(ok2.n,ok2.newton_ms,'^-',color='tab:green',label="dense Newton FI (O(n³))",lw=2,ms=8)
ax.annotate("dense: memory/time\ninfeasible beyond\nplotted range", xy=(50000, ax.get_ylim()[1]*0.2), fontsize=9, color='tab:red', ha='center')
ax.set_xlabel("Catalog size n (products)"); ax.set_ylabel("Wall-clock time (ms)")
ax.set_title("Solve time vs. catalog size (same machine, same markets)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
for dst in ["/sessions/epic-brave-gauss/mnt/MktSci/scalability_plot.png",
            "/sessions/epic-brave-gauss/mnt/Gamma/repl_check/gamma-equalization-replication/figures/scalability_plot.png",
            "/sessions/epic-brave-gauss/mnt/outputs/scalability_plot.png"]:
    plt.savefig(dst, dpi=150)
print("figure saved")
