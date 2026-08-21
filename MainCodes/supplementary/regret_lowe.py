"""Regret under misspecified cross-price structure.
Truth: mixed logit (consumer-heterogeneous alpha). Pricer fits MCI at
observed prices, matching TRUE shares and TRUE own elasticities (the
own-price side is estimated correctly); MCI's proportional cross
pattern is then an approximation. Compare, evaluated under TRUTH:
  - oracle FI (dense solver under truth)          -> regret 0
  - MCI-FI  (exact scalar-A optimum of fitted MCI) [uses model cross]
  - gamma rule (own-elasticity iteration in fitted MCI) [no cross]
Sweep consumer heterogeneity sigma (misspecification severity).
"""
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
warnings.simplefilter("ignore")
rng = np.random.default_rng(2026)
N, R = 60, 80
SIGMAS = [0.05, 0.30, 0.60, 0.90]
MKTS = 16
TOL, MAXIT = 1e-10, 4000

def true_shares(p, delta, alph):        # alph: (R,)
    U = delta[None,:] - alph[:,None]*p[None,:]          # R x N
    E = np.exp(U - U.max(axis=1, keepdims=True))
    S = E/(1.0 + E.sum(axis=1, keepdims=True)*0 + (np.exp(-U.max(axis=1,keepdims=True)) + E.sum(axis=1,keepdims=True)))
    return S    # R x N individual shares (outside included via +exp(-max)) -- replaced below
def true_shares(p, delta, alph):
    U = delta[None,:] - alph[:,None]*p[None,:]
    m = np.maximum(U.max(axis=1, keepdims=True), 0.0)
    E = np.exp(U - m); E0 = np.exp(-m[:,0])
    denom = E0 + E.sum(axis=1)
    return E/denom[:,None]
def agg_shares(p, delta, alph): return true_shares(p, delta, alph).mean(axis=0)
def true_jac(p, delta, alph):
    S = true_shares(p, delta, alph)                     # R x N
    A = (alph[:,None]*S)                                # R x N
    Om = (A.T @ S)/R                                    # sum_r a s_ri s_rj /R  -> (i,j)
    Om -= np.diag((alph[:,None]*S).mean(axis=0))
    return Om                                           # Om[i,j] = d s_j / d p_i
def true_profit(p, c, delta, alph, M=1.0):
    return float(np.sum((p-c)*agg_shares(p, delta, alph))*M)
def oracle_fi(p0, c, delta, alph):
    p = p0.copy()
    for k in range(MAXIT):
        s = agg_shares(p, delta, alph); Om = true_jac(p, delta, alph)
        d = np.diag(Om).copy(); Ga = Om - np.diag(d)
        pn = np.maximum(c - (s + Ga@(p-c))/d, c*1.0001)
        if np.max(np.abs(pn-p)) < TOL: return pn
        p = 0.5*p + 0.5*pn if k > 2000 else pn
    return p
def fit_mci(p0, c, delta, alph):
    s = agg_shares(p0, delta, alph); Om = true_jac(p0, delta, alph)
    eta_own = -np.diag(Om)*p0/s                          # |eta_ii| true
    beta = np.maximum(eta_own/(1.0-s), 1.05)
    s0 = 1.0 - s.sum()
    kap = (s/s0)*np.power(p0, beta)
    return beta, kap
def mci_shares(p, kap, beta):
    A = kap*np.power(p, -beta); return A/(1.0+A.sum())
def mci_fi_scalar(c, kap, beta):
    def g(A):
        p = (c+A)/(1.0-1.0/beta); s = mci_shares(p, kap, beta)
        return float(s@(p-c)) - A
    hi = 1.0
    while g(hi) > 0 and hi < 1e6: hi *= 4.0
    A = brentq(g, 0.0, hi, xtol=1e-12, maxiter=300)
    return (c+A)/(1.0-1.0/beta)
def gamma_rule(p0, c, kap, beta):
    p = p0.copy()
    for k in range(MAXIT):
        s = mci_shares(p, kap, beta); eta = np.maximum(beta*(1.0-s), 1.01)
        pn = np.maximum(c/(1.0-1.0/eta), c*1.0001)
        if np.max(np.abs(pn-p)) < 1e-8: return pn
        p = pn
    return p

rows = []
for sg in SIGMAS:
    for m in range(MKTS):
        delta_base = rng.standard_normal(N)*0.6
        alph = np.exp(rng.normal(0.6, sg, R))
        c = np.exp(rng.normal(0.0, 0.2, N)) + 0.5
        S_target = rng.uniform(0.06, 0.16)
        shift = -4.0
        for _cal in range(4):                              # calibrate outside so S(p*) ~= S_target
            delta = delta_base + shift
            p_star = oracle_fi(c*1.4, c, delta, alph)
            S_now = float(agg_shares(p_star, delta, alph).sum())
            S_now = min(max(S_now, 1e-4), 1-1e-4)
            shift += np.log(S_target/(1-S_target)) - np.log(S_now/(1-S_now))
        delta = delta_base + shift
        p_star = oracle_fi(c*1.4, c, delta, alph)        # truth optimum
        pi_star = true_profit(p_star, c, delta, alph)
        if pi_star <= 0: continue
        beta, kap = fit_mci(p_star, c, delta, alph)       # fit at true-FI prices
        p_fi = mci_fi_scalar(c, kap, beta)
        p_g  = gamma_rule(p_star, c, kap, beta)
        # misspecification index: rel Frobenius distance of cross blocks
        Om_t = true_jac(p_star, delta, alph)
        u = beta*mci_shares(p_star, kap, beta)/p_star
        s_m = mci_shares(p_star, kap, beta)
        Om_m = np.outer(u, s_m); np.fill_diagonal(Om_m, -u*(1.0-s_m))
        off = ~np.eye(N, dtype=bool)
        mis = float(np.linalg.norm((Om_t-Om_m)[off])/np.linalg.norm(Om_t[off]))
        s_fit = mci_shares(p_star, kap, beta); S_in = float(s_fit.sum())
        eb = float(np.max((S_in - s_fit)/(1.0 - s_fit)))
        rows.append(dict(sigma=sg, mkt=m, mis=mis, S=S_in, ebar=eb,
            regret_mcifi=1.0-true_profit(p_fi, c, delta, alph)/pi_star,
            regret_gamma=1.0-true_profit(p_g, c, delta, alph)/pi_star))
d = pd.DataFrame(rows)
d.to_csv("/sessions/epic-brave-gauss/mnt/outputs/regret_misspec_lowe.csv", index=False)
g = d.groupby("sigma").median(numeric_only=True)
print(g.round(4).to_string())
print()
d["ebin"] = pd.cut(d.ebar, [0,0.15,0.3,1.0], labels=["low e<0.15","mid 0.15-0.3","high e>0.3"])
print(d.groupby("ebin")[["regret_mcifi","regret_gamma","mis"]].median().round(4).to_string())
print()
print("overall: markets where gamma regret <= MCI-FI regret: %d / %d" % ((d.regret_gamma<=d.regret_mcifi).sum(), len(d)))
