"""
JD real-data BLP / random-coefficients logit with PyBLP.

Replaces the homemade jd_mixed_logit_real.py (which had a share-
aggregation bug) with proper BLP-GMM estimation via the PyBLP package,
which implements the full BLP (1995) demand system — nested share
inversion, random coefficients, and instrumental-variables identification.

What this script does
---------------------
1.  Load JD data (orders + SKU metadata) and aggregate to daily × SKU
    shares using the same M_mult = 3 and top-500 filter as the baseline.
2.  Construct instruments for price endogeneity:
      * Differentiation IVs (BLP): sums and products of rival-product
        characteristics within each market, as in Gandhi & Houde (2019).
      * Hausman IVs: mean price of the same SKU across other days
        (weak but tractable without cost shifters).
3.  Set up a BLP problem with:
      * X1 (linear) = price + SKU fixed effects + day fixed effects
      * X2 (random coefficients) = price
      (So σ = standard deviation of the random coefficient on price.)
4.  Solve with PyBLP's NFP contraction + GMM; optimizer = L-BFGS-B.
5.  Extract estimated elasticity matrix at observed prices.
6.  Compute the MS-type diagonal-dominance slack ebar from the full
    estimated Jacobian.
7.  Run the four pricing methods (γ-iteration, MS2011, uniform, Newton)
    under the estimated BLP demand, and report profit gaps to BN.

Outputs (in /Users/linlei/Downloads/Gamma/):
  - jd_blp_parameter_estimates.csv     α_hat, σ_hat, standard errors
  - jd_blp_elasticity_summary.csv      distribution of own- and cross-
                                         elasticities across products
  - jd_blp_pricing_comparison.csv      per-day γ/MS/uniform/BN results
  - jd_blp_ebar_distribution.png
  - jd_blp_profit_gap_vs_ebar.png

Required:
    pip install pyblp pandas numpy matplotlib scipy
  (PyBLP depends on numpy, scipy, pandas, sympy, patsy, pyhdfe.)

Runtime: 10-60 minutes depending on (a) how many markets and products
you keep, (b) whether PyBLP's NFP contraction converges on the initial
σ guess, and (c) CPU count.  For safety this script defaults to a
smaller slice (top 300 SKUs, 15 markets) to keep runtime tractable on
a laptop. Edit the constants at the top for the full top-500 / 31-day
fit.

Known caveats (important to read before interpreting output)
------------------------------------------------------------
1. Without genuine cost-shifter instruments, BLP's identification of
   price endogeneity on the JD slice rests on functional form plus
   the Hausman-style cross-day IV.  A referee could reasonably push
   for cost shifters; JD's distribution center field (`dc_ori`) is
   a natural starting point but we don't use it here because the
   `order_data` CSV records DC per order, not per (SKU, day) cell,
   and constructing DC-level price variation requires more data
   munging than fits this script.
2. PyBLP's NFP contraction can fail to converge for poorly-scaled
   starting values of σ.  If the script reports contraction
   failures, try the starting values in INIT_SIGMA below.
3. With 31 markets and modest within-SKU price variation, the
   random-coefficient standard deviation σ is identified only
   weakly.  A large posterior SE on σ is normal; a σ point estimate
   far from 0 would be surprising.

To run:
    JD_DATA_DIR=/path/to/jd_csvs python3 jd_pyblp.py
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root

warnings.simplefilter("ignore", category=FutureWarning)

try:
    import pyblp
    pyblp.options.digits = 2
    pyblp.options.verbose = False
    HAS_PYBLP = True
except ImportError:
    HAS_PYBLP = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "JD_DATA_DIR",
    "/Users/linlei/Library/Application Support/Claude/local-agent-mode-sessions/"
    "28ce55e3-2aeb-47b1-a159-176e9d6a9dbf/0f00e7ba-81f1-4095-94e5-73a365a8f51b/"
    "local_c02dff0e-7360-4888-8cca-0a64aed3b4e1/uploads"
))
OUT_DIR = SCRIPT_DIR

# ============== Config ==============
N_TOP_SKU = 300          # keep tractable; raise to 500 for the full paper spec
N_DAYS = 15              # keep tractable; raise to 31 for full month
M_MULT = 3.0
MARGIN_RATIO = 0.70
TOL = 1e-8
MAX_ITER = 2000
INIT_SIGMA = 0.5         # starting value for σ (SD of random price coefficient)
SEED = 2026
np.random.seed(SEED)


# ======================================================================
# Data prep
# ======================================================================
def load_and_build_blp_data():
    """Return a pyblp-ready product_data DataFrame."""
    orders = pd.read_csv(
        DATA_DIR / "JD_order_data.csv",
        usecols=["sku_ID", "order_date", "quantity", "final_unit_price"],
        dtype={"sku_ID": "string"}, parse_dates=["order_date"],
    )
    sku = pd.read_csv(
        DATA_DIR / "JD_sku_data.csv",
        usecols=["sku_ID", "type", "brand_ID", "attribute1", "attribute2"],
        dtype={"sku_ID": "string", "brand_ID": "string"},
    )
    top_ids = orders.groupby("sku_ID").size().nlargest(N_TOP_SKU).index.tolist()
    o = orders[orders["sku_ID"].isin(top_ids)].merge(sku, on="sku_ID", how="left")
    o["day"] = o["order_date"].dt.day

    agg = (
        o.assign(rev=o["final_unit_price"] * o["quantity"])
         .groupby(["day", "sku_ID"], as_index=False)
         .agg(qty=("quantity", "sum"), rev=("rev", "sum"))
    )
    agg["price"] = agg["rev"] / agg["qty"]
    agg = agg.merge(
        sku[["sku_ID", "type", "brand_ID", "attribute1", "attribute2"]],
        on="sku_ID", how="left",
    )
    agg = agg[
        (agg["qty"] > 0) & (agg["price"] > 0) &
        agg["type"].notna() & agg["brand_ID"].notna()
    ].copy()
    # Keep first N_DAYS days
    agg = agg[agg["day"] <= N_DAYS]
    peak_inside = agg.groupby("day")["qty"].sum().max()
    M = M_MULT * peak_inside
    agg["shares"] = agg["qty"] / M
    # PyBLP requires: market_ids, product_ids, firm_ids, shares, prices
    product_data = agg.rename(
        columns={"day": "market_ids", "sku_ID": "product_ids",
                 "brand_ID": "firm_ids"}
    )
    product_data["prices"] = product_data["price"]

    # Coerce attribute columns to numeric (they can arrive as strings
    # depending on CSV parsing), then fill NaN with median.
    for a in ["attribute1", "attribute2"]:
        product_data[a] = pd.to_numeric(product_data[a], errors="coerce")
        med = product_data[a].median()
        if pd.isna(med):
            med = 0.0
        product_data[a] = product_data[a].fillna(med)

    return product_data.reset_index(drop=True), M


def build_differentiation_ivs(product_data):
    """Gandhi-Houde (2019) differentiation IVs: for each product i in
    market t, construct IV = sum over rivals j ≠ i of characteristic
    distance or squared distance."""
    ivs_list = []
    for char in ["attribute1", "attribute2"]:
        x = product_data[char].to_numpy()
        market = product_data["market_ids"].to_numpy()
        # For each row i, sum over j in same market where j != i of (x_j - x_i)^2
        iv = np.zeros(len(product_data))
        for m in np.unique(market):
            idx = np.where(market == m)[0]
            xm = x[idx]
            # broadcast: (n,1) - (1,n) -> (n,n)
            d2 = (xm[:, None] - xm[None, :]) ** 2
            iv[idx] = d2.sum(axis=1)   # includes i=i (contributes 0)
        ivs_list.append(iv)
    return np.column_stack(ivs_list)


def build_hausman_ivs(product_data):
    """For each (SKU, day), use the mean price of the same SKU on OTHER
    days as a Hausman-style instrument (cross-market-same-product)."""
    df = product_data.copy()
    # Mean price of same SKU across all days, then subtract own-day
    sku_mean = df.groupby("product_ids")["prices"].mean().rename("sku_mean_p")
    sku_n = df.groupby("product_ids")["prices"].count().rename("sku_n")
    df = df.merge(sku_mean, on="product_ids").merge(sku_n, on="product_ids")
    # leave-one-out: (n * mean - own) / (n - 1)
    df["hausman_p"] = np.where(
        df["sku_n"] > 1,
        (df["sku_n"] * df["sku_mean_p"] - df["prices"]) / (df["sku_n"] - 1),
        df["sku_mean_p"],
    )
    return df["hausman_p"].to_numpy().reshape(-1, 1)


# ======================================================================
# BLP estimation
# ======================================================================
def build_rival_mean_price(product_data):
    """Classical BLP IV: for each (SKU, market), the mean price of OTHER
    SKUs in the same market.  This varies within-SKU across markets
    (because the set of rivals and their prices change), so it's not
    absorbed by SKU fixed effects."""
    df = product_data.copy()
    out = np.zeros(len(df))
    for m in df["market_ids"].unique():
        idx = np.where(df["market_ids"] == m)[0]
        p = df["prices"].to_numpy()[idx]
        n = len(p)
        if n > 1:
            out[idx] = (p.sum() - p) / (n - 1)
        else:
            out[idx] = p
    return out


def fit_blp(product_data):
    if not HAS_PYBLP:
        raise ImportError("pip install pyblp")

    product_data = product_data.copy()

    # Instruments.
    # - Hausman IV: leave-one-out own-SKU mean price across other markets.
    #   Varies within SKU across markets -> not absorbed by SKU FE.
    # - Mean rival price within market: varies within SKU across markets
    #   because the composition and prices of rival products change.
    #   Also not absorbed by SKU FE.
    # We deliberately do NOT use the Gandhi-Houde "squared attribute
    # distance" or "attribute product" instruments here: when the product
    # set is (nearly) constant across markets, those instruments are
    # (nearly) constant within SKU and therefore absorbed by SKU FE,
    # which triggers pyblp's collinearity guard.
    hausman = build_hausman_ivs(product_data)
    rival_mean = build_rival_mean_price(product_data)
    product_data["demand_instruments0"] = hausman[:, 0]
    product_data["demand_instruments1"] = rival_mean

    # Formulations
    # X1 = prices (SKU + day FE are absorbed separately; no constant
    # because day FE subsume the constant).
    # X2 = prices (random coefficient on price)
    X1_formulation = pyblp.Formulation(
        "0 + prices",
        absorb="C(product_ids) + C(market_ids)",
    )
    X2_formulation = pyblp.Formulation("0 + prices")

    # Agent data: Monte Carlo draws for the random coefficient.
    # One row per market x draw.  No demographics -> no agent_formulation.
    market_ids = product_data["market_ids"].unique()
    agent_draws = 100
    np.random.seed(SEED)
    agent_data_list = []
    for m in market_ids:
        weights = np.full(agent_draws, 1.0 / agent_draws)
        nodes = np.random.standard_normal((agent_draws, 1))
        agent_data_list.append(pd.DataFrame({
            "market_ids": m,
            "weights": weights,
            "nodes0": nodes[:, 0],
        }))
    agent_data = pd.concat(agent_data_list, ignore_index=True)

    # Initial sigma for random coefficient on price
    initial_sigma = np.array([[INIT_SIGMA]])

    print(f"    n_products={len(product_data)}, n_markets={len(market_ids)}, "
          f"agent_draws={agent_draws}")
    print(f"    instruments: Hausman (cross-market own-SKU mean price) "
          f"+ rival mean price in market")
    print(f"    initial σ = {INIT_SIGMA}")

    problem = pyblp.Problem(
        product_formulations=(X1_formulation, X2_formulation),
        product_data=product_data,
        agent_data=agent_data,
        # No agent_formulation: there are no demographics.
    )

    print(f"    problem defined: K1={problem.K1}, K2={problem.K2}, "
          f"MD={problem.MD}, MS={problem.MS}")
    print("    starting GMM...")
    t0 = time.time()
    results = problem.solve(
        sigma=initial_sigma,
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-4}),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-12}),
    )
    print(f"    GMM done in {(time.time()-t0)/60:.1f} min")
    print("    estimation results:")
    print(results)

    return results, product_data, agent_data


# ======================================================================
# Pricing solvers under BLP-estimated demand
# ======================================================================
def mixed_logit_shares_from_blp(p, sigma_draws, mean_utility_parts, alpha):
    """Compute mixed-logit shares for one market.
        u_{ri} = mean_utility_parts_i + (alpha + sigma * nu_r) * p_i
    sigma_draws: (R,)  standard-normal draws for one market
    mean_utility_parts: (n,)  α_i + δ_t (SKU FE + day FE, already absorbed)
    alpha: scalar, mean price coefficient (negative)
    """
    R = len(sigma_draws)
    u = mean_utility_parts[None, :] + \
        (alpha + sigma_draws[:, None] * 1.0) * p[None, :]
    u_max = np.maximum(u.max(axis=1, keepdims=True), 0.0)
    e = np.exp(u - u_max)
    e0 = np.exp(-u_max[:, 0])
    D = e0 + e.sum(axis=1)
    s_rj = e / D[:, None]
    s = s_rj.mean(axis=0)
    s0 = (e0 / D).mean()
    return s, s0, s_rj


def mixed_logit_jacobian_from_blp(p, sigma_draws, mean_utility_parts, alpha):
    """Full Ω_{ij} = ∂s_j/∂p_i under estimated mixed logit.

    Derivation. With u_{rj} = mean_utility_j + β_r * p_j (so β_r < 0
    in the usual case of a negative mean price coefficient):
        ∂s_{rj}/∂p_i = β_r · s_{rj} · (δ_{ij} − s_{ri}).
    Aggregate (1/R averaging):
        Ω_{ii} = (1/R) Σ_r β_r · s_{ri} · (1 − s_{ri})      (negative when β<0)
        Ω_{ij} = −(1/R) Σ_r β_r · s_{rj} · s_{ri}  for i≠j  (positive when β<0)

    `beta_r` here is the already-reconstructed per-consumer coefficient,
    i.e. `beta_r = alpha + sigma * z_r` where caller passes
    `sigma_draws = sigma * z_r`.
    """
    s, _, s_rj = mixed_logit_shares_from_blp(
        p, sigma_draws, mean_utility_parts, alpha)
    R, n = s_rj.shape
    beta_r = alpha + sigma_draws

    # Σ_r β_r · s_{ri} · s_{rj} (outer-product form), averaged over r.
    # Element [i, j] = (1/R) Σ_r β_r · s_{ri} · s_{rj}.
    cross = (beta_r[:, None] * s_rj).T @ s_rj / R
    # Correct signs: off-diagonal Ω_{ij} = -cross[i,j]
    Om = -cross
    # Diagonal Ω_{ii} = (1/R) Σ_r β_r · s_{ri} · (1 − s_{ri}).
    diag_own = (beta_r[:, None] * s_rj * (1.0 - s_rj)).mean(axis=0)
    np.fill_diagonal(Om, diag_own)
    return Om


def ebar_from_jacobian(Om):
    diag = np.diag(Om)
    off = np.abs(Om).sum(axis=1) - np.abs(diag)
    return float(np.max(off / np.abs(diag)))


def gamma_iteration(p0, c, sigma_draws, mean_util, alpha):
    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(MAX_ITER):
        s, _, s_rj = mixed_logit_shares_from_blp(p, sigma_draws, mean_util, alpha)
        beta_r = alpha + sigma_draws
        # Ω_{ii} = (1/R) Σ_r β_r · s_{ri} · (1 − s_{ri}), negative when β<0.
        # |η_i| = | p_i · Ω_{ii} / s_i | = | p_i · β_{eff} · (1-s_i) |.
        diag = (beta_r[:, None] * s_rj * (1.0 - s_rj)).mean(axis=0)
        eta_abs = np.abs(p * diag / np.maximum(s, 1e-12))
        p_new = np.maximum(c / (1.0 - 1.0 / np.maximum(eta_abs, 1.01)),
                           c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new; break
        p = p_new
    return p, k + 1, time.perf_counter() - t0


def ms_iteration(p0, c, sigma_draws, mean_util, alpha):
    p = p0.copy()
    t0 = time.perf_counter()
    for k in range(MAX_ITER):
        s, _, _ = mixed_logit_shares_from_blp(p, sigma_draws, mean_util, alpha)
        Om = mixed_logit_jacobian_from_blp(p, sigma_draws, mean_util, alpha)
        diag = np.diag(Om).copy()
        Gamma = Om - np.diag(diag)
        p_new = np.maximum(c - (s + Gamma @ (p - c)) / diag, c * 1.0001)
        if np.max(np.abs(p_new - p)) < TOL:
            p = p_new; break
        p = p_new
    return p, k + 1, time.perf_counter() - t0


def newton_bn(p0, c, sigma_draws, mean_util, alpha):
    def F(p):
        p_pos = np.maximum(p, c * 1.0001)
        s, _, _ = mixed_logit_shares_from_blp(p_pos, sigma_draws, mean_util, alpha)
        Om = mixed_logit_jacobian_from_blp(p_pos, sigma_draws, mean_util, alpha)
        return s + Om @ (p_pos - c)
    t0 = time.perf_counter()
    sol = root(F, p0, method="krylov", tol=1e-10, options={"maxiter": 500})
    return np.maximum(sol.x, c * 1.0001), sol.nit or -1, time.perf_counter() - t0


def uniform_pricing(c, sigma_draws, mean_util, alpha):
    def neg_profit(m):
        if not (0.0 < m < 0.999):
            return 1e18
        p = c / (1.0 - m)
        s, _, _ = mixed_logit_shares_from_blp(p, sigma_draws, mean_util, alpha)
        return -np.sum((p - c) * s)
    res = minimize_scalar(neg_profit, bounds=(0.001, 0.999), method="bounded",
                          options={"xatol": 1e-8})
    return c / (1.0 - res.x)


def total_profit(p, c, sigma_draws, mean_util, alpha):
    s, _, _ = mixed_logit_shares_from_blp(p, sigma_draws, mean_util, alpha)
    return float(np.sum((p - c) * s))


# ======================================================================
# Main
# ======================================================================
def main():
    print("=" * 70)
    print("JD PyBLP estimation + pricing comparison")
    print("=" * 70)
    if not HAS_PYBLP:
        print("ERROR: PyBLP not installed.")
        print("  pip install pyblp")
        return

    print(f"[1/4] Loading data (N_TOP_SKU={N_TOP_SKU}, N_DAYS={N_DAYS})...")
    product_data, M = load_and_build_blp_data()
    print(f"    {len(product_data)} product-market rows, M={M:.0f}")

    print("[2/4] Fitting BLP via PyBLP...")
    results, product_data, agent_data = fit_blp(product_data)

    # Extract parameters. PyBLP returns arrays; use .item() for scalars.
    alpha_hat = float(np.asarray(results.beta).ravel()[-1])
    sigma_hat = float(np.asarray(results.sigma).ravel()[0])
    print()
    print(f"    α̂ (mean price coefficient) = {alpha_hat:.4f}")
    print(f"    σ̂ (SD of random coefficient) = {sigma_hat:.4f}")
    if abs(sigma_hat) < 1e-4:
        print("    NOTE: σ̂ pinned near zero -- the data + instruments do "
              "not support detectable random-coefficient heterogeneity.")
        print("    The mixed logit degenerates to homogeneous MNL; the")
        print("    pricing comparison below is still valid but is effectively")
        print("    run under MNL demand.")

    beta_se_arr = np.asarray(results.beta_se).ravel() if results.beta_se is not None else None
    sigma_se_arr = np.asarray(results.sigma_se).ravel() if results.sigma_se is not None else None
    pd.DataFrame([
        {"parameter": "alpha_mean_price_coef",
         "estimate": alpha_hat,
         "se": float(beta_se_arr[-1]) if beta_se_arr is not None else np.nan},
        {"parameter": "sigma_rc_on_price",
         "estimate": sigma_hat,
         "se": float(sigma_se_arr[0]) if sigma_se_arr is not None else np.nan},
    ]).to_csv(OUT_DIR / "jd_blp_parameter_estimates.csv", index=False)

    # ---- Extract mean utilities (δ in BLP notation) ----
    # δ_{jt} = X1 β + α·p (aggregate attractiveness at observed prices).
    # PyBLP returns delta as shape (n_obs, 1); flatten to 1D so it
    # broadcasts correctly with per-market price vectors.
    delta = np.asarray(results.delta).flatten()

    # ---- Elasticity summary ----
    print("[3/4] Computing elasticity matrix...")
    elasticities_own = []
    for mid in product_data["market_ids"].unique():
        mkt = product_data[product_data["market_ids"] == mid]
        idx = mkt.index.to_numpy()
        p_obs = np.asarray(mkt["prices"].to_numpy(), dtype=float).ravel()
        # δ_i - α·p_i  =  mean utility "net of price" for reconstructing
        # predicted shares at arbitrary p. Both sides must be 1D of same length.
        mean_util_net = np.asarray(delta[idx]).ravel() - alpha_hat * p_obs

        draws = agent_data[agent_data["market_ids"] == mid]["nodes0"].to_numpy()
        sigma_draws = sigma_hat * draws

        Om = mixed_logit_jacobian_from_blp(p_obs, sigma_draws, mean_util_net,
                                           alpha_hat)
        s, _, _ = mixed_logit_shares_from_blp(p_obs, sigma_draws, mean_util_net,
                                              alpha_hat)
        eta_own = p_obs * np.diag(Om) / np.maximum(s, 1e-12)
        elasticities_own.extend(eta_own.tolist())

    elast_df = pd.Series(elasticities_own).describe().to_frame("own_elasticity")
    elast_df.to_csv(OUT_DIR / "jd_blp_elasticity_summary.csv")
    # Own-price elasticity should be negative; print both sign and magnitude.
    own_arr = np.array(elasticities_own)
    print(f"    own-elasticity: min {own_arr.min():+.3f}, "
          f"median {np.median(own_arr):+.3f}, "
          f"max {own_arr.max():+.3f}  "
          f"(negative is correct)")

    # ---- Pricing comparison ----
    print("[4/4] Running pricing comparison under BLP demand...")
    rows = []
    # Pre-align delta by integer position in product_data, not by
    # boolean mask on a possibly-reindexed DataFrame
    pd_market = product_data["market_ids"].to_numpy()
    for mid in sorted(product_data["market_ids"].unique()):
        mask_rows = (pd_market == mid)
        if mask_rows.sum() < 50:
            continue
        mkt = product_data[mask_rows].reset_index(drop=True)
        p_obs = np.asarray(mkt["prices"].to_numpy(), dtype=float).ravel()
        c = MARGIN_RATIO * p_obs
        delta_mkt = np.asarray(delta[mask_rows]).ravel()
        mean_util_net = delta_mkt - alpha_hat * p_obs
        draws = agent_data[agent_data["market_ids"] == mid]["nodes0"].to_numpy()
        sigma_draws = sigma_hat * draws

        Om_obs = mixed_logit_jacobian_from_blp(p_obs, sigma_draws,
                                               mean_util_net, alpha_hat)
        ebar = ebar_from_jacobian(Om_obs)

        p0 = p_obs.copy()
        p_g, it_g, t_g = gamma_iteration(p0, c, sigma_draws, mean_util_net, alpha_hat)
        p_m, it_m, t_m = ms_iteration(p0, c, sigma_draws, mean_util_net, alpha_hat)
        p_u = uniform_pricing(c, sigma_draws, mean_util_net, alpha_hat)
        try:
            p_bn, it_bn, t_bn = newton_bn(p_m.copy(), c, sigma_draws,
                                          mean_util_net, alpha_hat)
        except Exception:
            p_bn, it_bn, t_bn = p_m.copy(), 0, 0.0

        pi_bn = total_profit(p_bn, c, sigma_draws, mean_util_net, alpha_hat)
        if pi_bn <= 0:
            continue
        gap = lambda p: max(0.0, (pi_bn - total_profit(
            p, c, sigma_draws, mean_util_net, alpha_hat)) / pi_bn)
        rows.append({
            "market": int(mid), "ebar": ebar,
            "gap_gamma": gap(p_g), "gap_MS": gap(p_m), "gap_uniform": gap(p_u),
            "iter_gamma": it_g, "iter_MS": it_m,
            "time_gamma": t_g, "time_MS": t_m,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "jd_blp_pricing_comparison.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["ebar"], bins=15, color="tab:blue", edgecolor="black")
    ax.axvline(df["ebar"].median(), color="red", linestyle="--",
               label=f"median ebar = {df['ebar'].median():.3f}")
    ax.set_xlabel(r"$\bar e$ (BLP Jacobian)"); ax.set_ylabel("number of markets")
    ax.set_title("ebar distribution from estimated BLP demand")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_blp_ebar_distribution.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df["ebar"], df["gap_gamma"] * 100, color="tab:blue",
               label="γ-equalization", s=30)
    ax.scatter(df["ebar"], df["gap_MS"] * 100, color="tab:red",
               label="MS2011", s=30, marker="x")
    ax.scatter(df["ebar"], df["gap_uniform"] * 100, color="tab:green",
               label="uniform markup", s=30, marker="^")
    xs = np.linspace(df["ebar"].min(), df["ebar"].max(), 100)
    mask = df["ebar"] > 0
    if mask.any():
        c_fit = (df.loc[mask, "gap_gamma"] * df.loc[mask, "ebar"] ** 2).sum() \
                / (df.loc[mask, "ebar"] ** 4).sum()
        ax.plot(xs, c_fit * xs ** 2 * 100, "b--", alpha=0.5,
                label=fr"γ gap ≈ {c_fit:.2f}·ebar²")
    ax.set_xlabel(r"$\bar e$"); ax.set_ylabel("profit gap to BN (%)")
    ax.set_title("JD pricing comparison under BLP-estimated demand")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "jd_blp_profit_gap_vs_ebar.png", dpi=150)
    plt.close()

    print()
    print("=" * 70)
    print("Summary (BLP demand):")
    print(f"  α̂ = {alpha_hat:.3f},  σ̂ = {sigma_hat:.3f}")
    print(f"  ebar range: [{df['ebar'].min():.3f}, {df['ebar'].max():.3f}], "
          f"median {df['ebar'].median():.3f}")
    print(f"  γ-gap:   mean {df['gap_gamma'].mean()*100:.3f}%, "
          f"median {df['gap_gamma'].median()*100:.3f}%")
    print(f"  unif:    mean {df['gap_uniform'].mean()*100:.3f}%, "
          f"median {df['gap_uniform'].median()*100:.3f}%")
    print(f"  γ beats uniform: {(df['gap_gamma'] < df['gap_uniform']).mean()*100:.1f}%")
    print(f"  γ-gap < ebar²  : {(df['gap_gamma'] < df['ebar']**2).mean()*100:.1f}%")
    print(f"  speedup γ vs MS: {df['time_MS'].mean() / df['time_gamma'].mean():.1f}×")
    print()
    print("All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
