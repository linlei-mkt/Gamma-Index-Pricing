# Replication Package — $\gamma$-Equalization

This repository contains the complete replication package for

> *Revenue-Constrained Multi-Product Pricing via $\gamma$-Equalization:
> A Diagonal Approximation Under MCI Demand.*
> Working paper, 2026.

The paper develops a diagonal pricing rule, $\gamma$-equalization,
that approximates the constrained Bertrand–Nash optimum under MCI
demand using only own-price elasticity information. The
share-only diagonal-dominance slack
$\bar e(\mathbf{p}) = \max_i (S - s_i) / (1 - s_i)$ provides a
parameter-free regime indicator. Empirical illustration on the
JD.com MSOM 2020 dataset under hierarchical-Bayesian MCI demand.

## Repository contents

```
gamma-equalization-replication/
├── README.md                    this file
├── REPLICATION.md               step-by-step replication protocol
├── requirements.txt             pinned Python dependencies
├── MainCodes/                   all replication scripts (.py)
├── JD_MSOM/                     MSOM 2020 dataset (3 CSVs)
├── figures/                     pre-generated reference figures
└── reference_output_results/    pre-generated reference CSVs
```

## Quick start

```bash
git clone <repo-url>
cd gamma-equalization-replication
pip install -r requirements.txt

cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM

python3 jd_hierarchical_bayes.py    # ~10 min  → Table 3, Figure 2 (left)
python3 jd_gmv_constrained.py        # ~1 min   → Table 4, Figure 3
```

After Step 2, the median values of `gap_gamma_pct` (~5.6%) and
`gap_unif_pct` (~15.9%) in `jd_gmv_pricing_comparison.csv` confirm
the paper's headline GMV-constrained result: under a 15% revenue
floor, tuned $\gamma^\star$ achieves a 5.6% median profit gap to
constrained Bertrand–Nash, versus 15.9% for tuned uniform markup.

Full step-by-step instructions for every numbered Table (3–10) and
Figure (1–11) are in [`REPLICATION.md`](REPLICATION.md).

## Paper structure → script flow

The paper proceeds from theory (§1–§6) to a JD calibrated
illustration (§7), GMV-constrained pricing (§8), a one-page
robustness summary (§9), managerial implications (§10), discussion
(§11), and conclusion (§12). The detailed robustness package, the
calibrated Monte Carlo, and the conduct-assumption analysis for
marginal-cost inversion are in **Appendices L, M, and N**
respectively.
Appendices A–K contain the proofs and the contraction condition
for the $\gamma$-update map.

| Paper element | Content | Replication script |
|---|---|---|
| §7 (Table 3, Figures 1–2) | JD calibrated illustration with HB-MCI demand | `jd_hierarchical_bayes.py`, `jd_experiment.py` |
| §8 (Table 4, Figure 3) | GMV-constrained pricing on JD (15% floor) | `jd_gmv_constrained.py` |
| App L.1 (Table 5, Figure 4) | GMV-floor sensitivity ($\phi \in \{1.05,\ldots,1.25\}$) | `jd_gmv_floor_sensitivity.py` |
| App L.2 (Table 6) | Hausman-IV from cross-DC variation | `jd_hausman_iv.py` |
| App L.3 (Table 7, Figure 5) | Wall-clock scalability at $n \in \{500,\ldots,50{,}000\}$ | `scalability_demo.py` |
| App L.4 (Table 8) | $M_{\mathrm{mult}} \times c/\bar p$ sensitivity grid | `jd_hb_sensitivity.py` |
| App L.5 | Mixed-logit Monte Carlo robustness (narrative only) | `mixed_logit_robustness.py` |
| App L.6 (Figure 6) | Empirical local operator-norm contraction diagnostic | `jd_cbar_diagnostic.py` |
| App L.7 (Figures 7–8) | Posterior predictive check, holdout, uncertainty propagation | `jd_hb_validation.py` |
| App L.8 (Table 9, Figure 9) | Top-$N$ catalog-truncation robustness | `jd_topn_sensitivity.py` |
| App L.9 (Figures 10–11) | $\gamma^\star$-misspecification sweep + outcome-based calibration | `jd_gamma_robustness.py` |
| App M (Table 10) | Calibrated Monte Carlo + HKMR (1995) illustration | `empirical_gamma.py` |
| App N | Conduct assumptions in marginal-cost inversion | (analytical, no script) |

## Reference outputs

`figures/` and `reference_output_results/` contain pre-generated
versions of every paper figure and table CSV. Compare your fresh
outputs to these to verify reproducibility. Numerical agreement
should be exact under the tested software environment (see
`requirements.txt`) and to the 3rd decimal under reasonable
version drift.

## Citation

```bibtex
@unpublished{gamma_equalization_2026,
  author = {Authors blinded for review},
  title  = {Revenue-Constrained Multi-Product Pricing via $\gamma$-Equalization:
            A Diagonal Approximation Under MCI Demand},
  year   = {2026},
  note   = {Working paper}
}
```

## License

Code: MIT License. The JD MSOM 2020 dataset is distributed under
the MSOM challenge terms; see the
[challenge page](https://connect.informs.org/msom/events/datadriven-call).
