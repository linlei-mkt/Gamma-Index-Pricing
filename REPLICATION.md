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
├── JD_MSOM/                     place the 3 MSOM 2020 CSVs here (see below)
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
Figure (1–12) are in [`REPLICATION.md`](REPLICATION.md).

## Paper structure → script flow

The paper proceeds from theory (§1–§6) to a JD calibrated
illustration (§7), GMV-constrained pricing (§8), a one-page
robustness summary (§9), managerial implications (§10), discussion
(§11), and conclusion (§12). The detailed robustness package, the
calibrated Monte Carlo, and the conduct-assumption analysis for
marginal-cost inversion are in **Appendices L, M, and N**
respectively.
Appendices A–L contain the proofs and the contraction condition (Appendix I proves uniqueness and global optimality of the scalar-FI rule, Proposition 5)
for the $\gamma$-update map.

| Paper element | Content | Replication script |
|---|---|---|
| §7 (Table 3, Figures 1–2) | JD calibrated illustration with HB-MCI demand | `jd_hierarchical_bayes.py`, `jd_experiment.py` |
| §8 (Table 4, Figure 3) | GMV-constrained pricing on JD (15% floor) | `jd_gmv_constrained.py` |
| App M.1 (Table 5, Figure 4) | GMV-floor sensitivity ($\phi \in \{1.05,\ldots,1.25\}$) | `round8_sync.py` (supersedes `round7_common_feasible.py`, `jd_gmv_floor_sensitivity.py`) |
| App M.2 (Table 6) | Hausman-IV from cross-DC variation | `jd_hausman_iv.py` |
| App M.3 (Table 7, Figure 5) | Wall-clock scalability at $n \in \{500,\ldots,50{,}000\}$ | `scalability_demo.py` |
| App M.4 (Table 8) | $M_{\mathrm{mult}} \times c/\bar p$ sensitivity grid | `jd_hb_sensitivity.py` |
| App M.5 | Mixed-logit Monte Carlo robustness (narrative only) | `mixed_logit_robustness.py` |
| App M.6 (Figure 6) | Empirical local operator-norm contraction diagnostic | `jd_cbar_diagnostic.py` |
| App M.7 (Figures 7–8) | Conditional-mean residual check, holdout, uncertainty propagation | `jd_hb_validation.py` |
| App M.8 (Table 9, Figure 9) | Top-$N$ catalog-truncation robustness | `jd_topn_sensitivity.py` |
| App M.9 (Figures 10–11) | $\gamma^\star$-misspecification sweep + outcome-based calibration | `round8_sync.py` (supersedes `jd_gamma_robustness.py`) |
| App M.10 (Figure 12) | Misspecified cross-price information regret; scalar-FI benchmark | `regret_misspec.py`, `scalar_fi_benchmark.py` |
| App I (Prop. 5(ii)–(iii)) | Numerical check: Φ′(A*)=0, unique root, global optimality | `prop5_uniqueness_check.py` |
| App M (Table 10) | Calibrated Monte Carlo + HKMR (1995) illustration | `empirical_gamma.py` |
| App N | Conduct assumptions in marginal-cost inversion | (analytical, no script) |

## Reference outputs

The JD MSOM 2020 dataset is **not** redistributed in this
repository: download the three CSVs from the MSOM Data-Driven
Research Challenge page (link under License) and place them in
`JD_MSOM/` before running the scripts.

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

> Round-6 note (2026-08-21): the validation bands in App M.7 are conditional-mean credible bands (parameter draws only, no ε draws) — renamed accordingly in the paper. All pricing pipelines floor β at max(β̂, 1.2) (43/415 point estimates floored; median 34 per posterior draw within a daily market), keeping every solve inside Proposition 5's domain. The constrained experiments impose no price boxes; at the scalar-FI optimum 254/10,408 day–product pairs price below marginal cost (min p/c = 0.84), while the γ-implementation keeps p ≥ 1.0001c (binding on 636/10,408 pairs). See `prop5_uniqueness_check.py` for the Prop 5(ii)–(iii) numerical check.

> Round-7 note (2026-08-21): the constrained comparison now uses a common feasible set — the cost safeguard (p ≥ 1.0001c) is removed from the γ-iteration, so both γ and scalar-FI may price below cost (γ: 582/10,408 pairs at φ=1.15; FI: 254). Revenue along the γ-path is hump-shaped once below-cost prices are allowed; the outer search brackets the least-negative crossing (`round7_common_feasible.py`, canonical for Table 4, Fig. jd-gmv, App M.1). New headline: γ-tuned median gap 5.36% (30/31 feasible; March 1 infeasible, hump peak = 98.3% of target), uniform 15.93%, mean advantage 9.9pp. β-floor sensitivity in `round7_beta_floor_sens.csv` (unconstrained gap 1.60–1.61% across floors 1.05–1.5). Constrained posterior propagation rerun: 4.5% median, 2.3pp width (`round7_constrained_posterior_prop.csv`). Prop 5(i) prior art now cited: Gallego–Wang (2014), Li–Huh (2011), Keller–Levi–Perakis (2014).

> Round-8 note (2026-08-21): uniform markup is now tuned over m < 1 with no cost floor, matching the unclamped γ rule; the tuned m* stays positive (0.03–0.21 at φ=1.15) and feasibility counts are unchanged, so this is a fairness fix rather than a numbers change. Appendix M.9 (tuning-error sweep, outcome-based feedback) is re-run on the 30 γ-feasible days: 270/270 (day, Δγ) pairs converge in a median of 6 sweeps, GMV slope −74%/unit, Lagrangian loss measured against the constrained FI optimum = 5.36% + 0.55|Δγ| + 283.8Δγ²; feedback reaches |γ₁₂ − γ̂*| = 7e-4 (no noise) and 0.013 (2% noise); round 1 attains 72% of target. Canonical script: `round8_sync.py`. Bibliography: Huh & Li (2015, OR 63(4):840–850) added; Keller–Levi–Perakis corrected to Math. Prog. 145(1–2):223–261.
