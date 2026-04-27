# Replication Package — $\\gamma$-Equalization

This repository contains the complete replication package for

*Revenue-Constrained Multi-Product Pricing via $\\gamma$-Equalization: A Diagonal Approximation Under MCI Demand.* Working paper, 2026\.

The paper develops a diagonal pricing rule, $\\gamma$-equalization, that approximates the constrained Bertrand–Nash optimum under MCI demand using only own-price elasticity information. Empirical illustration on the JD.com MSOM 2020 dataset.

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

```shell
git clone <repo-url>
cd gamma-equalization-replication
pip install -r requirements.txt

cd MainCodes
export JD_DATA_DIR=$(pwd)/../JD_MSOM

python3 jd_hierarchical_bayes.py    # ~10 min  → Table 4, Figure 2
python3 jd_gmv_constrained.py        # ~1 min   → Table 5, Figure 3
```

After Step 2, the median values of `gap_gamma_pct` (\~5.6%) and `gap_unif_pct` (\~15.9%) in `jd_gmv_pricing_comparison.csv` confirm the paper's headline GMV-constrained result. Full step-by-step instructions for every numbered Table and Figure are in [`REPLICATION.md`](http://REPLICATION.md).

## Reference outputs

`figures/` and `reference_output_results/` contain pre-generated versions of every paper figure and table CSV. Compare your fresh outputs to these to verify reproducibility. Numerical agreement should be exact under the tested software environment (see `requirements.txt`) and to the 3rd decimal under reasonable version drift.

## Citation

```
@unpublished{gamma_equalization_2026,
  author = {Authors blinded for review},
  title  = {Revenue-Constrained Multi-Product Pricing via $\gamma$-Equalization:
            A Diagonal Approximation Under MCI Demand},
  year   = {2026},
  note   = {Working paper}
}
```

## License

Code: MIT License. The JD MSOM 2020 dataset is distributed under the MSOM challenge terms; see the [challenge page](https://connect.informs.org/msom/events/datadriven-call).  
