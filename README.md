# ActivitySCOPE

**Activity-focused Survey-Completeness Observational Probability Estimator**

A machine learning method for identifying active asteroid and comet candidates via observational history anomalies in the asteroid orbit database.

## Paper Release

**Version 1.0.1 is the reference release for the paper.** Select `v1.0.1` from the tags menu (the branch/tag picker above the file list on GitHub). Enhancements made since the paper may be present in HEAD, so anyone looking to replicate the paper should use the tagged version. Also set `use_reproducibility_snapshot = True` near the top of the notebook to use the paper's snapshot of the orbit database (`orb_snapshot.parquet`) instead of downloading the latest orbit databases.

## Overview

ActivitySCOPE predicts how many oppositions an asteroid *should* have been observed on, given its orbit, its absolute magnitude, and the depth and duration of modern sky surveys. It then compares that prediction with the observed history. Objects observed on far fewer oppositions than expected are flagged as candidates. The hypothesis is that transient activity brightened them at discovery, so their cataloged $H_V$ is brighter than the quiescent body.

Three AutoGluon models (CPU-only gradient-boosted trees, 8-fold out-of-fold predictions) are trained on the MPC orbit database using 12 features:

1. **Poisson regression** → expected opposition count $E[N_{\rm opp}]$.
2. **Quantile regression** (0.006 quantile, pinball loss) → lower-tail opposition count $Q_{0.006}$. Also uses the out-of-fold $E[N_{\rm opp}]$ as a feature.
3. **Binary classifier** (log loss) → $P(N_{\rm opp} \ge 4)$. Also uses the out-of-fold $E[N_{\rm opp}]$ and $Q_{0.006}$ as features.

These give four anomaly metrics:

| Metric | More anomalous when | Best for |
|---|---|---|
| $P(N_{\rm opp} \ge 4)$ | higher | single-opposition objects |
| $\Delta Q = Q_{0.006} - N_{\rm opp}$ | higher | unified single- and multi-opposition ranking |
| $S_{\rm EV} = F_{\rm Pois}(N_{\rm opp}-1;\ E[N_{\rm opp}]-1)$ | lower | multi-opposition ACOs |
| $\Delta H$ | higher | robustness check against $H_V$ error |

$\Delta H$ is the smallest dimming of $H_V$ (in 0.05 mag steps, with features recomputed) for which $E[N_{\rm opp}] - N_{\rm opp} \le 0.5$. It is 0 if no step up to $H = 30$ reaches that condition, which `neutral_within_0p5 = False` marks. See `h_neutrality.py`.

A secondary classifier, `extension_difficulty`, flags likely mislinkages (chimera orbits) and orbits too uncertain to extend. It is used to filter candidate lists or to sort them.

## Paper ↔ Code Notation

Paper symbols and the column names used in `activityscope_utils.py`, `h_neutrality.py` and the notebooks.

| Paper | Code | Notes |
|---|---|---|
| $a$, $e$, $i$, $\Omega$, $\omega$, $M$ | `a`, `e`, `i`, `Node`, `Peri`, `M` | Keplerian elements (au, degrees) |
| $q$, $Q$ | `Perihelion_dist`, `Aphelion_dist` | Also `q` / `Q` in the output tables |
| $T_J$ | `TJ` | Tisserand parameter with respect to Jupiter |
| $H_V$ | `H` | Median of MPC/AstDyS/JPL when all three exist, otherwise the dimmest. Per-catalog values: `H_MPC`, `H_astdys`, `H_jpl` |
| $N_{\rm opp}$ | `Num_opps` | Observed oppositions |
| $t_{\rm ref}$ | `REFERENCE_EPOCH_JD` | MPCORB standard epoch, pinned with `utils.set_reference_epoch` |
| 12 model features | `mlcols` | `a`, `i`, `Node`, `H`, `vis_q`, `Perihelion_direction_x_e`, `Perihelion_direction_y_e`, `days_since_last_opp`, `n_detectable`, `n_detectable_var`, `years_since_first_detectable`, `days_since_last_detectable` (same names as in the paper) |
| $\mathbb{1}[N_{\rm opp} \ge 4]$ | `Is_Past_Threshold` | Classifier label |
| $N_{\rm opp} - 1$ | `Num_opps_minus_one` | Label for both regressors (add 1 to predictions) |
| $E[N_{\rm opp}]$ | `exp_Num_opps` | Poisson regressor output |
| $Q_{0.006}$ | `quantile_Opps` | Quantile regressor output |
| $P(N_{\rm opp} \ge 4)$ | `prob` | Classifier output |
| $\Delta Q$ | `DeltaQ` | |
| $S_{\rm EV}$ | `poisson_cdf` | `S_EV` in the seed-sweep notebook's tables |
| $\Delta H$ | `H_change_to_neutral` | `DeltaH` in the seed-sweep notebook's tables. $H'$ = `H_neutral` |
| `extension_difficulty` | `extension_difficulty` | Mislinkage / too-uncertain-to-extend classifier (Appendix) |
| Priority Score | `priority_score` | `DeltaQ - 15.5 * extension_difficulty` |

Out-of-fold predictions are used for every object in the training set, and full-data predictions for everything else.

## Installation

Python 3.11 is used (3.10–3.12 are likely workable), with Jupyter or VS Code with the Jupyter extension.

```
conda create -n activityscope python=3.11
conda activate activityscope
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

(or `python -m venv .venv && source .venv/bin/activate` in place of conda). AutoGluon has its own OS/hardware-specific requirements; see the [AutoGluon Installation Guide](https://auto.gluon.ai/stable/install.html). No GPU is needed; about 16 GB of RAM and 15–60 minutes per run is typical.

## Usage

-   **`Start Here - ActivitySCOPE simplified demo.ipynb`**: the full pipeline. It loads the data, trains the three models and the extension-difficulty classifier, predicts for every asteroid, computes all four anomaly metrics, and outputs ranked candidate lists (unified, single-opposition, cometary, $a > 5.4$ au, and the comet file).
-   **`More Complicated Workbook - including paper-specific items.ipynb`**: the same pipeline plus various adhoc explorations (including a few unfinished ideas). Also includes some code that generates the tables and figures used in the paper.
-   **`Multi-run Table 3 - seed sweep.ipynb`**: retrains over several seeds to produce the paper's results table with run-to-run ± uncertainties.

Supporting modules: `activityscope_utils.py` (data loading, feature engineering, extension-difficulty classifier), `h_neutrality.py` ($\Delta H$ sweep), `paper_table.py` (LaTeX results table), plus plotting and tuning scripts (`shap_simple.py`, `quantile_tuner.py`).

## License

This software is released under the [MIT License](LICENSE). Copyright (c) 2026 Peter VanWylen.

The MIT License covers the original code in this repository. Several bundled
data files originate wholly or partly from third-party sources and remain subject
to those sources' own terms of use and acknowledgment requirements:

-   `allcometels.json.gz` and `comet_orbits/allcometels.json` — comet orbital elements from the Minor Planet Center (MPC).

Items derived using third-party data or software:

-   `astrometry_counter/astrometry_counts.json` — derived from MPC astrometry
    (observation counts computed from MPC80-format observations).
-   `comet_orbits/comet_els_fo.txt` — our own orbit solutions and inert H_V values, computed with
    [find_orb](https://www.projectpluto.com/find_orb.htm) (B. Gray, Project Pluto).

In addition, the code downloads the MPC orbit database (MPCORB), the AstDyS
catalogs, and the JPL small-body elements at runtime; these are referenced rather
than redistributed here. Please cite and acknowledge the MPC, AstDyS, JPL, and
find_orb as appropriate when using this data. The remaining `.csv`/`.json` files
(filter lists, overrides, and known-object lists) are original products
of this project.
