# ActivitySCOPE

**Activity Survey-Completeness Observational Probability Estimator**

A machine learning method for identifying active asteroid and comet candidates via orbital survey detection expectancy anomalies.

The paper release is labeled as "Paper Release, 1.0" and there may be changes since that release. If you wish to use the exact model from the paper release, pick the v1.0 tag from the Tag picker in GitHub (above). Furthermore set use_reproducibility_snapshot = True near the top of either Jupyter notebook if you wish to use the historical reproducibility snapshot from the paper instead of getting the latest alerts/flags.

## Overview

ActivitySCOPE is a catalog-level anomaly detection method that identifies active asteroid and comet candidates using features present in the typical asteroid orbit database. It operates on the premise that the aggregate history of asteroid surveys can be used to infer the expected observational history for any given orbit and absolute magnitude.

By predicting the expected number of oppositions any given asteroid is likely to have been observed (given the depth and duration of modern sky surveys) and comparing it to the actual observed history, ActivitySCOPE identifies objects that are "missing" or underobserved from the historical record. These anomalies often correspond to objects that are physically fainter than their calculated absolute magnitude suggests, and their absolute magnitude as calculated and cataloged in the orbit database is likely due to transient activity that increased their brightness one or more times during the observational arc.

## Key Features

-   **Survey Completeness Modeling:** Uses the collective history of sky surveys as an indirect sensor of activity.
-   **Machine Learning:** Employs AutoGluon to train models on the MPC asteroid orbit database.
    -   **Binary Classification Model:** Predicts the probability that an object should have been observed on at least 4 oppositions. High probabilities for single-opposition objects flag potential candidates.
    -   **Regression Model:** Predicts the expected count of observed oppositions to quantify the observational deficit.
-   **Mislinkage Detection:** Includes a secondary classifier to help filter out potential false linkages (chimera orbits) and objects with orbits too uncertain to extend easily.

## Installation

### Prerequisites

*   Python 3.11 is the one we use (but 3.10 - 3.12 are likely all workable)
*   Jupyter Notebook or compatible (we prefer VS Code with the Jupyter plugin)

### Dependencies

Install the required Python packages using the provided `requirements.txt`. It is recommended to create a dedicated environment first:

**Using conda:**
```
conda create -n activityscope python=3.11
conda activate activityscope
```

**Or using venv:**
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then install dependencies:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

*Note: AutoGluon has its own specific installation requirements depending on your OS and hardware (CPU/GPU). Please refer to the [AutoGluon Installation Guide](https://auto.gluon.ai/stable/install.html).*

## Usage

Start with the Jupyter Notebook `Start Here - ActivitySCOPE simplified demo.ipynb`. A second notebook, `More Complicated Workbook - including paper-specific items.ipynb`, contains additional explorations and some code used for tables and figures in the paper.

1.  **Data Preparation:** The notebook works with the MPC (Minor Planet Center) orbit database.
2.  **Run the Notebook:** Execute the cells in order. The notebook performs the following steps:
    -   Loads and filters the training data.
    -   Trains the AutoGluon regression and classification models (the binary classifier references the regression model's out-of-fold predictions).
    -   Generates predictions for every asteroid in the database.
    -   Trains the extension-difficulty (mislinkage) classifier on those predictions.
    -   Outputs ranked lists of candidates (Cometary vs. Asteroidal orbits).

## License

This software is released under the [MIT License](LICENSE). Copyright (c) 2026 Peter VanWylen.

The MIT License covers the original code in this repository. Several bundled
data files originate wholly or partly from third-party sources and remain subject
to those sources' own terms of use and acknowledgment requirements:

-   `comet_orbits/allcometels.json` — comet orbital elements from the Minor
    Planet Center (MPC).

Items derived using third-party data or software:

-   `astrometry_counter/astrometry_counts.json` — derived from MPC astrometry
    (observation counts computed from MPC80-format observations).
-   `comet_orbits/comet_els_fo.txt` — our own orbit solutions and inert H_V values, computed with
    [find_orb](https://www.projectpluto.com/find_orb.htm) (B. Gray, Project Pluto).

In addition, the code downloads the MPC orbit database (MPCORB), the AstDyS
catalogs, and the JPL small-body elements at runtime; these are referenced rather
than redistributed here. Please cite and acknowledge the MPC, AstDyS, JPL, and
find_orb as appropriate when using this data. The remaining `.csv`/`.json` files
(filter lists, overrides, and known-object lists) are original derived products
of this project.
