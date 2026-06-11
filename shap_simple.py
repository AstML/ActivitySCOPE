"""SHAP scatter / mean-effect plots for the simplified paper model (vis_orbit_mag_multi, spatial_discoverability_fraction, Node).

Uses AutoGluon's TabularPredictor (Poisson regression, good_quality bagged ensemble)
for maximum accuracy, mirroring the main pipeline in ActivitySCOPE_simplified_demo.ipynb.
SHAP values are computed via shap.Explainer (model-agnostic PermutationExplainer) since
AutoGluon's WeightedEnsemble is not a single tree model.
"""
import os
import datetime
import tempfile
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

import activityscope_utils as utils


SIMPLE_FEATURES = ["vis_orbit_mag_multi", "spatial_discoverability_fraction", "Node"]
LABEL_COL = "Num_opps_minus_one"

FEATURE_LABELS_LATEX = {
    "vis_orbit_mag_multi": r"vis\_orbit\_mag\_multi",
    "spatial_discoverability_fraction": r"spatial\_discoverability\_fraction",
    "Node": r"$\Omega$",
}

FEATURE_LABELS_PLAIN = {
    "vis_orbit_mag_multi": "vis_orbit_mag_multi",
    "spatial_discoverability_fraction": "spatial_discoverability_fraction",
    "Node": r"$\Omega$",
}

FEATURE_XLIMS = {
    "vis_orbit_mag_multi": (17, 30.5),
    "Node": (0, 360),
}

FEATURE_XTICKS = {
    "vis_orbit_mag_multi": [20, 25, 30],
    "Node": [0, 90, 180, 270, 360],
}


def train_simple_model(orb_training, features=SIMPLE_FEATURES, label=LABEL_COL,
                       time_limit=600, save_path=None, presets="good_quality",
                       num_bag_folds=8):
    train_df = orb_training[list(features) + [label]].copy()
    for f in features:
        train_df[f] = train_df[f].astype(np.float32)

    if save_path is None:
        save_path = os.path.join(
            tempfile.gettempdir(),
            "shap_simple_ag",
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    predictor = TabularPredictor(
        label=label,
        eval_metric=utils.POISSON_SCORER,
        problem_type="regression",
        path=save_path,
    )
    predictor.fit(
        train_df,
        presets=presets,
        hyperparameters=utils.HYPERPARAMETERS_POISSON,
        num_stack_levels=0,
        num_bag_folds=num_bag_folds,
        dynamic_stacking=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        time_limit=time_limit,
    )
    return predictor


def compute_shap(predictor, orb_test, features=SIMPLE_FEATURES,
                 background_n=100, sample_n=2000, random_state=42):
    """Model-agnostic SHAP for the AutoGluon ensemble.

    PermutationExplainer evaluates predictor.predict on ~(2*F+1)*background_n rows
    per sample, so keep background_n and sample_n modest. With 4 features this is
    typically a few minutes on the WeightedEnsemble.
    """
    X_test = orb_test[list(features)].astype(np.float32).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    if len(X_test) > background_n:
        background = X_test.iloc[rng.choice(len(X_test), background_n, replace=False)].reset_index(drop=True)
    else:
        background = X_test
    if len(X_test) > sample_n:
        X_sampled = X_test.iloc[rng.choice(len(X_test), sample_n, replace=False)].reset_index(drop=True)
    else:
        X_sampled = X_test

    feat_list = list(features)

    def predict_fn(arr):
        return predictor.predict(pd.DataFrame(arr, columns=feat_list)).to_numpy()

    explainer = shap.Explainer(predict_fn, background)
    explanation = explainer(X_sampled)
    return X_sampled, explanation.values


def _plot_panel(X_test, shap_values, features, labels_map, sample_n=200000):
    """Render all features as a single 1xN panel with a shared y-axis.

    Sized for ~6.5in of usable width (8.5in paper with 1in margins) and a
    common SHAP-value axis spanning -10 to +15.
    """
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.7), sharey=True)
    if n == 1:
        axes = [axes]

    usetex = matplotlib.rcParams.get("text.usetex", False)
    for j, feature in enumerate(features):
        ax = axes[j]
        tag = f"({chr(ord('a') + j)})"
        tag = rf"\textbf{{{tag}}}" if usetex else tag
        ax.text(0.04, 0.96, tag, transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        x_data = X_test[feature].values
        y_data = shap_values[:, j]
        if len(x_data) > sample_n:
            sel = np.random.choice(len(x_data), sample_n, replace=False)
            x_data = x_data[sel]
            y_data = y_data[sel]

        ax.scatter(x_data, y_data, color="#1f77b4", s=10, alpha=0.5,
                   rasterized=True, edgecolors='none')
        ax.set_ylim(-10, 15)
        ax.set_yticks(np.arange(-10, 16, 5))
        xlim = FEATURE_XLIMS.get(feature)
        if xlim is not None:
            ax.set_xlim(*xlim)
        xticks = FEATURE_XTICKS.get(feature)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_xlabel(labels_map.get(feature, feature))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.8)

    axes[0].set_ylabel(r"SHAP Value")
    fig.tight_layout()
    return fig, axes


def plot_shap_display(X_test, shap_values, features=SIMPLE_FEATURES):
    fig, _ = _plot_panel(X_test, shap_values, features, FEATURE_LABELS_PLAIN)
    plt.show()
    return fig


def save_shap_pgf(X_test, shap_values, out_dir, features=SIMPLE_FEATURES, prefix="shap_simple"):
    os.makedirs(out_dir, exist_ok=True)
    pgf_rc = {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 11,
    }
    with matplotlib.rc_context(pgf_rc):
        fig, _ = _plot_panel(X_test, shap_values, features, FEATURE_LABELS_LATEX)
        out = os.path.join(out_dir, f"{prefix}_panel.pgf")
        fig.savefig(out, backend="pgf")
        plt.close(fig)
    return [out]


def run(orb_training, orb_test, features=SIMPLE_FEATURES, out_dir=None,
        time_limit=600, background_n=100, sample_n=2000):
    predictor = train_simple_model(orb_training, features=features, time_limit=time_limit)
    X_test, shap_values = compute_shap(
        predictor, orb_test, features=features,
        background_n=background_n, sample_n=sample_n,
    )
    plot_shap_display(X_test, shap_values, features=features)
    if out_dir is not None:
        saved = save_shap_pgf(X_test, shap_values, out_dir, features=features)
        print(f"Saved {len(saved)} PGF files to {out_dir}")
    return predictor, X_test, shap_values
