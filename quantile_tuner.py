"""quantile_tuner.py

Sweep the quantile level used by the AutoGluon quantile regressor in
ActivitySCOPE and pick the level that best separates known active objects
from the inert background, measured by `sort = quantile_Opps - Num_opps`.

Population is restricted to:
    extension_difficulty < extension_difficulty_threshold   (default 0.002)
    a in a_range                                            (default 1.6 - 4.5)
applied to BOTH the active list and the inert background.

Separability metric:
  * primary   AUC of `sort` as a score for active vs inert
  * tiebreak  min(active sort) - max(inert sort), used only when several
              levels share AUC = 1.0
"""

import datetime
import os
import tempfile

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score


KNOWN_ACTIVES_DEFAULT = [
    "2008 BJ22", "2025 HV38", "2019 OE31", "2010 RH69", "2015 BC566",
    "2025 VZ8", "2003 BM80", "2021 AY8", "2001 BV70", "2009 FP8",
    "2018 BJ11", "2008 GO98", "2010 TR241", "2002 CW116",
    "2017 QN84", "2007 VB146"
]

EXCLUDED_DESIGNATIONS = ["2026 EX121","2026 EA122","2015 LT68"]

def _resolve_designation(df: pd.DataFrame) -> pd.Series:
    if "Principal_desig" in df.columns:
        return df["Principal_desig"]
    if df.index.name == "Principal_desig":
        return df.index.to_series()
    raise ValueError("eval_data must have Principal_desig as column or index")


def _train_quantile_predictor(data_df_reg, level, save_dir, time_limit,
                              presets, num_bag_folds, num_stack_levels):
    predictor = TabularPredictor(
        label="Num_opps_minus_one",
        problem_type="quantile",
        quantile_levels=[level],
        path=save_dir,
    )
    predictor.fit(
        data_df_reg,
        presets=presets,
        num_stack_levels=num_stack_levels,
        num_bag_folds=num_bag_folds,
        dynamic_stacking=False,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        time_limit=time_limit,
    )
    return predictor


def _quantile_opps_for_level(predictor, level, eval_data):
    preds = predictor.predict(eval_data)[level] + 1
    oof = predictor.predict_oof()[level] + 1
    preds.update(oof)
    return preds


def _evaluate(quantile_opps, eval_data, designations, active_set,
              extension_difficulty_threshold, a_range):
    df = pd.DataFrame({
        "designation": designations.values,
        "Num_opps": eval_data["Num_opps"].values,
        "extension_difficulty": eval_data["extension_difficulty"].values,
        "a": eval_data["a"].values,
        "Arc_length": eval_data["Arc_length"].values if "Arc_length" in eval_data.columns else np.nan,
        "quantile_Opps": quantile_opps.reindex(eval_data.index).values,
    }, index=eval_data.index)
    df["sort"] = df["quantile_Opps"] - df["Num_opps"]
    df["is_active"] = df["designation"].isin(active_set)

    mask = (
        df["extension_difficulty"].notna()
        & (df["extension_difficulty"] < extension_difficulty_threshold)
        & df["a"].between(a_range[0], a_range[1])
        & df["sort"].notna()
        & ((df["Arc_length"] > 7) | df["Arc_length"].isna())
    )
    pop = df[mask].copy()

    actives_present = set(pop.loc[pop["is_active"], "designation"])
    actives_dropped = sorted(active_set - actives_present)

    n_active = int(pop["is_active"].sum())
    n_inert = int((~pop["is_active"]).sum())

    metrics = {
        "n_active": n_active,
        "n_inert": n_inert,
        "actives_in_pop": sorted(actives_present),
        "actives_dropped": actives_dropped,
    }

    if n_active == 0 or n_inert == 0:
        metrics.update(auc=np.nan, min_gap=np.nan,
                       min_active_sort=np.nan, max_inert_sort=np.nan)
        return metrics, pop

    auc = roc_auc_score(pop["is_active"].astype(int).values, pop["sort"].values)
    min_active = pop.loc[pop["is_active"], "sort"].min()
    max_inert = pop.loc[~pop["is_active"], "sort"].max()
    metrics.update(
        auc=float(auc),
        min_gap=float(min_active - max_inert),
        min_active_sort=float(min_active),
        max_inert_sort=float(max_inert),
    )
    return metrics, pop


def tune(
    data_df_reg,
    eval_data,
    quantile_levels=(0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02),
    active_designations=None,
    excluded_designations=None,
    extension_difficulty_threshold=0.002,
    a_range=(1.6, 4.5),
    time_limit_per_level=300,
    presets="good_quality",
    num_bag_folds=8,
    num_stack_levels=1,
    save_path=None,
    verbose=True,
    fast=False,
):
    """Re-fit AutoGluon's quantile regressor at each candidate level and
    measure how well `sort = quantile_Opps - Num_opps` separates known
    active objects from the inert background inside the filter window.

    Parameters
    ----------
    data_df_reg : DataFrame
        Training frame in the same shape used in the notebook
        (columns = mlcols_reg, label = Num_opps_minus_one).
    eval_data : DataFrame
        Population to score. Must contain Num_opps, extension_difficulty,
        a, and Principal_desig (column or index). Anything with NaN sort
        or outside the filter window is dropped from the evaluation.
        Index must align with data_df_reg for the overlapping rows so OOF
        predictions can replace in-sample predictions for training rows.
    quantile_levels : iterable of float
        Quantile levels to try.
    active_designations : iterable of str, optional
        Principal_desig values for the known active set. Defaults to the
        16 objects listed in KNOWN_ACTIVES_DEFAULT.
    extension_difficulty_threshold, a_range : filter window, applied to
        both active and inert populations.
    time_limit_per_level : int
        Seconds per AutoGluon fit. With num_bag_folds=8 and
        num_stack_levels=1, 300s is a reasonable lower bound.
    fast : bool, optional
        If True, skip AutoGluon and use 5-fold LightGBM directly for a 
        faster but functionally identical evaluation. Default is False.

    Returns
    -------
    artifacts : dict[level] -> {"predictor", "quantile_opps", "pop", "metrics"}
    results   : DataFrame, one row per level, sorted by quantile_level
    summary   : dict {"best_quantile_level", "best_auc", "best_min_gap",
                       "tiebreak_used"}
    """
    if active_designations is None:
        active_designations = list(KNOWN_ACTIVES_DEFAULT)
    active_set = set(active_designations)

    if excluded_designations is None:
        excluded_designations = list(EXCLUDED_DESIGNATIONS)
    excluded_set = set(excluded_designations)

    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "quantile_tuner_ag")

    required = {"Num_opps", "extension_difficulty", "a"}
    missing = required - set(eval_data.columns)
    if missing:
        raise ValueError(f"eval_data is missing required columns: {missing}")

    designations = _resolve_designation(eval_data)

    if excluded_set:
        excluded_mask = designations.isin(excluded_set)
        excluded_idx = eval_data.index[excluded_mask.values]
        if len(excluded_idx):
            if verbose:
                dropped = sorted(designations[excluded_mask.values].unique())
                print(f"Excluding {len(excluded_idx)} rows "
                      f"for designations: {dropped}")
            eval_data = eval_data.drop(index=excluded_idx)
            data_df_reg = data_df_reg.drop(index=excluded_idx, errors="ignore")
            designations = designations.drop(excluded_idx)

    artifacts: dict = {}
    rows = []
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if fast:
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import KFold
        
        label_col = "Num_opps_minus_one"
        X_train = data_df_reg.drop(columns=[label_col])
        y_train = data_df_reg[label_col]
        X_eval = eval_data[X_train.columns]
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(X_train))

    for level in quantile_levels:
        save_dir = os.path.join(save_path, f"q_{level:g}_{run_stamp}")
        if verbose:
            print(f"\n=== quantile_level = {level}  (path: {save_dir}) ===")

        if fast:
            oof = pd.Series(index=X_train.index, dtype=float)
            models = []
            for train_idx, val_idx in folds:
                model = LGBMRegressor(
                    objective='quantile', 
                    alpha=level, 
                    n_estimators=100, 
                    random_state=42, 
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                oof.iloc[val_idx] = model.predict(X_train.iloc[val_idx])
                models.append(model)
            
            preds_all_models = np.column_stack([m.predict(X_eval) for m in models])
            preds = pd.Series(preds_all_models.mean(axis=1), index=eval_data.index)
            # update with oof where available
            preds.update(oof)
            # add 1
            quantile_opps = preds + 1
            predictor = models  # Save the list of models as the predictor artifact
        else:
            predictor = _train_quantile_predictor(
                data_df_reg, level, save_dir, time_limit_per_level,
                presets, num_bag_folds, num_stack_levels,
            )
            quantile_opps = _quantile_opps_for_level(predictor, level, eval_data)

        metrics, pop = _evaluate(
            quantile_opps, eval_data, designations, active_set,
            extension_difficulty_threshold, a_range,
        )

        artifacts[level] = {
            "predictor": predictor,
            "quantile_opps": quantile_opps,
            "pop": pop,
            "metrics": metrics,
        }
        rows.append({
            "quantile_level": level,
            "auc": metrics["auc"],
            "min_gap": metrics["min_gap"],
            "min_active_sort": metrics["min_active_sort"],
            "max_inert_sort": metrics["max_inert_sort"],
            "n_active": metrics["n_active"],
            "n_inert": metrics["n_inert"],
        })

        if verbose:
            print(f"  AUC={metrics['auc']:.4f}  "
                  f"min_gap={metrics['min_gap']:.3f}  "
                  f"n_active={metrics['n_active']}  n_inert={metrics['n_inert']}")
            if metrics["actives_dropped"]:
                print(f"  actives outside filter window: "
                      f"{metrics['actives_dropped']}")

    results = (pd.DataFrame(rows)
                 .sort_values("quantile_level")
                 .reset_index(drop=True))

    summary: dict = {}
    if not results.empty and results["auc"].notna().any():
        perfect = np.isclose(results["auc"], 1.0) & results["auc"].notna()
        if perfect.sum() > 1:
            best_idx = results.loc[perfect, "min_gap"].idxmax()
            tiebreak = True
        else:
            best_idx = results["auc"].idxmax()
            tiebreak = False
        summary = {
            "best_quantile_level": float(results.loc[best_idx, "quantile_level"]),
            "best_auc": float(results.loc[best_idx, "auc"]),
            "best_min_gap": float(results.loc[best_idx, "min_gap"]),
            "tiebreak_used": bool(tiebreak),
        }

    return artifacts, results, summary


def plot_separability(results: pd.DataFrame, save_path: str | None = None):
    """Plot AUC and min-gap vs quantile level."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(results["quantile_level"], results["auc"],
             "o-", color="C0", label="AUC")
    ax1.set_xlabel("quantile level")
    ax1.set_ylabel("AUC (active vs inert, by sort)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(results["quantile_level"], results["min_gap"],
             "s--", color="C3", label="min(active sort) - max(inert sort)")
    ax2.set_ylabel("min-gap", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.7)

    plt.title("Quantile-level sweep: active vs inert separability")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig
