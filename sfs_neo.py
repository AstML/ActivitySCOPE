"""sfs_neo.py

Sequential Feature Selection (SFS) restricted to the **NEO population**
(perihelion distance q = a(1 - e) < 1.3 AU).

Motivation
----------
The notebook's hand-picked `mlcols` were tuned against the whole catalog,
which is overwhelmingly main-belt. NEOs are a small, dynamically distinct
minority whose *discoverability* is governed by close Earth approaches,
encounter velocity, large solar phase angles and sun-glare geometry --
effects the main-belt-tuned feature set barely exercises. This script
re-runs feature selection on NEOs only, against a candidate pool that is

    (every numeric column present in `orb` after feature_engineering)
  + (a set of NEO-specific features engineered ONLY in this file).

It uses mlxtend's SequentialFeatureSelector (sequential floating selection
by default -- strictly stronger than the plain sklearn forward/backward
selector used elsewhere in the repo, because it can re-add or drop a
feature after a greedy step to escape local optima).

Novel NEO features engineered here (NOT in activityscope_utils.feature_engineering
and NOT in tune_new_features.py)
------------------------------------------------------------------------
 q                       Perihelion distance a(1-e). Defines the population
                         but still varies 0..1.3 within it; trees benefit
                         from it directly (it is NOT in orb -- AstDyS computes
                         a Perihelion_dist but it is never merged into orb).
 Q                       Aphelion distance a(1+e). Aten/Apollo/Amor split and
                         "how far out the faint part of the orbit reaches".
 opik_U                  Opik unperturbed encounter velocity wrt Earth, in
                         units of Earth's heliocentric speed:
                             U = sqrt(3 - 1/a - 2 sqrt(a(1-e^2)) cos i).
                         Governs how fast/brief/bright Earth encounters are;
                         the single most physical NEO-encounter scalar.
 earth_moid              Minimum orbit-intersection distance to Earth's orbit
                         (Earth idealised as a 1 AU circle in the ecliptic).
                         For a fixed asteroid point the closest Earth point is
                         analytic (sub-asteroid longitude), so MOID is the
                         min over true anomaly of sqrt(r^2 + 1 - 2 rho),
                         rho = sqrt(x^2 + y^2). The best apparition's distance.
 geo_dist_opp_typ        Time-weighted (Kepler 2nd law) median of the
                         opposition geocentric distance over the orbit. The
                         *typical* apparition distance, vs earth_moid's best.
 vis_at_moid             IAU (H,G=0.15) apparent V at the MOID encounter
                         geometry -- brightness of the best possible apparition.
 phase_at_moid           Solar phase angle (deg) at the MOID encounter. NEOs
                         can make their closest approaches at large phase
                         angles (dim, gibbous), which the symmetric vis_*
                         columns never see.
 sky_rate_moid           Sky-plane angular rate (deg/day) at the MOID
                         encounter ~ U * v_earth / Delta. Fast movers trail
                         and are harder to link -- the NEO detection penalty.
 frac_time_inside_1au    Time-weighted fraction of the orbit spent at r < 1 AU.
                         Separates Atens/Atiras (much time sunward, lost in
                         glare) from Amors (always exterior).
 frac_time_close         Time-weighted fraction of the orbit whose opposition
                         geocentric distance is < CLOSE_DELTA_AU (default 0.3).
                         How often the object presents a close, bright NEO
                         apparition rather than a distant faint one.

Candidate pool / NaN handling
-----------------------------
After the NEO cut and the usual quality filters, the candidate pool is every
numeric column that is not a target, identifier, cross-database QC column,
observation-count leakage column, or model output (see BLOCKLIST /
LEAK_SUBSTRINGS). Per the requested rule, any candidate column more than
10% empty *within the NEO subset* is dropped; rows are then dropna'd over
the surviving (>=90% filled) candidate columns.

Targets
-------
--target regression     (default) Num_opps_minus_one, XGB count:poisson,
                        scored by negative mean Poisson deviance. Matches the
                        repo's existing SFS / tune_new_features workflow.
--target classification Is_Past_Threshold, XGB binary:logistic, scored by
                        negative log loss. Matches the notebook's binary model.

Run (from the repo root):
    python sfs_neo.py [--target regression|classification] [--k-min K] [--k-max K]
                      [--cv F] [--direction forward|backward] [--no-floating]
                      [--subsample N] [--test-frac F] [--seed S]
                      [--max-nan-frac F] [--q-max Q]

Writes nothing to disk; prints the candidate pool, NaN report, the SFS
trajectory, the selected feature set, and a held-out comparison of
{SFS-selected, all-candidates, current-mlcols} feature sets.
"""

import argparse
import json

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as MLXSFS
from sklearn.metrics import (
    log_loss,
    mean_poisson_deviance,
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

import activityscope_utils as utils


EPS = 1e-3

# IAU HG phase function constants (G = 0.15), matching activityscope_utils.
HG_A1, HG_B1 = 3.33, 0.63
HG_A2, HG_B2 = 1.87, 1.22
HG_G = 0.15
PHI_FLOOR = 1e-30

# Earth heliocentric speed (AU / day): 2*pi AU per sidereal year.
V_EARTH_AU_PER_DAY = 2.0 * np.pi / 365.25

# True-anomaly samples per orbit for the orbit-averaged NEO features.
N_NU_SAMPLES = 96
# Row-chunk size to bound peak memory of the (chunk, N_NU) temporaries.
NEO_CHUNK_SIZE = 20_000
# Geocentric-distance threshold (AU) for frac_time_close.
CLOSE_DELTA_AU = 0.3

# The current notebook classification feature set (target stripped). Used only
# as a held-out baseline to compare the SFS-selected set against.
CURRENT_MLCOLS = [
    'Node', 'a', 'Perihelion_direction_y_e', 'vis_orbit_mag_multi',
       'vis_opp_mean', 'vis_q', 'vis_timeavg',
       'spatial_discoverability_fraction', 'vis_mid'
]

# The NEO features added by this file (so we can report how many survive SFS).
NEO_FEATURES = [
    "q", "Q", "opik_U", "earth_moid", "geo_dist_opp_typ", "vis_at_moid",
    "phase_at_moid", "sky_rate_moid", "frac_time_inside_1au", "frac_time_close",
]

# Columns that must never enter the candidate pool: targets, identifiers,
# cross-database QC fields, observation-count leakage, and model outputs.
BLOCKLIST = {
    # targets
    "Is_Past_Threshold", "Num_opps", "Num_opps_minus_one",
    # redundant with 'a' (already in the feature set): mean motion n and
    # orbital period are strictly monotonic functions of the semi-major axis
    # (Kepler's third law), so they carry no information beyond 'a'.
    "n", "Orbital_period",
    # arbitrary epoch / phase (used to BUILD geometry features, not raw inputs)
    "Epoch", "M", "Number",
    # cross-database QC
    "H_astdys", "H_jpl", "a_astdys", "e_astdys", "i_astdys",
    "H_diff_abs", "a_diff_abs", "e_diff_abs", "i_diff_abs",
    "H_diff_abs_jpl", "H_diff_abs_max", "multi_opp_disagree", "filtered_out",
    # observation / quality leakage (these effectively encode the answer)
    "Num_obs", "Arc_length", "U", "rms", "nights_total",
    "extension_difficulty",
    # model outputs / CV plumbing
    "exp_Num_opps", "quantile_Opps", "prob", "poisson_cdf", "DeltaQ",
    "Shared_Fold",
    # Other columns that should be ignored
    'G', 'H_MPC',
}

# Substring matches for the same leakage families (covers variants like
# nights_*, *astrometry*, n_obs_*, *_diff_abs).
LEAK_SUBSTRINGS = (
    "nights", "astrometry", "n_obs", "num_obs", "_diff_abs",
    "diff_abs", "filtered", "quantile", "extension_diff",
)


# ---------------------------------------------------------------------------
# Novel NEO feature engineering
# ---------------------------------------------------------------------------

def _phase_mag(tan_half):
    """IAU HG (G=0.15) phase-function magnitude penalty -2.5 log10(Phi)."""
    th = np.maximum(tan_half, 0.0)
    phi1 = np.exp(-HG_A1 * np.power(th, HG_B1))
    phi2 = np.exp(-HG_A2 * np.power(th, HG_B2))
    phi = np.maximum((1.0 - HG_G) * phi1 + HG_G * phi2, PHI_FLOOR)
    return -2.5 * np.log10(phi)


def _weighted_median(values, weights):
    """Row-wise weighted median of `values` (M, K) with `weights` (M, K)."""
    order = np.argsort(values, axis=1)
    v_sorted = np.take_along_axis(values, order, axis=1)
    w_sorted = np.take_along_axis(weights, order, axis=1)
    cw = np.cumsum(w_sorted, axis=1)
    cw = cw / np.maximum(cw[:, -1:], EPS)
    idx = np.argmax(cw >= 0.5, axis=1)
    return v_sorted[np.arange(v_sorted.shape[0]), idx]


def add_neo_features(orb):
    """Attach the NEO-specific candidate features to `orb` (in place).

    Assumes a, e, i, H, Node, Peri are present (MPC load) -- it does NOT
    require feature_engineering to have run first, but is happy if it has.
    """
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_rad = np.radians(orb["i"].to_numpy(dtype=np.float64))
    H = orb["H"].to_numpy(dtype=np.float64)
    Node = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri = np.radians(orb["Peri"].to_numpy(dtype=np.float64))

    # --- Closed-form scalars --------------------------------------------------
    q = a * (1.0 - e)
    Q = a * (1.0 + e)
    orb["q"] = q.astype(float)
    orb["Q"] = Q.astype(float)

    # Opik unperturbed encounter velocity wrt Earth (units of Earth's speed).
    # U^2 = 3 - 1/a - 2 sqrt(a (1 - e^2)) cos i. Clamp tiny negatives to 0.
    U2 = 3.0 - 1.0 / np.maximum(a, EPS) - 2.0 * np.sqrt(
        np.maximum(a * (1.0 - e ** 2), 0.0)
    ) * np.cos(i_rad)
    opik_U = np.sqrt(np.maximum(U2, 0.0))
    orb["opik_U"] = opik_U.astype(float)

    # --- Orbit-sampled close-approach geometry --------------------------------
    N = len(a)
    nu = np.linspace(0.0, 2.0 * np.pi, N_NU_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu)

    earth_moid = np.empty(N, dtype=np.float64)
    geo_dist_opp_typ = np.empty(N, dtype=np.float64)
    vis_at_moid = np.empty(N, dtype=np.float64)
    phase_at_moid = np.empty(N, dtype=np.float64)
    sky_rate_moid = np.empty(N, dtype=np.float64)
    frac_time_inside_1au = np.empty(N, dtype=np.float64)
    frac_time_close = np.empty(N, dtype=np.float64)

    for start in range(0, N, NEO_CHUNK_SIZE):
        end = min(start + NEO_CHUNK_SIZE, N)
        a_c = a[start:end, None]
        e_c = e[start:end, None]
        H_c = H[start:end, None]
        Node_c = Node[start:end, None]
        Peri_c = Peri[start:end, None]
        i_c = i_rad[start:end, None]
        U_c = opik_U[start:end]

        # Heliocentric distance and ecliptic position at each true anomaly.
        r = a_c * (1.0 - e_c ** 2) / (1.0 + e_c * cos_nu[None, :])
        r_safe = np.maximum(r, EPS)
        u = Peri_c + nu[None, :]
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_i, sin_i = np.cos(i_c), np.sin(i_c)
        cos_O, sin_O = np.cos(Node_c), np.sin(Node_c)
        x = r * (cos_O * cos_u - sin_O * sin_u * cos_i)
        y = r * (sin_O * cos_u + cos_O * sin_u * cos_i)
        z = r * sin_u * sin_i
        rho = np.sqrt(x * x + y * y)  # cylindrical radius

        # Opposition geocentric distance at each nu: the closest point on a
        # 1 AU circular ecliptic Earth orbit sits at the sub-asteroid longitude,
        # so d^2 = r^2 + 1 - 2 rho (minimised analytically over Earth phase).
        d = np.sqrt(np.maximum(r ** 2 + 1.0 - 2.0 * rho, EPS))

        # Kepler 2nd-law time weights (uniform-in-nu samples -> equal time).
        w = r ** 2
        w_sum = np.maximum(w.sum(axis=1), EPS)

        # earth_moid = best (minimum) opposition distance over the orbit.
        j_min = np.argmin(d, axis=1)
        rows = np.arange(end - start)
        earth_moid[start:end] = d[rows, j_min]

        # Typical opposition distance (time-weighted median over the orbit).
        geo_dist_opp_typ[start:end] = _weighted_median(d, w)

        # Geometry at the MOID encounter for brightness / phase / sky-rate.
        r_m = r[rows, j_min]
        rho_m = rho[rows, j_min]
        d_m = d[rows, j_min]
        d_m_safe = np.maximum(d_m, EPS)
        r_m_safe = np.maximum(r_m, EPS)
        # cos(phase) at asteroid vertex = (r^2 - rho) / (r * Delta)  (Earth on
        # the sub-asteroid longitude at unit distance).
        cos_alpha = np.clip((r_m ** 2 - rho_m) / (r_m_safe * d_m_safe), -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        phase_at_moid[start:end] = np.degrees(alpha)
        vis_at_moid[start:end] = (
            H[start:end]
            + 5.0 * np.log10(r_m_safe * d_m_safe)
            + _phase_mag(np.tan(alpha / 2.0))
        )
        # Sky-plane angular rate ~ relative speed / distance. Relative speed at
        # encounter ~ U * v_earth (Opik). deg/day.
        sky_rate_moid[start:end] = np.degrees(
            U_c * V_EARTH_AU_PER_DAY / d_m_safe
        )

        # Time-weighted fractions over the orbit.
        frac_time_inside_1au[start:end] = (
            ((r < 1.0) * w).sum(axis=1) / w_sum
        )
        frac_time_close[start:end] = (
            ((d < CLOSE_DELTA_AU) * w).sum(axis=1) / w_sum
        )

    orb["earth_moid"] = earth_moid.astype(float)
    orb["geo_dist_opp_typ"] = geo_dist_opp_typ.astype(float)
    orb["vis_at_moid"] = vis_at_moid.astype(float)
    orb["phase_at_moid"] = phase_at_moid.astype(float)
    orb["sky_rate_moid"] = sky_rate_moid.astype(float)
    orb["frac_time_inside_1au"] = frac_time_inside_1au.astype(float)
    orb["frac_time_close"] = frac_time_close.astype(float)

    return orb


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_neo_data(q_max=1.3, max_nan_frac=0.10, target="regression",
                     subsample=None, seed=0):
    """Load -> feature_engineering -> add NEO features -> NEO cut + quality
    filters -> build candidate pool (drop >max_nan_frac-empty cols) -> dropna.

    Returns (orb_neo, candidate_cols, target_col).
    """
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases()
    orb = orb[~orb["filtered_out"].astype(bool)]

    # NEO cut FIRST, on the cheap closed-form q = a(1-e). feature_engineering
    # and add_neo_features are both strictly row-independent (no cross-row
    # statistics), so applying the heavy orbit-sampling only to the small NEO
    # subset gives identical values at a fraction of the cost.
    q_full = orb["a"] * (1.0 - np.clip(orb["e"], 0.0, 0.999))
    orb = orb[q_full < q_max].copy()
    print(f"Rows after NEO cut (q < {q_max}): {len(orb)}")

    orb = utils.feature_engineering(orb)
    orb = add_neo_features(orb)

    print("Merging cached extension_difficulty.csv...")
    extension_difficulty = pd.read_csv("extension_difficulty.csv")
    orb = orb.merge(extension_difficulty, on="Principal_desig", how="left")

    with open("known_active_objects.json", "r") as f:
        known_active = json.load(f)
    with open("dual_designation_list.json", "r") as f:
        dual_designation_list = json.load(f)

    print(f"Rows before quality filter: {len(orb)}")
    # Same clean-training-set quality filters used by tune_new_features / the
    # notebook. These shape the training population; they are NOT features.
    orb = orb[
        ((orb["Arc_length"] >= 3) | orb["Arc_length"].isna())
        & (orb["Num_obs"] >= 16)
        & (orb["H_diff_abs_max"] < 0.4)
        & (orb["a_diff_abs"] < 0.0008)
        & (orb["e_diff_abs"] < 0.00025)
        & (orb["i_diff_abs"] < 0.004)
        & (orb["multi_opp_disagree"] == 0)
        & (orb["extension_difficulty"] < 0.25)
        # & (orb["U"] < 9)
        & ~orb["Principal_desig"].isin(known_active)
        & ~orb["Number"].isin(dual_designation_list)
    ]
    print(f"Rows after quality filter: {len(orb)}")

    target_col = "Num_opps_minus_one" if target == "regression" else "Is_Past_Threshold"

    # Candidate pool: numeric, not blocklisted, not a leakage-substring match.
    candidate_cols = []
    for c in orb.columns:
        if c in BLOCKLIST or c == target_col:
            continue
        if any(s in c.lower() for s in LEAK_SUBSTRINGS):
            continue
        if not pd.api.types.is_numeric_dtype(orb[c]):
            continue
        candidate_cols.append(c)

    # Drop candidate columns that are > max_nan_frac empty within the NEO subset.
    nan_frac = orb[candidate_cols].isna().mean()
    kept = [c for c in candidate_cols if nan_frac[c] <= max_nan_frac]
    dropped = [c for c in candidate_cols if nan_frac[c] > max_nan_frac]

    print(f"\nCandidate columns considered: {len(candidate_cols)}")
    print(f"  Kept (<= {max_nan_frac:.0%} NaN): {len(kept)}")
    if dropped:
        print(f"  Dropped (> {max_nan_frac:.0%} NaN):")
        for c in dropped:
            print(f"    {c:32s}  {nan_frac[c]:.1%} NaN")

    # dropna over the surviving (>= 90% filled) candidates + target.
    before = len(orb)
    orb = orb.dropna(subset=kept + [target_col]).reset_index(drop=True)
    print(f"\nRows after dropna over kept candidates + target: "
          f"{len(orb)} (dropped {before - len(orb)})")

    if subsample is not None and subsample < len(orb):
        orb = orb.sample(n=subsample, random_state=seed).reset_index(drop=True)
        print(f"Subsampled to {len(orb)} rows.")

    return orb, kept, target_col


# ---------------------------------------------------------------------------
# Estimator / scoring factory
# ---------------------------------------------------------------------------

def make_estimator(target, seed):
    """Return (estimator, sklearn_scoring_string) for the chosen target."""
    if target == "regression":
        est = XGBRegressor(
            objective="count:poisson",
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, verbosity=0,
        )
        return est, "neg_mean_poisson_deviance"
    est = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=1, verbosity=0,
    )
    return est, "neg_log_loss"


def evaluate_holdout(target, cols, X_tr, y_tr, X_te, y_te, seed):
    """Fit a fresh estimator on `cols` and report held-out metrics."""
    est, _ = make_estimator(target, seed)
    est.fit(X_tr[cols], y_tr)
    if target == "regression":
        pred = np.maximum(est.predict(X_te[cols]), EPS)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        pdev = float(mean_poisson_deviance(y_te, pred))
        return {"RMSE": rmse, "PoissonDev": pdev}
    proba = est.predict_proba(X_te[cols])[:, 1]
    return {
        "LogLoss": float(log_loss(y_te, proba)),
        "AUC": float(roc_auc_score(y_te, proba)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=["regression", "classification"],
                        default="regression",
                        help="Num_opps_minus_one (Poisson) or Is_Past_Threshold "
                             "(binary). Default: regression.")
    parser.add_argument("--q-max", type=float, default=1.3,
                        help="NEO perihelion-distance cut q < q_max (default 1.3).")
    parser.add_argument("--lock-mlcols", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Force the current notebook mlcols to be FIXED "
                             "(always in the model) via mlxtend fixed_features; "
                             "SFS then chooses additional features on top. "
                             "Default: on. Use --no-lock-mlcols for a free search.")
    parser.add_argument("--extra-min", type=int, default=1,
                        help="When locking mlcols: minimum number of ADDITIONAL "
                             "features to select beyond the locked set (>=1, "
                             "default 1).")
    parser.add_argument("--extra-max", type=int, default=8,
                        help="When locking mlcols: maximum number of ADDITIONAL "
                             "features to select beyond the locked set (default 8).")
    parser.add_argument("--k-min", type=int, default=4,
                        help="Minimum SFS subset size for a FREE search "
                             "(--no-lock-mlcols only; default 4).")
    parser.add_argument("--k-max", type=int, default=16,
                        help="Maximum SFS subset size for a FREE search "
                             "(--no-lock-mlcols only; default 16).")
    parser.add_argument("--direction", choices=["forward", "backward"],
                        default="forward",
                        help="SFS direction (default forward).")
    parser.add_argument("--no-floating", action="store_true",
                        help="Disable floating (plain SFS instead of SFFS/SFBS).")
    parser.add_argument("--cv", type=int, default=5, help="CV folds (default 5).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N NEO rows for faster iteration.")
    parser.add_argument("--test-frac", type=float, default=0.35,
                        help="Held-out test fraction (default 0.35).")
    parser.add_argument("--max-nan-frac", type=float, default=0.10,
                        help="Drop candidate cols more than this fraction empty "
                             "within the NEO subset (default 0.10).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sfs-jobs", type=int, default=-1,
                        help="n_jobs for the SFS subset search (default -1).")
    args = parser.parse_args()

    if args.k_max < args.k_min:
        parser.error("--k-max must be >= --k-min")
    if args.lock_mlcols and args.extra_min < 1:
        parser.error("--extra-min must be >= 1 (mlxtend needs k > #fixed).")
    if args.extra_max < args.extra_min:
        parser.error("--extra-max must be >= --extra-min")

    np.random.seed(args.seed)

    orb, candidate_cols, target_col = prepare_neo_data(
        q_max=args.q_max, max_nan_frac=args.max_nan_frac,
        target=args.target, subsample=args.subsample, seed=args.seed,
    )

    if len(orb) < 50:
        raise SystemExit(f"Too few NEO rows ({len(orb)}) after filtering.")

    n_neo_in_pool = sum(c in candidate_cols for c in NEO_FEATURES)
    print(f"\nTarget: {target_col}  ({args.target})")
    print(f"Candidate features ({len(candidate_cols)}), of which "
          f"{n_neo_in_pool}/{len(NEO_FEATURES)} are NEO-specific:")
    print(f"  {candidate_cols}")
    print(f"NEO features in pool: "
          f"{[c for c in NEO_FEATURES if c in candidate_cols]}")
    missing_mlcols = [c for c in CURRENT_MLCOLS if c not in candidate_cols]
    if missing_mlcols:
        print(f"[note] current mlcols not in candidate pool (NaN/blocked): "
              f"{missing_mlcols}")

    # Train / test split.
    X = orb[candidate_cols].astype(np.float32)
    y = orb[target_col].astype(np.float32 if args.target == "regression" else int)
    test_mask = np.zeros(len(orb), dtype=bool)
    rng = np.random.RandomState(args.seed)
    test_idx = rng.choice(len(orb), size=int(args.test_frac * len(orb)),
                          replace=False)
    test_mask[test_idx] = True
    X_tr, X_te = X[~test_mask], X[test_mask]
    y_tr, y_te = y[~test_mask].to_numpy(), y[test_mask].to_numpy()
    print(f"\nTrain rows: {len(X_tr)}   Test rows: {len(X_te)}")

    est, scoring = make_estimator(args.target, args.seed)
    floating = not args.no_floating

    # Locked ("non-negotiable") features: the current notebook mlcols that are
    # present in the candidate pool. mlxtend's fixed_features keeps them in every
    # evaluated subset; SFS only ever adds/removes the remaining candidates.
    if args.lock_mlcols:
        fixed = [c for c in CURRENT_MLCOLS if c in candidate_cols]
        n_extra_max = min(args.extra_max, len(candidate_cols) - len(fixed))
        n_extra_min = min(args.extra_min, n_extra_max)
        k_min = len(fixed) + n_extra_min
        k_max = len(fixed) + n_extra_max
        fixed_features = tuple(fixed)
        print(f"\nLocking {len(fixed)} mlcols as fixed (always in model): {fixed}")
        if len(fixed) < len(CURRENT_MLCOLS):
            print(f"[note] mlcols not lockable (absent from candidate pool): "
                  f"{[c for c in CURRENT_MLCOLS if c not in candidate_cols]}")
        print(f"SFS will choose {n_extra_min}..{n_extra_max} ADDITIONAL "
              f"features on top of them.")
    else:
        fixed_features = None
        k_max = min(args.k_max, len(candidate_cols))
        k_min = min(args.k_min, k_max)

    print(f"\nRunning mlxtend SequentialFeatureSelector "
          f"(direction={args.direction}, floating={floating}, "
          f"k=({k_min},{k_max}), cv={args.cv}, scoring={scoring}, "
          f"fixed={0 if fixed_features is None else len(fixed_features)})...")
    sfs = MLXSFS(
        est,
        k_features=(k_min, k_max),
        forward=(args.direction == "forward"),
        floating=floating,
        scoring=scoring,
        cv=args.cv,
        n_jobs=args.sfs_jobs,
        fixed_features=fixed_features,
        verbose=2,
    )
    # X_tr is a DataFrame, so mlxtend takes feature names from its columns
    # (do NOT pass custom_feature_names -- mlxtend forwards it to the estimator).
    sfs.fit(X_tr, y_tr)

    selected = list(sfs.k_feature_names_)
    locked = set(fixed_features) if fixed_features is not None else set()
    added = [c for c in selected if c not in locked]
    print("\n" + "=" * 78)
    print(f"SFS selected {len(selected)} features "
          f"(best CV {scoring} = {sfs.k_score_:.6f}):")
    print(f"  {selected}")
    if locked:
        print(f"  Locked mlcols ({len(locked)}): {[c for c in selected if c in locked]}")
        # NOTE: `added` is in feature-index order (mlxtend sorts k_feature_names_),
        # NOT the order SFS added them. See the addition-order trace below.
        print(f"  ADDED by SFS ({len(added)}, feature-index order): {added}")
    sel_neo = [c for c in selected if c in NEO_FEATURES]
    print(f"  NEO-specific among selected ({len(sel_neo)}): {sel_neo}")

    # Reconstruct the order in which SFS changed the subset, by diffing the
    # best subset at each consecutive size in sfs.subsets_. For plain forward
    # SFS this is exactly the addition order; with floating enabled a step can
    # both add and remove features, so we report both (locked features are
    # never touched, so they're suppressed from the trace).
    def _names(info):
        return set(info.get("feature_names", info["feature_idx"]))

    sizes = sorted(sfs.subsets_.keys())
    print("\nSubset change order (best subset at each size, diffed step to step):")
    prev = _names(sfs.subsets_[sizes[0]]) if sizes else set()
    print(f"  start k={sizes[0] if sizes else 0:2d}: "
          f"{sorted(prev - locked)} (+ {len(prev & locked)} locked)")
    for k in sizes[1:]:
        cur = _names(sfs.subsets_[k])
        gained = sorted((cur - prev) - locked)
        lost = sorted((prev - cur) - locked)
        change = []
        if gained:
            change.append(f"+{gained}")
        if lost:
            change.append(f"-{lost}")
        score = sfs.subsets_[k]["avg_score"]
        print(f"  k={k:2d}  {'  '.join(change) if change else '(no net change)'}"
              f"   cv_score={score:.6f}")
        prev = cur

    # Per-subset-size CV trajectory.
    print("\nCV score by subset size (mlxtend subsets_):")
    for k in sorted(sfs.subsets_.keys()):
        info = sfs.subsets_[k]
        names = info.get("feature_names", info["feature_idx"])
        print(f"  k={k:2d}  cv_score={info['avg_score']:.6f}  "
              f"+/-{np.std(info['cv_scores']):.6f}")

    # Held-out comparison: SFS-selected vs all candidates vs current mlcols.
    print("\n" + "=" * 78)
    print("Held-out test comparison (lower RMSE/PoissonDev/LogLoss is better; "
          "higher AUC is better):")
    avail_mlcols = [c for c in CURRENT_MLCOLS if c in candidate_cols]
    comparisons = {
        f"SFS-selected ({len(selected)})": selected,
        f"all candidates ({len(candidate_cols)})": candidate_cols,
        f"current mlcols ({len(avail_mlcols)})": avail_mlcols,
    }
    for label, cols in comparisons.items():
        if not cols:
            continue
        metrics = evaluate_holdout(args.target, cols, X_tr, y_tr, X_te, y_te,
                                   args.seed)
        metric_str = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        print(f"  {label:34s}  {metric_str}")
    print("=" * 78)


if __name__ == "__main__":
    main()
