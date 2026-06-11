"""tune_elong_ramp.py

2-D grid search over the two solar-elongation endpoints of the observing-
efficiency ramp inside `vis_orbit_mag_multi` (the model's single most important
feature).

In activityscope_utils.feature_engineering, vis_orbit_mag_multi sweeps Earth
around the full synodic circle (N_E = 24 heliocentric longitudes) at each of 32
true-anomaly samples, and weights every (anomaly, Earth-longitude) geometry by
an observing-efficiency ramp on the solar elongation epsilon:

    obs_eff = clip( (epsilon - ELONG_MIN_DEG) / (ELONG_FULL_DEG - ELONG_MIN_DEG), 0, 1 )

i.e. zero below ELONG_MIN_DEG (surveys avoid the Sun), ramping to full at
ELONG_FULL_DEG. The feature is then

    mag_when_obs = sum(V * w_kepler * obs_eff) / sum(w_kepler * obs_eff)   (gated mean V)
                   [fallback to the ungated full-grid mean when nothing clears the gate]
    duty         = sum(w_kepler * obs_eff) / (N_E * sum(w_kepler))
    vis_orbit_mag_multi = mag_when_obs - 2.5 log10( max(duty, DUTY_FLOOR) )

ELONG_MIN_DEG = 60 and ELONG_FULL_DEG = 90 are physically motivated but
otherwise hand-picked and untuned. This script finds the (epsilon_min,
epsilon_full) pair that makes the resulting feature most useful to the
production opposition-count regressor.

Efficiency. The endpoints only enter through obs_eff, so everything else is
precomputed ONCE: for every (object, anomaly, Earth-longitude) cell we store
the apparent magnitude V and the solar elongation, plus the per-object Kepler
weights and the (endpoint-independent) ungated magnitude sum. Each candidate
(epsilon_min, epsilon_full) is then a handful of vectorised reductions over
those stored cells -- no re-derivation of the orbital geometry.

  MEMORY: the per-cell arrays are N x 32 x 24 float32 each (V and elongation),
  ~6 KB per object. With the default stratified set (~all interior + 50k
  exterior, so tens of thousands of rows) this is only a few hundred MB; the
  script prints the estimate before allocating. Raise --exterior-sample to
  trade memory for a larger exterior sample.

The fixed baseline feature set is the notebook's production regression feature
list (mlcols_reg) with vis_orbit_mag_multi swapped for the recomputed column.
Each candidate is scored with k-fold CV blending a LightGBM and an XGBoost
Poisson regressor, reporting fold-averaged RMSE, Poisson deviance, and R^2.
The incumbent production endpoints (60, 90) are always evaluated and printed; a
candidate is flagged a winner if it beats the incumbent on >= 2 of the 3
metrics.

STRATIFIED SAMPLING. The elongation gate barely moves vis_orbit_mag_multi for
exterior (main-belt) objects -- their observable window is near opposition
(elongation ~180 deg), which clears any gate from 30-130 deg identically -- so
in a natural population sample (where interior objects are ~1%) their signal is
drowned out. Instead of a full random subsample, we build a stratified set:
keep ALL interior/low-q objects (q = a(1-e) < --strat-q-max, default 1.5 AU)
and cap the abundant exterior population at --exterior-sample (default 50000).
CV then trains and scores over this combined set, covering every dynamical
class while structurally over-weighting NEOs so the metric is sensitive to the
objects the gate is meant to help. (Optionally, --score-q-max additionally
restricts the metric to interior test rows; by default the whole set is scored.
--strat-q-max <= 0 disables stratification.)

Run:
    python tune_elong_ramp.py [--min-lo 30] [--min-hi 75]
                              [--full-lo 75] [--full-hi 130]
                              [--min-steps 5] [--full-steps 5]
                              [--strat-q-max 1.5] [--exterior-sample 50000]
                              [--seed 0] [--n-folds 8]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# This tuner lives in modeling/parameter tuners/ but imports the repo-root
# activityscope_utils module and reads data files (CSVs, JSON, the MPCORB
# cache) by repo-root-relative paths. Put the repo root on sys.path and make
# it the working directory so both resolve regardless of the launch directory.
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import activityscope_utils as utils  # noqa: E402  (after sys.path bootstrap)


# ---------------------------------------------------------------------------
# Constants (kept in lock-step with activityscope_utils.feature_engineering)
# ---------------------------------------------------------------------------

EPS = 1e-3

# The production endpoints being tuned (deg). Always evaluated for comparison.
INCUMBENT_EMIN = 52
INCUMBENT_EFULL = 160

# Orbit-sampling / photometry constants -- copied verbatim from
# feature_engineering so the recomputed feature matches the notebook bit for
# bit at the incumbent (60, 90).
N_ANOMALY_SAMPLES = 32
N_EARTH_OFFSETS = 24  # full synodic circle, linspace(0, 360, 24, endpoint=False)
ANOMALY_CHUNK_SIZE = 100_000
HG_A1, HG_B1 = 3.33, 0.63
HG_A2, HG_B2 = 1.87, 1.22
HG_G = 0.15
PHI_FLOOR = 1e-30
DUTY_FLOOR = 1e-3

# Production regression feature list (notebook mlcols_reg, target dropped).
# vis_orbit_mag_multi is REPLACED per candidate, so it is held in its own
# constant and excluded from the fixed baseline.
TUNED_COL = "vis_orbit_mag_multi"
BASELINE_COLS = [
    "H", "Node", "a", "i",
    "Perihelion_direction_x_e", "Perihelion_direction_y_e",
    "dec_flux_weighted", "vis_opp_mean",
    "e", "vis_q", "vis_timeavg", "vis_inc", "orbital_period_sync",
    "spatial_discoverability_fraction",
]
TARGET = "Num_opps_minus_one"


# ---------------------------------------------------------------------------
# Precompute the (epsilon_min, epsilon_full)-independent per-cell arrays
# ---------------------------------------------------------------------------

def precompute_cells(orb):
    """Replicate feature_engineering's full-synodic-circle sampling and return
    the pieces of vis_orbit_mag_multi that do NOT depend on the elongation ramp:

        V_cell     (N, n_nu, n_off) float32 : apparent V at each cell
        elong_cell (N, n_nu, n_off) float32 : solar elongation (deg) at each cell
        weights    (N, n_nu)        float32 : Kepler weight w_nu = r_nu^2
        w_sum      (N,)             float64 : sum of w_nu per object
        num_all    (N,)             float64 : ungated sum_{nu,off}(V * w_nu)
                                              (fallback numerator; ramp-independent)
    """
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_rad = np.radians(orb["i"].to_numpy(dtype=np.float64))
    H = orb["H"].to_numpy(dtype=np.float64)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))

    nu_arr = np.linspace(0.0, 2.0 * np.pi, N_ANOMALY_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu_arr)
    offsets = np.radians(np.linspace(0.0, 360.0, N_EARTH_OFFSETS, endpoint=False))
    eps_val = EPS

    N = len(a)
    n_nu = N_ANOMALY_SAMPLES
    n_off = N_EARTH_OFFSETS
    V_cell = np.empty((N, n_nu, n_off), dtype=np.float32)
    elong_cell = np.empty((N, n_nu, n_off), dtype=np.float32)
    weights_out = np.empty((N, n_nu), dtype=np.float32)
    w_sum = np.empty(N, dtype=np.float64)
    num_all = np.empty(N, dtype=np.float64)

    for start in range(0, N, ANOMALY_CHUNK_SIZE):
        end = min(start + ANOMALY_CHUNK_SIZE, N)
        a_c = a[start:end, None]
        e_c = e[start:end, None]
        Node_c = Node_rad[start:end, None]
        Peri_c = Peri_rad[start:end, None]
        i_c = i_rad[start:end, None]
        H_c = H[start:end, None]

        # Heliocentric distance and Kepler-2nd-law weights at each anomaly.
        r_orb = a_c * (1.0 - e_c ** 2) / (1.0 + e_c * cos_nu[None, :])
        r_safe = np.maximum(r_orb, eps_val)
        weights = r_orb ** 2
        w_sum[start:end] = np.maximum(weights.sum(axis=1), eps_val)
        weights_out[start:end] = weights.astype(np.float32)

        # Heliocentric ecliptic coordinates of the asteroid at each nu.
        u = Peri_c + nu_arr[None, :]
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        cos_i = np.cos(i_c)
        sin_i_arr = np.sin(i_c)
        cos_Node = np.cos(Node_c)
        sin_Node = np.sin(Node_c)
        x_ecl = r_orb * (cos_Node * cos_u - sin_Node * sin_u * cos_i)
        y_ecl = r_orb * (sin_Node * cos_u + cos_Node * sin_u * cos_i)
        z_ecl = r_orb * sin_u * sin_i_arr
        lambda_k = np.arctan2(y_ecl, x_ecl)

        num_all_chunk = np.zeros(end - start, dtype=np.float64)
        for j, offset_rad in enumerate(offsets):
            lambda_E = lambda_k + offset_rad
            cos_E = np.cos(lambda_E)
            sin_E = np.sin(lambda_E)
            dx = x_ecl - cos_E
            dy = y_ecl - sin_E
            # Earth z = 0, so dz = z_ecl (unchanged across Earth offsets)
            Delta = np.sqrt(dx * dx + dy * dy + z_ecl * z_ecl)
            Delta_safe = np.maximum(Delta, eps_val)
            cos_alpha = np.clip(
                (r_orb ** 2 + Delta ** 2 - 1.0) / (2.0 * r_safe * Delta_safe),
                -1.0, 1.0,
            )
            alpha = np.arccos(cos_alpha)
            tan_half = np.maximum(np.tan(alpha / 2.0), 0.0)
            phi1 = np.exp(-HG_A1 * np.power(tan_half, HG_B1))
            phi2 = np.exp(-HG_A2 * np.power(tan_half, HG_B2))
            phi_blend = np.maximum((1.0 - HG_G) * phi1 + HG_G * phi2, PHI_FLOOR)
            V_n = (H_c
                   + 5.0 * np.log10(r_safe * Delta_safe)
                   - 2.5 * np.log10(phi_blend))

            # Solar elongation: angle Sun-Earth-asteroid as seen from Earth.
            cos_elong = np.clip(
                (-cos_E * dx - sin_E * dy) / Delta_safe, -1.0, 1.0
            )
            elong_deg = np.degrees(np.arccos(cos_elong))

            V_cell[start:end, :, j] = V_n.astype(np.float32)
            elong_cell[start:end, :, j] = elong_deg.astype(np.float32)
            num_all_chunk += (V_n * weights).sum(axis=1)

        num_all[start:end] = num_all_chunk

    return V_cell, elong_cell, weights_out, w_sum, num_all


def vis_orbit_mag_multi_for(V_cell, elong_cell, weights, w_sum, num_all,
                            emin, efull, row_chunk=ANOMALY_CHUNK_SIZE):
    """Recompute vis_orbit_mag_multi under one (epsilon_min, epsilon_full).
    Chunked over rows so the (chunk, n_nu, n_off) temporaries stay small."""
    N = V_cell.shape[0]
    n_off = V_cell.shape[2]
    width = efull - emin
    out = np.empty(N, dtype=np.float64)
    for s in range(0, N, row_chunk):
        e = min(s + row_chunk, N)
        elong = elong_cell[s:e].astype(np.float64)
        Vc = V_cell[s:e].astype(np.float64)
        wc_nu = weights[s:e].astype(np.float64)[:, :, None]  # (chunk, n_nu, 1)
        if width <= 0.0:
            ramp = (elong >= emin).astype(np.float64)  # degenerate -> hard step
        else:
            ramp = np.clip((elong - emin) / width, 0.0, 1.0)
        w_cell = wc_nu * ramp
        den = w_cell.sum(axis=(1, 2))
        num = (Vc * w_cell).sum(axis=(1, 2))
        total_weight = np.maximum(w_sum[s:e] * n_off, EPS)
        mean_all = num_all[s:e] / total_weight
        mag_when_obs = np.where(den > EPS, num / np.maximum(den, EPS), mean_all)
        duty = den / total_weight
        penalty = -2.5 * np.log10(np.maximum(duty, DUTY_FLOOR))
        out[s:e] = mag_when_obs + penalty
    return out


# ---------------------------------------------------------------------------
# Evaluation: k-fold CV blending LightGBM + XGBoost Poisson regressors
# (identical evaluator to the other tuners)
# ---------------------------------------------------------------------------

def _fit_predict_xgb(X_tr, y_tr, X_te, seed):
    model = XGBRegressor(objective="count:poisson", random_state=seed,
                         n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=np.float64)


def _fit_predict_lgbm(X_tr, y_tr, X_te, seed):
    model = LGBMRegressor(objective="poisson", random_state=seed,
                          n_jobs=-1, verbosity=-1)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=np.float64)


def evaluate_cv(X, y, folds, seed, score_mask=None):
    """Train on the full training fold; score on the test fold. If score_mask
    (a length-N boolean over orb's row order) is given, the RMSE/Poisson/R^2 are
    computed ONLY on the test rows where the mask is True -- i.e. we train on the
    whole population (as deployed) but measure on a subpopulation. Folds with
    fewer than 2 in-mask test rows are skipped for the metric averages."""
    rmses, pdevs, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        p_xgb = _fit_predict_xgb(X_tr, y_tr, X_te, seed)
        p_lgb = _fit_predict_lgbm(X_tr, y_tr, X_te, seed)
        y_pred = np.maximum(0.5 * (p_xgb + p_lgb), EPS)
        if score_mask is not None:
            m = score_mask[test_idx]
            if m.sum() < 2:
                continue
            y_te, y_pred = y_te[m], y_pred[m]
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        pdevs.append(mean_poisson_deviance(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    if not rmses:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(rmses)), float(np.mean(pdevs)), float(np.mean(r2s))


def score_ramp(orb_base, y, folds, seed, V_cell, elong_cell, weights, w_sum,
               num_all, emin, efull, score_mask=None):
    """Build the feature matrix for one (epsilon_min, epsilon_full) and CV-score it."""
    X = orb_base.copy()
    X[TUNED_COL] = vis_orbit_mag_multi_for(
        V_cell, elong_cell, weights, w_sum, num_all, emin, efull
    ).astype(np.float32)
    return evaluate_cv(X.astype(np.float32), y, folds, seed, score_mask=score_mask)


# ---------------------------------------------------------------------------
# Data preparation (mirrors the other tuners / the notebook training filter)
# ---------------------------------------------------------------------------

def prepare_training_data(strat_q_max=1.5, exterior_sample=50000, seed=0):
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases()
    orb = orb[~orb["filtered_out"].astype(bool)]
    orb = utils.feature_engineering(orb)

    print("Merging cached extension_difficulty.csv...")
    extension_difficulty = pd.read_csv("extension_difficulty.csv")
    orb = orb.merge(extension_difficulty, on="Principal_desig", how="left")

    with open("known_active_objects.json", "r") as f:
        known_active = json.load(f)
    with open("dual_designation_list.json", "r") as f:
        dual_designation = json.load(f)

    print(f"Pre-filter row count: {len(orb)}")
    orb = orb[
        ((orb["Arc_length"] >= 20) | orb["Arc_length"].isna())
        & (orb["Num_obs"] >= 16)
        & (orb["H_diff_abs_max"] < 0.3)
        & (orb["a_diff_abs"] < 0.0005)
        & (orb["e_diff_abs"] < 0.00015)
        & (orb["i_diff_abs"] < 0.003)
        & (orb["multi_opp_disagree"] == 0)
        & (orb["extension_difficulty"] < 0.1)
        & (orb["U"] < 9)
        & ~orb["Principal_desig"].isin(known_active)
        & ~orb["Number"].isin(dual_designation)
    ]
    # vis_orbit_mag_multi is recomputed from a/e/i/Node/Peri/H, so require those.
    needed = BASELINE_COLS + [TARGET, "Peri"]
    orb = orb.dropna(subset=needed)
    print(f"Post-filter row count: {len(orb)}")

    # Stratified subsample: keep ALL interior/low-q objects (q < strat_q_max),
    # where the elongation gate actually moves vis_orbit_mag_multi, and cap the
    # abundant exterior population at exterior_sample. This evaluates over every
    # dynamical class while structurally over-weighting NEOs relative to their
    # ~1% natural share, so the metric is sensitive to the interior objects the
    # gate is meant to help. (strat_q_max <= 0 disables stratification, making
    # this a plain random sample of exterior_sample rows.)
    q = orb["a"].to_numpy(dtype=np.float64) * (1.0 - orb["e"].to_numpy(dtype=np.float64))
    interior = orb[q < strat_q_max]
    exterior = orb[q >= strat_q_max]
    if exterior_sample is not None and exterior_sample < len(exterior):
        exterior = exterior.sample(n=exterior_sample, random_state=seed)
    orb = pd.concat([interior, exterior])
    print(f"Stratified subsample (q split at {strat_q_max} AU): "
          f"{len(interior)} interior (all kept) + {len(exterior)} exterior "
          f"(of {len(q[q >= strat_q_max])}) = {len(orb)} rows")

    return orb.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-lo", type=float, default=48.0,
                        help="Lowest epsilon_min, deg (default: 45).")
    parser.add_argument("--min-hi", type=float, default=90.0,
                        help="Highest epsilon_min, deg (default: 70).")
    parser.add_argument("--full-lo", type=float, default=135.0,
                        help="Lowest epsilon_full, deg (default: 100).")
    parser.add_argument("--full-hi", type=float, default=215.0,
                        help="Highest epsilon_full, deg (default: 170).")
    parser.add_argument("--min-steps", type=int, default=5,
                        help="Grid points along the epsilon_min axis (default: 5).")
    parser.add_argument("--full-steps", type=int, default=5,
                        help="Grid points along the epsilon_full axis (default: 5).")
    parser.add_argument("--strat-q-max", type=float, default=1.5,
                        help="Perihelion q = a(1-e) split (AU) for stratified "
                             "subsampling: ALL rows below this are kept (interior/"
                             "NEO), the rest are capped at --exterior-sample. "
                             "Set <= 0 to disable stratification (default: 1.5).")
    parser.add_argument("--exterior-sample", type=int, default=170000,
                        help="Cap on exterior (q >= --strat-q-max) rows kept "
                             "(default: 120000). Set very large to keep all.")
    parser.add_argument("--score-q-max", type=float, default=None,
                        help="If set, score each fold ONLY on test rows with "
                             "q < this (AU). By default the metric is computed "
                             "over the whole stratified set, since the "
                             "oversampling already prioritizes interior objects.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV folds, subsample, and "
                             "model fits (default: 0).")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="CV folds per candidate (default: 8).")
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")
    if args.min_steps < 1 or args.full_steps < 1:
        parser.error("--min-steps and --full-steps must be >= 1.")
    if args.min_hi < args.min_lo:
        parser.error("--min-hi must be >= --min-lo.")
    if args.full_hi < args.full_lo:
        parser.error("--full-hi must be >= --full-lo.")

    np.random.seed(args.seed)

    print(f"Deterministic seed (folds + model fits): {args.seed}")
    print(f"CV: {args.n_folds}-fold, blending LightGBM + XGBoost (Poisson)")
    print(f"Tuning: solar-elongation efficiency ramp inside {TUNED_COL}")
    print(f"  epsilon_min  in [{args.min_lo}, {args.min_hi}] deg "
          f"({args.min_steps} pts)")
    print(f"  epsilon_full in [{args.full_lo}, {args.full_hi}] deg "
          f"({args.full_steps} pts)  => up to "
          f"{args.min_steps * args.full_steps} candidates")
    print(f"  obs_eff = clip((epsilon - epsilon_min)/(epsilon_full - epsilon_min), 0, 1); "
          f"incumbent = ({INCUMBENT_EMIN}, {INCUMBENT_EFULL})")
    print(f"  stratified sample: all q < {args.strat_q_max} AU + "
          f"<= {args.exterior_sample} of q >= {args.strat_q_max} AU")
    print(f"  scoring: " + (f"TEST rows with q < {args.score_q_max} AU only"
                            if args.score_q_max is not None
                            else "all rows in the stratified set"))
    print(f"Fixed baseline features ({len(BASELINE_COLS)}): {BASELINE_COLS}")

    orb = prepare_training_data(strat_q_max=args.strat_q_max,
                                exterior_sample=args.exterior_sample,
                                seed=args.seed)
    y = orb[TARGET].astype(np.float32).to_numpy()
    orb_base = orb[BASELINE_COLS].copy()

    # Perihelion distance per row, for the interior-count diagnostics and the
    # optional scoring restriction.
    q_arr = (orb["a"].to_numpy(dtype=np.float64)
             * (1.0 - orb["e"].to_numpy(dtype=np.float64)))
    score_mask = None if args.score_q_max is None else (q_arr < args.score_q_max)
    n_interior = int((q_arr < args.strat_q_max).sum())
    print(f"\nInterior (q < {args.strat_q_max}) objects in set: {n_interior} / "
          f"{len(orb)} ({100.0 * n_interior / len(orb):.2f}%)")

    n_cells = len(orb) * N_ANOMALY_SAMPLES * N_EARTH_OFFSETS
    est_gb = 2 * n_cells * 4 / 1e9  # V_cell + elong_cell, float32
    print(f"\nPrecomputing per-cell geometry "
          f"({len(orb)} x {N_ANOMALY_SAMPLES} x {N_EARTH_OFFSETS} cells, "
          f"~{est_gb:.2f} GB for V + elongation)...")
    t0 = time.time()
    V_cell, elong_cell, weights, w_sum, num_all = precompute_cells(orb)
    print(f"  done ({time.time() - t0:.1f}s)  V_cell{V_cell.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    interior_mask = q_arr < args.strat_q_max
    interior_per_fold = [int(interior_mask[te].sum()) for _, te in folds]
    print(f"Rows: {len(orb)}  Fold sizes: {[len(te) for _, te in folds]}")
    print(f"Interior (q<{args.strat_q_max}) test rows per fold: {interior_per_fold}")

    def evaluate(emin, efull):
        return score_ramp(orb_base, y, folds, args.seed,
                          V_cell, elong_cell, weights, w_sum, num_all,
                          emin, efull, score_mask=score_mask)

    # --- Incumbent (current production 60 / 90) ----------------------------
    print(f"\nScoring incumbent ramp ({INCUMBENT_EMIN}, {INCUMBENT_EFULL}) deg "
          f"(current production values)...")
    t0 = time.time()
    inc_rmse, inc_pdev, inc_r2 = evaluate(INCUMBENT_EMIN, INCUMBENT_EFULL)
    print(f"  Incumbent: RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  "
          f"R2={inc_r2:.5f}  ({time.time() - t0:.1f}s)\n")

    # --- 2-D grid sweep ----------------------------------------------------
    emins = np.linspace(args.min_lo, args.min_hi, args.min_steps)
    efulls = np.linspace(args.full_lo, args.full_hi, args.full_steps)
    results = []  # (emin, efull, rmse, pdev, r2)
    print("=" * 100)
    print(f"{'eps_min':>8}  {'eps_full':>8}  {'width':>7}  {'RMSE':>9}  "
          f"{'Poisson':>9}  {'R2':>9}  {'imp/3':>5}  {'vmm mean':>9}  {'sec':>5}")
    print("-" * 100)
    for emin in emins:
        for efull in efulls:
            if efull <= emin:
                # Zero/negative ramp width is degenerate; skip to keep the
                # feature well defined (a hard step is reported separately if
                # the incumbent-style behavior is wanted).
                print(f"{emin:8.2f}  {efull:8.2f}  {'--':>7}  "
                      f"(skipped: epsilon_full <= epsilon_min)")
                continue
            t0 = time.time()
            rmse, pdev, r2 = evaluate(emin, efull)
            dt = time.time() - t0
            imp = (int(rmse < inc_rmse) + int(pdev < inc_pdev) + int(r2 > inc_r2))
            # Mean of the recomputed feature over the interior (q<strat) rows,
            # the subpopulation the gate actually moves.
            vmm_mean = float(
                vis_orbit_mag_multi_for(V_cell, elong_cell, weights, w_sum,
                                        num_all, emin, efull)[interior_mask].mean()
            )
            marker = "  <-- beats incumbent" if imp >= 2 else ""
            print(f"{emin:8.2f}  {efull:8.2f}  {efull - emin:7.2f}  "
                  f"{rmse:9.5f}  {pdev:9.5f}  {r2:9.5f}  "
                  f"{imp:5d}  {vmm_mean:9.5f}  {dt:5.1f}{marker}")
            results.append((emin, efull, rmse, pdev, r2))
    print("=" * 100)

    if not results:
        print("No valid (epsilon_full > epsilon_min) candidates were evaluated.")
        return

    # --- Pick the best grid point (min Poisson deviance) -------------------
    best = min(results, key=lambda r: r[3])
    best_emin, best_efull, best_rmse, best_pdev, best_r2 = best

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Incumbent  ({INCUMBENT_EMIN:.1f}, {INCUMBENT_EFULL:.1f}) deg   "
          f"RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  R2={inc_r2:.5f}")
    print(f"Best       ({best_emin:.2f}, {best_efull:.2f}) deg  "
          f"(width {best_efull - best_emin:.2f})   "
          f"RMSE={best_rmse:.5f}  Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
    d_rmse = best_rmse - inc_rmse
    d_pdev = best_pdev - inc_pdev
    d_r2 = best_r2 - inc_r2
    improvements = int(d_rmse < 0) + int(d_pdev < 0) + int(d_r2 > 0)
    print(f"Delta vs incumbent: RMSE={d_rmse:+.5f}  Poisson={d_pdev:+.5f}  "
          f"R2={d_r2:+.5f}  ({improvements}/3 improved)")
    best_rmse_pt = min(results, key=lambda r: r[2])
    best_r2_pt = max(results, key=lambda r: r[4])
    print(f"\nPer-metric grid optima:")
    print(f"  min RMSE    at ({best_rmse_pt[0]:.2f}, {best_rmse_pt[1]:.2f})  "
          f"(RMSE={best_rmse_pt[2]:.5f})")
    print(f"  min Poisson at ({best[0]:.2f}, {best[1]:.2f})  "
          f"(Poisson={best[3]:.5f})")
    print(f"  max R2      at ({best_r2_pt[0]:.2f}, {best_r2_pt[1]:.2f})  "
          f"(R2={best_r2_pt[4]:.5f})")
    if improvements >= 2:
        print(f"\n==> Recommend ELONG_MIN_DEG = {best_emin:.2f}, "
              f"ELONG_FULL_DEG = {best_efull:.2f}  "
              f"(beats the incumbent ({INCUMBENT_EMIN}, {INCUMBENT_EFULL}) on "
              f"{improvements}/3 metrics).")
    else:
        print(f"\n==> The incumbent ({INCUMBENT_EMIN}, {INCUMBENT_EFULL}) is hard "
              f"to beat; best grid point improves only {improvements}/3 metrics. "
              f"Keeping (60, 90) is defensible.")
    print("=" * 100)


if __name__ == "__main__":
    main()
