"""tune_v_lim.py

2-D grid search over a linear *brightness roll-off* that gates
`spatial_discoverability_fraction`.

In activityscope_utils.feature_engineering, spatial_disc is the Kepler-time-
weighted, latitude-weighted fraction of an orbit at which an idealised
opposition apparition is bright enough to be discovered. Today the brightness
test is a hard cut at V_lim = 22.5. But survey depth is not a sharp edge:
detection efficiency tapers as objects approach the limiting magnitude. This
tuner replaces the hard cut with a one-sided linear roll-off in apparent
magnitude, parameterised by two numbers:

    lower : magnitude at/below which an apparition has full weight (1)
    width : magnitude span of the ramp; weight reaches 0 at lower + width

    v_weight(V_app) = clip( (lower + width - V_app) / width, 0, 1 )

      = 1                     for V_app <= lower            (easily detected)
      = linearly 1 -> 0       for lower < V_app < lower+width
      = 0                     for V_app >= lower + width    (too faint)

The (fixed) latitude roll-off w(beta) is unchanged. spatial_disc becomes:

    spatial_disc(lower, width)
        = sum_k( v_weight(V_app_k) * w(beta_k) * r_k^2 ) / sum_k(r_k^2)

We grid-search lower in [20.8, 21.8] mag and width in [0.8, 2.2] mag on a
4x4 grid by default (16 candidates). The (lower, width)-independent geometry
is precomputed ONCE: we store the per-anomaly apparent magnitude V_app and the
latitude-weighted Kepler weights w(beta_k) * r_k^2, so each candidate is a
single vectorised reduction.

The fixed baseline feature set is the notebook's production regression feature
list (mlcols_reg) with spatial_discoverability_fraction swapped for the
recomputed column. Each candidate is scored with k-fold CV blending a LightGBM
and an XGBoost Poisson regressor, reporting fold-averaged RMSE, Poisson
deviance, and R^2. The incumbent production hard cut (V_lim = 22.5) is always
evaluated and printed; a candidate is flagged a winner if it beats the
incumbent on >= 2 of the 3 metrics.

Run:
    python tune_v_lim.py [--lower-lo 20.8] [--lower-hi 21.8]
                         [--width-lo 0.8] [--width-hi 2.2]
                         [--lower-steps 4] [--width-steps 4]
                         [--seed 0] [--subsample K] [--n-folds 8]
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

# The production hard cut (mag). Always evaluated for comparison.
INCUMBENT_VLIM = 22.5

# Latitude roll-off endpoints (degrees), held FIXED at the production values.
LAT_ROLLOFF_FULL_DEG = 2.0
LAT_ROLLOFF_ZERO_DEG = 32.0

# Orbit-sampling / photometry constants -- copied verbatim from
# feature_engineering so the recomputed spatial_disc matches the notebook bit
# for bit at the incumbent hard cut V_lim = 22.5.
N_ANOMALY_SAMPLES = 32
ANOMALY_CHUNK_SIZE = 100_000
OBL_DEG = 23.44
HG_A1, HG_B1 = 3.33, 0.63
HG_A2, HG_B2 = 1.87, 1.22
HG_G = 0.15
PHI_FLOOR = 1e-30

# Production regression feature list (notebook mlcols_reg, target dropped).
# spatial_discoverability_fraction is REPLACED per candidate, so it is held in
# its own constant and excluded from the fixed baseline.
TUNED_COL = "spatial_discoverability_fraction"
BASELINE_COLS = [
    "H", "Node", "a", "i",
    "Perihelion_direction_x_e", "Perihelion_direction_y_e",
    "vis_orbit_mag_multi", "dec_flux_weighted", "vis_opp_mean",
    "e", "vis_q", "vis_timeavg", "vis_inc", "orbital_period_sync",
]
TARGET = "Num_opps_minus_one"


# ---------------------------------------------------------------------------
# Precompute the (lower, width)-independent per-anomaly arrays
# ---------------------------------------------------------------------------

def precompute_spatial_arrays(orb):
    """Replicate feature_engineering's opposition-geometry sampling and return
    the pieces of spatial_discoverability_fraction that do NOT depend on the
    brightness roll-off:

        V_app      (N, n_nu) float32 : apparent V at each (object, anomaly)
        lw_weights (N, n_nu) float32 : latitude-weighted Kepler weight,
                                       w(beta_k) * r_k^2 (roll-off applied once)
        w_sum      (N,)      float64 : sum of all Kepler weights per object

    spatial_disc(lower, width) is then
        sum_k( v_weight(V_app; lower, width) * lw_weights ) / w_sum.
    """
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_rad = np.radians(orb["i"].to_numpy(dtype=np.float64))
    H = orb["H"].to_numpy(dtype=np.float64)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))

    nu_arr = np.linspace(0.0, 2.0 * np.pi, N_ANOMALY_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu_arr)
    eps_val = EPS

    N = len(a)
    n_nu = N_ANOMALY_SAMPLES
    V_app_out = np.empty((N, n_nu), dtype=np.float32)
    lw_weights = np.empty((N, n_nu), dtype=np.float32)
    w_sum = np.empty(N, dtype=np.float64)

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

        # Idealised opposition Earth: r=1 AU, in-ecliptic, at the asteroid's
        # heliocentric longitude. Geocentric vector + apparent V via IAU HG.
        lambda_k = np.arctan2(y_ecl, x_ecl)
        dx = x_ecl - np.cos(lambda_k)
        dy = y_ecl - np.sin(lambda_k)
        dz = z_ecl  # Earth z = 0
        Delta_opp = np.sqrt(dx * dx + dy * dy + dz * dz)
        Delta_opp_safe = np.maximum(Delta_opp, eps_val)
        cos_alpha = np.clip(
            (r_orb ** 2 + Delta_opp ** 2 - 1.0) / (2.0 * r_safe * Delta_opp_safe),
            -1.0, 1.0,
        )
        alpha = np.arccos(cos_alpha)
        tan_half = np.maximum(np.tan(alpha / 2.0), 0.0)
        phi1 = np.exp(-HG_A1 * np.power(tan_half, HG_B1))
        phi2 = np.exp(-HG_A2 * np.power(tan_half, HG_B2))
        phi_blend = np.maximum((1.0 - HG_G) * phi1 + HG_G * phi2, PHI_FLOOR)
        V_app = (H_c
                 + 5.0 * np.log10(r_safe * Delta_opp_safe)
                 - 2.5 * np.log10(phi_blend))

        # Geocentric ecliptic latitude (matches what surveys see), then the
        # fixed linear roll-off weight w(beta) (same form as feature_engineering).
        beta_geo = np.arcsin(np.clip(dz / Delta_opp_safe, -1.0, 1.0))
        beta_deg = np.degrees(np.abs(beta_geo))
        lat_w = np.clip(
            (LAT_ROLLOFF_ZERO_DEG - beta_deg)
            / (LAT_ROLLOFF_ZERO_DEG - LAT_ROLLOFF_FULL_DEG),
            0.0, 1.0,
        )

        V_app_out[start:end] = V_app.astype(np.float32)
        lw_weights[start:end] = (lat_w * weights).astype(np.float32)

    return V_app_out, lw_weights, w_sum


def v_weight(V_app, lower, width):
    """One-sided linear brightness roll-off in [0, 1]:

        = 1                          for V_app <= lower
        = (lower+width - V_app)/width in the ramp
        = 0                          for V_app >= lower + width

    width <= 0 reduces to the hard cut at `lower`."""
    if width <= 0.0:
        return (V_app <= np.float32(lower)).astype(np.float32)
    return np.clip(
        (np.float32(lower + width) - V_app) / np.float32(width),
        0.0, 1.0,
    ).astype(np.float32)


def spatial_disc_for_rolloff(V_app, lw_weights, w_sum, lower, width):
    """Recompute spatial_discoverability_fraction under the brightness roll-off."""
    passed_w = v_weight(V_app, lower, width) * lw_weights
    return passed_w.sum(axis=1) / w_sum


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


def evaluate_cv(X, y, folds, seed):
    rmses, pdevs, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        p_xgb = _fit_predict_xgb(X_tr, y_tr, X_te, seed)
        p_lgb = _fit_predict_lgbm(X_tr, y_tr, X_te, seed)
        y_pred = np.maximum(0.5 * (p_xgb + p_lgb), EPS)
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        pdevs.append(mean_poisson_deviance(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(rmses)), float(np.mean(pdevs)), float(np.mean(r2s))


def score_rolloff(orb_base, y, folds, seed, V_app, lw_weights, w_sum, lower, width):
    """Build the feature matrix for one (lower, width) and CV-score it."""
    X = orb_base.copy()
    X[TUNED_COL] = spatial_disc_for_rolloff(
        V_app, lw_weights, w_sum, lower, width
    ).astype(np.float32)
    return evaluate_cv(X.astype(np.float32), y, folds, seed)


# ---------------------------------------------------------------------------
# Data preparation (mirrors the other tuners / the notebook training filter)
# ---------------------------------------------------------------------------

def prepare_training_data(subsample=None, seed=0):
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
    # spatial_disc is recomputed from a/e/i/Node/Peri/H, so require those too.
    needed = BASELINE_COLS + [TARGET, "Peri"]
    orb = orb.dropna(subset=needed)
    print(f"Post-filter row count: {len(orb)}")

    if subsample is not None and subsample < len(orb):
        orb = orb.sample(n=subsample, random_state=seed)
        print(f"Subsampled to {len(orb)} rows for faster iteration.")

    return orb.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lower-lo", type=float, default=20.8,
                        help="Lowest roll-off lower bound, mag (default: 20.8).")
    parser.add_argument("--lower-hi", type=float, default=21.8,
                        help="Highest roll-off lower bound, mag (default: 21.8).")
    parser.add_argument("--width-lo", type=float, default=0.8,
                        help="Narrowest roll-off width, mag (default: 0.8).")
    parser.add_argument("--width-hi", type=float, default=2.2,
                        help="Widest roll-off width, mag (default: 2.2).")
    parser.add_argument("--lower-steps", type=int, default=4,
                        help="Grid points along the lower-bound axis (default: 4).")
    parser.add_argument("--width-steps", type=int, default=4,
                        help="Grid points along the width axis (default: 4).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV folds, subsample, and "
                             "model fits (default: 0).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="CV folds per candidate (default: 8).")
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")
    if args.lower_steps < 1 or args.width_steps < 1:
        parser.error("--lower-steps and --width-steps must be >= 1.")
    if args.lower_hi < args.lower_lo:
        parser.error("--lower-hi must be >= --lower-lo.")
    if args.width_hi < args.width_lo:
        parser.error("--width-hi must be >= --width-lo.")
    if args.width_lo <= 0.0:
        parser.error("--width-lo must be > 0.")

    np.random.seed(args.seed)

    print(f"Deterministic seed (folds + model fits): {args.seed}")
    print(f"CV: {args.n_folds}-fold, blending LightGBM + XGBoost (Poisson)")
    print(f"Tuning: linear brightness roll-off on {TUNED_COL}")
    print(f"  lower in [{args.lower_lo}, {args.lower_hi}] mag "
          f"({args.lower_steps} pts);  width in [{args.width_lo}, "
          f"{args.width_hi}] mag ({args.width_steps} pts)  "
          f"=> {args.lower_steps * args.width_steps} candidates")
    print(f"  v_weight = 1 at V_app<=lower, linearly to 0 at V_app=lower+width")
    print(f"Latitude roll-off held fixed at "
          f"[{LAT_ROLLOFF_FULL_DEG}, {LAT_ROLLOFF_ZERO_DEG}] deg")
    print(f"Fixed baseline features ({len(BASELINE_COLS)}): {BASELINE_COLS}")

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb[TARGET].astype(np.float32).to_numpy()
    orb_base = orb[BASELINE_COLS].copy()

    print("\nPrecomputing roll-off-independent opposition geometry...")
    t0 = time.time()
    V_app, lw_weights, w_sum = precompute_spatial_arrays(orb)
    print(f"  done ({time.time() - t0:.1f}s)  "
          f"arrays: V_app{V_app.shape}, lw_weights{lw_weights.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    print(f"Rows: {len(orb)}  Fold sizes: {[len(te) for _, te in folds]}")

    def evaluate(lower, width):
        return score_rolloff(orb_base, y, folds, args.seed,
                             V_app, lw_weights, w_sum, lower, width)

    # --- Incumbent (current production hard cut at 22.5) -------------------
    print(f"\nScoring incumbent hard cut V_lim = {INCUMBENT_VLIM} mag "
          f"(current production value)...")
    t0 = time.time()
    inc_rmse, inc_pdev, inc_r2 = score_rolloff(
        orb_base, y, folds, args.seed, V_app, lw_weights, w_sum,
        INCUMBENT_VLIM, 0.0,  # width=0 => hard cut
    )
    print(f"  Incumbent: RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  "
          f"R2={inc_r2:.5f}  ({time.time() - t0:.1f}s)\n")

    # --- 2-D grid sweep ----------------------------------------------------
    lowers = np.linspace(args.lower_lo, args.lower_hi, args.lower_steps)
    widths = np.linspace(args.width_lo, args.width_hi, args.width_steps)
    results = []  # (lower, width, rmse, pdev, r2)
    print("=" * 100)
    print(f"{'lower':>7}  {'width':>7}  {'zero@':>7}  {'RMSE':>9}  {'Poisson':>9}  "
          f"{'R2':>9}  {'imp/3':>5}  {'sd mean':>9}  {'sec':>5}")
    print("-" * 100)
    for lower in lowers:
        for width in widths:
            t0 = time.time()
            rmse, pdev, r2 = evaluate(lower, width)
            dt = time.time() - t0
            imp = (int(rmse < inc_rmse) + int(pdev < inc_pdev) + int(r2 > inc_r2))
            sd_mean = float(
                spatial_disc_for_rolloff(V_app, lw_weights, w_sum, lower, width).mean()
            )
            marker = "  <-- beats incumbent" if imp >= 2 else ""
            print(f"{lower:7.3f}  {width:7.3f}  {lower + width:7.3f}  "
                  f"{rmse:9.5f}  {pdev:9.5f}  {r2:9.5f}  "
                  f"{imp:5d}  {sd_mean:9.5f}  {dt:5.1f}{marker}")
            results.append((lower, width, rmse, pdev, r2))
    print("=" * 100)

    # --- Pick the best grid point (min Poisson deviance) -------------------
    best = min(results, key=lambda r: r[3])
    best_lower, best_width, best_rmse, best_pdev, best_r2 = best

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Incumbent  hard cut V_lim = {INCUMBENT_VLIM:.2f} mag   "
          f"RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  R2={inc_r2:.5f}")
    print(f"Best  lower={best_lower:.3f}  width={best_width:.3f} mag  "
          f"(zero at {best_lower + best_width:.3f})   "
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
    print(f"  min RMSE    at lower={best_rmse_pt[0]:.3f} width={best_rmse_pt[1]:.3f}  "
          f"(RMSE={best_rmse_pt[2]:.5f})")
    print(f"  min Poisson at lower={best[0]:.3f} width={best[1]:.3f}  "
          f"(Poisson={best[3]:.5f})")
    print(f"  max R2      at lower={best_r2_pt[0]:.3f} width={best_r2_pt[1]:.3f}  "
          f"(R2={best_r2_pt[4]:.5f})")

    if improvements >= 2:
        lo, w = best_lower, best_width
        print(f"\n==> Recommend a brightness roll-off with lower={lo:.3f}, "
              f"width={w:.3f} (full weight to V_app<={lo:.3f}, zero at "
              f"{lo + w:.3f}); beats the hard {INCUMBENT_VLIM} cut on "
              f"{improvements}/3 metrics.")
        print(f"    In feature_engineering, replace the hard brightness gate")
        print(f"        passed = (V_app <= V_LIM) * lat_weight")
        print(f"    with a soft weight:")
        print(f"        v_weight = np.clip(({lo + w:.3f} - V_app) / {w:.3f}, 0.0, 1.0)")
        print(f"        passed   = v_weight * lat_weight")
    else:
        print(f"\n==> No roll-off clearly beats the hard {INCUMBENT_VLIM} cut "
              f"(best improves {improvements}/3 metrics); the sharp cut is "
              f"defensible.")
    print("=" * 100)


if __name__ == "__main__":
    main()
