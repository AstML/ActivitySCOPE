"""tune_lat_lim.py

Grid (line) search over the single geocentric-ecliptic-latitude cutoff that
gates `spatial_discoverability_fraction`.

In activityscope_utils.feature_engineering, spatial_discoverability_fraction
is the Kepler-time-weighted fraction of an orbit at which an idealised
opposition apparition is BOTH bright enough (V_app <= V_LIM) AND close enough
to the ecliptic to fall inside a survey's latitude coverage:

    passed = (V_app <= V_LIM) & (|beta_geo| <= LAT_LIM_RAD)
    spatial_discoverability_fraction = sum_k(passed_k * w_k) / sum_k(w_k)

LAT_LIM_RAD is currently hard-coded to radians(25.0) -- an arbitrary, untuned
guess at "how far off the ecliptic the surveys still find things." This script
finds the latitude cutoff that makes the resulting feature most useful to the
production opposition-count regressor.

Why a grid sweep rather than the stochastic search used by the other tuners:
LAT_LIM is a single bounded scalar, so the whole landscape is a 1-D curve we
can simply trace. We do it efficiently by precomputing, ONCE, the per-anomaly
quantities that do not depend on the cutoff -- the V-magnitude-passing Kepler
weights (vpass_w) and |beta_geo| for every (object, anomaly) sample -- so each
candidate cutoff is a single vectorised reduction:

    spatial_disc(lat) = sum_k( vpass_w_k * 1[|beta_geo_k| <= lat] ) / w_sum

The fixed baseline feature set is the notebook's production regression feature
list (mlcols_reg) with spatial_discoverability_fraction swapped out for the
recomputed column. Each candidate cutoff is scored with k-fold CV blending a
LightGBM and an XGBoost Poisson regressor (the NEW_tuner_for_vis evaluator),
reporting fold-averaged RMSE, Poisson deviance, and R^2. The incumbent
production value (25 deg) is always evaluated and printed for comparison; a
candidate is flagged a winner if it beats the incumbent on >= 2 of the 3
metrics.

The search range is restricted to 10-35 deg by default -- the physically
plausible band for "how far off the ecliptic surveys still find things"
(the incumbent 25 deg sits inside it). Widen with --lo / --hi if desired.

Run:
    python tune_lat_lim.py [--lo 10] [--hi 35] [--steps 26] [--seed 0]
                           [--subsample K] [--n-folds 8] [--refine]
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

# The production value being tuned (degrees). Always evaluated for comparison.
INCUMBENT_LAT_DEG = 25.0

# Orbit-sampling / photometry constants -- copied verbatim from
# feature_engineering so the recomputed spatial_disc matches the notebook bit
# for bit at LAT_LIM = 25 deg.
N_ANOMALY_SAMPLES = 32
ANOMALY_CHUNK_SIZE = 100_000
OBL_DEG = 23.44
V_LIM = 22.5
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
# Precompute the cutoff-independent per-anomaly arrays
# ---------------------------------------------------------------------------

def precompute_spatial_arrays(orb):
    """Replicate feature_engineering's opposition-geometry sampling and return
    the pieces of spatial_discoverability_fraction that do NOT depend on the
    latitude cutoff:

        vpass_w  (N, n_nu) float32 : Kepler weight w_k where V_app <= V_LIM,
                                     else 0  (the V-magnitude gate, applied once)
        abs_beta (N, n_nu) float32 : |geocentric ecliptic latitude| at each
                                     anomaly, in RADIANS
        w_sum    (N,)      float64 : sum of all Kepler weights per object

    spatial_disc(lat) is then sum_k(vpass_w * (abs_beta <= lat)) / w_sum.
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
    vpass_w = np.empty((N, n_nu), dtype=np.float32)
    abs_beta = np.empty((N, n_nu), dtype=np.float32)
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

        # Geocentric ecliptic latitude (matches what surveys see).
        beta_geo = np.arcsin(np.clip(dz / Delta_opp_safe, -1.0, 1.0))

        v_pass = V_app <= V_LIM
        vpass_w[start:end] = (weights * v_pass).astype(np.float32)
        abs_beta[start:end] = np.abs(beta_geo).astype(np.float32)

    return vpass_w, abs_beta, w_sum


def spatial_disc_for_lat(vpass_w, abs_beta, w_sum, lat_rad):
    """Recompute spatial_discoverability_fraction for one latitude cutoff."""
    passed_w = vpass_w * (abs_beta <= np.float32(lat_rad))
    return passed_w.sum(axis=1) / w_sum


# ---------------------------------------------------------------------------
# Evaluation: k-fold CV blending LightGBM + XGBoost Poisson regressors
# (identical evaluator to NEW_tuner_for_vis)
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


def score_lat(orb_base, y, folds, seed, vpass_w, abs_beta, w_sum, lat_deg):
    """Build the feature matrix for one latitude cutoff and CV-score it."""
    lat_rad = np.radians(lat_deg)
    X = orb_base.copy()
    X[TUNED_COL] = spatial_disc_for_lat(vpass_w, abs_beta, w_sum, lat_rad).astype(np.float32)
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
    parser.add_argument("--lo", type=float, default=10.0,
                        help="Lowest latitude cutoff in degrees (default: 10).")
    parser.add_argument("--hi", type=float, default=35.0,
                        help="Highest latitude cutoff in degrees (default: 35).")
    parser.add_argument("--steps", type=int, default=26,
                        help="Number of grid points from --lo to --hi inclusive "
                             "(default: 26, i.e. 1 deg spacing over 10-35).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV folds, subsample, and "
                             "model fits (default: 0).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="CV folds per candidate (default: 8).")
    parser.add_argument("--refine", action="store_true",
                        help="After the grid sweep, run a golden-section refine "
                             "between the grid points bracketing the best one.")
    parser.add_argument("--refine-iters", type=int, default=8,
                        help="Golden-section iterations when --refine is set "
                             "(default: 8).")
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")
    if args.steps < 2:
        parser.error("--steps must be >= 2.")
    if args.hi <= args.lo:
        parser.error("--hi must exceed --lo.")

    np.random.seed(args.seed)

    print(f"Deterministic seed (folds + model fits): {args.seed}")
    print(f"CV: {args.n_folds}-fold, blending LightGBM + XGBoost (Poisson)")
    print(f"Tuning: LAT_LIM (gate on {TUNED_COL})")
    print(f"Grid: {args.steps} points in [{args.lo}, {args.hi}] deg "
          f"(incumbent = {INCUMBENT_LAT_DEG} deg)")
    print(f"Fixed baseline features ({len(BASELINE_COLS)}): {BASELINE_COLS}")

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb[TARGET].astype(np.float32).to_numpy()
    orb_base = orb[BASELINE_COLS].copy()

    print("\nPrecomputing cutoff-independent opposition geometry...")
    t0 = time.time()
    vpass_w, abs_beta, w_sum = precompute_spatial_arrays(orb)
    print(f"  done ({time.time() - t0:.1f}s)  "
          f"arrays: vpass_w{vpass_w.shape}, abs_beta{abs_beta.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    print(f"Rows: {len(orb)}  Fold sizes: {[len(te) for _, te in folds]}")

    def evaluate(lat_deg):
        return score_lat(orb_base, y, folds, args.seed,
                         vpass_w, abs_beta, w_sum, lat_deg)

    # --- Incumbent (current production 25 deg) -----------------------------
    print(f"\nScoring incumbent LAT_LIM = {INCUMBENT_LAT_DEG} deg "
          f"(current production value)...")
    t0 = time.time()
    inc_rmse, inc_pdev, inc_r2 = evaluate(INCUMBENT_LAT_DEG)
    print(f"  Incumbent: RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  "
          f"R2={inc_r2:.5f}  ({time.time() - t0:.1f}s)\n")

    # --- Grid sweep --------------------------------------------------------
    grid = np.linspace(args.lo, args.hi, args.steps)
    results = []  # (lat_deg, rmse, pdev, r2)
    print("=" * 90)
    print(f"{'lat(deg)':>9}  {'RMSE':>9}  {'Poisson':>9}  {'R2':>9}  "
          f"{'imp/3':>5}  {'spatial_disc mean':>18}  {'sec':>5}")
    print("-" * 90)
    for lat_deg in grid:
        t0 = time.time()
        rmse, pdev, r2 = evaluate(lat_deg)
        dt = time.time() - t0
        imp = (int(rmse < inc_rmse) + int(pdev < inc_pdev) + int(r2 > inc_r2))
        # Diagnostic: how much "discoverable" mass this cutoff admits on average.
        sd_mean = float(
            spatial_disc_for_lat(vpass_w, abs_beta, w_sum, np.radians(lat_deg)).mean()
        )
        marker = "  <-- beats incumbent" if imp >= 2 else ""
        print(f"{lat_deg:9.2f}  {rmse:9.5f}  {pdev:9.5f}  {r2:9.5f}  "
              f"{imp:5d}  {sd_mean:18.5f}  {dt:5.1f}{marker}")
        results.append((lat_deg, rmse, pdev, r2))
    print("=" * 90)

    # --- Pick the best grid point ------------------------------------------
    # Primary objective: Poisson deviance (the training objective). RMSE and R2
    # are reported alongside; the >=2/3 rule decides "beats incumbent."
    def keyfn(r):
        return r[2]  # pdev
    best = min(results, key=keyfn)
    best_lat, best_rmse, best_pdev, best_r2 = best

    # --- Optional golden-section refine around the best grid point ---------
    if args.refine:
        step = (args.hi - args.lo) / (args.steps - 1)
        lo = max(args.lo, best_lat - step)
        hi = min(args.hi, best_lat + step)
        print(f"\nGolden-section refine in [{lo:.3f}, {hi:.3f}] deg "
              f"({args.refine_iters} iters), minimizing Poisson deviance...")
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = evaluate(c)[1]
        fd = evaluate(d)[1]
        for _ in range(args.refine_iters):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = evaluate(c)[1]
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = evaluate(d)[1]
            print(f"  bracket [{lo:.3f}, {hi:.3f}]  "
                  f"Poisson@{(0.5*(lo+hi)):.3f}deg")
        ref_lat = 0.5 * (lo + hi)
        ref_rmse, ref_pdev, ref_r2 = evaluate(ref_lat)
        print(f"  Refined: lat={ref_lat:.3f} deg  RMSE={ref_rmse:.5f}  "
              f"Poisson={ref_pdev:.5f}  R2={ref_r2:.5f}")
        if ref_pdev < best_pdev:
            best_lat, best_rmse, best_pdev, best_r2 = ref_lat, ref_rmse, ref_pdev, ref_r2

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Incumbent  LAT_LIM = {INCUMBENT_LAT_DEG:6.2f} deg   "
          f"RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  R2={inc_r2:.5f}")
    print(f"Best (min Poisson) = {best_lat:6.2f} deg   "
          f"RMSE={best_rmse:.5f}  Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
    d_rmse = best_rmse - inc_rmse
    d_pdev = best_pdev - inc_pdev
    d_r2 = best_r2 - inc_r2
    improvements = int(d_rmse < 0) + int(d_pdev < 0) + int(d_r2 > 0)
    print(f"Delta vs incumbent: RMSE={d_rmse:+.5f}  Poisson={d_pdev:+.5f}  "
          f"R2={d_r2:+.5f}  ({improvements}/3 improved)")
    # Also report the per-metric grid optima, in case they disagree.
    best_rmse_pt = min(results, key=lambda r: r[1])
    best_r2_pt = max(results, key=lambda r: r[3])
    print(f"\nPer-metric grid optima:")
    print(f"  min RMSE    at {best_rmse_pt[0]:6.2f} deg  (RMSE={best_rmse_pt[1]:.5f})")
    print(f"  min Poisson at {best[0]:6.2f} deg  (Poisson={best[2]:.5f})")
    print(f"  max R2      at {best_r2_pt[0]:6.2f} deg  (R2={best_r2_pt[3]:.5f})")
    if improvements >= 2:
        print(f"\n==> Recommend LAT_LIM_RAD = np.radians({best_lat:.2f})  "
              f"(beats the arbitrary {INCUMBENT_LAT_DEG} deg on {improvements}/3 metrics).")
    else:
        print(f"\n==> The incumbent {INCUMBENT_LAT_DEG} deg is hard to beat; "
              f"best grid point improves only {improvements}/3 metrics. "
              f"Keeping 25 deg is defensible.")
    print("=" * 90)


if __name__ == "__main__":
    main()
