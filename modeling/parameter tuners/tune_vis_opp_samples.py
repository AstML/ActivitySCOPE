"""tune_vis_opp_samples.py

1-D sweep over the number of recent equal-longitude "opposition" apparitions
`N_OPP` averaged to produce the `vis_opp_mean` feature.

In activityscope_utils.feature_engineering, recent equal-longitude Earth-asteroid
geometries are analytically solved and their apparent magnitude computed. At
present, the `vis_opp_mean` regression feature averages the `N_OPP = 5` most
recent valid apparitions. This tuner sweeps `N_OPP` from 10 to 20 inclusive to
see if a longer or shorter history provides a better signal.

Run:
    python "tune_vis_opp_samples.py"
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

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import activityscope_utils as utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-3

# The production hard cut. Always evaluated for comparison.
INCUMBENT_N_OPP = 5
MAX_N_OPP = 20
MIN_N_OPP = 10

TUNED_COL = "vis_opp_mean_disc"
BASELINE_COLS = [
    "H", "Node", "a", "i",
    "Perihelion_direction_x_e", "Perihelion_direction_y_e",
    "vis_orbit_mag_multi", "dec_flux_weighted", "vis_opp_mean_disc",
    "e", "vis_q", "vis_timeavg", "vis_inc", "orbital_period_sync",
]
TARGET = "Num_opps_minus_one"

TWO_PI = 2.0 * np.pi
OPP_MAX_LOOKBACK_DAYS = 20.0 * 365.25
VIS_OPP_FAINT = 28.0
HG_A1, HG_B1 = 3.33, 0.63
HG_A2, HG_B2 = 1.87, 1.22
HG_G = 0.15
PHI_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# Precompute the V_opp arrays for N_OPP up to max_opps
# ---------------------------------------------------------------------------

def _wrap_pi(ang):
    return (ang + np.pi) % TWO_PI - np.pi

def _earth_lon(t):
    T = (t - 2451545.0) / 36525.0
    Me = np.radians(357.52911 + 35999.05029 * T)
    lam_sun = (np.radians(280.46646 + 36000.76983 * T)
               + 0.033416 * np.sin(Me)
               + 0.000349 * np.sin(2.0 * Me))
    return lam_sun + np.pi

def _earth_xy(t):
    T = (t - 2451545.0) / 36525.0
    Me = np.radians(357.52911 + 35999.05029 * T)
    lam_sun = (np.radians(280.46646 + 36000.76983 * T)
               + 0.033416 * np.sin(Me)
               + 0.000349 * np.sin(2.0 * Me))
    lam_e = lam_sun + np.pi
    r_e = (1.00014061
           - 0.01670861 * np.cos(Me)
           - 0.00013957 * np.cos(2.0 * Me))
    return r_e * np.cos(lam_e), r_e * np.sin(lam_e)

def _kepler_E(Marr, earr):
    E = Marr + earr * np.sin(Marr) * (1.0 + earr * np.cos(Marr))
    for _ in range(12):
        E = E - (E - earr * np.sin(E) - Marr) / (1.0 - earr * np.cos(E))
    return E

def precompute_v_opp_array(orb, max_opps=MAX_N_OPP):
    MU_SUN = 0.0002959122082855911  # AU^3 / day^2 (k^2)
    a_np = orb['a'].to_numpy(dtype=np.float64)
    e_np = np.clip(orb['e'].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_np = np.radians(orb['i'].to_numpy(dtype=np.float64))
    H_np = orb['H'].to_numpy(dtype=np.float64)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))
    M0 = np.radians(orb["M"].to_numpy(dtype=np.float64)) % (2.0 * np.pi)
    Epoch_jd = orb["Epoch"].to_numpy(dtype=np.float64)
    
    eps_val = EPS
    n_mm = np.sqrt(MU_SUN) / np.power(np.maximum(a_np, eps_val), 1.5)

    a_col = a_np[:, None]
    e_col = e_np[:, None]
    i_col = i_np[:, None]
    H_col = H_np[:, None]
    Node_col = Node_rad[:, None]
    Peri_col = Peri_rad[:, None]
    M0_col = M0[:, None]
    Epoch_col = Epoch_jd[:, None]
    n_col = n_mm[:, None]
    
    cos_Node_o = np.cos(Node_col)
    sin_Node_o = np.sin(Node_col)
    cos_i_o = np.cos(i_col)
    sin_i_o = np.sin(i_col)

    n_earth_rate = np.radians(36000.76983 / 36525.0)
    n_syn = n_mm - n_earth_rate
    n_syn_safe = np.where(np.abs(n_syn) < 1e-12, 1e-12, n_syn)
    P_syn = TWO_PI / np.abs(n_syn_safe)  # synodic period (days), > 0

    varpi = Node_rad + Peri_rad
    psi_E = _wrap_pi((varpi + M0) - _earth_lon(Epoch_jd))
    t_near = Epoch_jd - psi_E / n_syn_safe
    t_last = np.where(t_near > Epoch_jd, t_near - P_syn, t_near)
    
    k_idx = np.arange(max_opps)
    t_opp = t_last[:, None] - k_idx[None, :] * P_syn[:, None]  # (N, max_opps)
    t_guess = t_opp.copy()
    half_syn = 0.5 * P_syn[:, None]

    for _ in range(4):
        M_o = (M0_col + n_col * (t_opp - Epoch_col)) % TWO_PI
        E_o = _kepler_E(M_o, e_col)
        nu_o = 2.0 * np.arctan2(
            np.sqrt(1.0 + e_col) * np.sin(E_o / 2.0),
            np.sqrt(1.0 - e_col) * np.cos(E_o / 2.0),
        )
        u_o = nu_o + Peri_col
        cos_u_o = np.cos(u_o)
        sin_u_o = np.sin(u_o)
        x_dir = cos_Node_o * cos_u_o - sin_Node_o * sin_u_o * cos_i_o
        y_dir = sin_Node_o * cos_u_o + cos_Node_o * sin_u_o * cos_i_o
        lam_ast = np.arctan2(y_dir, x_dir)
        psi = _wrap_pi(lam_ast - _earth_lon(t_opp))
        t_opp = t_opp - psi / n_syn_safe[:, None]
        t_opp = np.clip(t_opp, t_guess - half_syn, t_guess + half_syn)

    M_o = (M0_col + n_col * (t_opp - Epoch_col)) % TWO_PI
    E_o = _kepler_E(M_o, e_col)
    nu_o = 2.0 * np.arctan2(
        np.sqrt(1.0 + e_col) * np.sin(E_o / 2.0),
        np.sqrt(1.0 - e_col) * np.cos(E_o / 2.0),
    )
    r_o = a_col * (1.0 - e_col * np.cos(E_o))
    u_o = nu_o + Peri_col
    cos_u_o = np.cos(u_o)
    sin_u_o = np.sin(u_o)
    x_o = r_o * (cos_Node_o * cos_u_o - sin_Node_o * sin_u_o * cos_i_o)
    y_o = r_o * (sin_Node_o * cos_u_o + cos_Node_o * sin_u_o * cos_i_o)
    z_o = r_o * sin_u_o * sin_i_o

    x_eo, y_eo = _earth_xy(t_opp)
    dx_o = x_o - x_eo
    dy_o = y_o - y_eo
    dz_o = z_o
    Delta_o = np.sqrt(dx_o * dx_o + dy_o * dy_o + dz_o * dz_o)
    Delta_o_safe = np.maximum(Delta_o, eps_val)
    r_o_safe = np.maximum(r_o, eps_val)

    dot_o = x_o * dx_o + y_o * dy_o + z_o * dz_o
    cos_alpha_o = np.clip(dot_o / (r_o_safe * Delta_o_safe), -1.0, 1.0)
    alpha_o = np.arccos(cos_alpha_o)
    tan_half_o = np.maximum(np.tan(alpha_o / 2.0), 0.0)
    phi1_o = np.exp(-HG_A1 * np.power(tan_half_o, HG_B1))
    phi2_o = np.exp(-HG_A2 * np.power(tan_half_o, HG_B2))
    phi_blend_o = np.maximum((1.0 - HG_G) * phi1_o + HG_G * phi2_o, PHI_FLOOR)

    V_opp = (H_col
             + 5.0 * np.log10(r_o_safe * Delta_o_safe)
             - 2.5 * np.log10(phi_blend_o))

    stale = (Epoch_col - t_opp) > OPP_MAX_LOOKBACK_DAYS
    V_opp = np.where(stale, VIS_OPP_FAINT, V_opp)

    return V_opp


# ---------------------------------------------------------------------------
# Evaluation: k-fold CV blending LightGBM + XGBoost Poisson regressors
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

def score_opps(orb_base, y, folds, seed, V_opp, num_opps, disc_penalty):
    X = orb_base.copy()
    X[TUNED_COL] = V_opp[:, :num_opps].mean(axis=1).astype(np.float32) + disc_penalty.astype(np.float32)
    
    # Drop baseline columns that might not exist yet, though our orb_base is built cleanly.
    return evaluate_cv(X.astype(np.float32), y, folds, seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(subsample=None, seed=0):
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases()
    orb = orb[~orb["filtered_out"].astype(bool)]
    orb = utils.feature_engineering(orb)

    print("Merging cached extension_difficulty.csv...")
    try:
        extension_difficulty = pd.read_csv("extension_difficulty.csv")
        orb = orb.merge(extension_difficulty, on="Principal_desig", how="left")
    except Exception as e:
        print("Note: extension_difficulty.csv not found or error loading it:", e)
        orb["extension_difficulty"] = 0.0

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
    needed = BASELINE_COLS + [TARGET, "Epoch", "M", "Peri", "spatial_discoverability_fraction"]
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
    parser = argparse.ArgumentParser(description="Tune number of recent opposition apparitions.")
    parser.add_argument("--min-opps", type=int, default=10,
                        help="Minimum number of apparitions to average (default: 10).")
    parser.add_argument("--max-opps", type=int, default=20,
                        help="Maximum number of apparitions to average (default: 20).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV folds, subsample, and model fits (default: 0).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="CV folds per candidate (default: 8).")
    args = parser.parse_args()

    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")
    if args.min_opps < 1:
        parser.error("--min-opps must be >= 1.")
    if args.max_opps < args.min_opps:
        parser.error("--max-opps must be >= --min-opps.")

    np.random.seed(args.seed)

    print(f"Deterministic seed: {args.seed}")
    print(f"CV: {args.n_folds}-fold, blending LightGBM + XGBoost (Poisson)")
    print(f"Tuning: N_OPP over the range [{args.min_opps}, {args.max_opps}] on {TUNED_COL}")
    print(f"Fixed baseline features ({len(BASELINE_COLS)}): {BASELINE_COLS}")

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb[TARGET].astype(np.float32).to_numpy()
    
    DISC_FRAC_FLOOR = 1e-3
    disc_penalty = -2.5 * np.log10(
        np.maximum(
            orb["spatial_discoverability_fraction"].to_numpy(dtype=np.float64),
            DISC_FRAC_FLOOR,
        )
    )
    
    orb_base = orb[BASELINE_COLS].copy()

    print(f"\\nPrecomputing V_opp geometries for up to {args.max_opps} apparitions...")
    t0 = time.time()
    V_opp = precompute_v_opp_array(orb, max_opps=args.max_opps)
    print(f"  done ({time.time() - t0:.1f}s)  array: V_opp{V_opp.shape}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))

    # --- Incumbent ---------------------------------------------------------
    print(f"\\nScoring incumbent N_OPP = {INCUMBENT_N_OPP} (production value)...")
    t0 = time.time()
    inc_rmse, inc_pdev, inc_r2 = score_opps(orb_base, y, folds, args.seed, V_opp, INCUMBENT_N_OPP, disc_penalty)
    print(f"  Incumbent: RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  R2={inc_r2:.5f}  ({time.time() - t0:.1f}s)\\n")

    # --- Sweep -------------------------------------------------------------
    results = []
    print("=" * 90)
    print(f"{'N_OPP':>7}  {'RMSE':>9}  {'Poisson':>9}  {'R2':>9}  {'imp/3':>5}  {'sd mean':>9}  {'sec':>5}")
    print("-" * 90)
    
    for n in range(args.min_opps, args.max_opps + 1):
        t0 = time.time()
        rmse, pdev, r2 = score_opps(orb_base, y, folds, args.seed, V_opp, n, disc_penalty)
        dt = time.time() - t0
        
        imp = (int(rmse < inc_rmse) + int(pdev < inc_pdev) + int(r2 > inc_r2))
        sd_mean = float(V_opp[:, :n].mean(axis=1).mean())
        marker = "  <-- beats incumbent" if imp >= 2 else ""
        
        print(f"{n:7d}  {rmse:9.5f}  {pdev:9.5f}  {r2:9.5f}  {imp:5d}  {sd_mean:9.5f}  {dt:5.1f}{marker}")
        results.append((n, rmse, pdev, r2))
        
    print("=" * 90)

    # --- Pick the best -----------------------------------------------------
    best = min(results, key=lambda r: r[2]) # index 2 is Poisson Deviance
    best_n, best_rmse, best_pdev, best_r2 = best

    print("\\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Incumbent  N_OPP = {INCUMBENT_N_OPP:d}   RMSE={inc_rmse:.5f}  Poisson={inc_pdev:.5f}  R2={inc_r2:.5f}")
    print(f"Best       N_OPP = {best_n:d}   RMSE={best_rmse:.5f}  Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
    
    d_rmse = best_rmse - inc_rmse
    d_pdev = best_pdev - inc_pdev
    d_r2 = best_r2 - inc_r2
    improvements = int(d_rmse < 0) + int(d_pdev < 0) + int(d_r2 > 0)
    
    print(f"Delta vs incumbent: RMSE={d_rmse:+.5f}  Poisson={d_pdev:+.5f}  R2={d_r2:+.5f}  ({improvements}/3 improved)")
    
    best_rmse_pt = min(results, key=lambda r: r[1])
    best_r2_pt = max(results, key=lambda r: r[3])
    
    print(f"\\nPer-metric optima:")
    print(f"  min RMSE    at N_OPP = {best_rmse_pt[0]:d}  (RMSE={best_rmse_pt[1]:.5f})")
    print(f"  min Poisson at N_OPP = {best[0]:d}  (Poisson={best[2]:.5f})")
    print(f"  max R2      at N_OPP = {best_r2_pt[0]:d}  (R2={best_r2_pt[3]:.5f})")

    if improvements >= 2:
        print(f"\\n==> Recommend changing N_OPP to {best_n:d} apparitions.")
        print(f"    In activityscope_utils.py feature_engineering, update:")
        print(f"        N_OPP = {best_n:d}")
    else:
        print(f"\\n==> No length clearly beats the incumbent N_OPP = {INCUMBENT_N_OPP:d}.")
    print("=" * 90)


if __name__ == "__main__":
    main()
