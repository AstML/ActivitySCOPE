"""tune_new_features.py

Stochastic switch-flipping search to evaluate five newly engineered features
against the current notebook base. The base feature set is the notebook's
`mlcols` minus `Is_Past_Threshold` and `Shared_Fold` (i.e. the regression
feature set without the CV grouping column). Each base feature plus each
new feature is a switch that can be flipped on or off.

------------------------------------------------------------------------
The five new candidate features
------------------------------------------------------------------------

1. vis_inc (the required vis-alternative; uses a, e, i, H_V; no tunable d)
       r_t = a (1 + e^2/2)                              # time-avg heliocentric
       Δ_inc = sqrt(r_t^2 - 2 r_t cos(i) + 1)            # Earth at (1,0,0),
                                                         #   object tilted by i
       vis_inc = H + 5 log10(r_t * Δ_inc)
   Inclination shows up *inside* the geocentric distance, not as a separate
   tunable offset. For i=0 this reduces to the standard opposition vis with
   no offset (d=0 case); for inclined orbits Δ_inc grows naturally, which
   captures the under-observation tendency of high-i orbits in a single
   visibility-style column.

2. mag_amplitude (orbit-averaged brightness range, no H dependence)
       q = a(1-e),   Q = a(1+e)
       mag_amplitude = 5 log10[ Q(Q-1) / max(q(q-1), eps) ]   # magnitudes
   Peak-to-trough opposition apparent-magnitude swing between perihelion
   and aphelion apparitions. Captures eccentricity-driven brightness
   variability in a way the linear `e` cannot.

3. synodic_period (Earth-object opposition recurrence)
       synodic_period = 1 / |1 - 1/P_orb|     # years between oppositions
   Different physical quantity from `orbital_period_sync` (distance to
   integer year). Hildas, JFCs and MBAs all have characteristic synodic
   cadences which matter for cumulative survey coverage.

4. gal_lat_perihelion (galactic latitude of the perihelion direction)
       (p_x, p_y, p_z) = perihelion-direction unit vector (ecliptic frame)
       gal_lat_perihelion = arcsin( p . n_gal )   # degrees
   Existing `galactic_inc` captures *orbital plane* vs galactic plane,
   but not WHERE on the orbit the object crosses crowded fields. If
   perihelion (the brightest apparition) sits at high galactic latitude
   the object is found more easily even on a galactic-plane-aligned
   orbit, and vice versa.

5. frac_high_lat (fraction of orbit outside surveys' ecliptic band)
       L_band = 10 deg
       frac_high_lat = 1 - (2/pi) arcsin( sin(L_band) / sin(i) ),   i > L
                       0,                                           i <= L
   Closed-form fraction of true-anomaly orbit (uniform-weighted) where
   the object's ecliptic latitude exceeds the survey band. Nonlinear
   saturating function of i, distinct from the raw i column.

A second wave (features 9–13) chases physical effects not covered above:

 9. vis_phase_corr (HG=0.15 phase-corrected typical visibility)
        α_max = arcsin(min(1/r_t, 1)),   α_typ = α_max / 2
        Φ(α)  = 0.85·Φ₁(α) + 0.15·Φ₂(α)   (Bowell HG with G=0.15)
        vis_phase_corr = H + 5 log10(r_t * (r_t - 1)) - 2.5 log10(Φ(α_typ))
    Existing vis_* columns ignore phase entirely (the paper notes this).
    Negligible correction for MBAs, large for NEOs — gives the model a
    direct handle on something it currently can't see.

10. moid_proxy (geometric proxy for Earth MOID)
        closest_helio = clip(1, q, Q)
        moid_proxy    = sqrt( (closest_helio - 1)^2 + (closest_helio·sin(i))^2 )
    For q > 1 → q-1 in plane; for Q < 1 → 1-Q; otherwise the orbit
    crosses 1 AU radially and the in-plane radial offset is 0 with only
    the inclination-induced out-of-plane component remaining.

11. bright_frac (orbit-averaged fraction with opposition m_V < 22)
        m_V(ν) = H + 5 log10(r(ν) * (r(ν)-1))
        bright_frac = Σ_ν 1[m_V(ν)<V_LIM]·r(ν)² / Σ_ν r(ν)²
    Time-weighted (Kepler's 2nd law) sample over true anomaly. One scalar
    that summarizes "how often is this thing actually detectable."

12. mean_opp_dec (time-weighted mean equatorial declination at opposition)
        ν-averaged arcsin( (y_ecl·sin(ε) + z_ecl·cos(ε)) / r )
    Distinct from dec_perihelion (which samples a single orbital phase).
    Captures systematic north/south bias in survey access for a given
    orbit, not just the latitude of one apparition.

13. solar_elong_quality (fraction of synodic period with elongation > 60°)
        cos(E) = (1 - r_t cos θ) / sqrt(1 + r_t² - 2 r_t cos θ)
        solar_elong_quality = mean_θ [ E > 60° ]
    Cadence partner to synodic_period: synodic_period says how often
    oppositions recur; this says how usable each apparition is.

------------------------------------------------------------------------
Search semantics
------------------------------------------------------------------------

State: a dict mapping every base feature and every new feature to a
bool. Active feature matrix has one column per True entry. A *segment*
begins with all five new features OFF and exactly one of them turned
ON (rotating through the five new features across segments). The
baseline feature matrix for the segment thus equals the base mlcols
plus one new feature.

Per iteration ("a flip"):
  1) With probability NEW_ONLY_PROB (default 2/3) restrict the flip pool
     to NEW_FEATURES only; otherwise (1/3 of the time) the pool is all
     features (base + new).
  2) With 50% probability k=1; with 50% probability k=2.
  3) Sample k distinct features from the chosen pool and toggle each.

Acceptance: improve >= 2 of (RMSE, Poisson deviance, R^2) — same rule
used by tune_vis_offsets.

Segment end: 100 flips per segment (configurable via --segment-iters).
The next segment starts fresh with the next new feature in the rotation.

Writes nothing to disk. Prints baseline, every iteration, periodic
segment-best summary, and final per-segment + global-best summary.

Run:
    python tune_new_features.py [--iters N] [--seed S] [--subsample K]
                                [--n-folds F] [--segment-iters M]
"""

import argparse
import json
import random
import signal
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


EPS = 1e-3

# Notebook mlcols minus Is_Past_Threshold (target) and Shared_Fold (CV grouping).
BASE_FEATURES = [
    "H", "Node", "a", "e", "i",
    "vis_typ", "vis_q", "vis_flux", "vis_timeavg", 
    "orbital_period_sync", "galactic_inc",
    "Perihelion_direction_x_e", "Perihelion_direction_y_e", "Perihelion_direction_z_e",
    "TJ", "dec_perihelion",
]

NEW_FEATURES = [
    "moid_proxy",
    "vis_inc",
    "gal_lat_perihelion",
    "vis_phase_corr",
    "mean_opp_dec",
    "synodic_period",
    "solar_elong_quality",
    "opp_angular_speed_typ",
    "lon_perihelion",
    "ecl_lat_perihelion",
]

# Galactic pole in J2000 ecliptic coords (matches activityscope_utils).
N_GAL = np.array([-0.86767, -0.00041, 0.49717])

# Ecliptic survey band used by frac_high_lat (degrees).
SURVEY_BAND_DEG = 10.0

# Earth obliquity (deg) for the ecliptic-to-equatorial rotation in mean_opp_dec.
OBL_DEG = 23.44

# Survey-depth proxy (V-band apparent magnitude) used by bright_frac.
V_LIM = 22.0

# Elongation threshold (deg) used by solar_elong_quality.
ELONG_THRESH_DEG = 60.0

# True-anomaly / synodic-azimuth samples per orbit for the orbit-averaged
# features (bright_frac, mean_opp_dec, solar_elong_quality).
N_ANOMALY_SAMPLES = 32

# Row-chunk size used for the orbit-averaged computations to bound memory.
ANOMALY_CHUNK_SIZE = 100_000

DEFAULT_SEGMENT_ITERS = 100

# Probability a flip is restricted to NEW_FEATURES only. The complement
# (1 - NEW_ONLY_PROB) lets the flip pool span every base + new feature.
NEW_ONLY_PROB = 2.0 / 3.0


# ---------------------------------------------------------------------------
# New feature engineering
# ---------------------------------------------------------------------------

def add_new_features(orb):
    """Attach every NEW_FEATURES column to `orb`. Assumes that
    `activityscope_utils.feature_engineering` has already run (so the
    Perihelion_direction_{x,y,z} columns exist) and that Orbital_period,
    Node, and Peri are present from the MPC database load."""
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_deg = orb["i"].to_numpy(dtype=np.float64)
    H = orb["H"].to_numpy(dtype=np.float64)
    i_rad = np.radians(i_deg)

    # 1) vis_inc — inclination-aware visibility, no tunable d.
    r_t = a * (1.0 + e ** 2 / 2.0)
    delta_inc = np.sqrt(np.maximum(r_t ** 2 - 2.0 * r_t * np.cos(i_rad) + 1.0, EPS))
    orb["vis_inc"] = (5.0 * np.log10(np.maximum(r_t, EPS) * delta_inc) + H).astype(float)

    # 2) mag_amplitude — opposition brightness range, perihelion -> aphelion.
    q = a * (1.0 - e)
    Q = a * (1.0 + e)
    num = np.maximum(Q * (Q - 1.0), EPS)
    den = np.maximum(q * (q - 1.0), EPS)
    orb["mag_amplitude"] = (5.0 * np.log10(num / den)).astype(float)

    # 3) synodic_period — years between consecutive oppositions, capped.
    P_orb = orb["Orbital_period"].to_numpy(dtype=np.float64)
    inv_synodic = np.abs(1.0 - 1.0 / np.maximum(P_orb, EPS))
    orb["synodic_period"] = np.minimum(
        1.0 / np.maximum(inv_synodic, 1e-4), 1e4
    ).astype(float)

    # 4) gal_lat_perihelion — galactic latitude of perihelion direction.
    px = orb["Perihelion_direction_x"].to_numpy(dtype=np.float64)
    py = orb["Perihelion_direction_y"].to_numpy(dtype=np.float64)
    pz = orb["Perihelion_direction_z"].to_numpy(dtype=np.float64)
    gal_dot = px * N_GAL[0] + py * N_GAL[1] + pz * N_GAL[2]
    orb["gal_lat_perihelion"] = np.degrees(
        np.arcsin(np.clip(gal_dot, -1.0, 1.0))
    ).astype(float)

    # 5) frac_high_lat — fraction of orbit beyond +/-SURVEY_BAND_DEG ecliptic.
    L_band = np.radians(SURVEY_BAND_DEG)
    sin_i = np.sin(i_rad)
    sin_L = np.sin(L_band)
    ratio = np.where(sin_i > sin_L, sin_L / np.maximum(sin_i, EPS), 1.0)
    frac = np.where(
        sin_i > sin_L,
        1.0 - (2.0 / np.pi) * np.arcsin(np.clip(ratio, 0.0, 1.0)),
        0.0,
    )
    orb["frac_high_lat"] = frac.astype(float)

    # 6) vis_Q — apparent magnitude at aphelion
    orb["vis_Q"] = (5.0 * np.log10(np.maximum(Q, EPS) * np.maximum(Q - 1.0, EPS)) + H).astype(float)
    
    # 7) ecl_lat_perihelion — ecliptic latitude of perihelion (degrees)
    orb["ecl_lat_perihelion"] = np.degrees(np.arcsin(np.clip(pz, -1.0, 1.0))).astype(float)
    
    # 8) opp_angular_speed_typ — approximate relative angular speed at typical opposition
    # Proportional to |1/sqrt(r_t) - 1| / |r_t - 1|
    speed_factor = np.abs(1.0 / np.maximum(np.sqrt(np.maximum(r_t, EPS)), EPS) - 1.0)
    orb["opp_angular_speed_typ"] = (speed_factor / np.maximum(np.abs(r_t - 1.0), EPS)).astype(float)

    # 9) vis_phase_corr — HG=0.15 phase-corrected typical visibility.
    # α_max = arcsin(min(1/r_t, 1)) is the geometric maximum phase angle;
    # α_typ = α_max / 2 stands in for the "typical" observed phase.  The
    # Bowell HG phase function with G=0.15 is then applied as a magnitude
    # penalty on top of the standard opposition-geometry vis.
    alpha_max = np.arcsin(np.minimum(1.0 / np.maximum(r_t, EPS), 1.0))
    alpha_typ = 0.5 * alpha_max
    tan_half_alpha = np.tan(alpha_typ / 2.0)
    th_safe = np.maximum(tan_half_alpha, 1e-12)
    phi1 = np.exp(-3.332 * np.power(th_safe, 0.631))
    phi2 = np.exp(-1.862 * np.power(th_safe, 1.218))
    phase_factor = 0.85 * phi1 + 0.15 * phi2
    phase_corr_mag = -2.5 * np.log10(np.maximum(phase_factor, EPS))
    delta_opp_t = np.maximum(r_t - 1.0, EPS)
    orb["vis_phase_corr"] = (
        5.0 * np.log10(np.maximum(r_t, EPS) * delta_opp_t) + H + phase_corr_mag
    ).astype(float)

    # 10) moid_proxy — geometric proxy for Earth MOID.
    # closest_helio = 1 AU clamped to [q, Q] is the closest 1D radial distance
    # the object reaches to Earth's solar distance.  The full proxy folds in
    # an out-of-plane offset closest_helio * sin(i) for inclined orbits.
    closest_helio = np.clip(1.0, q, Q)
    moid_radial = closest_helio - 1.0
    moid_oop = closest_helio * np.sin(i_rad)
    orb["moid_proxy"] = np.sqrt(moid_radial ** 2 + moid_oop ** 2).astype(float)

    # 11) jupiter_moid_approx — distance proxy for Jupiter MOID (approx 5.2 AU).
    # Similar method as earth moid proxy but using Jupiter's typical distance (approx 5.204 AU)
    a_jup = 5.204
    closest_helio_jup = np.clip(a_jup, q, Q)
    jup_moid_radial = closest_helio_jup - a_jup
    jup_moid_oop = closest_helio_jup * np.sin(i_rad) # Using object's inclination relative to ecliptic
    orb["jupiter_moid_approx"] = np.sqrt(jup_moid_radial ** 2 + jup_moid_oop ** 2).astype(float)

    # 13) lon_perihelion — Longitude of perihelion (Node + Peri)
    # Important for knowing which season perihelion oppositions happen in.
    orb["lon_perihelion"] = (orb["Node"].to_numpy(dtype=np.float64) + orb["Peri"].to_numpy(dtype=np.float64)) % 360.0

    # 14) bright_frac, 15) mean_opp_dec, 16) solar_elong_quality
    # All three are orbit-averaged quantities computed via anomaly / synodic
    # sampling.  Chunked over rows to bound peak memory at
    # ANOMALY_CHUNK_SIZE * N_ANOMALY_SAMPLES * ~8 bytes per temporary array.
    N = len(a)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))
    nu = np.linspace(0.0, 2.0 * np.pi, N_ANOMALY_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu)
    obl_rad = np.radians(OBL_DEG)
    cos_obl = np.cos(obl_rad)
    sin_obl = np.sin(obl_rad)
    elong_thresh_rad = np.radians(ELONG_THRESH_DEG)

    bright_frac = np.empty(N, dtype=np.float64)
    mean_opp_dec = np.empty(N, dtype=np.float64)
    solar_elong_quality = np.empty(N, dtype=np.float64)

    for start in range(0, N, ANOMALY_CHUNK_SIZE):
        end = min(start + ANOMALY_CHUNK_SIZE, N)
        a_c = a[start:end, None]
        e_c = e[start:end, None]
        H_c = H[start:end, None]
        Node_c = Node_rad[start:end, None]
        Peri_c = Peri_rad[start:end, None]
        i_c = i_rad[start:end, None]
        r_t_c = r_t[start:end, None]

        # Heliocentric distance and Kepler-2nd-law weights at each anomaly.
        r_orb = a_c * (1.0 - e_c ** 2) / (1.0 + e_c * cos_nu[None, :])
        r_safe = np.maximum(r_orb, EPS)
        weights = r_orb ** 2
        w_sum = np.maximum(weights.sum(axis=1), EPS)

        # 11) bright_frac — time-weighted fraction with opposition m_V < V_LIM.
        m_V = H_c + 5.0 * np.log10(r_safe * np.maximum(r_orb - 1.0, EPS))
        bright_frac[start:end] = (
            (m_V < V_LIM).astype(np.float64) * weights
        ).sum(axis=1) / w_sum

        # 12) mean_opp_dec — time-weighted mean equatorial declination of the
        # heliocentric position direction (which approximates the geocentric
        # direction at opposition).  Ecliptic z and y are computed per anomaly
        # then rotated to equatorial by the obliquity.
        u = Peri_c + nu[None, :]
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        y_ecl = r_orb * (np.sin(Node_c) * cos_u + np.cos(Node_c) * sin_u * np.cos(i_c))
        z_ecl = r_orb * sin_u * np.sin(i_c)
        z_eq = y_ecl * sin_obl + z_ecl * cos_obl
        dec_rad = np.arcsin(np.clip(z_eq / r_safe, -1.0, 1.0))
        mean_opp_dec[start:end] = np.degrees(
            (dec_rad * weights).sum(axis=1) / w_sum
        )

        # 13) solar_elong_quality — fraction of one synodic period with
        # solar elongation > ELONG_THRESH_DEG, evaluated at r = r_t.
        # cos(E) = (1 - r cos θ) / Δ,  Δ = sqrt(1 + r² - 2 r cos θ).
        Delta_sq = 1.0 + r_t_c ** 2 - 2.0 * r_t_c * cos_nu[None, :]
        Delta_safe = np.sqrt(np.maximum(Delta_sq, EPS))
        cos_E = (1.0 - r_t_c * cos_nu[None, :]) / Delta_safe
        E_rad = np.arccos(np.clip(cos_E, -1.0, 1.0))
        solar_elong_quality[start:end] = (E_rad > elong_thresh_rad).mean(axis=1)

    orb["bright_frac"] = bright_frac.astype(float)
    orb["mean_opp_dec"] = mean_opp_dec.astype(float)
    orb["solar_elong_quality"] = solar_elong_quality.astype(float)

    return orb


# ---------------------------------------------------------------------------
# State / move helpers
# ---------------------------------------------------------------------------

def initial_state(active_new_list):
    """Base features all ON, all new features OFF except those in `active_new_list`."""
    state = {f: True for f in BASE_FEATURES}
    for f in NEW_FEATURES:
        state[f] = (f in active_new_list)
    return state


def active_columns(state):
    return [f for f, on in state.items() if on]


def state_str(state):
    on_new = [f for f in NEW_FEATURES if state[f]]
    off_base = [f for f in BASE_FEATURES if not state[f]]
    parts = [f"new_on={'+'.join(on_new) if on_new else '(none)'}"]
    if off_base:
        parts.append(f"base_off={'+'.join(off_base)}")
    return "; ".join(parts)


def apply_flips(rng, state):
    """Return (trial_state, op_label, change_strs).

    With probability NEW_ONLY_PROB the flip pool is restricted to
    NEW_FEATURES; otherwise it is all features (base + new). Then 50/50
    k=1 vs k=2. The 'all features' branch can still happen to land on
    only new features by chance.
    """
    restrict_new = rng.random() < NEW_ONLY_PROB
    pool = list(NEW_FEATURES) if restrict_new else list(state.keys())
    k = 1 if rng.random() < 0.5 else 2
    k = min(k, len(pool))
    targets = rng.sample(pool, k)
    trial = dict(state)
    for t in targets:
        trial[t] = not trial[t]
    pool_tag = "new" if restrict_new else "any"
    change_strs = [f"{'+' if trial[t] else '-'}{t}" for t in targets]
    return trial, f"k={k}/{pool_tag}", change_strs


def _fit_predict_xgb(X_tr, y_tr, X_te, seed):
    model = XGBRegressor(
        objective="count:poisson",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=np.float64)


def _fit_predict_lgbm(X_tr, y_tr, X_te, seed):
    model = LGBMRegressor(
        objective="poisson",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=np.float64)


def evaluate_cv(X, y, folds, seed):
    rmses, pdevs, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        
        p_xgb = _fit_predict_xgb(X_tr, y_tr, X_te, seed)
        p_lgb = _fit_predict_lgbm(X_tr, y_tr, X_te, seed)
        y_pred = np.maximum(0.5 * (p_xgb + p_lgb), EPS)
        
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        pdevs.append(mean_poisson_deviance(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(rmses)), float(np.mean(pdevs)), float(np.mean(r2s))


def build_X(orb, state):
    return orb[active_columns(state)].astype(np.float32)


# ---------------------------------------------------------------------------
# Data preparation (mirrors tune_vis_offsets / the notebook)
# ---------------------------------------------------------------------------

def prepare_training_data(subsample=None, seed=0):
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases()
    # filter out objects marked in either of the two named filter csvs
    orb = orb[~orb["filtered_out"].astype(bool)]
    orb = utils.feature_engineering(orb)
    orb = add_new_features(orb)

    print("Merging cached extension_difficulty.csv...")
    extension_difficulty = pd.read_csv("extension_difficulty.csv")
    orb = orb.merge(extension_difficulty, on="Principal_desig", how="left")

    with open("known_active_objects.json", "r") as f:
        known_active = json.load(f)
    with open("dual_designation_list.json", "r") as f:
        dual_designation_list = json.load(f)
    print(len(orb))

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
        & ~orb["Number"].isin(dual_designation_list)
    ]
    common_dropna_cols = [
        "H", "a", "e", "i", "Node", "Peri", "Orbital_period", "Num_opps_minus_one",
        "vis_timeavg", "vis_typ", "vis_flux", "vis_q",
        "orbital_period_sync", "galactic_inc",
        "Perihelion_direction_x_e", "Perihelion_direction_y_e", "Perihelion_direction_z_e",
        "TJ", "dec_perihelion"
    ]
    orb = orb.dropna(subset=common_dropna_cols)
    print(len(orb))

    if subsample is not None and subsample < len(orb):
        orb = orb.sample(n=subsample, random_state=seed)
        print(f"Subsampled to {len(orb)} rows.")
    return orb.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=1000,
                        help="Total iterations across all segments (default: 1000).")
    parser.add_argument("--segment-iters", type=int, default=DEFAULT_SEGMENT_ITERS,
                        help=f"Flips per segment before rotating to the next "
                             f"new feature (default: {DEFAULT_SEGMENT_ITERS}).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV folds, subsample, XGB fit.")
    parser.add_argument("--search-seed", type=int, default=None,
                        help="Seed for the stochastic search (flip choices). "
                             "Default: int(time.time()) — pass to reproduce a run.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="CV folds (default 5, min 2).")
    parser.add_argument("--summary-every", type=int, default=25,
                        help="Print a segment-best summary every N iters.")
    args = parser.parse_args()

    if args.search_seed is None:
        args.search_seed = int(time.time())
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2.")
    if args.segment_iters < 1:
        parser.error("--segment-iters must be at least 1.")

    print(f"Deterministic seed (CV folds / XGB):  {args.seed}")
    print(f"Search seed (flip sampling):          {args.search_seed}  "
          f"(pass --search-seed {args.search_seed} to reproduce)")
    print(f"CV: {args.n_folds}-fold cross-validation")
    print(f"Segment length: {args.segment_iters} flips")
    print(f"Base features ({len(BASE_FEATURES)}): {BASE_FEATURES}")
    print(f"New features  ({len(NEW_FEATURES)}): {NEW_FEATURES}")
    rng = random.Random(args.search_seed)
    np.random.seed(args.seed)

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb["Num_opps_minus_one"].astype(np.float32).to_numpy()

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    print(f"Rows: {len(orb)}   {args.n_folds}-fold sizes: {[len(t) for _, t in folds]}")

    # Diagnostic: RMSE on the 16 BASE_FEATURES columns (all base ON, no new
    # features). Lets us cross-check that two simultaneously running tuners
    # see the same underlying data.
    print(f"\n[diagnostic] RMSE on the {len(BASE_FEATURES)} base columns: "
          f"{BASE_FEATURES}")
    X_diag = orb[BASE_FEATURES].astype(np.float32)
    t0 = time.time()
    diag_rmse, diag_pdev, diag_r2 = evaluate_cv(X_diag, y, folds, args.seed)
    dt = time.time() - t0
    print(f"[diagnostic] RMSE={diag_rmse:.5f}  Poisson={diag_pdev:.5f}  "
          f"R2={diag_r2:.5f}  ({dt:.1f}s for {args.n_folds} fits)")

    # SIGINT handler — flip a flag rather than raising (XGBoost can swallow signals).
    interrupted_flag = [False]

    def _sigint_handler(signum, frame):
        if interrupted_flag[0]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[second Ctrl-C — exiting immediately]")
            raise KeyboardInterrupt
        interrupted_flag[0] = True
        print("\n[Ctrl-C received — will exit after current iteration completes]")

    signal.signal(signal.SIGINT, _sigint_handler)

    # Global best across all segments.
    global_best_state = None
    global_best_rmse = float("inf")
    global_best_pdev = float("inf")
    global_best_r2 = -float("inf")

    def maybe_update_global(state, rmse, pdev, r2):
        nonlocal global_best_state, global_best_rmse, global_best_pdev, global_best_r2
        if global_best_state is None:
            global_best_state = dict(state)
            global_best_rmse, global_best_pdev, global_best_r2 = rmse, pdev, r2
            return True
        imp = (int(rmse < global_best_rmse)
               + int(pdev < global_best_pdev)
               + int(r2 > global_best_r2))
        if imp >= 2:
            global_best_state = dict(state)
            global_best_rmse, global_best_pdev, global_best_r2 = rmse, pdev, r2
            return True
        return False

    segment_history = []
    n_accept_total = 0
    n_reject_total = 0
    op_counts = {}  # op_label -> [accepts, rejects]

    it = 0
    seg_idx = 0
    interrupted = False

    try:
        while it < args.iters and not interrupted_flag[0]:
            # ------------------------------------------------------------
            # Start a fresh segment with two randomly selected new features ON.
            # ------------------------------------------------------------
            active_new_list = rng.sample(NEW_FEATURES, 2)
            state = initial_state(active_new_list)
            X = build_X(orb, state)

            print()
            print("=" * 78)
            print(f"Segment {seg_idx}: starting with {', '.join(active_new_list)} ON, "
                  f"all other new features OFF, all base features ON.")
            print(f"   {state_str(state)}")
            t0 = time.time()
            best_rmse, best_pdev, best_r2 = evaluate_cv(X, y, folds, args.seed)
            dt = time.time() - t0
            marker = ""
            if maybe_update_global(state, best_rmse, best_pdev, best_r2):
                marker = "  [NEW GLOBAL BEST]"
            print(f"   Segment baseline: RMSE={best_rmse:.5f}  "
                  f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}  "
                  f"({dt:.1f}s){marker}")
            print("=" * 78)

            best_state = dict(state)
            seg_start_iter = it + 1
            seg_accepts = 0
            seg_rejects = 0

            # ------------------------------------------------------------
            # Inner loop: run args.segment_iters flips.
            # ------------------------------------------------------------
            seg_it = 0
            while seg_it < args.segment_iters and it < args.iters:
                if interrupted_flag[0]:
                    interrupted = True
                    break
                it += 1
                seg_it += 1

                trial_state, op_label, change_strs = apply_flips(rng, best_state)
                if not any(trial_state.values()):
                    # Safety: never evaluate an empty feature matrix.
                    n_reject_total += 1
                    seg_rejects += 1
                    op_counts.setdefault(op_label, [0, 0])[1] += 1
                    print(f"[s{seg_idx} {it:4d}] reject  {op_label}  "
                          f"(would empty feature set; auto-rejected)")
                    continue

                X_trial = build_X(orb, trial_state)
                t0 = time.time()
                rmse, pdev, r2 = evaluate_cv(X_trial, y, folds, args.seed)
                dt = time.time() - t0

                improvements = (int(rmse < best_rmse)
                                + int(pdev < best_pdev)
                                + int(r2 > best_r2))
                accept = improvements >= 2

                op_counts.setdefault(op_label, [0, 0])
                global_marker = ""
                if accept:
                    op_counts[op_label][0] += 1
                    n_accept_total += 1
                    seg_accepts += 1
                    best_state = trial_state
                    best_rmse, best_pdev, best_r2 = rmse, pdev, r2
                    if maybe_update_global(best_state, best_rmse, best_pdev, best_r2):
                        global_marker = "  [NEW GLOBAL BEST]"
                else:
                    op_counts[op_label][1] += 1
                    n_reject_total += 1
                    seg_rejects += 1

                changes_str = " ".join(change_strs).ljust(50)
                print(
                    f"[s{seg_idx} {it:4d}] {'ACCEPT' if accept else 'reject'}  "
                    f"{op_label:9s}  {changes_str}  "
                    f"RMSE={rmse:.5f} ({rmse - best_rmse:+.5f})  "
                    f"Poisson={pdev:.5f} ({pdev - best_pdev:+.5f})  "
                    f"R2={r2:.5f} ({r2 - best_r2:+.5f})  "
                    f"imp={improvements}/3  "
                    f"Nfeat={sum(trial_state.values())}  "
                    f"({dt:.1f}s){global_marker}"
                )

                if args.summary_every and it % args.summary_every == 0:
                    print("    " + "-" * 70)
                    print(f"    Segment {seg_idx} best "
                          f"({sum(best_state.values())} feat): "
                          f"{state_str(best_state)}")
                    print(f"    Segment best metrics: RMSE={best_rmse:.5f}  "
                          f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
                    print(f"    GLOBAL best metrics:  RMSE={global_best_rmse:.5f}  "
                          f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}")
                    print("    " + "-" * 70)

            # ------------------------------------------------------------
            # Archive segment.
            # ------------------------------------------------------------
            segment_history.append({
                "index": seg_idx,
                "active_new": active_new_list,
                "start": seg_start_iter,
                "end": it,
                "state": dict(best_state),
                "rmse": best_rmse,
                "pdev": best_pdev,
                "r2": best_r2,
                "accepts": seg_accepts,
                "rejects": seg_rejects,
            })
            print()
            print(f"[END SEG {seg_idx}] start={active_new_list}  iters "
                  f"{seg_start_iter}-{it}  "
                  f"accepts={seg_accepts}  rejects={seg_rejects}")
            print(f"    final: RMSE={best_rmse:.5f}  Poisson={best_pdev:.5f}  "
                  f"R2={best_r2:.5f}")
            print(f"    final state: {state_str(best_state)}")
            seg_idx += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted — printing summary so far]")

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    if interrupted or interrupted_flag[0]:
        print(f"Stopped early at iteration {it} of {args.iters}.")
    else:
        print(f"Done after {it} iterations.")
    print(f"  Segments: {len(segment_history)}   "
          f"Total accepts: {n_accept_total}   Total rejects: {n_reject_total}")

    if op_counts:
        print(f"  By move type (accepted / total):")
        for op in sorted(op_counts):
            a, r = op_counts[op]
            print(f"    {op:9s}  {a:4d} / {a + r:4d}")

    print(f"\n  Per-segment final metrics:")
    for seg in segment_history:
        on_new = "+".join(f for f in NEW_FEATURES if seg["state"][f]) or "(none)"
        off_base = "+".join(f for f in BASE_FEATURES if not seg["state"][f]) or "(none)"
        print(f"    seg {seg['index']:2d}  start={str(seg['active_new']):25s}  "
              f"iters {seg['start']:4d}-{seg['end']:4d}   "
              f"RMSE={seg['rmse']:.5f}  Poisson={seg['pdev']:.5f}  "
              f"R2={seg['r2']:.5f}")
        print(f"           new on:   {on_new}")
        print(f"           base off: {off_base}")

    # New-feature usage across segments — does each one survive to the end?
    if segment_history:
        print(f"\n  New-feature retention rate "
              f"(across {len(segment_history)} segments):")
        for f in NEW_FEATURES:
            n_on = sum(1 for s in segment_history if s["state"][f])
            print(f"    {f:22s}  retained in {n_on}/{len(segment_history)} "
                  f"({100*n_on/len(segment_history):5.1f}%)")
        print(f"\n  Base-feature drop rate "
              f"(across {len(segment_history)} segments):")
        for f in BASE_FEATURES:
            n_off = sum(1 for s in segment_history if not s["state"][f])
            if n_off > 0:
                print(f"    {f:30s}  dropped in {n_off}/{len(segment_history)} "
                      f"({100*n_off/len(segment_history):5.1f}%)")

    if global_best_state is not None:
        print(f"\n*** GLOBAL BEST across all segments ***")
        print(f"  RMSE={global_best_rmse:.5f}  Poisson={global_best_pdev:.5f}  "
              f"R2={global_best_r2:.5f}")
        on_new = "+".join(f for f in NEW_FEATURES if global_best_state[f]) or "(none)"
        off_base = "+".join(f for f in BASE_FEATURES if not global_best_state[f]) or "(none)"
        print(f"  new on:   {on_new}")
        print(f"  base off: {off_base}")
        cols = active_columns(global_best_state)
        print(f"  active columns ({len(cols)}): {cols}")
    print("=" * 78)


if __name__ == "__main__":
    main()
