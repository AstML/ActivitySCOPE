"""
ActivitySCOPE Utilities Module

This module contains utility functions for:
- Loading orbital databases (MPC, AstDyS, JPL)
- Loading astrometry counts
- Comparing databases
- Feature engineering for machine learning models
- Model hyperparameters and scoring functions
- Extension difficulty classification
"""

import os
import shutil
import time
import urllib.request
import pandas as pd
import numpy as np
from sbpy.data import Names
import json
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import cross_val_predict
from autogluon.core.metrics import make_scorer
from xgboost import XGBClassifier


# ==============================================================================
# MPC ORBIT DOWNLOAD CACHE
# ==============================================================================

_MPCORB_URL = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"
_MPCORB_CACHE_PATH = ".cache/mpcorb_extended.json.gz"
_MPCORB_CACHE_TTL_SECONDS = 60*20  # 20 minutes


def get_mpcorb_extended_path(max_age_seconds=_MPCORB_CACHE_TTL_SECONDS):
    """
    Return a local path to mpcorb_extended.json.gz, downloading it to a local
    cache on first use and reusing it on subsequent calls within the TTL.
    Pass max_age_seconds=0 to force a refresh.
    """
    cache_dir = os.path.dirname(_MPCORB_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    fresh = (
        os.path.exists(_MPCORB_CACHE_PATH)
        and (time.time() - os.path.getmtime(_MPCORB_CACHE_PATH)) < max_age_seconds
    )
    if not fresh:
        tmp_path = _MPCORB_CACHE_PATH + ".part"
        if os.path.exists(_MPCORB_CACHE_PATH):
            os.remove(_MPCORB_CACHE_PATH)

        req = urllib.request.Request(
            _MPCORB_URL,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"},
        )
        with urllib.request.urlopen(req) as r, open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f)

        os.replace(tmp_path, _MPCORB_CACHE_PATH)
    return _MPCORB_CACHE_PATH


# ==============================================================================
# MODEL HYPERPARAMETERS AND SCORING
# ==============================================================================

# Hyperparameters for binary classification models
HYPERPARAMETERS_BINARY = {
    "GBM": [
        {},
        {
            "learning_rate": 0.03,
            "num_leaves": 128,
            "feature_fraction": 0.9,
            "min_data_in_leaf": 3,
            "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
        },
    ],
    "XGB": [{}, {"learning_rate": 0.5, "max_depth": 3, "min_child_weight": 18, "subsample": 1.0, "ag_args": {"name_suffix": "_tuned"}}]
}

# Hyperparameters for Poisson regression models
HYPERPARAMETERS_POISSON = {
    'GBM': {'objective': 'poisson', 'num_iterations': 1000, 'learning_rate': 0.1},
    'XGB': {'objective': 'count:poisson'},
    'CAT': {'objective': 'Poisson'}
}

# Custom Poisson scorer for regression model evaluation
POISSON_SCORER = make_scorer(
    name='mean_poisson_deviance',
    score_func=mean_poisson_deviance,
    optimum=0,
    greater_is_better=False
)


# ==============================================================================
# ORBITAL DATABASE LOADING
# ==============================================================================

def load_mpc_orbits(apply_filters=True):
    """
    Load the MPC orbit database and apply initial processing.
    
    Parameters
    ----------
    apply_filters : bool, optional
        Whether to apply filter lists (default: True)
    
    Returns
    -------
    pd.DataFrame
        Processed MPC orbit database
    """
    orb = pd.read_json(get_mpcorb_extended_path(), compression='gzip')

    # MPC leaves the U parameter blank for some older objects.
    # Any recent object should have a U, but some older ones without a well defined orbit 
    # are deemed lost and have a blank H.
    # For our purposes we fill it with 10 here, even though some objects with missing U 
    # actually have a much better defined orbit than U=9.
    orb['U'] = pd.to_numeric(orb['U'], errors='coerce').fillna(10)
    
    orb = orb.convert_dtypes()
    orb.drop(["Other_desigs"], axis=1, inplace=True)
    
    if apply_filters:
        # Filter based on filter lists
        filter_until_further_notice = pd.read_csv("filter until further notice.csv")
        filter_out_unless_updated = pd.read_csv("filter out unless updated.csv")
        
        orb = orb[~orb["Principal_desig"].isin(filter_until_further_notice["Object"])]
        
        for _, row in filter_out_unless_updated.iterrows():
            orb = orb[~((orb["Principal_desig"] == row["Object"]) & 
                       (orb["Arc_length"] == row["Arc_length"]))]
    
    return orb


def load_astrometry_counts(orb):
    """
    Load astrometry counts and merge with orbit dataframe.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe to merge with astrometry counts
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with astrometry counts merged
    """
    with open('./astrometry_counter/astrometry_counts.json', 'r') as f:
        astrometry_counts = json.load(f)
    
    astrometry_counts_df = pd.DataFrame.from_dict(astrometry_counts, orient='index')
    astrometry_counts_df.index = astrometry_counts_df.index.map(
        lambda x: Names.from_packed(x).format("desig")
    )
    astrometry_counts_df.index.name = "Principal_desig"
    astrometry_counts_df["other_opps"] = (
        astrometry_counts_df["nights_total"] - 
        astrometry_counts_df["opp_with_most_nights"]
    )
    
    orb = orb.merge(astrometry_counts_df, how="left", 
                   left_on="Principal_desig", right_index=True)
    
    return orb


def load_astdys_orbits():
    """
    Load AstDyS orbit database (multi-opposition and single-opposition).
    
    Returns
    -------
    pd.DataFrame
        AstDyS orbit database
    """
    astdys_names = ["Astdys Name", "Epoch-MJD", "a", "e", "i", "Node", "Peri", 
                    "M", "H", "G", "rand"]
    astdys_widths = [15, 12, 25, 25, 25, 25, 25, 25, 6, 6, 3]
    
    # Multi-opposition objects
    astdys = pd.read_fwf(
        "https://newton.spacedys.com/~astdys2/catalogs/ufitobs.cat",
        index_col=False, names=astdys_names, widths=astdys_widths, skiprows=6
    )
    astdys["Astdys Multiopp"] = 1
    
    # Single-opposition objects
    astdys_sing = pd.read_fwf(
        "https://newton.spacedys.com/~astdys2/catalogs/singopp.cat",
        index_col=False, names=astdys_names, widths=astdys_widths, skiprows=6
    )
    astdys_sing["Astdys Multiopp"] = 0
    
    # Combine
    astdys = pd.concat([astdys, astdys_sing])
    print(f"Loaded {len(astdys)} AstDyS orbits")
    
    astdys = astdys.convert_dtypes()
    astdys["n"] = 360 / (astdys["a"])**1.5 / 365.2569
    astdys["Perihelion_dist"] = astdys["a"] * (1 - astdys["e"])
    astdys["Aphelion_dist"] = astdys["a"] * (1 + astdys["e"])
    astdys["Epoch"] = astdys["Epoch-MJD"] + 2400000.5
    astdys["Astdys Name"] = astdys["Astdys Name"].str.replace("'", "")
    astdys["Astdys Name"] = (astdys["Astdys Name"].str.slice(0, 4) + " " + 
                             astdys["Astdys Name"].str.slice(4))
    astdys["Ref"] = "AstDyS"
    
    return astdys


def load_jpl_orbits():
    """
    Load JPL orbit database.
    
    Returns
    -------
    pd.DataFrame
        JPL orbit database
    """
    jpl_names = ["Desig", "Epoch-MJD", "a", "e", "i", "Peri", "Node", "M", 
                 "H", "G", "Ref"]
    jpl_widths = [14, 6, 12, 11, 10, 10, 10, 12, 6, 5, 10]
    
    jpl = pd.read_fwf(
        "https://ssd.jpl.nasa.gov/dat/ELEMENTS.UNNUM.gz",
        compression='gzip', index_col=False, names=jpl_names, 
        widths=jpl_widths, skiprows=2
    )
    jpl = jpl.convert_dtypes()
    
    return jpl


# ==============================================================================
# DATABASE COMPARISON
# ==============================================================================

def compare_with_astdys(orb, astdys):
    """
    Compare MPC orbits with AstDyS and add comparison metrics.
    
    Parameters
    ----------
    orb : pd.DataFrame
        MPC orbit dataframe
    astdys : pd.DataFrame
        AstDyS orbit dataframe
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with AstDyS comparison columns added
    """
    orb = orb.merge(
        astdys[["Astdys Name", "a", "e", "i", "H", "Astdys Multiopp"]], 
        how="left", 
        left_on="Principal_desig", 
        right_on="Astdys Name", 
        suffixes=("", "_astdys")
    )
    
    # Set multi_opp_disagree to 1 any time Astdys Multiopp is defined and 
    # disagrees with Num_opps
    mpc_multiopp = (orb["Num_opps"] > 1).astype(int)
    orb["multi_opp_disagree"] = (
        (orb["Astdys Multiopp"].notna()) & 
        (orb["Astdys Multiopp"] != mpc_multiopp)
    ).astype(int)
    
    return orb


def compare_with_jpl(orb, jpl):
    """
    Compare MPC orbits with JPL and add comparison metrics.
    
    Parameters
    ----------
    orb : pd.DataFrame
        MPC orbit dataframe
    jpl : pd.DataFrame
        JPL orbit dataframe
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with JPL comparison columns added
    """
    orb = orb.merge(
        jpl[["Desig", "H"]], 
        how="left", 
        left_on="Principal_desig", 
        right_on="Desig", 
        suffixes=("", "_jpl")
    )
    
    return orb


def compute_database_differences(orb):
    """
    Compute differences between MPC, AstDyS, and JPL orbital elements.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe with MPC, AstDyS, and JPL data
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with difference columns added
    """
    # For any H above 33, make it into an NA
    orb.loc[orb['H'] > 33, 'H'] = pd.NA
    orb['H_MPC'] = orb['H']
    orb.loc[orb['H_astdys'] > 33, 'H_astdys'] = pd.NA
    orb.loc[orb['H_jpl'] > 33, 'H_jpl'] = pd.NA
    
    # Figure out difference between H and H_astdys
    orb["H_diff_abs"] = (orb["H"] - orb["H_astdys"]).abs()
    orb["a_diff_abs"] = (orb["a"] - orb["a_astdys"]).abs()
    orb["e_diff_abs"] = (orb["e"] - orb["e_astdys"]).abs()
    orb["i_diff_abs"] = (orb["i"] - orb["i_astdys"]).abs()
    
    # Fill NAs with small default values
    orb["H_diff_abs"] = orb["H_diff_abs"].fillna(0.011)
    orb["a_diff_abs"] = orb["a_diff_abs"].fillna(0)
    orb["e_diff_abs"] = orb["e_diff_abs"].fillna(0)
    orb["i_diff_abs"] = orb["i_diff_abs"].fillna(0)
    
    # For JPL, we will only compare H for now as that's the thing we most need 
    # to be certain of
    orb["H_diff_abs_jpl"] = (orb["H"] - orb["H_jpl"]).abs()
    orb["H_diff_abs_jpl"] = orb["H_diff_abs_jpl"].fillna(0.012)
    
    orb["H_diff_abs_max"] = orb[["H_diff_abs", "H_diff_abs_jpl"]].max(axis=1)
    
    return orb


def apply_magnitude_corrections(orb, corrections_file="absolute magnitude fixes.csv"):
    """
    Apply corrected H magnitudes from known photometry issues.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    corrections_file : str, optional
        Path to corrections CSV file
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with corrections applied
    """
    corrections = pd.read_csv(corrections_file)
    for _, row in corrections.iterrows():
        orb.loc[orb["Principal_desig"] == row["Object"], "H"] = row["Corrected H"]
    
    return orb


def apply_nights_overrides(orb, overrides_file="nights_override.csv"):
    """
    Apply overrides for nights_total when the CSV specifies a higher value.

    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    overrides_file : str, optional
        Path to overrides CSV file

    Returns
    -------
    pd.DataFrame
        Orbit dataframe with overrides applied
    """
    try:
        overrides = pd.read_csv(overrides_file, skipinitialspace=True)
        for _, row in overrides.iterrows():
            mask = orb["Principal_desig"] == row["Object"]
            if mask.any():
                current_nights = orb.loc[mask, "nights_total"].values[0]
                if pd.isna(current_nights) or row["Min Nights"] > current_nights:
                    orb.loc[mask, "nights_total"] = row["Min Nights"]
    except FileNotFoundError:
        pass

    return orb


def apply_num_opps_overrides(orb, overrides_file="num_opps_overrides.csv"):
    """
    Apply overrides for number of oppositions when the CSV specifies a higher value.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    overrides_file : str, optional
        Path to overrides CSV file
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with overrides applied
    """
    try:
        overrides = pd.read_csv(overrides_file, skipinitialspace=True)
        for _, row in overrides.iterrows():
            mask = orb["Principal_desig"] == row["Object"]
            if mask.any():
                current_opps = orb.loc[mask, "Num_opps"].values[0]
                if row["Opps"] > current_opps:
                    orb.loc[mask, "Num_opps"] = row["Opps"]
    except FileNotFoundError:
        pass
    
    # Anytime the number of oppositions is >= 2, set Arc_length to NaN
    orb.loc[orb["Num_opps"] >= 2, "Arc_length"] = np.nan
    
    return orb


def add_training_targets(orb, num_opps_threshold=4):
    """
    Add binary classification and regression targets for training.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe
    num_opps_threshold : int, optional
        Threshold for binary classification (default: 4)
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with training target columns added
    """
    # Indicator variable, the training target for binary classification model
    orb["Is_Past_Threshold"] = (orb["Num_opps"] >= num_opps_threshold) * 1
    
    # This is the training target for the regression model as we take the first 
    # opposition as granted (it wouldn't be designated if it hadn't been observed 
    # at least once) then it's the number of additional oppositions beyond the 
    # first one that we are trying to predict, which is just Num_opps - 1
    orb["Num_opps_minus_one"] = orb["Num_opps"] - 1
    
    return orb


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def feature_engineering(orb):
    """
    Engineer predictive features from orbital elements.
    
    This function adds visibility metrics, orbital dynamics features, and 
    geometric properties to the dataframe.
    
    Parameters
    ----------
    orb : pd.DataFrame
        Orbit dataframe with at least columns: a, e, i, H, Peri, Node
    
    Returns
    -------
    pd.DataFrame
        Orbit dataframe with engineered features added (modifies in place)
    """
    # ============================================================================
    # VISIBILITY FEATURES
    # ============================================================================
    
    a = orb['a']
    e = np.clip(orb['e'], 0, 0.999)
    H = orb['H']
    eps_val = 1e-3
    
    # vis_timeavg    d = 0.9
    # Typical-geometry visibility magnitude, with both distance factors taken as
    # orbit-averages:
    #   r = a(1 + e^2/2) is the time-averaged heliocentric distance: an asteroid
    #       spends more time near aphelion (Kepler's 2nd law), so <r>_t > a.
    #   Delta = sqrt(r^2 - 1) = sqrt((r - 1)(r + 1)) is the geometric mean of the
    #       opposition (r - 1, closest) and conjunction (r + 1, farthest)
    #       geocentric distances. Because magnitude is logarithmic in distance
    #       (the 5 log10 Delta term), this geometric mean is the distance whose
    #       magnitude equals the average of the best- and worst-case observing
    #       geometries -- a "typical visibility distance".
    #   d (AU) is a small empirical calibration: surveys preferentially detect
    #       objects near opposition, so the detection-weighted distance sits below
    #       the symmetric mean; d nudges Delta toward the near end. Fitted, not
    #       derived.
    d_timeavg = 0.9
    r_timeavg = a * (1.0 + e**2 / 2.0)
    delta_geom = np.sqrt(np.maximum(r_timeavg**2 - 1.0, eps_val))
    delta_timeavg = np.maximum(delta_geom - d_timeavg, eps_val)
    orb['vis_timeavg'] = (5.0 * np.log10(np.maximum(r_timeavg, eps_val) * delta_timeavg) + H).astype(float)

    # vis_typ
    d_vis_typ = 0.51
    r_vis_typ = a * (1.0 + e / 2.0)
    delta_vis_typ = np.maximum(r_vis_typ - d_vis_typ, eps_val)
    orb['vis_typ'] = (5.0 * np.log10(np.maximum(r_vis_typ, eps_val) * delta_vis_typ) + H).astype(float)

    # vis_flux
    d_flux = 1
    r_flux = a * np.power(np.maximum(1.0 - e**2, 0.0), 0.25)
    delta_flux = np.maximum(r_flux - d_flux, eps_val)
    orb['vis_flux'] = (5.0 * np.log10(np.maximum(r_flux, eps_val) * delta_flux) + H).astype(float)

    # vis_q
    d_vis_q = 0.8
    r_vis_q = a * (1.0 - e)
    delta_vis_q = np.maximum(r_vis_q - d_vis_q, eps_val)
    orb['vis_q'] = (5.0 * np.log10(np.maximum(r_vis_q, eps_val) * delta_vis_q) + H).astype(float)
    
    # vis_inc
    r_t = a * (1.0 + e**2 / 2.0)
    i_rad_temp = np.radians(orb['i'])
    delta_inc = np.sqrt(np.maximum(r_t**2 - 2.0 * r_t * np.cos(i_rad_temp) + 1.0, eps_val))
    orb['vis_inc'] = (5.0 * np.log10(np.maximum(r_t, eps_val) * delta_inc) + H).astype(float)
    
    # ============================================================================
    # ORBITAL DYNAMICS FEATURES
    # ============================================================================
    
    # Orbital period resonance with Earth
    # Measures how closely the orbital period matches an integer number of years
    orb['orbital_period_sync'] = np.abs(
        orb['Orbital_period'] - np.round(orb['Orbital_period'])
    )
    
    # Tisserand parameter relative to Jupiter
    # Distinguishes dynamical classes (asteroids vs comets, Trojans, etc.)
    orb["TJ"] = (5.203 / orb["a"] + 
                 2 * np.cos(np.radians(orb["i"])) * 
                 np.sqrt(orb["a"] / 5.203 * (1 - orb["e"]**2)))
    
    # Jupiter Trojan classification
    # Objects in 1:1 resonance with Jupiter near L4/L5 Lagrange points
    orb["is_trojan"] = ((orb["a"] > 5.0) & (orb["a"] < 5.4) &
                        (orb["e"] < 0.3) & (orb["i"] < 40)).astype(int)
    
    # ============================================================================
    # GEOMETRIC FEATURES
    # ============================================================================
    
    # Perihelion direction unit vector in heliocentric ecliptic coordinates
    # Captures seasonal visibility patterns based on perihelion orientation
    perihelion_directions = np.array([
        np.cos(np.radians(orb["Node"])) * np.cos(np.radians(orb["Peri"])) - 
        np.sin(np.radians(orb["Node"])) * np.sin(np.radians(orb["Peri"])) * 
        np.cos(np.radians(orb["i"])),
        
        np.sin(np.radians(orb["Node"])) * np.cos(np.radians(orb["Peri"])) + 
        np.cos(np.radians(orb["Node"])) * np.sin(np.radians(orb["Peri"])) * 
        np.cos(np.radians(orb["i"])),
        
        np.sin(np.radians(orb["Peri"])) * np.sin(np.radians(orb["i"]))
    ])
    
    orb["Perihelion_direction_x"] = perihelion_directions[0]
    orb["Perihelion_direction_y"] = perihelion_directions[1]
    orb["Perihelion_direction_z"] = perihelion_directions[2]
    
    # Eccentricity-weighted perihelion vectors
    # For circular orbits (e≈0), perihelion direction is undefined; 
    # weighting by e resolves this
    orb["Perihelion_direction_x_e"] = orb["Perihelion_direction_x"] * orb["e"]
    orb["Perihelion_direction_y_e"] = orb["Perihelion_direction_y"] * orb["e"]
    orb["Perihelion_direction_z_e"] = orb["Perihelion_direction_z"] * orb["e"]
    
    # Declination of perihelion
    # Accounts for northern vs southern hemisphere observational bias
    eps = np.radians(23.44)  # Earth's axial tilt
    i_rad = np.radians(orb['i'])
    node_rad = np.radians(orb['Node'])
    peri_rad = np.radians(orb['Peri'])
    
    sin_dec = (np.sin(i_rad) * np.sin(peri_rad) * np.cos(eps) + 
               (np.cos(peri_rad) * np.sin(node_rad) + 
                np.sin(peri_rad) * np.cos(i_rad) * np.cos(node_rad)) * 
               np.sin(eps))
    orb['dec_perihelion'] = np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    
    # Galactic plane alignment
    # Angle between orbital plane and galactic plane (affects stellar 
    # background density)
    n_gal = np.array([-0.86767, -0.00041, 0.49717])  # J2000 NGP in J2000 ecliptic coords
    n_ast = [
        np.sin(np.radians(orb["i"])) * np.sin(np.radians(orb["Node"])),
        -np.sin(np.radians(orb["i"])) * np.cos(np.radians(orb["Node"])),
        np.cos(np.radians(orb["i"]))
    ]
    orb["galactic_inc"] = np.degrees(np.arccos(np.clip(
        n_gal[0]*n_ast[0] + n_gal[1]*n_ast[1] + n_gal[2]*n_ast[2], -1.0, 1.0
    )))
    
    # Combined angular elements (exploratory feature)
    orb["node_plus_peri"] = (orb["Node"] + orb["Peri"]) % 360
    
    # ============================================================================
    # ORBIT-AVERAGED FEATURES
    # ============================================================================
    
    N_ANOMALY_SAMPLES = 32
    ANOMALY_CHUNK_SIZE = 100_000
    N = len(orb)
    Node_rad = np.radians(orb["Node"].to_numpy(dtype=np.float64))
    Peri_rad = np.radians(orb["Peri"].to_numpy(dtype=np.float64))
    nu_arr = np.linspace(0.0, 2.0 * np.pi, N_ANOMALY_SAMPLES, endpoint=False)
    cos_nu = np.cos(nu_arr)
    obl_rad = np.radians(23.44)
    cos_obl = np.cos(obl_rad)
    sin_obl = np.sin(obl_rad)
    elong_thresh_rad = np.radians(60.0)

    # spatial_discoverability_fraction thresholds and HG phase-function constants
    V_LIM = 22.5
    LAT_LIM_RAD = np.radians(25.0)
    HG_A1, HG_B1 = 3.33, 0.63
    HG_A2, HG_B2 = 1.87, 1.22
    HG_G = 0.15
    PHI_FLOOR = 1e-30

    # mean_opp_dec_arr = np.empty(N, dtype=np.float64)
    spatial_disc_arr = np.empty(N, dtype=np.float64)
    dec_flux_weighted_arr = np.empty(N, dtype=np.float64)
    dec_orbit_min_arr = np.empty(N, dtype=np.float64)
    frac_flux_south30_arr = np.empty(N, dtype=np.float64)
    vis_orbit_flux_opp_arr = np.empty(N, dtype=np.float64)
    # vis_mag_timeavg_arr = np.empty(N, dtype=np.float64)
    vis_orbit_flux_multi_arr = np.empty(N, dtype=np.float64)
    vis_orbit_mag_multi_arr = np.empty(N, dtype=np.float64)

    # Earth heliocentric-longitude offsets for vis_orbit_flux_multi.
    EARTH_LON_OFFSETS_RAD = np.radians(np.array([0.0, 30.0, 60.0]))

    a_np = orb['a'].to_numpy(dtype=np.float64)
    e_np = np.clip(orb['e'].to_numpy(dtype=np.float64), 0.0, 0.999)
    i_np = np.radians(orb['i'].to_numpy(dtype=np.float64))
    H_np = orb['H'].to_numpy(dtype=np.float64)
    r_t_np = a_np * (1.0 + e_np ** 2 / 2.0)

    for start in range(0, N, ANOMALY_CHUNK_SIZE):
        end = min(start + ANOMALY_CHUNK_SIZE, N)
        a_c = a_np[start:end, None]
        e_c = e_np[start:end, None]
        Node_c = Node_rad[start:end, None]
        Peri_c = Peri_rad[start:end, None]
        i_c = i_np[start:end, None]
        H_c = H_np[start:end, None]
        r_t_c = r_t_np[start:end, None]

        # Heliocentric distance and Kepler-2nd-law weights at each anomaly.
        r_orb = a_c * (1.0 - e_c ** 2) / (1.0 + e_c * cos_nu[None, :])
        r_safe = np.maximum(r_orb, eps_val)
        weights = r_orb ** 2
        w_sum = np.maximum(weights.sum(axis=1), eps_val)

        # Heliocentric ecliptic coordinates of the asteroid at each nu
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

        # # mean_opp_dec
        # z_eq = y_ecl * sin_obl + z_ecl * cos_obl
        # dec_rad = np.arcsin(np.clip(z_eq / r_safe, -1.0, 1.0))
        # mean_opp_dec_arr[start:end] = np.degrees(
        #     (dec_rad * weights).sum(axis=1) / w_sum
        # )

        # spatial_discoverability_fraction
        # Place an idealised Earth at the asteroid's heliocentric ecliptic longitude
        # at r=1 AU in the ecliptic plane (z=0), then compute the apparent V via the
        # IAU HG phase function and check the geocentric ecliptic latitude.
        lambda_k = np.arctan2(y_ecl, x_ecl)
        dx = x_ecl - np.cos(lambda_k)
        dy = y_ecl - np.sin(lambda_k)
        dz = z_ecl  # Earth z = 0
        Delta_opp = np.sqrt(dx * dx + dy * dy + dz * dz)
        Delta_opp_safe = np.maximum(Delta_opp, eps_val)
        # Phase angle via law of cosines (Sun-asteroid-Earth, vertex at asteroid)
        cos_alpha = np.clip(
            (r_orb ** 2 + Delta_opp ** 2 - 1.0) / (2.0 * r_safe * Delta_opp_safe),
            -1.0, 1.0
        )
        alpha = np.arccos(cos_alpha)
        tan_half = np.tan(alpha / 2.0)
        tan_half_safe = np.maximum(tan_half, 0.0)
        phi1 = np.exp(-HG_A1 * np.power(tan_half_safe, HG_B1))
        phi2 = np.exp(-HG_A2 * np.power(tan_half_safe, HG_B2))
        phi_blend = np.maximum((1.0 - HG_G) * phi1 + HG_G * phi2, PHI_FLOOR)
        V_app = (H_c
                 + 5.0 * np.log10(r_safe * Delta_opp_safe)
                 - 2.5 * np.log10(phi_blend))
        # Geocentric ecliptic latitude (not heliocentric -- matches what surveys see)
        beta_geo = np.arcsin(np.clip(dz / Delta_opp_safe, -1.0, 1.0))
        passed = (V_app <= V_LIM) & (np.abs(beta_geo) <= LAT_LIM_RAD)
        spatial_disc_arr[start:end] = (
            (passed * weights).sum(axis=1) / w_sum
        )

        # vis_orbit_flux_opp: flux-weighted mean apparent V over the orbit, in mag.
        # Averaging in linear flux (not in mag) is the physically correct way to
        # combine an exponentially distributed observable: a V=19 segment of the
        # orbit contributes ~40x more flux than a V=23 segment, so the brightest
        # portions dominate the result. The Kepler 2nd-law weights w_k = r^2 give
        # equal-time sampling.
        # Floor at 1e-30 (V ~= 75) just to keep log10 finite in pathological cases;
        # do NOT reuse the distance-scale eps_val here, which is ~6 orders of
        # magnitude larger than typical asteroid fluxes and would clamp every row.
        flux_app = np.power(10.0, -0.4 * V_app)
        flux_mean = (flux_app * weights).sum(axis=1) / w_sum
        vis_orbit_flux_opp_arr[start:end] = (
            -2.5 * np.log10(np.maximum(flux_mean, 1e-30))
        )

        # ----------------------------------------------------------------------
        # FLUX-WEIGHTED DECLINATION OF THE APPARITIONS
        # A more rigorous replacement for dec_perihelion (a single heliocentric
        # instant): asks where, in equatorial declination, the orbit's *observable*
        # light actually comes from. Uses the same idealised opposition Earth
        # (1 AU, in-ecliptic) as spatial_discoverability_fraction. The geocentric
        # vector (dx, dy, dz) is in the ecliptic frame; rotating about the x-axis
        # by the obliquity gives the equatorial z component, hence the geocentric
        # declination an observer would see at that orbital phase.
        z_eq_geo = dy * sin_obl + dz * cos_obl
        dec_geo = np.arcsin(np.clip(z_eq_geo / Delta_opp_safe, -1.0, 1.0))
        dec_geo_deg = np.degrees(dec_geo)

        # Flux x time weights. flux_app emphasises the brightest apparitions (a
        # V=19 phase outweighs a V=23 phase ~40x), while the r^2 Kepler-2nd-law
        # weights convert the uniform-in-true-anomaly samples to equal time. The
        # product is the physically correct weighting for "where does the light
        # we would actually receive over an orbit sit in declination."
        w_ft = flux_app * weights
        w_ft_sum = np.maximum(w_ft.sum(axis=1), 1e-30)

        # dec_flux_weighted: flux- and time-weighted mean geocentric declination.
        # Declination is bounded to [-90, 90] and never wraps, so the weighted
        # arithmetic mean is well defined (no circular-mean discontinuity).
        dec_flux_weighted_arr[start:end] = (
            (dec_geo_deg * w_ft).sum(axis=1) / w_ft_sum
        )

        # dec_orbit_min: southernmost declination reached at opposition anywhere on
        # the orbit (unweighted extreme) -- "how far south can this object ever get."
        dec_orbit_min_arr[start:end] = dec_geo_deg.min(axis=1)

        # frac_flux_south30: flux- and time-weighted fraction of the orbit's
        # observable light emitted while south of -30 deg declination, where major
        # northern surveys lose coverage. Distinguishes objects whose *bright*
        # apparitions fall in the deep south from those that only dip south while
        # faint near aphelion.
        south30 = (dec_geo_deg < -30.0)
        frac_flux_south30_arr[start:end] = (
            (south30 * w_ft).sum(axis=1) / w_ft_sum
        )

        # # vis_mag_timeavg: Time-weighted mean of apparent magnitude V over the orbit.
        # # As opposed to vis_orbit_flux_opp, directly averaging magnitudes prevents the 
        # # result from being overwhelmingly dominated by short-lived bright flashes 
        # # at close approaches. Time-averaging the magnitude acts as the geometric 
        # # mean of the flux, penalising objects that are very faint for the majority 
        # # of their orbits (e.g. highly eccentric NEOs).
        # mag_mean = (V_app * weights).sum(axis=1) / w_sum
        # vis_mag_timeavg_arr[start:end] = mag_mean

        # vis_orbit_flux_multi: same flux-averaging as vis_orbit_flux_opp, but averaged across
        # three Earth heliocentric-longitude positions per asteroid sample rather
        # than only at opposition. Earth at the asteroid's longitude (opposition),
        # +30 deg ahead, and +60 deg ahead.
        #
        # Physical motivation: real surveys rarely catch objects at exact opposition.
        # Sampling off-opposition geometries penalises NEOs much more than MBAs
        # because for low-r objects modest Earth offsets produce large changes in
        # geocentric distance and phase angle, while for distant objects the
        # Sun-Earth baseline is a small perturbation on the geometry.
        flux_geom_sum = flux_app  # offset = 0 (opposition); same as vis_orbit_flux_opp
        mag_geom_sum = V_app      # For the new geometric magnitude sum
        for offset_rad in EARTH_LON_OFFSETS_RAD[1:]:
            lambda_E = lambda_k + offset_rad
            dx_g = x_ecl - np.cos(lambda_E)
            dy_g = y_ecl - np.sin(lambda_E)
            # Earth z = 0, so dz_g = z_ecl (unchanged across Earth offsets)
            Delta_g = np.sqrt(dx_g * dx_g + dy_g * dy_g + z_ecl * z_ecl)
            Delta_g_safe = np.maximum(Delta_g, eps_val)
            cos_alpha_g = np.clip(
                (r_orb ** 2 + Delta_g ** 2 - 1.0) / (2.0 * r_safe * Delta_g_safe),
                -1.0, 1.0
            )
            alpha_g = np.arccos(cos_alpha_g)
            tan_half_g = np.maximum(np.tan(alpha_g / 2.0), 0.0)
            phi1_g = np.exp(-HG_A1 * np.power(tan_half_g, HG_B1))
            phi2_g = np.exp(-HG_A2 * np.power(tan_half_g, HG_B2))
            phi_blend_g = np.maximum(
                (1.0 - HG_G) * phi1_g + HG_G * phi2_g, PHI_FLOOR
            )
            V_g = (H_c
                   + 5.0 * np.log10(r_safe * Delta_g_safe)
                   - 2.5 * np.log10(phi_blend_g))
            flux_geom_sum = flux_geom_sum + np.power(10.0, -0.4 * V_g)
            mag_geom_sum = mag_geom_sum + V_g
            
        flux_geom_per_nu = flux_geom_sum / float(len(EARTH_LON_OFFSETS_RAD))
        flux_geom_mean = (flux_geom_per_nu * weights).sum(axis=1) / w_sum
        vis_orbit_flux_multi_arr[start:end] = (
            -2.5 * np.log10(np.maximum(flux_geom_mean, 1e-30))
        )
        
        mag_geom_per_nu = mag_geom_sum / float(len(EARTH_LON_OFFSETS_RAD))
        mag_geom_mean = (mag_geom_per_nu * weights).sum(axis=1) / w_sum
        vis_orbit_mag_multi_arr[start:end] = mag_geom_mean

    # orb["mean_opp_dec"] = mean_opp_dec_arr.astype(float)
    orb["spatial_discoverability_fraction"] = spatial_disc_arr.astype(float)
    orb["dec_flux_weighted"] = dec_flux_weighted_arr.astype(float)
    orb["dec_orbit_min"] = dec_orbit_min_arr.astype(float)
    orb["frac_flux_south30"] = frac_flux_south30_arr.astype(float)
    orb["vis_orbit_flux_opp"] = vis_orbit_flux_opp_arr.astype(float)
    # orb["vis_mag_timeavg"] = vis_mag_timeavg_arr.astype(float)
    orb["vis_orbit_flux_multi"] = vis_orbit_flux_multi_arr.astype(float)
    orb["vis_orbit_mag_multi"] = vis_orbit_mag_multi_arr.astype(float)

    # ============================================================================
    # ALIGNMENT AT LAST PERIHELION
    # ============================================================================
    # Reconstructs Sun-Earth-asteroid geometry at the most recent perihelion passage
    # prior to the catalog epoch and returns apparent V, geocentric distance, and
    # equatorial declination at that moment. Uses the catalog M and Epoch to
    # back-propagate to t_peri, an analytical Earth ephemeris (valid over the
    # decades-to-centuries lookbacks relevant here), and the IAU HG phase function.
    if "Epoch" in orb.columns and "M" in orb.columns:
        MU_SUN = 0.0002959122082855911  # AU^3 / day^2 (k^2)
        OBL_J2000 = np.radians(23.439291)
        sin_obl_p = np.sin(OBL_J2000)
        cos_obl_p = np.cos(OBL_J2000)

        M0 = np.radians(orb["M"].to_numpy(dtype=np.float64)) % (2.0 * np.pi)
        Epoch_jd = orb["Epoch"].to_numpy(dtype=np.float64)
        n_mm = np.sqrt(MU_SUN) / np.power(np.maximum(a_np, eps_val), 1.5)
        t_peri = Epoch_jd - M0 / np.maximum(n_mm, eps_val)

        # Asteroid heliocentric position at perihelion (nu = 0, r = a(1-e))
        r_peri = a_np * (1.0 - e_np)
        cos_u_p = np.cos(Peri_rad)
        sin_u_p = np.sin(Peri_rad)
        cos_i_p = np.cos(i_np)
        sin_i_p = np.sin(i_np)
        cos_Node_p = np.cos(Node_rad)
        sin_Node_p = np.sin(Node_rad)
        x_p = r_peri * (cos_Node_p * cos_u_p - sin_Node_p * sin_u_p * cos_i_p)
        y_p = r_peri * (sin_Node_p * cos_u_p + cos_Node_p * sin_u_p * cos_i_p)
        z_p = r_peri * sin_u_p * sin_i_p

        # Analytical Earth heliocentric position at t_peri (J2000 ecliptic frame).
        # The Meeus expressions below give the Sun's geocentric apparent longitude
        # Theta; Earth's heliocentric longitude is Theta + pi.
        T_jc = (t_peri - 2451545.0) / 36525.0
        M_earth = np.radians(357.52911 + 35999.05029 * T_jc)
        lambda_sun_geo = (np.radians(280.46646 + 36000.76983 * T_jc)
                          + 0.033416 * np.sin(M_earth)
                          + 0.000349 * np.sin(2.0 * M_earth))
        lambda_earth = lambda_sun_geo + np.pi
        r_earth = (1.00014061
                   - 0.01670861 * np.cos(M_earth)
                   - 0.00013957 * np.cos(2.0 * M_earth))
        x_e = r_earth * np.cos(lambda_earth)
        y_e = r_earth * np.sin(lambda_earth)
        # Earth z = 0 in the ecliptic frame

        # Geocentric vector and distance
        dx_p = x_p - x_e
        dy_p = y_p - y_e
        dz_p = z_p
        Delta_p = np.sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p)
        Delta_p_safe = np.maximum(Delta_p, eps_val)
        r_peri_safe = np.maximum(r_peri, eps_val)

        # Phase angle at the asteroid vertex: cos a = (r_ast . Delta) / (|r| |Delta|)
        dot_rd = x_p * dx_p + y_p * dy_p + z_p * dz_p
        cos_alpha_p = np.clip(dot_rd / (r_peri_safe * Delta_p_safe), -1.0, 1.0)
        alpha_p = np.arccos(cos_alpha_p)
        tan_half_p = np.maximum(np.tan(alpha_p / 2.0), 0.0)
        phi1_p = np.exp(-HG_A1 * np.power(tan_half_p, HG_B1))
        phi2_p = np.exp(-HG_A2 * np.power(tan_half_p, HG_B2))
        phi_blend_p = np.maximum((1.0 - HG_G) * phi1_p + HG_G * phi2_p, PHI_FLOOR)

        V_peri = (H_np
                  + 5.0 * np.log10(r_peri_safe * Delta_p_safe)
                  - 2.5 * np.log10(phi_blend_p))

        # Equatorial declination at last perihelion
        z_eq_p = dy_p * sin_obl_p + dz_p * cos_obl_p
        dec_peri_true = np.arcsin(np.clip(z_eq_p / Delta_p_safe, -1.0, 1.0))

        orb["vis_last_perihelion"] = V_peri.astype(float)
        orb["perihelion_delta_true"] = Delta_p.astype(float)
        orb["perihelion_dec_true"] = dec_peri_true.astype(float)

        # --- Second-to-last perihelion -----------------------------------------
        # One orbital period (P = 2*pi / n) earlier than t_peri. To first order the
        # asteroid returns to the SAME heliocentric position at perihelion each
        # revolution (precession over a single period is negligible here), so x_p,
        # y_p, z_p are reused; only Earth has moved. Re-evaluating the apparent V at
        # this earlier passage samples a different, independent Sun-Earth-asteroid
        # geometry -- useful because whether an object was favourably placed at its
        # most recent perihelion is partly luck of the Earth phasing.
        t_peri_2 = t_peri - 2.0 * np.pi / np.maximum(n_mm, eps_val)

        T_jc2 = (t_peri_2 - 2451545.0) / 36525.0
        M_earth2 = np.radians(357.52911 + 35999.05029 * T_jc2)
        lambda_sun_geo2 = (np.radians(280.46646 + 36000.76983 * T_jc2)
                           + 0.033416 * np.sin(M_earth2)
                           + 0.000349 * np.sin(2.0 * M_earth2))
        lambda_earth2 = lambda_sun_geo2 + np.pi
        r_earth2 = (1.00014061
                    - 0.01670861 * np.cos(M_earth2)
                    - 0.00013957 * np.cos(2.0 * M_earth2))
        x_e2 = r_earth2 * np.cos(lambda_earth2)
        y_e2 = r_earth2 * np.sin(lambda_earth2)
        # Earth z = 0 in the ecliptic frame

        dx_p2 = x_p - x_e2
        dy_p2 = y_p - y_e2
        dz_p2 = z_p
        Delta_p2 = np.sqrt(dx_p2 * dx_p2 + dy_p2 * dy_p2 + dz_p2 * dz_p2)
        Delta_p2_safe = np.maximum(Delta_p2, eps_val)

        dot_rd2 = x_p * dx_p2 + y_p * dy_p2 + z_p * dz_p2
        cos_alpha_p2 = np.clip(dot_rd2 / (r_peri_safe * Delta_p2_safe), -1.0, 1.0)
        alpha_p2 = np.arccos(cos_alpha_p2)
        tan_half_p2 = np.maximum(np.tan(alpha_p2 / 2.0), 0.0)
        phi1_p2 = np.exp(-HG_A1 * np.power(tan_half_p2, HG_B1))
        phi2_p2 = np.exp(-HG_A2 * np.power(tan_half_p2, HG_B2))
        phi_blend_p2 = np.maximum((1.0 - HG_G) * phi1_p2 + HG_G * phi2_p2, PHI_FLOOR)

        V_peri2 = (H_np
                   + 5.0 * np.log10(r_peri_safe * Delta_p2_safe)
                   - 2.5 * np.log10(phi_blend_p2))

        orb["vis_2nd_last_perihelion"] = V_peri2.astype(float)

        # --- Third-to-last perihelion ------------------------------------------
        # Same construction, three orbital periods (3 * 2*pi / n) before t_peri.
        # Reuses the asteroid perihelion position (x_p, y_p, z_p); only Earth's
        # ephemeris is re-evaluated. Over three revolutions orbital precession is
        # still small for the populations here, so the fixed-perihelion-position
        # approximation continues to hold.
        t_peri_3 = t_peri - 3.0 * (2.0 * np.pi / np.maximum(n_mm, eps_val))

        T_jc3 = (t_peri_3 - 2451545.0) / 36525.0
        M_earth3 = np.radians(357.52911 + 35999.05029 * T_jc3)
        lambda_sun_geo3 = (np.radians(280.46646 + 36000.76983 * T_jc3)
                           + 0.033416 * np.sin(M_earth3)
                           + 0.000349 * np.sin(2.0 * M_earth3))
        lambda_earth3 = lambda_sun_geo3 + np.pi
        r_earth3 = (1.00014061
                    - 0.01670861 * np.cos(M_earth3)
                    - 0.00013957 * np.cos(2.0 * M_earth3))
        x_e3 = r_earth3 * np.cos(lambda_earth3)
        y_e3 = r_earth3 * np.sin(lambda_earth3)
        # Earth z = 0 in the ecliptic frame

        dx_p3 = x_p - x_e3
        dy_p3 = y_p - y_e3
        dz_p3 = z_p
        Delta_p3 = np.sqrt(dx_p3 * dx_p3 + dy_p3 * dy_p3 + dz_p3 * dz_p3)
        Delta_p3_safe = np.maximum(Delta_p3, eps_val)

        dot_rd3 = x_p * dx_p3 + y_p * dy_p3 + z_p * dz_p3
        cos_alpha_p3 = np.clip(dot_rd3 / (r_peri_safe * Delta_p3_safe), -1.0, 1.0)
        alpha_p3 = np.arccos(cos_alpha_p3)
        tan_half_p3 = np.maximum(np.tan(alpha_p3 / 2.0), 0.0)
        phi1_p3 = np.exp(-HG_A1 * np.power(tan_half_p3, HG_B1))
        phi2_p3 = np.exp(-HG_A2 * np.power(tan_half_p3, HG_B2))
        phi_blend_p3 = np.maximum((1.0 - HG_G) * phi1_p3 + HG_G * phi2_p3, PHI_FLOOR)

        V_peri3 = (H_np
                   + 5.0 * np.log10(r_peri_safe * Delta_p3_safe)
                   - 2.5 * np.log10(phi_blend_p3))

        orb["vis_3rd_last_perihelion"] = V_peri3.astype(float)

        # ====================================================================
        # BRIGHTNESS AT THE LAST FIVE OPPOSITIONS
        # ====================================================================
        # Opposition is the instant the asteroid and Earth share the same
        # heliocentric ecliptic longitude (the asteroid sits opposite the Sun in
        # the sky): the geometry of each apparition's peak brightness -- minimum
        # solar phase angle and near-minimum geocentric distance. Unlike perihelion
        # it is NOT a fixed orbital position; it recurs once per synodic period at
        # times set by the Earth-asteroid longitude beat, so each opposition must be
        # solved for. We locate the five most recent oppositions before the catalog
        # epoch and evaluate the IAU (H, G) apparent V at each.
        #
        # Method:
        #   (1) A linear mean-longitude model gives each opposition time to within a
        #       fraction of a synodic period. The synodic angle is
        #           psi(t) = lambda_ast(t) - lambda_earth(t),
        #       and oppositions are psi == 0 (mod 2*pi). The asteroid mean longitude
        #       is varpi + M(t) with varpi = Node + Peri; Earth's is the Meeus mean.
        #   (2) A few Newton steps on the TRUE psi(t) -- lambda_ast from a Kepler
        #       solve (so eccentricity and inclination enter exactly) and
        #       lambda_earth from the analytic Meeus ephemeris -- refine each time.
        #       Steps are clamped to +/- half a synodic period so each estimate stays
        #       locked to its own opposition. Because V is at a local minimum at
        #       opposition, residual timing error contributes negligibly to V.
        #
        # All five share the same per-object orbital elements (no precession over the
        # few-year lookback). Interior objects (aphelion < 1 AU) never reach
        # opposition; for them the equal-longitude alignment is inferior conjunction
        # and the returned V is correspondingly faint -- the honest signal that such
        # objects lack good apparitions.
        TWO_PI = 2.0 * np.pi
        N_OPP = 5

        def _wrap_pi(ang):
            return (ang + np.pi) % TWO_PI - np.pi

        def _earth_lon(t):
            # Earth heliocentric ecliptic longitude (rad): Sun's geocentric
            # longitude + pi (Meeus low-precision series).
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
            # Solve M = E - e sin E (Newton). Seed handles moderate-to-high e.
            E = Marr + earr * np.sin(Marr) * (1.0 + earr * np.cos(Marr))
            for _ in range(12):
                E = E - (E - earr * np.sin(E) - Marr) / (1.0 - earr * np.cos(E))
            return E

        # Per-object columns broadcastable against the (N, 5) opposition times.
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

        # Synodic angular rate (rad/day); Earth's mean rate from the Meeus series.
        n_earth_rate = np.radians(36000.76983 / 36525.0)
        n_syn = n_mm - n_earth_rate
        n_syn_safe = np.where(np.abs(n_syn) < 1e-12, 1e-12, n_syn)
        P_syn = TWO_PI / np.abs(n_syn_safe)  # synodic period (days), > 0

        # (1) Linear initial guesses for the five most recent oppositions <= Epoch.
        varpi = Node_rad + Peri_rad
        psi_E = _wrap_pi((varpi + M0) - _earth_lon(Epoch_jd))
        t_near = Epoch_jd - psi_E / n_syn_safe
        t_last = np.where(t_near > Epoch_jd, t_near - P_syn, t_near)
        k_idx = np.arange(N_OPP)
        t_opp = t_last[:, None] - k_idx[None, :] * P_syn[:, None]  # (N, 5)
        t_guess = t_opp.copy()
        half_syn = 0.5 * P_syn[:, None]

        # (2) Newton refinement on the true synodic angle, clamped to each window.
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

        # Final apparent V at each refined opposition time.
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

        # Phase angle from the exact vectors (Sun-asteroid-Earth at the asteroid
        # vertex): cos alpha = (r_ast . (r_ast - r_earth)) / (|r_ast| |Delta|).
        dot_o = x_o * dx_o + y_o * dy_o + z_o * dz_o
        cos_alpha_o = np.clip(dot_o / (r_o_safe * Delta_o_safe), -1.0, 1.0)
        alpha_o = np.arccos(cos_alpha_o)
        tan_half_o = np.maximum(np.tan(alpha_o / 2.0), 0.0)
        phi1_o = np.exp(-HG_A1 * np.power(tan_half_o, HG_B1))
        phi2_o = np.exp(-HG_A2 * np.power(tan_half_o, HG_B2))
        phi_blend_o = np.maximum((1.0 - HG_G) * phi1_o + HG_G * phi2_o, PHI_FLOOR)

        V_opp = (H_col
                 + 5.0 * np.log10(r_o_safe * Delta_o_safe)
                 - 2.5 * np.log10(phi_blend_o))  # (N, 5), col 0 = most recent

        for j in range(N_OPP):
            orb[f"vis_opp_{j + 1}"] = V_opp[:, j].astype(float)

        # create vis_opp_1 through vis_opp_5 columns for the five most recent oppositions, with vis_opp_1
        orb["vis_opp_1"] = V_opp[:, 0].astype(float)  # most recent opposition  
        orb["vis_opp_2"] = V_opp[:, 1].astype(float)  # second most recent opposition
        orb["vis_opp_3"] = V_opp[:, 2].astype(float)  # third most recent opposition
        orb["vis_opp_4"] = V_opp[:, 3].astype(float)  # fourth most recent opposition
        orb["vis_opp_5"] = V_opp[:, 4].astype(float)  # fifth most recent

        # Summary statistics across the five oppositions. vis_opp_min is the
        # brightest (best) apparition; vis_opp_max the faintest; mean/median
        # describe the typical apparition brightness.
        orb["vis_opp_mean"] = V_opp.mean(axis=1).astype(float)
        orb["vis_opp_median"] = np.median(V_opp, axis=1).astype(float)
        orb["vis_opp_min"] = V_opp.min(axis=1).astype(float)
        orb["vis_opp_max"] = V_opp.max(axis=1).astype(float)
    else:
        orb["vis_last_perihelion"] = np.nan
        orb["perihelion_delta_true"] = np.nan
        orb["perihelion_dec_true"] = np.nan
        orb["vis_2nd_last_perihelion"] = np.nan
        orb["vis_3rd_last_perihelion"] = np.nan
        for j in range(5):
            orb[f"vis_opp_{j + 1}"] = np.nan
        orb["vis_opp_mean"] = np.nan
        orb["vis_opp_median"] = np.nan
        orb["vis_opp_min"] = np.nan
        orb["vis_opp_max"] = np.nan

    return orb


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_all_databases(apply_filters=True):
    """
    Load all orbit databases (MPC, AstDyS, JPL) and merge them.
    
    Parameters
    ----------
    apply_filters : bool, optional
        Whether to apply filter lists (default: True)
    
    Returns
    -------
    pd.DataFrame
        Combined orbit dataframe with all databases merged
    """
    print("Loading MPC orbits...")
    orb = load_mpc_orbits(apply_filters=apply_filters)
    
    print("Loading astrometry counts...")
    orb = load_astrometry_counts(orb)
    print("Applying nights overrides...")
    orb = apply_nights_overrides(orb)
    
    print("Loading AstDyS orbits...")
    astdys = load_astdys_orbits()
    
    print("Loading JPL orbits...")
    jpl = load_jpl_orbits()
    
    print("Comparing with AstDyS...")
    orb = compare_with_astdys(orb, astdys)
    
    print("Comparing with JPL...")
    orb = compare_with_jpl(orb, jpl)
    
    print("Computing database differences...")
    orb = compute_database_differences(orb)
    
    print("Applying number of oppositions overrides...")
    orb = apply_num_opps_overrides(orb)
    
    # Apply robust H: take the median when all three databases provide a value (to drop outliers),
    # else fall back to the dimmest (max) across available databases.
    # This must happen here, before feature_engineering is called, so that all
    # visibility features (vis_typ, vis_q, etc.) are computed using the combined H.
    # Using orb["H"] (post-corrections) rather than H_MPC so that known photometry
    # fixes are incorporated before combination.
    print("Applying robust H (median if 3 values, else max (dimmest) across MPC/AstDyS/JPL, after corrections)...")
    h_cols = orb[["H", "H_astdys", "H_jpl"]]
    has_3 = h_cols.notna().sum(axis=1) == 3
    orb["H"] = np.where(has_3, h_cols.median(axis=1), h_cols.max(axis=1))

    # Not sure which database is doing it, but drop any rows with H values == -9.99
    orb = orb[orb["H"] != -9.99]

    print("Applying magnitude corrections...")
    orb = apply_magnitude_corrections(orb)
    
    print("Adding training targets...")
    orb = add_training_targets(orb)
    
    return orb


# ==============================================================================
# EXTENSION DIFFICULTY CLASSIFIER
# ==============================================================================

def train_extension_difficulty_classifier(orb_pred, final, filter_csv="filter out unless updated.csv"):
    """
    Train extension difficulty classifier using iterative refinement.
    
    Extension difficulty quantifies objects that are challenging to extend due to:
    - Too uncertain to recover with ITF (Isolated Tracklet File) techniques
    - Too much of a "stretch" linkage that may represent a chimera orbit 
      (two different objects incorrectly linked together)
    
    The classifier uses the principle that highly-rated objects from prior ML 
    models that remain single-opposition are more representative of "difficult 
    to extend" objects. The difficulty is based primarily on astrometry metadata 
    (e.g., number of nights, arc length) rather than orbital elements.
    
    Parameters
    ----------
    orb_pred : pd.DataFrame
        Orbit predictions dataframe with prob, Num_opps, and astrometry metadata
    final : pd.DataFrame
        Final results dataframe to apply predictions to
    filter_csv : str, optional
        Path to filter CSV file (default: "filter out unless updated.csv")
    
    Returns
    -------
    tuple
        (orb_pred, final)
        - orb_pred: Updated orb_pred with extension_difficulty column
        - final: Updated final with extension_difficulty column
    """
    for df in (orb_pred, final):
        df["v_mag_gap"] = df["v_mag_max"] - df["v_mag_min"]
        df["second_minmax_gap"] = df["v_mag_second_max"] - df["v_mag_second_min"]
        df["v_mag_gap_1"] = df["v_mag_max"] - df["v_mag_avg"]
        df["v_mag_gap_2"] = df["v_mag_avg"] - df["v_mag_min"]

    # Load filter list, only keep the mislinkage comments
    filter_out_unless_updated = pd.read_csv(filter_csv)
    filter_out_unless_updated = filter_out_unless_updated[filter_out_unless_updated["reason"].str.contains("misl", case=False, na=False)]
    
    # Exclude recent observations and objects without astrometry metadata
    # Anything with E2026 may be too recent to have had the ITF community complete the extension if it is possible to extend.
    
    use_to_train_misl = orb_pred[
        (orb_pred["Ref"].str[0:5] != "E2026") & 
        (orb_pred["nights_total"].notna())
    ]
    
    # =========================================================================
    # POSITIVE CLASS: High extension difficulty objects
    # (too uncertain for ITF recovery or potential chimera orbits)
    # =========================================================================
    
    # Single opposition, high probability objects, we will include all with fewer than 4 nights and a sample of those with 4 or more nights
    poss_misl_or_unc_1opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.975) & 
        (use_to_train_misl["Num_opps"] == 1)
    ]
    poss_misl_or_unc_lt4nights = poss_misl_or_unc_1opp[
        poss_misl_or_unc_1opp["nights_total"] < 4
    ]
    poss_misl_or_unc_ge4nights = poss_misl_or_unc_1opp[
        poss_misl_or_unc_1opp["nights_total"] >= 4
    ]
    # but drop from this 4+ nights group any with a pre-existing extension_difficulty lower than 0.01 (likely okay)
    poss_misl_or_unc_ge4nights = poss_misl_or_unc_ge4nights[
        poss_misl_or_unc_ge4nights["extension_difficulty"] >= 0.01
    ]

    # high mag residuals likely
    poss_misl_or_unc_1opp_high_magresids = use_to_train_misl[
        (use_to_train_misl["v_mag_gap"] > 3)
        & (use_to_train_misl["v_mag_gap_1"] > 0.5)
        & (use_to_train_misl["v_mag_gap_2"] > 1.5)
        & (use_to_train_misl["second_minmax_gap"] > 1.5)
        & (use_to_train_misl["Arc_length"].between(7, 30))
        & (use_to_train_misl["nights_total"] <= 4)
        & (use_to_train_misl["Perihelion_dist"].between(1.6, 3.5))]
    
    # Objects in filter list and 2-3 opposition high-prob objects.
    # Reload from the cached raw MPC dump so Arc_length is the original (pre
    # apply_num_opps_overrides nulling), which is what the filter CSV matches.
    orb = load_mpc_orbits(apply_filters=False)
    orb = load_astrometry_counts(orb)
    orb = apply_nights_overrides(orb)
    poss_misl_or_unc_named = orb.merge(
        filter_out_unless_updated.rename(columns={"Object": "Principal_desig"})[["Principal_desig", "Arc_length"]],
        on=["Principal_desig", "Arc_length"],
        how="inner"
    )
    print(f"Positive examples from filter list: {len(poss_misl_or_unc_named)}")
    poss_misl_or_unc_23opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.99) & 
        (use_to_train_misl["Num_opps"].between(2, 3))
    ]
    
    # Combine positive examples
    poss_misl_or_unc = pd.concat([
        poss_misl_or_unc_23opp,
        poss_misl_or_unc_lt4nights,
        poss_misl_or_unc_ge4nights.sample(frac=0.2, random_state=42),
        poss_misl_or_unc_named,
        poss_misl_or_unc_1opp_high_magresids
    ])
    
    # Remove duplicates
    poss_misl_or_unc = poss_misl_or_unc[
        ~poss_misl_or_unc.index.duplicated(keep='first')
    ]
    
    # Filter out likely recoverable objects
    # 5+ nights over 12+ days in single opp is almost certainly not high difficulty
    poss_misl_or_unc = poss_misl_or_unc[~(
        (poss_misl_or_unc["Arc_length"] >= 12) & 
        (poss_misl_or_unc["nights_total"] >= 5) & 
        (poss_misl_or_unc["Num_opps"] == 1)
    )]
    
    # Objects with second opposition having 2+ nights have low extension difficulty
    poss_misl_or_unc = poss_misl_or_unc[
        ~(poss_misl_or_unc["opp_with_second_most_nights"] > 1)
    ]
    
    poss_misl_or_unc["label"] = 1
    print(f"Positive examples (high extension difficulty): {len(poss_misl_or_unc)}")
    
    # =========================================================================
    # NEGATIVE CLASS: Low extension difficulty (likely recoverable/reliable)
    # =========================================================================
    
    likely_okay = use_to_train_misl[
        (use_to_train_misl["prob"] < 0.7) & 
        (use_to_train_misl["Num_opps"] < 3)
    ]
    likely_okay_heavier = likely_okay[
        (likely_okay["nights_total"] >= 4) | 
        ((likely_okay["nights_total"] == 3) & (likely_okay["Arc_length"] < 22))
    ]
    likely_okay_gt3_opps = use_to_train_misl[use_to_train_misl["Num_opps"] >= 3]
    likely_okay_4night = likely_okay[likely_okay["nights_total"] == 4]
    likely_okay_gt5_nights_single_opp = likely_okay[
        (likely_okay["nights_total"] >= 5) & 
        (likely_okay["Num_opps"] == 1)
    ]
    likely_okay_gt4_opps_short_init_arc = use_to_train_misl[
        (use_to_train_misl["Num_opps"] >= 4) & 
        (use_to_train_misl["longest_opp_arc"] < 6)
    ]
    likely_okay_3opp = use_to_train_misl[
        (use_to_train_misl["prob"] < 0.97) & 
        (use_to_train_misl["Num_opps"] == 3)
    ]

    # low mag and astrometric residuals
    likely_okay_lowresids = likely_okay[
        (likely_okay["second_minmax_gap"] < 0.8)
      & (likely_okay["Arc_length"].between(11, 17))
      & (likely_okay["nights_total"].between(3,5))
      & (likely_okay["prob"]<0.4)
      & (likely_okay["rms"] < 0.09)]
    
    # Balanced sampling of negative examples
    likely_okay = pd.concat([
        likely_okay_gt5_nights_single_opp.sample(5000, random_state=91),
        likely_okay.sample(6000, random_state=42),
        likely_okay_heavier.sample(7000, random_state=11),
        likely_okay_gt3_opps.sample(2000, random_state=10),
        likely_okay_gt4_opps_short_init_arc.sample(500, random_state=9),
        likely_okay_4night.sample(4000, random_state=12),
        likely_okay_3opp.sample(500, random_state=8),
        likely_okay_lowresids
    ])

    # Remove duplicates
    likely_okay = likely_okay[
        ~likely_okay.index.duplicated(keep='first')
    ]

    likely_okay["label"] = 0
    print(f"Negative examples (low extension difficulty): {len(likely_okay)}")
    
    # =========================================================================
    # CLASSIFIER TRAINING WITH ITERATIVE REFINEMENT
    # =========================================================================
    
    misl_training = pd.concat([poss_misl_or_unc, likely_okay])

    # drop any rows where the same index appears in both positive and negative class (conflicting labels)
    misl_training = misl_training[~misl_training.index.duplicated(keep=False)]

    # Define feature columns
    misl_cols = [
        "U", "longest_opp_arc", "longest_gap_arc", "second_longest_gap_arc", 
        "shortest_gap_arc", "opposition_count", "opp_with_most_nights", 
        "opp_with_second_most_nights", "other_opps", "nights_total", 
        "Num_opps", "Num_obs", "Arc_length", #"v_mag_gap", "second_minmax_gap",
        
        "label"
    ]
    misl_cols_simple = ["U", "Num_opps", "Num_obs", "Arc_length"]
    
    # Initialize classifiers
    xgb_misl = XGBClassifier()
    xgb_misl_simple = XGBClassifier()
    
    def train_and_predict():
        """Helper function to train both classifiers and make predictions."""
        misl_training_mlcols = misl_training[misl_cols]
        
        # Train main classifier
        xgb_misl.fit(
            misl_training_mlcols.drop(columns=["label"]), 
            misl_training_mlcols["label"]
        )
        final["extension_difficulty"] = xgb_misl.predict_proba(
            final[misl_cols[:-1]].astype(float)
        )[:, 1]
        
        # Train simplified classifier (for missing astrometry data)
        xgb_misl_simple.fit(
            misl_training_mlcols[misl_cols_simple], 
            misl_training_mlcols["label"]
        )
        final["extension_difficulty_simple"] = xgb_misl_simple.predict_proba(
            final[misl_cols_simple].astype(float)
        )[:, 1]
        
        # Use simple classifier predictions where astrometry data is missing
        final.loc[final["nights_total"].isna(), "extension_difficulty"] = \
            final.loc[final["nights_total"].isna(), "extension_difficulty_simple"]
        final.drop(columns=["extension_difficulty_simple"], inplace=True)
    
    # Initial training
    train_and_predict()
    
    # Iterative refinement: Remove likely mislabeled examples
    misl_training["mr_temp"] = xgb_misl.predict_proba(
        misl_training[misl_cols[:-1]].astype(float)
    )[:, 1]
    
    # First refinement: Remove obvious mislabels
    misl_training = misl_training[~(
        (misl_training["label"] == 1) & (misl_training["mr_temp"] < 0.15)
    )]
    misl_training = misl_training[~(
        (misl_training["label"] == 0) & (misl_training["mr_temp"] > 0.88)
    )]
    
    # Retrain after first refinement
    train_and_predict()

    # Iterative refinement: Remove likely mislabeled examples
    misl_training["mr_temp"] = xgb_misl.predict_proba(
        misl_training[misl_cols[:-1]].astype(float)
    )[:, 1]
    
    # Second refinement: More aggressive filtering of positive class
    misl_training = misl_training[~(
        (misl_training["label"] == 1) & (misl_training["mr_temp"] < 0.04)
    )]
    print(f"Training set after refinement: {len(misl_training)}")
    
    # Final training
    train_and_predict()
    
    # Apply predictions to orb_pred
    orb_pred["extension_difficulty"] = xgb_misl.predict_proba(
        orb_pred[misl_cols[:-1]].astype(float)
    )[:, 1]
    
    return orb_pred, final
