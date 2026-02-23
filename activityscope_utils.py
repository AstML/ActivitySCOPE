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

import pandas as pd
import numpy as np
from sbpy.data import Names
import json
from sklearn.metrics import mean_poisson_deviance
from autogluon.core.metrics import make_scorer
from xgboost import XGBClassifier


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
    'GBM': {'objective': 'poisson'},
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
    orb = pd.read_json(
        "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz",
        compression='gzip'
    )
    
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
    
    # Set the overall H to be the dimmest of the three
    orb["H"] = orb[["H_MPC", "H_astdys", "H_jpl"]].max(axis=1)
    
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
    # Core visibility proxies - primary predictors of observational opportunities
    
    # Mean orbital distance proxy (semi-major axis weighted by eccentricity)
    r = orb['a'] * (1 + orb['e'] / 2)
    
    # Standard visibility: Apparent magnitude at mean distance
    # Formula: V = 5*log10(r*delta) + H, where delta approximates Earth distance
    orb['vis_0'] = (5 * np.log10(np.maximum(r, 1) * np.maximum(r - 1, 1)) + 
                    orb['H']).astype(float)
    
    # Perihelion visibility: Apparent magnitude at closest approach to Sun
    orb['visq_0'] = (5 * np.log10(np.maximum(orb["Perihelion_dist"], 1) * 
                                   np.maximum(orb["Perihelion_dist"] - 1, 1)) + 
                     orb['H']).astype(float)
    
    # Extended visibility variations at different Earth-object configurations
    e = np.clip(orb['e'], 0, 0.999)
    a = orb['a']
    H = orb['H']
    peri_rad = np.radians(orb['Peri'])
    
    # Aphelion visibility (faintest configuration)
    r_Q = a * (1 + e)
    delta_Q = np.maximum(r_Q - 1.6, 0.001)
    orb['vis_Q_0.6'] = (5 * np.log10(np.maximum(r_Q, 0.001) * delta_Q) + 
                        H).astype(float)
    
    # Semi-latus rectum visibility (at 90° true anomaly)
    r_p = a * (1 - e**2)
    delta_p = np.maximum(r_p - 1.6, 0.001)
    orb['vis_p_0.6'] = (5 * np.log10(np.maximum(r_p, 0.001) * delta_p) + 
                        H).astype(float)
    
    # Flux-weighted mean visibility (time-averaged photon collection)
    r_flux_0 = a * np.power(1 - e**2, 0.25)
    delta_flux_0 = np.maximum(r_flux_0 - 1.0, 0.001)
    orb['vis_flux_0.0'] = (5 * np.log10(np.maximum(r_flux_0, 0.001) * 
                                        delta_flux_0) + H).astype(float)
    
    r_flux_3 = a * np.power(1 - e**2, 0.25)
    delta_flux_3 = np.maximum(r_flux_3 - 1.3, 0.001)
    orb['vis_flux_0.3'] = (5 * np.log10(np.maximum(r_flux_3, 0.001) * 
                                        delta_flux_3) + H).astype(float)
    
    # Ecliptic node visibility (brightest crossing of ecliptic plane)
    p_semi = a * (1 - e**2)
    r_node1 = p_semi / (1 + e * np.cos(peri_rad))
    r_node2 = p_semi / (1 - e * np.cos(peri_rad))
    r_node = np.minimum(r_node1, r_node2)
    delta_node = np.maximum(r_node - 1.0, 0.001)
    orb['vis_node_0.0'] = (5 * np.log10(np.maximum(r_node, 0.001) * 
                                        delta_node) + H).astype(float)
    
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
    orb['dec_perihelion'] = np.degrees(np.arcsin(sin_dec))
    
    # Galactic plane alignment
    # Angle between orbital plane and galactic plane (affects stellar 
    # background density)
    n_gal = np.array([-0.8676, 0.0104, 0.4971])  # Galactic pole in ecliptic coords
    n_ast = [
        np.sin(np.radians(orb["i"])) * np.sin(np.radians(orb["Node"])),
        -np.sin(np.radians(orb["i"])) * np.cos(np.radians(orb["Node"])),
        np.cos(np.radians(orb["i"]))
    ]
    orb["gal_dist"] = np.degrees(np.arccos(
        n_gal[0]*n_ast[0] + n_gal[1]*n_ast[1] + n_gal[2]*n_ast[2]
    ))
    
    # Combined angular elements (exploratory feature)
    orb["node_plus_peri"] = (orb["Node"] + orb["Peri"]) % 360
    
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
    # Load filter list
    filter_out_unless_updated = pd.read_csv(filter_csv)
    
    # Exclude recent observations and objects without astrometry metadata
    use_to_train_misl = orb_pred[
        (orb_pred["Ref"].str[0:5] != "E2026") & 
        (orb_pred["nights_total"].notna())
    ]
    
    # =========================================================================
    # POSITIVE CLASS: High extension difficulty objects
    # (uncertain for ITF recovery or potential chimera orbits)
    # =========================================================================
    
    # Single opposition, high probability objects
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
    
    # Objects in filter list and 2-3 opposition high-prob objects
    poss_misl_or_unc_named = use_to_train_misl[
        use_to_train_misl["Principal_desig"].isin(filter_out_unless_updated["Object"])
    ]
    poss_misl_or_unc_23opp = use_to_train_misl[
        (use_to_train_misl["prob"] > 0.99) & 
        (use_to_train_misl["Num_opps"].between(2, 3))
    ]
    
    # Combine positive examples
    poss_misl_or_unc = pd.concat([
        poss_misl_or_unc_23opp,
        poss_misl_or_unc_lt4nights,
        poss_misl_or_unc_ge4nights.sample(frac=0.15, random_state=42),
        poss_misl_or_unc_named
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
        (use_to_train_misl["prob"] < 0.85) & 
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
    
    # Balanced sampling of negative examples
    likely_okay = pd.concat([
        likely_okay_gt5_nights_single_opp.sample(5000, random_state=91),
        likely_okay.sample(5000, random_state=42),
        likely_okay_heavier.sample(5000, random_state=11),
        likely_okay_gt3_opps.sample(2000, random_state=10),
        likely_okay_gt4_opps_short_init_arc.sample(500, random_state=9),
        likely_okay_4night.sample(4000, random_state=12),
        likely_okay_3opp.sample(500, random_state=8)
    ])
    likely_okay["label"] = 0
    print(f"Negative examples (low extension difficulty): {len(likely_okay)}")
    
    # =========================================================================
    # CLASSIFIER TRAINING WITH ITERATIVE REFINEMENT
    # =========================================================================
    
    misl_training = pd.concat([poss_misl_or_unc, likely_okay])
    
    # Define feature columns
    misl_cols = [
        "U", "longest_opp_arc", "longest_gap_arc", "second_longest_gap_arc", 
        "shortest_gap_arc", "opposition_count", "opp_with_most_nights", 
        "opp_with_second_most_nights", "other_opps", "nights_total", 
        "Num_opps", "Num_obs", "Arc_length", "label"
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
    
    # Second refinement: More aggressive filtering of positive class
    misl_training = misl_training[~(
        (misl_training["label"] == 1) & (misl_training["mr_temp"] < 0.07)
    )]
    print(f"Training set after refinement: {len(misl_training)}")
    
    # Final training
    train_and_predict()
    
    # Apply predictions to orb_pred
    orb_pred["extension_difficulty"] = xgb_misl.predict_proba(
        orb_pred[misl_cols[:-1]].astype(float)
    )[:, 1]
    
    return orb_pred, final
