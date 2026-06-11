"""tune_vis_offsets.py

Stochastic search for the optimal d in each `vis_*` family. Two formula
templates are in play:

  (1) Original families (4):

         vis = 5 log10( r_X * max(r_X - d, eps) ) + H_V               (eq. A)

      across four radius families:

         typical    r_X = a (1 + e/2)
         perihelion r_X = a (1 - e)       = q
         aphelion   r_X = a (1 + e)       = Q
         flux       r_X = a (1 - e^2)^(1/4)

  (2) Super families (4): physically-motivated alternatives that don't fit
      eq. A. Each has a single tunable d and a clear interpretation:

         super1 (vis_timeavg):  H + 5 log10(r * max(r-d, eps))
                           with r = a(1 + e^2/2). Time-averaged true
                           heliocentric distance, using the standard
                           tunable geocentric distance offset d.

         super4 (vis_S4):  H + 5 log10(r * max(r-1, eps)) + d * alpha_max
                           with r = a(1 + e^2/2) and
                           alpha_max = arcsin(min(1/r, 1)) the maximum
                           geometric phase angle (Sun-Object-Earth angle
                           when Earth sits at the tangent of its orbit
                           from the object; law of sines). Linear phase
                           darkening: d (mag/rad) is the effective phase
                           coefficient, folding the H,G phase slope and
                           the typical-alpha/alpha_max ratio into one
                           number. For r >> 1, alpha_max ~ 1/r, so this
                           asymptotes to a d/r magnitude penalty
                           (recovering S2's empirical behavior with a
                           defensible origin); for r <= 1 it saturates
                           at d*pi/2 instead of diverging.

         super5 (vis_S5):  H + 5 log10(r * max(Delta_quad - d, eps))
                           with r = a(1 + e^2/2) and
                           Delta_quad = sqrt(max(r^2 - 1, eps)) the
                           geocentric distance at quadrature (Earth-
                           Object line perpendicular to Sun-Earth line).
                           Uses quadrature rather than exact opposition
                           as the reference observing geometry; d (AU)
                           shifts the effective Delta from quadrature
                           toward opposition (d ~ 1 nearly recovers the
                           opposition formula for r >> 1). Because
                           Delta_quad grows faster than r-1 as r -> 1+,
                           this naturally penalizes small-r objects
                           without bolting on a separate phase term.

Search state: an ordered list of (family, d) entries with length between
MIN_SLOTS = 4 and MAX_SLOTS = 7. Duplicates are allowed (e.g. three vis_flux
entries with different d's) and any family may be absent. Each entry
contributes one column to the feature matrix.

Starting point: a random state with length drawn uniformly from [4, 7], each
slot assigned a random family (4 r_X choices plus 3 super families) and a
random d in [-0.7, 1.0]. Re-sampled each run; pass --search-seed to reproduce
a specific starting state.

Each iteration picks one of four move types:

    mutate (60% probability)
        Pick k ∈ {1, 2, 3} uniformly, then mutate k randomly chosen slots
        by sampling a fresh d for each from a gaussian centered at the
        slot's current d with std=1, truncated to [-0.7, 1.0]. Family is
        preserved.

    add (10% probability, suppressed if already at MAX_SLOTS)
        Append a new (family, d) slot with both drawn uniformly.

    remove (10% probability, suppressed if already at MIN_SLOTS)
        Drop one randomly chosen slot.

    swap (20% probability)
        Pick one slot and replace it with a fresh (family', d') where
        family' is drawn uniformly from the families other than the
        current one. Atomic family change without needing the search to
        survive a rejected remove+add pair.

The candidate state is scored via k-fold cross-validation (default 5 folds)
on a base XGBRegressor with objective='count:poisson' against
Num_opps_minus_one, averaging RMSE, Poisson deviance, and R^2 across folds.
The move is accepted if at least 2 of those three averaged metrics improve
on the current best, else rejected. Boundaries (size 4 or 7) reweight the
available moves rather than re-rolling, so the effective per-iteration mix
stays sensible at the edges.

Writes nothing to disk. Prints baseline, every iteration, periodic best
summary, and final state.

Run:
    python tune_vis_offsets.py [--iters N] [--seed S] [--subsample K] [--n-folds F]
"""

import argparse
import json
import random
import signal
import time

import numpy as np
import pandas as pd
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
# Family definitions
# ---------------------------------------------------------------------------

FAMILIES = {
    # Original families: vis = 5 log10(r_X * max(r_X - d, eps)) + H,
    # where d plays the role of a (geocentric) distance offset.
    "typical":    lambda a, e: a * (1.0 + e / 2.0),
    "perihelion": lambda a, e: a * (1.0 - e),
    "aphelion":   lambda a, e: a * (1.0 + e),
    "flux":       lambda a, e: a * np.power(np.clip(1.0 - e ** 2, 0.0, None), 0.25),

    # Super families: physically motivated alternatives that don't fit the
    # 5 log10(r * (r-d)) template. Dispatched in visibility() by name; the
    # lambda below is documentation only (the "characteristic r" the formula
    # uses, with None where there isn't a single one).
    #
    # super1: time-averaged r = a(1 + e^2/2), exact geocentric Δ = r - d
    #         at opposition, with tunable distance offset d (in AU).
    "super1":     lambda a, e: a * (1.0 + e ** 2 / 2.0),
    # super4: time-averaged r = a(1 + e^2/2), opposition Δ = r - 1, plus
    #         linear phase darkening d * α_max where
    #         α_max = arcsin(min(1/r, 1)) is the max geometric phase angle.
    #         d in mag/rad. Asymptotes to ~d/r for r >> 1 (the regime where
    #         S2's d/r penalty was working), but with a defensible origin
    #         and no divergence as r → 1.
    "super4":     lambda a, e: a * (1.0 + e ** 2 / 2.0),
    # super5: time-averaged r = a(1 + e^2/2), but geocentric distance taken
    #         at quadrature: Δ_quad = √(max(r²-1, ε)). d (AU) shifts the
    #         effective Δ from quadrature toward opposition (d ≈ 1 nearly
    #         recovers the opposition formula for r >> 1).
    "super5":     lambda a, e: a * (1.0 + e ** 2 / 2.0),
}

# Human-facing display name per family (used in progress output)
DISPLAY_NAME = {
    "typical":    "vis_typ",
    "perihelion": "vis_q",
    "aphelion":   "vis_Q",
    "flux":       "vis_flux",
    "super1":     "vis_timeavg",
    "super4":     "vis_S4",
    "super5":     "vis_S5",
}

# Non-vis features kept verbatim from the notebook's mlcols
NON_VIS_COLS = [
    'H', 'Node', 'a', 'i',
       'Perihelion_direction_x_e', 'Perihelion_direction_y_e',
       'vis_orbit_mag_multi', 'dec_flux_weighted', 'vis_opp_mean', 'e',
]

EPS = 1e-3
D_LOW, D_HIGH = -0.05, 1.05

# Search-space bounds and per-iteration move-type weights.
# Weights are over (mutate, add, remove, swap); at the size bounds the
# unavailable move (add at MAX_SLOTS, remove at MIN_SLOTS) is dropped and
# the remaining weights normalize naturally. `swap` and `mutate` are always
# available since they don't change the slot count.
MIN_SLOTS = 4
MAX_SLOTS = 7
MOVE_WEIGHTS = (0.60, 0.10, 0.10, 0.20)


# ---------------------------------------------------------------------------
# Feature construction and evaluation
# ---------------------------------------------------------------------------

def visibility(a, e, H, family, d):
    """Compute the visibility-proxy column for the given family and d.

    Standard families ("typical", "perihelion", "aphelion", "flux") all share
    the form  5 log10(r_X * max(r_X - d, eps)) + H with r_X set per family.

    Super families use distinct physically-motivated formulas:

      super1 (vis_timeavg): time-averaged true distance with standard offset
          r = a(1 + e^2/2)              # true time-averaged heliocentric
          Δ = max(r - d, eps)           # tunable distance offset
          vis = H + 5 log10(r * Δ)

      super4 (vis_S4): linear phase darkening at the geometric max α
          r = a(1 + e^2/2)
          Δ = max(r - 1, eps)
          α_max = arcsin(min(1/r, 1))            # max S-O-E angle, in rad
          vis = H + 5 log10(r * Δ) + d * α_max   # d in mag/rad
          For r >> 1: α_max ≈ 1/r, so vis ≈ H + 5 log10(r*(r-1)) + d/r,
          recovering S2's empirical behavior with a Bowell-H,G origin.

      super5 (vis_S5): quadrature-geometry geocentric distance
          r = a(1 + e^2/2)
          Δ_quad = sqrt(max(r^2 - 1, eps))       # geocentric Δ at quadrature
          vis = H + 5 log10(r * max(Δ_quad - d, eps))
          d (AU) pulls Δ from quadrature toward opposition; d ≈ 1 nearly
          recovers the standard opposition formula for r >> 1.
    """
    if family in {"typical", "perihelion", "aphelion", "flux"}:
        r = FAMILIES[family](a, e)
        r_safe = np.maximum(r, EPS)
        delta = np.maximum(r - d, EPS)
        return 5.0 * np.log10(r_safe * delta) + H
    if family == "super1":
        r = a * (1.0 + e ** 2 / 2.0)
        r_safe = np.maximum(r, EPS)
        delta = np.maximum(r - d, EPS)
        return 5.0 * np.log10(r_safe * delta) + H
    if family == "super4":
        r = a * (1.0 + e ** 2 / 2.0)
        r_safe = np.maximum(r, EPS)
        delta = np.maximum(r - 1.0, EPS)
        alpha_max = np.arcsin(np.minimum(1.0 / r_safe, 1.0))
        return 5.0 * np.log10(r_safe * delta) + H + d * alpha_max
    if family == "super5":
        r = a * (1.0 + e ** 2 / 2.0)
        r_safe = np.maximum(r, EPS)
        delta_quad = np.sqrt(np.maximum(np.abs(r ** 2 - 1.0), EPS))
        delta_eff = np.maximum(delta_quad - d, EPS)
        return 5.0 * np.log10(r_safe * delta_eff) + H
    raise ValueError(f"Unknown family: {family}")


def build_feature_matrix(orb, state):
    """Return X with NON_VIS_COLS plus one column per (family, d) entry in
    `state`, named slot_0, slot_1, ..., slot_{n-1}."""
    X = orb[NON_VIS_COLS].copy()
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0, 0.999)
    H = orb["H"].to_numpy(dtype=np.float64)
    for i, (fam, d) in enumerate(state):
        X[f"slot_{i}"] = visibility(a, e, H, fam, d).astype(np.float32)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Search helpers (state initialization and move generation)
# ---------------------------------------------------------------------------

def random_state(rng, families, min_slots=MIN_SLOTS, max_slots=MAX_SLOTS):
    """Sample an initial state: length ∈ [min_slots, max_slots], each entry
    a random (family, d) with d ∈ [D_LOW, D_HIGH]."""
    n = rng.randint(min_slots, max_slots)
    return [(rng.choice(families), rng.uniform(D_LOW, D_HIGH)) for _ in range(n)]


def pick_move(rng, n_slots, weights=MOVE_WEIGHTS,
              min_slots=MIN_SLOTS, max_slots=MAX_SLOTS):
    """Return one of 'mutate' / 'add' / 'remove' / 'swap'. At size bounds
    the unavailable size-changing move is dropped and the remaining weights
    normalize. `swap` (change one slot's family + d) and `mutate` are
    always available."""
    options = ["mutate"]
    w = [weights[0]]
    if n_slots < max_slots:
        options.append("add")
        w.append(weights[1])
    if n_slots > min_slots:
        options.append("remove")
        w.append(weights[2])
    options.append("swap")
    w.append(weights[3])
    return rng.choices(options, weights=w, k=1)[0]


def apply_move(rng, state, move, families, mutate_std=1.0):
    """Return (trial_state, op_label, change_strs).
    op_label is 'k=1' / 'k=2' / 'k=3' / 'add ' / 'rm  ' / 'swap'."""
    if move == "mutate":
        # k must not exceed the number of slots; tiny states (e.g. when the
        # user sets --max-slots 2) need k ∈ {1, 2} or {1}.
        max_k = min(3, len(state))
        k = rng.randint(1, max_k)
        slot_indices = rng.sample(range(len(state)), k)
        trial = list(state)
        change_strs = []
        for i in slot_indices:
            old_fam, old_d = trial[i]
            new_d = rng.gauss(old_d, mutate_std)
            while new_d < D_LOW or new_d > D_HIGH:
                new_d = rng.gauss(old_d, mutate_std)
            trial[i] = (old_fam, new_d)
            change_strs.append(
                f"{DISPLAY_NAME[old_fam]}:{old_d:+.3f}→{new_d:+.3f}"
            )
        return trial, f"k={k} ", change_strs
    if move == "add":
        fam = rng.choice(families)
        d = rng.uniform(D_LOW, D_HIGH)
        trial = list(state) + [(fam, d)]
        return trial, "add ", [
            f"+{DISPLAY_NAME[fam]}@{d:+.3f} ({len(state)}→{len(trial)})"
        ]
    if move == "remove":
        i = rng.randrange(len(state))
        fam, d = state[i]
        trial = list(state[:i]) + list(state[i+1:])
        return trial, "rm  ", [
            f"-{DISPLAY_NAME[fam]}@{d:+.3f} ({len(state)}→{len(trial)})"
        ]
    if move == "swap":
        # Change one slot to a different family with a fresh d. Same count.
        i = rng.randrange(len(state))
        old_fam, old_d = state[i]
        other_families = [f for f in families if f != old_fam]
        new_fam = rng.choice(other_families)
        new_d = rng.uniform(D_LOW, D_HIGH)
        trial = list(state)
        trial[i] = (new_fam, new_d)
        return trial, "swap", [
            f"{DISPLAY_NAME[old_fam]}@{old_d:+.3f}→{DISPLAY_NAME[new_fam]}@{new_d:+.3f}"
        ]
    raise ValueError(f"Unknown move: {move}")


def state_str(state):
    """One-line compact representation of a state for periodic summaries."""
    return ", ".join(f"{DISPLAY_NAME[fam]}@{d:+.3f}" for fam, d in state)


def evaluate_cv(X, y, folds, seed):
    """Train base Poisson XGBRegressor across `folds` and return the
    fold-averaged (rmse, poisson_dev, r2). `folds` is a list of
    (train_idx, test_idx) tuples produced by KFold.split()."""
    rmses, pdevs, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        model = XGBRegressor(
            objective="count:poisson",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_pred = np.maximum(y_pred, EPS)  # Poisson deviance needs strictly > 0
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        pdevs.append(mean_poisson_deviance(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(rmses)), float(np.mean(pdevs)), float(np.mean(r2s))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(subsample=None, seed=0):
    """Load orb, feature-engineer, then apply the same training filter the
    notebook uses (cells 1, 3, 5, and 7 of ActivitySCOPE_simplified_demo.ipynb)."""
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases()
    # filter out objects marked in either of the two named filter csvs
    orb = orb[~orb["filtered_out"].astype(bool)]
    orb = utils.feature_engineering(orb)

    # Cell 5: merge cached extension_difficulty scores so the training filter
    # below can use them just like the notebook does.
    print("Merging cached extension_difficulty.csv...")
    extension_difficulty = pd.read_csv("extension_difficulty.csv")
    orb = orb.merge(extension_difficulty, on="Principal_desig", how="left")

    # Cell 1: load active-object and dual-designation lists for the filter below.
    with open("known_active_objects.json", "r") as f:
        known_active = json.load(f)
    with open("dual_designation_list.json", "r") as f:
        dual_designation_list = json.load(f)

    print(len(orb))  # mirrors the notebook's `print(len(orb))` in cell 6

    # Cell 7: training-data filter (identical to the notebook).
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
    print(len(orb))  # mirrors the notebook's `print(len(orb_decent_orbit))` in cell 7

    if subsample is not None and subsample < len(orb):
        orb = orb.sample(n=subsample, random_state=seed)
        print(f"Subsampled to {len(orb)} rows for faster iteration.")

    return orb.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=1000,
                        help="Total stochastic iterations across all restart "
                             "segments (default: 1000).")
    parser.add_argument("--reseed-after", type=int, default=75,
                        help="Re-seed (random restart) after this many "
                             "consecutive rejected iterations within a segment "
                             "(default: 75). Set to a very large number to "
                             "disable restarts.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for the train/test split, "
                             "subsample, and XGB fit (default: 0). Keep this "
                             "fixed across runs so the metrics are comparable.")
    parser.add_argument("--search-seed", type=int, default=None,
                        help="Seed for the stochastic search (family choice and "
                             "d sampling). Default: int(time.time()) at startup, "
                             "so every invocation explores differently. Pass an "
                             "integer to reproduce a specific run.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration (default: use all).")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds used to score each candidate "
                             "(default: 5, minimum: 2).")
    parser.add_argument("--min-slots", type=int, default=MIN_SLOTS,
                        help=f"Minimum number of vis-family features in the "
                             f"state (default: {MIN_SLOTS}).")
    parser.add_argument("--max-slots", type=int, default=MAX_SLOTS,
                        help=f"Maximum number of vis-family features in the "
                             f"state (default: {MAX_SLOTS}).")
    parser.add_argument("--summary-every", type=int, default=25,
                        help="Print a current-best summary every N iters (default: 25).")
    args = parser.parse_args()

    if args.search_seed is None:
        args.search_seed = int(time.time())
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2.")
    if args.min_slots < 1:
        parser.error("--min-slots must be at least 1.")
    if args.max_slots < args.min_slots:
        parser.error("--max-slots must be >= --min-slots.")
    print(f"Deterministic seed (CV folds / XGB):  {args.seed}")
    print(f"Search seed (family + d sampling):    {args.search_seed}  "
          f"(pass --search-seed {args.search_seed} to reproduce this run)")
    print(f"CV: {args.n_folds}-fold cross-validation")
    print(f"Slot count range: [{args.min_slots}, {args.max_slots}]")
    rng = random.Random(args.search_seed)
    np.random.seed(args.seed)

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb["Num_opps_minus_one"].astype(np.float32).to_numpy()

    families = list(FAMILIES.keys())
    best_state = random_state(rng, families,
                              min_slots=args.min_slots, max_slots=args.max_slots)
    print()
    print("=" * 78)
    print(f"Starting random state ({len(best_state)} features):")
    for i, (fam, d) in enumerate(best_state):
        print(f"   slot_{i}  {DISPLAY_NAME[fam]:9s}  d = {d:+.4f}")
    print("=" * 78)

    print("\nBuilding initial feature matrix and preparing CV folds...")
    X = build_feature_matrix(orb, best_state)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    fold_sizes = [len(test) for _, test in folds]
    print(f"Rows: {len(X)}   Features: {X.shape[1]}   "
          f"{args.n_folds}-fold sizes: {fold_sizes}")
    print(f"Columns: {list(X.columns)}")

    # Diagnostic: RMSE on the 16 base columns from tune_new_features (the
    # 12 NON_VIS_COLS plus the 4 precomputed vis_* columns from
    # feature_engineering). Lets us cross-check that two simultaneously
    # running tuners see the same underlying data.
    diag_cols = NON_VIS_COLS + ["vis_typ", "vis_q"]
    print(f"\n[diagnostic] RMSE on the {len(diag_cols)} base columns shared with "
          f"tune_new_features: {diag_cols}")
    X_diag = orb[diag_cols].astype(np.float32)
    t0 = time.time()
    diag_rmse, diag_pdev, diag_r2 = evaluate_cv(X_diag, y, folds, args.seed)
    dt = time.time() - t0
    print(f"[diagnostic] RMSE={diag_rmse:.5f}  Poisson={diag_pdev:.5f}  "
          f"R2={diag_r2:.5f}  ({dt:.1f}s for {args.n_folds} fits)")

    print(f"\nTraining baseline ({args.n_folds} fits)...")
    t0 = time.time()
    best_rmse, best_pdev, best_r2 = evaluate_cv(X, y, folds, args.seed)
    dt = time.time() - t0
    print(f"Baseline:  RMSE={best_rmse:.5f}   Poisson={best_pdev:.5f}   "
          f"R2={best_r2:.5f}   ({dt:.1f}s for {args.n_folds} fits)")
    print()

    # Global best across all restart segments. Initialized to the first
    # segment's baseline; updated whenever a segment-local accept (or a
    # fresh segment baseline) beats it on ≥ 2 of the 3 metrics.
    global_best_state = list(best_state)
    global_best_rmse = best_rmse
    global_best_pdev = best_pdev
    global_best_r2 = best_r2

    # Per-segment history (for the final report). Each entry holds the
    # segment's final state and metrics.
    segment_history = []
    segment_index = 0
    segment_start_iter = 1

    consecutive_rejects = 0

    n_accept = 0
    n_reject = 0
    op_accepts = {"k=1 ": 0, "k=2 ": 0, "k=3 ": 0, "add ": 0, "rm  ": 0, "swap": 0}
    op_rejects = {"k=1 ": 0, "k=2 ": 0, "k=3 ": 0, "add ": 0, "rm  ": 0, "swap": 0}
    last_it = 0
    interrupted = False

    # Install a SIGINT handler that flips a flag instead of raising
    # KeyboardInterrupt. XGBoost's multithreaded fit() can swallow Python
    # signals, so relying on KeyboardInterrupt propagation is unreliable.
    # We instead check the flag at the top of each iteration; the in-flight
    # iteration completes (at most a few seconds) and then we break out
    # cleanly so the final summary still prints.
    interrupted_flag = [False]

    def _sigint_handler(signum, frame):
        if interrupted_flag[0]:
            # Second Ctrl-C — restore default handler and re-raise so the
            # user can force-quit if they really mean it.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[second Ctrl-C — exiting immediately]")
            raise KeyboardInterrupt
        interrupted_flag[0] = True
        print("\n[Ctrl-C received — will exit after current iteration "
              "completes; press again to force-quit]")

    signal.signal(signal.SIGINT, _sigint_handler)

    def maybe_update_global(state, rmse, pdev, r2):
        """Return True if this candidate beats the running global best on
        ≥ 2 of the 3 metrics; updates the global tracking variables."""
        nonlocal global_best_state, global_best_rmse, global_best_pdev, global_best_r2
        g_imp = (int(rmse < global_best_rmse)
                 + int(pdev < global_best_pdev)
                 + int(r2 > global_best_r2))
        if g_imp >= 2:
            global_best_state = list(state)
            global_best_rmse = rmse
            global_best_pdev = pdev
            global_best_r2 = r2
            return True
        return False

    try:
      for it in range(1, args.iters + 1):
        if interrupted_flag[0]:
            interrupted = True
            break
        last_it = it

        # Simulated annealing cooling schedule (1.0 -> 0.3 over first 200 segment iters)
        iters_in_segment = it - segment_start_iter
        mutate_std = max(0.3, 1.0 - (iters_in_segment / 200.0) * 0.7)

        # Decide on a move type respecting --min-slots / --max-slots bounds.
        move = pick_move(rng, len(best_state),
                         min_slots=args.min_slots, max_slots=args.max_slots)
        trial_state, op_label, change_strs = apply_move(
            rng, best_state, move, families, mutate_std=mutate_std
        )

        # Rebuild the entire feature matrix. The vis-column block is small
        # (4-7 columns out of ~17) and add/remove invalidates slot indexing
        # anyway, so a full rebuild keeps the code simple.
        X_trial = build_feature_matrix(orb, trial_state)

        t0 = time.time()
        rmse, pdev, r2 = evaluate_cv(X_trial, y, folds, args.seed)
        dt = time.time() - t0

        # Improvement count (lower is better for RMSE and Poisson; higher for R2)
        improvements = int(rmse < best_rmse) + int(pdev < best_pdev) + int(r2 > best_r2)
        accept = improvements >= 2

        changes_str = " ".join(change_strs).ljust(70)

        global_marker = ""
        if accept:
            n_accept += 1
            op_accepts[op_label] += 1
            meaningful_improvement = (best_rmse - rmse >= 0.00001) or \
                                     (best_pdev - pdev >= 0.00001) or \
                                     (r2 - best_r2 >= 0.00001)
            best_state = trial_state
            best_rmse, best_pdev, best_r2 = rmse, pdev, r2
            X = X_trial  # carry forward
            if meaningful_improvement:
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
            if maybe_update_global(best_state, best_rmse, best_pdev, best_r2):
                global_marker = "  [NEW GLOBAL BEST]"
        else:
            n_reject += 1
            op_rejects[op_label] += 1
            consecutive_rejects += 1

        print(
            f"[{it:4d}] {'ACCEPT' if accept else 'reject'}  {op_label}  {changes_str}  "
            f"RMSE={rmse:.5f} ({rmse - best_rmse:+.5f})  "
            f"Poisson={pdev:.5f} ({pdev - best_pdev:+.5f})  "
            f"R2={r2:.5f} ({r2 - best_r2:+.5f})  "
            f"imp={improvements}/3  N={len(trial_state)}  "
            f"rej_streak={consecutive_rejects}  "
            f"std={mutate_std:.3f}  ({dt:.1f}s)"
            f"{global_marker}"
        )

        if args.summary_every and it % args.summary_every == 0:
            print("    " + "-" * 70)
            print(f"    Segment {segment_index} best ({len(best_state)} feat): "
                  f"{state_str(best_state)}")
            print(f"    Segment best metrics: RMSE={best_rmse:.5f}  "
                  f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
            print(f"    GLOBAL best   metrics: RMSE={global_best_rmse:.5f}  "
                  f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}  "
                  f"({len(global_best_state)} feat)")
            ops_summary = "  ".join(
                f"{op.strip()}:{op_accepts[op]}/{op_accepts[op]+op_rejects[op]}"
                for op in op_accepts
            )
            print(f"    Accepted {n_accept}/{it}   Rejected {n_reject}/{it}   "
                  f"ops (acc/tot): {ops_summary}")
            print("    " + "-" * 70)

        # --- Restart trigger -----------------------------------------------
        # If the segment has stalled (too many consecutive rejects) AND we
        # still have iteration budget left, archive the segment and start
        # over with a fresh random state.
        if (consecutive_rejects >= args.reseed_after
                and it < args.iters
                and not interrupted_flag[0]):
            segment_history.append({
                "index": segment_index,
                "start": segment_start_iter,
                "end": it,
                "state": list(best_state),
                "rmse": best_rmse,
                "pdev": best_pdev,
                "r2": best_r2,
                "stalled": True,
            })
            print()
            print("=" * 78)
            print(f"[RESTART] Segment {segment_index} stalled "
                  f"({consecutive_rejects} consecutive rejects, "
                  f"iters {segment_start_iter}-{it}). Reseeding.")
            print(f"   Segment best:  RMSE={best_rmse:.5f}  "
                  f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
            print(f"   Global best:   RMSE={global_best_rmse:.5f}  "
                  f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}")

            segment_index += 1
            segment_start_iter = it + 1
            best_state = random_state(rng, families,
                                      min_slots=args.min_slots,
                                      max_slots=args.max_slots)
            print(f"   New segment {segment_index} random state "
                  f"({len(best_state)} features):")
            for i, (fam, d) in enumerate(best_state):
                print(f"     slot_{i}  {DISPLAY_NAME[fam]:9s}  d = {d:+.4f}")

            X = build_feature_matrix(orb, best_state)
            t0 = time.time()
            best_rmse, best_pdev, best_r2 = evaluate_cv(X, y, folds, args.seed)
            dt = time.time() - t0
            new_baseline_marker = ""
            if maybe_update_global(best_state, best_rmse, best_pdev, best_r2):
                new_baseline_marker = "  [NEW GLOBAL BEST]"
            print(f"   New baseline: RMSE={best_rmse:.5f}  "
                  f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}  "
                  f"({dt:.1f}s){new_baseline_marker}")
            print("=" * 78)
            consecutive_rejects = 0
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted — printing summary so far]")

    # Archive the final (in-progress) segment. Marked stalled=False since it
    # ended due to iter-budget exhaustion or Ctrl-C, not the reseed trigger.
    segment_history.append({
        "index": segment_index,
        "start": segment_start_iter,
        "end": last_it,
        "state": list(best_state),
        "rmse": best_rmse,
        "pdev": best_pdev,
        "r2": best_r2,
        "stalled": False,
    })

    print()
    print("=" * 78)
    if interrupted:
        print(f"Stopped early at iteration {last_it} of {args.iters}.")
    else:
        print(f"Done after {args.iters} iterations.")
    print(f"  Segments: {len(segment_history)}   "
          f"Accepted: {n_accept}   Rejected: {n_reject}")
    print(f"  By move type (accepted / tried):")
    for op in ["k=1 ", "k=2 ", "k=3 ", "add ", "rm  ", "swap"]:
        total = op_accepts[op] + op_rejects[op]
        print(f"    {op.strip():4s}  {op_accepts[op]:4d} / {total:4d}")

    print(f"\n  Per-segment final metrics:")
    for seg in segment_history:
        print(f"    seg {seg['index']:2d}  iters {seg['start']:4d}-{seg['end']:4d}"
              f"  ({seg['end'] - seg['start'] + 1:4d} long)   "
              f"RMSE={seg['rmse']:.5f}  Poisson={seg['pdev']:.5f}  "
              f"R2={seg['r2']:.5f}  N={len(seg['state'])}")

    # Tally family occurrences across stalled (local-minimum) segments.
    # Each slot counts once, so a segment with two vis_flux slots adds 2.
    stalled_segs = [s for s in segment_history if s["stalled"]]
    fam_counts = {fam: 0 for fam in FAMILIES}
    total_slots_stalled = 0
    for seg in stalled_segs:
        for fam, _ in seg["state"]:
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            total_slots_stalled += 1
    print(f"\n  Family usage at stalled local minima "
          f"({len(stalled_segs)} stalled segments, "
          f"{total_slots_stalled} total slots):")
    if total_slots_stalled == 0:
        print("    (none — no segments reached the reseed-after threshold)")
    else:
        ranked = sorted(fam_counts.items(), key=lambda kv: -kv[1])
        for fam, cnt in ranked:
            pct_slots = 100.0 * cnt / total_slots_stalled
            n_segs_with_fam = sum(
                1 for seg in stalled_segs if any(f == fam for f, _ in seg["state"])
            )
            pct_segs = 100.0 * n_segs_with_fam / len(stalled_segs)
            print(f"    {DISPLAY_NAME[fam]:9s}  slots={cnt:4d} ({pct_slots:5.1f}%)   "
                  f"in {n_segs_with_fam}/{len(stalled_segs)} segments ({pct_segs:5.1f}%)")

    print(f"\n  Current segment final state ({len(best_state)} features):")
    for i, (fam, d) in enumerate(best_state):
        print(f"    slot_{i}  {DISPLAY_NAME[fam]:9s}  d = {d:+.4f}")
    print(f"  Current segment metrics: RMSE={best_rmse:.5f}   "
          f"Poisson={best_pdev:.5f}   R2={best_r2:.5f}")

    print(f"\n*** GLOBAL BEST across all segments ({len(global_best_state)} features) ***")
    for i, (fam, d) in enumerate(global_best_state):
        print(f"    slot_{i}  {DISPLAY_NAME[fam]:9s}  d = {d:+.4f}")
    print(f"  Global best metrics: RMSE={global_best_rmse:.5f}   "
          f"Poisson={global_best_pdev:.5f}   R2={global_best_r2:.5f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
