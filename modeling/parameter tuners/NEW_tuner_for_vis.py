"""NEW_tuner_for_vis.py

Stochastic search for the optimal set of visibility-proxy "slots" that get
appended to a fixed baseline of orbital features when predicting how many
additional oppositions an asteroid should accumulate.

A *state* is an ordered list of (family, d) pairs, length between MIN_SLOTS
and MAX_SLOTS. Each pair contributes one column to the feature matrix; the
column is a visibility magnitude proxy computed from a, e, H and the
per-slot offset d.

Families come in two flavors:

  Standard families (single template):

      vis = 5 log10( r_X * max(r_X - d, eps) ) + H

    with characteristic radius r_X chosen per family:

      typical     r_X = a (1 + e/2)
      perihelion  r_X = a (1 - e) = q
      aphelion    r_X = a (1 + e) = Q
      flux        r_X = a (1 - e^2)^(1/4)

  Super families (physically-motivated alternatives):

      super1 (vis_timeavg): r = a(1 + e^2/2), Delta = max(r - d, eps),
                            vis = H + 5 log10(r * Delta)
      super4 (vis_S4):      r = a(1 + e^2/2), Delta = max(r - 1, eps),
                            alpha_max = arcsin(min(1/r, 1)),
                            vis = H + 5 log10(r * Delta) + d * alpha_max
                            (linear phase darkening; d in mag/rad)
      super5 (vis_S5):      r = a(1 + e^2/2),
                            Delta_quad = sqrt(max(r^2 - 1, eps)),
                            vis = H + 5 log10(r * max(Delta_quad - d, eps))
                            (quadrature geometry; d in AU)

Per-iteration move (drawn from MOVE_WEIGHTS):

    mutate   resample d on k ∈ {1,2,3} random slots from N(d, sigma)
             truncated to [D_LOW, D_HIGH]; family preserved.
    add      append a new (family, d) slot drawn uniformly.
    remove   drop one randomly chosen slot.
    swap     replace one slot's family (with d resampled uniformly).

Each candidate state is evaluated with 8-fold CV; per fold we fit one
LightGBM and one XGBoost regressor (both with a Poisson objective) and
average their predictions. We then compute three fold-averaged metrics
(RMSE, mean Poisson deviance, R^2) and accept if at least 2 of the 3
improve on the current best. This evaluator matches the prototype spec;
a future revision will swap it for AutoGluon OOF predictions.

Run:
    python NEW_tuner_for_vis.py [--iters N] [--seed S] [--subsample K]
                                [--n-folds F] [--reseed-after K]
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

import activityscope_utils as utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-3
D_LOW, D_HIGH = -0.2, 1.2

MIN_SLOTS = 4
MAX_SLOTS = 7

# (mutate, add, remove, swap)
MOVE_WEIGHTS = (0.60, 0.10, 0.10, 0.20)

STANDARD_FAMILIES = ("typical", "perihelion", "aphelion", "flux")
SUPER_FAMILIES = ("super1", "super4", "super5")
FAMILIES = STANDARD_FAMILIES + SUPER_FAMILIES

DISPLAY_NAME = {
    "typical":    "vis_typ",
    "perihelion": "vis_q",
    "aphelion":   "vis_Q",
    "flux":       "vis_flux",
    "super1":     "vis_timeavg",
    "super4":     "vis_S4",
    "super5":     "vis_S5",
}

# Fixed baseline feature columns (visibility slots get appended to these).
BASELINE_COLS = [
    'H', 'Node', "a", "e", 'i', 'orbital_period_sync', 'galactic_inc', 'Perihelion_direction_x_e', 'Perihelion_direction_y_e', 'Perihelion_direction_z_e', 'dec_perihelion',
           'vis_last_perihelion', 'vis_orbit_flux_opp', 'vis_orbit_flux_multi', 'vis_orbit_mag_multi',
]


# ---------------------------------------------------------------------------
# Visibility families
# ---------------------------------------------------------------------------

def _r_standard(family, a, e):
    if family == "typical":
        return a * (1.0 + e / 2.0)
    if family == "perihelion":
        return a * (1.0 - e)
    if family == "aphelion":
        return a * (1.0 + e)
    if family == "flux":
        return a * np.power(np.clip(1.0 - e ** 2, 0.0, None), 0.25)
    raise ValueError(f"not a standard family: {family}")


def visibility(a, e, H, family, d):
    """One visibility-proxy column for the given family and offset d."""
    if family in STANDARD_FAMILIES:
        r = _r_standard(family, a, e)
        r_safe = np.maximum(r, EPS)
        delta = np.maximum(r - d, EPS)
        return 5.0 * np.log10(r_safe * delta) + H

    # All super families share r_timeavg = a(1 + e^2/2)
    r = a * (1.0 + e ** 2 / 2.0)
    r_safe = np.maximum(r, EPS)

    if family == "super1":
        delta = np.maximum(r - d, EPS)
        return 5.0 * np.log10(r_safe * delta) + H
    if family == "super4":
        delta = np.maximum(r - 1.0, EPS)
        alpha_max = np.arcsin(np.minimum(1.0 / r_safe, 1.0))
        return 5.0 * np.log10(r_safe * delta) + H + d * alpha_max
    if family == "super5":
        delta_quad = np.sqrt(np.maximum(r ** 2 - 1.0, EPS))
        delta_eff = np.maximum(delta_quad - d, EPS)
        return 5.0 * np.log10(r_safe * delta_eff) + H

    raise ValueError(f"Unknown family: {family}")


def build_feature_matrix(orb, state):
    """Baseline columns + one visibility column per (family, d) in state."""
    X = orb[BASELINE_COLS].copy()
    a = orb["a"].to_numpy(dtype=np.float64)
    e = np.clip(orb["e"].to_numpy(dtype=np.float64), 0.0, 0.999)
    H = orb["H"].to_numpy(dtype=np.float64)
    for i, (fam, d) in enumerate(state):
        X[f"slot_{i}"] = visibility(a, e, H, fam, d).astype(np.float32)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Search-state primitives
# ---------------------------------------------------------------------------

def random_state(rng, min_slots=MIN_SLOTS, max_slots=MAX_SLOTS):
    n = rng.randint(min_slots, max_slots)
    return [(rng.choice(FAMILIES), rng.uniform(D_LOW, D_HIGH)) for _ in range(n)]


def state_str(state):
    return ", ".join(f"{DISPLAY_NAME[fam]}@{d:+.3f}" for fam, d in state)


def pick_move(rng, n_slots, min_slots=MIN_SLOTS, max_slots=MAX_SLOTS):
    """Drop the unavailable size-changing move at the bounds; remaining
    weights renormalize via random.choices."""
    options = ["mutate"]
    weights = [MOVE_WEIGHTS[0]]
    if n_slots < max_slots:
        options.append("add")
        weights.append(MOVE_WEIGHTS[1])
    if n_slots > min_slots:
        options.append("remove")
        weights.append(MOVE_WEIGHTS[2])
    options.append("swap")
    weights.append(MOVE_WEIGHTS[3])
    return rng.choices(options, weights=weights, k=1)[0]


def apply_move(rng, state, move, mutate_std):
    """Return (trial_state, op_label, change_descriptions)."""
    if move == "mutate":
        max_k = min(3, len(state))
        k = rng.randint(1, max_k)
        idxs = rng.sample(range(len(state)), k)
        trial = list(state)
        changes = []
        for i in idxs:
            fam, old_d = trial[i]
            new_d = rng.gauss(old_d, mutate_std)
            while new_d < D_LOW or new_d > D_HIGH:
                new_d = rng.gauss(old_d, mutate_std)
            trial[i] = (fam, new_d)
            changes.append(f"{DISPLAY_NAME[fam]}:{old_d:+.3f}->{new_d:+.3f}")
        return trial, f"k={k} ", changes

    if move == "add":
        fam = rng.choice(FAMILIES)
        d = rng.uniform(D_LOW, D_HIGH)
        trial = list(state) + [(fam, d)]
        return trial, "add ", [
            f"+{DISPLAY_NAME[fam]}@{d:+.3f} ({len(state)}->{len(trial)})"
        ]

    if move == "remove":
        i = rng.randrange(len(state))
        fam, d = state[i]
        trial = list(state[:i]) + list(state[i+1:])
        return trial, "rm  ", [
            f"-{DISPLAY_NAME[fam]}@{d:+.3f} ({len(state)}->{len(trial)})"
        ]

    if move == "swap":
        i = rng.randrange(len(state))
        old_fam, old_d = state[i]
        new_fam = rng.choice([f for f in FAMILIES if f != old_fam])
        new_d = rng.uniform(D_LOW, D_HIGH)
        trial = list(state)
        trial[i] = (new_fam, new_d)
        return trial, "swap", [
            f"{DISPLAY_NAME[old_fam]}@{old_d:+.3f}->{DISPLAY_NAME[new_fam]}@{new_d:+.3f}"
        ]

    raise ValueError(f"Unknown move: {move}")


# ---------------------------------------------------------------------------
# Evaluation: 8-fold CV averaging LightGBM + XGBoost Poisson regressors
# ---------------------------------------------------------------------------

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
    """Per fold: fit one LightGBM and one XGBoost Poisson regressor, average
    their predictions, score the average. Returns (rmse, pdev, r2) averaged
    across folds."""
    rmses, pdevs, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        p_xgb = _fit_predict_xgb(X_tr, y_tr, X_te, seed)
        p_lgb = _fit_predict_lgbm(X_tr, y_tr, X_te, seed)
        # Both models target the same scale (Poisson mean); plain mean blend.
        y_pred = np.maximum(0.5 * (p_xgb + p_lgb), EPS)

        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        pdevs.append(mean_poisson_deviance(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(rmses)), float(np.mean(pdevs)), float(np.mean(r2s))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(subsample=None, seed=0):
    """Mirror the notebook's training filter so this tuner sees the same
    'decent orbit' population the production models train on."""
    print("Loading orbit databases (this can take a minute)...")
    orb = utils.load_all_databases(apply_filters=True)
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
    orb = orb.dropna(subset=BASELINE_COLS + ["Num_opps_minus_one"])
    print(f"Post-filter row count: {len(orb)}")

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
                        help="Total stochastic iterations (default: 1000).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed for CV splits and model fits "
                             "(default: 0).")
    parser.add_argument("--search-seed", type=int, default=None,
                        help="Seed for the stochastic search itself. Default: "
                             "int(time.time()) so each run explores differently.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration.")
    parser.add_argument("--n-folds", type=int, default=8,
                        help="CV folds per candidate (default: 8).")
    parser.add_argument("--min-slots", type=int, default=MIN_SLOTS,
                        help=f"Minimum slot count (default: {MIN_SLOTS}).")
    parser.add_argument("--max-slots", type=int, default=MAX_SLOTS,
                        help=f"Maximum slot count (default: {MAX_SLOTS}).")
    parser.add_argument("--summary-every", type=int, default=25,
                        help="Print a check-in every N iters (default: 25).")
    parser.add_argument("--reseed-after", type=int, default=75,
                        help="Random-restart after this many consecutive "
                             "rejects within a segment (default: 75).")
    args = parser.parse_args()

    if args.search_seed is None:
        args.search_seed = int(time.time())
    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2.")
    if args.min_slots < 1 or args.max_slots < args.min_slots:
        parser.error("Bad --min-slots / --max-slots.")

    print(f"Deterministic seed (folds + model fits): {args.seed}")
    print(f"Search seed (family + d sampling):       {args.search_seed}  "
          f"(reproduce with --search-seed {args.search_seed})")
    print(f"CV: {args.n_folds}-fold, blending LightGBM + XGBoost (Poisson)")
    print(f"Slot range: [{args.min_slots}, {args.max_slots}]")

    rng = random.Random(args.search_seed)
    np.random.seed(args.seed)

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb["Num_opps_minus_one"].astype(np.float32).to_numpy()

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))

    # --- Initialization / baseline -----------------------------------------
    best_state = random_state(rng,
                              min_slots=args.min_slots,
                              max_slots=args.max_slots)
    print()
    print("=" * 78)
    print(f"Initial random state ({len(best_state)} slots):")
    for i, (fam, d) in enumerate(best_state):
        print(f"   slot_{i}  {DISPLAY_NAME[fam]:11s}  d = {d:+.4f}")
    print("=" * 78)

    X = build_feature_matrix(orb, best_state)
    print(f"Rows: {len(X)}  Features: {X.shape[1]}  "
          f"Fold sizes: {[len(te) for _, te in folds]}")

    print(f"\nScoring baseline ({args.n_folds}-fold, LGBM+XGB blend)...")
    t0 = time.time()
    best_rmse, best_pdev, best_r2 = evaluate_cv(X, y, folds, args.seed)
    dt = time.time() - t0
    print(f"Baseline: RMSE={best_rmse:.5f}  Poisson={best_pdev:.5f}  "
          f"R2={best_r2:.5f}  ({dt:.1f}s)")
    print()

    # Global best across all restart segments.
    global_best_state = list(best_state)
    global_best_rmse = best_rmse
    global_best_pdev = best_pdev
    global_best_r2 = best_r2

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
    interrupted_flag = [False]

    def _sigint_handler(signum, frame):
        if interrupted_flag[0]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[second Ctrl-C — exiting immediately]")
            raise KeyboardInterrupt
        interrupted_flag[0] = True
        print("\n[Ctrl-C received — finishing this iteration, then exiting "
              "(press again to force-quit)]")

    signal.signal(signal.SIGINT, _sigint_handler)

    def maybe_update_global(state, rmse, pdev, r2):
        nonlocal global_best_state, global_best_rmse, global_best_pdev, global_best_r2
        improvements = (
            int(rmse < global_best_rmse)
            + int(pdev < global_best_pdev)
            + int(r2 > global_best_r2)
        )
        if improvements >= 2:
            global_best_state = list(state)
            global_best_rmse = rmse
            global_best_pdev = pdev
            global_best_r2 = r2
            return True
        return False

    # --- Iteration loop ----------------------------------------------------
    try:
        for it in range(1, args.iters + 1):
            if interrupted_flag[0]:
                interrupted = True
                break
            last_it = it

            # Light annealing on the mutation jump size within each segment.
            iters_in_segment = it - segment_start_iter
            mutate_std = max(0.3, 1.0 - (iters_in_segment / 200.0) * 0.7)

            move = pick_move(rng, len(best_state),
                             min_slots=args.min_slots,
                             max_slots=args.max_slots)
            trial_state, op_label, changes = apply_move(
                rng, best_state, move, mutate_std
            )

            X_trial = build_feature_matrix(orb, trial_state)
            t0 = time.time()
            rmse, pdev, r2 = evaluate_cv(X_trial, y, folds, args.seed)
            dt = time.time() - t0

            improvements = (
                int(rmse < best_rmse)
                + int(pdev < best_pdev)
                + int(r2 > best_r2)
            )
            accept = improvements >= 2

            global_marker = ""
            if accept:
                n_accept += 1
                op_accepts[op_label] += 1
                best_state = trial_state
                best_rmse, best_pdev, best_r2 = rmse, pdev, r2
                X = X_trial
                consecutive_rejects = 0
                if maybe_update_global(best_state, best_rmse, best_pdev, best_r2):
                    global_marker = "  [NEW GLOBAL BEST]"
            else:
                n_reject += 1
                op_rejects[op_label] += 1
                consecutive_rejects += 1

            changes_str = " ".join(changes).ljust(70)
            print(
                f"[{it:4d}] {'ACCEPT' if accept else 'reject'}  {op_label}  "
                f"{changes_str}  "
                f"RMSE={rmse:.5f} ({rmse - best_rmse:+.5f})  "
                f"Poisson={pdev:.5f} ({pdev - best_pdev:+.5f})  "
                f"R2={r2:.5f} ({r2 - best_r2:+.5f})  "
                f"imp={improvements}/3  N={len(trial_state)}  "
                f"rej_streak={consecutive_rejects}  std={mutate_std:.2f}  "
                f"({dt:.1f}s){global_marker}"
            )

            if args.summary_every and it % args.summary_every == 0:
                print("    " + "-" * 70)
                print(f"    Segment {segment_index} best ({len(best_state)} slots): "
                      f"{state_str(best_state)}")
                print(f"    Segment metrics: RMSE={best_rmse:.5f}  "
                      f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
                print(f"    GLOBAL  metrics: RMSE={global_best_rmse:.5f}  "
                      f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}  "
                      f"({len(global_best_state)} slots)")
                ops_summary = "  ".join(
                    f"{op.strip()}:{op_accepts[op]}/{op_accepts[op]+op_rejects[op]}"
                    for op in op_accepts
                )
                print(f"    Accepted {n_accept}/{it}  Rejected {n_reject}/{it}  "
                      f"ops (acc/tot): {ops_summary}")
                print("    " + "-" * 70)

            # --- Restart on stall ------------------------------------------
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
                })
                print()
                print("=" * 78)
                print(f"[RESTART] Segment {segment_index} stalled at iter {it} "
                      f"({consecutive_rejects} consecutive rejects).")
                print(f"   Segment best: RMSE={best_rmse:.5f}  "
                      f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
                print(f"   Global  best: RMSE={global_best_rmse:.5f}  "
                      f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}")

                segment_index += 1
                segment_start_iter = it + 1
                best_state = random_state(rng,
                                          min_slots=args.min_slots,
                                          max_slots=args.max_slots)
                print(f"   New segment {segment_index} random state "
                      f"({len(best_state)} slots):")
                for i, (fam, d) in enumerate(best_state):
                    print(f"     slot_{i}  {DISPLAY_NAME[fam]:11s}  d = {d:+.4f}")

                X = build_feature_matrix(orb, best_state)
                t0 = time.time()
                best_rmse, best_pdev, best_r2 = evaluate_cv(X, y, folds, args.seed)
                dt = time.time() - t0
                marker = ""
                if maybe_update_global(best_state, best_rmse, best_pdev, best_r2):
                    marker = "  [NEW GLOBAL BEST]"
                print(f"   New baseline: RMSE={best_rmse:.5f}  "
                      f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}  "
                      f"({dt:.1f}s){marker}")
                print("=" * 78)
                consecutive_rejects = 0

    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted — printing summary]")

    # Record the final segment.
    segment_history.append({
        "index": segment_index,
        "start": segment_start_iter,
        "end": last_it,
        "state": list(best_state),
        "rmse": best_rmse,
        "pdev": best_pdev,
        "r2": best_r2,
    })

    # --- Final summary -----------------------------------------------------
    print()
    print("=" * 78)
    print("FINAL RUN SUMMARY")
    print("=" * 78)
    print(f"Iterations completed: {last_it} / {args.iters}"
          f"{'  (interrupted)' if interrupted else ''}")
    print(f"Accepted: {n_accept}   Rejected: {n_reject}")
    print(f"Ops (accept/total):  "
          + "  ".join(
              f"{op.strip()}:{op_accepts[op]}/{op_accepts[op]+op_rejects[op]}"
              for op in op_accepts
          ))
    print()
    print(f"Segments explored: {len(segment_history)}")
    for seg in segment_history:
        print(f"  seg {seg['index']:2d}  iters {seg['start']:4d}-{seg['end']:4d}  "
              f"({len(seg['state'])} slots)  "
              f"RMSE={seg['rmse']:.5f}  Poisson={seg['pdev']:.5f}  "
              f"R2={seg['r2']:.5f}")
        print(f"           {state_str(seg['state'])}")

    print()
    print("GLOBAL BEST")
    print("-" * 78)
    print(f"  ({len(global_best_state)} slots)")
    for i, (fam, d) in enumerate(global_best_state):
        print(f"   slot_{i}  {DISPLAY_NAME[fam]:11s}  d = {d:+.4f}")
    print(f"  RMSE={global_best_rmse:.5f}   Poisson={global_best_pdev:.5f}   "
          f"R2={global_best_r2:.5f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
