"""tune_vis_offsets.py

Simulated-annealing search over the four `d` parameters of a fixed
four-feature visibility model:

    slot_0  vis_timeavg r = a (1 + e^2/2)            (super1)
    slot_1  vis_typ     r = a (1 + e/2)              (typical)
    slot_2  vis_flux    r = a (1 - e^2)^(1/4)        (flux)
    slot_3  vis_q        r = a (1 - e)                (perihelion)

All four slots use the same magnitude form:

    vis = 5 log10( r * max(r - d, eps) ) + H

The search state is just (d_0, d_1, d_2, d_3) — the slot identities are
fixed forever. Each iteration mutates either k=1 or k=2 of the d values
by sampling from a gaussian centered on the current d, with std cooling
from 1.0 to 0.3 over the first 200 iterations of each segment, truncated
to [D_LOW, D_HIGH].

Candidates are scored via k-fold cross-validation on a base XGBRegressor
with objective='count:poisson' against Num_opps_minus_one, averaging
RMSE, Poisson deviance, and R^2 across folds. The move is accepted if
at least 2 of those 3 metrics improve on the current best.

Restart: if `reseed-after` consecutive rejections accumulate within a
segment, the segment is archived and a fresh random d-tuple is sampled
to start a new segment.

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

import activityscope_utils as utils


# ---------------------------------------------------------------------------
# Fixed four-slot model
# ---------------------------------------------------------------------------

# (family_key, display_name) per slot, in display order. Each slot uses
# the same vis = 5 log10(r * max(r-d, eps)) + H form with r per family.
FIXED_SLOTS = [
    ("super1",     "vis_timeavg"),
    ("typical",    "vis_typ"),
    ("flux",       "vis_flux"),
    ("perihelion", "vis_q"),
]
N_SLOTS = len(FIXED_SLOTS)

# Non-vis features kept verbatim from the notebook's mlcols
NON_VIS_COLS = [
    "H", "Node", "a", "e", "i",
    "orbital_period_sync", "galactic_inc",
    "Perihelion_direction_x_e", "Perihelion_direction_y_e", "Perihelion_direction_z_e",
    "TJ", "dec_perihelion",
]

EPS = 1e-3
D_LOW, D_HIGH = -0.7, 1.0


# ---------------------------------------------------------------------------
# Feature construction and evaluation
# ---------------------------------------------------------------------------

def _r_for_family(a, e, family):
    if family == "super1":
        return a * (1.0 + e ** 2 / 2.0)
    if family == "typical":
        return a * (1.0 + e / 2.0)
    if family == "flux":
        return a * np.power(np.clip(1.0 - e ** 2, 0.0, None), 0.25)
    if family == "perihelion":
        return a * (1.0 - e)
    raise ValueError(f"Unknown family: {family}")


def visibility(a, e, H, family, d):
    """vis = 5 log10( r * max(r - d, eps) ) + H  with r per family."""
    r = _r_for_family(a, e, family)
    r_safe = np.maximum(r, EPS)
    delta = np.maximum(r - d, EPS)
    return 5.0 * np.log10(r_safe * delta) + H


def build_feature_matrix(orb, state):
    """Return X with NON_VIS_COLS plus one column per fixed slot, named
    slot_0, slot_1, slot_2, slot_3."""
    X = orb[NON_VIS_COLS].copy()
    a = orb["a"].to_numpy(dtype=np.float64)
    e = orb["e"].to_numpy(dtype=np.float64)
    H = orb["H"].to_numpy(dtype=np.float64)
    for i, ((fam, _name), d) in enumerate(zip(FIXED_SLOTS, state)):
        X[f"slot_{i}"] = visibility(a, e, H, fam, d).astype(np.float32)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def random_state(rng):
    """Sample a fresh uniform d ∈ [D_LOW, D_HIGH] for each fixed slot."""
    return [rng.uniform(D_LOW, D_HIGH) for _ in range(N_SLOTS)]


def apply_mutation(rng, state, mutate_std=1.0):
    """Mutate k ∈ {1, 2} slots' d-values from a truncated gaussian
    centered on the current d. Returns (trial_state, op_label, change_strs)."""
    k = rng.randint(1, 2)
    slot_indices = rng.sample(range(N_SLOTS), k)
    trial = list(state)
    change_strs = []
    for i in slot_indices:
        old_d = trial[i]
        new_d = rng.gauss(old_d, mutate_std)
        while new_d < D_LOW or new_d > D_HIGH:
            new_d = rng.gauss(old_d, mutate_std)
        trial[i] = new_d
        _fam, name = FIXED_SLOTS[i]
        change_strs.append(f"{name}:{old_d:+.3f}→{new_d:+.3f}")
    return trial, f"k={k}", change_strs


def state_str(state):
    """One-line compact representation of a state for periodic summaries."""
    return ", ".join(
        f"{name}@{d:+.3f}" for (_fam, name), d in zip(FIXED_SLOTS, state)
    )


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
    orb = utils.load_all_databases(apply_filters=True)
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
    orb = orb.dropna(subset=["H", "a", "e", "i", "Num_opps_minus_one"])
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
                        help="Seed for the stochastic search (d sampling). "
                             "Default: int(time.time()) at startup, so every "
                             "invocation explores differently. Pass an integer "
                             "to reproduce a specific run.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N rows for faster iteration (default: use all).")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds used to score each candidate "
                             "(default: 5, minimum: 2).")
    parser.add_argument("--summary-every", type=int, default=25,
                        help="Print a current-best summary every N iters (default: 25).")
    args = parser.parse_args()

    if args.search_seed is None:
        args.search_seed = int(time.time())
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2.")
    print(f"Deterministic seed (CV folds / XGB):  {args.seed}")
    print(f"Search seed (d sampling):             {args.search_seed}  "
          f"(pass --search-seed {args.search_seed} to reproduce this run)")
    print(f"CV: {args.n_folds}-fold cross-validation")
    print(f"Fixed slots ({N_SLOTS}): "
          + ", ".join(name for _fam, name in FIXED_SLOTS))
    rng = random.Random(args.search_seed)
    np.random.seed(args.seed)

    orb = prepare_training_data(subsample=args.subsample, seed=args.seed)
    y = orb["Num_opps_minus_one"].astype(np.float32).to_numpy()

    best_state = random_state(rng)
    print()
    print("=" * 78)
    print(f"Starting random state ({N_SLOTS} features):")
    for i, ((_fam, name), d) in enumerate(zip(FIXED_SLOTS, best_state)):
        print(f"   slot_{i}  {name:9s}  d = {d:+.4f}")
    print("=" * 78)

    print("\nBuilding initial feature matrix and preparing CV folds...")
    X = build_feature_matrix(orb, best_state)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(np.arange(len(orb))))
    fold_sizes = [len(test) for _, test in folds]
    print(f"Rows: {len(X)}   Features: {X.shape[1]}   "
          f"{args.n_folds}-fold sizes: {fold_sizes}")
    print(f"Columns: {list(X.columns)}")

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
    op_accepts = {"k=1": 0, "k=2": 0}
    op_rejects = {"k=1": 0, "k=2": 0}
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

        trial_state, op_label, change_strs = apply_mutation(
            rng, best_state, mutate_std=mutate_std
        )

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
            best_state = trial_state
            best_rmse, best_pdev, best_r2 = rmse, pdev, r2
            X = X_trial  # carry forward
            consecutive_rejects = 0
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
            f"imp={improvements}/3  "
            f"rej_streak={consecutive_rejects}  "
            f"std={mutate_std:.3f}  ({dt:.1f}s)"
            f"{global_marker}"
        )

        if args.summary_every and it % args.summary_every == 0:
            print("    " + "-" * 70)
            print(f"    Segment {segment_index} best: {state_str(best_state)}")
            print(f"    Segment best metrics: RMSE={best_rmse:.5f}  "
                  f"Poisson={best_pdev:.5f}  R2={best_r2:.5f}")
            print(f"    GLOBAL best   metrics: RMSE={global_best_rmse:.5f}  "
                  f"Poisson={global_best_pdev:.5f}  R2={global_best_r2:.5f}")
            ops_summary = "  ".join(
                f"{op}:{op_accepts[op]}/{op_accepts[op]+op_rejects[op]}"
                for op in op_accepts
            )
            print(f"    Accepted {n_accept}/{it}   Rejected {n_reject}/{it}   "
                  f"ops (acc/tot): {ops_summary}")
            print("    " + "-" * 70)

        # --- Restart trigger -----------------------------------------------
        # If the segment has stalled (too many consecutive rejects) AND we
        # still have iteration budget left, archive the segment and start
        # over with a fresh random d-tuple.
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
            best_state = random_state(rng)
            print(f"   New segment {segment_index} random state:")
            for i, ((_fam, name), d) in enumerate(zip(FIXED_SLOTS, best_state)):
                print(f"     slot_{i}  {name:9s}  d = {d:+.4f}")

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
    for op in ["k=1", "k=2"]:
        total = op_accepts[op] + op_rejects[op]
        print(f"    {op:4s}  {op_accepts[op]:4d} / {total:4d}")

    print(f"\n  Per-segment final metrics:")
    for seg in segment_history:
        print(f"    seg {seg['index']:2d}  iters {seg['start']:4d}-{seg['end']:4d}"
              f"  ({seg['end'] - seg['start'] + 1:4d} long)   "
              f"RMSE={seg['rmse']:.5f}  Poisson={seg['pdev']:.5f}  "
              f"R2={seg['r2']:.5f}")

    # Per-slot d statistics across ALL segments (each segment contributes
    # its final d-tuple, whether it stalled into a local minimum or ran
    # out of iteration budget).
    print(f"\n  Per-slot d across all {len(segment_history)} segments "
          f"(end-of-segment values):")
    for i, (_fam, name) in enumerate(FIXED_SLOTS):
        vals = np.array([seg["state"][i] for seg in segment_history])
        print(f"    slot_{i}  {name:9s}  "
              f"mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  max={vals.max():+.4f}  "
              f"n={len(vals)}")

    # Same statistics restricted to stalled (local-minimum) segments. These
    # are the points where the search actually got stuck, so spread here is
    # a sharper read on where the d-landscape has basins.
    stalled_segs = [s for s in segment_history if s["stalled"]]
    print(f"\n  Per-slot d at stalled local minima only "
          f"({len(stalled_segs)} stalled segments):")
    if not stalled_segs:
        print("    (none — no segments reached the reseed-after threshold)")
    else:
        for i, (_fam, name) in enumerate(FIXED_SLOTS):
            vals = np.array([seg["state"][i] for seg in stalled_segs])
            print(f"    slot_{i}  {name:9s}  "
                  f"mean={vals.mean():+.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():+.4f}  max={vals.max():+.4f}")

    print(f"\n  Current segment final state:")
    for i, ((_fam, name), d) in enumerate(zip(FIXED_SLOTS, best_state)):
        print(f"    slot_{i}  {name:9s}  d = {d:+.4f}")
    print(f"  Current segment metrics: RMSE={best_rmse:.5f}   "
          f"Poisson={best_pdev:.5f}   R2={best_r2:.5f}")

    print(f"\n*** GLOBAL BEST across all segments ***")
    for i, ((_fam, name), d) in enumerate(zip(FIXED_SLOTS, global_best_state)):
        print(f"    slot_{i}  {name:9s}  d = {d:+.4f}")
    print(f"  Global best metrics: RMSE={global_best_rmse:.5f}   "
          f"Poisson={global_best_pdev:.5f}   R2={global_best_r2:.5f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
