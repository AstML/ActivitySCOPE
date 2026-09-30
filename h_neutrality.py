"""Counterfactual absolute-magnitude sweep (Delta-H) for ActivitySCOPE.

Implements the paper's definition (Section "Candidate Ranking and Flagging Thresholds"):
holding each object's orbital and astrometric data fixed, step H_V *dimmer* from the
catalog value in 0.05 mag increments, recomputing the engineered features and the
regression expectation E[N_opp] at each step.  Delta-H = H' - H_V, where H' is the first
(i.e. closest to catalog) step at which E[N_opp] - N_opp <= 0.5.

    * An object already reconciled at its catalog H gets Delta-H = 0.
    * An object that never reconciles before H' reaches H_MAX also gets Delta-H = 0, with
      neutral_within_0p5 = False so it can be told apart from a genuine zero.

Shared by all three notebooks, which differ only in how they pick the candidates.

The reference epoch must be pinned (utils.set_reference_epoch) before calling: each
sweep frame holds only a handful of objects, and an unpinned frame would pick its own
"now" from those objects' (often displaced) epochs.
"""

import numpy as np
import pandas as pd

import activityscope_utils as utils

H_STEP = 0.05
H_MAX = 30.0
H_NEUTRALITY_TOLERANCE = 0.5

DH_COLS = ["H_neutral", "H_change_to_neutral", "exp_Num_opps_at_neutral_H",
           "neutrality_residual", "neutral_within_0p5"]

# Rows fed to feature_engineering / predict at once (candidates x steps).
_CHUNK_ROWS = 100_000


def delta_h_sweep(candidates, predictor_reg, observed_num_opps=None,
                  h_step=H_STEP, h_max=H_MAX, tolerance=H_NEUTRALITY_TOLERANCE):
    """Delta-H for every row of ``candidates`` (indexed by designation).

    observed_num_opps=None uses each candidate's Num_opps (orbit database); the comet
    file passes 1.  Returns a frame indexed by designation with columns DH_COLS.
    """
    if utils.configured_reference_epoch() is None:
        raise RuntimeError("Pin the reference epoch first: "
                           "utils.set_reference_epoch(utils.catalogue_reference_epoch(orb))")

    cand = candidates[~candidates.index.duplicated(keep="first")]
    cand = cand[cand["H"].notna()]
    if observed_num_opps is None:
        cand = cand[cand["Num_opps"].notna()]
        observed = pd.to_numeric(cand["Num_opps"], errors="coerce").to_numpy(dtype=float)
    else:
        observed = np.full(len(cand), float(observed_num_opps))
    h0 = pd.to_numeric(cand["H"], errors="coerce").to_numpy(dtype=float)
    n_steps = np.floor((h_max - h0) / h_step + 1e-9).astype(int) + 1
    n_steps = np.maximum(n_steps, 1)

    # E[N_opp] at every step for every candidate, built in chunks of whole candidates
    exp_by_cand = [None] * len(cand)
    start = 0
    while start < len(cand):
        stop = start
        rows = 0
        while stop < len(cand) and (rows == 0 or rows + n_steps[stop] <= _CHUNK_ROWS):
            rows += n_steps[stop]
            stop += 1
        pos = np.arange(start, stop)
        rep = np.repeat(pos, n_steps[pos])
        k = np.concatenate([np.arange(n) for n in n_steps[pos]])
        scen = cand.iloc[rep].reset_index(drop=True)
        scen["H"] = h0[rep] + k * h_step
        scen = utils.feature_engineering(scen)
        exp = np.asarray(predictor_reg.predict(scen), dtype=float) + 1
        for p, e in zip(pos, np.split(exp, np.cumsum(n_steps[pos])[:-1])):
            exp_by_cand[p] = e
        start = stop

    results = []
    for p, designation in enumerate(cand.index):
        residuals = exp_by_cand[p] - observed[p]
        hits = np.flatnonzero(residuals <= tolerance)
        found = len(hits) > 0
        k = hits[0] if found else 0
        results.append({
            "Designation": designation,
            "H_neutral": h0[p] + k * h_step,
            "H_change_to_neutral": k * h_step,
            "exp_Num_opps_at_neutral_H": exp_by_cand[p][k],
            "neutrality_residual": residuals[k],
            "neutral_within_0p5": found,
        })
    return pd.DataFrame(results, columns=["Designation"] + DH_COLS).set_index("Designation")


def attach_delta_h(frame, dh, keys):
    """Copy the Delta-H columns onto frame (NaN for rows that weren't swept). keys aligns frame rows to dh's index."""
    for col in DH_COLS:
        frame[col] = pd.Series(keys, index=frame.index).map(dh[col])
    frame["neutral_within_0p5"] = frame["neutral_within_0p5"].astype("boolean")
