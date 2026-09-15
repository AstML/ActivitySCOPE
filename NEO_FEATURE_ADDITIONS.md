# Columns to add when NEO performance matters

The 12-column `mlcols` was selected on the full training frame, which is main-belt dominated.
Trained and scored on near-Earth objects alone (q < 1.5 au, 29,602 training rows, paired 5-fold
LightGBM; the study is `modeling/neo_sfs/REPORT.md` on the `paper-diff-combined` branch,
2026-09-14) it is 9 % worse in Poisson deviance and 6 % worse in log-loss than the earlier
15-column survey-era model. The gap is almost entirely one column, and the columns below close it.
All of them are already produced by `feature_engineering` in `activityscope_utils.py`; adding them
is a change to `mlcols` only.

## What to add, in order

Gains are paired-CV deltas on the q < 1.5 rows, cumulative, on top of the current 12 (lower is
better; s.e. about 0.3 / 0.5 points).

| step | column | Poisson deviance | log-loss | where it comes from |
|---|---|---|---|---|
| 1 | `n_detect_windows` | -6.8 % | -2.8 % | default path since 04f8c5c (0.12 mag/yr ramp threshold) |
| 2 | `q` | -8.1 % | -4.3 % | `a * (1 - e)`; the MPC `Perihelion_dist` column is the same number |
| 3 | `orbital_period_sync` | | | default path |
| 4 | `vis_orbit_mag_multi` | -11.8 % | -6.9 % | `feature_engineering(orb, extra_features=True)` only |

Step 1 alone is worth about 70-80 % of the whole gain on both targets. Steps 1-2 match the old
15-column model on both targets. Steps 3-4 together beat it; they were measured as a pair.

What the columns do for NEOs:

* **`n_detect_windows`** walks the true geometry through the 20-yr window on a 10-day grid and
  counts separate intervals that are both brighter than the survey limit and more than 60 deg from
  the Sun. `n_detectable` scores each solved apparition at one instant with opposition geometry, so
  it counts an interior object's equal-longitude events next to the Sun and misses an Apollo's
  close approach far from opposition. For main-belt objects the two agree (which is why the
  compact revision dropped it, at -0.3 % on the full frame); for NEOs they do not. The current
  version thresholds on the same 0.12 mag/yr ramp as `n_detectable`, so it needs no survey
  calibration; the measured-calibration threshold was only marginally better (-7.1 % / -4.0 %).
* **`q`** separates Atens, Apollos and Amors, and lets the trees condition the other columns on
  how deep inside 1 au the orbit goes. `vis_q` already acts as a q < 1 indicator (see below), but
  the distance itself still adds 1.2 / 1.5 points.
* **`orbital_period_sync`** (|P - round(P)| in years): an NEO with a period near 1 yr has a very
  long synodic period and returns to the same Earth-relative geometry only after many years.
  `Synodic_period` and a synodic count carry the same information less well.
* **`vis_orbit_mag_multi`** is the orbit-averaged brightness from the 32 x 24 grid. It helps the
  regression (-1.6 / -1.0 points once steps 1-2 are in) and is the third greedy pick; the two other
  orbit-averaged columns (`spatial_discoverability_fraction`, `dec_flux_weighted`) add only
  0.1-0.3 points and are not worth their explanation.

## How to add them

1. Add the column names to `mlcols` in both notebooks (`Start Here - ActivitySCOPE simplified
   demo.ipynb` and `More Complicated Workbook - including paper-specific items.ipynb`; the lists
   must stay identical) and to `CURRENT_MLCOLS` in `sfs_neo.py`.
2. `q` is not created by `feature_engineering`; add `orb["q"] = orb["a"] * (1 - orb["e"])` next to
   the `mlcols` definition, or use `Perihelion_dist`.
3. Step 4 needs `feature_engineering(orb, extra_features=True)` wherever the training and
   prediction frames are built; it adds several minutes on the full catalogue.
4. Update the feature descriptions in the paper (Section: Model Features) and
   `paper_code_notation.md`.

## Before changing `mlcols`

The gains above are NEO-only fits. The production model is trained on the whole catalogue, so run
one paired 5-fold on the full training frame (the protocol of `modeling/compact_revision`) scored
both on all rows and on the q < 1.5 subset, to confirm the NEO gain carries over and the main belt
does not lose. `n_detect_windows` cost -0.3 % on the full frame in the compact-revision study, so
no loss is expected, but it has not been measured with the ramp threshold.

## What not to change

* **`vis_q`.** Replacing its clamped `max(q - 1, 1e-3)` with `|q - 1|` is worse for NEOs
  (+1.9 % / +0.7 %). The clamp sends every q < 1 object to a very bright value no exterior object
  can reach, so the column doubles as an interior-orbit indicator; an explicit q < 1 flag adds
  nothing. Keep the clamped form. Adding the absolute form as a second column gives the
  classifier -0.85 % and the regression nothing.
* **Encounter-geometry columns.** Twenty-five NEO-specific candidates (Earth MOID, Opik encounter
  velocity, sky rate, phase and elongation at closest approach, time fractions inside 1 au or
  brighter than the limit, nodal distances, Aten/Apollo/Amor class, ...; `modeling/neo_sfs/
  neo_features.py`) are each worth under 1.5 points alone and are redundant once
  `n_detect_windows` is in. Only `earth_moid` (-0.9 %, regression) and `sky_rate_moid` (-0.5 %,
  classifier) still made the greedy path afterwards.
* **The survey-era `E_*` columns.** `E_50yr` is -0.8 / -1.1 points as a single add and needs the
  measured survey calibration; not worth re-introducing for this.
