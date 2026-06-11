# Paper ↔ Code Notation Reference

A bidirectional glossary linking the symbolic notation used in the paper to the column names used in the implementation ([activityscope_utils.py](activityscope_utils.py) and [ActivitySCOPE_simplified_demo.ipynb](ActivitySCOPE_simplified_demo.ipynb)). The intent is that the paper can use whatever notation reads best (typically Greek letters and math italic) without forcing the code to follow suit, and vice versa.

In the tables below, the **Paper** column shows the LaTeX source as it appears in the manuscript; the **Code** column shows the literal Python identifier or pandas column name; **Units** and **Notes** give additional context where relevant.

---

## 1. Keplerian orbital elements

| Paper | Code | Units | Notes |
|---|---|---|---|
| `$a$` | `a` | AU | Semi-major axis |
| `$e$` | `e` | — | Eccentricity |
| `$i$` | `i` | degrees | Inclination |
| `$\Omega$` | `Node` | degrees | Longitude of ascending node |
| `$\omega$` | `Peri` | degrees | Argument of perihelion |
| `$M$` | `M` | degrees | Mean anomaly at epoch |
| `$P_{\rm yr}$` | `Orbital_period` | years | Orbital period (sidereal) |

## 2. Derived orbital quantities

| Paper | Code | Units | Notes |
|---|---|---|---|
| `$q$` | `Perihelion_dist` | AU | Perihelion distance $a(1-e)$; renamed to `q` after sorting in `final` ([cell 20](ActivitySCOPE_simplified_demo.ipynb)) |
| `$Q$` | `Aphelion_dist` | AU | Aphelion distance $a(1+e)$; renamed to `Q` in `final` |
| `$p$` | `r_p` (local) | AU | Semi-latus rectum $a(1-e^2)$; not stored as a column |
| `$r_{\rm typ}$` | `r` (local) | AU | Characteristic radius $a(1 + e/2)$ used inside `feature_engineering` |
| `$T_J$` | `TJ` | — | Tisserand parameter with respect to Jupiter ($a_J = 5.203$ AU) |
| `$H$` / `$H_V$` | `H` | mag | Absolute magnitude in the $HG$ system; after [activityscope_utils.py:678](activityscope_utils.py#L678) this is the dimmest of the corrected MPC, AstDyS, and JPL values |
| `$H_V^{\rm MPC}$` | `H_MPC` | mag | The MPC-only $H_V$ before the pessimistic-max step |

## 3. Observational metadata (from MPC + astrometry counter)

| Paper | Code | Notes |
|---|---|---|
| `$N_{\rm opp}$` | `Num_opps` | Number of oppositions on which observations have been linked |
| `$N_{\rm obs}$` | `Num_obs` | Total number of astrometric observations |
| `$U$` | `U` | MPC uncertainty parameter (0–9, with 9 = worst; missing values are filled with 10 in [activityscope_utils.py:125](activityscope_utils.py#L125)) |
| Arc length | `Arc_length` | Days between first and last observation (NaN for multi-opposition objects after [activityscope_utils.py:433](activityscope_utils.py#L433)) |
| — | `nights_total` | Number of distinct observing nights, from `astrometry_counts.json` |
| — | `opp_with_most_nights` | Number of nights on the most heavily-observed opposition |
| — | `opp_with_second_most_nights` | Number of nights on the second-most-observed opposition |
| — | `other_opps` | `nights_total − opp_with_most_nights` |
| — | `longest_opp_arc`, `longest_gap_arc`, `second_longest_gap_arc`, `shortest_gap_arc` | Arc-length statistics across oppositions |
| — | `v_mag_min`, `v_mag_max`, `v_mag_avg`, `v_mag_gap` | Reported magnitudes across the observation history |

## 4. Model targets, predictions, and derived rankings

| Paper | Code | Notes |
|---|---|---|
| Positive class indicator $\mathbb{1}[N_{\rm opp} \ge 4]$ | `Is_Past_Threshold` | Binary classifier training label, defined in [activityscope_utils.py:455](activityscope_utils.py#L455) |
| $N_{\rm opp} - 1$ | `Num_opps_minus_one` | Poisson- and quantile-regression training label |
| `$P(N_{\rm opp} \ge 4)$` | `prob` | Out-of-fold or held-out probability from the binary classifier |
| `$E[N_{\rm opp}]$` | `exp_Num_opps` | Expected number of oppositions from the Poisson regressor (`+1` to undo the `minus_one` shift) |
| 0.006-quantile expectation | `quantile_Opps` | Lower-quantile prediction from the pinball regressor (again `+1`) |
| `$E[N_{\rm opp}] - N_{\rm opp}$` | `Num_opps_diff` | Observed-opposition deficit |
| `$E[N_{\rm opp}] / N_{\rm opp}$` | `Num_opps_mult` | Observed-opposition ratio |
| `quantile_Opps` $- N_{\rm opp}$ | `DeltaQ` | Sorting key used to rank under-observed multi-opposition candidates |
| `$S_{EV}$` | `poisson_cdf` | No notes |
| Extension-difficulty score | `extension_difficulty` | Output of the secondary classifier described in Appendix A.3 |

## 5. Visibility and Engineered Features

| Paper | Code | Notes |
|---|---|---|
| `vis_orbit_mag_multi` | `vis_orbit_mag_multi` | Spatial integration of visibility across the orbit (using $a$, $e$, $H_V$) at multiple Earth longitude offsets |
| `vis_opp_mean` | `vis_opp_mean` | Mean expected magnitude over the last five equal-longitude apparitions (typically oppositions) |
| `vis_timeavg` | `vis_timeavg` | Typical-geometry visibility magnitude using time-averaged heliocentric distance |
| `vis_q` | `vis_q` | Typical perihelion visibility magnitude |
| `dec_flux_weighted` | `dec_flux_weighted` | Ecliptic (or Equatorial) declination time-weighted by $1/r^2$ received flux |
| `Perihelion_direction_x_e` | `Perihelion_direction_x_e` | x-component of perihelion direction weighted by eccentricity $e$ |
| `Perihelion_direction_y_e` | `Perihelion_direction_y_e` | y-component of perihelion direction weighted by eccentricity $e$ |
| `orbital_period_sync` | `orbital_period_sync` | Distance of the orbital period to the nearest integer year |
