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

## 3. Engineered visibility (`vis`-family) features

All visibility features have units of magnitudes and are computed inside [activityscope_utils.py:470-622](activityscope_utils.py#L470-L622).

| Paper | Code | Notes |
|---|---|---|
| `$\texttt{vis}$` | `vis` | Typical magnitude proxy using $r_{\rm typ}$ |
| `$\texttt{vis\_q}$` | `vis_q` | Perihelion magnitude proxy (note: code uses no underscore) |
| `$\texttt{vis\_Q\_0.6}$` | `vis_Q_0.6` | Aphelion magnitude proxy with explicit geocentric distance |
| `$\texttt{vis\_p\_0.6}$` | `vis_p_0.6` | Semi-latus-rectum magnitude proxy |
| `$\texttt{vis\_flux\_0.0}$` | `vis_flux_0.0` | Flux-weighted mean magnitude proxy, $d = 0.0$ AU offset |
| `$\texttt{vis\_flux\_0.3}$` | `vis_flux_0.3` | Flux-weighted mean magnitude proxy, $d = 0.3$ AU offset |

The numeric suffix in the `_0.6`, `_0.3`, `_0.0` names is the offset $d$ (AU) added to Earth's orbital radius to form the geocentric-distance approximation $r_X - (1 + d)$.

> **Naming caveat.** The `vis_q` form in the paper is purely typographic. The code column is `vis_q` (no underscore). Both refer to the same quantity. If consistency is desired in a future cleanup, renaming `orb['vis_q']` → `orb['vis_q']` in [activityscope_utils.py:501](activityscope_utils.py#L501) and downstream is straightforward; until then, the paper's `vis_q` should be read as the same column.

## 4. Engineered geometric / orientation features

| Paper | Code | Notes |
|---|---|---|
| `$x e$, $y e$` | `Perihelion_direction_x_e`, `Perihelion_direction_y_e` | Eccentricity-scaled heliocentric ecliptic direction cosines of the perihelion point |
| `$\texttt{orbital\_period\_sync}$` | `orbital_period_sync` | $|P_{\rm yr} - \mathrm{round}(P_{\rm yr})|$ |
| `$\hat{\mathbf{n}}_{\rm orb}$` | `n_ast` (local) | Unit normal to the orbital plane, in ecliptic coords |
| `$\hat{\mathbf{n}}_{\rm gal}$` | `n_gal` (local) | Unit vector toward the north galactic pole, in ecliptic coords |

## 5. Observational metadata (from MPC + astrometry counter)

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

## 6. Model targets, predictions, and derived rankings

| Paper | Code | Notes |
|---|---|---|
| Positive class indicator $\mathbb{1}[N_{\rm opp} \ge 4]$ | `Is_Past_Threshold` | Binary classifier training label, defined in [activityscope_utils.py:455](activityscope_utils.py#L455) |
| $N_{\rm opp} - 1$ | `Num_opps_minus_one` | Poisson- and quantile-regression training label |
| `$P(N_{\rm opp} \ge 4)$` | `prob` | Out-of-fold or held-out probability from the binary classifier |
| `$E[N_{\rm opp}]$` | `exp_Num_opps` | Expected number of oppositions from the Poisson regressor (`+1` to undo the `minus_one` shift) |
| 0.006-quantile expectation | `quantile_Opps` | Lower-quantile prediction from the pinball regressor (again `+1`) |
| `$E[N_{\rm opp}] - N_{\rm opp}$` | `Num_opps_diff` | Observed-opposition deficit |
| `$E[N_{\rm opp}] / N_{\rm opp}$` | `Num_opps_mult` | Observed-opposition ratio |
| `quantile_Opps` $- N_{\rm opp}$ | `sort` | Sorting key used to rank under-observed multi-opposition candidates ([cell 27](ActivitySCOPE_simplified_demo.ipynb)) |
| `$S_{EV}$` | `poisson_cdf` | No notes
| Extension-difficulty score | `extension_difficulty` | Output of the secondary classifier described in Appendix A.3 |

## 7. Catalog comparison columns

| Paper | Code | Notes |
|---|---|---|
| `$\Delta a$` | `a_diff_abs` | $\lvert a_{\rm MPC} - a_{\rm AstDyS} \rvert$ |
| `$\Delta e$` | `e_diff_abs` | analogous |
| `$\Delta i$` | `i_diff_abs` | analogous |
| `$\Delta H$` (MPC vs AstDyS) | `H_diff_abs` | analogous |
| `$\Delta H$` (MPC vs JPL) | `H_diff_abs_jpl` | analogous |
| $\max(\Delta H)$ across catalogs | `H_diff_abs_max` | Largest of the two $\Delta H$ values; used as a training filter |
| Catalog-disagreement flag | `multi_opp_disagree` | 1 if MPC and AstDyS disagree on single- vs multi-opposition status |

## 8. Physical and conventional constants used in the code

| Paper | Code (literal) | Value | Notes |
|---|---|---|---|
| `$a_J$` | `5.203` | 5.203 AU | Jupiter semi-major axis used in $T_J$ |
| `$\epsilon$` (obliquity) | `np.radians(23.44)` | 23.44° | Obliquity of the ecliptic used for $\delta_{\rm peri}$ |
| `$\hat{\mathbf{n}}_{\rm gal}$` | `np.array([-0.8676, 0.0104, 0.4971])` | — | North galactic pole direction in ecliptic frame |
| Geocentric-distance offset $d$ | `0.0`, `0.3`, `0.6` | AU | Encoded in `vis_*` suffixes |

## 9. Columns computed but not in `mlcols`

These are produced by [activityscope_utils.py: feature_engineering](activityscope_utils.py#L470-L622) but are not fed to the model (see [cell 9](ActivitySCOPE_simplified_demo.ipynb)). They are listed here for completeness in case the paper references them in passing or future work re-introduces them.

| Code | Definition |
|---|---|
| `Perihelion_direction_x`, `_y`, `_z` | Unscaled (no $e$ multiplier) perihelion direction cosines |
| `is_trojan` | Boolean: $5.0 < a < 5.4$, $e < 0.3$, $i < 40^\circ$ |
| `vis_node_0.0` | Visibility proxy at the closer of the two ecliptic nodes |
| `node_plus_peri` | $(\Omega + \omega) \bmod 360$ |

---

## Cross-reference: where each notation choice appears

- Greek letters $\Omega$, $\omega$, $i$ are used throughout the paper (§2, §A.1, §A.2) for orbital elements; they map to `Node`, `Peri`, `i` in code.
- The Tisserand symbol $T_J$ corresponds to the column `TJ` (no subscript) — typical for ASCII-only environments.
- The classifier-output column `prob` is the value rendered in the paper's tables as $P(N_{\rm opp} \ge 4)$.
