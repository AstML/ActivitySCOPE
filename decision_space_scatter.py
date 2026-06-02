"""Decision-space scatter for the ActivitySCOPE paper.

Plots P(N_opp >= 4) vs. dQ = quantile_Opps - Num_opps over the population
returned by the same filter used in the notebook's "NEW WOW generous ext diff"
cell, with confirmed/candidate actives highlighted.

Typical use from the notebook (after `final` has been built):

    import sys
    sys.path.insert(0, "additional code for paper")
    import decision_space_scatter as dss

    fig, ax, info = dss.decision_space_scatter(final)
    # info["actives_missing"] lists labeled actives rejected by the filter.

Pass output_path=... to save a PNG; pass pgf=True (and a .pgf path) for the
paper.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


DEFAULT_ACTIVES = [
    "2008 BJ22", "2025 HV38", "2019 OE31", "2010 RH69", "2015 BC566",
    "2025 VZ8", "2003 BM80", "2021 AY8", "2001 BV70", "2009 FP8",
    "2018 BJ11", "2008 GO98", "2010 TR241", "2026 EA122", "2002 CW116",
    "2017 QN84",
]

# Optional second labeled group; pass to highlight false positives separately.
DEFAULT_FALSE_POSITIVES = [
    "2025 HB52", "2023 UQ87", "2025 WW35", "2019 GX101",
    "2015 FH479", "2014 DZ209", "2009 US21", "2009 AN42", "2009 VW92",
    "2002 GR31", "2024 SG49",
]


def _safe_logit(p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _get_designations(df):
    if df.index.name == "Principal_desig" or "Principal_desig" not in df.columns:
        return df.index
    return df["Principal_desig"]


def apply_default_filter(
    final,
    ext_diff_threshold=0.002,
    a_range=(1.15, 20.0),
    arc_length_min=7,
    single_opp_only=True,
):
    """Same filter as the 'NEW WOW generous ext diff' notebook cell, with the
    additional restriction (on by default) to single-opposition objects only.

    Multi-opposition labeled actives (2010 RH69, 2003 BM80, 2008 GO98, 2017 QN84)
    will be dropped by this and surface in info['actives_missing'].
    """
    mask = (
        ((final["Arc_length"] > arc_length_min) | final["Arc_length"].isna())
        & final["a"].between(*a_range)
        & final["extension_difficulty"].between(0, ext_diff_threshold)
    )
    if single_opp_only:
        mask = mask & (final["Num_opps"] == 1)
    return final[mask]


def decision_space_scatter(
    final,
    active_designations=DEFAULT_ACTIVES,
    false_positive_designations=None,
    ext_diff_threshold=0.002,
    a_range=(1.15, 20.0),
    arc_length_min=7,
    single_opp_only=True,
    p_threshold_asteroidal=0.999,
    p_threshold_cometary=0.95,
    sort_threshold=3.0,
    tj_split=3.05,
    label_actives=True,
    label_fp=True,
    annotate_fontsize=7,
    ax=None,
    figsize=(8.5, 6.0),
    output_path=None,
    pgf=False,
    title="ActivitySCOPE decision space",
):
    """Plot P(N_opp>=4) vs. dQ for the WOW-generous filtered population.

    Parameters
    ----------
    final : pd.DataFrame
        The post-prediction frame (Principal_desig as index or column),
        containing at minimum: prob, quantile_Opps, Num_opps, Arc_length,
        a, extension_difficulty, TJ.
    active_designations : list[str]
        Designations to overplot as "active" (positive class).
    false_positive_designations : list[str] or None
        Optional second labeled group plotted as a distinct marker.
    ext_diff_threshold, a_range, arc_length_min :
        Filter parameters matching the notebook cell.
    p_threshold_asteroidal, p_threshold_cometary, sort_threshold :
        Reference threshold lines drawn on the plot.
    tj_split :
        T_J value used to color/shape the asteroidal vs. cometary split.
    label_actives, label_fp :
        Whether to annotate the labeled points with their designation.
    ax, figsize :
        Standard matplotlib; pass an existing axis to compose with other plots.
    output_path : str or None
        If given, savefig to this path.
    pgf : bool
        If True and output_path is given, use matplotlib's pgf backend with
        a serif/TeX rcparams context (for the paper).

    Returns
    -------
    fig, ax, info
        info contains:
          - 'actives_in_filter': labeled actives that survive the filter
          - 'actives_missing':   labeled actives rejected by the filter
          - 'fp_in_filter':      labeled FPs that survive the filter (if given)
          - 'fp_missing':        labeled FPs rejected by the filter
          - 'n_background':      number of unlabeled points plotted
    """
    df = apply_default_filter(
        final,
        ext_diff_threshold=ext_diff_threshold,
        a_range=a_range,
        arc_length_min=arc_length_min,
        single_opp_only=single_opp_only,
    ).copy()
    df["dq"] = df["quantile_Opps"] - df["Num_opps"]
    df["logit_prob"] = _safe_logit(df["prob"])

    desigs = _get_designations(df)
    df = df.assign(_desig=desigs.values)

    active_set = set(active_designations or [])
    fp_set = set(false_positive_designations or [])

    actives_mask = df["_desig"].isin(active_set)
    fp_mask = df["_desig"].isin(fp_set) & ~actives_mask
    bg_mask = ~actives_mask & ~fp_mask

    actives_in_filter = sorted(set(df.loc[actives_mask, "_desig"]))
    actives_missing = sorted(active_set - set(actives_in_filter))
    fp_in_filter = sorted(set(df.loc[fp_mask, "_desig"]))
    fp_missing = sorted(fp_set - set(fp_in_filter))

    def _draw(ax):
        bg = df[bg_mask]
        ast_bg = bg[bg["TJ"] >= tj_split]
        com_bg = bg[bg["TJ"] < tj_split]
        ax.scatter(
            ast_bg["dq"], ast_bg["logit_prob"],
            s=8, alpha=0.35, color="0.55",
            label=fr"Background, $T_J \geq {tj_split}$",
            rasterized=True, linewidths=0,
        )
        ax.scatter(
            com_bg["dq"], com_bg["logit_prob"],
            s=14, alpha=0.55, color="steelblue",
            label=fr"Background, $T_J < {tj_split}$",
            rasterized=True, linewidths=0,
        )

        act = df[actives_mask]
        ast_act = act[act["TJ"] >= tj_split]
        com_act = act[act["TJ"] < tj_split]
        ax.scatter(
            ast_act["dq"], ast_act["logit_prob"],
            s=70, color="crimson", edgecolor="black", marker="o",
            label=fr"Active/candidate, $T_J \geq {tj_split}$", zorder=5,
        )
        ax.scatter(
            com_act["dq"], com_act["logit_prob"],
            s=70, color="darkorange", edgecolor="black", marker="s",
            label=fr"Active/candidate, $T_J < {tj_split}$", zorder=5,
        )

        if label_actives:
            for _, row in act.iterrows():
                ax.annotate(
                    row["_desig"], (row["dq"], row["logit_prob"]),
                    fontsize=annotate_fontsize, xytext=(4, 4),
                    textcoords="offset points", zorder=6,
                )

        if len(fp_in_filter) > 0:
            fp = df[fp_mask]
            ax.scatter(
                fp["dq"], fp["logit_prob"],
                s=55, facecolor="none", edgecolor="black", marker="X",
                linewidths=1.2, label="False positives", zorder=4,
            )
            if label_fp:
                for _, row in fp.iterrows():
                    ax.annotate(
                        row["_desig"], (row["dq"], row["logit_prob"]),
                        fontsize=annotate_fontsize, color="0.25",
                        xytext=(4, -8), textcoords="offset points", zorder=6,
                    )

        ax.axvline(sort_threshold, ls="--", color="0.35", alpha=0.8,
                   label=fr"$\Delta Q = {sort_threshold:g}$")
        ax.axhline(_safe_logit(p_threshold_asteroidal), ls=":",
                   color="crimson", alpha=0.85,
                   label=fr"$P > {p_threshold_asteroidal}$ (ast.)")
        ax.axhline(_safe_logit(p_threshold_cometary), ls=":",
                   color="darkorange", alpha=0.85,
                   label=fr"$P > {p_threshold_cometary}$ (com.)")

        p_ticks = np.array([0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
        ax.set_yticks(_safe_logit(p_ticks))
        ax.set_yticklabels([f"{p:g}" for p in p_ticks])

        ax.set_xlabel(r"$\Delta Q = Q_{0.006} - N_{\mathrm{opp}}$")
        ax.set_ylabel(r"$P(N_{\mathrm{opp}} \geq 4)$")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    rc = {}
    if pgf:
        rc = {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }

    with matplotlib.rc_context(rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        _draw(ax)
        fig.tight_layout()

        if output_path is not None:
            if pgf:
                fig.savefig(output_path, backend="pgf", bbox_inches="tight")
            else:
                fig.savefig(output_path, bbox_inches="tight", dpi=200)

    info = {
        "actives_in_filter": actives_in_filter,
        "actives_missing": actives_missing,
        "fp_in_filter": fp_in_filter,
        "fp_missing": fp_missing,
        "n_background": int(bg_mask.sum()),
        "ext_diff_threshold": ext_diff_threshold,
        "a_range": a_range,
        "arc_length_min": arc_length_min,
    }
    return fig, ax, info
