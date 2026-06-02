"""Render the ActivitySCOPE results table (``tab:activityscope_results``) for the paper.

Everything in the table comes from the model DataFrame (``final`` / ``cometmerge``)
*except* the free-text per-object note, the confirmed-active asterisk, and the
cometary/asteroidal grouping.  Those three live in ``table_notes.csv``:

    designation , group , confirmed , note

  * ``designation`` -- plain (un-subscripted) object name.  It is both the join key
    against the DataFrame's ``Object`` column and the displayed name.
  * ``group``       -- ``cometary`` or ``asteroidal``.  The cometary group is further
    split by opposition count into three rendered sections:
      1. Multi-opposition cometary (N_opp > 1)  -- sorted by S_EV ascending.
      2. Single-opposition cometary (N_opp <= 1 or unknown) -- sorted by P(N_opp>=4) desc.
      3. Asteroidal -- sorted by P(N_opp>=4) desc.
    Objects with no model match (or NA in the sort column) sort to the bottom of
    their section.  Comet-file objects have NA ``Num_opps`` and therefore land in
    the single-opposition section unless you supply ``Num_opps`` for them.
  * ``confirmed``   -- ``1`` appends ``$^{\\ast}$`` (confirmed active), ``0`` does not.
  * ``note``        -- raw LaTeX placed verbatim in the final column (``\\notecell{...}``,
    ``\\citet{}``, ``\\tabnote{}`` and bare text are all preserved as-is).

Usage from the notebook::

    import paper_table, importlib; importlib.reload(paper_table)
    combined = paper_table.combine_sources(for_paper, cometmerge)   # build the data
    latex = paper_table.build_table(combined, "table_notes.csv")
    print(latex)                                                    # paste into main.tex

Numbers reproduce the existing column meanings:
    a, e, i, H_V (=H), T_J (=TJ), N_opp (=Num_opps), E[N_opp] (=exp_Num_opps),
    P(N_opp>=4) (=prob), DeltaQ (=quantile_Opps - Num_opps),
    S_EV (=poisson.cdf(Num_opps-1, exp_Num_opps-1)).
"""

import csv
import re

import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- data columns the table reads from the combined DataFrame ----------------
# Each entry: (DataFrame column, format function).  Order matches the table.
DATA_COLUMNS = ["a", "e", "i", "H", "TJ", "Num_opps", "exp_Num_opps", "prob", "DeltaQ", "S_EV"]

# Subscript the cycle number of a provisional designation: "2010 RH69" -> "2010 RH$_{69}$".
# Matches a 4-digit year, a space, two capital letters, then the trailing digits.
_SUBSCRIPT_RE = re.compile(r"(\b\d{4}\s[A-Z]{2})(\d+)")

# Leading periodic-comet number prefix, e.g. "282P/" in "282P/(323137) 2003 BM80".
_COMET_PREFIX_RE = re.compile(r"^\d+[PDAI]/")


def format_designation(name, confirmed=False):
    """Plain designation -> LaTeX object cell content (subscripts + optional asterisk)."""
    formatted = _SUBSCRIPT_RE.sub(r"\1$_{\2}$", name)
    if confirmed:
        formatted += r"$^{\ast}$"
    return formatted


# --- numeric formatters ------------------------------------------------------
def _fixed(decimals):
    def fmt(v):
        if pd.isna(v):
            return ""
        return f"{v:.{decimals}f}"
    return fmt


def _int_fmt(v):
    if pd.isna(v):
        return ""
    return f"{v:.0f}"


def _sci_latex(v):
    """Small probability -> LaTeX scientific, e.g. 5.6e-05 -> ``$5.6\\times10^{-5}$``."""
    if pd.isna(v):
        return ""
    if v == 0:
        return "0"
    if v >= 1e-3:                      # not tiny: show plainly
        return f"{v:.4f}"
    mantissa, exponent = f"{v:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


# DataFrame column -> formatter.  Tweak here to change precision.
FORMATTERS = {
    "a": _fixed(2),
    "e": _fixed(3),
    "i": _fixed(2),
    "H": _fixed(1),
    "TJ": _fixed(2),
    "Num_opps": _int_fmt,
    "exp_Num_opps": _fixed(1),
    "prob": _fixed(6),
    "DeltaQ": _fixed(1),
    "S_EV": _sci_latex,
}


# --- data assembly -----------------------------------------------------------
def combine_sources(for_paper, cometmerge):
    """Concatenate the orbit-derived and comet-file objects into one lookup frame.

    Returns a DataFrame with an ``Object`` column plus all columns needed by the
    table.  ``DeltaQ`` and ``S_EV`` are computed if absent.
    """
    base_cols = ["Object", "a", "e", "i", "H", "TJ", "Num_opps", "exp_Num_opps", "prob",
                 "quantile_Opps", "DeltaQ", "poisson_cdf"]

    a = for_paper.copy()
    c = cometmerge.rename(columns={"Designation_and_name": "Object"}).copy()

    frames = []
    for df in (a, c):
        present = [col for col in base_cols if col in df.columns]
        frames.append(df[present])
    combined = pd.concat(frames, ignore_index=True)

    return _ensure_derived(combined)


def _ensure_derived(df):
    df = df.copy()
    if "DeltaQ" not in df.columns:
        if "quantile_Opps" in df.columns and "Num_opps" in df.columns:
            df["DeltaQ"] = df["quantile_Opps"] - df["Num_opps"]
        else:
            df["DeltaQ"] = np.nan
    if "S_EV" not in df.columns:
        if "poisson_cdf" in df.columns:
            df["S_EV"] = df["poisson_cdf"]
        elif {"Num_opps", "exp_Num_opps"}.issubset(df.columns):
            n = pd.to_numeric(df["Num_opps"], errors="coerce")
            e = pd.to_numeric(df["exp_Num_opps"], errors="coerce")
            df["S_EV"] = poisson.cdf(n - 1, e - 1)
        else:
            df["S_EV"] = np.nan
    return df


# --- note CSV ----------------------------------------------------------------
def load_notes(notes_csv):
    """Read the notes CSV, preserving row order. Returns a list of dicts."""
    with open(notes_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["confirmed"] = str(r.get("confirmed", "")).strip() in ("1", "True", "true")
        r["nopp_dagger"] = str(r.get("nopp_dagger", "")).strip() in ("1", "True", "true")
        r["group"] = r.get("group", "").strip().lower()
        r["designation"] = r["designation"].strip()
    return rows


def _lookup(df_by_object, designation):
    """Find the data row for a designation, with a comet-prefix-strip fallback."""
    if designation in df_by_object.index:
        return df_by_object.loc[designation]
    stripped = _COMET_PREFIX_RE.sub("", designation)
    if stripped != designation and stripped in df_by_object.index:
        return df_by_object.loc[stripped]
    return None


# --- LaTeX templates (verbatim from main.tex) --------------------------------
_HEADER = r"""\begin{longrotatetable}

\begin{longtable}{lrrrrrrrrrrl}
\caption{\activityscope\ candidates, recoveries, and false positives. Confirmed active objects are marked with an asterisk ($^{\ast}$).}
\label{tab:activityscope_results}\\
\toprule
Object & $a$ & $e$ & $i~(^\circ)$ &
$H_V$\tabnote{Absolute magnitude in the $V$ band, assuming an inert object.} &
$T_J$\tabnote{Tisserand parameter with respect to Jupiter.} &
$N_{\rm opp}$\tabnote{Number of oppositions on which the object has been observed.} &
$E[N_{\rm opp}]$\tabnote{Expected number of observed oppositions for an inert object of the same orbit and $H_V$, predicted by the regression model.} &
$P(N_{\rm opp}\ge4)$\tabnote{Binary-classifier probability that an inert object of the same orbit and $H_V$ would be observed on at least four oppositions.} &
$\Delta Q$\tabnote{Quantile deficit $\Delta Q = Q_{0.006} - N_{\rm opp}$, where $Q_{0.006}$ is the model's $0.006$-quantile prediction of the opposition count. Larger positive values indicate a larger deficit.} &
$S_{\rm EV}$\tabnote{Poisson lower-tail anomaly score $S_{\rm EV}=F_{\rm Pois}(N_{\rm opp}-1;\,E[N_{\rm opp}]-1)$. Smaller values indicate a larger deficit relative to $E[N_{\rm opp}]$.} &
Note \\
\midrule
\endfirsthead

\caption[]{\activityscope\ candidates, recoveries, and false positives (continued).}\\
\toprule
Object & $a$ & $e$ & $i~(^\circ)$ &
$H_V$\textsuperscript{a} & $T_J$\textsuperscript{b} & $N_{\rm opp}$\textsuperscript{c} &
$E[N_{\rm opp}]$\textsuperscript{d} & $P(N_{\rm opp}\ge4)$\textsuperscript{e} &
$\Delta Q$\textsuperscript{f} & $S_{\rm EV}$\textsuperscript{g} & Note \\
\midrule
\endhead

\midrule
\multicolumn{12}{r}{Continued on next page}\\
\endfoot

\endlastfoot
"""

_FOOTER = r"""\bottomrule
\multicolumn{12}{l}{\footnotesize $^{\ast}$\,Confirmed active object.}\\
\multicolumn{12}{l}{\footnotesize $^{\dagger}$\,Single-opposition at the time of flagging; tabulated $N_{\rm opp}$, $\Delta Q$, and $S_{\rm EV}$ are the values the model assigns to an otherwise-identical single-opposition object.}\\
\printtabnotes

\end{longtable}

\end{longrotatetable}"""

# Section labels and sort captions (edit here to retitle / re-caption a section).
SECTION_MULTI_COMET = "Cometary orbits (multiple oppositions when flagged)"
SECTION_SINGLE_COMET = "Cometary orbits (single opposition when flagged)"
SECTION_ASTEROID = "Asteroidal orbits"

_SORT_SEV_ASC = r"sorted by $S_{\rm EV}$, ascending"
_SORT_PROB_DESC = r"sorted by $P(N_{\rm opp}\ge4)$, descending"


_DAGGER = r"$^{\dagger}$"


def _num(data, col):
    """Scalar value of ``col`` as a float, or NaN if missing/NA/non-numeric."""
    if data is None or col not in data:
        return float("nan")
    try:
        return float(data[col])
    except (TypeError, ValueError):
        return float("nan")


def _as_single_opp(data):
    """Return a copy of ``data`` displayed as a single-opposition object.

    Forces ``Num_opps = 1`` and recomputes the opposition-count-dependent fields
    ``DeltaQ`` (= quantile_Opps - 1) and ``S_EV`` (= poisson.cdf(0, exp-1)) from the
    model's orbit/H predictions, so daggered comet rows (NA ``Num_opps``) show the
    values the model assigns to an otherwise-identical single-opposition object.
    """
    if data is None:
        return None
    d = data.copy()
    d["Num_opps"] = 1
    q = _num(data, "quantile_Opps")
    e = _num(data, "exp_Num_opps")
    if not pd.isna(q):
        d["DeltaQ"] = q - 1
    if not pd.isna(e):
        d["S_EV"] = float(poisson.cdf(0, e - 1))
    return d


def _key_asc(v):
    na = pd.isna(v)
    return (na, v if not na else 0.0)


def _key_desc(v):
    na = pd.isna(v)
    return (na, -v if not na else 0.0)


def _render_row(designation, confirmed, note, data, dagger=False):
    obj = format_designation(designation, confirmed)
    if data is None:
        cells = ["" for _ in DATA_COLUMNS]
    else:
        cells = [FORMATTERS[col](data.get(col)) for col in DATA_COLUMNS]
    if dagger:
        nopp_idx = DATA_COLUMNS.index("Num_opps")
        cells[nopp_idx] = (cells[nopp_idx] or "") + _DAGGER
    cells = [c if c else "  " for c in cells]
    return (
        f"\\objectcell{{{obj}}} &\n"
        + " & ".join(cells)
        + f" &\n{note} \\\\"
    )


def build_table(df, notes_csv, warn=True, multi_opp_threshold=1):
    """Build the full ``longrotatetable`` LaTeX block.

    ``df`` is the combined DataFrame (see :func:`combine_sources`) with an
    ``Object`` column.  ``notes_csv`` supplies group/confirmed/note per object.

    The cometary group is split into multi-opposition (``Num_opps >
    multi_opp_threshold``) and single-opposition sections.  Sorting:
    multi-opp cometary by ``S_EV`` ascending; single-opp cometary and asteroidal
    by ``prob`` descending.  Missing/NA sort keys go to the bottom of the section.
    """
    df = _ensure_derived(df)
    df_by_object = df.drop_duplicates(subset="Object").set_index("Object")

    notes = load_notes(notes_csv)
    missing = []

    multi, single, asteroid = [], [], []
    for r in notes:
        data = _lookup(df_by_object, r["designation"])
        if data is None:
            missing.append(r["designation"])
        if r["nopp_dagger"]:
            data = _as_single_opp(data)  # display single-opposition model values
        entry = (r, data)
        if r["group"] == "asteroidal":
            asteroid.append(entry)
        elif r["group"] == "cometary":
            nopp = _num(data, "Num_opps")
            if not pd.isna(nopp) and nopp > multi_opp_threshold:
                multi.append(entry)
            else:
                single.append(entry)
        else:
            if warn:
                print(f"WARNING: unknown group {r['group']!r} for {r['designation']!r}; "
                      "treating as asteroidal")
            asteroid.append(entry)

    multi.sort(key=lambda e: _key_asc(_num(e[1], "S_EV")))
    single.sort(key=lambda e: _key_desc(_num(e[1], "prob")))
    asteroid.sort(key=lambda e: _key_desc(_num(e[1], "prob")))

    # (label, entries, sort caption, force a page break before this section)
    sections = [
        (SECTION_MULTI_COMET, multi, _SORT_SEV_ASC, False),
        (SECTION_SINGLE_COMET, single, _SORT_PROB_DESC, False),
        (SECTION_ASTEROID, asteroid, _SORT_PROB_DESC, True),  # hard-coded page break
    ]

    parts = [_HEADER]
    for label, entries, sortcap, newpage in sections:
        if not entries:
            continue
        if newpage:
            parts.append(r"\newpage")
        heading = f"\\textbf{{{label}}} \\textnormal{{\\footnotesize ({sortcap})}}"
        parts.append(f"\\multicolumn{{12}}{{l}}{{{heading}}}\\\\")
        parts.append(r"\midrule")
        parts.append("")
        for r, data in entries:
            parts.append(_render_row(r["designation"], r["confirmed"], r["note"], data,
                                     dagger=r["nopp_dagger"]))
            parts.append("")
        parts.append(r"\midrule")

    # drop the trailing \midrule after the last section; replace with footer
    if parts and parts[-1] == r"\midrule":
        parts.pop()
    parts.append(_FOOTER)

    if warn and missing:
        print("WARNING: no DataFrame match (numbers left blank) for:")
        for m in missing:
            print(f"  - {m}")

    return "\n".join(parts)
