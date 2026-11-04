"""Change-point detection on monthly demand, run honestly at each origin.

At every origin only months <= origin are used, so the reported detection
origin is what a planner would have seen at the time, not hindsight.

PELT with an RBF kernel (ruptures). Each window is divided by its own mean
before fitting so one penalty reads the same across families of different
size. (The RBF kernel's bandwidth is set by the median heuristic on the
window, which is already scale-free; the division keeps that explicit and
keeps the fit independent of units.)
"""

import numpy as np
import pandas as pd
import ruptures as rpt

ONSET = 37                    # step onset month (synth default)
LONG_LEAD_OFFSET = 1          # long-lead families step this many months before the onset
EARLY_ALLOWANCE = 2           # a break placed up to this many months before a family's onset counts (the ramp)
NEW_FAMILY_HISTORY = 12       # months of history before a new family is trusted
DEFAULT_ORIGINS = range(24, 61)
PENALTY_GRID = tuple(np.round(np.arange(0.5, 10.01, 0.25), 2))
DEFAULT_PENALTY = 2.25        # choose_penalty() on the default synth panel; see its docstring


def detect_at_origin(series: pd.Series, origin: int, penalty: float, min_size: int = 3) -> int | None:
    """Month of the last change point in `series` (index = month) up to `origin`.

    Returns the start month of the newest regime, or None if PELT finds none.
    """
    s = series[series.index <= origin].dropna().sort_index()
    if len(s) < 2 * min_size or s.mean() <= 0:
        return None
    x = (s.to_numpy(dtype=float) / s.mean()).reshape(-1, 1)
    # jump=1: consider every month. The library default (5) only places breaks on a 5-month grid.
    bkps = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(x).predict(pen=penalty)
    bkps = [b for b in bkps if b < len(s)]          # last entry is the series end
    return int(s.index[bkps[-1]]) if bkps else None


def monthly_totals(actuals: pd.DataFrame, level: str = "family") -> dict:
    """{name: monthly series}. Total treats NaN as 0; families keep NaN pre-launch."""
    if level == "total":
        return {"TOTAL": actuals.groupby("month")["actual"].sum(min_count=0)}
    g = actuals.groupby(["family", "month"])["actual"].sum(min_count=1)
    return {f: g.loc[f] for f in g.index.get_level_values(0).unique()}


def scan(actuals: pd.DataFrame, origins=None, penalty: float = DEFAULT_PENALTY,
         level: str = "family") -> pd.DataFrame:
    """Detected last break month per series and origin: columns name, origin, break_month."""
    origins = DEFAULT_ORIGINS if origins is None else origins
    rows = [(name, o, detect_at_origin(s, o, penalty))
            for name, s in monthly_totals(actuals, level).items() for o in origins]
    df = pd.DataFrame(rows, columns=["name", "origin", "break_month"])
    df["break_month"] = df["break_month"].astype("Int64")
    return df


def false_alarms(scanned: pd.DataFrame, onset: int = ONSET, recent: int = 20) -> pd.Series:
    """Per name, count of pre-onset origins whose last break is recent (>= `recent`)."""
    pre = scanned[scanned.origin < onset]
    return pre.groupby("name")["break_month"].apply(lambda b: int((b.fillna(0) >= recent).sum()))


def choose_penalty(actuals: pd.DataFrame, grid=PENALTY_GRID, onset: int = ONSET) -> float:
    """Smallest penalty with zero pre-onset false alarms on the total and on
    at least 5 of the 6 smooth families, scanning origins 24..onset-1 only.

    Uses no data after the step, so the penalty is not tuned on detection.
    """
    origins = range(24, onset)
    smooth = actuals.loc[~actuals.is_intermittent & ~actuals.is_new]
    for pen in grid:
        if false_alarms(scan(actuals, origins, pen, "total")).sum() > 0:
            continue
        fa = false_alarms(scan(smooth, origins, pen, "family"))
        if (fa == 0).sum() >= len(fa) - 1:
            return float(pen)
    return float(grid[-1])


def family_onsets(actuals: pd.DataFrame, onset: int = ONSET) -> dict:
    """Onset month per series name: long-lead families step LONG_LEAD_OFFSET months early."""
    ll = actuals.drop_duplicates("family").set_index("family")["is_long_lead"]
    return {"TOTAL": onset, **{f: onset - LONG_LEAD_OFFSET * int(v) for f, v in ll.items()}}


def first_detection(scanned: pd.DataFrame, onsets: dict) -> pd.Series:
    """Per name, the first origin whose last break month is within EARLY_ALLOWANCE
    months before that series' onset, or later (NaN if never)."""
    threshold = scanned.name.map(onsets) - EARLY_ALLOWANCE
    hit = scanned[scanned.break_month.fillna(0) >= threshold]
    out = hit.groupby("name")["origin"].min()
    return out.reindex(scanned.name.unique()).astype(float)


def alert_table(actuals: pd.DataFrame, onset: int = ONSET, origins=None,
                penalty: float = DEFAULT_PENALTY) -> pd.DataFrame:
    """One row per family plus TOTAL.

    pre_break_level: mean monthly total over the 12 months before the series' own
    onset (long-lead families step LONG_LEAD_OFFSET months early).
    post_break_level: mean monthly total from break_month (PELT break month at
    the detection origin) to the last origin scanned; both NaN if never detected.
    months_after_onset is measured from the series' own onset.
    New families: reason "new_family", break month = launch (NaN if not launched yet).
    """
    origins = list(DEFAULT_ORIGINS if origins is None else origins)
    new = set(actuals.loc[actuals.is_new, "family"])
    est = actuals[~actuals.family.isin(new)]
    scanned = pd.concat([scan(actuals, origins, penalty, "total"),
                         scan(est, origins, penalty, "family")], ignore_index=True)
    onsets = family_onsets(actuals, onset)
    first = first_detection(scanned, onsets)
    series = {**monthly_totals(actuals, "total"), **monthly_totals(actuals, "family")}
    rows = []
    for name, full in series.items():
        s = full[full.index <= origins[-1]]
        own = onsets[name]
        pre = s[(s.index >= own - 12) & (s.index < own)].mean()
        if name in new:
            at = start = float(s.dropna().index.min()) if s.notna().any() else np.nan
            reason, pre = "new_family", np.nan
        else:
            at = first.get(name, np.nan)
            reason = "detected" if pd.notna(at) else np.nan
            bm = scanned[(scanned.name == name) & (scanned.origin == at)].break_month
            start = float(bm.iloc[0]) if pd.notna(at) else np.nan
        post = s[s.index >= start].mean() if pd.notna(start) else np.nan
        rows.append({"family": name, "break_detected_at_origin": at, "break_month": start,
                     "months_after_onset": at - own if pd.notna(at) else np.nan,
                     "pre_break_level": pre, "post_break_level": post, "reason": reason})
    return pd.DataFrame(rows)


def history_unreliable(alerts: pd.DataFrame, family: str, origin: int) -> bool:
    """True once a break is detected at an origin <= `origin`; never auto-clears.

    New families are unreliable from launch until they have NEW_FAMILY_HISTORY
    months of history.
    """
    row = alerts[alerts.family == family]
    if row.empty or pd.isna(row.break_detected_at_origin.iloc[0]):
        return False
    at = row.break_detected_at_origin.iloc[0]
    if row.reason.iloc[0] == "new_family":
        return bool(at <= origin < at + NEW_FAMILY_HISTORY)
    return bool(at <= origin)
