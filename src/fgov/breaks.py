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
EARLY_ALLOWANCE = 1           # long-lead families step one month early
NEW_FAMILY_HISTORY = 12       # months of history before a new family is trusted
DEFAULT_ORIGINS = range(24, 61)
PENALTY_GRID = tuple(np.round(np.arange(0.5, 10.01, 0.25), 2))
DEFAULT_PENALTY = 2.0         # choose_penalty() on the default synth panel; see its docstring


def detect_at_origin(series: pd.Series, origin: int, penalty: float, min_size: int = 3) -> int | None:
    """Month of the last change point in `series` (index = month) up to `origin`.

    Returns the start month of the newest regime, or None if PELT finds none.
    """
    s = series[series.index <= origin].dropna().sort_index()
    if len(s) < 2 * min_size or s.mean() <= 0:
        return None
    x = (s.to_numpy(dtype=float) / s.mean()).reshape(-1, 1)
    bkps = rpt.Pelt(model="rbf", min_size=min_size).fit(x).predict(pen=penalty)
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


def first_detection(scanned: pd.DataFrame, onset: int = ONSET) -> pd.Series:
    """Per name, the first origin whose last break month is >= onset - 1 (NaN if never)."""
    hit = scanned[scanned.break_month.fillna(0) >= onset - EARLY_ALLOWANCE]
    out = hit.groupby("name")["origin"].min()
    return out.reindex(scanned.name.unique()).astype(float)


def alert_table(actuals: pd.DataFrame, onset: int = ONSET, origins=None,
                penalty: float = DEFAULT_PENALTY) -> pd.DataFrame:
    """One row per family plus TOTAL.

    pre_break_level: mean monthly total over the 12 months before onset.
    post_break_level: mean monthly total from the detected break month to the
    last origin scanned (NaN if never detected). New families are flagged by
    rule at launch (reason "new_family"); their post level runs from launch.
    """
    origins = list(DEFAULT_ORIGINS if origins is None else origins)
    new = set(actuals.loc[actuals.is_new, "family"])
    est = actuals[~actuals.family.isin(new)]
    scanned = pd.concat([scan(actuals, origins, penalty, "total"),
                         scan(est, origins, penalty, "family")], ignore_index=True)
    first = first_detection(scanned, onset)
    series = {**monthly_totals(actuals, "total"), **monthly_totals(actuals, "family")}
    rows = []
    for name, full in series.items():
        s = full[full.index <= origins[-1]]
        pre = s[(s.index >= onset - 12) & (s.index < onset)].mean()
        if name in new:
            launch = int(s.dropna().index.min())
            at, reason, start, pre = float(launch), "new_family", launch, np.nan
        else:
            at = first.get(name, np.nan)
            reason = "detected" if pd.notna(at) else np.nan
            bm = scanned[(scanned.name == name) & (scanned.origin == at)].break_month
            start = int(bm.iloc[0]) if pd.notna(at) else None
        post = s[s.index >= start].mean() if start is not None else np.nan
        rows.append({"family": name, "break_detected_at_origin": at,
                     "months_after_onset": at - onset if pd.notna(at) else np.nan,
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
