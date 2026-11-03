"""Candidate forecasts produced by rolling-origin backtesting.

Every method sees only what was known at the origin: actuals for months
<= origin, customer forecasts and order book made at or before the origin,
and trailing ratios over targets already realised. Output is one long frame
(origin_month, target_month, horizon, customer, family, method, forecast,
lo80, hi80) that metrics.py consumes, plus the blend's choice per family.
"""

import time
import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import TSB, AutoETS
from statsforecast.utils import ConformalIntervals

from fgov import breaks, metrics

KEYS = ["customer", "family"]
COLS = ["origin_month", "target_month", "horizon", "customer", "family",
        "method", "forecast", "lo80", "hi80"]
CANDIDATES = ["stat_baseline", "customer_bias_corrected", "order_book"]
INTERMITTENT_ZERO_SHARE = 0.40   # zero share above this -> TSB
MIN_MODEL_POINTS = 13            # fewer points -> mean of available history
CONFORMAL_WINDOWS = 2            # conformal needs CONFORMAL_WINDOWS * 6 extra points
MAX_H = 6


def _finish(df, method):
    """Common tidy-up: method label, missing interval columns, non-negative values."""
    out = df.assign(method=method)
    for col in ("lo80", "hi80"):
        if col not in out:
            out[col] = np.nan
    out[["forecast", "lo80", "hi80"]] = out[["forecast", "lo80", "hi80"]].astype(float).clip(lower=0)
    return out[COLS].reset_index(drop=True)


def _grid(origins, horizons, last_month):
    rows = [(o, o + h, h) for o in origins for h in horizons if o + h <= last_month]
    return pd.DataFrame(rows, columns=["origin_month", "target_month", "horizon"])


def seasonal_naive(actuals, origins, horizons, last_month=60):
    """Actual at target - 12; if that is NaN, the last actual at or before the origin."""
    act = actuals[KEYS + ["month", "actual"]].sort_values(KEYS + ["month"])
    act["last_seen"] = act.groupby(KEYS)["actual"].ffill()
    g = _grid(origins, horizons, last_month).merge(act[KEYS].drop_duplicates(), how="cross")
    g["lag_month"] = g["target_month"] - 12
    lag = act[KEYS + ["month", "actual"]].rename(columns={"month": "lag_month", "actual": "lag12"})
    last = act[KEYS + ["month", "last_seen"]].rename(columns={"month": "origin_month"})
    g = g.merge(lag, on=KEYS + ["lag_month"], how="left").merge(
        last, on=KEYS + ["origin_month"], how="left")
    g["forecast"] = g["lag12"].fillna(g["last_seen"])
    return _finish(g[g["forecast"].notna()], "seasonal_naive")


def _sf_forecast(model, df, conformal, n_jobs):
    """One statsforecast call for one model group; returns unique_id, ds, forecast, lo80, hi80."""
    name = model.alias
    sf = StatsForecast(models=[model], freq=1, n_jobs=n_jobs)
    if conformal:
        out = sf.forecast(h=MAX_H, df=df, level=[80],
                          prediction_intervals=ConformalIntervals(n_windows=CONFORMAL_WINDOWS, h=MAX_H))
    else:
        out = sf.forecast(h=MAX_H, df=df)
    assert isinstance(out, pd.DataFrame)    # pandas in, pandas out
    return out.rename(columns={name: "forecast", f"{name}-lo-80": "lo80", f"{name}-hi-80": "hi80"})

def _stat_one_origin(actuals, origin, n_jobs):
    """All series at one origin: TSB if intermittent, AutoETS otherwise, mean if short."""
    d = actuals[(actuals["month"] <= origin) & actuals["actual"].notna()]
    df = pd.DataFrame({"unique_id": d["customer"] + "|" + d["family"],
                       "ds": d["month"], "y": d["actual"].astype(float)})
    info = df.groupby("unique_id")["y"].agg(n="size", zero_share=lambda y: (y == 0).mean(), mean="mean")
    long_enough = info["n"] >= MIN_MODEL_POINTS + CONFORMAL_WINDOWS * MAX_H
    parts = []
    short = info[info["n"] < MIN_MODEL_POINTS]
    if len(short):
        parts.append(pd.DataFrame({"unique_id": np.repeat(short.index, MAX_H),
                                   "ds": np.tile(origin + np.arange(1, MAX_H + 1), len(short)),
                                   "forecast": np.repeat(short["mean"].to_numpy(), MAX_H)}))
    modelled = info["n"] >= MIN_MODEL_POINTS
    intermittent = info["zero_share"] > INTERMITTENT_ZERO_SHARE
    groups = [(lambda: TSB(alpha_d=0.2, alpha_p=0.2), modelled & intermittent),
              (lambda: AutoETS(season_length=12), modelled & ~intermittent)]
    for make, mask in groups:
        for conformal in (True, False):
            ids = info.index[mask & (long_enough == conformal)]
            if len(ids):
                parts.append(_sf_forecast(make(), df[df["unique_id"].isin(ids)], conformal, n_jobs))
    out = pd.concat(parts, ignore_index=True)
    out[KEYS] = out["unique_id"].str.split("|", expand=True)
    return out.assign(origin_month=origin, target_month=out["ds"].astype(int),
                      horizon=out["ds"].astype(int) - origin)


def stat_baseline(actuals, origins, horizons, last_month=60, n_jobs=4):
    """History-only statistical baseline, fitted per origin on every series at once."""
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ETS candidate fits with zero dof
        for origin in origins:
            parts.append(_stat_one_origin(actuals, origin, n_jobs))
    out = pd.concat(parts, ignore_index=True)
    out = out[out["horizon"].isin(horizons) & (out["target_month"] <= last_month)]
    return _finish(out, "stat_baseline")


def customer_raw(customer_forecasts, origins, horizons, last_month=60):
    """The customer's own number for the target, made exactly `horizon` months earlier."""
    cf = customer_forecasts
    d = cf[cf["origin_month"].isin(list(origins)) & cf["horizon"].isin(horizons)
           & (cf["target_month"] <= last_month) & cf["customer_forecast"].notna()]
    return _finish(d.rename(columns={"customer_forecast": "forecast"}), "customer_raw")


def _trailing_ratios(df, actuals, origins, horizons, value_col):
    """Per origin, horizon and customer: metrics.customer_bias_trailing (window 12)."""
    parts = [metrics.customer_bias_trailing(df, actuals, o, h, window=12, value_col=value_col)
             .assign(origin_month=o, horizon=h) for o in origins for h in horizons]
    return pd.concat(parts, ignore_index=True)


def customer_bias_corrected(customer_forecasts, actuals, origins, horizons, last_month=60):
    """Raw customer forecast divided by that customer's trailing bias ratio at the horizon."""
    raw = customer_raw(customer_forecasts, origins, horizons, last_month)
    ratio = _trailing_ratios(customer_forecasts, actuals, origins, horizons, "customer_forecast")
    d = raw.merge(ratio, on=["origin_month", "horizon", "customer"], how="left")
    d["forecast"] = d["forecast"] / d["bias_ratio"].fillna(1.0)
    return _finish(d.drop(columns=["lo80", "hi80"]), "customer_bias_corrected")


def order_book(book, actuals, origins, horizons, last_month=60):
    """Confirmed quantity as of the origin divided by the trailing fill ratio."""
    b = book.assign(horizon=book["target_month"] - book["as_of_month"])
    d = b[b["as_of_month"].isin(list(origins)) & b["horizon"].isin(horizons)
          & (b["target_month"] <= last_month) & b["confirmed_qty"].notna()]
    d = d.rename(columns={"as_of_month": "origin_month"})
    ratio = _trailing_ratios(b, actuals, origins, horizons, "confirmed_qty")
    d = d.merge(ratio, on=["origin_month", "horizon", "customer"], how="left")
    ratio_used = d["bias_ratio"].where(d["bias_ratio"] > 0).fillna(1.0)
    d["forecast"] = d["confirmed_qty"] / ratio_used
    return _finish(d, "order_book")


def series_level(actuals, window=12):
    """Per series and origin month: mean actual over realised months in
    (origin - window, origin]; falls back to the mean over all months <= origin
    when that window is empty or zero. NaN when no positive history exists."""
    wide = actuals.pivot(index="month", columns=KEYS, values="actual").sort_index()
    recent = wide.rolling(window, min_periods=1).mean()
    level = recent.where(recent > 0, wide.expanding(min_periods=1).mean())
    level = level.where(level > 0)
    return level.stack(KEYS, future_stack=True).rename("level").reset_index().rename(
        columns={"month": "origin_month"})


def empirical_interval(forecast_rows, actuals, origin=None, window=12):
    """80% interval from past relative residuals of the same method, horizon and family.

    A residual (actual - forecast) from origin o' with target t' is usable at
    origin o when t' <= o (realised) and o - window < o' <= o. Each residual is
    divided by its series' level at o (see series_level, which uses months
    <= o only), pooled over customers, and the 10th/90th percentiles are
    multiplied back by the row's own series level at o. The interval is widened if needed to
    contain the point forecast and clipped at zero. Fewer than 5 residuals, or
    no level, -> NaN. Runs once over the frame; `origin` only restricts which
    rows are returned.
    """
    act = actuals[KEYS + ["month", "actual"]].rename(columns={"month": "target_month"})
    level = series_level(actuals, window)
    on_level = KEYS + ["origin_month"]
    r = forecast_rows.merge(act, on=KEYS + ["target_month"]).dropna(subset=["actual", "forecast"])
    r = r.assign(resid=r["actual"] - r["forecast"]).reset_index(drop=True)
    r = r.loc[r.index.repeat(window)].assign(use_origin=np.tile(np.arange(window), len(r)))
    r["use_origin"] += r["origin_month"]
    r = r[r["use_origin"] >= r["target_month"]]
    r = r.merge(level.rename(columns={"origin_month": "use_origin"}), on=KEYS + ["use_origin"])
    r = r.assign(resid=r["resid"] / r["level"]).dropna(subset=["resid"])
    g = r.groupby(["method", "family", "horizon", "use_origin"])["resid"]
    q = pd.DataFrame({"q10": g.quantile(0.1), "q90": g.quantile(0.9), "n": g.size()})
    q = q[q["n"] >= 5].reset_index().rename(columns={"use_origin": "origin_month"})
    rows = forecast_rows if origin is None else forecast_rows[forecast_rows["origin_month"] == origin]
    m = rows[["method", "family", "horizon", "origin_month", "customer", "forecast"]].merge(
        q, on=["method", "family", "horizon", "origin_month"], how="left").merge(
        level, on=on_level, how="left")
    lo = (m["forecast"] + (m["q10"] * m["level"]).clip(upper=0)).clip(lower=0)
    hi = (m["forecast"] + (m["q90"] * m["level"]).clip(lower=0)).clip(lower=0)
    return pd.DataFrame({"lo80": lo.to_numpy(), "hi80": hi.to_numpy()}, index=rows.index)


def fill_intervals(rows, actuals, window=12):
    """Keep model intervals where present; fill the rest with empirical ones."""
    emp = empirical_interval(rows, actuals, window=window)
    missing = rows["lo80"].isna() | rows["hi80"].isna()
    out = rows.copy()
    out.loc[missing, ["lo80", "hi80"]] = emp.loc[missing, ["lo80", "hi80"]]
    return out


def average_candidate(candidate_rows):
    """Simple mean of the three candidates' point forecasts, row by row (method 'average')."""
    keys = ["origin_month", "target_month", "horizon"] + KEYS
    c = candidate_rows[candidate_rows["method"].isin(CANDIDATES)]
    avg = c.groupby(keys, as_index=False)["forecast"].mean()
    return _finish(avg, "average")


def blend(candidate_rows, actuals, alerts, origins, recent=12, unreliable_recent=6, min_origins=3):
    """Per origin and family, pick the candidate with lowest trailing pooled MASE.

    `candidate_rows` holds the three candidates plus 'average', with intervals,
    for origins reaching back at least `recent` before the first blend origin.
    The trailing window is the previous 12 origins (6 when history is flagged
    unreliable), using only rows whose target was realised by the origin.
    """
    ev = metrics.evaluate(candidate_rows, actuals)
    ev = ev[ev["abs_scaled_error"].notna()]
    families = sorted(actuals["family"].unique())
    choices = []
    for o in origins:
        past = ev[(ev["origin_month"] >= o - recent) & (ev["origin_month"] < o) & (ev["target_month"] <= o)]
        for fam in families:
            window = unreliable_recent if breaks.history_unreliable(alerts, fam, o) else recent
            p = past[(past["family"] == fam) & (past["origin_month"] >= o - window)]
            chosen = "average"
            if p["origin_month"].nunique() >= min_origins:
                scores = p.groupby("method")["abs_scaled_error"].mean()
                chosen = scores.reindex(CANDIDATES + ["average"]).idxmin()
            choices.append((o, fam, chosen))
    choices = pd.DataFrame(choices, columns=["origin_month", "family", "chosen"])
    picked = candidate_rows.merge(choices, left_on=["origin_month", "family", "method"],
                                  right_on=["origin_month", "family", "chosen"])
    return _finish(picked, "blend"), choices


def run_all(frames, alerts, origins=range(25, 61), horizons=(1, 3, 6), warmup=12, n_jobs=4,
            timings=None):
    """All six methods for `origins`. Methods are also run for `warmup` origins
    before the first, only so empirical intervals and the blend have a trailing
    history at the first origin; those rows are dropped from the output.

    Pass a dict as `timings` to receive wall seconds per method.
    """
    actuals = frames["actuals"]
    last = int(actuals["month"].max())
    origins = list(origins)
    run_origins = list(range(max(1, origins[0] - warmup), origins[-1] + 1))
    timings = {} if timings is None else timings
    calls = {
        "seasonal_naive": lambda: seasonal_naive(actuals, run_origins, horizons, last),
        "stat_baseline": lambda: stat_baseline(actuals, run_origins, horizons, last, n_jobs),
        "customer_raw": lambda: customer_raw(frames["customer_forecasts"], run_origins, horizons, last),
        "customer_bias_corrected": lambda: customer_bias_corrected(
            frames["customer_forecasts"], actuals, run_origins, horizons, last),
        "order_book": lambda: order_book(frames["order_book"], actuals, run_origins, horizons, last),
    }
    parts = []
    for name, call in calls.items():
        t0 = time.perf_counter()
        parts.append(call())
        timings[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = pd.concat(parts, ignore_index=True)
    rows = pd.concat([rows, average_candidate(rows)], ignore_index=True)
    rows = fill_intervals(rows, actuals)
    timings["intervals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    blended, choices = blend(rows[rows["method"].isin(CANDIDATES + ["average"])], actuals, alerts, origins)
    timings["blend"] = time.perf_counter() - t0
    out = pd.concat([rows[rows["method"] != "average"], blended], ignore_index=True)
    out = out[out["origin_month"].isin(origins)]
    out = out.sort_values(["method", "origin_month", "horizon"] + KEYS).reset_index(drop=True)
    return out, choices
