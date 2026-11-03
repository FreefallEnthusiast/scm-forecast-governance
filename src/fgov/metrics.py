"""Forecast metrics: MASE, bias, FVA, stability, interval coverage, trailing bias.

Plain pandas functions. `forecasts` is the long frame produced by forecast.py
(origin_month, target_month, horizon, customer, family, method, forecast,
lo80, hi80); `actuals` is the frame from synth.generate().
"""

import numpy as np
import pandas as pd

KEYS = ["customer", "family"]
METHODS_REF = {"naive": "seasonal_naive", "baseline": "stat_baseline"}


def mase_scale(actuals, origin, season=12):
    """In-sample seasonal-naive MAE per series over months <= origin."""
    wide = actuals[actuals["month"] <= origin].pivot(
        index="month", columns=KEYS, values="actual").reindex(range(1, origin + 1))
    seasonal = (wide - wide.shift(season)).abs().mean()
    lag1 = (wide - wide.shift(1)).abs().mean()
    enough = (wide.notna().sum() >= season + 1) & seasonal.notna()
    scale = seasonal.where(enough, lag1)
    scale = scale.where(scale > 0)          # constant series: scale undefined
    return scale.rename("scale").reset_index()


def evaluate(forecasts, actuals, break_month=37):
    """Row-level errors: actuals joined on target, pre/post label, scaled error."""
    act = actuals[KEYS + ["month", "actual"]].rename(columns={"month": "target_month"})
    ev = forecasts.merge(act, on=KEYS + ["target_month"], how="inner")
    ev = ev[ev["actual"].notna()].copy()
    ev["period"] = np.where(ev["origin_month"] < break_month, "pre", "post")
    scales = pd.concat([mase_scale(actuals, o).assign(origin_month=o)
                        for o in sorted(ev["origin_month"].unique())])
    ev = ev.merge(scales, on=KEYS + ["origin_month"], how="left")
    ev["error"] = ev["forecast"] - ev["actual"]
    ev["abs_scaled_error"] = ev["error"].abs() / ev["scale"]
    has_interval = ev["lo80"].notna() & ev["hi80"].notna()
    inside = (ev["lo80"] <= ev["actual"]) & (ev["actual"] <= ev["hi80"])
    ev["inside80"] = inside.astype(float).where(has_interval)
    return ev.reset_index(drop=True)


def summarise(evaluated, by=()):
    """MASE, signed bias, 80% coverage and n, pooled over all rows in each group."""
    by = list(by)
    pooled = pd.concat([evaluated, evaluated.assign(period="all")], ignore_index=True)
    grouped = pooled.groupby(["method", "horizon", "period"] + by)
    out = grouped.agg(mase=("abs_scaled_error", "mean"), mean_error=("error", "mean"),
                      mean_actual=("actual", "mean"), coverage80=("inside80", "mean"),
                      n=("actual", "size")).reset_index()
    out["bias"] = out["mean_error"] / out["mean_actual"].where(out["mean_actual"] > 0)
    return out.drop(columns=["mean_error", "mean_actual"])[
        ["method", "horizon", "period"] + by + ["mase", "bias", "coverage80", "n"]]


def fva(summary):
    """FVA in percentage points versus seasonal naive and versus the stat baseline."""
    group_cols = [c for c in summary.columns if c not in
                  ("method", "mase", "bias", "coverage80", "n")]
    out = summary.copy()
    for short, ref in METHODS_REF.items():
        ref_mase = (summary[summary["method"] == ref][group_cols + ["mase"]]
                    .rename(columns={"mase": f"mase_{short}"}))
        out = out.merge(ref_mase, on=group_cols, how="left")
        out[f"fva_vs_{short}"] = 100 * (out[f"mase_{short}"] - out["mase"]) / out[f"mase_{short}"]
        out = out.drop(columns=f"mase_{short}")
    return out


def stability(forecasts, actuals, break_month=37):
    """Mean absolute revision between consecutive origins, over series mean actual.

    Revisions are taken per (customer, family, target_month, method) between
    consecutive available origins; the period is set by the later origin. The
    divisor is the series' mean actual over all realised months in `actuals`.
    """
    fc = forecasts.sort_values("origin_month")
    fc = fc.assign(revision=fc.groupby(KEYS + ["target_month", "method"])["forecast"].diff().abs())
    fc = fc[fc["revision"].notna()].copy()
    level = actuals.groupby(KEYS)["actual"].mean().rename("series_mean").reset_index()
    fc = fc.merge(level, on=KEYS, how="left")
    fc["scaled_revision"] = fc["revision"] / fc["series_mean"].where(fc["series_mean"] > 0)
    fc["period"] = np.where(fc["origin_month"] < break_month, "pre", "post")
    pooled = pd.concat([fc, fc.assign(period="all")], ignore_index=True)
    return (pooled.groupby(["method", "period"])["scaled_revision"].mean()
            .rename("stability").reset_index())


def customer_bias_trailing(df, actuals, origin, horizon, window=12,
                           value_col="customer_forecast", by_family=False):
    """Sum of forecasts / sum of actuals over the last `window` realised target months.

    Uses only targets in (origin - window, origin] at the given horizon, so
    everything was realised by the origin. Returns bias_ratio per customer
    (and family when by_family=True).
    """
    keys = KEYS if by_family else ["customer"]
    d = df[(df["horizon"] == horizon) & (df["target_month"] <= origin)
           & (df["target_month"] > origin - window)]
    act = actuals[KEYS + ["month", "actual"]].rename(columns={"month": "target_month"})
    d = d.merge(act, on=KEYS + ["target_month"])
    d = d[d["actual"].notna() & d[value_col].notna()]
    sums = d.groupby(keys)[[value_col, "actual"]].sum()
    ratio = sums[value_col] / sums["actual"].where(sums["actual"] > 0)
    return ratio.rename("bias_ratio").reset_index()
