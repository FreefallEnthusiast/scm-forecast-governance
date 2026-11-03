"""Governance report: tidy tables, five charts, the planner table and one markdown page.

Every number in the report text is computed here from the tables; nothing is
hardcoded. Charts are static PNGs on a light surface with one fixed colour per
method, so a method looks the same in every chart.
"""

from pathlib import Path

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fgov import breaks, metrics  # noqa: E402

ONSET = breaks.ONSET
HISTORY_END = 48
TRACK_CUSTOMER = "C5"
TRACK_HORIZON = 3
LABEL = {"seasonal_naive": "Seasonal naive", "stat_baseline": "Stat baseline",
         "customer_raw": "Customer raw", "customer_bias_corrected": "Customer bias-corrected",
         "order_book": "Order book", "blend": "Blend", "actual": "Actual"}
COLOR = {"seasonal_naive": "#898781", "stat_baseline": "#2a78d6", "customer_raw": "#eb6834",
         "customer_bias_corrected": "#1baf7a", "order_book": "#eda100", "blend": "#e87ba4",
         "actual": "#0b0b0b"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PERIOD_COLOR = {"pre": "#c3c2b7", "post": "#52514e"}
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
      "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
      "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
      "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
      "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
      "savefig.dpi": 150}
CHARTS = ("total_demand_and_break", "customer_bias_by_horizon", "fva_by_method",
          "tracking_c5", "stability_vs_accuracy")


def detected_break_month(actuals, name, origin):
    """Break month PELT reported at the detection origin (alert table has only the origin)."""
    if pd.isna(origin):
        return np.nan
    series = {**breaks.monthly_totals(actuals, "total"), **breaks.monthly_totals(actuals)}[name]
    month = breaks.detect_at_origin(series, int(origin), breaks.DEFAULT_PENALTY)
    return np.nan if month is None else month


def planner_table(frames, forecasts, choices, alerts):
    """One row per customer x family for the next planning month.

    The nominal planning origin is the last origin (60) with target 61, but
    forecasts stop at target 60. So the planning origin is the latest origin
    with horizon-1 blend rows (59 in the full run) and the target is origin + 1.
    """
    actuals = frames["actuals"]
    h1 = forecasts[(forecasts.method == "blend") & (forecasts.horizon == 1)]
    origin = int(h1.origin_month.max())
    grid = actuals[["customer", "family"]].drop_duplicates()
    blend = h1[h1.origin_month == origin][["customer", "family", "forecast", "lo80", "hi80"]]
    out = grid.merge(blend, on=["customer", "family"], how="left")
    chosen = choices.set_index(["origin_month", "family"])["chosen"]
    out["recommended_source"] = [chosen.get((origin, f), np.nan) for f in out.family]
    previous = [chosen.get((origin - 1, f), np.nan) for f in out.family]
    bias = metrics.customer_bias_trailing(frames["customer_forecasts"], actuals, origin, 1)
    out["customer_bias_trailing"] = out.customer.map(bias.set_index("customer").bias_ratio)
    out["history_unreliable"] = [breaks.history_unreliable(alerts, f, origin) for f in out.family]
    detected_now = set(alerts.loc[alerts.break_detected_at_origin == origin, "family"])
    reasons = []
    for row, prev in zip(out.itertuples(), previous, strict=True):
        r = []
        if abs(row.customer_bias_trailing - 1) > 0.20:
            r.append(f"customer bias {100 * (row.customer_bias_trailing - 1):+.0f}%")
        if pd.notna(prev) and prev != row.recommended_source:
            r.append(f"source changed from {prev}")
        if row.family in detected_now:
            r.append("break detected this origin")
        reasons.append("; ".join(r))
    out["attention_flag"] = [bool(r) for r in reasons]
    out["attention_reason"] = reasons
    out["customer_bias_trailing"] = out.customer_bias_trailing.round(3)
    out[["forecast", "lo80", "hi80"]] = out[["forecast", "lo80", "hi80"]].round(1)
    out.insert(0, "target_month", origin + 1)
    out.insert(0, "origin_month", origin)
    return out.sort_values(["attention_flag", "customer", "family"],
                           ascending=[False, True, True]).reset_index(drop=True)


def build_tables(frames, forecasts, choices, alerts):
    actuals = frames["actuals"]
    evaluated = metrics.evaluate(forecasts, actuals)
    fams = frames["families"]
    long_lead = fams[fams.is_long_lead]                 # F01-F04 in the default panel
    track_family = long_lead.loc[long_lead.step_factor.idxmax(), "family"]
    total = alerts[alerts.family == "TOTAL"].iloc[0]
    fam_alert = alerts[alerts.family == track_family].iloc[0]
    context = pd.DataFrame([{
        "total_detected_at": total.break_detected_at_origin,
        "total_months_after_onset": total.months_after_onset,
        "total_break_month": detected_break_month(actuals, "TOTAL", total.break_detected_at_origin),
        "total_pre_level": total.pre_break_level, "total_post_level": total.post_break_level,
        "track_family": track_family, "track_step_factor": long_lead.step_factor.max(),
        "track_detected_at": fam_alert.break_detected_at_origin,
        "track_break_month": detected_break_month(actuals, track_family,
                                                  fam_alert.break_detected_at_origin)}])
    sel = forecasts[(forecasts.customer == TRACK_CUSTOMER) & (forecasts.family == track_family)
                    & (forecasts.horizon == TRACK_HORIZON) & forecasts.origin_month.between(25, 57)
                    & forecasts.method.isin(["stat_baseline", "customer_raw",
                                             "customer_bias_corrected"])]
    act = actuals[(actuals.customer == TRACK_CUSTOMER) & (actuals.family == track_family)
                  & (actuals.month >= 25)]
    tracking = pd.concat([
        sel.rename(columns={"method": "series", "forecast": "value"})[["target_month", "series", "value"]],
        act.assign(series="actual").rename(columns={"month": "target_month", "actual": "value"})
        [["target_month", "series", "value"]]], ignore_index=True)
    return {"summary": metrics.fva(metrics.summarise(evaluated)),
            "by_customer": metrics.summarise(evaluated, by=("customer",)),
            "by_family": metrics.summarise(evaluated, by=("family",)),
            "stability": metrics.stability(forecasts, actuals),
            "alerts": alerts,
            "planner": planner_table(frames, forecasts, choices, alerts),
            "tracking": tracking, "context": context,
            "total_demand": actuals.groupby("month")["actual"].sum(min_count=0)
            .fillna(0).rename("total").reset_index()}


def _save(fig, outdir, name):
    path = Path(outdir) / "charts" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _mark_break(ax, break_month, detected_at, top):
    """Line at the detected break month; shaded lag from onset to the detection origin."""
    if pd.isna(break_month):
        return
    ax.axvline(break_month, color=INK2, lw=1.2, label=f"Detected break (month {break_month:.0f})")
    if pd.notna(detected_at):
        ax.axvspan(ONSET, detected_at, color="#eb6834", alpha=0.14, lw=0,
                   label=f"Detection lag (onset {ONSET} to alert at origin {detected_at:.0f})")
        ax.plot([detected_at], [top], marker="v", color="#eb6834", ms=8, clip_on=False, ls="")


def chart_total(tables, outdir):
    t, c = tables["total_demand"], tables["context"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t.month, t.total / 1000, color=COLOR["stat_baseline"], lw=2, label="Total demand")
    top = ax.get_ylim()[1]
    ax.axvline(HISTORY_END + 0.5, color=AXIS, lw=0.8)
    ax.text(HISTORY_END + 1, top * 0.04, "holdout", color=INK2, fontsize=9)
    _mark_break(ax, c.total_break_month, c.total_detected_at, top)
    if pd.notna(c.total_detected_at):
        ax.annotate(f"detected {c.total_months_after_onset:.0f} months after onset",
                    xy=(c.total_detected_at + 0.3, top), xytext=(c.total_detected_at + 2, top * 0.97),
                    color=INK, fontsize=9, va="center",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.legend(loc="upper left")
    ax.set_ylim(0, top * 1.02)
    ax.set_xlim(1, t.month.max())
    ax.set_xlabel("Month")
    ax.set_ylabel("Total demand (thousand units / month)")
    ax.set_title(f"Total demand rose from {c.total_pre_level / 1000:.1f}k to "
                 f"{c.total_post_level / 1000:.1f}k units a month; "
                 f"the break was flagged {c.total_months_after_onset:.0f} months after onset")
    return _save(fig, outdir, "total_demand_and_break")


def chart_bias(tables, outdir):
    b = tables["by_customer"]
    b = b[(b.period == "all") & b.method.isin(["customer_raw", "customer_bias_corrected"])]
    horizons = sorted(b.horizon.unique())
    customers = sorted(b.customer.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 4), sharey=True, squeeze=False)
    x, w = np.arange(len(customers)), 0.38
    for ax, h in zip(axes[0], horizons, strict=True):
        ax.axhspan(-20, 20, color=GRID, alpha=0.5, lw=0)
        for k, m in enumerate(["customer_raw", "customer_bias_corrected"]):
            v = b[(b.horizon == h) & (b.method == m)].set_index("customer").bias.reindex(customers)
            ax.bar(x + (k - 0.5) * w, 100 * v, w * 0.92, color=COLOR[m], label=LABEL[m])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, customers)
        ax.set_title(f"{h}-month horizon", fontweight="normal")
        ax.set_xlabel("Customer")
    axes[0][0].set_ylabel("Bias (% of actual, + = over-forecast)")
    axes[0][0].legend(loc="upper left")
    raw = b[b.method == "customer_raw"]
    worst = raw.loc[raw.bias.abs().idxmax()]
    cor = b[b.method == "customer_bias_corrected"].set_index(["customer", "horizon"]).bias
    closer = cor.abs() <= raw.set_index(["customer", "horizon"]).bias.abs()
    fig.suptitle(f"{worst.customer} is the most biased customer forecast "
                 f"({100 * worst.bias:+.0f}% at {worst.horizon} months); bias correction moves "
                 f"{closer.sum()} of {len(closer)} customer-horizon pairs closer to zero\n"
                 f"Grey band: +/-20% attention threshold", fontweight="bold", fontsize=11)
    return _save(fig, outdir, "customer_bias_by_horizon")


def chart_fva(tables, outdir):
    s = tables["summary"]
    s = s[(s.horizon == TRACK_HORIZON) & (s.method != "seasonal_naive") & (s.period != "all")]
    methods = [m for m in LABEL if m in set(s.method)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x, w = np.arange(len(methods)), 0.38
    for k, p in enumerate(["pre", "post"]):
        v = s[s.period == p].set_index("method").fva_vs_naive.reindex(methods)
        ax.bar(x + (k - 0.5) * w, v.fillna(0), w * 0.92, color=PERIOD_COLOR[p],
               label=f"{p}-break origins" + (" (none in this run)" if v.isna().all() else ""))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [LABEL[m] for m in methods])
    ax.set_ylabel("FVA vs seasonal naive (percentage points)")
    ax.set_xlabel("Forecast method")
    ax.legend(loc="best")
    post = s[s.period == "post"].set_index("method").fva_vs_naive
    best = post.idxmax() if post.notna().any() else methods[0]
    ax.set_title(f"After the break {LABEL[best].lower()} adds the most value over seasonal "
                 f"naive ({post.get(best, np.nan):+.0f} pp, horizon {TRACK_HORIZON} months)")
    return _save(fig, outdir, "fva_by_method")


def tracking_bias(tracking, series):
    """Sum of forecasts over sum of actuals minus one, for target months from the onset."""
    post = tracking[tracking.target_month >= ONSET]
    d = post[post.series == series].set_index("target_month").value
    a = post[post.series == "actual"].set_index("target_month").value.reindex(d.index)
    return d.sum() / a.sum() - 1 if a.sum() > 0 else np.nan


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:+.0f}%"


def chart_tracking(tables, outdir):
    t, c = tables["tracking"], tables["context"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    styles = {"actual": {"lw": 2.4}, "stat_baseline": {"lw": 1.6, "ls": "--"},
              "customer_raw": {"lw": 1.6}, "customer_bias_corrected": {"lw": 2.0}}
    for name, st in styles.items():
        d = t[t.series == name].sort_values("target_month")
        if len(d):
            ax.plot(d.target_month, d.value, color=COLOR[name], label=LABEL[name], **st)
    top = ax.get_ylim()[1] * 1.2
    _mark_break(ax, c.track_break_month, c.track_detected_at, top)
    ax.set_ylim(0, top * 1.02)
    ax.set_xlabel(f"Target month (forecast made {TRACK_HORIZON} months earlier)")
    ax.set_ylabel("Units / month")
    ax.legend(loc="upper left", ncol=3, handlelength=2.5, fontsize=9)
    ax.set_title(f"{TRACK_CUSTOMER}, family {c.track_family}, {TRACK_HORIZON} months ahead: after the "
                 f"break the raw customer forecast runs {_pct(tracking_bias(t, 'customer_raw'))}"
                 f" vs actual, bias-corrected {_pct(tracking_bias(t, 'customer_bias_corrected'))}")
    return _save(fig, outdir, "tracking_c5")


def chart_stability(tables, outdir):
    s = tables["summary"]
    s = s[(s.period == "all") & (s.horizon == TRACK_HORIZON)].set_index("method").mase
    st = tables["stability"]
    st = st[st.period == "all"].set_index("method").stability
    d = pd.DataFrame({"mase": s, "stability": st}).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.grid(True, axis="both")
    span_x = max(d.mase.max() - d.mase.min(), 0.05)
    span_y = max(d.stability.max(), 0.01)
    ax.set_xlim(d.mase.min() - 0.35 * span_x, d.mase.max() + 0.45 * span_x)
    ax.set_ylim(bottom=-0.04 * span_y, top=span_y * 1.15)
    for m, r in d.iterrows():
        ax.scatter(r.mase, r.stability, s=70, color=COLOR.get(m, INK2), edgecolor=SURFACE,
                   linewidth=2, zorder=3)
    _place_labels(fig, ax, d)
    ax.set_xlabel(f"MASE at {TRACK_HORIZON}-month horizon (lower = more accurate)")
    ax.set_ylabel("Mean revision between origins (share of mean demand)")
    calm = d.drop("seasonal_naive", errors="ignore").stability.idxmin()
    ax.set_title(f"Apart from seasonal naive, {LABEL.get(calm, calm).lower()} revises least; "
                 f"lower-left is better", loc="left")
    return _save(fig, outdir, "stability_vs_accuracy")


def _place_labels(fig, ax, d):
    """Label each point at the first candidate offset that overlaps no other label or point."""
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(d[["mase", "stability"]].to_numpy())
    taken = [matplotlib.transforms.Bbox.from_extents(x - 7, y - 7, x + 7, y + 7) for x, y in pts]
    offsets = ((8, 2, "left", "bottom"), (8, -2, "left", "top"), (-8, 2, "right", "bottom"),
               (-8, -2, "right", "top"), (0, 9, "center", "bottom"), (0, -9, "center", "top"))
    for i, (m, r) in enumerate(d.iterrows()):
        for k, (dx, dy, ha, va) in enumerate(offsets):     # keeps the last one if all collide
            label = ax.annotate(LABEL.get(m, m), (r.mase, r.stability), xytext=(dx, dy),
                                textcoords="offset points", ha=ha, va=va, fontsize=9, color=INK)
            box = label.get_window_extent(renderer)
            if k == len(offsets) - 1 or not any(box.overlaps(b) for j, b in enumerate(taken) if j != i):
                break
            label.remove()
        taken.append(box)


def make_charts(tables, outdir="outputs"):
    with plt.rc_context(RC):
        return [f(tables, outdir) for f in (chart_total, chart_bias, chart_fva,
                                             chart_tracking, chart_stability)]


def md_table(df, fmt="{:,.2f}"):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else fmt.format(v)
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(tables, outdir="outputs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tables["summary"].to_csv(out / "summary_metrics.csv", index=False)
    c, s, plan = tables["context"].iloc[0], tables["summary"], tables["planner"]
    bc = tables["by_customer"][tables["by_customer"].period == "all"]
    raw, cor = (bc[bc.method == m] for m in ("customer_raw", "customer_bias_corrected"))
    worst = raw.loc[raw.bias.abs().idxmax()]
    worst_cor = cor[(cor.customer == worst.customer) & (cor.horizon == worst.horizon)].bias
    trusted = sorted(raw.groupby("customer").bias.apply(lambda b: b.abs().max() <= 0.05)
                     .loc[lambda v: v].index)
    bias_tab = (bc[bc.method.isin(["customer_raw", "customer_bias_corrected"])]
                .assign(col=lambda d: d.method.map({"customer_raw": "raw",
                                                    "customer_bias_corrected": "corrected"})
                        + " h" + d.horizon.astype(str), pct=lambda d: 100 * d.bias)
                .pivot(index="customer", columns="col", values="pct"))
    bias_tab = bias_tab[sorted(bias_tab.columns, key=lambda k: (k[0] == "c", k))].reset_index()
    track_bias = lambda m: tracking_bias(tables["tracking"], m)  # noqa: E731
    h3 = s[s.horizon == TRACK_HORIZON]
    fv = h3.pivot(index="method", columns="period", values="fva_vs_naive")
    fv_post = fv["post"].drop("seasonal_naive", errors="ignore") if "post" in fv else pd.Series()
    best = fv_post.idxmax() if fv_post.notna().any() else "blend"
    base_pre = fv.get("pre", pd.Series()).get("stat_baseline", np.nan)
    base_post = fv_post.get("stat_baseline", np.nan)
    baseline_note = (f"stat baseline {base_pre:+.0f} pp before the break and {base_post:+.0f} pp after"
                     if pd.notna(base_pre) else f"stat baseline {base_post:+.0f} pp; no pre-break origins in this run")
    stab = tables["stability"][tables["stability"].period == "all"].set_index("method").stability
    mase = h3[h3.period == "all"].set_index("method").mase
    calm, jumpy = stab.drop("seasonal_naive", errors="ignore").idxmin(), stab.idxmax()
    missed = sorted(tables["alerts"].loc[tables["alerts"].break_detected_at_origin.isna(), "family"])
    n_flag = int(plan.attention_flag.sum())
    sources = plan.drop_duplicates("family").recommended_source.value_counts()
    alerts_md = tables["alerts"].rename(columns={"break_detected_at_origin": "detected_at_origin"})
    for col in ("detected_at_origin", "months_after_onset"):
        alerts_md[col] = alerts_md[col].astype("Int64")
    plan_md = plan.drop(columns=["origin_month", "target_month"])
    img = lambda n: f"![{n}](charts/{n}.png)"  # noqa: E731
    text = f"""# Forecast governance report

## 1. What this is

All data in this report is synthetic, generated by code in this repository with a fixed seed; no real company, customer or product appears. It evaluates {s.method.nunique()} forecast sources for {plan.customer.nunique()} customers and {plan.family.nunique()} product families on rolling origins, and reports bias, value added, stability and a break alert for each. It ends with a planner table for target month {plan.target_month.iloc[0]}.

## 2. Which customer forecasts to trust

{worst.customer}'s own forecast is the least trustworthy ({_pct(worst.bias)} bias at {worst.horizon} months, {_pct(worst_cor.iloc[0] if len(worst_cor) else np.nan)} after bias correction); {(", ".join(trusted) + " stay") if trusted else "no customer stays"} within 5% at every horizon.

{img("customer_bias_by_horizon")}

Bias in percent of actual demand (positive = customer over-forecasts), all origins pooled:

{md_table(bias_tab, "{:+.1f}")}

## 3. What happened at the break

Total demand stepped from {c.total_pre_level:,.0f} to {c.total_post_level:,.0f} units a month and the alert fired at origin {c.total_detected_at:.0f}, {c.total_months_after_onset:.0f} months after the onset at month {ONSET}.

{img("total_demand_and_break")}

For {TRACK_CUSTOMER} on {c.track_family} (the long-lead family with the largest step, {c.track_step_factor:.2f}x), the raw customer forecast ran {_pct(track_bias("customer_raw"))} against actuals after the break at {TRACK_HORIZON} months ahead, the bias-corrected forecast {_pct(track_bias("customer_bias_corrected"))} and the stat baseline {_pct(track_bias("stat_baseline"))}.

{img("tracking_c5")}

Families never detected: {", ".join(missed) or "none"}.

{md_table(alerts_md)}

## 4. Which forecast source to use where

After the break {LABEL.get(best, best).lower()} adds {fv_post.get(best, np.nan):+.0f} pp over seasonal naive at {TRACK_HORIZON} months ({baseline_note}), and {n_flag} of {len(plan)} planner rows need attention.

{img("fva_by_method")}

Recommended source by family at origin {plan.origin_month.iloc[0]}: {", ".join(f"{k} for {v} of {plan.family.nunique()} families" for k, v in sources.items())}. The planner table below is for target month {plan.target_month.iloc[0]} at horizon 1: the backtest stops at month 60, so the planning month is the latest target that still has a horizon-1 forecast. Flagged rows come first.

{md_table(plan_md)}

## 5. Stability

Apart from seasonal naive ({stab.get("seasonal_naive", np.nan):.2f}; it repeats last year's actual, so it can hardly revise), {LABEL.get(calm, calm).lower()} revises least ({stab[calm]:.2f} of mean demand per revision, MASE {mase.get(calm, np.nan):.2f}) and {LABEL.get(jumpy, jumpy).lower()} most ({stab[jumpy]:.2f}, MASE {mase.get(jumpy, np.nan):.2f}).

{img("stability_vs_accuracy")}

Planners care because every revision moves purchase orders and supplier schedules; a forecast that is slightly less accurate but much calmer can be the cheaper one to plan against.

## 6. What a forecast must report before a planner sees it

A forecast is not ready for a planner until it reports the items below; this report shows each of them, and the planner table carries the per-row ones. The standard itself is in [docs/governance_standard.md](../docs/governance_standard.md).

- Point forecast with an 80% interval, and the interval's achieved coverage.
- Signed bias against actuals, by customer, over a trailing window.
- Value added versus seasonal naive and versus the stat baseline.
- Stability: how much the number moved since the last origin.
- Whether history is reliable (break alert) and which source was chosen, and whether that changed.

## 7. Limitations

- Synthetic data from one fixed seed; break detection and bias numbers vary across seeds.
- No price or promotion effects.
- No capacity or supply constraints.
- Five customers and twelve families; a real portfolio has far more series and messier data.
- Break detection misses or lags lumpy families ({", ".join(missed) or "none missed in this run"}).
"""
    (out / "governance_report.md").write_text(text)
    return out / "governance_report.md"


def build_all(frames, forecasts, choices, alerts, outdir="outputs"):
    tables = build_tables(frames, forecasts, choices, alerts)
    make_charts(tables, outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    tables["planner"].to_csv(Path(outdir) / "planner_table.csv", index=False)
    write_report(tables, outdir)
    return tables
