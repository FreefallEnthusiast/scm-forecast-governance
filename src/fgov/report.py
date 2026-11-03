"""Governance report: tidy tables, five charts, the planner table and one markdown page.

Every number in the report text is computed from the tables; nothing is hardcoded.
Charts are static PNGs on a light surface with one fixed colour per method.
"""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fgov import breaks, metrics  # noqa: E402

HISTORY_END, TRACK_CUSTOMER, TRACK_HORIZON = 48, "C5", 3
CUSTOMER_METHODS = ["customer_raw", "customer_bias_corrected"]
LABEL = {"seasonal_naive": "Seasonal naive", "stat_baseline": "Stat baseline",
         "customer_raw": "Customer raw", "customer_bias_corrected": "Customer bias-corrected",
         "order_book": "Order book", "blend": "Blend", "actual": "Actual"}
COLOR = {"seasonal_naive": "#898781", "stat_baseline": "#2a78d6", "customer_raw": "#eb6834",
         "customer_bias_corrected": "#1baf7a", "order_book": "#eda100", "blend": "#e87ba4",
         "actual": "#0b0b0b"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LAG, PERIOD_COLOR = "#eb6834", {"pre": "#c3c2b7", "post": "#52514e"}
RC: dict[Any, Any] = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 150}
CHARTS = ("total_demand_and_break", "customer_bias_by_horizon", "fva_by_method",
          "tracking_c5", "stability_vs_accuracy")


def name(method) -> str:
    return LABEL.get(str(method), str(method))


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:+.0f}%"


def num(x, fmt) -> str:
    return "n/a" if pd.isna(x) else format(x, fmt)


def planner_table(frames, forecasts, choices, alerts):
    """One row per customer x family for the next planning month.

    The nominal planning origin is the last origin (60) with target 61, but
    forecasts stop at target 60. So the planning origin is the latest origin
    with horizon-1 blend rows (59 in the full run) and the target is origin + 1.
    """
    actuals = frames["actuals"]
    h1 = forecasts[(forecasts.method == "blend") & (forecasts.horizon == 1)]
    if h1.empty:
        raise ValueError("planner table needs horizon-1 blend rows")
    origin = int(h1.origin_month.max())
    blend = h1[h1.origin_month == origin][["customer", "family", "forecast", "lo80", "hi80"]]
    out = actuals[["customer", "family"]].drop_duplicates().merge(blend, how="left")
    chosen = choices.set_index(["origin_month", "family"])["chosen"]
    out["recommended_source"] = [chosen.get((origin, f), np.nan) for f in out.family]
    previous = [chosen.get((origin - 1, f), np.nan) for f in out.family]
    bias = metrics.customer_bias_trailing(frames["customer_forecasts"], actuals, origin, 1)
    out["customer_bias_trailing"] = out.customer.map(bias.set_index("customer").bias_ratio)
    out["history_unreliable"] = [breaks.history_unreliable(alerts, f, origin) for f in out.family]
    detected_now = set(alerts.loc[alerts.break_detected_at_origin == origin, "family"])
    triggers = zip([f"customer bias {100 * (b - 1):+.0f}%" if abs(b - 1) > 0.20 else ""
                    for b in out.customer_bias_trailing],
                   [f"source changed from {p}" if pd.notna(p) and p != r else ""
                    for p, r in zip(previous, out.recommended_source, strict=True)],
                   ["break detected this origin" if f in detected_now else "" for f in out.family], strict=True)
    reasons = ["; ".join(t for t in row if t) for row in triggers]
    out["attention_flag"] = [bool(r) for r in reasons]
    out["attention_reason"] = reasons
    out["customer_bias_trailing"] = out.customer_bias_trailing.round(3)
    out[["forecast", "lo80", "hi80"]] = out[["forecast", "lo80", "hi80"]].round(1)
    out.insert(0, "target_month", origin + 1)
    out.insert(0, "origin_month", origin)
    return out.sort_values(["attention_flag", "customer", "family"],
                           ascending=[False, True, True]).reset_index(drop=True)


def build_tables(frames, forecasts, choices, alerts, onset=breaks.ONSET):
    """Spec tables plus three the charts and text need: tracking, context, total_demand."""
    actuals, fams = frames["actuals"], frames["families"]
    evaluated = metrics.evaluate(forecasts, actuals, break_month=onset)
    summary = metrics.fva(metrics.summarise(evaluated))
    by_customer = metrics.summarise(evaluated, by=("customer",))
    stab = metrics.stability(forecasts, actuals, break_month=onset)
    long_lead = fams[fams.is_long_lead]                 # F01-F04 in the default panel
    fam = long_lead.loc[long_lead.step_factor.idxmax(), "family"]
    total, track = alerts[alerts.family == "TOTAL"].iloc[0], alerts[alerts.family == fam].iloc[0]
    # Headlines picked once here so chart titles and report text cannot disagree.
    raw = by_customer[(by_customer.period == "all") & (by_customer.method == "customer_raw")]
    worst = raw.loc[raw.bias.abs().idxmax()]
    pair = by_customer[by_customer.period == "all"].pivot_table(index=["customer", "horizon"], columns="method", values="bias")
    closer = (pair["customer_bias_corrected"].abs() <= pair["customer_raw"].abs())
    post = summary[(summary.horizon == TRACK_HORIZON) & (summary.period == "post")
                   & (summary.method != "seasonal_naive")].set_index("method").fva_vs_naive
    st = stab[stab.period == "all"].set_index("method").stability
    context = pd.DataFrame([{
        "onset": onset, "total_detected_at": total.break_detected_at_origin,
        "total_months_after_onset": total.months_after_onset, "total_break_month": total.break_month,
        "total_pre_level": total.pre_break_level, "total_post_level": total.post_break_level,
        "track_family": fam, "track_step_factor": long_lead.step_factor.max(),
        "track_detected_at": track.break_detected_at_origin, "track_break_month": track.break_month,
        "worst_customer": worst.customer, "worst_horizon": worst.horizon, "worst_bias": worst.bias,
        "pairs_closer": int(closer.sum()), "pairs_total": int(len(closer)),
        "best_post": post.idxmax() if post.notna().any() else "blend", "best_post_fva": post.max(),
        "calm": st.drop("seasonal_naive", errors="ignore").idxmin(), "jumpy": st.idxmax()}])
    f, a = forecasts, actuals
    lo, hi = f.origin_month.min(), f[f.horizon == TRACK_HORIZON].origin_month.max()
    sel = f[(f.customer == TRACK_CUSTOMER) & (f.family == fam) & (f.horizon == TRACK_HORIZON)
            & f.origin_month.between(lo, hi) & f.method.isin(["stat_baseline"] + CUSTOMER_METHODS)]
    act = a[(a.customer == TRACK_CUSTOMER) & (a.family == fam) & (a.month >= lo)]
    cols = ["target_month", "series", "value"]
    tracking = pd.concat([sel.rename(columns={"method": "series", "forecast": "value"})[cols],
                          act.assign(series="actual").rename(
                              columns={"month": "target_month", "actual": "value"})[cols]],
                         ignore_index=True)
    total_demand = breaks.monthly_totals(actuals, "total")["TOTAL"].fillna(0)
    typed = evaluated.merge(fams[["family", "is_intermittent"]], on="family")
    return {"summary": summary, "by_customer": by_customer,
            "by_family": metrics.summarise(evaluated, by=("family",)), "stability": stab,
            "by_type": metrics.fva(metrics.summarise(typed, by=("is_intermittent",))),
            "alerts": alerts, "planner": planner_table(frames, forecasts, choices, alerts),
            "tracking": tracking, "context": context,
            "total_demand": total_demand.rename("total").reset_index()}


def tracking_bias(tracking, series, onset):
    """Sum of forecasts over sum of actuals minus one, for target months from the onset."""
    post = tracking[tracking.target_month >= onset]
    d = post[post.series == series].set_index("target_month").value
    a = post[post.series == "actual"].set_index("target_month").value.reindex(d.index)
    return d.sum() / a.sum() - 1 if a.sum() > 0 else np.nan


def finish(fig, ax, outdir, chart, title, xlabel, ylabel, legend=None, **title_kw):
    """Shared chart ending: labels, optional legend, tight layout, 150 dpi PNG, close."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, **title_kw)
    if legend:
        ax.legend(**legend)
    path = Path(outdir) / "charts" / f"{chart}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def mark_break(ax, break_month, detected_at, top, onset):
    """Line at the detected break month; shaded lag from onset to the detection origin."""
    if pd.isna(break_month):
        return
    ax.axvline(break_month, color=INK2, lw=1.2, label=f"Detected break (month {break_month:.0f})")
    if pd.notna(detected_at):
        ax.axvspan(onset, detected_at, color=LAG, alpha=0.14, lw=0,
                   label=f"Detection lag (onset {onset} to alert at origin {detected_at:.0f})")
        ax.plot([detected_at], [top], marker="v", color=LAG, ms=8, clip_on=False, ls="")


def paired_bars(ax, x, series, colors, labels, w=0.38):
    for k, (v, c, lab) in enumerate(zip(series, colors, labels, strict=True)):
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=c, label=lab)


def chart_total(tables, outdir):
    t, c = tables["total_demand"], tables["context"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t.month, t.total / 1000, color=COLOR["stat_baseline"], lw=2, label="Total demand")
    top = ax.get_ylim()[1]
    ax.axvline(HISTORY_END + 0.5, color=AXIS, lw=0.8)
    ax.text(HISTORY_END + 1, top * 0.04, "holdout", color=INK2, fontsize=9)
    mark_break(ax, c.total_break_month, c.total_detected_at, top, c.onset)
    if pd.notna(c.total_detected_at):
        ax.annotate(f"detected {c.total_months_after_onset:.0f} months after onset",
                    xy=(c.total_detected_at + 0.3, top), xytext=(c.total_detected_at + 2, top * 0.97),
                    color=INK, fontsize=9, va="center",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_ylim(0, top * 1.02)
    ax.set_xlim(1, t.month.max())
    return finish(fig, ax, outdir, CHARTS[0], f"Total demand stepped from {c.total_pre_level / 1000:.1f}k to "
                  f"{c.total_post_level / 1000:.1f}k a month; break flagged "
                  f"{num(c.total_months_after_onset, '.0f')} months after onset",
                  "Month", "Total demand (thousand units / month)", {"loc": "upper left"})


def chart_bias(tables, outdir):
    b, c = tables["by_customer"], tables["context"].iloc[0]
    b = b[(b.period == "all") & b.method.isin(CUSTOMER_METHODS)]
    horizons, customers = sorted(b.horizon.unique()), sorted(b.customer.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 4), sharey=True, squeeze=False)
    x = np.arange(len(customers))
    for ax, h in zip(axes[0], horizons, strict=True):
        ax.axhspan(-20, 20, color=GRID, alpha=0.5, lw=0, label="+/-20% attention band" if h == horizons[0] else None)
        vals = [100 * b[(b.horizon == h) & (b.method == m)].set_index("customer").bias.reindex(customers)
                for m in CUSTOMER_METHODS]
        paired_bars(ax, x, vals, [COLOR[m] for m in CUSTOMER_METHODS], [LABEL[m] for m in CUSTOMER_METHODS])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, customers)
        ax.set_title(f"{h}-month horizon", fontweight="normal")
        ax.set_xlabel("Customer")
    fig.suptitle(f"Customer forecast bias by horizon: {c.worst_customer} is the most biased "
                 f"({pct(c.worst_bias)} at {c.worst_horizon} months)", fontweight="bold", fontsize=11)
    return finish(fig, axes[0][0], outdir, CHARTS[1], None, "Customer",
                  "Bias (% of actual, + = over-forecast)", {"loc": "upper left"})


def chart_fva(tables, outdir):
    s, c = tables["summary"], tables["context"].iloc[0]
    s = s[(s.horizon == TRACK_HORIZON) & (s.method != "seasonal_naive") & (s.period != "all")]
    methods = [m for m in LABEL if m in set(s.method)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(methods))
    vals = [s[s.period == p].set_index("method").fva_vs_naive.reindex(methods) for p in ("pre", "post")]
    paired_bars(ax, x, [v.fillna(0) for v in vals], [PERIOD_COLOR["pre"], PERIOD_COLOR["post"]],
                [f"{p}-break origins" + (" (none in this run)" if v.isna().all() else "")
                 for p, v in zip(("pre", "post"), vals, strict=True)])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [LABEL[m] for m in methods])
    return finish(fig, ax, outdir, CHARTS[2], f"After the break {name(c.best_post).lower()} adds the most "
                  f"value over seasonal naive ({num(c.best_post_fva, '+.0f')} pp, horizon {TRACK_HORIZON} months)",
                  "Forecast method", "FVA vs seasonal naive (percentage points)", {"loc": "best"})


def chart_tracking(tables, outdir):
    t, c = tables["tracking"], tables["context"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    styles = {"actual": {"lw": 2.4}, "stat_baseline": {"lw": 1.6, "ls": "--"},
              "customer_raw": {"lw": 1.6}, "customer_bias_corrected": {"lw": 2.0}}
    for series, st in styles.items():
        d = t[t.series == series].sort_values("target_month")
        if len(d):
            ax.plot(d.target_month, d.value, color=COLOR[series], label=LABEL[series], **st)
    top = ax.get_ylim()[1] * 1.2
    mark_break(ax, c.track_break_month, c.track_detected_at, top, c.onset)
    ax.set_ylim(0, top * 1.02)
    return finish(fig, ax, outdir, CHARTS[3], f"{TRACK_CUSTOMER} on {c.track_family}, {TRACK_HORIZON} months "
                  f"ahead: after the break raw customer forecast {pct(tracking_bias(t, 'customer_raw', c.onset))}, "
                  f"bias-corrected {pct(tracking_bias(t, 'customer_bias_corrected', c.onset))}",
                  f"Target month (forecast made {TRACK_HORIZON} months earlier)", "Units / month",
                  {"loc": "lower right", "ncol": 2, "handlelength": 2.5, "fontsize": 9})


def place_labels(fig, ax, d):
    """Label each point at the first offset that overlaps no other label or point."""
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(d[["mase", "stability"]].to_numpy())
    taken = [matplotlib.transforms.Bbox.from_extents(x - 7, y - 7, x + 7, y + 7) for x, y in pts]
    offsets = ((8, 2, "left", "bottom"), (8, -2, "left", "top"), (-8, 2, "right", "bottom"),
               (-8, -2, "right", "top"), (0, 9, "center", "bottom"), (0, -9, "center", "top"))
    for i, (m, r) in enumerate(d.iterrows()):
        others = [b for j, b in enumerate(taken) if j != i]
        for k, (dx, dy, ha, va) in enumerate(offsets):
            label = ax.annotate(name(m), (r.mase, r.stability), xytext=(dx, dy), ha=ha, va=va,
                                textcoords="offset points", fontsize=9, color=INK)
            box = label.get_window_extent(renderer)
            if k == len(offsets) - 1 or not any(box.overlaps(b) for b in others):
                taken.append(box)          # the last offset is kept even if it collides
                break
            label.remove()


def chart_stability(tables, outdir):
    s, st, c = tables["summary"], tables["stability"], tables["context"].iloc[0]
    d = pd.DataFrame({"mase": s[(s.period == "all") & (s.horizon == TRACK_HORIZON)].set_index("method").mase,
                      "stability": st[st.period == "all"].set_index("method").stability}).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.grid(True, axis="both")
    span_x, span_y = max(d.mase.max() - d.mase.min(), 0.05), max(d.stability.max(), 0.01)
    ax.set_xlim(d.mase.min() - 0.35 * span_x, d.mase.max() + 0.45 * span_x)
    ax.set_ylim(bottom=-0.04 * span_y, top=span_y * 1.15)
    for m, r in d.iterrows():
        ax.scatter(r.mase, r.stability, s=70, color=COLOR.get(str(m), INK2), edgecolor=SURFACE,
                   linewidth=2, zorder=3)
    place_labels(fig, ax, d)
    return finish(fig, ax, outdir, CHARTS[4],
                  f"Apart from seasonal naive, {name(c.calm).lower()} revises least; lower-left is better",
                  f"MASE at {TRACK_HORIZON}-month horizon (lower = more accurate)",
                  "Mean revision between origins (share of mean demand)", loc="left")


def make_charts(tables, outdir: str | Path = "outputs"):
    with plt.rc_context(RC):
        return [chart(tables, outdir)
                for chart in (chart_total, chart_bias, chart_fva, chart_tracking, chart_stability)]


def md_table(df, fmt="{:,.2f}"):
    def cell(v):
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else fmt.format(v)
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(cell(v) for v in row) + " |"
                              for row in df.itertuples(index=False)])


def write_report(tables, outdir: str | Path = "outputs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tables["summary"].to_csv(out / "summary_metrics.csv", index=False)
    s, plan, alerts, c = tables["summary"], tables["planner"], tables["alerts"], tables["context"].iloc[0]
    bc = tables["by_customer"][tables["by_customer"].period == "all"]
    worst_cor = bc[(bc.method == "customer_bias_corrected") & (bc.customer == c.worst_customer)
                   & (bc.horizon == c.worst_horizon)].bias
    worst_abs = bc[bc.method == "customer_raw"].groupby("customer").bias.apply(lambda b: b.abs().max())
    trusted = sorted(worst_abs[worst_abs <= 0.10].index)
    bias_tab = (bc[bc.method.isin(CUSTOMER_METHODS)]
                .assign(col=lambda d: d.method.map(dict(zip(CUSTOMER_METHODS, ("raw", "corrected"), strict=True)))
                        + " h" + d.horizon.astype(str), pct=lambda d: 100 * d.bias)
                .pivot(index="customer", columns="col", values="pct"))
    bias_tab = bias_tab[sorted(bias_tab.columns, key=lambda k: (k[0] == "c", k))].reset_index()
    fv = s[s.method != "seasonal_naive"].pivot(index=["horizon", "method"], columns="period", values="fva_vs_naive")
    fv3 = fv.loc[TRACK_HORIZON]
    fv_post = fv3["post"] if "post" in fv3 else pd.Series(dtype=float)
    base_pre = fv3["pre"].get("stat_baseline", np.nan) if "pre" in fv3 else np.nan
    base_post = fv_post.get("stat_baseline", np.nan)
    baseline_note = (f"stat baseline {base_pre:+.0f} pp before the break and {base_post:+.0f} pp after"
                     if pd.notna(base_pre)
                     else f"stat baseline {base_post:+.0f} pp; no pre-break origins in this run")
    blend_h1 = fv["all"].get((1, "blend"), np.nan)
    bt = tables["by_type"]
    bt = bt[(bt.horizon == TRACK_HORIZON) & (bt.period == ("post" if (bt.period == "post").any() else "all"))]
    fva_of = lambda m, inter: bt[(bt.method == m) & (bt.is_intermittent == inter)].fva_vs_naive.max()  # noqa: E731
    singles = pd.Series({m: fva_of(m, False) for m in ("stat_baseline", "customer_bias_corrected", "order_book")})
    best_smooth = singles.idxmax() if singles.notna().any() else "stat_baseline"
    book_note = (f"The blend's lead is routing, not a model. On the intermittent families it hands the forecast "
                 f"to the order book, which still sees most lumpy orders {TRACK_HORIZON} months out and adds "
                 f"{num(fva_of('order_book', True), '+.0f')} pp there against {num(fva_of('customer_bias_corrected', True), '+.0f')} pp "
                 f"for the bias-corrected customer forecast. On the smooth families it averages the three sources and "
                 f"adds {num(fva_of('blend', False), '+.0f')} pp, above the best single source ({name(best_smooth).lower()}, "
                 f"{num(singles.max(), '+.0f')} pp), because their errors are not aligned. At 1 month ahead the blend "
                 f"adds {num(blend_h1, '+.0f')} pp, and there too it is the confirmed order book doing the work.")
    mase = s[(s.horizon == TRACK_HORIZON) & (s.period == "all")].set_index("method").mase
    stab = tables["stability"][tables["stability"].period == "all"].set_index("method").stability
    missed = ", ".join(sorted(alerts.family[alerts.break_detected_at_origin.isna() & alerts.reason.ne("new_family")]))
    sources = plan.drop_duplicates("family").recommended_source.value_counts()
    alerts_md = alerts.drop(columns="break_month").rename(columns={"break_detected_at_origin": "detected_at_origin"})
    alert_note = (f"the alert fired at origin {c.total_detected_at:.0f}, {c.total_months_after_onset:.0f} months "
                  f"after the onset at month {c.onset}" if pd.notna(c.total_detected_at)
                  else f"no break alert fired (onset at month {c.onset})")
    for col in ("detected_at_origin", "months_after_onset", "pre_break_level", "post_break_level"):
        alerts_md[col] = alerts_md[col].round().astype("Int64")
    cov = lambda m, p: s[(s.horizon == TRACK_HORIZON) & (s.period == p) & (s.method == m)].coverage80.max()  # noqa: E731
    cov_note = (f"The stat baseline's 80% interval covered the actual {num(cov('stat_baseline', 'pre'), '.0%')} of the "
                f"time before the break and {num(cov('stat_baseline', 'post'), '.0%')} after, against "
                f"{num(cov('blend', 'pre'), '.0%')} and {num(cov('blend', 'post'), '.0%')} for the blend; intervals "
                f"built on stale history are too narrow once the level has moved, so a planner reading them as "
                f"80% bands would be surprised far more often than one time in five.")
    img = lambda n: f"![{n}](charts/{n}.png)"  # noqa: E731
    track = lambda m: pct(tracking_bias(tables["tracking"], m, c.onset))  # noqa: E731
    text = f"""# Forecast governance report

## 1. What this is

All data in this report is synthetic, generated by code in this repository with a fixed seed; no real company, customer or product appears. It evaluates {s.method.nunique()} forecast sources for {plan.customer.nunique()} customers and {plan.family.nunique()} product families on rolling origins, and reports bias, value added, stability and a break alert for each. It ends with a planner table for target month {plan.target_month.iloc[0]}.

## 2. Which customer forecasts to trust

{c.worst_customer}'s own forecast is the least trustworthy ({pct(c.worst_bias)} bias at {c.worst_horizon} months, {pct(worst_cor.iloc[0] if len(worst_cor) else np.nan)} after bias correction); {(", ".join(trusted) + " stay") if trusted else "no customer stays"} within 10% at every horizon. Bias correction moves {c.pairs_closer} of {c.pairs_total} customer-horizon pairs closer to zero.

{img("customer_bias_by_horizon")}

Bias in percent of actual demand (positive = customer over-forecasts), all origins pooled:

{md_table(bias_tab, "{:+.1f}")}

## 3. What happened at the break

Total demand stepped from {c.total_pre_level:,.0f} to {c.total_post_level:,.0f} units a month and {alert_note}.

{img("total_demand_and_break")}

For {TRACK_CUSTOMER} on {c.track_family} (the long-lead family with the largest step, {c.track_step_factor:.2f}x), the raw customer forecast ran {track("customer_raw")} against actuals after the break at {TRACK_HORIZON} months ahead, the bias-corrected forecast {track("customer_bias_corrected")} and the stat baseline {track("stat_baseline")}.

{img("tracking_c5")}

{cov_note}

Families never detected: {missed or "none"}.

{md_table(alerts_md)}

## 4. Which forecast source to use where

After the break {name(c.best_post).lower()} adds {num(c.best_post_fva, "+.0f")} pp over seasonal naive at {TRACK_HORIZON} months ({baseline_note}), and {int(plan.attention_flag.sum())} of {len(plan)} planner rows need attention. {book_note}

{img("fva_by_method")}

Recommended source by family at origin {plan.origin_month.iloc[0]}: {", ".join(f"{k} for {v} of {plan.family.nunique()} families" for k, v in sources.items())}. The planner table below is for target month {plan.target_month.iloc[0]} at horizon 1: the backtest stops at month 60, so the planning month is the latest target that still has a horizon-1 forecast. Flagged rows come first.

{md_table(plan.drop(columns=["origin_month", "target_month"]))}

## 5. Stability

Apart from seasonal naive ({stab.get("seasonal_naive", np.nan):.2f}; it repeats last year's actual, so it can hardly revise), {name(c.calm).lower()} revises least ({stab[c.calm]:.2f} of mean demand per revision, MASE {mase.get(str(c.calm), np.nan):.2f}) and {name(c.jumpy).lower()} most ({stab[c.jumpy]:.2f}, MASE {mase.get(str(c.jumpy), np.nan):.2f}).

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
- Break detection misses or lags lumpy families ({missed or "none missed in this run"}).
"""
    (out / "governance_report.md").write_text(text)
    return out / "governance_report.md"


def build_all(frames, forecasts, choices, alerts, outdir: str | Path = "outputs", onset=breaks.ONSET):
    tables = build_tables(frames, forecasts, choices, alerts, onset)
    make_charts(tables, outdir)
    tables["planner"].to_csv(Path(outdir) / "planner_table.csv", index=False)
    write_report(tables, outdir)
    return tables
