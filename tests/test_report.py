"""Report and charts, driven by a stub in place of the real candidate forecasts."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fgov import breaks, metrics, report, synth

METHODS = ["seasonal_naive", "stat_baseline", "customer_raw", "customer_bias_corrected",
           "order_book", "blend"]
FACTORS = {"stat_baseline": 0.95, "order_book": 0.9, "blend": 1.0}
CANDIDATES = ["stat_baseline", "customer_bias_corrected", "order_book", "average"]


def stub_forecasts(frames, origins, horizons=(1, 3, 6)):
    """Stand-in for forecast.run_all. Seasonal naive = actual 12 months before the
    target; customer_raw = the generated customer forecast; bias-corrected = raw
    over the customer's trailing bias; the rest = last actual x a factor.
    Intervals are +/-20% around every point."""
    actuals = frames["actuals"]
    a = actuals.set_index(["customer", "family", "month"]).actual
    cf = frames["customer_forecasts"].set_index(
        ["customer", "family", "origin_month", "horizon"]).customer_forecast
    keys = actuals[["customer", "family"]].drop_duplicates()
    parts = []
    for o in origins:
        for h in horizons:
            if o + h > 60:
                continue
            k = keys.assign(origin_month=o, target_month=o + h, horizon=h)
            idx = lambda *cols: pd.MultiIndex.from_arrays(list(cols))  # noqa: E731
            snaive = a.reindex(idx(k.customer, k.family, k.target_month - 12)).to_numpy()
            last = a.reindex(idx(k.customer, k.family, k.origin_month)).to_numpy()
            raw = cf.reindex(idx(k.customer, k.family, k.origin_month, k.horizon)).to_numpy()
            ratio = metrics.customer_bias_trailing(frames["customer_forecasts"], actuals, o, h)
            corrected = raw / k.customer.map(ratio.set_index("customer").bias_ratio).to_numpy()
            values = {"seasonal_naive": snaive, "customer_raw": raw,
                      "customer_bias_corrected": corrected}
            for m in METHODS:
                v = values.get(m, last * FACTORS.get(m, 1.0))
                parts.append(k.assign(method=m, forecast=v, lo80=0.8 * v, hi80=1.2 * v))
    return pd.concat(parts, ignore_index=True)


def stub_choices(families, origins):
    """Rotate the chosen candidate by origin so some families change source."""
    rows = [(o, f, CANDIDATES[(o + i) % len(CANDIDATES)] if i % 3 == 0 else "stat_baseline")
            for o in origins for i, f in enumerate(families)]
    return pd.DataFrame(rows, columns=["origin_month", "family", "chosen"])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    frames = synth.generate()
    alerts = breaks.alert_table(frames["actuals"])
    origins = range(37, 49)
    forecasts = stub_forecasts(frames, origins)
    planning_origin = 48     # latest origin with horizon-1 rows in the stub (target 49)
    # Force one family to be "newly detected" at the planning origin to exercise that trigger.
    alerts = alerts.copy()
    alerts.loc[alerts.family == "F08", ["break_detected_at_origin", "reason"]] = [planning_origin, "detected"]
    choices = stub_choices(frames["families"].family, origins)
    out = tmp_path_factory.mktemp("outputs")
    tables = report.build_all(frames, forecasts, choices, alerts, outdir=out)
    return out, tables


def test_all_outputs_exist(built):
    out, _ = built
    for name in report.CHARTS:
        assert (out / "charts" / f"{name}.png").stat().st_size > 10_000
    for f in ("planner_table.csv", "governance_report.md", "summary_metrics.csv"):
        assert (out / f).exists()


def test_planner_table(built):
    out, _ = built
    p = pd.read_csv(out / "planner_table.csv")
    assert len(p) == 60
    need = {"customer", "family", "recommended_source", "forecast", "lo80", "hi80",
            "customer_bias_trailing", "history_unreliable", "attention_flag", "attention_reason"}
    assert need <= set(p.columns)
    flags = p.attention_flag.to_numpy()
    assert flags.any() and not flags.all()
    assert not np.any(np.diff(flags.astype(int)) > 0)     # flagged rows first
    assert (p.origin_month == 48).all() and (p.target_month == 49).all()
    reasons = " ".join(p.attention_reason.dropna())
    for trigger in ("customer bias", "source changed", "break detected this origin"):
        assert trigger in reasons
    assert p.loc[p.family == "F08", "history_unreliable"].all()


def test_report_sections_in_order(built):
    out, _ = built
    text = (out / "governance_report.md").read_text()
    heads = ["## 1. What this is", "## 2. Which customer forecasts to trust",
             "## 3. What happened at the break", "## 4. Which forecast source to use where",
             "## 5. Stability", "## 6. What a forecast must report", "## 7. Limitations"]
    pos = [text.index(h) for h in heads]
    assert pos == sorted(pos)
    assert any(ch.isdigit() for ch in text[pos[1]:pos[2]])
    for name in report.CHARTS:
        assert f"](charts/{name}.png)" in text
    assert "docs/governance_standard.md" in text
    assert not re.search(r"\bnan\b", text, flags=re.I)


def test_fva_chart_with_pre_and_post_origins(tmp_path):
    """Origins straddling the break exercise the pre-break bars."""
    frames = synth.generate()
    alerts = breaks.alert_table(frames["actuals"])
    origins = range(33, 41)
    fc = stub_forecasts(frames, origins, horizons=(1, 3))
    tables = report.build_tables(frames, fc, stub_choices(frames["families"].family, origins), alerts)
    assert {"pre", "post"} <= set(tables["summary"].period)
    with plt.rc_context(report.RC):
        path = report.chart_fva(tables, tmp_path)
    assert path.exists()


def test_onset_threads_through(tmp_path):
    """A different step onset moves the alert lag, the pre/post split and the text with it."""
    params = synth.SynthParams(step_onset=40)
    frames = synth.generate(params)
    alerts = breaks.alert_table(frames["actuals"], onset=params.step_onset)
    det = alerts.dropna(subset=["break_detected_at_origin"])
    own = det.family.map(breaks.family_onsets(frames["actuals"], 40))
    assert (det.months_after_onset == det.break_detected_at_origin - own).all()
    origins = range(37, 49)
    fc = stub_forecasts(frames, origins, horizons=(1, 3))
    tables = report.build_all(frames, fc, stub_choices(frames["families"].family, origins), alerts,
                              outdir=tmp_path, onset=params.step_onset)
    ev = metrics.evaluate(fc, frames["actuals"], break_month=40)
    assert set(ev.loc[ev.origin_month < 40, "period"]) == {"pre"}
    assert set(ev.loc[ev.origin_month >= 40, "period"]) == {"post"}
    s = tables["summary"]
    assert {"pre", "post"} <= set(s.period)
    pre_n = s[(s.period == "pre") & (s.method == "blend") & (s.horizon == 1)].n.iloc[0]
    assert pre_n == len(ev[(ev.period == "pre") & (ev.method == "blend") & (ev.horizon == 1)])
    assert "onset at month 40" in (tmp_path / "governance_report.md").read_text()
