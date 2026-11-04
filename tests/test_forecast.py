import time

import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.models import TSB

from fgov import breaks, forecast, synth

pytestmark = pytest.mark.filterwarnings("ignore:This process.*fork:DeprecationWarning")

ORIGINS = list(range(37, 43))
HORIZONS = (1, 3, 6)
KEYS = ["origin_month", "horizon", "customer", "family"]


@pytest.fixture(scope="module")
def frames():
    return synth.generate()


@pytest.fixture(scope="module")
def run(frames):
    alerts = breaks.alert_table(frames["actuals"])
    return forecast.run_all(frames, alerts, origins=ORIGINS, horizons=HORIZONS)


def expected_keys(frames, method):
    a = frames["actuals"]
    if method in ("seasonal_naive", "stat_baseline"):
        rows = [(o, h, c, f) for o in ORIGINS for h in HORIZONS
                for (c, f), g in a.groupby(["customer", "family"])
                if g.loc[g.month <= o, "actual"].notna().any()]
        return set(rows)
    if method in ("customer_raw", "customer_bias_corrected"):
        d = frames["customer_forecasts"].dropna(subset=["customer_forecast"])
    else:
        b = frames["order_book"].dropna(subset=["confirmed_qty"])
        d = b.assign(origin_month=b.as_of_month, horizon=b.target_month - b.as_of_month)
    d = d[d.origin_month.isin(ORIGINS) & d.horizon.isin(HORIZONS)]
    return set(map(tuple, d[KEYS].to_numpy()))


def test_seasonal_naive_is_actual_twelve_months_back(frames):
    a = frames["actuals"]
    sn = forecast.seasonal_naive(a, ORIGINS, HORIZONS)
    lag = a.assign(target_month=a.month + 12)[["customer", "family", "target_month", "actual"]]
    m = sn.merge(lag, on=["customer", "family", "target_month"]).dropna(subset=["actual"])
    assert len(m) > 0
    np.testing.assert_allclose(m.forecast, m.actual)


@pytest.mark.parametrize("method", ["seasonal_naive", "stat_baseline", "customer_raw",
                                    "customer_bias_corrected", "order_book"])
def test_every_method_covers_its_grid(run, frames, method):
    fc, _ = run
    got = fc[fc.method == method]
    assert set(got.origin_month) == set(ORIGINS)
    assert not got.duplicated(KEYS).any()
    assert set(map(tuple, got[KEYS].to_numpy())) == expected_keys(frames, method)


def test_blend_rows_follow_choices(run):
    fc, choices = run
    candidates = fc[fc.method.isin(["stat_baseline", "customer_bias_corrected", "order_book"])]
    n_ref = candidates.groupby(["origin_month", "family", "method"]).size()
    n_any = candidates.drop_duplicates(KEYS).groupby(["origin_month", "family"]).size()
    n_blend = fc[fc.method == "blend"].groupby(["origin_month", "family"]).size()
    for o, f, chosen in choices.itertuples(index=False):
        expected = n_any.get((o, f), 0) if chosen == "average" else n_ref.get((o, f, chosen), 0)
        assert n_blend.get((o, f), 0) == expected


def test_non_negative_and_ordered_intervals(run):
    fc, _ = run
    assert set(fc.method) == {"seasonal_naive", "stat_baseline", "customer_raw",
                              "customer_bias_corrected", "order_book", "blend"}
    assert fc.target_month.max() <= 60
    assert fc.forecast.notna().all() and (fc.forecast >= 0).all()
    assert (fc.lo80.dropna() >= 0).all() and (fc.hi80.dropna() >= 0).all()
    has = fc.lo80.notna() & fc.hi80.notna()
    assert has.mean() > 0.95
    sub = fc[has]
    assert (sub.lo80 <= sub.forecast + 1e-9).all() and (sub.forecast <= sub.hi80 + 1e-9).all()


def test_blend_choices_valid(run):
    _, choices = run
    assert set(choices.chosen) <= {"stat_baseline", "customer_bias_corrected", "order_book", "average"}
    assert len(choices) == len(ORIGINS) * 12
    assert not choices.duplicated(["origin_month", "family"]).any()


def pooled_bias(rows, actuals):
    m = rows.merge(actuals.rename(columns={"month": "target_month"}),
                   on=["customer", "family", "target_month"]).dropna(subset=["actual"])
    sums = m.assign(error=m.forecast - m.actual).groupby("customer")[["error", "actual"]].sum()
    return sums.error / sums.actual


def test_bias_correction_reduces_bias_for_c2_c3(frames):
    a, cf = frames["actuals"], frames["customer_forecasts"]
    origins = range(37, 49)
    raw = pooled_bias(forecast.customer_raw(cf, origins, (6,)), a)
    cor = pooled_bias(forecast.customer_bias_corrected(cf, a, origins, (6,)), a)
    for c in ("C2", "C3"):
        assert abs(cor[c]) < abs(raw[c])


def test_bias_correction_uses_no_future_actuals(frames):
    a, cf = frames["actuals"], frames["customer_forecasts"]
    origin = 40
    garbage = a.copy()
    future = garbage.month > origin
    garbage.loc[future, "actual"] = np.random.default_rng(0).uniform(0, 1e6, future.sum())
    real = forecast.customer_bias_corrected(cf, a, [origin], HORIZONS)
    fake = forecast.customer_bias_corrected(cf, garbage, [origin], HORIZONS)
    pd.testing.assert_frame_equal(real, fake)
    book_real = forecast.order_book(frames["order_book"], a, [origin], HORIZONS)
    book_fake = forecast.order_book(frames["order_book"], garbage, [origin], HORIZONS)
    pd.testing.assert_frame_equal(book_real, book_fake)


def test_empirical_interval_needs_five_residuals(frames):
    a = frames["actuals"]
    sn = forecast.seasonal_naive(a, range(25, 31), (1,))
    emp = forecast.empirical_interval(sn, a)
    assert emp.loc[sn.origin_month == 25].isna().all().all()   # no realised past forecasts yet
    assert emp.loc[sn.origin_month == 30].notna().all().all()  # 5 origins x 5 customers


def test_empirical_interval_scales_with_customer_level(frames):
    a = frames["actuals"]
    sn = forecast.seasonal_naive(a, range(25, 41), (1,))
    emp = forecast.empirical_interval(sn, a)
    rows = sn.join(emp, rsuffix="_emp")
    rows = rows[(rows.origin_month == 40) & (rows.family == "F01")]
    level = forecast.series_level(a)
    rows = rows.merge(level, on=["customer", "family", "origin_month"])
    big, small = rows.loc[rows.level.idxmax()], rows.loc[rows.level.idxmin()]
    assert big.level > 2 * small.level
    up = (rows.hi80_emp - rows.forecast) / rows.level     # the same relative q90 for every customer
    down = (rows.forecast - rows.lo80_emp) / rows.level
    np.testing.assert_allclose(up, up.iloc[0])
    np.testing.assert_allclose(down, down.iloc[0])
    assert (big.hi80_emp - big.lo80_emp) > 2 * (small.hi80_emp - small.lo80_emp)


def test_stat_baseline_three_origins_is_fast(frames):
    t0 = time.perf_counter()
    out = forecast.stat_baseline(frames["actuals"], [40, 41, 42], HORIZONS)
    assert time.perf_counter() - t0 < 30      # about 6 s idle; guards the 116 s single-worker regression
    assert len(out) > 0 and (out.forecast >= 0).all()


# --- Tests added from mutation testing -------------------------------------------------


def _garbage_after(actuals, origin):
    g = actuals.copy()
    future = g.month > origin
    g.loc[future, "actual"] = np.random.default_rng(0).uniform(0, 1e6, future.sum())
    return g


def test_intervals_use_no_future_actuals(frames):
    a, origin = frames["actuals"], 40
    sn = forecast.seasonal_naive(a, range(25, origin + 1), HORIZONS)
    real = forecast.empirical_interval(sn, a, origin=origin)
    fake = forecast.empirical_interval(sn, _garbage_after(a, origin), origin=origin)
    assert real.notna().all(axis=1).mean() > 0.9
    pd.testing.assert_frame_equal(real, fake)


def test_stat_baseline_uses_no_future_actuals(frames):
    a, origin = frames["actuals"], 40
    real = forecast.stat_baseline(a, [origin], (1,))
    fake = forecast.stat_baseline(_garbage_after(a, origin), [origin], (1,))
    pd.testing.assert_frame_equal(real, fake)


def _one_series(values, customer="C1", family="F01"):
    return pd.DataFrame({"customer": customer, "family": family,
                         "month": np.arange(1, len(values) + 1), "actual": np.asarray(values, float)})


@pytest.mark.parametrize("zero_share,expect_tsb", [(0.5, True), (0.375, False)])
def test_intermittent_threshold_routes_to_tsb(zero_share, expect_tsb):
    n = 24
    rng = np.random.default_rng(1)
    y = rng.uniform(50, 150, n).round()
    y[rng.permutation(n)[:int(zero_share * n)]] = 0.0
    got = forecast._stat_one_origin(_one_series(y), n, n_jobs=1).sort_values("ds")["forecast"]
    df = pd.DataFrame({"unique_id": "C1|F01", "ds": np.arange(1, n + 1), "y": y})
    tsb = StatsForecast(models=[TSB(alpha_d=0.2, alpha_p=0.2)], freq=1).forecast(h=forecast.MAX_H, df=df)
    assert np.allclose(got.to_numpy(), tsb["TSB"].to_numpy()) == expect_tsb


def _flat_customer_frames(forecast_by_target, n=30, value=100.0):
    act = _one_series(np.full(n, value))
    t = np.arange(2, n + 1)
    cf = pd.DataFrame({"origin_month": t - 1, "target_month": t, "horizon": 1, "customer": "C1",
                       "family": "F01", "customer_forecast": [forecast_by_target(x) for x in t]})
    return act, cf


def test_bias_correction_trailing_window_is_twelve_targets():
    # Origin 20: the window is targets 9..20, all unbiased; target 8 is far off and must not count.
    act, cf = _flat_customer_frames(lambda t: 1000.0 if t == 8 else 100.0)
    out = forecast.customer_bias_corrected(cf, act, [20], (1,))
    assert out["forecast"].iloc[0] == pytest.approx(100.0)


def test_bias_correction_falls_back_when_trailing_ratio_is_zero():
    act, cf = _flat_customer_frames(lambda t: 0.0 if t <= 20 else 50.0)
    out = forecast.customer_bias_corrected(cf, act, [20], (1,))
    assert out["forecast"].iloc[0] == pytest.approx(50.0)


def test_order_book_falls_back_when_trailing_fill_is_zero():
    act = _one_series(np.full(30, 100.0))
    t = np.arange(2, 31)
    book = pd.DataFrame({"as_of_month": t - 1, "target_month": t, "customer": "C1", "family": "F01",
                         "confirmed_qty": np.where(t <= 20, 0.0, 40.0)})
    out = forecast.order_book(book, act, [20], (1,))
    assert out["forecast"].iloc[0] == pytest.approx(40.0)


BLEND_O = 26
BASE_ERR = {"stat_baseline": 10.0, "customer_bias_corrected": 20.0, "order_book": 30.0, "average": 25.0}
NO_ALERTS = pd.DataFrame(columns=["family", "break_detected_at_origin", "reason"])


def _blend_case(huge=None, origins=range(13, BLEND_O), extra=(), base=None):
    """One series, horizon-1 candidate rows at `origins`; `huge` = (method, origin) gets error 1000.
    `extra` adds (method, origin, target, error) rows."""
    act = _one_series(np.random.default_rng(2).uniform(50, 150, 40).round())
    base = BASE_ERR if base is None else base
    rows = [(m, o, o + 1, 1000.0 if huge == (m, o) else e) for m, e in base.items() for o in origins]
    rows += list(extra)
    df = pd.DataFrame(rows, columns=["method", "origin_month", "target_month", "err"])
    df = df.merge(act.rename(columns={"month": "target_month"}), on="target_month")
    df = df.assign(horizon=df.target_month - df.origin_month, forecast=df.actual + df.err,
                   lo80=np.nan, hi80=np.nan)
    return df[forecast.COLS], act


def _chosen(rows, act, alerts=NO_ALERTS):
    _, choices = forecast.blend(rows, act, alerts, [BLEND_O])
    return choices["chosen"].iloc[0]


def test_blend_window_includes_origin_minus_12_and_excludes_minus_13():
    assert _chosen(*_blend_case(huge=("stat_baseline", BLEND_O - 12))) == "customer_bias_corrected"
    assert _chosen(*_blend_case(huge=("stat_baseline", BLEND_O - 13))) == "stat_baseline"


def test_blend_ignores_targets_not_yet_realised():
    leak = [("stat_baseline", BLEND_O - 2, BLEND_O + 1, 1e5)]   # target after the origin
    assert _chosen(*_blend_case(extra=leak)) == "stat_baseline"


def test_blend_needs_min_origins_and_breaks_ties_in_candidate_order():
    assert _chosen(*_blend_case(origins=range(BLEND_O - 3, BLEND_O))) == "stat_baseline"
    assert _chosen(*_blend_case(origins=range(BLEND_O - 2, BLEND_O))) == "average"
    same = dict.fromkeys(BASE_ERR, 10.0)
    assert _chosen(*_blend_case(base=same)) == "stat_baseline"


def test_blend_shorter_window_when_history_unreliable():
    rows, act = _blend_case(huge=("stat_baseline", BLEND_O - 8))
    flagged = pd.DataFrame({"family": ["F01"], "break_detected_at_origin": [20.0], "reason": ["detected"]})
    assert _chosen(rows, act) == "customer_bias_corrected"           # 12-origin window sees it
    assert _chosen(rows, act, flagged) == "stat_baseline"            # 6-origin window does not
