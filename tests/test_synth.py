from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fgov import synth
from fgov.synth import SynthParams, generate, write_outputs

P = SynthParams()


@pytest.fixture(scope="module")
def frames():
    return generate(P)


def _with_actuals(df, frames, month_col):
    a = frames["actuals"][["customer", "family", "month", "actual", "is_intermittent"]]
    m = df.merge(a, left_on=["customer", "family", month_col], right_on=["customer", "family", "month"])
    return m.dropna(subset=["actual"])


def _bias(frames, post):
    m = _with_actuals(frames["customer_forecasts"], frames, "target_month")
    m = m[m.target_month >= P.step_onset] if post else m[m.target_month < P.step_onset]
    g = m.groupby(["customer", "horizon"])[["customer_forecast", "actual"]].sum()
    return g.customer_forecast / g.actual


def test_shapes(frames):
    a, fc, ob = frames["actuals"], frames["customer_forecasts"], frames["order_book"]
    n_m = P.n_history + P.n_holdout
    assert n_m == 60
    assert a.customer.nunique() == 5 and a.family.nunique() == 12
    assert len(a) == 60 * 5 * 12
    assert sorted(a.month.unique()) == list(range(1, 61))
    expected_fc = sum(n_m - h for h in P.horizons) * 5 * 12
    assert len(fc) == expected_fc
    assert (fc.origin_month >= 1).all()
    assert (fc.target_month - fc.origin_month == fc.horizon).all()
    assert sorted(fc.horizon.unique()) == [1, 3, 6]
    assert len(ob) == sum(n_m - h for h in P.book_horizons) * 5 * 12
    assert set(a.columns) >= {"customer", "family", "month", "actual",
                              "is_intermittent", "is_new", "is_long_lead"}
    assert list(fc.columns) == ["origin_month", "target_month", "horizon",
                                "customer", "family", "customer_forecast"]
    assert list(ob.columns) == ["as_of_month", "target_month", "customer", "family", "confirmed_qty"]


def test_family_mix(frames):
    fam = frames["families"]
    assert fam.is_intermittent.sum() == 4 and fam.is_new.sum() == 2
    assert (~fam.is_intermittent & ~fam.is_new).sum() == 6
    assert not (fam.is_long_lead & (fam.is_intermittent | fam.is_new)).any()


def test_no_negatives(frames):
    assert (frames["actuals"].actual.dropna() >= 0).all()
    assert (frames["customer_forecasts"].customer_forecast.dropna() >= 0).all()
    assert (frames["order_book"].confirmed_qty.dropna() >= 0).all()


def test_new_families_launch(frames):
    a = frames["actuals"]
    new = a[a.is_new]
    assert new[new.month < P.new_launch_month].actual.isna().all()
    assert (new[new.month >= P.new_launch_month].actual > 0).all()
    assert a[~a.is_new].actual.notna().all()


def test_intermittent_zero_share(frames):
    a = frames["actuals"]
    train = a[a.is_intermittent & (a.month <= P.n_history)]
    zero = train.groupby("family").actual.apply(lambda s: (s == 0).mean())
    assert ((zero >= 0.40) & (zero <= 0.70)).all(), zero
    smooth = a[~a.is_intermittent & ~a.is_new]
    assert (smooth.actual == 0).mean() < 0.01


def test_step_factor_applied(frames):
    total = frames["actuals"].groupby("month").actual.sum()
    drawn = frames["families"].total_step_factor.iloc[0]
    assert P.total_step_range[0] <= drawn <= P.total_step_range[1]
    ratio = total.loc[43:48].mean() / total.loc[25:36].mean()
    assert abs(ratio / drawn - 1) <= 0.25
    est = frames["families"][~frames["families"].is_new]
    assert est.step_factor.between(*P.family_step_range).all()


def test_long_lead_steps_first(frames):
    a = frames["actuals"]
    smooth = a[~a.is_new & ~a.is_intermittent].groupby(["is_long_lead", "month"]).actual.sum()
    before = P.step_onset - P.long_lead_offset
    jump = {flag: smooth.loc[flag].loc[before] / smooth.loc[flag].loc[before - 12:before - 1].mean()
            for flag in (True, False)}
    assert jump[True] > 1.05        # long-lead families already moving
    assert jump[False] < 1.10       # the rest still flat
    assert jump[True] > jump[False]


def test_customer_bias_pre_break(frames):
    b = _bias(frames, post=False)
    assert b["C2", 6] == pytest.approx(1.30, abs=0.08)
    assert b["C2", 3] == pytest.approx(1.15, abs=0.08)
    assert b["C2", 1] < b["C2", 3] < b["C2", 6]
    for h in P.horizons:
        assert b["C3", h] == pytest.approx(0.85, abs=0.06)
        assert b["C1", h] == pytest.approx(1.0, abs=0.08)
        assert b["C4", h] == pytest.approx(1.0, abs=0.10)
        assert b["C5", h] == pytest.approx(1.0, abs=0.08)


def test_c5_panics_post_break(frames):
    b = _bias(frames, post=True)
    for h in P.horizons:
        assert b["C5", h] > 1.2


def test_c4_revisions_are_large(frames):
    fc = frames["customer_forecasts"].dropna()
    wide = fc.pivot_table(index=["customer", "family", "target_month"], columns="horizon",
                          values="customer_forecast").dropna()
    rev = (wide[3] - wide[1]).abs().groupby("customer").sum() / wide[1].groupby("customer").sum()
    assert rev["C4"] > 2 * rev.drop("C4").max()


def test_order_book_fill_rises_as_horizon_shortens(frames):
    m = _with_actuals(frames["order_book"], frames, "target_month")
    m["h"] = m.target_month - m.as_of_month
    for post in (False, True):
        sub = m[(m.target_month >= P.step_onset) == post]
        g = sub.groupby("h")[["confirmed_qty", "actual"]].sum()
        fill = (g.confirmed_qty / g.actual).sort_index()
        assert np.all(np.diff(fill.to_numpy()) < 0), fill
    # The fill parameter applies directly to smooth families; lumpy orders may
    # not be on the book yet far out, which lowers the pooled ratio.
    smooth = m[(m.target_month < P.step_onset) & ~m.is_intermittent]
    pre = smooth.groupby("h")[["confirmed_qty", "actual"]].sum()
    pre_fill = pre.confirmed_qty / pre.actual
    assert pre_fill[6] == pytest.approx(0.30, abs=0.05)
    assert pre_fill[1] == pytest.approx(0.85, abs=0.05)
    post = m[(m.target_month >= P.step_onset) & ~m.is_intermittent].groupby("h")[["confirmed_qty", "actual"]].sum()
    assert (post.confirmed_qty / post.actual)[6] == pytest.approx(0.40, abs=0.05)   # +0.10 upcycle boost
    assert (m.confirmed_qty <= m.actual).all()


def test_seed_is_deterministic(frames):
    again = generate(P)
    assert again["actuals"].equals(frames["actuals"])


def test_write_outputs(frames, tmp_path):
    write_outputs(frames, tmp_path)
    for name in ["actuals.parquet", "customer_forecasts.csv", "order_book.csv", "families.csv"]:
        assert (tmp_path / name).exists()
    # No index column leaks into the files.
    assert list(pd.read_parquet(tmp_path / "actuals.parquet").columns) == list(frames["actuals"].columns)
    for name in ["customer_forecasts", "order_book", "families"]:
        assert list(pd.read_csv(tmp_path / f"{name}.csv").columns) == list(frames[name].columns)


# --- Tests added from mutation testing -------------------------------------------------


def test_family_labels_and_long_lead_set(frames):
    fam = frames["families"]
    assert list(fam.family) == [f"F{i:02d}" for i in range(1, 13)]
    assert list(fam.family[fam.is_long_lead]) == ["F01", "F02", "F03", "F04"]


def test_params_are_used(frames):
    other = generate(SynthParams(seed=1))
    assert not other["actuals"].actual.equals(frames["actuals"].actual)


@pytest.mark.parametrize("seed", [3, 5])   # seed 3 clips at the floor, seed 5 at the ceiling
def test_step_factors_hit_total_and_stay_in_range(seed):
    rng = np.random.default_rng(0)
    weights = rng.lognormal(0, 0.4, 12)
    is_new = np.arange(12) >= 10
    total, g = synth.step_factors(P, np.random.default_rng(seed), weights, is_new)
    lo, hi = P.family_step_range
    assert np.isclose(g[~is_new], lo).any() or np.isclose(g[~is_new], hi).any()
    assert P.total_step_range[0] <= total <= P.total_step_range[1]
    assert (g[is_new] == 1.0).all()
    assert ((g[~is_new] >= lo) & (g[~is_new] <= hi)).all()
    new_total = (weights[~is_new] * g[~is_new]).sum() + weights[is_new].sum()
    assert new_total == pytest.approx(total * weights[~is_new].sum(), rel=1e-9)


def test_step_profile_ramp_and_long_lead_offset():
    months = np.arange(34, 41)
    f = 1.9
    ramp = [1.0, 1.0, 1.0, 1 + 0.9 / 3, 1 + 1.8 / 3, f, f]             # months 34..40, onset 37
    np.testing.assert_allclose(synth.step_profile(P, months, f, False), ramp)
    np.testing.assert_allclose(synth.step_profile(P, months, f, True), ramp[1:] + [f])


def _long_run(n_months=600):
    months = np.arange(1, n_months + 1)
    fam = synth.family_table(P)
    shape = (len(P.customers), len(fam), n_months)
    return months, fam, shape


def test_customer_forecasts_track_the_target_month_with_unbiased_noise():
    months, fam, shape = _long_run()
    # Alternating signal: any off-by-one in the target month shows up as a 1.25 ratio.
    signal = np.broadcast_to(np.where(months % 2 == 0, 1000.0, 2000.0), shape)
    cf = synth.customer_forecasts(P, np.random.default_rng(0), fam, signal, months)
    cf["signal"] = np.where(cf.target_month % 2 == 0, 1000.0, 2000.0)
    cf["ratio"] = cf.customer_forecast / cf.signal
    pre = cf[cf.target_month < P.step_onset - 1]
    late = cf[cf.target_month >= P.step_onset]
    r = late.groupby(["customer", "horizon"]).ratio.mean()
    for h, b in zip(P.horizons, P.upcycle_bias, strict=True):
        assert r["C1", h] == pytest.approx(1.0, abs=0.01)
        assert r["C4", h] == pytest.approx(1.0, abs=0.015)   # mean-one noise even at sigma 0.4
        assert r["C5", h] == pytest.approx(b, rel=0.01)
    assert pre[pre.customer == "C5"].ratio.mean() == pytest.approx(1.0, abs=0.03)
    # Panic starts exactly at the onset target month.
    c5 = cf[(cf.customer == "C5") & (cf.horizon == 1)].groupby("target_month").ratio.mean()
    assert c5[P.step_onset - 1] < 1.15 < c5[P.step_onset]


def test_order_book_fill_ratio_and_upcycle_boost_exact():
    months, fam, shape = _long_run(60)
    quiet = replace(P, fill_noise_sigma=(1e-12,) * 6)
    ob = synth.order_book(quiet, np.random.default_rng(0), fam, np.full(shape, 1e6), months)
    ob = ob.merge(fam[["family", "is_intermittent"]], on="family")
    ob = ob[~ob.is_intermittent]
    h = ob.target_month - ob.as_of_month
    expected = np.array(P.fill_ratio)[h - 1] + P.upcycle_fill_boost * (ob.target_month >= P.step_onset)
    np.testing.assert_allclose(ob.confirmed_qty / 1e6, expected, atol=2e-6)


def test_order_book_noise_is_mean_one_and_lumpy_orders_can_be_missing():
    months, fam, shape = _long_run()
    low = replace(P, fill_ratio=(0.1,) * 6, upcycle_fill_boost=0.0)
    ob = synth.order_book(low, np.random.default_rng(0), fam, np.full(shape, 1e6), months)
    ob = ob.merge(fam[["family", "is_intermittent"]], on="family")
    ob["h"] = ob.target_month - ob.as_of_month
    smooth = ob[~ob.is_intermittent].groupby("h").confirmed_qty.mean() / 1e6
    np.testing.assert_allclose(smooth, 0.1, rtol=0.02)
    seen = ob[ob.is_intermittent].groupby("h").confirmed_qty.apply(lambda q: (q > 0).mean())
    np.testing.assert_allclose(seen, P.book_visibility, atol=0.02)


def test_family_sizes_follow_weights(frames):
    a = frames["actuals"]
    pre = a[a.month < P.step_onset - 1].groupby("family").actual.mean()
    smooth, lumpy = pre.loc["F01":"F06"], pre.loc["F07":"F10"]
    new = a[a.is_new & (a.month >= P.new_launch_month)].groupby("family").actual.mean()
    assert 1.3 < smooth.max() / smooth.min() < 10
    assert 0.3 < lumpy.mean() / smooth.mean() < 0.75      # 0.40 weight x lognormal spread (0.56 here)
    assert 0.2 < new.mean() / smooth.mean() < 0.7         # 0.35 weight, new families do not step


def test_total_level_and_customer_shares(frames):
    a = frames["actuals"]
    pre = a[a.month <= 36]
    assert pre.groupby("month").actual.sum().mean() == pytest.approx(P.base_total, rel=0.08)
    share = pre.groupby("customer").actual.sum() / pre.actual.sum()
    np.testing.assert_allclose(share.to_numpy(), P.customer_shares, atol=0.04)


def test_seasonality_phase_and_amplitude(frames):
    a = frames["actuals"]
    smooth = a[~a.is_intermittent & ~a.is_new & (a.month < P.step_onset - 2)]
    total = smooth.groupby("month").actual.sum()
    m = total.index.to_numpy()
    x = np.c_[np.ones(len(m)), np.sin(2 * np.pi * (m - 1) / 12), np.cos(2 * np.pi * (m - 1) / 12)]
    _, sin_coef, cos_coef = np.linalg.lstsq(x, np.log(total.to_numpy()), rcond=None)[0]
    assert sin_coef == pytest.approx(P.season_amplitude, abs=0.03)
    assert cos_coef == pytest.approx(0.0, abs=0.03)


def test_smooth_series_have_monthly_noise(frames):
    a = frames["actuals"]
    year2 = a[~a.is_intermittent & ~a.is_new & a.month.between(13, 24)]
    cv = year2.groupby(["customer", "family"]).actual.agg(lambda s: s.std() / s.mean())
    assert 0.11 < cv.median() < 0.22                     # sigma 0.15 noise plus +/-10% season


def test_intermittent_customer_forecast_is_expected_level(frames):
    # Customers cannot see which month a lumpy order lands: they forecast the level,
    # so a month with zero actual still has a positive forecast.
    m = _with_actuals(frames["customer_forecasts"], frames, "target_month")
    lumpy_zero = m[m.is_intermittent & (m.actual == 0)]
    assert len(lumpy_zero) > 100
    assert (lumpy_zero.customer_forecast > 0).mean() > 0.9


def test_month_start_is_calendar_month(frames):
    a = frames["actuals"]
    starts = a.drop_duplicates("month").set_index("month").month_start
    assert starts.loc[1] == pd.Timestamp(P.start_date)
    assert (starts.dt.day == 1).all()
    assert starts.loc[13] == pd.Timestamp(P.start_date) + pd.DateOffset(years=1)


def test_intermittent_demand_is_mean_one_around_the_expected_level():
    # Long panel: lumpy actuals against C1/C4 one-month forecasts, which are the
    # expected level times mean-one noise with no bias, so the pooled ratio is ~1.
    long = replace(P, n_history=2388)
    f = generate(long)
    m = _with_actuals(f["customer_forecasts"], f, "target_month")
    m = m[m.is_intermittent & m.customer.isin(["C1", "C4"]) & (m.horizon == 1)]
    assert m.actual.sum() / m.customer_forecast.sum() == pytest.approx(1.0, abs=0.04)


def test_story_constants_match_spec():
    # The rest of the pipeline (breaks.ONSET, report periods) assumes these.
    assert P.step_onset == 37 and P.new_launch_month == 37
    assert P.n_history == 48 and P.n_holdout == 12
