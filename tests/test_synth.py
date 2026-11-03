import numpy as np
import pytest

from fgov.synth import SynthParams, generate, write_outputs

P = SynthParams()


@pytest.fixture(scope="module")
def frames():
    return generate(P)


def _with_actuals(df, frames, month_col):
    a = frames["actuals"][["customer", "family", "month", "actual"]]
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
    pre = m[m.target_month < P.step_onset].groupby("h")[["confirmed_qty", "actual"]].sum()
    pre_fill = pre.confirmed_qty / pre.actual
    assert pre_fill[6] == pytest.approx(0.30, abs=0.05)
    assert pre_fill[1] == pytest.approx(0.85, abs=0.05)
    assert (m.confirmed_qty <= m.actual).all()


def test_seed_is_deterministic(frames):
    again = generate(P)
    assert again["actuals"].equals(frames["actuals"])


def test_write_outputs(frames, tmp_path):
    write_outputs(frames, tmp_path)
    for name in ["actuals.parquet", "customer_forecasts.csv", "order_book.csv"]:
        assert (tmp_path / name).exists()
