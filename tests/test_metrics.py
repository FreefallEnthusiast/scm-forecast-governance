import numpy as np
import pandas as pd
import pytest

from fgov import metrics, synth


def panel(values, customer="C1", family="F01"):
    return pd.DataFrame({"customer": customer, "family": family,
                         "month": np.arange(1, len(values) + 1), "actual": values})


def fc_rows(actuals, factor, method="m", origin=None, lo=None, hi=None):
    """Forecast = factor x actual for every month, all with one origin and horizon 1."""
    a = actuals
    return pd.DataFrame({
        "origin_month": a["month"] - 1 if origin is None else origin,
        "target_month": a["month"], "horizon": 1,
        "customer": a["customer"], "family": a["family"], "method": method,
        "forecast": a["actual"] * factor,
        "lo80": a["actual"] * lo if lo is not None else np.nan,
        "hi80": a["actual"] * hi if hi is not None else np.nan})


RNG = np.random.default_rng(1)
VALUES = RNG.uniform(50, 150, 24).round()


def test_mase_scale_matches_hand_value():
    s = metrics.mase_scale(panel(VALUES), origin=24)
    assert s["scale"].iloc[0] == pytest.approx(np.abs(VALUES[12:] - VALUES[:-12]).mean())


def test_mase_scale_fallback_to_lag1():
    s = metrics.mase_scale(panel(VALUES[:10]), origin=10)
    assert s["scale"].iloc[0] == pytest.approx(np.abs(np.diff(VALUES[:10])).mean())


def test_mase_of_seasonal_naive_against_itself_is_one():
    # Scaling window is months 1..24 (origin 24), whose seasonal differences are
    # y_t - y_{t-12}, t = 13..24. We evaluate seasonal-naive forecasts for exactly
    # those targets, all stamped with origin 24 (horizon label is irrelevant to the
    # maths), so in-sample and evaluated errors are identical and MASE is 1.
    act = panel(VALUES)
    t = np.arange(13, 25)
    f = pd.DataFrame({"origin_month": 24, "target_month": t, "horizon": 1,
                      "customer": "C1", "family": "F01", "method": "seasonal_naive",
                      "forecast": VALUES[t - 13], "lo80": np.nan, "hi80": np.nan})
    s = metrics.summarise(metrics.evaluate(f, act))
    assert s.loc[s["period"] == "all", "mase"].iloc[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("factor,expected", [(1.2, 0.2), (0.8, -0.2)])
def test_bias_sign(factor, expected):
    act = panel(VALUES)
    s = metrics.summarise(metrics.evaluate(fc_rows(act, factor), act))
    assert s.loc[s["period"] == "all", "bias"].iloc[0] == pytest.approx(expected)


def test_fva_zero_for_identical_forecasts():
    act = panel(VALUES)
    f = pd.concat([fc_rows(act, 1.1, "seasonal_naive"), fc_rows(act, 1.1, "stat_baseline"),
                   fc_rows(act, 1.1, "blend")])
    out = metrics.fva(metrics.summarise(metrics.evaluate(f, act)))
    assert out["fva_vs_naive"].abs().max() < 1e-9
    assert out["fva_vs_baseline"].abs().max() < 1e-9


def test_fva_sign_positive_when_better():
    act = panel(VALUES)
    f = pd.concat([fc_rows(act, 1.5, "seasonal_naive"), fc_rows(act, 1.1, "blend")])
    out = metrics.fva(metrics.summarise(metrics.evaluate(f, act)))
    row = out[(out["method"] == "blend") & (out["period"] == "all")].iloc[0]
    assert row["fva_vs_naive"] == pytest.approx(100 * (1 - 0.1 / 0.5))


def test_coverage_always_and_never():
    act = panel(VALUES)
    inside = metrics.summarise(metrics.evaluate(fc_rows(act, 1.0, lo=0.9, hi=1.1), act))
    outside = metrics.summarise(metrics.evaluate(fc_rows(act, 1.0, lo=1.1, hi=1.2), act))
    missing = metrics.summarise(metrics.evaluate(fc_rows(act, 1.0), act))
    assert inside["coverage80"].eq(1.0).all()
    assert outside["coverage80"].eq(0.0).all()
    assert missing["coverage80"].isna().all()


def test_period_split_uses_origin():
    act = panel(VALUES)
    ev = metrics.evaluate(fc_rows(act, 1.0), act, break_month=13)
    assert set(ev.loc[ev["origin_month"] < 13, "period"]) == {"pre"}
    assert set(ev.loc[ev["origin_month"] >= 13, "period"]) == {"post"}


def stability_frame(values_by_origin):
    rows = [{"origin_month": o, "target_month": 10, "horizon": 10 - o, "customer": "C1",
             "family": "F01", "method": "m", "forecast": v, "lo80": np.nan, "hi80": np.nan}
            for o, v in values_by_origin.items()]
    return pd.DataFrame(rows)


def test_stability_constant_is_zero_and_known_revision():
    act = panel([100.0] * 12)
    flat = metrics.stability(stability_frame({4: 90.0, 7: 90.0, 9: 90.0}), act)
    assert flat["stability"].eq(0).all()
    # Revisions 10 and 20 on a series with mean actual 100 -> mean 0.15.
    rev = metrics.stability(stability_frame({4: 90.0, 7: 100.0, 9: 80.0}), act)
    assert rev.loc[rev["period"] == "all", "stability"].iloc[0] == pytest.approx(0.15)


def test_customer_bias_trailing_recovers_ratio_and_ignores_future():
    act = pd.concat([panel([100.0] * 30, "C1", "F01"), panel([100.0] * 30, "C1", "F02"),
                     panel([100.0] * 30, "C2", "F01")])
    t = np.arange(10, 31)
    cf = pd.DataFrame({"origin_month": t - 3, "target_month": t, "horizon": 3,
                       "customer": "C1", "family": "F01",
                       "customer_forecast": np.where(t <= 20, 130.0, 1e6)})
    cf2 = cf.assign(family="F02", customer_forecast=np.where(t <= 20, 100.0, 1e6))
    cf3 = cf.assign(customer="C2", customer_forecast=np.where(t <= 20, 70.0, 1e6))
    cf = pd.concat([cf, cf2, cf3])
    by_c = metrics.customer_bias_trailing(cf, act, origin=20, horizon=3).set_index("customer")
    assert by_c.loc["C1", "bias_ratio"] == pytest.approx(1.15)
    assert by_c.loc["C2", "bias_ratio"] == pytest.approx(0.7)
    by_cf = metrics.customer_bias_trailing(cf, act, 20, 3, by_family=True)
    got = by_cf.set_index(["customer", "family"])["bias_ratio"]
    assert got[("C1", "F01")] == pytest.approx(1.3)
    assert got[("C1", "F02")] == pytest.approx(1.0)
    assert metrics.customer_bias_trailing(cf, act, 20, 1).empty


def test_integration_with_synth():
    frames = synth.generate()
    act = frames["actuals"]
    lookup = act.set_index(["customer", "family", "month"])["actual"]
    rows = act[(act["month"] >= 26) & (act["month"] <= 49)][["customer", "family", "month"]]
    rows = rows.rename(columns={"month": "target_month"})
    rows["origin_month"] = rows["target_month"] - 1
    rows["horizon"] = 1
    rows["forecast"] = lookup.reindex(
        pd.MultiIndex.from_arrays([rows["customer"], rows["family"], rows["target_month"] - 12])).to_numpy()
    rows = rows[rows["forecast"].notna()].assign(method="seasonal_naive", lo80=np.nan, hi80=np.nan)
    ev = metrics.evaluate(rows, act)
    by_family = metrics.summarise(ev, by=("family",))
    pre = by_family[(by_family["period"] == "pre") & by_family["family"].isin(
        [f"F{i:02d}" for i in range(1, 7)])]
    assert np.isfinite(pre["mase"]).all()
    assert pre["mase"].between(0.6, 1.6).all()
    overall = metrics.summarise(ev)
    assert np.isfinite(overall["mase"]).all()

    cb = metrics.customer_bias_trailing(frames["customer_forecasts"], act, 36, 6)
    cb = cb.set_index("customer")["bias_ratio"]
    assert cb["C2"] == pytest.approx(1.3, abs=0.1)
    assert cb["C3"] == pytest.approx(0.85, abs=0.1)
