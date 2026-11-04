import numpy as np
import pandas as pd
import pytest

from fgov import breaks
from fgov.synth import SynthParams, generate

ONSET = SynthParams().step_onset


@pytest.fixture(scope="module")
def actuals():
    return generate()["actuals"]


@pytest.fixture(scope="module")
def alerts(actuals):
    return breaks.alert_table(actuals, onset=ONSET)


def test_penalty_constant_matches_procedure(actuals):
    assert breaks.choose_penalty(actuals) == breaks.DEFAULT_PENALTY


def test_total_break_detected_within_4_months(alerts):
    at = alerts.set_index("family").loc["TOTAL", "break_detected_at_origin"]
    assert ONSET <= at <= ONSET + 3


def test_family_detection_count(alerts):
    est = alerts[(alerts.family != "TOTAL") & (alerts.reason != "new_family")]
    assert len(est) == 10
    within = (est.break_detected_at_origin <= ONSET + 4).sum()
    # Honest count with the pre-break-chosen penalty: 7 of 10 established
    # families within 4 origins (F07 and F10 lag at 6-7 months, F08 with a
    # 1.3x step is never detected). The spec target was 8 of 10; the penalty
    # was not tuned on the step to reach it. With the 2 new families flagged
    # by rule that is 9 of 12.
    assert within >= 7
    new = alerts[alerts.reason == "new_family"]
    assert set(new.family) == {"F11", "F12"}


def test_no_false_alarm_on_total_before_onset(actuals):
    sc = breaks.scan(actuals, range(24, ONSET), level="total")
    assert (sc.break_month.fillna(0) >= 20).sum() == 0


def test_history_unreliable(alerts):
    at = int(alerts.set_index("family").loc["F02", "break_detected_at_origin"])
    assert not breaks.history_unreliable(alerts, "F02", at - 1)
    assert breaks.history_unreliable(alerts, "F02", at)
    assert breaks.history_unreliable(alerts, "F02", 60)
    assert breaks.history_unreliable(alerts, "F11", ONSET)
    assert breaks.history_unreliable(alerts, "F11", ONSET + breaks.NEW_FAMILY_HISTORY - 1)
    assert breaks.NEW_FAMILY_HISTORY == 12
    assert not breaks.history_unreliable(alerts, "F11", ONSET + breaks.NEW_FAMILY_HISTORY)


def test_constructed_step_detected_at_month_30():
    rng = np.random.default_rng(0)
    months = np.arange(1, 41)
    values = np.where(months >= 30, 200.0, 100.0) * rng.lognormal(0, 0.05, len(months))
    bm = breaks.detect_at_origin(pd.Series(values, index=months), 40, breaks.DEFAULT_PENALTY)
    assert bm is not None and abs(bm - 30) <= 1


# --- Tests added from mutation testing -------------------------------------------------


def _family_series(actuals):
    total = actuals.groupby("month")["actual"].sum(min_count=0)
    fam = actuals.groupby(["family", "month"])["actual"].sum(min_count=1)
    return {"TOTAL": total, **{f: fam.loc[f] for f in fam.index.get_level_values(0).unique()}}


def test_alert_table_levels_lags_and_reasons(actuals):
    # A shorter scan (origins 24..42) so the level windows must stop at the last origin and
    # families first caught later (F07 at 43, F10 at 44) must read as not detected.
    last = 42
    al = breaks.alert_table(actuals, onset=ONSET, origins=range(24, last + 1)).set_index("family")
    series = _family_series(actuals)
    assert set(al.index) == set(series)
    assert set(al.index[al.reason.isna()]) == {"F07", "F08", "F10"}
    pen = breaks.DEFAULT_PENALTY
    for name, full in series.items():
        s, row = full[full.index <= last], al.loc[name]
        if name in ("F11", "F12"):
            assert row.reason == "new_family"
            assert row.break_detected_at_origin == row.break_month == ONSET
            assert row.months_after_onset == 0 and np.isnan(row.pre_break_level)
            assert row.post_break_level == pytest.approx(s.loc[ONSET:].mean())
            continue
        if name in ("F07", "F08", "F10"):          # not detected by origin 42 (F08 never)
            assert pd.isna(row.reason)
            assert row[["break_detected_at_origin", "break_month", "months_after_onset",
                        "post_break_level"]].isna().all()
            continue
        at = int(row.break_detected_at_origin)
        assert row.reason == "detected"
        assert row.months_after_onset == at - ONSET
        assert breaks.detect_at_origin(full, at, pen) >= ONSET - 1
        before = breaks.detect_at_origin(full, at - 1, pen)
        assert before is None or before < ONSET - 1
        assert row.break_month == breaks.detect_at_origin(full, at, pen)
        assert row.pre_break_level == pytest.approx(s.loc[ONSET - 12:ONSET - 1].mean())
        assert row.post_break_level == pytest.approx(s.loc[int(row.break_month):].mean())


def test_history_unreliable_false_when_unknown_or_never_detected(alerts):
    assert not breaks.history_unreliable(alerts, "F99", 60)
    assert pd.isna(alerts.set_index("family").loc["F08", "break_detected_at_origin"])
    assert not breaks.history_unreliable(alerts, "F08", 60)


def test_false_alarms_counts_recent_breaks_before_onset_only():
    sc = pd.DataFrame({"name": "A", "origin": [30, 31, 32, 33, 36, 37],
                       "break_month": pd.array([19, 20, 21, None, 25, 25], dtype="Int64")})
    # 20, 21 and 25 are recent (>= 20); 19 is not; NaN is not; origin 37 is not pre-onset.
    assert breaks.false_alarms(sc, onset=37)["A"] == 3


def _grid_step(scale=1.0):
    # PELT's default jump of 5 only places breaks at positions 5, 10, ...; position 30
    # is month 31, so a step at month 31 can be located exactly.
    months = np.arange(1, 41)
    rng = np.random.default_rng(0)
    values = np.where(months >= 31, 200.0, 100.0) * rng.lognormal(0, 0.05, len(months))
    return pd.Series(scale * values, index=months)


def test_step_detected_at_earliest_origin_using_the_origin_month():
    s = _grid_step()
    # min_size 3: the new regime needs months 31, 32 and 33, so origin 33 is the first hit.
    assert breaks.detect_at_origin(s, 32, breaks.DEFAULT_PENALTY) is None
    assert breaks.detect_at_origin(s, 33, breaks.DEFAULT_PENALTY) == 31


def test_detection_is_scale_free_and_zero_series_is_none():
    for scale in (0.001, 1000.0):
        assert breaks.detect_at_origin(_grid_step(scale), 33, breaks.DEFAULT_PENALTY) == 31
    zeros = pd.Series(0.0, index=np.arange(1, 41))
    assert breaks.detect_at_origin(zeros, 40, breaks.DEFAULT_PENALTY) is None


def test_choose_penalty_falls_back_to_largest_grid_value(actuals):
    # Penalties this small raise false alarms everywhere, so none qualifies.
    assert breaks.choose_penalty(actuals, grid=(0.05, 0.1, 0.15)) == 0.15


def test_onset_constant_and_default_scan_reach_month_60(actuals, alerts):
    assert breaks.ONSET == ONSET            # report.py uses breaks.ONSET as its default period split
    total = actuals.groupby("month")["actual"].sum()
    row = alerts.set_index("family").loc["TOTAL"]
    assert row.post_break_level == pytest.approx(total.loc[int(row.break_month):60].mean())
