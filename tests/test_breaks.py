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
    assert not breaks.history_unreliable(alerts, "F11", ONSET + breaks.NEW_FAMILY_HISTORY)


def test_constructed_step_detected_at_month_30():
    rng = np.random.default_rng(0)
    months = np.arange(1, 41)
    values = np.where(months >= 30, 200.0, 100.0) * rng.lognormal(0, 0.05, len(months))
    bm = breaks.detect_at_origin(pd.Series(values, index=months), 40, breaks.DEFAULT_PENALTY)
    assert bm is not None and abs(bm - 30) <= 1
