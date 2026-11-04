"""Blend value-added probe: error correlations, a leakage check, and the blend without the order book.
Run from the repo root: python scripts/blend_probe.py (about 3 minutes). Reporting only
nothing is written to outputs.
"""
import numpy as np
import pandas as pd
import warnings
from fgov import synth, breaks, forecast, metrics
warnings.simplefilter("ignore")
pd.set_option("display.width", 200)
fr = synth.generate()
a = fr["actuals"]
alerts = breaks.alert_table(a)
H = (1, 3, 6)
origins = list(range(25, 61))
run_origins = list(range(13, 61))
flags = a[["family", "is_intermittent"]].drop_duplicates()
def fva_table(fc, by=()):
    ev = metrics.evaluate(fc, a).merge(flags, on="family")
    s = metrics.fva(metrics.summarise(ev, by=by))
    return ev, s
print("=== (a) pairwise error correlation at h=3, errors scaled by series mean actual ===")
full, ch_full = forecast.run_all(fr, alerts)
ev, s = fva_table(full)
lvl = a.groupby(["customer", "family"]).actual.mean().rename("lvl").reset_index()
e3 = ev[(ev.horizon == 3) & ev.method.isin(forecast.CANDIDATES)].merge(lvl, on=["customer", "family"])
e3["scaled"] = e3.error / e3.lvl
for period in ("pre", "post"):
    w = e3[e3.period == period].pivot_table(index=["origin_month", "target_month", "customer", "family"], columns="method", values="scaled")
    print(period, "n rows", len(w))
    print(w.corr().round(2).to_string())
    for typ, lab in ((False, "smooth"), (True, "intermittent")):
        ww = w.reset_index().merge(flags, on="family")
        ww = ww[ww.is_intermittent == typ]
        print(f"  {period} {lab}:")
        print(ww[list(forecast.CANDIDATES)].corr().round(2).to_string())
print("\n=== (c) FVA vs naive by horizon, blend with and without the order book ===")
base = full[full.method != "blend"]
def rebuild(cands):
    forecast.CANDIDATES = cands
    parts = [forecast.seasonal_naive(a, run_origins, H, 60), forecast.stat_baseline(a, run_origins, H, 60, 4),
             forecast.customer_raw(fr["customer_forecasts"], run_origins, H, 60),
             forecast.customer_bias_corrected(fr["customer_forecasts"], a, run_origins, H, 60),
             forecast.order_book(fr["order_book"], a, run_origins, H, 60)]
    rows = pd.concat(parts, ignore_index=True)
    rows = pd.concat([rows, forecast.average_candidate(rows)], ignore_index=True)
    rows = forecast.fill_intervals(rows, a)
    blended, choices = forecast.blend(rows[rows.method.isin(cands + ["average"])], a, alerts, origins)
    out = pd.concat([rows[rows.method != "average"], blended], ignore_index=True)
    return out[out.origin_month.isin(origins)], choices
nobook, ch_nobook = rebuild(["stat_baseline", "customer_bias_corrected"])
forecast.CANDIDATES = ["stat_baseline", "customer_bias_corrected", "order_book"]
ev_nb, s_nb = fva_table(nobook)
def show(s, label):
    t = s[s.period.isin(["pre", "post"]) & s.method.isin(["blend", "order_book", "customer_bias_corrected", "stat_baseline"])]
    print(label)
    print(t.pivot_table(index="method", columns=["horizon", "period"], values="fva_vs_naive").round(1).to_string())
show(s, "with order book:")
show(s_nb, "without order book (average = stat + bias-corrected):")
print("choices with book:", ch_full.chosen.value_counts().to_dict())
print("choices without:", ch_nobook.chosen.value_counts().to_dict())
print("\n=== FVA vs naive at h=3 by family type (with order book) ===")
_, s_t = fva_table(full, by=("is_intermittent",))
t = s_t[(s_t.horizon == 3) & s_t.period.isin(["pre", "post"]) & (s_t.method != "customer_raw")]
print(t.pivot_table(index="method", columns=["is_intermittent", "period"], values=["mase", "fva_vs_naive"]).round(2).to_string())
_, s_tn = fva_table(nobook, by=("is_intermittent",))
tn = s_tn[(s_tn.horizon == 3) & s_tn.period.isin(["pre", "post"]) & (s_tn.method == "blend")]
print("blend without book by type:")
print(tn.pivot_table(index="method", columns=["is_intermittent", "period"], values="fva_vs_naive").round(1).to_string())
# does average beat its parts on smooth families? pooled MASE on smooth at h=3 for the 'average' candidate itself
print("\n=== (b) leakage: perturb actuals after month 45, rerun, compare everything at origins <= 45 ===")
rng = np.random.default_rng(1)
a2 = a.copy()
m = a2.month > 45
a2.loc[m, "actual"] = (a2.loc[m, "actual"] * rng.uniform(0.3, 3.0, m.sum())).round()
fr2 = dict(fr)
fr2["actuals"] = a2
alerts2 = breaks.alert_table(a2)
det = alerts.set_index("family").break_detected_at_origin
det2 = alerts2.set_index("family").break_detected_at_origin
print("detections at origin <= 45 identical:", det[det <= 45].equals(det2[det2 <= 45]))
full2, ch2 = forecast.run_all(fr2, alerts2)
k = ["method", "origin_month", "target_month", "horizon", "customer", "family"]
x = full[full.origin_month <= 45].sort_values(k).reset_index(drop=True)
y = full2[full2.origin_month <= 45].sort_values(k).reset_index(drop=True)
same_keys = x[k].equals(y[k])
close = all(np.allclose(x[c].fillna(-1), y[c].fillna(-1)) for c in ("forecast", "lo80", "hi80"))
print("rows identical (keys):", same_keys, "| forecasts and intervals identical:", close, "| n rows", len(x))
c1 = ch_full[ch_full.origin_month <= 45].sort_values(["origin_month", "family"]).reset_index(drop=True)
c2 = ch2[ch2.origin_month <= 45].sort_values(["origin_month", "family"]).reset_index(drop=True)
print("blend choices at origins <= 45 identical:", c1.equals(c2))
diff_after = (full2[full2.origin_month > 45].sort_values(k).forecast.to_numpy() != full[full.origin_month > 45].sort_values(k).forecast.to_numpy()).mean()
print("share of forecasts that changed at origins > 45 (sanity, should be > 0):", round(diff_after, 2))
