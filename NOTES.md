# Build notes

Judgement calls and open questions for the owner to review.

## Judgement calls

### Synthetic panel (`src/fgov/synth.py`)

- **Month index.** Months are integers 1..60 (1..48 history, 49..60 holdout). A cosmetic `month_start` date column (from 2019-01-01) is also written to `actuals.parquet`; nothing downstream should depend on the calendar.
- **Family layout.** F01-F06 smooth, F07-F10 intermittent, F11-F12 new. Long-lead means F01-F04 (the first four smooth families); they see the step one month earlier (onset 36 instead of 37). Intermittent and new families are never long-lead.
- **New families pre-launch.** F11 and F12 launch at month 37 (the first of the last 12 history months, which coincides with the step onset). Before launch their `actual`, `customer_forecast` and `confirmed_qty` are NaN, not zero: the product did not exist, so a zero would be a false "no demand" observation. The rows are kept so every table is a full rectangular grid; downstream code must drop NaN rows (statsforecast will not accept them). New families do not step themselves: they launch at their post-step level.
- **Step factor rescaling.** The total factor T is drawn from U(1.8, 2.2). Per-family factors are drawn from U(1.3, 2.8), then multiplied by a common constant and clipped to [1.3, 2.8], iterated until the base-weighted average over established families, plus the launch volume of the two new families, equals T times the pre-step total. So "total demand rises by T" includes the new families' volume. With the default seed T = 1.849 and two families (F01, F08) end up clipped at 1.3.
- **Step shape.** Factor ramps linearly: 1/3 of the increase in the onset month, 2/3 the next, full from the third month, then flat to month 60. The step multiplies the expected level; for intermittent families it scales order size, not occurrence, so zero share is unchanged by the upcycle.
- **Measured step.** Mean total demand months 43-48 over months 25-36 is 1.66 against the drawn 1.85 with the default seed. The gap is mostly seasonality (months 43-48 sit in the trough of the annual sine, about -6%) plus intermittent noise. The test allows +/-25%.
- **Demand process.** Expected total 20,000 units/month pre-step. Family sizes are lognormal (sigma 0.4); intermittent families are scaled to 0.4x and new to 0.35x a smooth family. Customer mix across families varies lognormally (sigma 0.3) around the customer shares 30/25/20/15/10%. Seasonality is one shared annual sine, amplitude 10%. Smooth noise is mean-one lognormal, sigma 0.15. Actuals are rounded to whole units.
- **Intermittent demand.** Occurrence p = 0.45 (expected zero share 55%); size is mean-one lognormal, sigma 0.6, divided by p so the expected value equals the level. Size draws are truncated at +/-3 sigma: one untruncated draw produced a single order bigger than the whole rest of the month's demand, which swamped every per-customer ratio. With the default seed the training-window zero share per family is 0.53-0.57; per customer x family series it ranges 0.42-0.65, so a few series in other seeds can fall just under the 40% classification threshold used in phase 2.
- **Customer forecasts.** forecast = signal x bias(customer, horizon) x mean-one lognormal noise(customer, horizon), rounded. Noise is drawn independently for every origin, which is what makes C4's revisions large. The signal is the realised actual for smooth and new families, but only the expected level for intermittent families (a customer cannot know in advance which month a lumpy order lands). Bias per horizon (1, 3, 6): C1 1.00/1.00/1.00, C2 1.05/1.15/1.30, C3 0.85 flat, C4 1.00, C5 1.00. Noise sigma per horizon: C1 0.04/0.06/0.08, C2 and C3 0.06/0.08/0.10, C4 0.30/0.35/0.40, C5 0.05/0.07/0.09.
- **C5 upcycle.** For targets at or after month 37 C5's forecasts get an extra multiplier of 1.35/1.50/1.60 at horizons 1/3/6 (panic is worst furthest out). It is applied by target month, not origin, and to all families including long-lead ones.
- **Order book.** For as-of month a and h = 1..6 months ahead, confirmed = floor(actual x clip(fill(h) x noise, 0, 1)), so it never exceeds the eventual actual. fill(h) runs linearly 0.85, 0.74, 0.63, 0.52, 0.41, 0.30; targets from month 37 get +0.10. Noise is lognormal sigma 0.10. Only as-of months >= 1 are generated, like the forecasts.
- **Extra output.** `families.csv` holds the family flags, the per-family step factor and the drawn total factor, so later phases and the report can quote them without re-deriving.
- **Test robustness.** All tests pass for the default seed. Across seeds 0-29, the pre-break bias test fails for 3 seeds (10, 15, 23), always because realised intermittent demand ran well above or below its expectation and moved a pooled customer ratio by more than the tolerance (e.g. C2 at h=6 = 1.21). The tolerances were not widened to cover them.
- **Tooling.** The `pytest` first on PATH here (`/root/.local/bin/pytest`) is an isolated tool install without numpy; use `python -m pytest`.

### Review of the synthetic panel (lead session)

- **Intermittent families dominate month-to-month noise in the total.** Their combined monthly total swings roughly 3k to 13k against a smooth-family total of about 13k pre-step, so the total series has a coefficient of variation near 13% before the step. The step to about 37k is still unmistakable, but per-family break detection on F07-F10 will lag and may miss F08 (factor only 1.3). Accepted as the honest lumpy case; phase 4 should report misses rather than tune them away.
- **Pooled customer bias carries noise from intermittent families.** Post-break (24 months only), C1 reads 0.94 and C4 reads 1.08 overall although both are 1.00 on smooth families; the gap is realised lumpy demand running away from the expected level the customer forecasts against. Within the 20% attention threshold, so accepted. The report should state bias by customer on smooth families as well as overall if the chart looks muddled.

### Metrics (`src/fgov/metrics.py`)

- **MASE scale** is the in-sample seasonal-naive MAE on months 1..origin per customer x family, recomputed at every origin. Series with fewer than 13 points (the new families at every origin) fall back to the lag-1 MAE. A zero or undefined scale becomes NaN and those rows drop out of the means.
- **Pooling.** `summarise` pools rows across origins rather than averaging per origin. Bias is the sum of errors over the sum of actuals. `period` is set by the origin month (`pre` if origin < 37), and an `all` copy is added.
- **Coverage** is NaN for methods without intervals rather than zero.
- **FVA** is joined on horizon, period and any extra grouping columns; the two reference methods score zero against themselves.
- **Stability** takes revisions between consecutive available origins for the same target (so a target seen at origins t-6, t-3, t-1 yields two revisions, crossing horizons), scaled by the series' whole-history mean actual, which includes the upcycle level. Reported by method and period only.
- **Trailing bias window** is target months in (origin - 12, origin] at the given horizon, so only realised months are used.
