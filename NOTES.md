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

### Review of the synthetic panel

- **Intermittent families dominate month-to-month noise in the total.** Their combined monthly total swings roughly 3k to 13k against a smooth-family total of about 13k pre-step, so the total series has a coefficient of variation near 13% before the step. The step to about 37k is still unmistakable, but per-family break detection on F07-F10 will lag and may miss F08 (factor only 1.3). Accepted as the honest lumpy case; phase 4 should report misses rather than tune them away.
- **Pooled customer bias carries noise from intermittent families.** Post-break (24 months only), C1 reads 0.94 and C4 reads 1.08 overall although both are 1.00 on smooth families; the gap is realised lumpy demand running away from the expected level the customer forecasts against. Within the 20% attention threshold, so accepted. The report should state bias by customer on smooth families as well as overall if the chart looks muddled.

### Metrics (`src/fgov/metrics.py`)

- **MASE scale** is the in-sample seasonal-naive MAE on months 1..origin per customer x family, recomputed at every origin. Series with fewer than 13 points (the new families at every origin) fall back to the lag-1 MAE. A zero or undefined scale becomes NaN and those rows drop out of the means.
- **Pooling.** `summarise` pools rows across origins rather than averaging per origin. Bias is the sum of errors over the sum of actuals. `period` is set by the origin month (`pre` if origin < 37), and an `all` copy is added.
- **Coverage** is NaN for methods without intervals rather than zero.
- **FVA** is joined on horizon, period and any extra grouping columns; the two reference methods score zero against themselves.
- **Stability** takes revisions between consecutive available origins for the same target (so a target seen at origins t-6, t-3, t-1 yields two revisions, crossing horizons), scaled by the series' whole-history mean actual, which includes the upcycle level. Reported by method and period only.
- **Trailing bias window** is target months in (origin - 12, origin] at the given horizon, so only realised months are used.

### Break detection (`src/fgov/breaks.py`)

- **Spec deviation, for the owner.** The spec expects the step detected within 4 origins in at least 10 of 12 families. The honest result on the default seed is 7 of 10 established families plus the 2 new families flagged by rule, so 9 of 12. F08 (step factor only 1.3 under lumpy demand) is never detected; F07 and F10 are 6 and 7 months late; F05 and F09 are right at the limit. The test asserts the honest count of 7 with a comment. The penalty was not tuned on the step; see next point. Options for the owner: accept and say so in the report (recommended, it is the point of the lumpy case), or raise the intermittent step factors in the synth parameters.
- **Penalty procedure.** PELT penalty 2.0 chosen as the smallest grid value with zero pre-onset false alarms on the total and on at least 5 of the 6 smooth families, scanning origins 24..36 only. No post-step data is used. F01 has one allowed pre-onset false alarm (origin 31, break at month 26).
- **Earliest possible detection** is two origins after the break month, because PELT with min_size 3 cannot place a break in the last two points. Every detected break lands on month 36 (long-lead families step early and the ramp pulls the change point one month early for the rest).
- **Flags are sticky.** The raw scan can lose a break at later origins (F09 after origin 43); `history_unreliable` never clears once set, as specified. New families are unreliable from launch (month 37) for 12 months by rule.
- **Levels.** Pre-break level is the mean over months 25..36; post-break level runs from the detected break month to month 60, so it includes the ramp. TOTAL counts new-family NaN as zero.
- **Other seeds (information only).** Across seeds 1..8 the total is detected at origins 39..42 and only 4 to 6 established families within 4 origins. The default seed is a good case and the family-count test is seed-specific.

### Quality gate

- **Tools.** `ruff check .` (lint, bug patterns, naming, McCabe complexity 10, max 50 statements and 12 branches per function) and `pyright src` (static typing as a language server would report it) must both be clean before a commit. Config lives in `pyproject.toml`. No AST-rewriting or graph tooling: the package is seven files with a fixed one-way layering (synth, then forecast/metrics/breaks, then report, then run) and every module is read in review before it is committed.
- **Synth refactor.** `generate` was over the statement limit; the customer-forecast and order-book blocks became `customer_forecasts` and `order_book` with a shared `long_table` helper. Random draws happen in the same order, and all four output frames hash identically to before.

### Candidate forecasts (`src/fgov/forecast.py`)

- **Warm-up origins.** Every method also runs for the 12 origins before the first requested one (13..24 for the full run) so empirical intervals and the blend have history at origin 25; those rows are dropped from the output.
- **Stat baseline groups at each origin.** Under 13 points: mean of available history with empirical intervals. 13 to 24 points: TSB (alpha_d = alpha_p = 0.2) or AutoETS(season_length=12) without conformal intervals. 25 or more: conformal with n_windows=2, h=6. One `StatsForecast.forecast` call per group per origin; `n_jobs=4` brings the 36 origins from 116 s to 38 s. AutoETS is unrestricted.
- **Empirical intervals** use residuals from the last 12 origins whose targets were realised by the origin, pooled per method x horizon x family. Where all residuals share a sign the interval is widened to contain the point forecast, then clipped at zero. Conformal intervals are kept where present.
- **Blend** scores stat baseline, bias-corrected customer, order book and their row-wise average by pooled scaled error over the previous 12 origins (6 when `history_unreliable`), all horizons pooled; fewer than 3 evaluable origins picks "average"; ties go to the first in that order. Ratios for bias correction and order-book fill are per customer, pooled over families; zero or NaN falls back to 1.0.
- **Rows.** Customer and order-book methods have 100 more rows than the history-based ones: the new families at origin 36 have forecasts for target 37 but no actual history.
- **Seasonal naive MASE is about 1.5, not 1.** Out of sample its errors are |y_t - y_{t-12}| like the in-sample scale, but pre-break origins forecast into the step and intermittent series are lumpier out of sample. In-sample equality is what the metrics test checks.

### Order book made honest (after the first forecast run)

- The first run showed the order-book method 66 to 74% better than seasonal naive at every horizon, above the spec's implausibility line. Cause: the generator revealed each month's demand six months ahead with 10% noise and revealed lumpy-order timing perfectly. Changed to horizon-dependent noise (lognormal sigma 0.10 at one month to 0.50 at six) and, for intermittent families, a probability that a lumpy order is not on the book yet (95% visible at one month, 50% at six). The fill test now measures the fill parameter on smooth families, where it applies directly.
- After the change the order book is strong one month out (FVA about 70%, where 88% of demand is already confirmed, which is real), modest at three months (10 to 20%) and worse than naive at six. The blend now picks between sources (average 239, order book 144, bias-corrected customer 48, stat baseline 1 of 432 choices) instead of always taking the book. Blend FVA at one month stays above 60%; that is the confirmed order book doing the work and the report says so.
- Stat baseline 80% coverage is 0.56 to 0.58: conformal with two windows under-covers by construction, and pre-break origins 31..36 forecast into the step. Left as found; it is part of the history-unreliable story.

### Report and runner (`src/fgov/report.py`, `src/fgov/run.py`)

- **Planner origin** is the latest origin with horizon-1 blend rows (59 in the full run), target 60; the CSV carries `origin_month` and `target_month`. Rows come from the full customer x family grid so the table always has 60 rows.
- **Attention triggers:** trailing customer bias beyond 20%, recommended source different from the previous origin's choice, or the family's break detected at exactly this origin. `attention_reason` lists which fired.
- **Charts.** One colour per method across all charts; the pre/post split uses two greys. The break line is the PELT break month at the detection origin (36); the shaded lag runs from onset to detection. The tracking family is the long-lead family with the largest step factor (F04).
- **Runner** deletes only the files it writes, not the output folder.

### Independent code review (after the first full run)

A separate review pass over `src/fgov` and `tests` returned ten findings; nine were applied, one skipped.

- **Applied.** The step onset is now threaded from `SynthParams.step_onset` through `alert_table`, `evaluate`, `stability` and the report (it was hardcoded as 37 in three places; a test now changes it to 40 and checks the labels follow). `alert_table` emits `break_month` so the chart and the alert can no longer disagree (the report used to re-run PELT with the default penalty). NaN guards: a new family not yet launched by the last scanned origin, and a planner table with no horizon-1 rows. A zero trailing bias ratio now falls back to the raw forecast instead of producing infinity. The report never prints "nan". The worst customer, best post-break method and calmest and jumpiest methods are derived once in `build_tables` and shared by chart titles and text. The total demand series and the series keys are defined once. The tracking chart window follows the origins passed in. Outputs on the default seed were byte-identical before and after.
- **Skipped.** `choose_penalty` refits PELT for every grid value although only the penalty changes; it costs about a second in the tests and is not worth the extra code.
- **Package size** is unchanged at about 1,240 lines in `src/fgov`.

### Robustness across seeds (`scripts/seed_study.py`)

Twenty seeds (0..19) through the full pipeline; results in `scripts/seed_study_20seeds.csv`. Median, min and max:

| quantity | median | min | max |
|---|---|---|---|
| Drawn total step factor | 1.97 | 1.80 | 2.18 |
| Total-series detection lag (months) | 3 | 1 | 4 |
| Established families detected within 4 months (of 10) | 6 | 4 | 8 |
| Established families never detected | 1 | 0 | 3 |
| C2 raw bias, 6 months, pre-break | +31% | +21% | +46% |
| C3 raw bias, 6 months, pre-break | -13% | -23% | -3% |
| C5 raw bias, 6 months, post-break | +61% | +49% | +80% |
| C5 corrected bias, 6 months, post-break | +13% | 0% | +21% |
| Blend FVA vs naive, 3 months, post-break (pp) | 63 | 55 | 68 |
| Stat baseline FVA vs naive, 3 months, post-break (pp) | 25 | 18 | 27 |
| Blend 80% coverage, 3 months | 0.73 | 0.71 | 0.74 |
| Stat baseline 80% coverage, 3 months | 0.57 | 0.55 | 0.60 |

- The customer-bias and value-added findings are stable. The total break is always flagged within four months. Per-family detection within four months never reaches the spec's 10 of 12 on any seed (best is 8 of 10 established plus the 2 new families by rule); the spec's expectation is too optimistic for lumpy families at these step sizes and penalties chosen without hindsight. The default seed, at 7 of 10, is slightly better than the median.
- The blend's one-month FVA stays above 60% on every seed because the confirmed order book carries it; the three-month figure sits at 55 to 68.
- Stat baseline interval coverage is 0.55 to 0.60 on every seed: the two-window conformal intervals under-cover by construction, which the report states.

## Owner review round

### 1. Repository scrub

- `PROPOSAL.md` (an internal build plan) is removed from the working tree and purged from history, so the repository reads as a reference implementation. References to it, and process narrative in these notes, are removed. Commit trailers no longer name a model.

### 2. Blend FVA investigation

Question: the blend adds about +60 pp over seasonal naive at 3 months, pre- and post-break, while its best single input (bias-corrected customer forecast) adds +33 pp post-break. Three checks on the default seed.

**(a) Error correlation at 3 months** (errors scaled by series mean actual, pooled over series; "stat" = stat baseline, "cust" = bias-corrected customer, "book" = order book):

| group | stat vs cust | stat vs book | cust vs book |
|---|---|---|---|
| all series, pre | 0.85 | 0.23 | 0.25 |
| all series, post | 0.89 | 0.12 | 0.14 |
| smooth families, pre | -0.28 | -0.36 | 0.09 |
| smooth families, post | -0.12 | -0.13 | 0.07 |
| intermittent families, pre | 0.91 | 0.30 | 0.28 |
| intermittent families, post | 0.94 | 0.13 | 0.16 |

On smooth families the three sources' errors are uncorrelated or negatively correlated, so averaging them cancels error. On intermittent families the stat baseline and the customer forecast both forecast the expected level and the lumpy actual drives both errors the same way (0.9 correlation); only the order book carries different information.

**(b) Leakage check.** Every actual after month 45 was multiplied by a random factor in 0.3 to 3.0 and the whole pipeline rerun. All 20,920 forecast rows, intervals and blend choices at origins up to 45, and every detection at origins up to 45, were identical to the unperturbed run; 63% of forecasts at later origins changed, as they should. The blend's choice at an origin uses only origins strictly before it with targets realised by it; the bias and fill ratios use only realised targets; the average uses the three inputs' own origin-time forecasts. No leak found.

**(c) Blend without the order book** (average = stat baseline + bias-corrected only), FVA vs naive in pp:

| horizon, period | blend with book | blend without book | best single input |
|---|---|---|---|
| 1, pre | +72 | +34 | order book +67 |
| 1, post | +76 | +45 | order book +74 |
| 3, pre | +60 | +33 | bias-corrected +26 |
| 3, post | +60 | +40 | bias-corrected +33 |
| 6, pre | +40 | +35 | bias-corrected +27 |
| 6, post | +25 | +37 | bias-corrected +30 |

By family type at 3 months (with the book): on intermittent families the blend routes to the order book at every origin and adds +73 pre / +69 post, where the bias-corrected customer forecast adds only +12 / +8. On smooth families the blend averages and adds +55 / +56, above the best single source (+32 / +43). Without the book the intermittent figure falls to +11 / +10 and the smooth figure to +43 / +52.

**Finding.** The +60 pp at 3 months is real and not leaked, but about two thirds of it is the order book: lumpy orders are mostly on the book three months out (80% visibility in the generator), so the blend hands intermittent families to it. The rest is genuine averaging gain on smooth families, where the inputs' errors are not aligned. At 6 months the order book hurts and the blend without it is as good or better. The report's section 4 now states the decomposition with generated numbers, and the README headline quotes the honest version.

### 3. Interval coverage in the report

Section 3 now states the stat baseline's achieved 80% coverage at 3 months pre- and post-break against the blend's, with one sentence on what the gap means. The numbers are generated from the summary table at every run.

### 4. Committed outputs

`outputs/charts/*.png`, `outputs/governance_report.md`, `outputs/planner_table.csv` and `outputs/summary_metrics.csv` are committed so the README renders on GitHub without a run; the data files stay ignored. The README links charts as `outputs/charts/...` from the repo root; the report links them as `charts/...` and the docs as `../docs/...` from `outputs/`, so both resolve where each file lives. Rerunning `python -m fgov.run` rewrites the committed files; a changed number shows up as a diff.

### 5. README tidy

Opening paragraph split in three with the synthetic-data sentence first; a fourth headline bullet on interval coverage; a short section on reading the planner table; and a line at the top of `SPEC.md` stating that the owner wrote it and reviewed the result.
