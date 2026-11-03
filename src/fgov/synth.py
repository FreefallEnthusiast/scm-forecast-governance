"""Synthetic demand panel: actuals, customer forecasts and order book.

Grain is customer x product family x month. Months are integer indices
1..60 (48 months of history followed by a 12-month holdout). Every number
in the story is set in `SynthParams`; change a field there to change it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynthParams:
    seed: int = 20240601                 # numpy default_rng seed; fixes every draw
    n_history: int = 48                  # months of history
    n_holdout: int = 12                  # months of holdout after the history
    start_date: str = "2019-01-01"       # calendar label for month 1 (cosmetic only)
    base_total: float = 20000.0          # expected total units per month before the step

    # Customers: labels, share of total demand.
    customers: tuple = ("C1", "C2", "C3", "C4", "C5")
    customer_shares: tuple = (0.30, 0.25, 0.20, 0.15, 0.10)
    cust_family_sigma: float = 0.30      # lognormal spread of a customer's mix across families

    # Customer forecast behaviour. Each tuple is aligned with `horizons`.
    horizons: tuple = (1, 3, 6)          # forecast horizons in months
    customer_bias: tuple = (             # multiplicative bias on true demand, per horizon
        (1.00, 1.00, 1.00),              # C1 reliable
        (1.05, 1.15, 1.30),              # C2 optimistic, shrinks as horizon shortens
        (0.85, 0.85, 0.85),              # C3 pessimistic, stable
        (1.00, 1.00, 1.00),              # C4 unbiased on average
        (1.00, 1.00, 1.00),              # C5 accurate in normal times
    )
    customer_noise: tuple = (            # lognormal sigma of forecast error, per horizon
        (0.04, 0.06, 0.08),              # C1 small error
        (0.06, 0.08, 0.10),              # C2
        (0.06, 0.08, 0.10),              # C3
        (0.30, 0.35, 0.40),              # C4 large, independent per origin -> big revisions
        (0.05, 0.07, 0.09),              # C5
    )
    upcycle_customer: str = "C5"         # customer that panics during the upcycle
    upcycle_bias: tuple = (1.35, 1.50, 1.60)  # extra multiplier for targets >= step onset

    # Product families: F01..F06 smooth, F07..F10 intermittent, F11..F12 new.
    n_smooth: int = 6
    n_intermittent: int = 4
    n_new: int = 2
    n_long_lead: int = 4                 # the first n smooth families are long-lead
    family_weight_sigma: float = 0.40    # lognormal spread of family sizes
    intermittent_weight: float = 0.40    # size of an intermittent family relative to smooth
    new_weight: float = 0.35             # size of a new family relative to smooth
    new_launch_month: int = 37           # new families have no demand (NaN) before this

    # Demand process.
    season_amplitude: float = 0.10       # peak-to-mean of the annual sine seasonality
    noise_sigma: float = 0.15            # lognormal monthly noise on smooth families
    occurrence_prob: float = 0.45        # P(order in a month) for intermittent families
    size_sigma: float = 0.60             # lognormal sigma of an intermittent order size
    size_clip: float = 3.0               # order-size draws truncated at +/- this many sigmas

    # The step change.
    step_onset: int = 37                 # first month the step is visible
    total_step_range: tuple = (1.8, 2.2) # total demand factor drawn uniformly from this
    family_step_range: tuple = (1.3, 2.8)  # per-family factors drawn and clipped to this
    ramp_months: int = 3                 # linear ramp length; full level at onset + ramp - 1
    long_lead_offset: int = 1            # long-lead families step this many months earlier

    # Order book: share of true demand already confirmed, by months ahead.
    book_horizons: tuple = (1, 2, 3, 4, 5, 6)
    fill_ratio: tuple = (0.85, 0.74, 0.63, 0.52, 0.41, 0.30)  # normal times
    upcycle_fill_boost: float = 0.10     # added to fill ratio for targets >= step onset
    fill_noise_sigma: float = 0.10       # lognormal noise on the fill ratio


def family_table(p: SynthParams) -> pd.DataFrame:
    n = p.n_smooth + p.n_intermittent + p.n_new
    idx = np.arange(n)
    return pd.DataFrame({
        "family": [f"F{i + 1:02d}" for i in idx],
        "is_intermittent": (idx >= p.n_smooth) & (idx < p.n_smooth + p.n_intermittent),
        "is_new": idx >= p.n_smooth + p.n_intermittent,
        "is_long_lead": idx < p.n_long_lead,
    })


def step_factors(p, rng, weights, is_new):
    """Draw per-family step factors and rescale so the total hits a drawn factor.

    New families do not step (they launch during the upcycle), but their
    launch volume counts towards the new total level.
    """
    total = rng.uniform(*p.total_step_range)
    lo, hi = p.family_step_range
    raw = rng.uniform(lo, hi, size=len(weights))
    old = weights[~is_new].sum()
    target = total * old - weights[is_new].sum()
    k = 1.0
    for _ in range(100):
        g = np.clip(raw * k, lo, hi)
        k *= target / (weights[~is_new] * g[~is_new]).sum()
    g = np.clip(raw * k, lo, hi)
    g[is_new] = 1.0
    return total, g


def step_profile(p, months, factor, long_lead):
    onset = p.step_onset - (p.long_lead_offset if long_lead else 0)
    progress = np.clip((months - onset + 1) / p.ramp_months, 0.0, 1.0)
    return 1.0 + (factor - 1.0) * progress


def long_table(p, fam, t, h, values, value_name, origin_name):
    """Flatten a (customer, family, target) cube into a long frame with an origin column."""
    n_c, n_f = len(p.customers), len(fam)
    c, f, k = np.meshgrid(np.arange(n_c), np.arange(n_f), np.arange(len(t)), indexing="ij")
    return pd.DataFrame({
        origin_name: t[k.ravel()] - h,
        "target_month": t[k.ravel()],
        "customer": np.array(p.customers)[c.ravel()],
        "family": fam["family"].to_numpy()[f.ravel()],
        value_name: values.ravel(),
    })


def customer_forecasts(p, rng, fam, signal, months):
    """Each customer's forecast of every target made 1, 3 and 6 months earlier."""
    n_c, n_f = len(p.customers), len(fam)
    parts = []
    for hi, h in enumerate(p.horizons):
        t = months[months - h >= 1]
        bias = np.array([b[hi] for b in p.customer_bias])[:, None, None]
        sig = np.array([s[hi] for s in p.customer_noise])[:, None, None]
        shape = (n_c, n_f, len(t))
        noise = np.exp(sig * rng.standard_normal(shape) - 0.5 * sig**2)
        panic = np.ones(shape)
        c5 = p.customers.index(p.upcycle_customer)
        panic[c5][:, t >= p.step_onset] = p.upcycle_bias[hi]
        fc = np.round(signal[:, :, t - 1] * bias * noise * panic)
        parts.append(long_table(p, fam, t, h, fc, "customer_forecast", "origin_month").assign(horizon=h))
    cols = ["origin_month", "target_month", "horizon", "customer", "family", "customer_forecast"]
    return pd.concat(parts, ignore_index=True)[cols]


def order_book(p, rng, fam, actual, months):
    """Confirmed quantity on the book 1..6 months ahead; never exceeds the eventual actual."""
    n_c, n_f = len(p.customers), len(fam)
    parts = []
    for hi, h in enumerate(p.book_horizons):
        t = months[months - h >= 1]
        shape = (n_c, n_f, len(t))
        ratio = p.fill_ratio[hi] + p.upcycle_fill_boost * (t >= p.step_onset)
        ratio = ratio[None, None, :] * rng.lognormal(0.0, p.fill_noise_sigma, shape)
        qty = np.floor(actual[:, :, t - 1] * np.clip(ratio, 0.0, 1.0))
        parts.append(long_table(p, fam, t, h, qty, "confirmed_qty", "as_of_month"))
    return pd.concat(parts, ignore_index=True)


def generate(p: SynthParams | None = None) -> dict:
    p = p or SynthParams()
    rng = np.random.default_rng(p.seed)
    fam = family_table(p)
    n_c, n_f, n_m = len(p.customers), len(fam), p.n_history + p.n_holdout
    months = np.arange(1, n_m + 1)
    is_int = fam["is_intermittent"].to_numpy()
    is_new = fam["is_new"].to_numpy()

    # Base level per family, then split across customers.
    fw = rng.lognormal(0.0, p.family_weight_sigma, n_f)
    fw = np.where(is_int, fw * p.intermittent_weight, fw)
    fw = np.where(is_new, fw * p.new_weight, fw)
    fw = fw / fw[~is_new].sum()          # established families sum to 1
    mix = rng.lognormal(0.0, p.cust_family_sigma, (n_c, n_f))
    mix = mix / mix.mean(axis=0)
    base = p.base_total * np.array(p.customer_shares)[:, None] * fw[None, :] * mix

    total_factor, g = step_factors(p, rng, base.sum(axis=0), is_new)
    fam["step_factor"] = g
    fam["total_step_factor"] = total_factor

    # Expected level (customer x family x month).
    season = 1.0 + p.season_amplitude * np.sin(2 * np.pi * (months - 1) / 12)
    step = np.stack([step_profile(p, months, g[f], fam["is_long_lead"][f]) for f in range(n_f)])
    level = base[:, :, None] * step[None, :, :] * season[None, None, :]

    # Realised demand.
    smooth_noise = rng.lognormal(-0.5 * p.noise_sigma**2, p.noise_sigma, level.shape)
    occur = rng.random(level.shape) < p.occurrence_prob
    z = np.clip(rng.standard_normal(level.shape), -p.size_clip, p.size_clip)
    size = np.exp(p.size_sigma * z - 0.5 * p.size_sigma**2) / p.occurrence_prob
    actual = np.where(is_int[None, :, None], level * occur * size, level * smooth_noise)
    actual = np.round(actual)
    launched = months[None, None, :] >= p.new_launch_month
    actual = np.where(is_new[None, :, None] & ~launched, np.nan, actual)
    level = np.where(np.isnan(actual), np.nan, level)

    cc, ff, mm = np.meshgrid(np.arange(n_c), np.arange(n_f), np.arange(n_m), indexing="ij")
    dates = pd.date_range(p.start_date, periods=n_m, freq="MS")
    actuals = pd.DataFrame({
        "customer": np.array(p.customers)[cc.ravel()],
        "family": fam["family"].to_numpy()[ff.ravel()],
        "month": months[mm.ravel()],
        "month_start": dates[mm.ravel()],
        "actual": actual.ravel(),
    }).merge(fam[["family", "is_intermittent", "is_new", "is_long_lead"]], on="family")

    # Customer forecasts. The customer sees the realised demand for smooth and
    # new families, and only the expected level for intermittent ones.
    signal = np.where(is_int[None, :, None], level, actual)
    forecasts = customer_forecasts(p, rng, fam, signal, months)
    book = order_book(p, rng, fam, actual, months)

    sort = ["customer", "family"]
    return {
        "actuals": actuals.sort_values(sort + ["month"]).reset_index(drop=True),
        "customer_forecasts": forecasts.sort_values(sort + ["target_month", "horizon"]).reset_index(drop=True),
        "order_book": book.sort_values(sort + ["as_of_month", "target_month"]).reset_index(drop=True),
        "families": fam,
    }


def write_outputs(frames: dict, outdir="outputs") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    frames["actuals"].to_parquet(out / "actuals.parquet", index=False)
    frames["customer_forecasts"].to_csv(out / "customer_forecasts.csv", index=False)
    frames["order_book"].to_csv(out / "order_book.csv", index=False)
    frames["families"].to_csv(out / "families.csv", index=False)


if __name__ == "__main__":
    write_outputs(generate(SynthParams()))
