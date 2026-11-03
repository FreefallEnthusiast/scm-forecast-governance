"""Robustness study: re-run the pipeline over many synth seeds, one CSV row per seed."""
import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fgov import breaks, forecast, metrics, synth  # noqa: E402

SHARES = ("order_book", "average", "customer_bias_corrected", "stat_baseline")


def pick(table, method, horizon, period, col, customer=None):
    """One value from a summarise/fva table; NaN if the row is missing."""
    m = (table["method"] == method) & (table["horizon"] == horizon) & (table["period"] == period)
    if customer is not None:
        m &= table["customer"] == customer
    vals = table.loc[m, col]
    return float(vals.iloc[0]) if len(vals) else float("nan")


def detection_stats(frames, alerts):
    fam = frames["families"]
    est = set(fam.loc[~fam["is_new"].astype(bool), "family"])
    lag = alerts.set_index("family")["months_after_onset"]
    est_lag = lag[lag.index.isin(est)]
    return {
        "total_step_factor": float(fam["total_step_factor"].iloc[0]),
        "total_detection_lag": float(lag.get("TOTAL", float("nan"))),
        "families_detected_within_4": int((est_lag <= 4).sum()),
        "families_never_detected": int(est_lag.isna().sum()),
    }


def metric_stats(forecasts, actuals):
    ev = metrics.evaluate(forecasts, actuals)
    by_cust = metrics.summarise(ev, by=("customer",))
    overall = metrics.fva(metrics.summarise(ev))
    return {
        "c2_raw_bias_h6_pre": pick(by_cust, "customer_raw", 6, "pre", "bias", "C2"),
        "c3_raw_bias_h6_pre": pick(by_cust, "customer_raw", 6, "pre", "bias", "C3"),
        "c5_raw_bias_h6_post": pick(by_cust, "customer_raw", 6, "post", "bias", "C5"),
        "c5_corrected_bias_h6_post": pick(by_cust, "customer_bias_corrected", 6, "post", "bias", "C5"),
        "blend_fva_naive_h3_pre": pick(overall, "blend", 3, "pre", "fva_vs_naive"),
        "blend_fva_naive_h3_post": pick(overall, "blend", 3, "post", "fva_vs_naive"),
        "stat_fva_naive_h3_post": pick(overall, "stat_baseline", 3, "post", "fva_vs_naive"),
        "corrected_fva_naive_h3_post": pick(overall, "customer_bias_corrected", 3, "post", "fva_vs_naive"),
        "blend_coverage80_h3_all": pick(overall, "blend", 3, "all", "coverage80"),
        "stat_coverage80_h3_all": pick(overall, "stat_baseline", 3, "all", "coverage80"),
    }


def run_seed(seed):
    t0 = time.perf_counter()
    frames = synth.generate(synth.SynthParams(seed=seed))
    alerts = breaks.alert_table(frames["actuals"])
    forecasts, choices = forecast.run_all(frames, alerts, n_jobs=2)
    row = {"seed": seed, **detection_stats(frames, alerts), **metric_stats(forecasts, frames["actuals"])}
    share = choices["chosen"].value_counts(normalize=True)
    row.update({f"share_{k}": float(share.get(k, 0.0)) for k in SHARES})
    row["wall_seconds"] = time.perf_counter() - t0
    return row


def append_row(row, path):
    """Add one row, rewriting the file so error rows and result rows share columns."""
    new = pd.DataFrame([row])
    old = pd.read_csv(path) if path.exists() else new.iloc[:0]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20, help="run seeds start..N-1")
    ap.add_argument("--start", type=int, default=0, help="first seed (resume support)")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "seed_study.csv"
    if args.start == 0 and path.exists():
        path.unlink()
    for seed in range(args.start, args.seeds):
        try:
            row = run_seed(seed)
        except Exception as exc:  # noqa: BLE001 - record and continue
            traceback.print_exc()
            row = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
        append_row(row, path)
        print(f"seed {seed} done", flush=True)
    df = pd.read_csv(path)
    num = df.drop(columns=["error"], errors="ignore").select_dtypes("number")
    pd.set_option("display.width", 200)
    print(num.describe().T[["50%", "min", "max"]].rename(columns={"50%": "median"}).round(3))


if __name__ == "__main__":
    main()
