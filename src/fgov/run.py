"""End to end: generate -> break alerts -> candidate forecasts -> report.

    python -m fgov.run [--outdir outputs] [--origins-from 25]

Regenerates every file in the output directory from the fixed seed.
"""

import argparse
import shutil
import time
from pathlib import Path

from fgov import breaks, forecast, report, synth


GENERATED = ("actuals.parquet", "customer_forecasts.csv", "order_book.csv", "families.csv",
             "planner_table.csv", "summary_metrics.csv", "governance_report.md")


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--origins-from", type=int, default=25,
                        help="first backtest origin (origins run to 60); raise it for a quick run")
    args = parser.parse_args(argv)

    out = Path(args.outdir)
    shutil.rmtree(out / "charts", ignore_errors=True)        # only what this run writes
    for name in GENERATED:
        (out / name).unlink(missing_ok=True)

    start = last = time.perf_counter()
    def step(label):
        nonlocal last
        now = time.perf_counter()
        print(f"{label:<28} {now - last:6.1f} s")
        last = now

    params = synth.SynthParams()
    frames = synth.generate(params)
    synth.write_outputs(frames, str(out))
    step("synthetic panel")
    alerts = breaks.alert_table(frames["actuals"], onset=params.step_onset)
    step("break alerts")
    forecasts, choices = forecast.run_all(frames, alerts, origins=range(args.origins_from, 61))
    step("candidate forecasts")
    report.build_all(frames, forecasts, choices, alerts, onset=params.step_onset, outdir=out)
    step("tables, charts, report")
    print(f"{'total':<28} {time.perf_counter() - start:6.1f} s\n")
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != ".gitkeep"):
        print(f"{f}  ({f.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
