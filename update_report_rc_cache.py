import argparse
import os
import time

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from setting import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update report_rc cache based on daily.hdf symbols/dates."
    )
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--report-path",
        default="data/report_rc.20140101-20251224.csv",
        help="report_rc cache csv path",
    )
    parser.add_argument(
        "--checkpoint",
        default="update_report_rc_cache.progress",
        help="checkpoint file path",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds per symbol")
    return parser.parse_args()


def decode_symbols(raw):
    return np.array([x.decode("utf-8") for x in raw])


def change_ts(sym):
    return sym.replace("SSE", "SH").replace("SZE", "SZ").replace("BSE", "BJ")


def load_or_empty(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def latest_report_date(df, ts_sym):
    if df.empty:
        return None
    x = df[df.ts_code == ts_sym]
    if x.empty:
        return None
    for col in ["report_date", "pub_date", "ann_date", "end_date"]:
        if col in x.columns:
            vals = x[col].dropna()
            if not vals.empty:
                return int(vals.max())
    return None


def fetch_incremental(fetch_fn, ts_sym, start_date, end_date):
    if start_date is None:
        return fetch_fn(ts_code=ts_sym, end_date=end_date)
    return fetch_fn(ts_code=ts_sym, start_date=str(start_date), end_date=end_date)


def main():
    args = parse_args()

    with h5.File(args.daily, "r") as f:
        symbols = decode_symbols(f["symbols"][()])
        dates = f["dates"][()]

    if dates.size == 0:
        raise SystemExit("daily.hdf has no dates")

    start_date = int(dates[0])
    end_date = int(dates[-1])
    end_date_str = str(end_date)

    report_df = load_or_empty(args.report_path)

    pro = ts.pro_api(token)

    report_new = []

    total = int(symbols.size)
    start_ts = time.time()
    start_idx = 0
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r") as f:
                start_idx = int(f.read().strip())
        except Exception:
            start_idx = 0
    if start_idx < 0 or start_idx > total:
        start_idx = 0

    for idx, sym in enumerate(symbols[start_idx:], start=start_idx + 1):
        ts_sym = change_ts(sym)

        last_date = latest_report_date(report_df, ts_sym)
        rep_start = start_date if last_date is None else last_date + 1

        if rep_start <= end_date:
            df_rep = fetch_incremental(pro.report_rc, ts_sym, rep_start, end_date_str)
            if df_rep is not None and not df_rep.empty:
                report_new.append(df_rep)

        time.sleep(args.sleep)

        if idx % 50 == 0 or idx == total:
            if report_new:
                report_df = pd.concat([report_df] + report_new, ignore_index=True)
                report_df = report_df.drop_duplicates()
                report_df.to_csv(args.report_path, index=False)
                report_new = []
            with open(args.checkpoint, "w") as f:
                f.write(str(idx))

            elapsed = time.time() - start_ts
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = (total - idx) / rate if rate > 0 else 0.0
            print(
                f"progress {idx}/{total}, elapsed {elapsed:.1f}s, "
                f"rate {rate:.2f} sym/s, eta {remaining/60:.1f} min"
            )

    if report_new:
        report_df = pd.concat([report_df] + report_new, ignore_index=True)
        report_df = report_df.drop_duplicates()
        report_df.to_csv(args.report_path, index=False)


if __name__ == "__main__":
    main()
