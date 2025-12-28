#!/usr/bin/env python
import argparse
import time

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from setting import token


FIELDS = ["open", "high", "low", "close", "pre_close", "vol", "amount"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing index.daily.hdf rows from tushare index_daily."
    )
    parser.add_argument("--path", default="index.daily.hdf", help="path to index.daily.hdf")
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds per date")
    parser.add_argument("--req-sleep", type=float, default=0.15, help="sleep seconds per request")
    parser.add_argument("--retry", type=int, default=3, help="retry per request")
    parser.add_argument("--limit", type=int, default=None, help="limit missing dates to first N")
    return parser.parse_args()


def main():
    args = parse_args()
    pro = ts.pro_api(token)

    with h5.File(args.path, "r+") as f:
        symbols = [x.decode("utf-8") for x in f["symbols"][()]]
        dates = f["dates"][()]
        close = f["close"][()]
        nan_mask = ~np.isfinite(close)
        missing_date_mask = nan_mask.any(axis=1)
        missing_dates = dates[missing_date_mask]

        if args.limit is not None:
            missing_dates = missing_dates[: args.limit]
        print("missing dates count", missing_dates.size)
        if missing_dates.size == 0:
            return 0
        print("missing dates sample", [int(d) for d in missing_dates[:50]])
        if missing_dates.size > 50:
            print("...")

        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        api_symbols = {s: s.replace("SSE", "SH").replace("SZE", "SZ") for s in symbols}

        def fetch(sym, trade_date):
            for i in range(args.retry):
                try:
                    df = pro.index_daily(ts_code=api_symbols[sym], trade_date=int(trade_date))
                    return df
                except Exception as e:
                    msg = str(e)
                    if "最多访问该接口" in msg:
                        print("rate limit hit; sleeping 60s")
                        time.sleep(60)
                        continue
                    if i == args.retry - 1:
                        print(f"fetch failed {sym} {trade_date}: {e}")
                        return None
                    time.sleep(1)

        missing_indices = np.where(missing_date_mask)[0]
        for i, (di, d) in enumerate(zip(missing_indices, missing_dates)):
            trade_date = int(d)
            frames = []
            for sym in symbols:
                df = fetch(sym, trade_date)
                time.sleep(args.req_sleep)
                if df is not None and not df.empty:
                    df["ts_code"] = sym
                    frames.append(df)
            if not frames:
                print("no data for", trade_date)
                continue
            df = pd.concat(frames, ignore_index=True)
            idx = df["ts_code"].map(sym_to_idx).to_numpy()
            for field in FIELDS:
                row = f[field][di, :]
                vals = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=np.float32)
                missing_here = ~np.isfinite(row[idx])
                if missing_here.any():
                    row[idx[missing_here]] = vals[missing_here]
                    f[field][di, :] = row
            if i % 10 == 0 or i == len(missing_dates) - 1:
                print(f"filled {i+1}/{len(missing_dates)} date={trade_date}")
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
