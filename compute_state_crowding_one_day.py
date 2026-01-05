#!/usr/bin/env python3
import argparse
import os
import time

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from setting import token


def parse_args():
    parser = argparse.ArgumentParser(description="Compute state ownership and crowding factors for one day.")
    parser.add_argument("--date", type=int, required=True, help="YYYYMMDD")
    parser.add_argument("--daily", default="daily.hdf", help="daily.hdf path")
    parser.add_argument("--index-daily", default="index.daily.hdf", help="index.daily.hdf path")
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds per symbol")
    parser.add_argument("--limit", type=int, default=None, help="limit number of symbols (debug)")
    parser.add_argument(
        "--out",
        default=None,
        help="output csv path (default data/state_crowding_YYYYMMDD.csv)",
    )
    return parser.parse_args()


def last_quarter_end(date_int):
    y = date_int // 10000
    mmdd = date_int % 10000
    quarters = [331, 630, 930, 1231]
    candidates = [q for q in quarters if q <= mmdd]
    if candidates:
        return y * 10000 + max(candidates)
    return (y - 1) * 10000 + 1231


def is_state_holder(holder_type, holder_name):
    text = f"{holder_type or ''} {holder_name or ''}"
    keywords = [
        "国有",
        "国资",
        "国家",
        "中央",
        "省",
        "市",
        "县",
        "政府",
        "财政",
        "国开",
        "汇金",
        "社保",
        "国资委",
        "国务院",
    ]
    return any(k in text for k in keywords)


def zscore_cs(arr):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return np.full_like(arr, np.nan, dtype=np.float64)
    return (arr - mean) / std


def main():
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()

    with h5.File(args.daily, "r") as f:
        dates = f["dates"][:]
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        if args.date not in dates:
            raise SystemExit(f"date not in daily.hdf: {args.date}")
        date_idx = int(np.where(dates == args.date)[0][0])
        start_idx = max(0, date_idx - 19)
        turnover = f["turnover_rate"][start_idx:date_idx + 1]
        pe_ttm = f["pe_ttm"][date_idx]
        pb = f["pb"][date_idx]
        ps = f["ps"][date_idx]

    if args.limit:
        symbols = symbols[: args.limit]
        turnover = turnover[:, : args.limit]
        pe_ttm = pe_ttm[: args.limit]
        pb = pb[: args.limit]
        ps = ps[: args.limit]

    turnover_20d = np.nanmean(turnover, axis=0)
    turnover_z = zscore_cs(turnover_20d)

    pe = np.where(pe_ttm > 0, np.log1p(pe_ttm), np.nan)
    pbv = np.where(pb > 0, np.log1p(pb), np.nan)
    psv = np.where(ps > 0, np.log1p(ps), np.nan)
    val_z = zscore_cs(pe) + zscore_cs(pbv) + zscore_cs(psv)
    val_z = val_z / 3.0

    crowding = np.nanmean(np.column_stack([turnover_z, val_z]), axis=1)

    pro = ts.pro_api(token)
    end_date = last_quarter_end(args.date)
    state_ratio = np.full(symbols.shape[0], np.nan, dtype=np.float64)

    for i, sym in enumerate(symbols, 1):
        try:
            df = pro.top10_holders(ts_code=sym, end_date=str(end_date))
        except Exception as exc:
            print(f"top10_holders failed {sym}: {exc}")
            df = None
        if df is not None and not df.empty:
            mask = df.apply(lambda r: is_state_holder(r.get("holder_type"), r.get("holder_name")), axis=1)
            if mask.any():
                state_ratio[i - 1] = df.loc[mask, "hold_ratio"].sum()
        if args.sleep:
            time.sleep(args.sleep)
        if i % 200 == 0 or i == symbols.size:
            print(f"progress {i}/{symbols.size}")

    out_path = args.out or f"data/state_crowding_{args.date}.csv"
    out_df = pd.DataFrame(
        {
            "ts_code": symbols,
            "state_hold_ratio": state_ratio,
            "crowding": crowding,
        }
    )
    out_df.to_csv(out_path, index=False)

    print("saved", out_path)
    print("state coverage", np.isfinite(state_ratio).mean())
    print("crowding coverage", np.isfinite(crowding).mean())
    print("crowding describe")
    print(pd.Series(crowding).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print("state_hold_ratio describe")
    print(pd.Series(state_ratio).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))


if __name__ == "__main__":
    main()
