import argparse
import os
import time

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts
from IPython import embed

from setting import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ci_index_member L1 data for all symbols in daily.hdf."
    )
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--out",
        default="data/ci_index_member_l1.csv",
        help="output csv path",
    )
    parser.add_argument(
        "--missing",
        default="data/ci_index_member_l1_missing.csv",
        help="missing list csv path",
    )
    parser.add_argument(
        "--checkpoint",
        default="export_ci_index_member_l1.progress",
        help="checkpoint file path",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds per symbol")
    return parser.parse_args()


def to_ts_code(sym):
    return sym.replace("SSE", "SH").replace("SZE", "SZ").replace("BSE", "BJ")


def write_rows(path, df, header):
    mode = "a" if os.path.exists(path) else "w"
    df.to_csv(path, mode=mode, header=header and mode == "w", index=False)


def main():
    args = parse_args()

    with h5.File(args.daily, "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])

    pro = ts.pro_api(token)

    start_idx = 0
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r") as ck:
                start_idx = int(ck.read().strip())
        except Exception:
            start_idx = 0
    if start_idx < 0 or start_idx >= symbols.size:
        start_idx = 0

    total = symbols.size
    for i, sym in enumerate(symbols[start_idx:], start=start_idx):
        ts_code = to_ts_code(sym)
        # if ts_code == "000003.SZ":
        #     embed()
        try:
            df = pro.ci_index_member(ts_code=ts_code)
        except Exception:
            df = None

        if df is None or df.empty:
            miss = pd.DataFrame([{"ts_code": ts_code, "h5_symbol": sym}])
            write_rows(args.missing, miss, header=True)
        else:
            df = df.copy()
            df["h5_symbol"] = sym
            df = df[
                [
                    "ts_code",
                    "h5_symbol",
                    "l1_code",
                    "l1_name",
                    "in_date",
                    "out_date",
                    "is_new",
                ]
            ]
            write_rows(args.out, df, header=True)

        if i % 50 == 0 or i == total - 1:
            with open(args.checkpoint, "w") as ck:
                ck.write(str(i))
            print(f"progress {i+1}/{total}")

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
