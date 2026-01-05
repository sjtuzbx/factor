import argparse

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/code/cb_cache")
import cb_cache as cbc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Read industry value and name for a symbol/date.")
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--industry-l1",
        default="data/ci_index_member_l1_map.csv",
        help="industry mapping csv",
    )
    parser.add_argument("--symbol", required=True, help="symbol like 000001.SZE")
    parser.add_argument("--date", type=int, required=True, help="trade date YYYYMMDD")
    return parser.parse_args()


def main():
    args = parse_args()

    cache = cbc.EqCache(args.symbol, args.date, args.date, root_path=os.getcwd())
    ind_val = cache.daily.industry[0, 0]

    l1 = pd.read_csv(args.industry_l1)
    if "industry_idx" in l1.columns:
        l1 = l1.sort_values("industry_idx")
        codes = l1["l1_code"].astype(str).values
        names = l1["l1_name"].astype(str).values if "l1_name" in l1.columns else None
    else:
        codes = l1["index_code"].astype(str).values
        names = l1["industry_name"].astype(str).values if "industry_name" in l1.columns else None

    if np.isnan(ind_val):
        print(args.symbol, args.date, "industry=NaN")
        return

    ind_idx = int(ind_val)
    ind_code = codes[ind_idx] if ind_idx < len(codes) else "UNKNOWN"
    if names is not None and ind_idx < len(names):
        ind_name = names[ind_idx]
        print(args.symbol, args.date, ind_code, ind_name)
    else:
        print(args.symbol, args.date, ind_code)


if __name__ == "__main__":
    main()
