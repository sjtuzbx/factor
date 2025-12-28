import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Update industry dataset in daily.hdf.")
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--industry-csv",
        default="/home/ubuntu/scripts/eqcache/industry.csv",
        help="industry membership csv",
    )
    parser.add_argument(
        "--industry-l1",
        default="/home/ubuntu/scripts/eqcache/industry.sw_L1.csv",
        help="industry L1 definition csv",
    )
    return parser.parse_args()


def to_h5_symbol(ts_code):
    return ts_code.replace("SH", "SSE").replace("SZ", "SZE")


def main():
    args = parse_args()

    if not os.path.exists(args.industry_csv):
        raise SystemExit(f"missing industry csv: {args.industry_csv}")
    if not os.path.exists(args.industry_l1):
        raise SystemExit(f"missing industry L1 csv: {args.industry_l1}")

    data = pd.read_csv(args.industry_csv)
    data = data.dropna(subset=["ts_code", "in_date", "l1_code"])
    data["code"] = data["ts_code"].astype(str).map(to_h5_symbol)
    data["in_date"] = data["in_date"].astype(str)
    data = data[data["in_date"].str.isnumeric()]
    data["in_date"] = data["in_date"].astype(int)

    l1 = pd.read_csv(args.industry_l1)
    if "index_code" not in l1.columns:
        raise SystemExit("industry L1 csv missing index_code")
    index_codes = l1["index_code"].astype(str).values
    code_to_idx = {c: i for i, c in enumerate(index_codes)}

    with h5.File(args.daily, "a") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]

        shape = (dates.shape[0], symbols.shape[0])
        if "industry" in f:
            dset = f["industry"]
            if dset.shape != shape:
                raise RuntimeError("industry dataset shape mismatch")
        else:
            dset = f.create_dataset("industry", shape, maxshape=(None, None))

        dset[:, :] = np.nan

        grouped = data.groupby("code")
        for g, symbol in enumerate(symbols):
            if symbol not in grouped.groups:
                continue
            df = grouped.get_group(symbol).sort_values("in_date")
            in_dates = df["in_date"].values
            l1_codes = df["l1_code"].astype(str).values

            idx = np.searchsorted(in_dates, dates, side="right") - 1
            valid = idx >= 0
            if not valid.any():
                continue
            pick = idx[valid]
            chosen = l1_codes[pick]
            mapped = np.array([code_to_idx.get(x, np.nan) for x in chosen], dtype=float)
            dset[valid, g] = mapped

    print("saved industry to", args.daily)


if __name__ == "__main__":
    main()
