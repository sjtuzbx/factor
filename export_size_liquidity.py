#!/usr/bin/env python
import argparse
import os

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SIZE and LIQUIDITY factors for a date.")
    parser.add_argument("--daily", default="/home/ubuntu/scripts/cache_all/daily.hdf", help="path to daily.hdf")
    parser.add_argument("--date", type=int, required=True, help="trade date YYYYMMDD")
    parser.add_argument(
        "--output",
        default=None,
        help="output csv path; default: ./size_liquidity_<date>.csv",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()

    with h5py.File(args.daily, "r") as f:
        dates = f["dates"][()]
        if args.date not in dates:
            raise SystemExit(f"date not found in daily.hdf: {args.date}")
        idx = int(np.where(dates == args.date)[0][0])

        symbols = [x.decode("utf-8") for x in f["symbols"][()]]
        if "SIZE" not in f or "LIQUIDITY" not in f:
            raise SystemExit("SIZE or LIQUIDITY not found in daily.hdf")

        size = f["SIZE"][idx, :]
        liquidity = f["LIQUIDITY"][idx, :]

    df = pd.DataFrame(
        {
            "date": args.date,
            "symbol": symbols,
            "SIZE": size,
            "LIQUIDITY": liquidity,
        }
    )

    out = args.output or os.path.join(os.getcwd(), f"size_liquidity_{args.date}.csv")
    df.to_csv(out, index=False)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
