import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from setting import *


def parse_args():
    parser = argparse.ArgumentParser(description="Update industry dataset in daily.hdf.")
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--ci-csv",
        default="data/ci_index_member_l1.csv",
        help="ci_index_member L1 csv",
    )
    parser.add_argument(
        "--industry-map",
        default="data/ci_index_member_l1_map.csv",
        help="output industry mapping csv path",
    )
    parser.add_argument(
        "--area-map",
        default="data/area_map.csv",
        help="output area mapping csv path",
    )
    return parser.parse_args()


def to_h5_symbol(ts_code):
    return ts_code.replace("SH", "SSE").replace("SZ", "SZE").replace("BJ", "BSE")


def main():
    args = parse_args()

    if not os.path.exists(args.ci_csv):
        raise SystemExit(f"missing ci_index_member csv: {args.ci_csv}")

    data = pd.read_csv(args.ci_csv)
    required_cols = {"l1_code", "l1_name", "in_date", "out_date"}
    if "h5_symbol" not in data.columns and "ts_code" not in data.columns:
        raise SystemExit("ci_index_member csv missing h5_symbol or ts_code")
    if not required_cols.issubset(set(data.columns)):
        raise SystemExit("ci_index_member csv missing required columns")
    if "h5_symbol" in data.columns:
        data["code"] = data["h5_symbol"].astype(str)
    else:
        data["code"] = data["ts_code"].astype(str).map(to_h5_symbol)
    data = data.dropna(subset=["code", "in_date", "l1_code"])
    data["in_date"] = data["in_date"].astype(str)
    data = data[data["in_date"].str.isnumeric()]
    data["in_date"] = data["in_date"].astype(int)
    data["out_date"] = data["out_date"].fillna(99999999).astype(int)

    l1_codes = data["l1_code"].astype(str)
    l1_names = data["l1_name"].astype(str)
    l1_unique = (
        pd.DataFrame({"l1_code": l1_codes, "l1_name": l1_names})
        .drop_duplicates(subset=["l1_code"])
        .sort_values("l1_code")
        .reset_index(drop=True)
    )
    l1_unique["industry_idx"] = np.arange(len(l1_unique), dtype=int)
    code_to_idx = dict(zip(l1_unique["l1_code"].values, l1_unique["industry_idx"].values))
    os.makedirs(os.path.dirname(args.industry_map), exist_ok=True)
    l1_unique.to_csv(args.industry_map, index=False)

    pro = ts.pro_api(token)
    area_df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,area")
    area_df = area_df.dropna(subset=["ts_code", "area"]).copy()
    area_df["code"] = area_df["ts_code"].astype(str).map(to_h5_symbol)
    areas = sorted(area_df["area"].unique())
    area_to_idx = {name: i for i, name in enumerate(areas)}
    area_df["area_idx"] = area_df["area"].map(area_to_idx)
    os.makedirs(os.path.dirname(args.area_map), exist_ok=True)
    pd.DataFrame({"area": areas, "area_idx": range(len(areas))}).to_csv(
        args.area_map, index=False
    )

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
            out_dates = df["out_date"].values
            l1_codes = df["l1_code"].astype(str).values

            idx = np.searchsorted(in_dates, dates, side="right") - 1
            valid = idx >= 0
            if not valid.any():
                continue
            pick = idx[valid]
            chosen = l1_codes[pick]
            active = out_dates[pick] >= dates[valid]
            mapped = np.full(valid.sum(), np.nan, dtype=float)
            if active.any():
                picked = chosen[active]
                mapped[active] = np.array([code_to_idx.get(x, np.nan) for x in picked], dtype=float)
            dset[valid, g] = mapped

        area_shape = (dates.shape[0], symbols.shape[0])
        if "area" in f:
            area_dset = f["area"]
            if area_dset.shape != area_shape:
                raise RuntimeError("area dataset shape mismatch")
        else:
            area_dset = f.create_dataset("area", area_shape, maxshape=(None, None))

        area_dset[:, :] = np.nan
        area_map = dict(zip(area_df["code"].values, area_df["area_idx"].values))
        for g, symbol in enumerate(symbols):
            val = area_map.get(symbol)
            if val is None:
                continue
            area_dset[:, g] = float(val)

    print("saved industry to", args.daily)
    print("saved industry mapping to", args.industry_map)
    print("saved area to", args.daily, "mapping", args.area_map)


if __name__ == "__main__":
    main()
