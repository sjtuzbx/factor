import argparse

import h5py as h5
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute analyst NP std / market cap factor for a given date."
    )
    parser.add_argument("--date", type=int, required=True, help="YYYYMMDD")
    parser.add_argument(
        "--window-days",
        type=int,
        default=365,
        help="lookback window in days (default 365)",
    )
    parser.add_argument(
        "--report",
        default="data/report_rc.20140101-20251224.csv",
        help="report_rc csv path",
    )
    parser.add_argument(
        "--daily",
        default="daily.hdf.bak.20251227",
        help="daily hdf path (for total_mv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output csv path (default: data/np_std_mv_YYYYMMDD.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    end_date = pd.to_datetime(str(args.date))
    start_date = end_date - pd.Timedelta(days=args.window_days - 1)

    df = pd.read_csv(
        args.report, usecols=["ts_code", "report_date", "quarter", "np"]
    )
    df["report_date"] = df["report_date"].astype(str)

    year = str(args.date)[:4]
    q4 = f"{year}Q4"

    sub = df[
        (df["report_date"] >= start_date.strftime("%Y%m%d"))
        & (df["report_date"] <= end_date.strftime("%Y%m%d"))
        & (df["quarter"] == q4)
    ].copy()

    if sub.empty:
        raise SystemExit("no matching reports")

    agg = sub.groupby(["ts_code"], as_index=False).agg(
        np_std=("np", "std"),
        np_cnt=("np", "count"),
    )

    with h5.File(args.daily, "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]
        total_mv = f["total_mv"][:]

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    h5_sym = (
        agg["ts_code"]
        .astype(str)
        .str.replace("SH", "SSE")
        .str.replace("SZ", "SZE")
        .str.replace("BJ", "BSE")
    )
    agg["sym_idx"] = h5_sym.map(sym_to_idx).fillna(-1).astype(int)

    date_idx = np.searchsorted(dates, args.date)
    if date_idx >= len(dates) or int(dates[date_idx]) != args.date:
        raise SystemExit("date not found in daily.hdf")

    mv_row = total_mv[date_idx]
    factor = np.full(len(agg), np.nan, dtype=float)

    valid = (agg["sym_idx"] >= 0) & np.isfinite(agg["np_std"])
    if valid.any():
        idx = agg.loc[valid, "sym_idx"].to_numpy()
        mv = mv_row[idx]
        ok = np.isfinite(mv) & (mv > 0)
        valid_idx = np.where(valid.to_numpy())[0]
        factor[valid_idx[ok]] = agg.loc[valid_idx[ok], "np_std"].to_numpy() / mv[ok]

    agg["np_std_mv"] = factor

    out_path = args.out or f"data/np_std_mv_{args.date}.csv"
    agg[["ts_code", "np_std", "np_cnt", "np_std_mv"]].to_csv(out_path, index=False)
    print("saved", out_path, "rows", len(agg))


if __name__ == "__main__":
    main()
