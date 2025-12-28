import argparse
import time

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from python_utils import quant
from python_utils import time_utils as tu
from setting import *


DF1_COLS = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
DF3_COLS = ["adj_factor"]
DF4_COLS = ["up_limit", "down_limit"]
DF5_COLS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]

ROUND_DECIMALS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache daily data based on existing symbols in daily.hdf."
    )
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf file")
    parser.add_argument("--start", type=int, default=20140101, help="start date YYYYMMDD")
    parser.add_argument("--end", type=int, default=20251224, help="end date YYYYMMDD")
    parser.add_argument(
        "--update",
        action="store_true",
        help="only fetch dates after the last date with data",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds per date")
    parser.add_argument("--retry", type=int, default=3, help="retry count per date")
    parser.add_argument(
        "--refresh-listed",
        action="store_true",
        help="recompute is_listed_in_5days dataset",
    )
    return parser.parse_args()


def ts_to_h5_symbol(ts_code):
    return ts_code.replace("SH", "SSE").replace("SZ", "SZE").replace("BJ", "BSE")


def decode_symbols(raw):
    return np.array([x.decode("utf-8") for x in raw])


def ensure_dates_dataset(f, target_dates):
    if "dates" not in f:
        f.create_dataset("dates", (target_dates.shape[0],), maxshape=(None,), data=target_dates)
        return target_dates

    existing = f["dates"][()]
    if existing.shape[0] == target_dates.shape[0] and np.all(existing == target_dates):
        return existing

    if (
        f["dates"].maxshape[0] is None
        and existing.shape[0] < target_dates.shape[0]
        and np.all(existing == target_dates[: existing.shape[0]])
    ):
        old_len = existing.shape[0]
        f["dates"].resize((target_dates.shape[0],))
        f["dates"][old_len:] = target_dates[old_len:]
        return f["dates"][()]

    del f["dates"]
    f.create_dataset("dates", (target_dates.shape[0],), maxshape=(None,), data=target_dates)
    return target_dates


def ensure_data_shapes(f, old_dates_len, old_symbols_len, new_dates_len, new_symbols_len):
    if new_dates_len == old_dates_len and new_symbols_len == old_symbols_len:
        return
    for key in f.keys():
        if key in ["dates", "symbols"]:
            continue
        dset = f[key]
        if dset.ndim != 2:
            continue
        if dset.shape == (new_dates_len, new_symbols_len):
            continue
        if dset.maxshape[0] is not None and new_dates_len > dset.maxshape[0]:
            raise RuntimeError(f"{key} cannot resize dates dimension to {new_dates_len}")
        if dset.maxshape[1] is not None and new_symbols_len > dset.maxshape[1]:
            raise RuntimeError(f"{key} cannot resize symbols dimension to {new_symbols_len}")
        dset.resize((new_dates_len, new_symbols_len))


def ensure_field_dataset(f, field, n_dates, n_symbols):
    if field in f:
        return f[field]
    return f.create_dataset(
        field, (n_dates, n_symbols), maxshape=(None, None), dtype=np.float32
    )


def last_filled_date(f, dates, field="open"):
    if field not in f or dates.size == 0:
        return None
    dset = f[field]
    for idx in range(dset.shape[0] - 1, -1, -1):
        row = dset[idx]
        if np.isfinite(row).any():
            return int(dates[idx])
    return None


def fetch_daily_data(pro, trade_date, retry):
    for attempt in range(retry):
        try:
            df1 = pro.daily(trade_date=int(trade_date))
            df3 = pro.adj_factor(trade_date=int(trade_date))
            df4 = pro.stk_limit(trade_date=int(trade_date))
            df5 = pro.daily_basic(trade_date=int(trade_date))
            if df5 is not None and not df5.empty and "close" in df5.columns:
                df5 = df5.drop(columns=["close"])
            return df1, df3, df4, df5
        except Exception:
            if attempt == retry - 1:
                raise
            time.sleep(1)


def fetch_list_dates(pro):
    frames = []
    for status in ["L", "D", "P"]:
        df = pro.stock_basic(
            exchange="",
            list_status=status,
            fields="ts_code,list_date",
        )
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["new_symbol"] = data["ts_code"].astype(str).map(ts_to_h5_symbol)
    data["list_date"] = pd.to_numeric(data["list_date"], errors="coerce")
    data = data.dropna(subset=["new_symbol", "list_date"])
    data = data.drop_duplicates(subset=["new_symbol"], keep="first")
    return dict(zip(data["new_symbol"], data["list_date"].astype(int)))


def update_is_listed_in_5days(f, symbols, dates, pro):
    n_dates = dates.shape[0]
    n_symbols = symbols.shape[0]
    if "is_listed_in_5days" not in f:
        dset = f.create_dataset(
            "is_listed_in_5days",
            (n_dates, n_symbols),
            maxshape=(None, None),
            dtype=np.float32,
        )
    else:
        dset = f["is_listed_in_5days"]
        if dset.shape != (n_dates, n_symbols):
            dset.resize((n_dates, n_symbols))

    dset[:, :] = 0.0
    list_dates = fetch_list_dates(pro)
    date_set = set(int(x) for x in dates)
    for idx, sym in enumerate(symbols):
        if "BSE" in sym:
            continue
        list_date = list_dates.get(sym)
        if list_date is None or list_date not in date_set:
            continue
        start = int(np.searchsorted(dates, list_date))
        end = min(start + 5, n_dates)
        dset[start:end, idx] = 1.0


def fill_field_row(dset, date_idx, df, col, idx, n_symbols):
    row = np.full(n_symbols, np.nan, dtype=np.float32)
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
    row[idx] = np.round(values, ROUND_DECIMALS)
    dset[date_idx, :] = row


def write_df_to_dsets(f, date_idx, df, cols, symbol_to_idx, n_symbols):
    if df is None or df.empty:
        return

    mapped = df["ts_code"].astype(str).map(ts_to_h5_symbol)
    idx = mapped.map(symbol_to_idx)
    mask = idx.notna()
    if not mask.any():
        return
    idx = idx[mask].astype(int).to_numpy()
    df = df.loc[mask].reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            continue
        dset = ensure_field_dataset(f, col, f["dates"].shape[0], n_symbols)
        fill_field_row(dset, date_idx, df, col, idx, n_symbols)


def main():
    args = parse_args()

    pro = ts.pro_api(token)

    end_date = min(args.end, int(tu.todayYYYYMMDD()))
    if end_date < args.start:
        raise SystemExit("end date is before start date")

    with h5.File(args.daily, "a") as f:
        if "symbols" not in f:
            raise SystemExit("daily.hdf is missing symbols dataset")

        symbols = decode_symbols(f["symbols"][()])
        n_symbols = symbols.shape[0]
        symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

        target_dates = quant.date_range(args.start, args.end)
        existing_dates = f["dates"][()] if "dates" in f else np.array([], dtype=int)
        if existing_dates.size == 0:
            all_dates = target_dates
        else:
            if existing_dates[0] > target_dates[0]:
                raise SystemExit("existing dates start after requested start date")
            if target_dates[-1] > existing_dates[-1]:
                extra = target_dates[target_dates > existing_dates[-1]]
                all_dates = np.concatenate([existing_dates, extra])
            else:
                all_dates = existing_dates

        old_dates_len = existing_dates.shape[0]
        new_dates = ensure_dates_dataset(f, all_dates)
        ensure_data_shapes(f, old_dates_len, n_symbols, new_dates.shape[0], n_symbols)

        date_to_idx = {int(d): i for i, d in enumerate(new_dates)}

        listed_dset = f.get("is_listed_in_5days")
        if (
            listed_dset is None
            or args.refresh_listed
            or listed_dset.shape != (new_dates.shape[0], n_symbols)
        ):
            update_is_listed_in_5days(f, symbols, new_dates, pro)

        if args.update:
            last_date = last_filled_date(f, new_dates)
            start_date = args.start if last_date is None else last_date + 1
        else:
            start_date = args.start

        fetch_dates = quant.date_range(start_date, end_date)
        if fetch_dates.size == 0:
            print("no dates to fetch")
            return

        total = int(fetch_dates.shape[0])
        for i, date in enumerate(fetch_dates, start=1):
            print(f"{date} ({i}/{total})")
            df1, df3, df4, df5 = fetch_daily_data(pro, date, args.retry)
            date_idx = date_to_idx.get(int(date))
            if date_idx is None:
                continue
            write_df_to_dsets(f, date_idx, df1, DF1_COLS, symbol_to_idx, n_symbols)
            write_df_to_dsets(f, date_idx, df3, DF3_COLS, symbol_to_idx, n_symbols)
            write_df_to_dsets(f, date_idx, df4, DF4_COLS, symbol_to_idx, n_symbols)
            write_df_to_dsets(f, date_idx, df5, DF5_COLS, symbol_to_idx, n_symbols)
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
