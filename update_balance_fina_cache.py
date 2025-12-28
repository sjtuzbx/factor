import argparse
import os
import time
from datetime import datetime

import h5py as h5
import numpy as np
import pandas as pd
import tushare as ts

from setting import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update balancesheet and fina_indicator caches based on daily.hdf symbols/dates."
    )
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--balance-path",
        default="data/balance_sheet.20070101-20251231.csv",
        help="balancesheet cache csv path",
    )
    parser.add_argument(
        "--fina-path",
        default="data/fina_indicator.20070101-20251231.csv",
        help="fina_indicator cache csv path",
    )
    parser.add_argument(
        "--checkpoint",
        default="update_balance_fina_cache.progress",
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


def latest_ann_date(df, ts_sym):
    if df.empty or "ann_date" not in df.columns:
        return None
    x = df[df.ts_code == ts_sym]
    if x.empty:
        return None
    ann = x["ann_date"].dropna()
    if ann.empty:
        return None
    return int(ann.max())


def fetch_incremental(fetch_fn, ts_sym, start_date, end_date):
    if start_date is None:
        return fetch_fn(ts_code=ts_sym, end_date=end_date)
    return fetch_fn(ts_code=ts_sym, start_date=str(start_date), end_date=end_date)


def main():
    args = parse_args()

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)

    with h5.File(args.daily, "r") as f:
        symbols = decode_symbols(f["symbols"][()])
        dates = f["dates"][()]

    if dates.size == 0:
        raise SystemExit("daily.hdf has no dates")

    start_date = int(dates[0])
    end_date = int(dates[-1])
    end_date_str = str(end_date)

    balance_df = load_or_empty(args.balance_path)
    fina_df = load_or_empty(args.fina_path)

    pro = ts.pro_api(token)

    balance_new = []
    fina_new = []

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
        if not isinstance(sym, str):
            sym = str(sym)
        ts_sym = change_ts(sym).strip()
        if not ts_sym or ts_sym.lower() == "nan":
            print(f"skip invalid symbol at index {idx}: {sym}")
            continue

        last_bal = latest_ann_date(balance_df, ts_sym)
        last_fina = latest_ann_date(fina_df, ts_sym)

        bal_start = start_date if last_bal is None else last_bal + 1
        fina_start = start_date if last_fina is None else last_fina + 1

        if bal_start <= end_date:
            try:
                df_bal = fetch_incremental(pro.balancesheet, ts_sym, bal_start, end_date_str)
                if df_bal is not None and not df_bal.empty:
                    balance_new.append(df_bal)
            except Exception as exc:
                print(f"balancesheet failed for {ts_sym}: {exc}")

        if fina_start <= end_date:
            try:
                df_fina = fetch_incremental(pro.fina_indicator, ts_sym, fina_start, end_date_str)
                if df_fina is not None and not df_fina.empty:
                    fina_new.append(df_fina)
            except Exception as exc:
                print(f"fina_indicator failed for {ts_sym}: {exc}")

        time.sleep(args.sleep)

        if idx % 50 == 0 or idx == total:
            if balance_new:
                balance_df = pd.concat([balance_df] + balance_new, ignore_index=True)
                balance_df = balance_df.drop_duplicates()
                balance_df.to_csv(args.balance_path, index=False)
                balance_new = []
            if fina_new:
                fina_df = pd.concat([fina_df] + fina_new, ignore_index=True)
                fina_df = fina_df.drop_duplicates()
                fina_df.to_csv(args.fina_path, index=False)
                fina_new = []
            with open(args.checkpoint, "w") as f:
                f.write(str(idx))

            elapsed = time.time() - start_ts
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = (total - idx) / rate if rate > 0 else 0.0
            print(
                f"progress {idx}/{total}, elapsed {elapsed:.1f}s, "
                f"rate {rate:.2f} sym/s, eta {remaining/60:.1f} min"
            )

    if balance_new:
        balance_df = pd.concat([balance_df] + balance_new, ignore_index=True)
        balance_df = balance_df.drop_duplicates()
        balance_df.to_csv(args.balance_path, index=False)

    if fina_new:
        fina_df = pd.concat([fina_df] + fina_new, ignore_index=True)
        fina_df = fina_df.drop_duplicates()
        fina_df.to_csv(args.fina_path, index=False)


if __name__ == "__main__":
    main()
