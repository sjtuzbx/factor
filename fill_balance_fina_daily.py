import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd


BALANCE_META = {
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
    "update_flag",
}

FINA_META = {
    "ts_code",
    "ann_date",
    "end_date",
    "update_flag",
}

INCOME_META = {
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
    "update_flag",
}

CASHFLOW_META = {
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
    "update_flag",
}

FORECAST_META = {
    "ts_code",
    "ann_date",
    "end_date",
    "type",
    "first_ann_date",
    "summary",
    "change_reason",
    "update_flag",
}

EXPRESS_META = {
    "ts_code",
    "ann_date",
    "end_date",
    "perf_summary",
    "update_flag",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill daily.hdf with balancesheet and fina_indicator fields using announcement dates."
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
        "--income-path",
        default="data/income.20070101-20251231.csv",
        help="income cache csv path",
    )
    parser.add_argument(
        "--cashflow-path",
        default="data/cashflow.20070101-20251231.csv",
        help="cashflow cache csv path",
    )
    parser.add_argument(
        "--forecast-path",
        default="data/forecast.20070101-20251231.csv",
        help="forecast cache csv path",
    )
    parser.add_argument(
        "--express-path",
        default="data/express.20070101-20251231.csv",
        help="express cache csv path",
    )
    parser.add_argument("--balance-prefix", default="balance_", help="prefix for balancesheet fields")
    parser.add_argument("--fina-prefix", default="fina_", help="prefix for fina_indicator fields")
    parser.add_argument("--income-prefix", default="income_", help="prefix for income fields")
    parser.add_argument("--cashflow-prefix", default="cashflow_", help="prefix for cashflow fields")
    parser.add_argument("--forecast-prefix", default="forecast_", help="prefix for forecast fields")
    parser.add_argument("--express-prefix", default="express_", help="prefix for express fields")
    parser.add_argument("--round", type=int, default=3, help="round decimals for numeric fields")
    parser.add_argument("--start-index", type=int, default=0, help="start symbol index")
    parser.add_argument("--end-index", type=int, default=None, help="end symbol index (exclusive)")
    parser.add_argument(
        "--only-meta",
        action="store_true",
        help="only fill meta date columns (ann_date/end_date), skip numeric fields",
    )
    parser.add_argument(
        "--checkpoint",
        default="fill_balance_fina_daily.progress",
        help="checkpoint file path",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="delete existing checkpoint file before running",
    )
    parser.add_argument("--flush-every", type=int, default=50, help="save checkpoint every N symbols")
    return parser.parse_args()


def change_ts(sym):
    return sym.replace("SSE", "SH").replace("SZE", "SZ").replace("BSE", "BJ")


def decode_symbols(raw):
    return np.array([x.decode("utf-8") for x in raw], dtype=str)


def date_slice(valid_dates, ann_date, next_ann_date):
    start = int(np.searchsorted(valid_dates, ann_date, side="right"))
    if start >= valid_dates.size:
        return None, None
    if next_ann_date is None:
        end = valid_dates.size
    else:
        end = int(np.searchsorted(valid_dates, next_ann_date, side="right"))
    if end <= start:
        return None, None
    return start, end


def ensure_dataset(dsets, f, name, n_dates, n_symbols, dtype):
    if name in dsets:
        return dsets[name]
    if name in f:
        dsets[name] = f[name]
        return dsets[name]
    if dtype == "str":
        dset = f.create_dataset(
            name,
            (n_dates, n_symbols),
            maxshape=(None, None),
            dtype=h5.string_dtype("utf-8", 100),
        )
    elif dtype == "int":
        dset = f.create_dataset(
            name,
            (n_dates, n_symbols),
            maxshape=(None, None),
            dtype=np.int32,
        )
    else:
        dset = f.create_dataset(
            name,
            (n_dates, n_symbols),
            maxshape=(None, None),
            dtype=np.float32,
        )
    dsets[name] = dset
    return dset


def fill_table(
    f,
    valid_dates,
    symbols,
    df,
    ann_col,
    prefix,
    meta_cols,
    meta_save_cols,
    round_decimals,
    start_index,
    end_index,
    checkpoint_path,
    flush_every,
    only_meta=False,
):
    if df.empty:
        return

    df = df.copy()
    df[ann_col] = pd.to_numeric(df[ann_col], errors="coerce")
    df = df.dropna(subset=[ann_col])

    if only_meta:
        columns = []
    else:
        columns = [c for c in df.columns if c not in meta_cols]
    col_is_numeric = {}
    for col in columns:
        col_is_numeric[col] = pd.api.types.is_numeric_dtype(df[col])

    groups = df.groupby("ts_code").groups
    dsets = {}
    n_dates = valid_dates.shape[0]
    n_symbols = symbols.shape[0]

    if end_index is None or end_index > n_symbols:
        end_index = n_symbols

    for idx in range(start_index, end_index):
        sym = symbols[idx]
        ts_sym = change_ts(sym)
        if ts_sym not in groups:
            continue
        rows = df.loc[groups[ts_sym]].sort_values(by=[ann_col, "end_date"]).drop_duplicates()
        rows = rows.reset_index(drop=True)
        if rows.empty:
            continue
        derived_profit_dedtQ = []
        if (
            (not only_meta)
            and ann_col == "ann_date"
            and "profit_dedt" in rows.columns
            and "end_date" in rows.columns
        ):
            for i in range(rows.shape[0]):
                end_date = str(rows.loc[i, "end_date"])
                if len(end_date) >= 6 and end_date[4:6] == "03":
                    derived_profit_dedtQ.append(rows.loc[i, "profit_dedt"])
                else:
                    prev_idx = rows[rows["end_date"] < rows.loc[i, "end_date"]].index
                    if prev_idx.size > 0:
                        prev_val = rows.loc[prev_idx[-1], "profit_dedt"]
                        derived_profit_dedtQ.append(rows.loc[i, "profit_dedt"] - prev_val)
                    else:
                        derived_profit_dedtQ.append(rows.loc[i, "profit_dedt"])

        for i in range(rows.shape[0]):
            ann_date = int(rows.loc[i, ann_col])
            next_ann = None
            if i < rows.shape[0] - 1:
                next_ann = int(rows.loc[i + 1, ann_col])
            start, end = date_slice(valid_dates, ann_date, next_ann)
            if start is None:
                continue

            for col in columns:
                name = f"{prefix}{col}"
                if col_is_numeric[col]:
                    val = pd.to_numeric(rows.loc[i, col], errors="coerce")
                    if pd.isna(val):
                        continue
                    dset = ensure_dataset(dsets, f, name, n_dates, n_symbols, "num")
                    dset[start:end, idx] = np.round(float(val), round_decimals)
                else:
                    val = rows.loc[i, col]
                    if pd.isna(val):
                        continue
                    dset = ensure_dataset(dsets, f, name, n_dates, n_symbols, "str")
                    dset[start:end, idx] = str(val)

            for col in meta_save_cols:
                if col not in rows.columns:
                    continue
                name = f"{prefix}{col}"
                val = pd.to_numeric(rows.loc[i, col], errors="coerce")
                if pd.isna(val):
                    continue
                dset = ensure_dataset(dsets, f, name, n_dates, n_symbols, "int")
                dset[start:end, idx] = np.int32(val)

            if derived_profit_dedtQ:
                val = pd.to_numeric(derived_profit_dedtQ[i], errors="coerce")
                if not pd.isna(val):
                    dset = ensure_dataset(dsets, f, "profit_dedtQ", n_dates, n_symbols, "num")
                    dset[start:end, idx] = np.round(float(val), round_decimals)

        if (idx + 1) % flush_every == 0:
            with open(checkpoint_path, "w") as fp:
                fp.write(str(idx + 1))
            print(f"{prefix} progress {idx + 1}/{end_index}")


def main():
    args = parse_args()

    if not os.path.exists(args.daily):
        raise SystemExit(f"daily.hdf not found: {args.daily}")

    balance_df = pd.read_csv(args.balance_path) if os.path.exists(args.balance_path) else pd.DataFrame()
    fina_df = pd.read_csv(args.fina_path) if os.path.exists(args.fina_path) else pd.DataFrame()
    income_df = pd.read_csv(args.income_path) if os.path.exists(args.income_path) else pd.DataFrame()
    cashflow_df = pd.read_csv(args.cashflow_path) if os.path.exists(args.cashflow_path) else pd.DataFrame()
    forecast_df = pd.read_csv(args.forecast_path) if os.path.exists(args.forecast_path) else pd.DataFrame()
    express_df = pd.read_csv(args.express_path) if os.path.exists(args.express_path) else pd.DataFrame()

    start_index = args.start_index
    if args.reset_checkpoint and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)

    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r") as fp:
                start_index = int(fp.read().strip())
        except Exception:
            start_index = args.start_index

    with h5.File(args.daily, "a") as f:
        if "symbols" not in f or "dates" not in f:
            raise SystemExit("daily.hdf missing symbols or dates dataset")

        symbols = decode_symbols(f["symbols"][()])
        valid_dates = f["dates"][()]

        fill_table(
            f,
            valid_dates,
            symbols,
            balance_df,
            "f_ann_date",
            args.balance_prefix,
            BALANCE_META,
            {"f_ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )
        fill_table(
            f,
            valid_dates,
            symbols,
            fina_df,
            "ann_date",
            args.fina_prefix,
            FINA_META,
            {"ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )
        fill_table(
            f,
            valid_dates,
            symbols,
            income_df,
            "ann_date",
            args.income_prefix,
            INCOME_META,
            {"ann_date", "f_ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )
        fill_table(
            f,
            valid_dates,
            symbols,
            cashflow_df,
            "ann_date",
            args.cashflow_prefix,
            CASHFLOW_META,
            {"ann_date", "f_ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )
        fill_table(
            f,
            valid_dates,
            symbols,
            forecast_df,
            "ann_date",
            args.forecast_prefix,
            FORECAST_META,
            {"ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )
        fill_table(
            f,
            valid_dates,
            symbols,
            express_df,
            "ann_date",
            args.express_prefix,
            EXPRESS_META,
            {"ann_date", "end_date"},
            args.round,
            start_index,
            args.end_index,
            args.checkpoint,
            args.flush_every,
            only_meta=args.only_meta,
        )


if __name__ == "__main__":
    main()
