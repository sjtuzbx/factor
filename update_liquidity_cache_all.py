#!/usr/bin/env python
import argparse
import os

import h5py
import numpy as np


LIQUIDITY_COLS = [
    "Monthly_share_turnover",
    "Quarterly_share_turnover",
    "Annual_share_turnover",
    "Annualized_traded_value_ratio",
    "LIQUIDITY",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update liquidity factors in cache_all daily.hdf")
    parser.add_argument("--daily", default="/home/ubuntu/scripts/cache_all/daily.hdf", help="path to daily.hdf")
    parser.add_argument("--date", type=int, default=None, help="trade date YYYYMMDD; default = last date")
    parser.add_argument("--start-date", type=int, default=None, help="start date YYYYMMDD for LIQUIDITY only")
    parser.add_argument("--end-date", type=int, default=None, help="end date YYYYMMDD for LIQUIDITY only")
    parser.add_argument("--full", action="store_true", help="recompute all liquidity factors for a date range")
    return parser.parse_args()


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    cumsum = np.cumsum(filled, axis=0, dtype=np.float64)
    count = np.cumsum(valid.astype(np.int32), axis=0)
    pad = np.zeros((1, arr.shape[1]), dtype=np.float64)
    cumsum = np.vstack([pad, cumsum])
    count = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.int32), count])
    out = cumsum[window:] - cumsum[:-window]
    cnt = count[window:] - count[:-window]
    nan_pad = np.full((window - 1, arr.shape[1]), np.nan, dtype=np.float64)
    out = np.vstack([nan_pad, out])
    cnt = np.vstack([np.zeros((window - 1, arr.shape[1]), dtype=np.int32), cnt])
    out[cnt < window] = np.nan
    return out


def main() -> int:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()

    with h5py.File(args.daily, "r+") as f:
        dates = f["dates"][()]
        symbols = [x.decode("utf-8") for x in f["symbols"][()]]
        sym_mask = ~np.char.startswith(np.array(symbols, dtype=str), "T")

        if args.full:
            if args.start_date is None or args.end_date is None:
                raise SystemExit("full update requires --start-date and --end-date")
            start_idx = int(np.searchsorted(dates, args.start_date))
            end_idx = int(np.searchsorted(dates, args.end_date))
            if end_idx < len(dates) and dates[end_idx] == args.end_date:
                end_idx += 1
            if start_idx >= end_idx:
                raise SystemExit("empty date range")

            for col in LIQUIDITY_COLS:
                if col not in f:
                    f.create_dataset(col, (dates.shape[0], len(symbols)), maxshape=(None, None))

            turnover_rate = f["turnover_rate"][:, sym_mask].astype(np.float64)
            turnover_ratio = turnover_rate / 100.0

            monthly_sum = rolling_sum(turnover_ratio, 21)
            monthly_sum[monthly_sum <= 0] = np.nan
            monthly_log = np.log(monthly_sum)

            # Annualized traded value ratio with half-life weighting
            window = 252
            half_life = 63
            decay = 0.5 ** (1 / half_life)
            weights = np.array([decay ** i for i in range(window)][::-1], dtype=np.float64)
            weights = weights / np.mean(weights)

            month_end_offsets = list(range(20, 252, 21))  # 12 month-ends
            for di in range(start_idx, end_idx):
                if di < 251:
                    continue
                window_vals = turnover_ratio[di - 251 : di + 1, :]
                valid = np.isfinite(window_vals)
                weighted_sum = np.nansum(window_vals * weights.reshape(-1, 1), axis=0)
                weight_sum = np.sum(weights.reshape(-1, 1) * valid, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    annualized = weighted_sum / weight_sum * window

                mst = monthly_sum[di - 251 : di + 1, :][month_end_offsets, :]
                quarterly = np.nanmean(mst[-3:, :], axis=0)
                annual = np.nanmean(mst, axis=0)
                quarterly[quarterly <= 0] = np.nan
                annual[annual <= 0] = np.nan
                quarterly_log = np.log(quarterly)
                annual_log = np.log(annual)

                row_monthly = monthly_log[di]
                liq_stack = np.vstack([row_monthly, quarterly_log, annual_log, annualized])
                with np.errstate(invalid="ignore"):
                    liquidity = np.nanmean(liq_stack, axis=0)

                row = np.full(len(symbols), np.nan, dtype=np.float64)
                row[sym_mask] = row_monthly
                f["Monthly_share_turnover"][di, :] = row

                row = np.full(len(symbols), np.nan, dtype=np.float64)
                row[sym_mask] = quarterly_log
                f["Quarterly_share_turnover"][di, :] = row

                row = np.full(len(symbols), np.nan, dtype=np.float64)
                row[sym_mask] = annual_log
                f["Annual_share_turnover"][di, :] = row

                row = np.full(len(symbols), np.nan, dtype=np.float64)
                row[sym_mask] = annualized
                f["Annualized_traded_value_ratio"][di, :] = row

                row = np.full(len(symbols), np.nan, dtype=np.float64)
                row[sym_mask] = liquidity
                f["LIQUIDITY"][di, :] = row

                if (di - start_idx) % 50 == 0:
                    print(f"processed {int(dates[di])}")

            print(f"full update done for {int(dates[start_idx])}-{int(dates[end_idx-1])}")
            return 0

        if args.start_date is not None or args.end_date is not None:
            if args.start_date is None or args.end_date is None:
                raise SystemExit("both --start-date and --end-date are required")
            if "LIQUIDITY" not in f:
                f.create_dataset("LIQUIDITY", (dates.shape[0], len(symbols)), maxshape=(None, None))
            for col in LIQUIDITY_COLS:
                if col not in f:
                    raise SystemExit(f"missing factor in daily.hdf: {col}")
            start_idx = int(np.searchsorted(dates, args.start_date))
            end_idx = int(np.searchsorted(dates, args.end_date))
            if end_idx < len(dates) and dates[end_idx] == args.end_date:
                end_idx += 1
            if start_idx >= end_idx:
                raise SystemExit("empty date range")

            stack = np.stack(
                [
                    f["Monthly_share_turnover"][start_idx:end_idx, :],
                    f["Quarterly_share_turnover"][start_idx:end_idx, :],
                    f["Annual_share_turnover"][start_idx:end_idx, :],
                    f["Annualized_traded_value_ratio"][start_idx:end_idx, :],
                ],
                axis=0,
            )
            with np.errstate(invalid="ignore"):
                liquidity = np.nanmean(stack, axis=0)
            f["LIQUIDITY"][start_idx:end_idx, :] = liquidity
            print(f"updated LIQUIDITY for {int(dates[start_idx])}-{int(dates[end_idx-1])}")
            return 0

        if args.date is None:
            target_date = int(dates[-1])
        else:
            target_date = args.date

        if target_date not in dates:
            raise SystemExit(f"date not found in daily.hdf: {target_date}")

        date_idx = int(np.where(dates == target_date)[0][0])
        if date_idx < 251:
            raise SystemExit("date is too early; need at least 252 trading days for liquidity factors")

        symbols_use = np.array(symbols)[sym_mask]

        turnover_rate = f["turnover_rate"][:, sym_mask].astype(np.float64)
        turnover_ratio = turnover_rate / 100.0

        # Monthly share turnover: log of 21-day sum
        monthly_sum = rolling_sum(turnover_ratio, 21)
        monthly_sum[monthly_sum <= 0] = np.nan
        monthly_log = np.log(monthly_sum)
        monthly_val = monthly_log[date_idx]

        # Quarterly / annual: use month-end positions in the 252-day window
        start_idx = date_idx - 251
        month_end_offsets = list(range(20, 252, 21))  # 12 month-ends
        month_end_idx = [start_idx + o for o in month_end_offsets]
        mst = monthly_sum[month_end_idx, :]
        quarterly = np.nanmean(mst[-3:, :], axis=0)
        annual = np.nanmean(mst, axis=0)
        quarterly[quarterly <= 0] = np.nan
        annual[annual <= 0] = np.nan
        quarterly_log = np.log(quarterly)
        annual_log = np.log(annual)

        # Annualized traded value ratio (252-day half-life weighting)
        window = 252
        half_life = 63
        decay = 0.5 ** (1 / half_life)
        weights = np.array([decay ** i for i in range(window)][::-1], dtype=np.float64)
        weights = weights / np.mean(weights)
        window_vals = turnover_ratio[date_idx - 251 : date_idx + 1, :]
        with np.errstate(invalid="ignore"):
            annualized = np.nanmean(window_vals * weights.reshape(-1, 1), axis=0) * window

        # Write back
        for col in LIQUIDITY_COLS:
            if col not in f:
                f.create_dataset(col, (dates.shape[0], len(symbols)), maxshape=(None, None))

        def write_col(col_name: str, data: np.ndarray) -> None:
            row = np.full(len(symbols), np.nan, dtype=np.float64)
            row[sym_mask] = data
            f[col_name][date_idx, :] = row

        write_col("Monthly_share_turnover", monthly_val)
        write_col("Quarterly_share_turnover", quarterly_log)
        write_col("Annual_share_turnover", annual_log)
        write_col("Annualized_traded_value_ratio", annualized)

        liq_stack = np.vstack([monthly_val, quarterly_log, annual_log, annualized])
        with np.errstate(invalid="ignore"):
            liquidity = np.nanmean(liq_stack, axis=0)
        write_col("LIQUIDITY", liquidity)

        print(f"updated liquidity factors for {target_date}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
