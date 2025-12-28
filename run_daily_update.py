#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily update pipeline for daily/index and factor caches."
    )
    parser.add_argument("--date", type=int, default=None, help="single trade date YYYYMMDD")
    parser.add_argument("--start-date", type=int, default=None, help="start date YYYYMMDD")
    parser.add_argument("--end-date", type=int, default=None, help="end date YYYYMMDD")
    parser.add_argument("--daily", default="/home/ubuntu/scripts/cache_all/daily.hdf")
    parser.add_argument("--index", default="/home/ubuntu/scripts/cache_all/index.daily.hdf")
    parser.add_argument("--refresh-listed", action="store_true", help="recompute is_listed_in_5days")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-liquidity", action="store_true")
    parser.add_argument("--skip-vol", action="store_true")
    parser.add_argument("--skip-size", action="store_true")
    return parser.parse_args()


def load_dates(daily_path):
    with h5py.File(daily_path, "r") as f:
        dates = f["dates"][()]
    return dates


def get_date_list(args, dates):
    if args.date is not None:
        if args.date not in dates:
            raise SystemExit(f"date not found in daily.hdf: {args.date}")
        return [int(args.date)]
    if args.start_date is None or args.end_date is None:
        raise SystemExit("use --date or both --start-date and --end-date")
    start_idx = int(np.searchsorted(dates, args.start_date))
    end_idx = int(np.searchsorted(dates, args.end_date))
    if end_idx < len(dates) and dates[end_idx] == args.end_date:
        end_idx += 1
    if start_idx >= end_idx:
        raise SystemExit("empty date range")
    return [int(x) for x in dates[start_idx:end_idx]]


def run(cmd, env=None):
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = parse_args()
    dates = load_dates(args.daily)
    date_list = get_date_list(args, dates)

    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    py = sys.executable

    if not args.skip_daily:
        cmd = [
            py,
            "/home/ubuntu/scripts/cache_all/update_daily_cache.py",
            "--daily",
            args.daily,
            "--start",
            str(date_list[0]),
            "--end",
            str(date_list[-1]),
        ]
        if args.refresh_listed:
            cmd.append("--refresh-listed")
        run(cmd, env=env)

    if not args.skip_index:
        cmd = [
            py,
            "/home/ubuntu/scripts/cache_all/generate_index_daily_hdf5.py",
            "--out",
            args.index,
            "--start",
            str(date_list[0]),
            "--end",
            str(date_list[-1]),
        ]
        run(cmd, env=env)

    if not args.skip_liquidity:
        for d in date_list:
            cmd = [
                py,
                "/home/ubuntu/scripts/cache_all/update_liquidity_cache_all.py",
                "--date",
                str(d),
            ]
            run(cmd, env=env)

    if not args.skip_vol:
        cmd = [
            py,
            "/home/ubuntu/scripts/cache_all/update_vol_cache_all.py",
            "--start-date",
            str(date_list[0]),
            "--end-date",
            str(date_list[-1]),
        ]
        run(cmd, env=env)

    if not args.skip_size:
        cmd = [py, "/home/ubuntu/scripts/cache_all/compute_size_factor.py"]
        run(cmd, env=env)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
