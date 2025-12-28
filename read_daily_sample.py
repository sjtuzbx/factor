import argparse
import random
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/code/cb_cache")
import cb_cache as cbc


def parse_args():
    parser = argparse.ArgumentParser(description="Read values from a daily.hdf5 file.")
    parser.add_argument("--field", default="open", help="dataset field name, e.g. open/close/vol")
    parser.add_argument("--symbol", default=None, help="symbol like 600000.SSE")
    parser.add_argument("--date", type=int, default=None, help="trade date like 20251224")
    parser.add_argument("--random", type=int, default=5, help="number of random samples")
    return parser.parse_args()


def main():
    args = parse_args()

    symbols = cbc.get_all_symbols()
    dates = cbc.get_all_dates()
    field = args.field.lower()

    def read_field(cache, field_name):
        if field_name == "open":
            return cache.daily.open
        if field_name == "high":
            return cache.daily.high
        if field_name == "low":
            return cache.daily.low
        if field_name == "close":
            return cache.daily.close
        if field_name == "pre_close":
            return cache.daily.pre_close
        if field_name == "vol":
            return cache.daily.vol
        if field_name == "amount":
            return cache.daily.amount
        if field_name == "adj_factor":
            return cache.daily.adj_factor
        if field_name == "up_limit":
            return cache.daily.up_limit
        if field_name == "down_limit":
            return cache.daily.down_limit
        raise SystemExit(f"unsupported field: {field_name}")

    if args.symbol and args.date:
        if args.symbol not in symbols:
            raise SystemExit(f"symbol not found: {args.symbol}")
        if args.date not in dates:
            raise SystemExit(f"date not found: {args.date}")
        cache = cbc.EqCache(args.symbol, args.date, args.date)
        val = read_field(cache, field)[0, 0]
        print(args.symbol, args.date, float(val))
        return

    rng = random.Random()
    max_samples = max(0, args.random)
    for _ in range(max_samples):
        sym = rng.choice(symbols)
        if isinstance(sym, bytes):
            sym = sym.decode("utf-8")
        date = int(rng.choice(dates))
        cache = cbc.EqCache(sym, date, date)
        val = read_field(cache, field)[0, 0]
        print(sym, date, float(val))


if __name__ == "__main__":
    main()
