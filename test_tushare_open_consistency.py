#!/usr/bin/env python
import time

import h5py as h5
import numpy as np
import tushare as ts

from setting import token


REQ_SLEEP = 0.15
RTOL = 1e-6
ATOL = 1e-4
SEED = 42


def to_ts_code(sym: str) -> str:
    return sym.replace("SSE", "SH").replace("SZE", "SZ").replace("BSE", "BJ")


def pick_symbol(symbols, predicate, rng):
    candidates = [s for s in symbols if predicate(s)]
    if not candidates:
        return None
    candidates = sorted(candidates)
    return rng.choice(candidates)


def pick_date_for_symbol(f, dataset_name, symbol_to_idx, symbol, rng):
    col = f[dataset_name][:, symbol_to_idx[symbol]]
    valid_idx = np.where(np.isfinite(col))[0]
    if valid_idx.size == 0:
        return None, None
    di = rng.choice(valid_idx)
    return int(di), float(col[di])


def main():
    rng = np.random.default_rng(SEED)
    pro = ts.pro_api(token)

    with h5.File("daily.hdf", "r") as fd:
        symbols = [x.decode("utf-8") for x in fd["symbols"][()]]
        dates = fd["dates"][()]
        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        boards = [
            ("mainboard", lambda s: s.endswith("SSE") and s[:3] in {"600", "601", "603", "605"}),
            ("gem", lambda s: s.endswith("SZE") and s.startswith("300")),
            ("star", lambda s: s.endswith("SSE") and s.startswith("688")),
            ("bse", lambda s: s.endswith("BSE")),
        ]

        results = []
        for name, pred in boards:
            sym = pick_symbol(symbols, pred, rng)
            if sym is None:
                raise SystemExit(f"no symbol found for {name}")
            di, local_open = pick_date_for_symbol(fd, "open", sym_to_idx, sym, rng)
            if di is None:
                raise SystemExit(f"no valid open for {sym}")
            trade_date = int(dates[di])
            df = pro.daily(ts_code=to_ts_code(sym), trade_date=trade_date)
            time.sleep(REQ_SLEEP)
            if df is None or df.empty:
                raise SystemExit(f"tushare daily missing {sym} {trade_date}")
            ts_open = float(df["open"].iloc[0])
            ok = np.isclose(local_open, ts_open, rtol=RTOL, atol=ATOL)
            results.append((name, sym, trade_date, local_open, ts_open, ok))

    with h5.File("index.daily.hdf", "r") as fi:
        idx_symbols = [x.decode("utf-8") for x in fi["symbols"][()]]
        idx_dates = fi["dates"][()]
        idx_to_idx = {s: i for i, s in enumerate(idx_symbols)}

        idx_pick = "000300.SSE" if "000300.SSE" in idx_symbols else rng.choice(sorted(idx_symbols))
        di, local_open = pick_date_for_symbol(fi, "open", idx_to_idx, idx_pick, rng)
        if di is None:
            raise SystemExit(f"no valid open for {idx_pick}")
        trade_date = int(idx_dates[di])
        df = pro.index_daily(ts_code=to_ts_code(idx_pick), trade_date=trade_date)
        time.sleep(REQ_SLEEP)
        if df is None or df.empty:
            raise SystemExit(f"tushare index_daily missing {idx_pick} {trade_date}")
        ts_open = float(df["open"].iloc[0])
        ok = np.isclose(local_open, ts_open, rtol=RTOL, atol=ATOL)
        results.append(("index", idx_pick, trade_date, local_open, ts_open, ok))

    failed = [r for r in results if not r[-1]]
    for name, sym, trade_date, local_open, ts_open, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"{status} {name} {sym} {trade_date} local={local_open} tushare={ts_open}")

    if failed:
        raise SystemExit(f"{len(failed)} mismatches")
    print("all checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
