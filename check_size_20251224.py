import os
import sys

import h5py as h5
import numpy as np

sys.path.insert(0, "/home/ubuntu/code/cb_cache")
import cb_cache as cbc  # noqa: E402


def main():
    h5_path = "daily.hdf"
    target_date = 20251224

    with h5.File(h5_path, "r") as f:
        dates = f["dates"][()]
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        idx = np.searchsorted(dates, target_date)
        if idx >= len(dates) or int(dates[idx]) != target_date:
            raise SystemExit(f"date {target_date} not found in daily.hdf")
        target_symbols = symbols[:10]

    cache = cbc.EqCache(target_symbols, target_date, target_date, root_path=os.getcwd())
    size_vals = cache.daily.SIZE[0]
    lncap_vals = cache.daily.LNCAP[0]
    midcap_vals = cache.daily.MIDCAP[0]
    non_nan = np.where(np.isfinite(size_vals))[0]
    print("date", target_date, "symbols", len(target_symbols), "non_nan", non_nan.size)
    for i in non_nan:
        print(target_symbols[i], float(size_vals[i]), float(lncap_vals[i]), float(midcap_vals[i]))


if __name__ == "__main__":
    main()
