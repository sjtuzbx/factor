import time

import h5py as h5
import numpy as np
import tushare as ts

from setting import token


def main():
    with h5.File("daily.hdf", "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])

    pro = ts.pro_api(token)

    n = 10
    start = time.time()
    for sym in symbols[:n]:
        ts_sym = sym.replace("SSE", "SH").replace("SZE", "SZ").replace("BSE", "BJ")
        pro.report_rc(ts_code=ts_sym, end_date="20251224")
    end = time.time()

    print(f"{n} symbols took {end-start:.2f}s ({(end-start)/n:.2f}s/symbol)")


if __name__ == "__main__":
    main()
