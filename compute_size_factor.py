import gc
import os
import sys

import h5py as h5
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/code/cb_cache")
import cb_cache as cbc  # noqa: E402


def ensure_symlink(target, link_path):
    if os.path.islink(link_path):
        if os.readlink(link_path) == target:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(f"{link_path} exists and is not a symlink")
    os.symlink(target, link_path)


def load_symbols_dates(h5_path):
    with h5.File(h5_path, "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]
    return symbols, dates


def mad_winsorize(arr, multiplier=5.0):
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    if not np.isfinite(mad) or mad == 0:
        return arr
    upper = med + multiplier * mad
    lower = med - multiplier * mad
    return np.clip(arr, lower, upper)


def compute_midcap(lncap, marcap):
    n_dates, n_syms = lncap.shape
    midcap = np.full_like(lncap, np.nan, dtype=np.float64)

    for d in range(n_dates):
        y = lncap[d] ** 3
        x = lncap[d]
        w = np.sqrt(marcap[d])
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if mask.sum() < 2:
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        W = w[mask]
        XtW = X.T * W
        beta = np.linalg.pinv(XtW @ X) @ (XtW @ y[mask])
        resid = y[mask] - (X @ beta)
        resid = mad_winsorize(resid, multiplier=5.0)
        mean = np.nanmean(resid)
        std = np.nanstd(resid)
        if not np.isfinite(std) or std == 0:
            continue
        resid = (resid - mean) / std
        midcap[d, mask] = resid
    return midcap




def ensure_dataset(f, name, shape):
    if name in f:
        dset = f[name]
        if dset.shape != shape:
            if dset.maxshape[0] is None and dset.maxshape[1] is None:
                dset.resize(shape)
            else:
                raise RuntimeError(f"{name} dataset shape mismatch: {dset.shape} vs {shape}")
        return dset
    return f.create_dataset(name, shape, maxshape=(None, None), dtype=np.float32)


def main():
    h5_path = "daily.hdf"
    # Use the local daily.hdf directly to avoid symlink indirection.
    tmp_root = os.getcwd()

    symbols, dates = load_symbols_dates(h5_path)
    # Exclude T* symbols if present.
    szsh_mask = ~np.char.startswith(symbols.astype(str), "T")
    szsh_symbols = symbols[szsh_mask]
    start_date = int(dates[0])
    end_date = int(dates[-1])

    cache = cbc.EqCache(szsh_symbols, start_date, end_date, root_path=tmp_root)
    circ_mv = cache.daily.circ_mv.astype(np.float64)
    # Tushare daily_basic circ_mv is in 10k yuan.
    marcap = circ_mv * 10000.0
    lncap_szsh = np.where(marcap > 0, np.log(marcap + 1.0), np.nan)
    midcap_szsh = compute_midcap(lncap_szsh, marcap)

    lncap = np.full((dates.size, symbols.size), np.nan, dtype=np.float64)
    midcap = np.full((dates.size, symbols.size), np.nan, dtype=np.float64)
    lncap[:, szsh_mask] = lncap_szsh
    midcap[:, szsh_mask] = midcap_szsh

    # release read-only handle before writing to daily.hdf
    del cache
    gc.collect()

    columns = [str(int(d)) for d in dates]
    lncap_df = pd.DataFrame(lncap.T, index=symbols, columns=columns)
    midcap_df = pd.DataFrame(midcap.T, index=symbols, columns=columns)

    out_dir = os.path.join(os.getcwd(), "Barra_CNE6-master", "factor_data")
    os.makedirs(out_dir, exist_ok=True)
    lncap_df.to_csv(os.path.join(out_dir, "LNCAP.csv"))
    midcap_df.to_csv(os.path.join(out_dir, "MIDCAP.csv"))
    size = 0.5 * midcap + 0.5 * lncap

    with h5.File(h5_path, "a") as f:
        if "symbols" not in f or "dates" not in f:
            raise RuntimeError("daily.hdf missing symbols/dates")
        if f["symbols"].shape[0] != symbols.shape[0] or f["dates"].shape[0] != dates.shape[0]:
            raise RuntimeError("daily.hdf symbols/dates mismatch with computed arrays")
        dset_lncap = ensure_dataset(f, "LNCAP", lncap.shape)
        dset_midcap = ensure_dataset(f, "MIDCAP", midcap.shape)
        dset_size = ensure_dataset(f, "SIZE", size.shape)
        dset_lncap[:, :] = lncap.astype(np.float32)
        dset_midcap[:, :] = midcap.astype(np.float32)
        dset_size[:, :] = size.astype(np.float32)
    note_path = os.path.join(out_dir, "SIZE_UNIT_NOTE.txt")
    with open(note_path, "w") as f:
        f.write("LNCAP uses log(circ_mv*10000 + 1). MIDCAP is cross-sectional, ")
        f.write("weighted by sqrt(marcap), MAD winsorized (5x), and z-scored.\n")
        f.write("circ_mv is based on same-day close; shift if you need next-day usage.\n")
        f.write("BSE symbols are included when data is available.\n")
    print("saved", os.path.join(out_dir, "LNCAP.csv"))
    print("saved", os.path.join(out_dir, "MIDCAP.csv"))
    print("saved", note_path)
    print("saved", h5_path, "dataset LNCAP/MIDCAP/SIZE")


if __name__ == "__main__":
    main()
