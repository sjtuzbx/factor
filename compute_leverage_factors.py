#!/usr/bin/env python3
import argparse

import h5py as h5
import numpy as np


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


def safe_divide(num, den):
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute leverage factors: MLEV, BLEV, DTOA.")
    parser.add_argument("--h5", default="daily.hdf", help="Path to daily.hdf")
    parser.add_argument("--me-dataset", default="total_mv", help="Market equity dataset")
    parser.add_argument("--be-dataset", default="balance_total_hldr_eqy_exc_min_int", help="Book equity dataset")
    parser.add_argument("--tl-dataset", default="balance_total_liab", help="Total liabilities dataset")
    parser.add_argument("--ta-dataset", default="balance_total_assets", help="Total assets dataset")
    parser.add_argument("--tcl-dataset", default="balance_total_cur_liab", help="Total current liabilities dataset")
    parser.add_argument("--ld-dataset", default="", help="Long-term debt/non-current liability dataset (optional)")
    parser.add_argument("--pe-dataset", default="balance_oth_eqt_tools_p_shr", help="Preferred equity dataset (optional)")
    parser.add_argument("--pe-lyr", action="store_true", help="Use previous fiscal year-end value for PE")
    parser.add_argument("--me-shift", type=int, default=1, help="Shift ME by N days (default 1)")
    parser.add_argument("--out-mlev", default="MLEV", help="Output dataset name for market leverage")
    parser.add_argument("--out-blev", default="BLEV", help="Output dataset name for book leverage")
    parser.add_argument("--out-dtoa", default="DTOA", help="Output dataset name for debt-to-assets")
    args = parser.parse_args()

    with h5.File(args.h5, "r") as f:
        dates = f["dates"][()].astype(int)
        me = f[args.me_dataset][()].astype(np.float64)
        be = f[args.be_dataset][()].astype(np.float64)
        tl = f[args.tl_dataset][()].astype(np.float64)
        ta = f[args.ta_dataset][()].astype(np.float64)
        tcl = f[args.tcl_dataset][()].astype(np.float64)

        if args.ld_dataset and args.ld_dataset in f:
            ld = f[args.ld_dataset][()].astype(np.float64)
        else:
            ld = tl - tcl

        if args.pe_dataset and args.pe_dataset in f:
            pe = f[args.pe_dataset][()].astype(np.float64)
        else:
            pe = np.zeros_like(ld, dtype=np.float64)

    if args.me_shift:
        me_shifted = np.full_like(me, np.nan, dtype=np.float64)
        if me.shape[0] > args.me_shift:
            me_shifted[args.me_shift :, :] = me[:-args.me_shift, :]
        me = me_shifted

    mlev_num = me + pe + ld
    blev_num = be + pe + ld
    mlev = safe_divide(mlev_num, me)
    blev = safe_divide(blev_num, be)
    dtoa = safe_divide(tl, ta)

    with h5.File(args.h5, "a") as f:
        dset_mlev = ensure_dataset(f, args.out_mlev, mlev.shape)
        dset_blev = ensure_dataset(f, args.out_blev, blev.shape)
        dset_dtoa = ensure_dataset(f, args.out_dtoa, dtoa.shape)
        dset_mlev[:, :] = mlev.astype(np.float32)
        dset_blev[:, :] = blev.astype(np.float32)
        dset_dtoa[:, :] = dtoa.astype(np.float32)

    print("saved", args.out_mlev, args.out_blev, args.out_dtoa, "in", args.h5)
    if not args.ld_dataset:
        print("LD = TL - TCL (non-current liabilities), because --ld-dataset not provided.")
    if not args.pe_dataset:
        print("PE = 0, because --pe-dataset not provided.")
    elif args.pe_lyr:
        print("PE uses previous fiscal year-end value (--pe-lyr).")


if __name__ == "__main__":
    main()
    if args.pe_lyr:
        years = dates // 10000
        pe_lyr = np.full_like(pe, np.nan, dtype=np.float64)
        year_to_last_idx = {}
        for y in np.unique(years):
            idx_y = np.where(years == y)[0]
            if idx_y.size == 0:
                continue
            year_to_last_idx[y] = idx_y[-1]
        for y in np.unique(years):
            prev_y = y - 1
            if prev_y not in year_to_last_idx:
                continue
            src_idx = year_to_last_idx[prev_y]
            tgt_idx = np.where(years == y)[0]
            if tgt_idx.size == 0:
                continue
            pe_lyr[tgt_idx, :] = pe[src_idx, :]
        pe = pe_lyr
