#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass

import h5py as h5
import numpy as np
import pandas as pd


def zscore(x):
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    return (x - mean) / std


def pearson_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x0 = x[mask] - np.nanmean(x[mask])
    y0 = y[mask] - np.nanmean(y[mask])
    denom = np.sqrt(np.nansum(x0 * x0) * np.nansum(y0 * y0))
    if denom == 0:
        return np.nan
    return float(np.nansum(x0 * y0) / denom)


def rank_ic(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rx = pd.Series(x[mask]).rank(method="average").to_numpy()
    ry = pd.Series(y[mask]).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def build_forward_returns(close, holding_days):
    n_dates = close.shape[0]
    fwd = np.full_like(close, np.nan, dtype=np.float64)
    if holding_days < n_dates:
        fwd[:-holding_days, :] = close[holding_days:, :] / close[:-holding_days, :] - 1.0
    return fwd


@dataclass
class EvalResult:
    w1: float
    w2: float
    mean_abs_corr: float
    mean_ic_pred: float
    mean_rank_ic_pred: float
    mean_rank_ic_size: float
    score: float


def evaluate_weights(
    lncap,
    midcap,
    liquidity,
    volatility,
    momentum,
    fwd_ret,
    w1_list,
    w_corr,
    w_pred,
):
    results = []
    n_dates = fwd_ret.shape[0]
    for w1 in w1_list:
        w2 = 1.0 - w1
        corr_vals = []
        ic_pred_vals = []
        ric_pred_vals = []
        ric_size_vals = []

        for d in range(n_dates):
            size = w1 * lncap[d] + w2 * midcap[d]
            size_z = zscore(size)
            liq_z = zscore(liquidity[d])
            vol_z = zscore(volatility[d])
            mom_z = zscore(momentum[d])
            ret = fwd_ret[d]

            mask = np.isfinite(size_z) & np.isfinite(liq_z) & np.isfinite(vol_z) & np.isfinite(mom_z) & np.isfinite(ret)
            if mask.sum() < 30:
                continue

            s = size_z[mask]
            l = liq_z[mask]
            v = vol_z[mask]
            m = mom_z[mask]
            y = ret[mask]

            corr_vals.append(np.nanmean([abs(pearson_corr(s, l)), abs(pearson_corr(s, v)), abs(pearson_corr(s, m))]))
            ric_size_vals.append(rank_ic(s, y))

            X = np.column_stack([s, l, v, m])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ic_pred_vals.append(pearson_corr(y_hat, y))
            ric_pred_vals.append(rank_ic(y_hat, y))

        if len(corr_vals) == 0:
            continue
        mean_abs_corr = float(np.nanmean(corr_vals))
        mean_ic_pred = float(np.nanmean(ic_pred_vals))
        mean_rank_ic_pred = float(np.nanmean(ric_pred_vals))
        mean_rank_ic_size = float(np.nanmean(ric_size_vals))
        score = w_pred * mean_rank_ic_pred - w_corr * mean_abs_corr
        results.append(
            EvalResult(
                w1=w1,
                w2=w2,
                mean_abs_corr=mean_abs_corr,
                mean_ic_pred=mean_ic_pred,
                mean_rank_ic_pred=mean_rank_ic_pred,
                mean_rank_ic_size=mean_rank_ic_size,
                score=score,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize SIZE = w1*LNCAP + w2*MIDCAP with independence and prediction.")
    parser.add_argument("--h5", default="daily.hdf", help="Path to daily.hdf")
    parser.add_argument("--holding-days", type=int, default=5, help="Forward return holding days")
    parser.add_argument("--start-date", type=int, default=None, help="Start date YYYYMMDD (inclusive)")
    parser.add_argument("--end-date", type=int, default=None, help="End date YYYYMMDD (inclusive)")
    parser.add_argument("--date-step", type=int, default=1, help="Sample every N dates to speed up")
    parser.add_argument("--w1-step", type=float, default=0.02, help="Grid step for w1 in [0,1]")
    parser.add_argument("--w-corr", type=float, default=1.0, help="Penalty weight for correlation (independence)")
    parser.add_argument("--w-pred", type=float, default=1.0, help="Reward weight for prediction (rank IC)")
    parser.add_argument("--out", default="size_weight_search.csv", help="Output CSV path")
    args = parser.parse_args()

    with h5.File(args.h5, "r") as f:
        dates = f["dates"][()].astype(int)
        lncap_all = f["LNCAP"][()]
        midcap_all = f["MIDCAP"][()]
        liq_all = f["LIQUIDITY"][()]
        vol_all = f["Volatility"][()]
        mom_all = f["Momentum"][()]
        close_all = f["close"][()]

    date_mask = np.ones(dates.shape[0], dtype=bool)
    if args.start_date is not None:
        date_mask &= dates >= args.start_date
    if args.end_date is not None:
        date_mask &= dates <= args.end_date
    idx = np.where(date_mask)[0]
    if args.date_step > 1:
        idx = idx[:: args.date_step]

    if len(idx) == 0:
        raise SystemExit("No dates selected after filtering.")

    lncap = lncap_all[idx]
    midcap = midcap_all[idx]
    liquidity = liq_all[idx]
    volatility = vol_all[idx]
    momentum = mom_all[idx]
    close = close_all[idx]

    fwd_ret = build_forward_returns(close, args.holding_days)

    w1_list = np.round(np.arange(0.0, 1.0 + 1e-12, args.w1_step), 8)
    results = evaluate_weights(
        lncap,
        midcap,
        liquidity,
        volatility,
        momentum,
        fwd_ret,
        w1_list,
        w_corr=args.w_corr,
        w_pred=args.w_pred,
    )

    if not results:
        raise SystemExit("No valid results. Check data coverage and filters.")

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("score", ascending=False)
    df.to_csv(args.out, index=False)

    best = df.iloc[0]
    print("best_w1", best["w1"])
    print("best_w2", best["w2"])
    print("mean_abs_corr", best["mean_abs_corr"])
    print("mean_rank_ic_pred", best["mean_rank_ic_pred"])
    print("mean_rank_ic_size", best["mean_rank_ic_size"])
    print("score", best["score"])
    print("saved", args.out)


if __name__ == "__main__":
    main()
