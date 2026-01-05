import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute analyst NP std / market cap for all dates and symbols."
    )
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--report",
        default="data/report_rc.20140101-20251224.csv",
        help="report_rc csv path",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=365,
        help="lookback window in days (default 365)",
    )
    parser.add_argument(
        "--dataset",
        default="ANALYST_NP_STD_MV",
        help="output dataset name in daily.hdf",
    )
    parser.add_argument(
        "--dataset-fwd12",
        default="ANALYST_NP_FWD12M_MEAN",
        help="output dataset name for fwd 12m NP mean (Q4 this year + next year)/2",
    )
    parser.add_argument(
        "--dataset-cagr",
        default="ANALYST_NP_CAGR_2Y",
        help="output dataset name for 2Y CAGR using Q4 this year -> Q4+2",
    )
    parser.add_argument(
        "--dataset-rribs",
        default="ANALYST_NP_RRIBS",
        help="output dataset name for revision ratio (RRIBS) using NP up/down",
    )
    parser.add_argument(
        "--dataset-ep-change",
        default="ANALYST_EP_CHANGE",
        help="output dataset name for EP change factor",
    )
    parser.add_argument(
        "--dataset-eps-change",
        default="ANALYST_EPS_CHANGE",
        help="output dataset name for EPS change factor",
    )
    parser.add_argument(
        "--dataset-rd-mean",
        default="ANALYST_RD_6M_MEAN",
        help="output dataset name for 6-month mean of rd",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=6,
        help="lookback months for fwd12m/cagr-style metrics (default 6)",
    )
    parser.add_argument(
        "--checkpoint",
        default="update_analyst_np_std_mv.progress",
        help="checkpoint file path",
    )
    return parser.parse_args()


def to_h5_symbol(ts_code):
    return ts_code.replace("SH", "SSE").replace("SZ", "SZE").replace("BJ", "BSE")


def _window_sum(cum, end, window):
    if end < 0:
        return None
    start = end - window + 1
    if start <= 0:
        return cum[end]
    return cum[end] - cum[start - 1]


def _window_mean(sum_cum, cnt_cum, end, window):
    if end < 0:
        return None
    sum_w = _window_sum(sum_cum, end, window)
    cnt_w = _window_sum(cnt_cum, end, window)
    if sum_w is None or cnt_w is None:
        return None
    mean = np.full_like(sum_w, np.nan, dtype=np.float64)
    mask = cnt_w > 0
    mean[mask] = sum_w[mask] / cnt_w[mask]
    return mean


def main():
    args = parse_args()

    report = pd.read_csv(args.report, usecols=["ts_code", "report_date", "quarter", "np", "eps", "rd"])
    report["report_date"] = report["report_date"].astype(str)
    report = report[report["report_date"].str.isnumeric()]
    report["report_date"] = report["report_date"].astype(int)
    report["quarter"] = report["quarter"].astype(str)
    report["h5_code"] = report["ts_code"].astype(str).map(to_h5_symbol)

    with h5.File(args.daily, "a") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]
        total_mv = f["total_mv"][()]

        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        report["sym_idx"] = report["h5_code"].map(sym_to_idx).fillna(-1).astype(int)

        shape = (dates.shape[0], symbols.shape[0])
        if args.dataset in f:
            dset = f[args.dataset]
            if dset.shape != shape:
                raise RuntimeError("dataset shape mismatch")
        else:
            dset = f.create_dataset(args.dataset, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_fwd12 in f:
            dset_fwd12 = f[args.dataset_fwd12]
            if dset_fwd12.shape != shape:
                raise RuntimeError("fwd12 dataset shape mismatch")
        else:
            dset_fwd12 = f.create_dataset(args.dataset_fwd12, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_cagr in f:
            dset_cagr = f[args.dataset_cagr]
            if dset_cagr.shape != shape:
                raise RuntimeError("cagr dataset shape mismatch")
        else:
            dset_cagr = f.create_dataset(args.dataset_cagr, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_rribs in f:
            dset_rribs = f[args.dataset_rribs]
            if dset_rribs.shape != shape:
                raise RuntimeError("rribs dataset shape mismatch")
        else:
            dset_rribs = f.create_dataset(args.dataset_rribs, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_ep_change in f:
            dset_ep = f[args.dataset_ep_change]
            if dset_ep.shape != shape:
                raise RuntimeError("ep change dataset shape mismatch")
        else:
            dset_ep = f.create_dataset(args.dataset_ep_change, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_eps_change in f:
            dset_eps = f[args.dataset_eps_change]
            if dset_eps.shape != shape:
                raise RuntimeError("eps change dataset shape mismatch")
        else:
            dset_eps = f.create_dataset(args.dataset_eps_change, shape, maxshape=(None, None), dtype=np.float32)

        if args.dataset_rd_mean in f:
            dset_rd = f[args.dataset_rd_mean]
            if dset_rd.shape != shape:
                raise RuntimeError("rd mean dataset shape mismatch")
        else:
            dset_rd = f.create_dataset(args.dataset_rd_mean, shape, maxshape=(None, None), dtype=np.float32)

        # map report_date to nearest trading date index
        date_idx = np.searchsorted(dates, report["report_date"].values, side="right") - 1
        report["date_idx"] = date_idx
        report = report[report["date_idx"] >= 0]

        # compute up/down based on NP change vs previous report of same quarter
        report = report.sort_values(["sym_idx", "quarter", "report_date"])
        report["prev_np"] = report.groupby(["sym_idx", "quarter"])["np"].shift(1)
        report["up"] = (report["np"] > report["prev_np"]).astype(int)
        report["down"] = (report["np"] < report["prev_np"]).astype(int)
        report["valid_ud"] = report["up"] + report["down"]

        # aggregate counts per date/symbol
        up_counts = np.zeros(shape, dtype=np.int32)
        down_counts = np.zeros(shape, dtype=np.int32)
        total_counts = np.zeros(shape, dtype=np.int32)
        if not report.empty:
            idx_pairs = report[["date_idx", "sym_idx"]].values.astype(int)
            np.add.at(up_counts, (idx_pairs[:, 0], idx_pairs[:, 1]), report["up"].values.astype(int))
            np.add.at(down_counts, (idx_pairs[:, 0], idx_pairs[:, 1]), report["down"].values.astype(int))
            np.add.at(total_counts, (idx_pairs[:, 0], idx_pairs[:, 1]), report["valid_ud"].values.astype(int))

        up_cum = np.cumsum(up_counts, axis=0)
        down_cum = np.cumsum(down_counts, axis=0)
        total_cum = np.cumsum(total_counts, axis=0)

        # prepare EP/EPS daily sums for rolling 63-day means
        ep_sum = np.zeros(shape, dtype=np.float64)
        ep_cnt = np.zeros(shape, dtype=np.int32)
        eps_sum = np.zeros(shape, dtype=np.float64)
        eps_cnt = np.zeros(shape, dtype=np.int32)
        rd_sum = np.zeros(shape, dtype=np.float64)
        rd_cnt = np.zeros(shape, dtype=np.int32)
        if not report.empty:
            mv = total_mv[report["date_idx"].values, report["sym_idx"].values]
            np_vals = report["np"].values.astype(np.float64)
            eps_vals = report["eps"].values.astype(np.float64)
            rd_vals = report["rd"].values.astype(np.float64)
            ep_vals = np.full_like(np_vals, np.nan, dtype=np.float64)
            mask = np.isfinite(np_vals) & np.isfinite(mv) & (mv > 0)
            ep_vals[mask] = np_vals[mask] / mv[mask]
            idx_pairs = report[["date_idx", "sym_idx"]].values.astype(int)
            valid_ep = np.isfinite(ep_vals)
            if valid_ep.any():
                np.add.at(ep_sum, (idx_pairs[valid_ep, 0], idx_pairs[valid_ep, 1]), ep_vals[valid_ep])
                np.add.at(ep_cnt, (idx_pairs[valid_ep, 0], idx_pairs[valid_ep, 1]), 1)
            valid_eps = np.isfinite(eps_vals)
            if valid_eps.any():
                np.add.at(eps_sum, (idx_pairs[valid_eps, 0], idx_pairs[valid_eps, 1]), eps_vals[valid_eps])
                np.add.at(eps_cnt, (idx_pairs[valid_eps, 0], idx_pairs[valid_eps, 1]), 1)
            valid_rd = np.isfinite(rd_vals)
            if valid_rd.any():
                np.add.at(rd_sum, (idx_pairs[valid_rd, 0], idx_pairs[valid_rd, 1]), rd_vals[valid_rd])
                np.add.at(rd_cnt, (idx_pairs[valid_rd, 0], idx_pairs[valid_rd, 1]), 1)

        ep_sum_cum = np.cumsum(ep_sum, axis=0)
        ep_cnt_cum = np.cumsum(ep_cnt, axis=0)
        eps_sum_cum = np.cumsum(eps_sum, axis=0)
        eps_cnt_cum = np.cumsum(eps_cnt, axis=0)
        rd_sum_cum = np.cumsum(rd_sum, axis=0)
        rd_cnt_cum = np.cumsum(rd_cnt, axis=0)

        start_idx = 0
        if os.path.exists(args.checkpoint):
            try:
                with open(args.checkpoint, "r") as ck:
                    start_idx = int(ck.read().strip())
            except Exception:
                start_idx = 0
        if start_idx < 0 or start_idx >= dates.shape[0]:
            start_idx = 0

        for i in range(start_idx, dates.shape[0]):
            date = int(dates[i])
            window_start = int(
                (pd.to_datetime(str(date)) - pd.Timedelta(days=args.window_days - 1)).strftime("%Y%m%d")
            )
            lookback_start = int(
                (pd.to_datetime(str(date)) - pd.DateOffset(months=args.lookback_months)).strftime("%Y%m%d")
            )
            year = str(date)[:4]
            q4 = f"{year}Q4"

            sub = report[
                (report["report_date"] >= window_start)
                & (report["report_date"] <= date)
                & (report["quarter"] == q4)
                & (report["sym_idx"] >= 0)
            ]

            row = np.full(symbols.shape[0], np.nan, dtype=np.float32)
            if not sub.empty:
                grp = sub.groupby("sym_idx", as_index=False)["np"].std()
                mv = total_mv[i]
                for _, r in grp.iterrows():
                    idx = int(r["sym_idx"])
                    val = r["np"]
                    if np.isfinite(val) and np.isfinite(mv[idx]) and mv[idx] > 0:
                        row[idx] = float(val / mv[idx])

            dset[i, :] = row

            year = int(str(date)[:4])
            q4_this = f"{year}Q4"
            q4_next = f"{year + 1}Q4"
            q4_next2 = f"{year + 2}Q4"
            sub2 = report[
                (report["report_date"] <= date)
                & (report["quarter"].isin([q4_this, q4_next]))
                & (report["sym_idx"] >= 0)
            ]
            row_fwd12 = np.full(symbols.shape[0], np.nan, dtype=np.float32)
            if not sub2.empty:
                sub2 = sub2[sub2["report_date"] >= lookback_start].sort_values("report_date")
                latest = sub2.groupby(["sym_idx", "quarter"], as_index=False).tail(1)
                pivot = latest.pivot(index="sym_idx", columns="quarter", values="np")
                if q4_this in pivot.columns and q4_next in pivot.columns:
                    mean_np = (pivot[q4_this] + pivot[q4_next]) / 2.0
                    for idx, val in mean_np.dropna().items():
                        row_fwd12[int(idx)] = float(val)
            dset_fwd12[i, :] = row_fwd12

            sub3 = report[
                (report["report_date"] <= date)
                & (report["report_date"] >= lookback_start)
                & (report["quarter"].isin([q4_this, q4_next2]))
                & (report["sym_idx"] >= 0)
            ].sort_values("report_date")
            row_cagr = np.full(symbols.shape[0], np.nan, dtype=np.float32)
            if not sub3.empty:
                latest = sub3.groupby(["sym_idx", "quarter"], as_index=False).tail(1)
                pivot = latest.pivot(index="sym_idx", columns="quarter", values="np")
                if q4_this in pivot.columns and q4_next2 in pivot.columns:
                    base = pivot[q4_this]
                    nxt2 = pivot[q4_next2]
                    valid = base > 0
                    cagr = (nxt2[valid] / base[valid]) ** 0.5 - 1.0
                    for idx, val in cagr.dropna().items():
                        row_cagr[int(idx)] = float(val)
            dset_cagr[i, :] = row_cagr

            # RRIBS (Revision ratio) using 21-day blocks, L={0,1,2}, W={3,2,1}
            rribs = np.full(symbols.shape[0], np.nan, dtype=np.float64)
            weights = [3.0, 2.0, 1.0]
            rr_sum = np.zeros(symbols.shape[0], dtype=np.float64)
            rr_w = 0.0
            for l, w in enumerate(weights):
                end = i - l * 21
                if end < 0:
                    continue
                up_w = _window_sum(up_cum, end, 21)
                down_w = _window_sum(down_cum, end, 21)
                total_w = _window_sum(total_cum, end, 21)
                if up_w is None or down_w is None or total_w is None:
                    continue
                mask = total_w > 0
                if not mask.any():
                    continue
                rr = np.full_like(rribs, np.nan, dtype=np.float64)
                rr[mask] = (up_w[mask] - down_w[mask]) / total_w[mask]
                rr_sum[mask] += w * rr[mask]
                rr_w += w
            if rr_w > 0:
                rribs = rr_sum / rr_w
            dset_rribs[i, :] = rribs.astype(np.float32)

            # EP change and EPS change using 63-day blocks, L={0,1,2,3}, W={9,7,5,3}
            weights = [9.0, 7.0, 5.0, 3.0]
            ep_change = np.full(symbols.shape[0], np.nan, dtype=np.float64)
            eps_change = np.full(symbols.shape[0], np.nan, dtype=np.float64)
            ep_sum_w = np.zeros(symbols.shape[0], dtype=np.float64)
            eps_sum_w = np.zeros(symbols.shape[0], dtype=np.float64)
            ep_w = 0.0
            eps_w = 0.0
            for l, w in enumerate(weights):
                end = i - l * 63
                prev_end = i - (l + 1) * 63
                if end < 0 or prev_end < 0:
                    continue
                ep_curr = _window_mean(ep_sum_cum, ep_cnt_cum, end, 63)
                ep_prev = _window_mean(ep_sum_cum, ep_cnt_cum, prev_end, 63)
                if ep_curr is not None and ep_prev is not None:
                    mask = np.isfinite(ep_curr) & np.isfinite(ep_prev) & (ep_prev != 0)
                    if mask.any():
                        ep_delta = np.full_like(ep_curr, np.nan, dtype=np.float64)
                        ep_delta[mask] = (ep_curr[mask] - ep_prev[mask]) / ep_prev[mask]
                        ep_sum_w[mask] += w * ep_delta[mask]
                        ep_w += w
                eps_curr = _window_mean(eps_sum_cum, eps_cnt_cum, end, 63)
                eps_prev = _window_mean(eps_sum_cum, eps_cnt_cum, prev_end, 63)
                if eps_curr is not None and eps_prev is not None:
                    mask = np.isfinite(eps_curr) & np.isfinite(eps_prev) & (eps_prev != 0)
                    if mask.any():
                        eps_delta = np.full_like(eps_curr, np.nan, dtype=np.float64)
                        eps_delta[mask] = (eps_curr[mask] - eps_prev[mask]) / eps_prev[mask]
                        eps_sum_w[mask] += w * eps_delta[mask]
                        eps_w += w
            if ep_w > 0:
                ep_change = ep_sum_w / ep_w
            if eps_w > 0:
                eps_change = eps_sum_w / eps_w
            dset_ep[i, :] = ep_change.astype(np.float32)
            dset_eps[i, :] = eps_change.astype(np.float32)

            start_idx_rd = int(np.searchsorted(dates, lookback_start, side="left"))
            if start_idx_rd > i:
                start_idx_rd = i
            if start_idx_rd > 0:
                rd_sum_w = rd_sum_cum[i] - rd_sum_cum[start_idx_rd - 1]
                rd_cnt_w = rd_cnt_cum[i] - rd_cnt_cum[start_idx_rd - 1]
            else:
                rd_sum_w = rd_sum_cum[i]
                rd_cnt_w = rd_cnt_cum[i]
            rd_mean = np.full(symbols.shape[0], np.nan, dtype=np.float64)
            mask = rd_cnt_w > 0
            if mask.any():
                rd_mean[mask] = rd_sum_w[mask] / rd_cnt_w[mask]
            dset_rd[i, :] = rd_mean.astype(np.float32)

            if i % 10 == 0 or i == dates.shape[0] - 1:
                with open(args.checkpoint, "w") as ck:
                    ck.write(str(i))
                print(f"{date} ({i+1}/{dates.shape[0]})")

    print("saved", args.dataset, "to", args.daily)


if __name__ == "__main__":
    main()
