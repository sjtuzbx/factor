#!/usr/bin/env python
import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd

from python_utils.quant import lag, prev_date

import sys
sys.path.insert(0, "/home/ubuntu/code/cb_cache")
import cb_cache as cbc

INDEX_LIST = np.array([
    "000001.SSE",
    "000005.SSE",
    "000006.SSE",
    "000016.SSE",
    "000300.SSE",
    "000905.SSE",
    "399001.SZE",
    "399005.SZE",
    "399006.SZE",
    "399016.SZE",
    "399300.SZE",
    "399905.SZE",
], dtype=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update volatility factors in cache_all daily.hdf")
    parser.add_argument("--daily", default="/home/ubuntu/scripts/cache_all/daily.hdf", help="path to daily.hdf")
    parser.add_argument("--index-daily", default="/home/ubuntu/scripts/cache_all/index.daily.hdf", help="path to index.daily.hdf")
    parser.add_argument("--start-date", type=int, required=True, help="start date YYYYMMDD")
    parser.add_argument("--end-date", type=int, required=True, help="end date YYYYMMDD")
    parser.add_argument(
        "--mode",
        choices=["daily", "full"],
        default="daily",
        help="daily reads start_date-260d window; full reads from first date",
    )
    parser.add_argument(
        "--beta-impl",
        choices=["legacy", "opt"],
        default="opt",
        help="beta calculation implementation",
    )
    return parser.parse_args()


def cal_volatility(df, start_date=None, end_date=None, window=252, half_life=63):
    ret = pd.pivot_table(df, values="ret", index="time", columns="code")

    L, Lambda = 0.5 ** (1 / half_life), 0.5 ** (1 / half_life)
    W = []
    for _ in range(window):
        W.append(Lambda)
        Lambda *= L
    W = W[::-1]

    # 计算BETA、Hist_sigma、Historical_alpha
    beta, hist_sigma, hist_alpha = [], [], []
    for i in range(len(ret) - window + 1):
        tmp = ret.iloc[i : i + window, :].copy()
        missing_counts = tmp.isna().sum(axis=0)
        if tmp["000300.SSE"].iloc[1:].isna().any():
            raise ValueError(f"000300.SSE has missing values in window ending {tmp.index[-1]}")
        valid_mask = ~tmp["000300.SSE"].isna().values
        tmp_valid = tmp.loc[valid_mask]
        idx_full = []
        beta_full = pd.Series(dtype=float, name=tmp.index[-1])
        hist_sigma_full = pd.Series(dtype=float, name=tmp.index[-1])
        hist_alpha_full = pd.Series(dtype=float, name=tmp.index[-1])
        if len(tmp_valid) >= half_life:
            W_full = np.diag(np.array(W)[valid_mask])
            Y_full = tmp_valid.dropna(axis=1).drop(columns="000300.SSE", errors="ignore")
            idx_full, Y_full = Y_full.columns, Y_full.values
            X_full = np.c_[np.ones((len(tmp_valid), 1)), tmp_valid.loc[:, "000300.SSE"].values]
            beta_full = np.linalg.pinv(X_full.T @ W_full @ X_full) @ X_full.T @ W_full @ Y_full
            hist_sigma_full = pd.Series(
                np.std(Y_full - X_full @ beta_full, axis=0),
                index=idx_full,
                name=tmp.index[-1],
            )
            hist_alpha_full = pd.Series(beta_full[0], index=idx_full, name=tmp.index[-1])
            beta_full = pd.Series(beta_full[1], index=idx_full, name=tmp.index[-1])

        beta_lack, hist_sigma_lack, hist_alpha_lack = {}, {}, {}
        for c in set(tmp.columns) - set(idx_full) - set("000300.SSE"):
            missing = int(missing_counts.get(c, 0))
            if missing > window / 2:
                continue
            tmp_ = tmp.loc[:, [c, "000300.SSE"]].copy()
            if missing <= 10:
                tmp_.loc[:, c] = tmp_[c].ffill()
            tmp_.loc[:, "W"] = W
            tmp_ = tmp_.dropna()
            if len(tmp_) < half_life:
                continue
            W_lack = np.diag(tmp_["W"])
            X_lack = np.c_[np.ones(len(tmp_)), tmp_["000300.SSE"].values]
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(X_lack.T @ W_lack @ X_lack) @ X_lack.T @ W_lack @ Y_lack
            hist_sigma_lack[c] = np.std(Y_lack - X_lack @ beta_tmp)
            beta_lack[c] = beta_tmp[1]
            hist_alpha_lack[c] = beta_tmp[0]
        beta_lack = pd.Series(beta_lack, name=tmp.index[-1])
        hist_sigma_lack = pd.Series(hist_sigma_lack, name=tmp.index[-1])
        hist_alpha_lack = pd.Series(hist_alpha_lack, name=tmp.index[-1])
        beta.append(pd.concat([beta_full, beta_lack]).sort_index())
        hist_sigma.append(pd.concat([hist_sigma_full, hist_sigma_lack]).sort_index())
        hist_alpha.append(pd.concat([hist_alpha_full, hist_alpha_lack]).sort_index())
    beta = pd.concat(beta, axis=1).T
    beta = pd.melt(beta.reset_index(), id_vars="index").dropna()
    beta.columns = ["time", "code", "BETA"]
    hist_sigma = pd.concat(hist_sigma, axis=1).T
    hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars="index").dropna()
    hist_sigma.columns = ["time", "code", "Hist_sigma"]
    hist_alpha = pd.concat(hist_alpha, axis=1).T
    hist_alpha = pd.melt(hist_alpha.reset_index(), id_vars="index").dropna()
    hist_alpha.columns = ["time", "code", "Historical_alpha"]
    factor = pd.merge(beta, hist_sigma)
    factor = factor.merge(hist_alpha)

    # EWMA daily std
    init_var = ret.var(axis=0)
    L = 0.5 ** (1 / 42)
    tmp = init_var.copy()
    daily_std = {}
    for t, k in ret.iterrows():
        tmp = tmp * L + k ** 2 * (1 - L)
        daily_std[t] = np.sqrt(tmp)
        tmp = tmp.fillna(init_var)
    daily_std = pd.DataFrame(daily_std).T
    daily_std.index.name = "time"
    daily_std = daily_std.loc[start_date:end_date, :]
    daily_std = pd.melt(daily_std.reset_index(), id_vars="time", value_name="Daily_std").dropna()

    factor = factor.merge(daily_std)

    close = pd.pivot_table(df, values="close", index="time", columns="code").fillna(method="ffill", limit=10)
    pre_close = pd.pivot_table(df, values="pre_close", index="time", columns="code").fillna(method="ffill", limit=10)
    idx = close.index
    CMRA = {}
    for i in range(len(close) - window + 1):
        close_ = close.iloc[i : i + window, :]
        pre_close_ = pre_close.iloc[i, :]
        pre_close_.name = prev_date(pre_close_.name)
        close_ = pd.concat([close_, pre_close_.to_frame().T]).sort_index().iloc[list(range(0, 253, 21)), :]
        r_tau = close_.pct_change().dropna(how="all")
        Z_T = np.log(r_tau + 1).cumsum(axis=0)
        CMRA[idx[i + window - 1]] = Z_T.max(axis=0) - Z_T.min(axis=0)

    CMRA = pd.DataFrame(CMRA).T
    CMRA.index.name = "time"
    CMRA = pd.melt(CMRA.reset_index(), id_vars="time", value_name="Cumulative_range").dropna()

    factor = factor.merge(CMRA)
    return factor.sort_values(by=["time", "code"])


def cal_volatility_opt(ret, close, pre_close, start_date=None, end_date=None, window=252, half_life=63):
    ret = ret.copy()
    close = close.copy()
    pre_close = pre_close.copy()
    ret.columns.name = "code"
    close.columns.name = "code"
    pre_close.columns.name = "code"
    dates = ret.index.values

    L, Lambda = 0.5 ** (1 / half_life), 0.5 ** (1 / half_life)
    W = []
    for _ in range(window):
        W.append(Lambda)
        Lambda *= L
    W = np.array(W[::-1])
    W_sqrt = np.sqrt(W)

    # compute only windows whose end date is within [start_date, end_date]
    start_pos = int(np.searchsorted(dates, start_date))
    end_pos = int(np.searchsorted(dates, end_date))
    if end_pos < len(dates) and dates[end_pos] == end_date:
        end_pos = end_pos
    else:
        end_pos -= 1
    end_pos = max(end_pos, window - 1)
    start_pos = max(start_pos, window - 1)
    end_positions = np.arange(start_pos, end_pos + 1)
    window_starts = end_positions - window + 1

    # 计算BETA、Hist_sigma、Historical_alpha（快速加权最小二乘）
    beta, hist_sigma, hist_alpha = [], [], []
    for i, (start, end) in enumerate(zip(window_starts, end_positions)):
        if i % 5 == 0 or i == len(end_positions) - 1:
            print(f"calc BETA/Hist_sigma {i+1}/{len(end_positions)} start_idx={start} end_idx={end} date={int(dates[end])}")
        tmp = ret.iloc[start : start + window, :].copy()
        missing_counts = tmp.isna().sum(axis=0)
        if tmp["000300.SSE"].iloc[1:].isna().any():
            raise ValueError(f"000300.SSE has missing values in window ending {dates[end]}")
        valid_mask = ~tmp["000300.SSE"].isna().values
        tmp_valid = tmp.loc[valid_mask]
        idx_full = []
        beta_full = pd.Series(dtype=float, name=dates[end])
        hist_sigma_full = pd.Series(dtype=float, name=dates[end])
        hist_alpha_full = pd.Series(dtype=float, name=dates[end])
        if len(tmp_valid) >= half_life:
            Y_full = tmp_valid.dropna(axis=1).drop(columns="000300.SSE", errors="ignore")
            idx_full, Y_full = Y_full.columns, Y_full.values
            X_full = np.c_[np.ones((len(tmp_valid), 1)), tmp_valid.loc[:, "000300.SSE"].values]
            beta_full = _wls_beta(X_full, Y_full, W_sqrt[valid_mask])
            hist_sigma_full = pd.Series(
                np.std(Y_full - X_full @ beta_full, axis=0),
                index=idx_full,
                name=dates[end],
            )
            hist_alpha_full = pd.Series(beta_full[0], index=idx_full, name=dates[end])
            beta_full = pd.Series(beta_full[1], index=idx_full, name=dates[end])

        beta_lack, hist_sigma_lack, hist_alpha_lack = {}, {}, {}
        missing_cols = [c for c in tmp.columns if c not in idx_full and c != "000300.SSE"]
        if missing_cols:
            base_mask = ~tmp["000300.SSE"].isna().values
            groups = {}
            for c in missing_cols:
                missing = int(missing_counts.get(c, 0))
                if missing > window / 2:
                    continue
                if missing <= 10:
                    tmp_ = tmp.loc[:, [c, "000300.SSE"]].copy()
                    tmp_.loc[:, c] = tmp_[c].ffill()
                    col_mask = base_mask & ~tmp_[c].isna().values
                    if col_mask.sum() < half_life:
                        continue
                    X_lack = np.c_[np.ones(col_mask.sum()), tmp_.loc[col_mask, "000300.SSE"].values]
                    Y_lack = tmp_.loc[col_mask, c].values
                    W_lack_sqrt = W_sqrt[col_mask]
                    beta_tmp = _wls_beta(X_lack, Y_lack, W_lack_sqrt)
                    hist_sigma_lack[c] = np.std(Y_lack - X_lack @ beta_tmp)
                    beta_lack[c] = beta_tmp[1]
                    hist_alpha_lack[c] = beta_tmp[0]
                    continue
                col_mask = base_mask & ~tmp[c].isna().values
                if col_mask.sum() < half_life:
                    continue
                key = col_mask.tobytes()
                if key in groups:
                    groups[key][1].append(c)
                else:
                    groups[key] = [col_mask, [c]]
            for col_mask, cols in groups.values():
                X_lack = np.c_[np.ones(col_mask.sum()), tmp.loc[col_mask, "000300.SSE"].values]
                Y_lack = tmp.loc[col_mask, cols].values
                W_lack_sqrt = W_sqrt[col_mask]
                beta_tmp = _wls_beta(X_lack, Y_lack, W_lack_sqrt)
                resid = Y_lack - X_lack @ beta_tmp
                sigma = np.std(resid, axis=0)
                for j, c in enumerate(cols):
                    hist_sigma_lack[c] = sigma[j]
                    beta_lack[c] = beta_tmp[1, j]
                    hist_alpha_lack[c] = beta_tmp[0, j]
        beta_lack = pd.Series(beta_lack, name=dates[end], dtype=float)
        
        # from IPython import embed
        # embed()
        hist_sigma_lack = pd.Series(hist_sigma_lack, name=dates[end], dtype=float)
        hist_alpha_lack = pd.Series(hist_alpha_lack, name=dates[end], dtype=float)
        beta_row = pd.concat([beta_full, beta_lack]).sort_index()
        if "000001.SZE" in beta_row.index and not np.isfinite(beta_row.loc["000001.SZE"]):
            print(f"warn: 000001.SZE beta nan at {int(dates[end])}")
        beta.append(beta_row)
        hist_sigma.append(pd.concat([hist_sigma_full, hist_sigma_lack]).sort_index())
        hist_alpha.append(pd.concat([hist_alpha_full, hist_alpha_lack]).sort_index())
    
    # from IPython import embed
    # embed()
    
    beta = pd.concat(beta, axis=1).T
    beta = pd.melt(beta.reset_index(), id_vars="index").dropna()
    beta.columns = ["time", "code", "BETA"]
    hist_sigma = pd.concat(hist_sigma, axis=1).T
    hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars="index").dropna()
    hist_sigma.columns = ["time", "code", "Hist_sigma"]
    hist_alpha = pd.concat(hist_alpha, axis=1).T
    hist_alpha = pd.melt(hist_alpha.reset_index(), id_vars="index").dropna()
    hist_alpha.columns = ["time", "code", "Historical_alpha"]
    factor = pd.merge(beta, hist_sigma)
    factor = factor.merge(hist_alpha)
    
    print('factor shape 1', factor[factor.time == 20221228].shape)

    # EWMA daily std
    init_var = ret.var(axis=0)
    L = 0.5 ** (1 / 42)
    tmp = init_var.copy()
    daily_std = {}
    for i, (t, k) in enumerate(ret.iterrows()):
        if i % 200 == 0 or i == len(ret) - 1:
            print(f"calc Daily_std {i+1}/{len(ret)} date={int(t)}")
        tmp = tmp * L + k ** 2 * (1 - L)
        daily_std[t] = np.sqrt(tmp)
        tmp = tmp.fillna(init_var)
    daily_std = pd.DataFrame(daily_std).T
    daily_std.index.name = "time"
    daily_std = daily_std.loc[start_date:end_date, :]
    daily_std = pd.melt(daily_std.reset_index(), id_vars="time", value_name="Daily_std").dropna()

    factor = factor.merge(daily_std)
    print('factor shape 2', factor[factor.time == 20221228].shape)

    close = close.fillna(method="ffill", limit=10)
    pre_close = pre_close.fillna(method="ffill", limit=10)
    CMRA = {}
    for i, (start, end) in enumerate(zip(window_starts, end_positions)):
        if i % 50 == 0 or i == len(end_positions) - 1:
            print(f"calc Cumulative_range {i+1}/{len(end_positions)} start_idx={start} end_idx={end} date={int(dates[end])}")
        close_ = close.iloc[start : start + window, :]
        pre_close_ = pre_close.iloc[start, :]
        pre_close_.name = prev_date(pre_close_.name)
        close_ = pd.concat([close_, pre_close_.to_frame().T]).sort_index().iloc[list(range(0, 253, 21)), :]
        r_tau = close_.pct_change().dropna(how="all")
        Z_T = np.log(r_tau + 1).cumsum(axis=0)
        CMRA[dates[end]] = Z_T.max(axis=0) - Z_T.min(axis=0)

    CMRA = pd.DataFrame(CMRA).T
    CMRA.index.name = "time"
    CMRA = pd.melt(CMRA.reset_index(), id_vars="time", value_name="Cumulative_range").dropna()

    factor = factor.merge(CMRA)
    print('factor shape 3', factor[factor.time == 20221228].shape)
    
    return factor.sort_values(by=["time", "code"])


def _wls_beta(X, Y, W_sqrt):
    if np.isnan(X).any() or np.isnan(Y).any():
        W_full = np.diag(W_sqrt ** 2)
        return np.linalg.pinv(X.T @ W_full @ X) @ X.T @ W_full @ Y
    Xw = X * W_sqrt[:, None]
    if Y.ndim == 1:
        Yw = Y * W_sqrt
    else:
        Yw = Y * W_sqrt[:, None]
    try:
        return np.linalg.lstsq(Xw, Yw, rcond=None)[0]
    except np.linalg.LinAlgError:
        W_full = np.diag(W_sqrt ** 2)
        return np.linalg.pinv(X.T @ W_full @ X) @ X.T @ W_full @ Y


def main():
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()
    print(f"start ")

    with h5.File(args.daily, "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]

    # Exclude T* symbols from updates; daily.hdf should not include them.
    szsh_mask = ~np.char.startswith(symbols, "T")
    szsh_symbols = symbols[szsh_mask]

    start_idx = int(np.searchsorted(dates, args.start_date))
    end_idx = int(np.searchsorted(dates, args.end_date))
    
    print(f"start 2")
    
    if end_idx < len(dates) and dates[end_idx] == args.end_date:
        end_idx += 1
    if start_idx >= end_idx:
        raise SystemExit("empty date range")

    # include enough history for 252-day windows
    print(f"start 3")

    if args.mode == "full":
        st_idx = 0
    else:
        st_idx = max(0, start_idx - 260)
    st_date = int(dates[st_idx])
    ed_date = int(dates[end_idx - 1])

    print(f"loading EqCache {st_date}-{ed_date}")
    index_cache = cbc.IndexCache(INDEX_LIST, st_date, ed_date, root_path=os.path.dirname(args.index_daily))
    eq_cache = cbc.EqCache(szsh_symbols, st_date, ed_date, root_path=os.path.dirname(args.daily))

    eq_daily = eq_cache.daily
    index_daily = index_cache.daily
    eq_dates = np.array(eq_daily.dates).copy()
    eq_close = np.array(eq_daily.close).copy()
    eq_adj = np.array(eq_daily.adj_factor).copy()
    eq_is_listed = np.array(eq_daily.is_listed_in_5days, dtype=bool).copy()
    index_close = np.array(index_daily.close).copy()
    print("EqCache data loaded")

    # release HDF5 handles early; keep numpy arrays in memory
    del eq_daily
    del index_daily
    del eq_cache
    del index_cache

    adj_close = eq_close * eq_adj
    adj_close = pd.DataFrame(adj_close).fillna(method="ffill").values
    lag_adj_close = lag(adj_close, 1, axis=0)

    adj_close[eq_is_listed] = np.nan
    lag_adj_close[eq_is_listed] = np.nan
    eq_ret_dg = adj_close / lag_adj_close - 1

    lag_index_close = lag(index_close, 1, axis=0)
    index_ret_dg = index_close / lag_index_close - 1

    hs300_idx = int(np.where(INDEX_LIST == "000300.SSE")[0][0])

    start_date = int(dates[start_idx])
    end_date = int(dates[end_idx - 1])

    if args.beta_impl == "opt":
        codes = np.concatenate([szsh_symbols, np.array(["000300.SSE"], dtype=str)])
        ret_df = pd.DataFrame(
            np.column_stack([eq_ret_dg, index_ret_dg[:, hs300_idx]]),
            index=eq_dates,
            columns=codes,
        )
        close_df = pd.DataFrame(
            np.column_stack([adj_close, index_close[:, hs300_idx]]),
            index=eq_dates,
            columns=codes,
        )
        pre_close_df = pd.DataFrame(
            np.column_stack([lag_adj_close, lag_index_close[:, hs300_idx]]),
            index=eq_dates,
            columns=codes,
        )
        data = cal_volatility_opt(
            ret=ret_df,
            close=close_df,
            pre_close=pre_close_df,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        df = pd.DataFrame({"time": [], "code": [], "ret": [], "close": [], "pre_close": []})
        df["ret"] = np.concatenate([eq_ret_dg.ravel(), index_ret_dg[:, hs300_idx].ravel()])
        df["time"] = np.concatenate([
            np.repeat(eq_dates, eq_ret_dg.shape[1]),
            eq_dates,
        ])
        df["code"] = np.concatenate([
            np.tile(szsh_symbols, eq_dates.size),
            np.tile("000300.SSE", eq_dates.size),
        ])
        df["close"] = np.concatenate([adj_close.ravel(), index_close[:, hs300_idx].ravel()])
        df["pre_close"] = np.concatenate([lag_adj_close.ravel(), lag_index_close[:, hs300_idx].ravel()])
        data = cal_volatility(
            df=df,
            start_date=start_date,
            end_date=end_date,
        )

    # from IPython import embed
    # embed()
    target_sym = "000001.SZE"
    target_present = target_sym in symbols

    # write to daily.hdf
    with h5.File(args.daily, "a") as f:
        for col in ["BETA", "Hist_sigma", "Historical_alpha", "Daily_std", "Cumulative_range", "Volatility"]:
            if col not in f:
                f.create_dataset(col, (dates.shape[0], symbols.shape[0]), maxshape=(None, None))

        total = end_idx - start_idx
        update_mask = szsh_mask
        for i, date in enumerate(dates[start_idx:end_idx]):
            print('assigning date ', date)
            x = data[data.time == date]
            x.index = x.code
            y = x.reindex(symbols)
            if target_present:
                beta_val = y.loc[target_sym, "BETA"] if "BETA" in y.columns else np.nan
                if not np.isfinite(beta_val):
                    print(f"nan BETA for {target_sym} at {int(date)}")
            if y.shape[0] > 0:
                idx = np.searchsorted(dates, date)
                f["BETA"][idx, update_mask] = y["BETA"].values.astype(float)[update_mask]
                f["Hist_sigma"][idx, update_mask] = y["Hist_sigma"].values.astype(float)[update_mask]
                f["Historical_alpha"][idx, update_mask] = y["Historical_alpha"].values.astype(float)[update_mask]
                f["Daily_std"][idx, update_mask] = y["Daily_std"].values.astype(float)[update_mask]
                f["Cumulative_range"][idx, update_mask] = y["Cumulative_range"].values.astype(float)[update_mask]
                vol = (
                    0.5 * y["BETA"].values
                    + (1.0 / 6.0) * y["Hist_sigma"].values
                    + (1.0 / 6.0) * y["Daily_std"].values
                    + (1.0 / 6.0) * y["Cumulative_range"].values
                )
                f["Volatility"][idx, update_mask] = vol.astype(float)[update_mask]
            if i % 5 == 0 or i == total - 1:
                print(f"progress {i+1}/{total} date={int(date)}")

    print(f"updated volatility factors for {int(dates[start_idx])}-{int(dates[end_idx-1])}")


if __name__ == "__main__":
    main()
