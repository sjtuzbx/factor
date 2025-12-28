import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd

import cb_cache as cbc
from python_utils.quant import lag, prev_date
from python_utils import time_utils as tu


def parse_args():
    parser = argparse.ArgumentParser(description="Update volatility factors in daily.hdf.")
    parser.add_argument("--daily", default="daily.hdf", help="path to daily.hdf")
    parser.add_argument(
        "--index-root",
        default="/mnt/sse_data/eq_cache",
        help="root path that contains index.daily.hdf",
    )
    parser.add_argument("--target-date", type=int, default=None, help="YYYYMMDD")
    return parser.parse_args()


def load_symbols_dates(h5_path):
    with h5.File(h5_path, "r") as f:
        symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]
    return symbols, dates


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


def cal_volatility(df, start_date=None, end_date=None, window=252, half_life=63):
    ret = pd.pivot_table(df, values="ret", index="time", columns="code")

    decay = 0.5 ** (1 / half_life)
    w = []
    lam = decay
    for _ in range(window):
        w.append(lam)
        lam *= decay
    w = w[::-1]

    beta, hist_sigma = [], []
    for i in range(len(ret) - window + 1):
        tmp = ret.iloc[i : i + window, :].copy()
        w_full = np.diag(w)
        y_full = tmp.dropna(axis=1).drop(columns="000300.SSE")
        idx_full, y_full = y_full.columns, y_full.values
        x_full = np.c_[np.ones((window, 1)), tmp.loc[:, "000300.SSE"].values]
        beta_full = np.linalg.pinv(x_full.T @ w_full @ x_full) @ x_full.T @ w_full @ y_full
        hist_sigma_full = pd.Series(
            np.std(y_full - x_full @ beta_full, axis=0), index=idx_full, name=tmp.index[-1]
        )
        beta_full = pd.Series(beta_full[1], index=idx_full, name=tmp.index[-1])

        beta_lack, hist_sigma_lack = {}, {}
        for c in set(tmp.columns) - set(idx_full) - set("000300.SSE"):
            tmp_ = tmp.loc[:, [c, "000300.SSE"]].copy()
            tmp_.loc[:, "W"] = w
            tmp_ = tmp_.dropna()
            w_lack = np.diag(tmp_["W"])
            if len(tmp_) < half_life:
                continue
            x_lack = np.c_[np.ones(len(tmp_)), tmp_["000300.SSE"].values]
            y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(x_lack.T @ w_lack @ x_lack) @ x_lack.T @ w_lack @ y_lack
            hist_sigma_lack[c] = np.std(y_lack - x_lack @ beta_tmp)
            beta_lack[c] = beta_tmp[1]
        beta_lack = pd.Series(beta_lack, name=tmp.index[-1])
        hist_sigma_lack = pd.Series(hist_sigma_lack, name=tmp.index[-1])
        beta.append(pd.concat([beta_full, beta_lack]).sort_index())
        hist_sigma.append(pd.concat([hist_sigma_full, hist_sigma_lack]).sort_index())
    beta = pd.concat(beta, axis=1).T
    beta = pd.melt(beta.reset_index(), id_vars="index").dropna()
    beta.columns = ["time", "code", "BETA"]
    hist_sigma = pd.concat(hist_sigma, axis=1).T
    hist_sigma = pd.melt(hist_sigma.reset_index(), id_vars="index").dropna()
    hist_sigma.columns = ["time", "code", "Hist_sigma"]
    factor = pd.merge(beta, hist_sigma)

    init_var = ret.var(axis=0)
    decay = 0.5 ** (1 / 42)
    tmp = init_var.copy()
    daily_std = {}
    for t, k in ret.iterrows():
        tmp = tmp * decay + k**2 * (1 - decay)
        daily_std[t] = np.sqrt(tmp)
        tmp = tmp.fillna(init_var)
    daily_std = pd.DataFrame(daily_std).T
    daily_std.index.name = "time"
    daily_std = daily_std.loc[start_date:end_date, :]
    daily_std = pd.melt(daily_std.reset_index(), id_vars="time", value_name="Daily_std").dropna()

    factor = factor.merge(daily_std)

    close = pd.pivot_table(df, values="close", index="time", columns="code").fillna(
        method="ffill", limit=10
    )
    pre_close = pd.pivot_table(df, values="pre_close", index="time", columns="code").fillna(
        method="ffill", limit=10
    )
    idx = close.index
    cmra = {}
    for i in range(len(close) - window + 1):
        close_ = close.iloc[i : i + window, :]
        pre_close_ = pre_close.iloc[i, :]
        pre_close_.name = prev_date(pre_close_.name)
        close_ = close_.append(pre_close_).sort_index().iloc[list(range(0, 253, 21)), :]
        r_tau = close_.pct_change().dropna(how="all")
        z_t = np.log(r_tau + 1).cumsum(axis=0)
        cmra[idx[i + window - 1]] = z_t.max(axis=0) - z_t.min(axis=0)

    cmra = pd.DataFrame(cmra).T
    cmra.index.name = "time"
    cmra = pd.melt(cmra.reset_index(), id_vars="time", value_name="Cumulative_range").dropna()

    factor = factor.merge(cmra)
    return factor.sort_values(by=["time", "code"])


def main():
    args = parse_args()
    all_symbols, dates = load_symbols_dates(args.daily)
    szsh_mask = np.array(["BSE" not in s for s in all_symbols])
    szsh_symbols = all_symbols[szsh_mask]

    target_date = args.target_date or int(tu.todayYYYYMMDD())
    idx = np.searchsorted(dates, target_date)
    if idx >= len(dates) or int(dates[idx]) != target_date:
        raise SystemExit(f"target date {target_date} not in daily.hdf")

    window = 252
    st_idx = idx - 260
    if st_idx < 0:
        raise SystemExit("not enough history for window")

    index_list = np.array(["000300.SSE"], dtype=str)
    index_cache = cbc.IndexCache(index_list, int(dates[st_idx]), target_date, root_path=args.index_root)
    eq_cache = cbc.EqCache(szsh_symbols, int(dates[st_idx]), target_date, root_path=os.getcwd())

    adj_close = eq_cache.daily.close * eq_cache.daily.adj_factor
    lag_adj_close = lag(adj_close, 1, axis=0)

    if "is_listed_in_5days" in h5.File(args.daily, "r"):
        is_listed_in_5days = eq_cache.daily.is_listed_in_5days.astype(bool)
        adj_close[is_listed_in_5days] = np.nan
        lag_adj_close[is_listed_in_5days] = np.nan

    eq_ret = adj_close / lag_adj_close - 1

    index_close = index_cache.daily.close
    lag_index_close = lag(index_close, 1, axis=0)
    index_ret = index_close / lag_index_close - 1

    df = pd.DataFrame({"time": [], "code": [], "ret": [], "close": [], "pre_close": []})
    df["ret"] = np.concatenate([eq_ret.ravel(), index_ret[:, 0].ravel()])
    df["time"] = np.concatenate(
        [np.repeat(eq_cache.daily.dates, eq_ret.shape[1]), eq_cache.daily.dates]
    )
    df["code"] = np.concatenate([np.tile(szsh_symbols, eq_ret.shape[0]), np.tile("000300.SSE", eq_ret.shape[0])])
    df["close"] = np.concatenate([adj_close.ravel(), index_close[:, 0].ravel()])
    df["pre_close"] = np.concatenate([lag_adj_close.ravel(), lag_index_close[:, 0].ravel()])

    data = cal_volatility(start_date=dates[st_idx], end_date=target_date, window=window, half_life=63)

    with h5.File(args.daily, "a") as f:
        shape = (dates.shape[0], all_symbols.shape[0])
        dset_beta = ensure_dataset(f, "BETA", shape)
        dset_hs = ensure_dataset(f, "Hist_sigma", shape)
        dset_std = ensure_dataset(f, "Daily_std", shape)
        dset_cmra = ensure_dataset(f, "Cumulative_range", shape)

        x = data[data.time == target_date]
        x.index = x.code
        y = x.reindex(all_symbols)
        if y.shape[0] > 0:
            dset_beta[idx] = y["BETA"].values.astype(float)
            dset_hs[idx] = y["Hist_sigma"].values.astype(float)
            dset_std[idx] = y["Daily_std"].values.astype(float)
            dset_cmra[idx] = y["Cumulative_range"].values.astype(float)


if __name__ == "__main__":
    main()
