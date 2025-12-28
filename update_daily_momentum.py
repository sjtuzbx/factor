import argparse
import sys
import numpy as np
import pandas as pd
import h5py as h5
import os
from python_utils.quant import *


def _parse_args():
    parser = argparse.ArgumentParser(description="Update daily momentum factors")
    parser.add_argument("--date", type=int, help="single date YYYYMMDD")
    parser.add_argument("--start-date", type=int, help="start date YYYYMMDD")
    parser.add_argument("--end-date", type=int, help="end date YYYYMMDD")
    parser.add_argument("--log-every", type=int, default=5, help="log every N updates")
    return parser.parse_args()

def get_exponent_weight(window, half_life, is_standardize=True):
    L, Lambda = 0.5 ** (1 / half_life), 0.5 ** (1 / half_life)
    W = []
    for _ in range(window):
        W.append(Lambda)
        Lambda *= L
    W = np.array(W[::-1])
    if is_standardize:
        W /= np.sum(W)
    return W


def _compute_for_idx(
    idx,
    today,
    eq_dates,
    ret_df,
    close_df,
    industry_arr,
    circ_mv_arr,
    szsh_symbols,
    w_strev,
    w_rs,
    w_ind,
    r_n,
):
    # Short_Term_reversal
    if idx >= 20:
        tmp = r_n.iloc[idx - 20:idx + 1, :].copy()
        tmp = np.log1p(tmp)
        strev = pd.Series(np.sum(w_strev.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns, name=today)
    else:
        strev = pd.Series(index=szsh_symbols, dtype=float, name=today)

    # Seasonality
    season_list = []
    for i in range(1, 6):
        last_year_date = int((pd.to_datetime(str(today)) - pd.Timedelta(days=365 * i)).strftime("%Y%m%d"))
        pos = int(np.searchsorted(eq_dates, last_year_date))
        if pos + 21 > len(eq_dates):
            continue
        close_win = close_df.iloc[pos:pos + 21].ffill()
        if close_win.shape[0] < 2:
            continue
        season_list.append(close_win.iloc[-1, :] / close_win.iloc[0, :] - 1)
    if season_list:
        seasonality = pd.concat(season_list, axis=1).mean(axis=1)
    else:
        seasonality = pd.Series(index=szsh_symbols, dtype=float, name=today)

    # Relative strength
    window = 252
    rs_dates = eq_dates[max(0, idx - 10):idx + 1]
    rs_list = []
    for d in rs_dates:
        end_pos = int(np.searchsorted(eq_dates, d))
        if end_pos < window - 1:
            continue
        tmp = ret_df.iloc[end_pos - window + 1:end_pos + 1, :].copy()
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / window <= 0.1].fillna(0.0)
        rs_list.append(pd.Series(np.sum(w_rs.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns, name=d))
    if rs_list:
        rs_df = pd.concat(rs_list, axis=1).T
        relative_strength = rs_df.mean(axis=0)
    else:
        relative_strength = pd.Series(index=szsh_symbols, dtype=float)

    # Industry_Momentum
    if len(ret_df) >= 126 and idx >= 125:
        tmp = ret_df.iloc[idx - 125:idx + 1, :].copy()
        tmp = tmp.loc[:, tmp.isnull().sum(axis=0) / 252 <= 0.1].fillna(0.0)
        tmp = np.log1p(tmp)
        rs_today = pd.Series(np.sum(w_ind.reshape(-1, 1) * tmp.values, axis=0), index=tmp.columns, name=today)
        rs_today = rs_today.reindex(szsh_symbols)
        industry = pd.Series(industry_arr[idx], index=szsh_symbols, name="industry")
        circ_mv = pd.Series(circ_mv_arr[idx], index=szsh_symbols, name="circ_mv")
        x = pd.DataFrame(
            {"RS": rs_today, "industry": industry, "c": np.sqrt(circ_mv)},
            index=szsh_symbols,
        )
        x = x.replace([np.inf, -np.inf], np.nan)
        valid = x["industry"].notna() & x["c"].notna() & (x["c"] > 0) & x["RS"].notna()
        x_valid = x.loc[valid]
        if not x_valid.empty:
            ind_rs = x_valid.groupby("industry").apply(
                lambda y: np.nansum(y["RS"] * y["c"]) / np.nansum(y["c"])
            )
            ind_rs.name = "ind_RS"
            x["ind_RS"] = x["industry"].map(ind_rs)
        else:
            x["ind_RS"] = np.nan
        ind_mom = (x["ind_RS"] - x["RS"]).rename("Industry_Momentum")
        ind_mom = ind_mom.fillna(0.0)
        ind_mom.index = x.index
    else:
        ind_mom = pd.Series(index=szsh_symbols, dtype=float, name="Industry_Momentum")

    return strev, seasonality, ind_mom, relative_strength


def main():
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = _parse_args()
    with h5.File("daily.hdf", "r") as f:
        all_symbols = np.array([x.decode("utf-8") for x in f["symbols"][()]])
        dates = f["dates"][()]
        close = f["close"][()]
        adj = f["adj_factor"][()]
        is_listed_in_5days = f["is_listed_in_5days"][()].astype(bool)
        industry = f["industry"][()]
        circ_mv = f["circ_mv"][()]
        hist_alpha = f["Historical_alpha"][()]
    szsh_symbols = np.array(all_symbols, dtype=str)

    if args.date is not None:
        start_date = end_date = int(args.date)
    elif args.start_date is not None and args.end_date is not None:
        start_date = int(args.start_date)
        end_date = int(args.end_date)
    else:
        start_date = end_date = int(tu.todayYYYYMMDD())

    adj_close = close * adj
    adj_close = pd.DataFrame(adj_close).ffill().values
    lag_adj_close = np.vstack([np.full((1, adj_close.shape[1]), np.nan), adj_close[:-1]])

    adj_close[is_listed_in_5days] = np.nan
    lag_adj_close[is_listed_in_5days] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = adj_close / lag_adj_close - 1
    ret[~np.isfinite(ret)] = np.nan

    eq_dates = np.array(dates)
    ret_df = pd.DataFrame(ret, index=eq_dates, columns=szsh_symbols)
    close_df = pd.DataFrame(adj_close, index=eq_dates, columns=szsh_symbols)
    r_n = ret_df.rolling(21).mean()
    w_strev = get_exponent_weight(window=21, half_life=5)
    w_rs = get_exponent_weight(window=252, half_life=126)
    w_ind = get_exponent_weight(window=126, half_life=21)

    start_idx = int(np.searchsorted(eq_dates, start_date))
    end_idx = int(np.searchsorted(eq_dates, end_date))
    if end_idx < len(eq_dates) and eq_dates[end_idx] == end_date:
        end_idx = end_idx
    else:
        end_idx -= 1
    end_idx = max(end_idx, start_idx)

    with h5.File("daily.hdf", "a") as f:
        for col in ["Short_Term_reversal", "Seasonality", "Industry_Momentum", "Relative_strength", "Momentum"]:
            if col not in f:
                f.create_dataset(col, (dates.shape[0], all_symbols.shape[0]), maxshape=(None, None))

        total = end_idx - start_idx + 1
        for i, idx in enumerate(range(start_idx, end_idx + 1), 1):
            today = int(eq_dates[idx])
            strev, seasonality, ind_mom, relative_strength = _compute_for_idx(
                idx,
                today,
                eq_dates,
                ret_df,
                close_df,
                industry,
                circ_mv,
                szsh_symbols,
                w_strev,
                w_rs,
                w_ind,
                r_n,
            )
            strev = strev.reindex(all_symbols)
            seasonality = seasonality.reindex(all_symbols)
            ind_mom = ind_mom.reindex(all_symbols)
            relative_strength = relative_strength.reindex(all_symbols)
            hist_alpha_series = pd.Series(hist_alpha[idx], index=all_symbols, dtype=float)
            hist_alpha_series = hist_alpha_series.fillna(0.0)
            momentum = (
                0.25 * strev.fillna(0.0)
                + 0.25 * seasonality.fillna(0.0)
                + 0.25 * ind_mom.fillna(0.0)
                + 0.125 * hist_alpha_series
                + 0.125 * relative_strength.fillna(0.0)
            )

            f["Short_Term_reversal"][idx] = strev.values.astype(float)
            f["Seasonality"][idx] = seasonality.values.astype(float)
            f["Industry_Momentum"][idx] = ind_mom.values.astype(float)
            f["Relative_strength"][idx] = relative_strength.values.astype(float)
            f["Momentum"][idx] = momentum.values.astype(float)

            if args.log_every > 0 and (i % args.log_every == 0 or i == total):
                print(f"progress {i}/{total} date={today}")


if __name__ == "__main__":
    main()
