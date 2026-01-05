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


def apply_lyr_by_end_date(values, end_dates, dates):
    n_dates, n_syms = values.shape
    prev_year_end = (dates // 10000 - 1) * 10000 + 1231
    out = np.full_like(values, np.nan, dtype=np.float64)
    for s in range(n_syms):
        ed = end_dates[:, s]
        match = (ed == prev_year_end) & np.isfinite(ed)
        idx = np.where(match, np.arange(n_dates), -1)
        last = np.maximum.accumulate(idx)
        valid = last >= 0
        out[valid, s] = values[last[valid], s]
    return out


def compute_ld_from_longdeb_ratio(total_liab, longdeb_ratio):
    out = np.full_like(total_liab, np.nan, dtype=np.float64)
    mask = np.isfinite(total_liab) & np.isfinite(longdeb_ratio)
    out[mask] = total_liab[mask] * longdeb_ratio[mask]
    return out


def compute_annual_variability(values, ann_dates, end_dates, years=5, annual_mmdd=1231):
    n_dates, n_syms = values.shape
    out = np.full_like(values, np.nan, dtype=np.float64)
    for s in range(n_syms):
        window = []
        prev_ann = np.nan
        for i in range(n_dates):
            ann = ann_dates[i, s]
            if np.isfinite(ann) and (not np.isfinite(prev_ann) or ann != prev_ann):
                prev_ann = ann
                end_date = end_dates[i, s]
                val = values[i, s]
                if np.isfinite(end_date) and np.isfinite(val) and int(end_date) % 10000 == annual_mmdd:
                    window.append(val)
                    if len(window) > years:
                        window.pop(0)
            if len(window) >= years:
                mean = float(np.mean(window))
                if np.isfinite(mean) and mean != 0:
                    out[i, s] = float(np.std(window, ddof=0)) / mean
    return out


def compute_ttm_from_cumulative(values, end_dates, ann_dates):
    n_dates, n_syms = values.shape
    out = np.full_like(values, np.nan, dtype=np.float64)
    for s in range(n_syms):
        values_by_end = {}
        prev_ann = np.nan
        prev_end = np.nan
        for i in range(n_dates):
            ann = ann_dates[i, s] if ann_dates is not None else np.nan
            end_date = end_dates[i, s]
            val = values[i, s]
            if np.isfinite(end_date) and np.isfinite(val):
                end_i = int(end_date)
                ann_changed = np.isfinite(ann) and (not np.isfinite(prev_ann) or ann != prev_ann)
                end_changed = not np.isfinite(ann) and (not np.isfinite(prev_end) or end_i != int(prev_end))
                if ann_changed or end_changed:
                    values_by_end[end_i] = float(val)
                    prev_ann = ann
                    prev_end = end_date
                mmdd = end_i % 10000
                if mmdd == 1231:
                    out[i, s] = float(val)
                else:
                    prev_year_end = (end_i // 10000 - 1) * 10000 + 1231
                    prev_year_same = end_i - 10000
                    if prev_year_end in values_by_end and prev_year_same in values_by_end:
                        out[i, s] = float(val) + values_by_end[prev_year_end] - values_by_end[prev_year_same]
            else:
                if np.isfinite(ann):
                    prev_ann = ann
    return out


def compute_growth_slope(values, end_dates, ann_dates, years=5, annual_mmdd=1231):
    n_dates, n_syms = values.shape
    out = np.full_like(values, np.nan, dtype=np.float64)
    x = np.arange(years, dtype=np.float64)
    mean_x = float(np.mean(x))
    denom = float(np.sum((x - mean_x) ** 2))
    for s in range(n_syms):
        window = []
        prev_ann = np.nan
        prev_end = np.nan
        for i in range(n_dates):
            ann = ann_dates[i, s] if ann_dates is not None else np.nan
            end_date = end_dates[i, s]
            val = values[i, s]
            if np.isfinite(end_date) and np.isfinite(val):
                end_i = int(end_date)
                ann_changed = np.isfinite(ann) and (not np.isfinite(prev_ann) or ann != prev_ann)
                end_changed = not np.isfinite(ann) and (not np.isfinite(prev_end) or end_i != int(prev_end))
                if ann_changed or end_changed:
                    prev_ann = ann
                    prev_end = end_date
                    if end_i % 10000 == annual_mmdd:
                        window.append(float(val))
                        if len(window) > years:
                            window.pop(0)
            else:
                if np.isfinite(ann):
                    prev_ann = ann
            if len(window) >= years:
                y = np.asarray(window, dtype=np.float64)
                mean_y = float(np.mean(y))
                if np.isfinite(mean_y) and mean_y != 0:
                    slope = float(np.sum((x - mean_x) * (y - mean_y)) / denom)
                    out[i, s] = -slope / mean_y
    return out


def mean_ignore_nan(*arrays):
    stacked = np.stack(arrays, axis=0)
    count = np.sum(np.isfinite(stacked), axis=0)
    total = np.nansum(stacked, axis=0)
    out = np.full_like(total, np.nan, dtype=np.float64)
    mask = count > 0
    out[mask] = total[mask] / count[mask]
    return out


def combine_primary_fallback(primary, fallback):
    if primary is None and fallback is None:
        return None
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    out = primary.copy()
    mask = ~np.isfinite(out)
    out[mask] = fallback[mask]
    return out


def compute_noa(total_assets, total_liab, cash_equ, total_debt):
    out = np.full_like(total_assets, np.nan, dtype=np.float64)
    mask = np.isfinite(total_assets) & np.isfinite(total_liab) & np.isfinite(cash_equ) & np.isfinite(total_debt)
    out[mask] = (total_assets[mask] - cash_equ[mask]) - (total_liab[mask] - total_debt[mask])
    return out


def compute_annual_regression_growth(values, ann_dates, end_dates, years=5, annual_mmdd=1231):
    n_dates, n_syms = values.shape
    out = np.full_like(values, np.nan, dtype=np.float64)
    t = np.arange(years, dtype=np.float64)
    t_mean = float(np.mean(t))
    t_denom = float(np.sum((t - t_mean) ** 2))
    if t_denom == 0:
        return out
    for s in range(n_syms):
        window = []
        prev_ann = np.nan
        for i in range(n_dates):
            ann = ann_dates[i, s]
            if np.isfinite(ann) and (not np.isfinite(prev_ann) or ann != prev_ann):
                prev_ann = ann
                end_date = end_dates[i, s]
                val = values[i, s]
                if np.isfinite(end_date) and np.isfinite(val) and int(end_date) % 10000 == annual_mmdd:
                    window.append(val)
                    if len(window) > years:
                        window.pop(0)
            if len(window) >= years:
                y = np.array(window[-years:], dtype=np.float64)
                y_mean = float(np.mean(y))
                if np.isfinite(y_mean) and y_mean != 0:
                    slope = float(np.sum((t - t_mean) * (y - y_mean)) / t_denom)
                    out[i, s] = slope / y_mean
    return out


def load_dataset(f, name):
    if not name:
        return None
    if name not in f:
        return None
    return f[name][()].astype(np.float64)


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
    parser.add_argument("--balance-lyr", action="store_true", help="Use previous fiscal year-end for PE/LD/BE/TL/TA")
    parser.add_argument("--balance-scale", type=float, default=10000.0, help="Scale factor for balance sheet fields")
    parser.add_argument("--me-scale", type=float, default=1.0, help="Scale factor for market equity")
    parser.add_argument(
        "--ld-from-longdeb",
        action="store_true",
        default=True,
        help="Estimate LD as total_liab * fina_longdeb_to_debt",
    )
    parser.add_argument(
        "--no-ld-from-longdeb",
        action="store_false",
        dest="ld_from_longdeb",
        help="Disable LD estimation from longdeb ratio",
    )
    parser.add_argument("--me-shift", type=int, default=1, help="Shift ME by N days (default 1)")
    parser.add_argument("--out-mlev", default="MLEV", help="Output dataset name for market leverage")
    parser.add_argument("--out-blev", default="BLEV", help="Output dataset name for book leverage")
    parser.add_argument("--out-dtoa", default="DTOA", help="Output dataset name for debt-to-assets")
    parser.add_argument("--ev-sales-dataset", default="income_total_revenue", help="Sales dataset")
    parser.add_argument("--ev-sales-fallback", default="income_revenue", help="Fallback sales dataset")
    parser.add_argument("--ev-earnings-dataset", default="income_n_income", help="Earnings dataset")
    parser.add_argument("--ev-ann-date-dataset", default="income_ann_date", help="Income announcement date dataset")
    parser.add_argument("--ev-end-date-dataset", default="income_end_date", help="Income report end date dataset")
    parser.add_argument("--ev-cfo-dataset", default="cashflow_n_incr_cash_cash_equ", help="Cashflow dataset")
    parser.add_argument("--ev-cfo-ann-date-dataset", default="cashflow_ann_date", help="Cashflow announcement date dataset")
    parser.add_argument("--ev-cfo-end-date-dataset", default="cashflow_end_date", help="Cashflow report end date dataset")
    parser.add_argument("--ev-years", type=int, default=5, help="Number of fiscal years for variability")
    parser.add_argument("--ev-annual-mmdd", type=int, default=1231, help="Fiscal year-end MMDD (default 1231)")
    parser.add_argument("--out-var-sales", default="VAR_SALES", help="Output dataset name for sales variability")
    parser.add_argument("--out-var-earnings", default="VAR_EARN", help="Output dataset name for earnings variability")
    parser.add_argument("--out-var-cfo", default="VAR_CFO", help="Output dataset name for cashflow variability")
    parser.add_argument("--out-abs", default="ABS", help="Output dataset name for accruals balance-sheet factor")
    parser.add_argument("--out-acf", default="ACF", help="Output dataset name for accruals cashflow factor")
    parser.add_argument("--gp-cogs-dataset", default="income_oper_cost", help="COGS dataset for profitability")
    parser.add_argument("--issue-dataset", default="float_share", help="Float share dataset for issuance growth")
    parser.add_argument(
        "--capex-dataset",
        default="c_pay_acq_const_fiolta",
        help="Capex dataset for capital expenditure growth",
    )
    parser.add_argument("--out-ato", default="ATO", help="Output dataset name for asset turnover")
    parser.add_argument("--out-gp", default="GP", help="Output dataset name for gross profitability")
    parser.add_argument("--out-gpm", default="GPM", help="Output dataset name for gross profit margin")
    parser.add_argument("--out-roa", default="ROA", help="Output dataset name for return on assets")
    parser.add_argument("--out-agrow", default="AGROW", help="Output dataset name for total assets growth rate")
    parser.add_argument("--out-issue-grow", default="ISSUE_GROW", help="Output dataset name for issuance growth")
    parser.add_argument("--out-capex-grow", default="CAPEX_GROW", help="Output dataset name for capex growth")
    parser.add_argument("--out-leverage", default="LEVERAGE", help="Output dataset name for leverage composite")
    parser.add_argument("--out-earn-var", default="EARN_VAR", help="Output dataset name for earnings variability composite")
    parser.add_argument("--out-earn-qual", default="EARN_QUAL", help="Output dataset name for earnings quality composite")
    parser.add_argument("--out-profit", default="PROFIT", help="Output dataset name for profitability composite")
    parser.add_argument("--out-invest-qual", default="INVEST_QUAL", help="Output dataset name for investment quality composite")
    parser.add_argument("--out-quality", default="QUALITY", help="Output dataset name for overall quality composite")
    parser.add_argument(
        "--af-std-eps-dataset",
        default="",
        help="Analyst forecast EPS std dataset (optional)",
    )
    parser.add_argument("--af-price-dataset", default="close", help="Price dataset for EPS-to-price")
    parser.add_argument(
        "--out-af-std-epsp",
        default="STD_AFEP",
        help="Output dataset name for analyst EPS std-to-price",
    )
    parser.add_argument(
        "--bp-be-dataset",
        default="balance_total_hldr_eqy_exc_min_int",
        help="Book equity dataset for book-to-price",
    )
    parser.add_argument(
        "--bp-me-dataset",
        default="total_mv",
        help="Market equity dataset for book-to-price",
    )
    parser.add_argument(
        "--ep-profit-dataset",
        default="fina_profit_dedt",
        help="Profit dataset for earnings-to-price (last fiscal year)",
    )
    parser.add_argument(
        "--ep-analyst-dataset",
        default="ANALYST_NP_FWD12M_MEAN",
        help="Analyst forward 12M net profit dataset",
    )
    parser.add_argument(
        "--cep-cashflow-dataset",
        default="cashflow_n_cashflow_act",
        help="Cashflow dataset for cash earnings-to-price (last fiscal year)",
    )
    parser.add_argument(
        "--em-ebit-dataset",
        default="income_ebit",
        help="EBIT dataset for enterprise multiple (last fiscal year)",
    )
    parser.add_argument(
        "--em-tl-dataset",
        default="balance_total_liab",
        help="Total liabilities dataset for EV",
    )
    parser.add_argument(
        "--em-cash-dataset",
        default="balance_cash_reser_cb",
        help="Cash dataset for EV",
    )
    parser.add_argument("--out-bp", default="BP", help="Output dataset name for book-to-price")
    parser.add_argument("--out-ep", default="EP", help="Output dataset name for earnings-to-price")
    parser.add_argument("--out-epf", default="EPF", help="Output dataset name for analyst forward EP")
    parser.add_argument("--out-cep", default="CEP", help="Output dataset name for cash earnings-to-price")
    parser.add_argument("--out-em", default="EM", help="Output dataset name for enterprise multiple")
    parser.add_argument("--value-lrs-dataset", default="Long_Relative_strength", help="Long relative strength dataset")
    parser.add_argument(
        "--value-lha-dataset",
        default="Long_Historical_alpha",
        help="Long historical alpha dataset",
    )
    parser.add_argument("--out-value", default="VALUE", help="Output dataset name for value composite")
    parser.add_argument("--eps-ps-dataset", default="income_basic_eps", help="EPS dataset for growth")
    parser.add_argument("--eps-ann-date-dataset", default="income_ann_date", help="EPS announcement date dataset")
    parser.add_argument("--eps-end-date-dataset", default="income_end_date", help="EPS report end date dataset")
    parser.add_argument(
        "--sales-ps-dataset",
        default="fina_total_revenue_ps",
        help="Revenue per share dataset for growth",
    )
    parser.add_argument("--sales-ann-date-dataset", default="fina_ann_date", help="Sales announcement date dataset")
    parser.add_argument("--sales-end-date-dataset", default="fina_end_date", help="Sales report end date dataset")
    parser.add_argument("--out-eps-growth", default="EPS_GROWTH_5Y", help="Output dataset name for EPS growth")
    parser.add_argument(
        "--out-sales-growth",
        default="SALES_GROWTH_5Y",
        help="Output dataset name for sales per share growth",
    )
    parser.add_argument(
        "--growth-analyst-dataset",
        default="ANALYST_NP_CAGR_2Y",
        help="Analyst net profit CAGR dataset for growth composite",
    )
    parser.add_argument("--out-growth", default="GROWTH", help="Output dataset name for growth composite")
    parser.add_argument(
        "--dividend-dataset",
        default="income_div_payt",
        help="Dividend dataset for dividend-to-price",
    )
    parser.add_argument(
        "--dividend-end-date-dataset",
        default="income_end_date",
        help="Dividend report end date dataset",
    )
    parser.add_argument(
        "--dividend-ann-date-dataset",
        default="income_ann_date",
        help="Dividend announcement date dataset",
    )
    parser.add_argument(
        "--dividend-share-dataset",
        default="total_share",
        help="Total share dataset for dividend per share",
    )
    parser.add_argument(
        "--dividend-share-scale",
        type=float,
        default=10000.0,
        help="Scale factor for total_share to shares",
    )
    parser.add_argument(
        "--dividend-price-dataset",
        default="close",
        help="Price dataset for dividend-to-price",
    )
    parser.add_argument(
        "--out-divp",
        default="DIVIDEND_TO_PRICE",
        help="Output dataset name for dividend-to-price ratio",
    )
    parser.add_argument(
        "--sentiment-np-rribs-dataset",
        default="ANALYST_NP_RRIBS",
        help="Analyst RRIBS net profit sentiment dataset",
    )
    parser.add_argument(
        "--sentiment-ep-change-dataset",
        default="ANALYST_EP_CHANGE",
        help="Analyst EP change dataset",
    )
    parser.add_argument(
        "--sentiment-eps-change-dataset",
        default="ANALYST_EPS_CHANGE",
        help="Analyst EPS change dataset",
    )
    parser.add_argument("--out-sentiment", default="SENTIMENT", help="Output dataset name for sentiment factor")
    parser.add_argument(
        "--dividend-analyst-rd-dataset",
        default="ANALYST_RD_6M_MEAN",
        help="Analyst 6M dividend yield dataset",
    )
    parser.add_argument(
        "--out-dividend-yield",
        default="DIVIDEND_YIELD_COMP",
        help="Output dataset name for dividend yield composite",
    )
    args = parser.parse_args()

    with h5.File(args.h5, "r") as f:
        dates = f["dates"][()].astype(int)
        me = f[args.me_dataset][()].astype(np.float64)
        be = f[args.be_dataset][()].astype(np.float64)
        tl = f[args.tl_dataset][()].astype(np.float64)
        ta = f[args.ta_dataset][()].astype(np.float64)
        tcl = f[args.tcl_dataset][()].astype(np.float64)
        end_dates = load_dataset(f, "balance_end_date")
        ev_ann_dates = load_dataset(f, args.ev_ann_date_dataset)
        ev_end_dates = load_dataset(f, args.ev_end_date_dataset)
        ev_sales = load_dataset(f, args.ev_sales_dataset)
        ev_sales_fallback = load_dataset(f, args.ev_sales_fallback)
        ev_sales = combine_primary_fallback(ev_sales, ev_sales_fallback)
        ev_earnings = load_dataset(f, args.ev_earnings_dataset)
        ev_cfo = load_dataset(f, args.ev_cfo_dataset)
        ev_cfo_ann_dates = load_dataset(f, args.ev_cfo_ann_date_dataset)
        ev_cfo_end_dates = load_dataset(f, args.ev_cfo_end_date_dataset)
        af_eps_std = load_dataset(f, args.af_std_eps_dataset)
        af_price = load_dataset(f, args.af_price_dataset)
        gp_cogs = load_dataset(f, args.gp_cogs_dataset)
        issue_shares = load_dataset(f, args.issue_dataset)
        capex = load_dataset(f, args.capex_dataset)
        bal_cash = load_dataset(f, "balance_cash_reser_cb")
        bal_st_borr = load_dataset(f, "balance_st_borr")
        bal_lt_borr = load_dataset(f, "balance_lt_borr")
        bal_cb_borr = load_dataset(f, "balance_cb_borr")
        bal_pledge_borr = load_dataset(f, "balance_pledge_borr")
        cf_depr = load_dataset(f, "cashflow_depr_fa_coga_dpba")
        cf_amort = load_dataset(f, "cashflow_amort_intang_assets")
        cf_cfo = load_dataset(f, "cashflow_n_cashflow_act")
        cf_cfi = load_dataset(f, "cashflow_n_cashflow_inv_act")
        inc_net = load_dataset(f, "income_n_income")
        bp_be = load_dataset(f, args.bp_be_dataset)
        bp_me = load_dataset(f, args.bp_me_dataset)
        ep_profit = load_dataset(f, args.ep_profit_dataset)
        ep_analyst = load_dataset(f, args.ep_analyst_dataset)
        cep_cashflow = load_dataset(f, args.cep_cashflow_dataset)
        em_ebit = load_dataset(f, args.em_ebit_dataset)
        em_tl = load_dataset(f, args.em_tl_dataset)
        em_cash = load_dataset(f, args.em_cash_dataset)
        income_end_dates = load_dataset(f, "income_end_date")
        value_lrs = load_dataset(f, args.value_lrs_dataset)
        value_lha = load_dataset(f, args.value_lha_dataset)
        eps_ps = load_dataset(f, args.eps_ps_dataset)
        eps_ann_dates = load_dataset(f, args.eps_ann_date_dataset)
        eps_end_dates = load_dataset(f, args.eps_end_date_dataset)
        sales_ps = load_dataset(f, args.sales_ps_dataset)
        sales_ann_dates = load_dataset(f, args.sales_ann_date_dataset)
        sales_end_dates = load_dataset(f, args.sales_end_date_dataset)
        growth_analyst = load_dataset(f, args.growth_analyst_dataset)
        div_payt = load_dataset(f, args.dividend_dataset)
        div_end_dates = load_dataset(f, args.dividend_end_date_dataset)
        div_ann_dates = load_dataset(f, args.dividend_ann_date_dataset)
        div_shares = load_dataset(f, args.dividend_share_dataset)
        div_price = load_dataset(f, args.dividend_price_dataset)
        sent_np_rribs = load_dataset(f, args.sentiment_np_rribs_dataset)
        sent_ep_change = load_dataset(f, args.sentiment_ep_change_dataset)
        sent_eps_change = load_dataset(f, args.sentiment_eps_change_dataset)
        div_analyst_rd = load_dataset(f, args.dividend_analyst_rd_dataset)

        if args.ld_from_longdeb:
            if "fina_longdeb_to_debt" not in f:
                raise SystemExit("fina_longdeb_to_debt not found for --ld-from-longdeb")
            longdeb_ratio = f["fina_longdeb_to_debt"][()].astype(np.float64)
            ld = compute_ld_from_longdeb_ratio(tl, longdeb_ratio)
        elif args.ld_dataset and args.ld_dataset in f:
            ld = f[args.ld_dataset][()].astype(np.float64)
        else:
            ld = tl - tcl

        if args.pe_dataset and args.pe_dataset in f:
            pe = f[args.pe_dataset][()].astype(np.float64)
        else:
            pe = np.zeros_like(ld, dtype=np.float64)

    if args.balance_scale != 1.0:
        be = be / args.balance_scale
        tl = tl / args.balance_scale
        ta = ta / args.balance_scale
        tcl = tcl / args.balance_scale
        pe = pe / args.balance_scale
        ld = ld / args.balance_scale
    if args.me_scale != 1.0:
        me = me / args.me_scale

    if args.balance_lyr:
        if end_dates is None:
            raise SystemExit("balance_end_date is required for --balance-lyr")
        be = apply_lyr_by_end_date(be, end_dates, dates)
        tl = apply_lyr_by_end_date(tl, end_dates, dates)
        ta = apply_lyr_by_end_date(ta, end_dates, dates)
        tcl = apply_lyr_by_end_date(tcl, end_dates, dates)
        pe = apply_lyr_by_end_date(pe, end_dates, dates)
        if args.ld_from_longdeb:
            ld = apply_lyr_by_end_date(ld, end_dates, dates)
        else:
            ld = tl - tcl
    elif args.pe_lyr:
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

    if args.me_shift:
        me_shifted = np.full_like(me, np.nan, dtype=np.float64)
        if me.shape[0] > args.me_shift:
            me_shifted[args.me_shift :, :] = me[:-args.me_shift, :]
        me = me_shifted

    mlev_num = me + pe + ld
    blev_num = be + pe + ld
    mlev = safe_divide(mlev_num, me)
    blev = safe_divide(blev_num, me)
    dtoa = safe_divide(tl, ta)

    var_sales = None
    var_earnings = None
    var_cfo = None
    abs_factor = None
    acf_factor = None
    af_eps_to_price = None
    bp = None
    ep = None
    epf = None
    cep = None
    em = None
    value = None
    eps_growth = None
    sales_growth = None
    growth_comp = None
    divp = None
    sentiment = None
    dividend_yield = None
    ato = None
    gp = None
    gpm = None
    roa = None
    agrow = None
    issue_grow = None
    capex_grow = None
    leverage_comp = None
    earn_var_comp = None
    earn_qual_comp = None
    profit_comp = None
    invest_qual_comp = None
    quality_comp = None
    if ev_ann_dates is None or ev_end_dates is None:
        print("earnings variability skipped: missing ann/end date datasets")
    else:
        if ev_sales is not None:
            print("computing VAR_SALES...")
            var_sales = compute_annual_variability(
                ev_sales, ev_ann_dates, ev_end_dates, years=args.ev_years, annual_mmdd=args.ev_annual_mmdd
            )
        else:
            print("earnings variability skipped for sales: dataset missing")
        if ev_earnings is not None:
            print("computing VAR_EARN...")
            var_earnings = compute_annual_variability(
                ev_earnings, ev_ann_dates, ev_end_dates, years=args.ev_years, annual_mmdd=args.ev_annual_mmdd
            )
        else:
            print("earnings variability skipped for earnings: dataset missing")
    if ev_cfo is not None and ev_cfo_ann_dates is not None and ev_cfo_end_dates is not None:
        print("computing VAR_CFO...")
        var_cfo = compute_annual_variability(
            ev_cfo, ev_cfo_ann_dates, ev_cfo_end_dates, years=args.ev_years, annual_mmdd=args.ev_annual_mmdd
        )
    elif ev_cfo is None:
        print("earnings variability skipped for cashflows: dataset missing")
    else:
        print("earnings variability skipped for cashflows: missing ann/end date datasets")

    if bal_cash is None:
        print("accruals skipped: balance_cash_reser_cb missing")
    elif cf_depr is None or cf_amort is None or cf_cfo is None or cf_cfi is None or inc_net is None:
        print("accruals skipped: cashflow/income datasets missing")
    elif end_dates is None:
        print("accruals skipped: balance_end_date missing")
    else:
        da = cf_depr + cf_amort
        total_debt = np.zeros_like(tl, dtype=np.float64)
        for part in (bal_st_borr, bal_lt_borr, bal_cb_borr, bal_pledge_borr):
            if part is not None:
                total_debt += part
        noa = compute_noa(ta, tl, bal_cash, total_debt)
        noa_prev = apply_lyr_by_end_date(noa, end_dates, dates)
        accr_bs = noa - noa_prev - da
        abs_factor = safe_divide(-accr_bs, ta)
        accr_cf = inc_net - (cf_cfo + cf_cfi) + da
        acf_factor = safe_divide(-accr_cf, ta)

    if af_eps_std is not None and af_price is not None:
        af_eps_to_price = safe_divide(af_eps_std, af_price)
    elif args.af_std_eps_dataset:
        print("analyst EPS std-to-price skipped: dataset missing")

    if bp_be is not None:
        bp_be_use = bp_be
        if args.balance_scale != 1.0:
            bp_be_use = bp_be_use / args.balance_scale
        if end_dates is not None:
            bp_be_use = apply_lyr_by_end_date(bp_be_use, end_dates, dates)
        bp_me_use = bp_me if bp_me is not None else me
        if bp_me_use is not me and args.me_scale != 1.0:
            bp_me_use = bp_me_use / args.me_scale
        bp = safe_divide(bp_be_use, bp_me_use)
    else:
        print("BP skipped: book equity dataset missing")

    income_dates_for_lyr = income_end_dates if income_end_dates is not None else ev_end_dates
    if income_dates_for_lyr is None:
        print("EP/CEP/EM skipped: income end_date missing")
    else:
        if ep_profit is not None:
            ep_profit_use = apply_lyr_by_end_date(ep_profit, income_dates_for_lyr, dates)
            ep = safe_divide(ep_profit_use, me)
        else:
            print("EP skipped: profit dataset missing")
        if cep_cashflow is not None:
            cep_cashflow_use = apply_lyr_by_end_date(cep_cashflow, income_dates_for_lyr, dates)
            cep = safe_divide(cep_cashflow_use, me)
        else:
            print("CEP skipped: cashflow dataset missing")
        if em_ebit is not None and em_tl is not None and em_cash is not None:
            em_ebit_use = apply_lyr_by_end_date(em_ebit, income_dates_for_lyr, dates)
            em_tl_use = em_tl
            em_cash_use = em_cash
            if args.balance_scale != 1.0:
                em_tl_use = em_tl_use / args.balance_scale
                em_cash_use = em_cash_use / args.balance_scale
            if end_dates is not None:
                em_tl_use = apply_lyr_by_end_date(em_tl_use, end_dates, dates)
                em_cash_use = apply_lyr_by_end_date(em_cash_use, end_dates, dates)
            ev = me + em_tl_use - em_cash_use
            em = safe_divide(em_ebit_use, ev)
        else:
            print("EM skipped: EBIT/total_liab/cash dataset missing")

    if ep_analyst is not None:
        epf = safe_divide(ep_analyst, me)
    else:
        print("EPF skipped: analyst dataset missing")

    if (
        bp is not None
        and ep is not None
        and epf is not None
        and cep is not None
        and em is not None
        and value_lrs is not None
        and value_lha is not None
    ):
        value_earn = mean_ignore_nan(ep, epf, cep, em)
        value_long = mean_ignore_nan(value_lrs, value_lha)
        value = mean_ignore_nan(bp, value_earn, value_long)
        missing_dates = np.all(~np.isfinite(value), axis=1)
        if missing_dates.any():
            bad_dates = dates[missing_dates]
            print("WARNING: VALUE all-NaN dates:", bad_dates[:10], "count", bad_dates.size)
    else:
        print("VALUE skipped: missing BP/EP/EPF/CEP/EM or long-term momentum datasets")

    if eps_ps is not None and eps_ann_dates is not None and eps_end_dates is not None:
        eps_growth = compute_annual_regression_growth(
            eps_ps, eps_ann_dates, eps_end_dates, years=args.ev_years, annual_mmdd=args.ev_annual_mmdd
        )
    else:
        print("EPS growth skipped: missing EPS datasets or dates")

    if sales_ps is not None and sales_ann_dates is not None and sales_end_dates is not None:
        sales_growth = compute_annual_regression_growth(
            sales_ps, sales_ann_dates, sales_end_dates, years=args.ev_years, annual_mmdd=args.ev_annual_mmdd
        )
    else:
        print("Sales growth skipped: missing sales datasets or dates")

    if growth_analyst is not None or eps_growth is not None or sales_growth is not None:
        growth_comp = mean_ignore_nan(
            growth_analyst if growth_analyst is not None else np.full_like(mlev, np.nan, dtype=np.float64),
            eps_growth if eps_growth is not None else np.full_like(mlev, np.nan, dtype=np.float64),
            sales_growth if sales_growth is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        )
    else:
        print("Growth composite skipped: missing analyst/eps/sales growth datasets")

    if div_payt is None or div_end_dates is None or div_shares is None or div_price is None:
        print("Dividend-to-price skipped: missing dividend/share/price datasets")
    else:
        div_end_dates_use = div_end_dates
        if div_end_dates_use is None and div_ann_dates is not None:
            div_end_dates_use = div_ann_dates
        if div_end_dates_use is None:
            print("Dividend-to-price skipped: dividend end/ann date missing")
        else:
            div_payt_use = apply_lyr_by_end_date(div_payt, div_end_dates_use, dates)
            shares = div_shares * args.dividend_share_scale
            div_ps = safe_divide(div_payt_use, shares)
            divp = safe_divide(div_ps, div_price)

    if sent_np_rribs is not None or sent_ep_change is not None or sent_eps_change is not None:
        sentiment = mean_ignore_nan(
            sent_np_rribs if sent_np_rribs is not None else np.full_like(mlev, np.nan, dtype=np.float64),
            sent_ep_change if sent_ep_change is not None else np.full_like(mlev, np.nan, dtype=np.float64),
            sent_eps_change if sent_eps_change is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        )
    else:
        print("Sentiment skipped: missing analyst sentiment datasets")

    if divp is not None or div_analyst_rd is not None:
        dividend_yield = mean_ignore_nan(
            divp if divp is not None else np.full_like(mlev, np.nan, dtype=np.float64),
            div_analyst_rd if div_analyst_rd is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        )
    else:
        print("Dividend yield skipped: missing dividend datasets")

    if ev_sales is None or ev_earnings is None or gp_cogs is None or ev_end_dates is None:
        print("profitability factors skipped: income datasets or end_date missing")
    else:
        ttm_sales = compute_ttm_from_cumulative(ev_sales, ev_end_dates, ev_ann_dates)
        ttm_earnings = compute_ttm_from_cumulative(ev_earnings, ev_end_dates, ev_ann_dates)
        lfy_sales = apply_lyr_by_end_date(ev_sales, ev_end_dates, dates)
        lfy_cogs = apply_lyr_by_end_date(gp_cogs, ev_end_dates, dates)
        lfy_ta = apply_lyr_by_end_date(ta, end_dates, dates) if end_dates is not None else None
        if lfy_ta is None:
            print("profitability factors skipped: balance_end_date missing")
        else:
            ato = safe_divide(ttm_sales, ta)
            gp = safe_divide(lfy_sales - lfy_cogs, lfy_ta)
            gpm = safe_divide(lfy_sales - lfy_cogs, lfy_sales)
            roa = safe_divide(ttm_earnings, ta)

    if end_dates is None:
        print("growth factors skipped: balance_end_date missing")
    else:
        agrow = compute_growth_slope(ta, end_dates, None, years=5)
        if issue_shares is None:
            print("issuance growth skipped: float_share missing")
        else:
            issue_grow = compute_growth_slope(issue_shares, end_dates, None, years=5)
    if capex is None or ev_cfo_end_dates is None:
        print("capex growth skipped: capex or cashflow_end_date missing")
    else:
        capex_grow = compute_growth_slope(capex, ev_cfo_end_dates, ev_cfo_ann_dates, years=5)

    leverage_comp = mean_ignore_nan(mlev, blev, dtoa)
    earn_var_comp = mean_ignore_nan(
        var_sales if var_sales is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        var_earnings if var_earnings is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        var_cfo if var_cfo is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        af_eps_to_price if af_eps_to_price is not None else np.full_like(mlev, np.nan, dtype=np.float64),
    )
    earn_qual_comp = mean_ignore_nan(
        abs_factor if abs_factor is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        acf_factor if acf_factor is not None else np.full_like(mlev, np.nan, dtype=np.float64),
    )
    profit_comp = mean_ignore_nan(
        ato if ato is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        gp if gp is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        gpm if gpm is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        roa if roa is not None else np.full_like(mlev, np.nan, dtype=np.float64),
    )
    invest_qual_comp = mean_ignore_nan(
        agrow if agrow is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        issue_grow if issue_grow is not None else np.full_like(mlev, np.nan, dtype=np.float64),
        capex_grow if capex_grow is not None else np.full_like(mlev, np.nan, dtype=np.float64),
    )
    quality_comp = mean_ignore_nan(
        0.2 * leverage_comp,
        0.2 * earn_var_comp,
        0.2 * earn_qual_comp,
        0.2 * profit_comp,
        0.2 * invest_qual_comp,
    )

    with h5.File(args.h5, "a") as f:
        dset_mlev = ensure_dataset(f, args.out_mlev, mlev.shape)
        dset_blev = ensure_dataset(f, args.out_blev, blev.shape)
        dset_dtoa = ensure_dataset(f, args.out_dtoa, dtoa.shape)
        dset_mlev[:, :] = mlev.astype(np.float32)
        dset_blev[:, :] = blev.astype(np.float32)
        dset_dtoa[:, :] = dtoa.astype(np.float32)
        if var_sales is not None:
            dset_var_sales = ensure_dataset(f, args.out_var_sales, var_sales.shape)
            dset_var_sales[:, :] = var_sales.astype(np.float32)
        if var_earnings is not None:
            dset_var_earn = ensure_dataset(f, args.out_var_earnings, var_earnings.shape)
            dset_var_earn[:, :] = var_earnings.astype(np.float32)
        if var_cfo is not None:
            dset_var_cfo = ensure_dataset(f, args.out_var_cfo, var_cfo.shape)
            dset_var_cfo[:, :] = var_cfo.astype(np.float32)
        if abs_factor is not None:
            dset_abs = ensure_dataset(f, args.out_abs, abs_factor.shape)
            dset_abs[:, :] = abs_factor.astype(np.float32)
        if acf_factor is not None:
            dset_acf = ensure_dataset(f, args.out_acf, acf_factor.shape)
            dset_acf[:, :] = acf_factor.astype(np.float32)
        if af_eps_to_price is not None:
            dset_af_epsp = ensure_dataset(f, args.out_af_std_epsp, af_eps_to_price.shape)
            dset_af_epsp[:, :] = af_eps_to_price.astype(np.float32)
        if bp is not None:
            dset_bp = ensure_dataset(f, args.out_bp, bp.shape)
            dset_bp[:, :] = bp.astype(np.float32)
        if ep is not None:
            dset_ep = ensure_dataset(f, args.out_ep, ep.shape)
            dset_ep[:, :] = ep.astype(np.float32)
        if epf is not None:
            dset_epf = ensure_dataset(f, args.out_epf, epf.shape)
            dset_epf[:, :] = epf.astype(np.float32)
        if cep is not None:
            dset_cep = ensure_dataset(f, args.out_cep, cep.shape)
            dset_cep[:, :] = cep.astype(np.float32)
        if em is not None:
            dset_em = ensure_dataset(f, args.out_em, em.shape)
            dset_em[:, :] = em.astype(np.float32)
        if value is not None:
            dset_value = ensure_dataset(f, args.out_value, value.shape)
            dset_value[:, :] = value.astype(np.float32)
        if eps_growth is not None:
            dset_epsg = ensure_dataset(f, args.out_eps_growth, eps_growth.shape)
            dset_epsg[:, :] = eps_growth.astype(np.float32)
        if sales_growth is not None:
            dset_salesg = ensure_dataset(f, args.out_sales_growth, sales_growth.shape)
            dset_salesg[:, :] = sales_growth.astype(np.float32)
        if growth_comp is not None:
            dset_growth = ensure_dataset(f, args.out_growth, growth_comp.shape)
            dset_growth[:, :] = growth_comp.astype(np.float32)
        if divp is not None:
            dset_divp = ensure_dataset(f, args.out_divp, divp.shape)
            dset_divp[:, :] = divp.astype(np.float32)
        if sentiment is not None:
            dset_sent = ensure_dataset(f, args.out_sentiment, sentiment.shape)
            dset_sent[:, :] = sentiment.astype(np.float32)
        if dividend_yield is not None:
            dset_divy = ensure_dataset(f, args.out_dividend_yield, dividend_yield.shape)
            dset_divy[:, :] = dividend_yield.astype(np.float32)
        if ato is not None:
            dset_ato = ensure_dataset(f, args.out_ato, ato.shape)
            dset_ato[:, :] = ato.astype(np.float32)
        if gp is not None:
            dset_gp = ensure_dataset(f, args.out_gp, gp.shape)
            dset_gp[:, :] = gp.astype(np.float32)
        if gpm is not None:
            dset_gpm = ensure_dataset(f, args.out_gpm, gpm.shape)
            dset_gpm[:, :] = gpm.astype(np.float32)
        if roa is not None:
            dset_roa = ensure_dataset(f, args.out_roa, roa.shape)
            dset_roa[:, :] = roa.astype(np.float32)
        if agrow is not None:
            dset_agrow = ensure_dataset(f, args.out_agrow, agrow.shape)
            dset_agrow[:, :] = agrow.astype(np.float32)
        if issue_grow is not None:
            dset_issue = ensure_dataset(f, args.out_issue_grow, issue_grow.shape)
            dset_issue[:, :] = issue_grow.astype(np.float32)
        if capex_grow is not None:
            dset_capex = ensure_dataset(f, args.out_capex_grow, capex_grow.shape)
            dset_capex[:, :] = capex_grow.astype(np.float32)
        if leverage_comp is not None:
            dset_leverage = ensure_dataset(f, args.out_leverage, leverage_comp.shape)
            dset_leverage[:, :] = leverage_comp.astype(np.float32)
        if earn_var_comp is not None:
            dset_earn_var = ensure_dataset(f, args.out_earn_var, earn_var_comp.shape)
            dset_earn_var[:, :] = earn_var_comp.astype(np.float32)
        if earn_qual_comp is not None:
            dset_earn_qual = ensure_dataset(f, args.out_earn_qual, earn_qual_comp.shape)
            dset_earn_qual[:, :] = earn_qual_comp.astype(np.float32)
        if profit_comp is not None:
            dset_profit = ensure_dataset(f, args.out_profit, profit_comp.shape)
            dset_profit[:, :] = profit_comp.astype(np.float32)
        if invest_qual_comp is not None:
            dset_invest = ensure_dataset(f, args.out_invest_qual, invest_qual_comp.shape)
            dset_invest[:, :] = invest_qual_comp.astype(np.float32)
        if quality_comp is not None:
            dset_quality = ensure_dataset(f, args.out_quality, quality_comp.shape)
            dset_quality[:, :] = quality_comp.astype(np.float32)

    print("saved", args.out_mlev, args.out_blev, args.out_dtoa, "in", args.h5)
    if args.ld_from_longdeb:
        print("LD estimated as total_liab * fina_longdeb_to_debt (--ld-from-longdeb).")
    elif args.ld_dataset:
        print("LD from dataset:", args.ld_dataset)
    else:
        print("LD = TL - TCL (non-current liabilities), because --ld-dataset not provided.")
    if not args.pe_dataset:
        print("PE = 0, because --pe-dataset not provided.")
    elif args.balance_lyr:
        print("PE/LD/BE/TL/TA use previous fiscal year-end value (--balance-lyr).")
    elif args.pe_lyr:
        print("PE uses previous fiscal year-end value (--pe-lyr).")
    if args.balance_scale != 1.0 or args.me_scale != 1.0:
        print("scales: balance", args.balance_scale, "me", args.me_scale)
    if var_sales is not None or var_earnings is not None or var_cfo is not None:
        print("earnings variability window:", args.ev_years, "years, annual mmdd:", args.ev_annual_mmdd)
    if abs_factor is not None or acf_factor is not None:
        print("accruals factors saved as", args.out_abs, args.out_acf)
    if af_eps_to_price is not None:
        print("analyst EPS std-to-price saved as", args.out_af_std_epsp)
    if bp is not None or ep is not None or epf is not None or cep is not None or em is not None:
        print("valuation factors saved as", args.out_bp, args.out_ep, args.out_epf, args.out_cep, args.out_em)
    if value is not None:
        print("value composite saved as", args.out_value)
    if eps_growth is not None or sales_growth is not None or growth_comp is not None:
        print("growth factors saved as", args.out_eps_growth, args.out_sales_growth, args.out_growth)
    if divp is not None:
        print("dividend-to-price saved as", args.out_divp)
    if sentiment is not None:
        print("sentiment saved as", args.out_sentiment)
    if dividend_yield is not None:
        print("dividend yield saved as", args.out_dividend_yield)
    if ato is not None or gp is not None or gpm is not None or roa is not None:
        print("profitability factors saved as", args.out_ato, args.out_gp, args.out_gpm, args.out_roa)
    if agrow is not None or issue_grow is not None or capex_grow is not None:
        print("growth factors saved as", args.out_agrow, args.out_issue_grow, args.out_capex_grow)
    if quality_comp is not None:
        print(
            "quality composites saved as",
            args.out_leverage,
            args.out_earn_var,
            args.out_earn_qual,
            args.out_profit,
            args.out_invest_qual,
            args.out_quality,
        )


if __name__ == "__main__":
    main()
