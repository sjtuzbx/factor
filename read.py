import h5py
import numpy as np


def main():
    h5_path = "daily.hdf"
    symbols = ["000001.SZE", "000002.SZE", "000004.SZE"]
    target_date = 20221227
    fields = ["BETA", "Hist_sigma", "Daily_std", "Cumulative_range"]
    liq_symbols = ["000001.SZE", "689009.SSE"]
    liq_start = 20221226
    liq_end = 20221230
    liq_fields = [
        "Monthly_share_turnover",
        "Quarterly_share_turnover",
        "Annual_share_turnover",
        "Annualized_traded_value_ratio",
    ]
    mom_symbols = ["000001.SZE", "689009.SSE"]
    mom_start = 20211228
    mom_end = 20220104
    mom_fields = [
        "Short_Term_reversal",
        "Seasonality",
        "Industry_Momentum",
        "Relative_strength",
        "Momentum",
        "Long_Relative_strength",
        "Long_Historical_alpha",
    ]
    vol_symbols = ["000001.SZE"]
    vol_date = 20221230
    vol_fields = [
        "BETA",
        "Hist_sigma",
        "Historical_alpha",
        "Daily_std",
        "Cumulative_range",
        "Volatility",
    ]
    size_fields = ["LNCAP", "MIDCAP", "SIZE"]
    leverage_fields = ["MLEV", "BLEV", "DTOA"]
    variability_fields = ["VAR_SALES", "VAR_EARN", "VAR_CFO"]
    accrual_fields = ["ABS", "ACF"]
    profitability_fields = ["ATO", "GP", "GPM", "ROA"]
    growth_fields = ["AGROW", "ISSUE_GROW", "CAPEX_GROW"]
    valuation_fields = ["BP", "EP", "EPF", "CEP", "EM", "VALUE"]
    extra_style_fields = ["GROWTH", "SENTIMENT", "DIVIDEND_YIELD_COMP"]
    quality_fields = [
        "LEVERAGE",
        "EARN_VAR",
        "EARN_QUAL",
        "PROFIT",
        "INVEST_QUAL",
        "QUALITY",
    ]
    analyst_field = "ANALYST_NP_STD_MV"
    analyst_ranges = [
        ("000001.SZE", 20211228, 20211231),
        ("689009.SSE", 20230103, 20230106),
    ]

    with h5py.File(h5_path, "r") as f:
        dates = f["dates"][:]
        symbols_arr = f["symbols"][:]

        try:
            date_idx = int(np.where(dates == target_date)[0][0])
        except IndexError:
            raise SystemExit(f"date not found: {target_date}")

        symbol_to_idx = {s.decode("utf-8"): i for i, s in enumerate(symbols_arr)}

        print(f"date {target_date}")
        for sym in symbols:
            idx = symbol_to_idx.get(sym)
            if idx is None:
                print(f"{sym} not found")
                continue
            values = []
            for field in fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][date_idx, idx])
                values.append(f"{field}={val}")
            print(f"{sym} " + " ".join(values))

        liq_mask = (dates >= liq_start) & (dates <= liq_end)
        liq_date_idx = np.where(liq_mask)[0]
        liq_dates = dates[liq_date_idx]
        for liq_symbol in liq_symbols:
            liq_idx = symbol_to_idx.get(liq_symbol)
            if liq_idx is None:
                print(f"{liq_symbol} not found for liquidity output")
                continue
            print(f"liquidity {liq_symbol} {liq_start}-{liq_end}")
            for d, row_i in zip(liq_dates, liq_date_idx):
                values = []
                for field in liq_fields:
                    if field not in f:
                        values.append(f"{field}=NA")
                        continue
                    val = float(f[field][row_i, liq_idx])
                    values.append(f"{field}={val}")
                print(f"{int(d)} " + " ".join(values))

        mom_mask = (dates >= mom_start) & (dates <= mom_end)
        mom_date_idx = np.where(mom_mask)[0]
        mom_dates = dates[mom_date_idx]
        for mom_symbol in mom_symbols:
            mom_idx = symbol_to_idx.get(mom_symbol)
            if mom_idx is None:
                print(f"{mom_symbol} not found for momentum output")
                continue
            print(f"momentum {mom_symbol} {mom_start}-{mom_end}")
            for d, row_i in zip(mom_dates, mom_date_idx):
                values = []
                for field in mom_fields:
                    if field not in f:
                        values.append(f"{field}=NA")
                        continue
                    val = float(f[field][row_i, mom_idx])
                    values.append(f"{field}={val}")
                print(f"{int(d)} " + " ".join(values))

        vol_pos = np.where(dates == vol_date)[0]
        if vol_pos.size == 0:
            print(f"vol date not found: {vol_date}")
            return
        vol_date_idx = int(vol_pos[0])
        for vol_symbol in vol_symbols:
            vol_idx = symbol_to_idx.get(vol_symbol)
            if vol_idx is None:
                print(f"{vol_symbol} not found for vol output")
                continue
            values = []
            for field in vol_fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][vol_date_idx, vol_idx])
                values.append(f"{field}={val}")
            for field in size_fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][vol_date_idx, vol_idx])
                values.append(f"{field}={val}")
            for field in leverage_fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][vol_date_idx, vol_idx])
                values.append(f"{field}={val}")
            for field in variability_fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][vol_date_idx, vol_idx])
                values.append(f"{field}={val}")
        for field in accrual_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        for field in profitability_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        for field in growth_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        for field in valuation_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        for field in extra_style_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        for field in quality_fields:
            if field not in f:
                values.append(f"{field}=NA")
                continue
            val = float(f[field][vol_date_idx, vol_idx])
            values.append(f"{field}={val}")
        print(f"vol {vol_symbol} {vol_date} " + " ".join(values))

        quality_symbol = "689009.SSE"
        quality_idx = symbol_to_idx.get(quality_symbol)
        if quality_idx is None:
            print(f"{quality_symbol} not found for quality output")
        else:
            values = []
            for field in quality_fields:
                if field not in f:
                    values.append(f"{field}=NA")
                    continue
                val = float(f[field][vol_date_idx, quality_idx])
                values.append(f"{field}={val}")
            print(f"quality {quality_symbol} {vol_date} " + " ".join(values))

        for sym, start, end in analyst_ranges:
            idx = symbol_to_idx.get(sym)
            if idx is None:
                print(f"{sym} not found for analyst output")
                continue
            date_mask = (dates >= start) & (dates <= end)
            date_idx = np.where(date_mask)[0]
            print(f"analyst {sym} {start}-{end}")
            for d, row_i in zip(dates[date_idx], date_idx):
                if analyst_field not in f:
                    print(f"{int(d)} {analyst_field}=NA")
                    continue
                val = float(f[analyst_field][row_i, idx])
                print(f"{int(d)} {analyst_field}={val}")


if __name__ == "__main__":
    main()
