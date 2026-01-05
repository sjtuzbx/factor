# Daily Style Factors: Dependencies & Update Scripts

This document lists the daily update scripts and the upstream datasets required to compute the 9 style factors:
Size, Volatility, Liquidity, Momentum, Quality, Value, Growth, Sentiment, DividendYield.

## Summary Table

| Factor | Script | Output Datasets | Key Inputs (daily.hdf) | Other Inputs |
|---|---|---|---|---|
| Size | `compute_size_factor.py` | `LNCAP`, `MIDCAP`, `SIZE` | `circ_mv`, `dates`, `symbols` | — |
| Volatility | `update_vol_cache_all.py` | `BETA`, `Hist_sigma`, `Historical_alpha`, `Daily_std`, `Cumulative_range`, `Volatility` | `close`, `adj_factor`, `is_listed_in_5days` | `index.daily.hdf` (`000300.SSE` close) |
| Liquidity | `update_liquidity_cache_all.py` | `Monthly_share_turnover`, `Quarterly_share_turnover`, `Annual_share_turnover`, `Annualized_traded_value_ratio`, `LIQUIDITY` | `turnover_rate` | — |
| Momentum | `update_daily_momentum.py` | `Short_Term_reversal`, `Seasonality`, `Industry_Momentum`, `Relative_strength`, `Momentum`, `Long_Relative_strength`, `Long_Historical_alpha` | `close`, `adj_factor`, `is_listed_in_5days`, `industry`, `circ_mv`, `Historical_alpha` | `index.daily.hdf` (`000300.SSE` for long alpha) |
| Quality | `compute_leverage_factors.py` | `LEVERAGE`, `EARN_VAR`, `EARN_QUAL`, `PROFIT`, `INVEST_QUAL`, `QUALITY` | balance/income/cashflow fields listed below | — |
| Value | `compute_leverage_factors.py` | `VALUE` | `BP/EP/EPF/CEP/EM`, `Long_Relative_strength`, `Long_Historical_alpha` | — |
| Growth | `compute_leverage_factors.py` | `EPS_GROWTH_5Y`, `SALES_GROWTH_5Y`, `GROWTH` | `income_basic_eps`, `fina_total_revenue_ps`, `ANALYST_NP_CAGR_2Y` | — |
| Sentiment | `compute_leverage_factors.py` | `SENTIMENT` | `ANALYST_NP_RRIBS`, `ANALYST_EP_CHANGE`, `ANALYST_EPS_CHANGE` | — |
| DividendYield | `compute_leverage_factors.py` | `DIVIDEND_TO_PRICE`, `DIVIDEND_YIELD_COMP` | `income_div_payt`, `total_share`, `close`, `ANALYST_RD_6M_MEAN` | — |
| Analyst Signals | `update_analyst_np_std_mv.py` | `ANALYST_NP_STD_MV`, `ANALYST_NP_FWD12M_MEAN`, `ANALYST_NP_CAGR_2Y`, `ANALYST_NP_RRIBS`, `ANALYST_EP_CHANGE`, `ANALYST_EPS_CHANGE`, `ANALYST_RD_6M_MEAN` | `total_mv`, `dates`, `symbols` | `data/report_rc*.csv` (via `update_report_rc_cache.py`) |

## Detailed Dependencies

### Size
- Script: `compute_size_factor.py`
- Inputs: `circ_mv` (from `cb_cache` daily), `dates`, `symbols`.
- Outputs: `LNCAP`, `MIDCAP`, `SIZE`.

### Volatility
- Script: `update_vol_cache_all.py`
- Inputs (daily.hdf): `close`, `adj_factor`, `is_listed_in_5days`.
- Inputs (index.daily.hdf): `000300.SSE` close for market return.
- Outputs: `BETA`, `Hist_sigma`, `Historical_alpha`, `Daily_std`, `Cumulative_range`, `Volatility`.

### Liquidity
- Script: `update_liquidity_cache_all.py`
- Inputs: `turnover_rate`.
- Outputs: `Monthly_share_turnover`, `Quarterly_share_turnover`, `Annual_share_turnover`, `Annualized_traded_value_ratio`, `LIQUIDITY`.

### Momentum
- Script: `update_daily_momentum.py`
- Inputs: `close`, `adj_factor`, `is_listed_in_5days`, `industry`, `circ_mv`, `Historical_alpha`.
- Long alpha uses `index.daily.hdf` (`000300.SSE`).
- Outputs: `Short_Term_reversal`, `Seasonality`, `Industry_Momentum`, `Relative_strength`, `Momentum`, `Long_Relative_strength`, `Long_Historical_alpha`.

### Quality
- Script: `compute_leverage_factors.py`
- Inputs:
  - Balance sheet: `balance_total_liab`, `balance_total_assets`, `balance_total_cur_liab`, `balance_cash_reser_cb`, `balance_st_borr`, `balance_lt_borr`, `balance_cb_borr`, `balance_pledge_borr`, `balance_end_date`.
  - Income: `income_total_revenue`, `income_revenue`, `income_n_income`, `income_oper_cost`, `income_end_date`, `income_ann_date`.
  - Cashflow: `cashflow_n_cashflow_act`, `cashflow_n_cashflow_inv_act`, `cashflow_depr_fa_coga_dpba`, `cashflow_amort_intang_assets`, `cashflow_c_pay_acq_const_fiolta`, `cashflow_end_date`, `cashflow_ann_date`.
  - Other: `float_share`.
- Outputs: `LEVERAGE`, `EARN_VAR`, `EARN_QUAL`, `PROFIT`, `INVEST_QUAL`, `QUALITY`.

### Value
- Script: `compute_leverage_factors.py`
- Components:
  - `BP`: `balance_total_hldr_eqy_exc_min_int`, `total_mv`, `balance_end_date`.
  - `EP`: `fina_profit_dedt`, `fina_end_date`, `total_mv`.
  - `EPF`: `ANALYST_NP_FWD12M_MEAN`, `total_mv`.
  - `CEP`: `cashflow_n_cashflow_act`, `income_end_date`, `total_mv`.
  - `EM`: `income_ebit`, `balance_total_liab`, `balance_cash_reser_cb`, `total_mv`.
  - `Long_Relative_strength`, `Long_Historical_alpha`.
- Output: `VALUE`.

### Growth
- Script: `compute_leverage_factors.py`
- Components:
  - `EPS_GROWTH_5Y`: `income_basic_eps`, `income_ann_date`, `income_end_date`.
  - `SALES_GROWTH_5Y`: `fina_total_revenue_ps`, `fina_ann_date`, `fina_end_date`.
  - `ANALYST_NP_CAGR_2Y`.
- Output: `GROWTH`.

### Sentiment
- Script: `compute_leverage_factors.py`
- Inputs: `ANALYST_NP_RRIBS`, `ANALYST_EP_CHANGE`, `ANALYST_EPS_CHANGE`.
- Output: `SENTIMENT`.

### DividendYield
- Script: `compute_leverage_factors.py`
- Components:
  - `DIVIDEND_TO_PRICE`: `income_div_payt`, `income_end_date`, `total_share`, `close`.
  - `ANALYST_RD_6M_MEAN`.
- Output: `DIVIDEND_YIELD_COMP`.

### Analyst Signals
- CSV source: `update_report_rc_cache.py` -> `data/report_rc*.csv`
- Script: `update_analyst_np_std_mv.py`
- Inputs: `data/report_rc*.csv`, `total_mv` (from `daily.hdf`)
- Outputs: `ANALYST_NP_STD_MV`, `ANALYST_NP_FWD12M_MEAN`, `ANALYST_NP_CAGR_2Y`, `ANALYST_NP_RRIBS`, `ANALYST_EP_CHANGE`, `ANALYST_EPS_CHANGE`, `ANALYST_RD_6M_MEAN`

## Notes
- `compute_leverage_factors.py` writes many composite factors at once; make sure required datasets exist in `daily.hdf`.
- `fill_balance_fina_daily.py` populates income/cashflow fields into `daily.hdf`.
- `update_balance_fina_cache.py` refreshes the CSV caches used by `fill_balance_fina_daily.py`.
