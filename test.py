import h5py as h5
import os
import numpy as np
import pandas as pd 
import tushare as ts
import time 
from datetime import datetime

from python_utils.quant import date_range
from setting import *

def change_ts(sym):
    return sym.replace('SSE', 'SH').replace('SZE', 'SZ').replace('BSE', 'BJ')


def get_last_quarter_end(x, date, col_name='ann_date'):
    if x.shape[0] == 0:
        return None, None
    
    idx = np.searchsorted(x[col_name], date)
    if idx > 0:
        idx -= 1
    
    y = x[x[col_name] <= date]
    if y.shape[0] == 0:
        if x.iloc[idx][col_name] <= date:
            return x.iloc[idx]['end_date'], None
        else:
            return None, None
    else:
        prev_idx = np.searchsorted(y['end_date'], x.iloc[idx]['end_date']-1)
        if prev_idx > 0:
            prev_idx = prev_idx - 1 
        
    # print('***', idx, prev_idx, date, x.iloc[idx]['end_date'], x.iloc[prev_idx]['end_date'])
    if x.iloc[idx][col_name] <= date:
        return x.iloc[idx]['end_date'], x.iloc[prev_idx]['end_date']
    else:
        return None, None

class DataGenerator:
    def __init__(self, ts_token, h5_path, earliest_date=20070101, latest_date=20251231):
        self.ts_token = ts_token
        self.h5_path = h5_path
        
        self.pro = ts.pro_api(self.ts_token)
        self._all_dates = date_range(earliest_date, latest_date)
        
    def initialize(self):
        self._universe = self.load_universe()
        self._encode_symbols = np.array([x for x in self._universe], dtype=h5.string_dtype('utf-8', 10))
        
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                f.create_dataset('symbols', (self._universe.shape[0],), maxshape=(None,), dtype=h5.string_dtype('utf-8', 10), data=self._encode_symbols)
            
            self._valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            self._valid_dates = f['dates'][()]
            
        if False:
            balance_sheet_path = 'data/balance_sheet.20070101-20251231.csv'
            fina_indicator_path = 'data/fina_indicator.20070101-20251231.csv'
            self._balance_df = pd.read_csv(balance_sheet_path)
            self._fina_df = pd.read_csv(fina_indicator_path)
            self._express_df = pd.read_csv('data/express.20070101-20251231.csv')
            self._forecast_df = pd.read_csv('data/forecast.20070101-20251231.csv')
            
    def _locate_bsrow(self, symbol, date):
        ts_symbol = change_ts(symbol)
        x = self._balance_df[self._balance_df.ts_code == ts_symbol].sort_values(by=['f_ann_date', 'end_date'])
        last_quarter_end, _ = get_last_quarter_end(x, date, 'f_ann_date')
        if last_quarter_end is None:
            return None
        
        x = x[x.end_date == last_quarter_end]
        idx = np.searchsorted(x.f_ann_date, date)
        if idx > 0:
            idx -= 1

        if x.shape[0] == 0:
            print('no balance sheet data for symbol ', symbol, ' at date ', date)
            return None
        
        # print('x is ', x, x.iloc[idx].f_ann_date, date)
            
        assert x.iloc[idx].f_ann_date <= date
        return x.iloc[idx]
    
    def _locate_finarow(self, symbol, date):
        ts_symbol = change_ts(symbol)
        x = self._fina_df[self._fina_df.ts_code == ts_symbol].sort_values(by=['ann_date', 'end_date'])
        last_quarter_end, _ = get_last_quarter_end(x, date, 'ann_date')
        if last_quarter_end is None:
            return None
        
        x = x[x.end_date == last_quarter_end]
        idx = np.searchsorted(x.ann_date, date)
        if idx > 0:
            idx -= 1
            
        if x.shape[0] == 0:
            return None

        assert x.iloc[idx].ann_date <= date
        return x.iloc[idx]
        
    def load_universe(self):
        dataL = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date,market')
        dataP = self.pro.stock_basic(exchange='', list_status='P', fields='ts_code,symbol,name,area,industry,list_date,market')
        dataD = self.pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name,area,industry,list_date,market')
        all_data = pd.concat([dataL, dataP, dataD]).reset_index(drop=True)
        all_symbols = np.sort(all_data.ts_code)
        for i, sym in enumerate(all_symbols):
            if sym[0] not in ['0', '3', '6', '4', '8', '9']:
                print(i, sym)
                
        all_symbols = all_symbols[:-1]
        return np.array([sym.replace('SZ', 'SZE').replace('SH', 'SSE').replace('BJ', 'BSE') for sym in all_symbols])

    def save_balancesheet(self, path='data/balance_sheet.20070101-20251231.csv'):
        t1, t2, t3, t4 = '20070101', '20171231', '20180101', '20251231'
        concat_df = []
        for sym in self._universe:
            print(sym)
            df1 = self.pro.balancesheet(ts_code=change_ts(sym), start_date=t1, end_date=t2)
            df2 = self.pro.balancesheet(ts_code=change_ts(sym), start_date=t3, end_date=t4)
            concat_df.append(df1)
            concat_df.append(df2)
            time.sleep(0.2)
            
        final_df = pd.concat(concat_df)
        final_df.to_csv(path, index=False)
        
    def save_express(self, path='data/express.20070101-20251231.csv'):
        t1, t2, t3, t4 = '20070101', '20171231', '20180101', '20251231'
        concat_df = []
        for sym in self._universe:
            print(sym)
            df1 = self.pro.express(ts_code=change_ts(sym), start_date=t1, end_date=t2)
            df2 = self.pro.express(ts_code=change_ts(sym), start_date=t3, end_date=t4)
            concat_df.append(df1)
            concat_df.append(df2)
            time.sleep(0.5)
            
        final_df = pd.concat(concat_df)
        final_df.to_csv(path, index=False)
        
    def save_forecast(self, path='data/forecast.20070101-20251231.csv'):
        t1, t2, t3, t4 = '20070101', '20171231', '20180101', '20251231'
        concat_df = []
        for sym in self._universe:
            # if sym != '600768.SSE':
            #     continue
            print(sym)
            df1 = self.pro.forecast(ts_code=change_ts(sym), start_date=t1, end_date=t2)
            df2 = self.pro.forecast(ts_code=change_ts(sym), start_date=t3, end_date=t4)
            concat_df.append(df1)
            concat_df.append(df2)
            time.sleep(0.3)
            
        final_df = pd.concat(concat_df)
        final_df.to_csv(path, index=False)
        
    def save_fina(self, path='data/fina_indicator.20070101-20251231.csv'):
        t1, t2, t3, t4 = '20070101', '20171231', '20180101', '20251231'
        concat_df = []
        for sym in self._universe:
            print(sym)
            df1 = self.pro.fina_indicator(ts_code=change_ts(sym), start_date=t1, end_date=t2)
            df2 = self.pro.fina_indicator(ts_code=change_ts(sym), start_date=t3, end_date=t4)
            concat_df.append(df1)
            concat_df.append(df2)
            time.sleep(0.2)
            
        final_df = pd.concat(concat_df)
        final_df.to_csv(path, index=False)

    def update_balance_fina_cache(
        self,
        balance_path='data/balance_sheet.20070101-20251231.csv',
        fina_path='data/fina_indicator.20070101-20251231.csv',
    ):
        def _load_or_empty(path):
            if os.path.exists(path):
                return pd.read_csv(path)
            return pd.DataFrame()

        def _latest_ann_date(df, ts_sym):
            if df.empty or 'ann_date' not in df.columns:
                return None
            x = df[df.ts_code == ts_sym]
            if x.empty:
                return None
            ann = x['ann_date'].dropna()
            if ann.empty:
                return None
            return int(ann.max())

        def _fetch_incremental(fetch_fn, ts_sym, last_ann):
            start_date = None
            end_date = datetime.today().strftime('%Y%m%d')
            if last_ann is not None:
                start_date = str(last_ann + 1)
            if start_date:
                return fetch_fn(ts_code=ts_sym, start_date=start_date, end_date=end_date)
            return fetch_fn(ts_code=ts_sym)

        balance_df = _load_or_empty(balance_path)
        fina_df = _load_or_empty(fina_path)

        balance_new = []
        fina_new = []

        for sym in self._universe:
            ts_sym = change_ts(sym)
            last_bal = _latest_ann_date(balance_df, ts_sym)
            last_fina = _latest_ann_date(fina_df, ts_sym)

            df_bal = _fetch_incremental(self.pro.balancesheet, ts_sym, last_bal)
            df_fina = _fetch_incremental(self.pro.fina_indicator, ts_sym, last_fina)

            if df_bal is not None and not df_bal.empty:
                balance_new.append(df_bal)
            if df_fina is not None and not df_fina.empty:
                fina_new.append(df_fina)

            time.sleep(0.2)

        if balance_new:
            balance_df = pd.concat([balance_df] + balance_new, ignore_index=True)
            balance_df = balance_df.drop_duplicates()
            balance_df.to_csv(balance_path, index=False)

        if fina_new:
            fina_df = pd.concat([fina_df] + fina_new, ignore_index=True)
            fina_df = fina_df.drop_duplicates()
            fina_df.to_csv(fina_path, index=False)


    def update(self, start_date, end_date):
        dates = date_range(start_date, end_date)
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                f.create_dataset('symbols', (self._universe.shape[0],), maxshape=(None,), dtype=h5.string_dtype('utf-8', 10), data=self._encode_symbols)
            
            valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            valid_dates = f['dates'][()]
            print(valid_dates)
            print(valid_symbols)
            
            for d, date in enumerate(dates):
                date_idx = np.where(valid_dates == date)
                print(date, date_idx)
                
                self.update_daily(date, date_idx, valid_dates, valid_symbols, f)
                self.update_adj_factor(date, date_idx, valid_dates, valid_symbols, f)
                self.update_stk_limit(date, date_idx, valid_dates, valid_symbols, f)
                self.update_daily_basic(date, date_idx, valid_dates, valid_symbols, f)
    
    
    
          
    # Propagte 
    def propage(self):
        # TODO, must propagate total_mv , 不然计算pe有点难受
        pass

    def update_daily(self, trade_date, date_idx, valid_dates, valid_symbols, f):
        df = None 
        for _ in range(3):
            try:   
                df = self.pro.daily(trade_date=int(trade_date))
                break
            except:
                time.sleep(1)
        
        if df is None:
            print(f"daily Failed to fetch data for {trade_date}")
            return None
        else:
            df1_cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
            self.process(df, df1_cols, date_idx, valid_dates, valid_symbols, f)
            
    def update_adj_factor(self, trade_date, date_idx, valid_dates, valid_symbols, f):
        df = None 
        for _ in range(3):
            try:   
                df = self.pro.adj_factor(trade_date=int(trade_date))
                break
            except:
                time.sleep(1)
        
        if df is None:
            print(f"adj_factor Failed to fetch data for {trade_date}")
            return None
        else:
            self.process(df, ['adj_factor'], date_idx, valid_dates, valid_symbols, f)
            
    def update_stk_limit(self, trade_date, date_idx, valid_dates, valid_symbols, f):
        df = None 
        for _ in range(3):
            try:   
                df = self.pro.stk_limit(trade_date=int(trade_date))
                break
            except:
                time.sleep(1)
        
        if df is None:
            print(f"stk_limit Failed to fetch data for {trade_date}")
            return None
        else:
            self.process(df, ['up_limit', 'down_limit'], date_idx, valid_dates, valid_symbols, f)
            
    def update_daily_basic(self, trade_date, date_idx, valid_dates, valid_symbols, f):
        df = None 
        for _ in range(3):
            try:   
                df = self.pro.daily_basic(trade_date=int(trade_date))
                df = df.drop(['close'], axis=1)      
                break
            except:
                time.sleep(1)
        
        if df is None:
            print(f"daily_basic Failed to fetch data for {trade_date}")
            return None
        else:
            self.process(df, ['turnover_rate', 'turnover_rate_f',
       'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
       'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
       'circ_mv'], date_idx, valid_dates, valid_symbols, f)
            
    def process(self, df, target_cols, date_idx, valid_dates, valid_symbols, f):
        df_symbols = np.array([x.replace('SH', 'SSE').replace('SZ', 'SZE').replace('BJ', 'BSE') for x in df['ts_code'].values],dtype=str)
        idx = [np.where(df_symbols == sym) for sym in self._universe]   
        for col in target_cols:
            print(col)
            if col not in f:
                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
            else:
                dset = f[col]
                
            for g, ind in enumerate(idx):
                if ind[0].size != 0:                
                    # print(np.round(df.iloc[ind[0][0]][col].astype(np.float64), 3))
                    dset[date_idx[0][0], g] = np.round(df.iloc[ind[0][0]][col].astype(np.float64), 3)
                else:
                    dset[date_idx[0][0], g] = np.nan
                    
    def fill_express(self):
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                print('symbols is not created')
                return
            
            valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            valid_dates = f['dates'][()]
            print(valid_dates)
            print(valid_symbols)
            
            for g, sym in enumerate(dg._valid_symbols):
                # if sym != '600768.SSE':
                #     continue
                ts_sym = change_ts(sym)
                print(ts_sym)

                x = dg._express_df[dg._express_df.ts_code == ts_sym].sort_values(by=['ann_date', 'end_date']).drop_duplicates()
                x = x.dropna(subset=['ann_date'], inplace=False)

                for i in range(x.shape[0]):
                    if i != x.shape[0] - 1:
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        ed1 = np.where(valid_dates <= x.iloc[i+1].ann_date)
                        # print(x.iloc[i]['ann_date'], x.iloc[i]['end_date'], x.iloc[i+1]['ann_date'])
                        if ed1[0].shape[0] == 0:
                            continue
                        else:
                            # print(ts_sym, st1[0][0], ed1[0][-1]+1, np.round(x.iloc[i]['bps'], 3))
                            for col in ['total_hldr_eqy_exc_min_int', 'n_income', 'operate_profit']:
                                name = col + '_express'                                
                                if name not in f:
                                    dset = f.create_dataset(name, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[name]
                                # print(ts_sym, col)
                                dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i][col], 3)
                                
                    else:
                        # print('ann date 1 = ', x.iloc[i].f_ann_date)
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        # print('st1 , ', st1, ts_sym)
                        # print(st1[0][0])
                        for col in ['total_hldr_eqy_exc_min_int', 'n_income', 'operate_profit']:
                            name = col + '_express'                                
                            if name not in f:
                                dset = f.create_dataset(name, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[name]
                            dset[st1[0][0]:, g] = np.round(x.iloc[i][col], 3)
                            
    def fill_forecast(self):
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                print('symbols is not created')
                return
            
            valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            valid_dates = f['dates'][()]
            print(valid_dates)
            print(valid_symbols)
            
            for g, sym in enumerate(dg._valid_symbols):
                # if sym != '600768.SSE':
                #     continue
                ts_sym = change_ts(sym)
                print(ts_sym)

                x = dg._forecast_df[dg._forecast_df.ts_code == ts_sym].sort_values(by=['ann_date', 'end_date']).drop_duplicates()
                x = x.dropna(subset=['ann_date'], inplace=False)

                for i in range(x.shape[0]):
                    if i != x.shape[0] - 1:
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        ed1 = np.where(valid_dates <= x.iloc[i+1].ann_date)
                        # print(x.iloc[i]['ann_date'], x.iloc[i]['end_date'], x.iloc[i+1]['ann_date'])
                        if ed1[0].shape[0] == 0:
                            continue
                        else:
                            # print(ts_sym, st1[0][0], ed1[0][-1]+1, np.round(x.iloc[i]['bps'], 3))
                            for col in ['net_profit_min', 'net_profit_max']:
                                name = col + '_forecast'                                
                                if name not in f:
                                    dset = f.create_dataset(name, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[name]
                                # print(ts_sym, col)
                                dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i][col], 3)
                                
                    else:
                        # print('ann date 1 = ', x.iloc[i].f_ann_date)
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        # print('st1 , ', st1, ts_sym)
                        # print(st1[0][0])
                        for col in ['net_profit_min', 'net_profit_max']:
                            name = col + '_forecast'                                
                            if name not in f:
                                dset = f.create_dataset(name, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[name]
                            dset[st1[0][0]:, g] = np.round(x.iloc[i][col], 3)
            
    
    def fill_fina(self):
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                # f.create_dataset('symbols', (self._universe.shape[0],), maxshape=(None,), dtype=h5.string_dtype('utf-8', 10), data=self._encode_symbols)
                print('symbols is not created')
                return
            
            valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            valid_dates = f['dates'][()]
            print(valid_dates)
            print(valid_symbols)
            
            for g, sym in enumerate(dg._valid_symbols):
                # if sym != '600768.SSE':
                #     continue
                
                ts_sym = change_ts(sym)
                print(ts_sym)

                x = dg._fina_df[dg._fina_df.ts_code == ts_sym].sort_values(by=['ann_date', 'end_date']).drop_duplicates()
                x = x.dropna(subset=['ann_date'], inplace=False)
                
                exp = self._express_df[self._express_df.ts_code == ts_sym]
                exp = exp.dropna(axis=0, subset=['n_income'])
                fc = self._forecast_df[self._forecast_df.ts_code == ts_sym]
                fc = fc.dropna(axis=0, subset=['net_profit_min', 'net_profit_max'])

                last_end = None
                last_begin = None
                
                for i in range(x.shape[0]):
                    if i != x.shape[0] - 1:
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        ed1 = np.where(valid_dates <= x.iloc[i+1].ann_date)
                        # print(x.iloc[i]['ann_date'], x.iloc[i]['end_date'], x.iloc[i+1]['ann_date'])
                        if ed1[0].shape[0] == 0:
                            continue
                        else:
                            # print(ts_sym, st1[0][0], ed1[0][-1]+1, np.round(x.iloc[i]['bps'], 3))
                            for col in ['bps', 'profit_dedt', 'roe_dt', 'extra_item']:
                                if col not in f:
                                    dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[col]
                                # print(ts_sym, col)
                                dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i][col], 3)
                                # print('assigning from ', valid_dates[st1[0][0]], ' to ', valid_dates[ed1[0][-1]], ' for symbol ', ts_sym, ' col ', col)
                                
                                
                            for col in ['profit_dedtQ']:
                                if col not in f:
                                    dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[col]
                                
                                if str(x.iloc[i].end_date)[5] == '3':
                                    # print('2  ', ts_sym, col, dset[st1[0][0]:ed1[0][-1]+1, g])
                                    dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i]['profit_dedt'], 3) # same as profit_dedtQ
                                else:
                                    idx = np.where(x.end_date < x.iloc[i].end_date)
                                    if idx[0].size > 0:
                                        # print('idx is ', idx, 'idx -1 is ', idx[0], idx[0].shape, x.iloc[idx[0][-1]].end_date, x.iloc[i].end_date)
                                        dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i]['profit_dedt'], 3) - np.round(x.iloc[idx[0][-1]]['profit_dedt'], 3)
                                    else:
                                        dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i]['profit_dedt'], 3)
                    
                            for col in ['net_profit_fcast']:
                                if col not in f:
                                    dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[col]
                                    
                                if col+'Q' not in f:
                                    dset2 = f.create_dataset(col+'Q', (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset2 = f[col+'Q']
                                
                                lastExtra = 0
                                if str(x.iloc[i].end_date)[5] == '3':
                                    lastExtra = 0
                                else:
                                    idx = np.where(x.end_date < x.iloc[i].end_date)
                                    if idx[0].size > 0:
                                        lastExtra = np.round(x.iloc[idx[0][-1]]['profit_dedt'], 3)
                                print(col, x.iloc[i].end_date, valid_dates[last_begin] ,valid_dates[last_end], lastExtra)
                                
                                data = fc[fc.end_date == x.iloc[i].end_date]
                                dataExp = exp[exp.end_date == x.iloc[i].end_date]
                                
                                if (data.shape[0] > 0 or dataExp.shape[0] > 0) and last_begin != last_end:
                                    idx1 = 0
                                    idx2 = 0
                                    
                                    while idx1 < data.shape[0] and idx2 < dataExp.shape[0]:
                                        if data.iloc[idx1].ann_date < dataExp.iloc[idx2].ann_date:
                                            this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                            dset[this_one[0][0]:last_end, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4   
                                            dset2[this_one[0][0]:last_end, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra  
                                            idx1 += 1
                                        else:
                                            this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                            dset[this_one[0][0]:last_end, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                            dset2[this_one[0][0]:last_end, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                            idx2 += 1 
                                    
                                    while idx1 < data.shape[0]:
                                        this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                        dset[this_one[0][0]:last_end, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4
                                        dset2[this_one[0][0]:last_end, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra
                                        idx1 += 1
                                    
                                    while idx2 < dataExp.shape[0]:
                                        this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                        dset[this_one[0][0]:last_end, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                        dset2[this_one[0][0]:last_end, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                        idx2 += 1                              
                                else:
                                    dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i].extra_item + x.iloc[i].profit_dedt, 3)
                                    dset2[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i].profit_dedt, 3) 
                                        
                            
                            last_end = ed1[0][-1]+1
                            last_begin = st1[0][0]  
                    else:
                        # print('ann date 1 = ', x.iloc[i].ann_date)
                        st1 = np.where(valid_dates > x.iloc[i].ann_date)
                        # print('st1 , ', st1, ts_sym)
                        # print(st1[0][0])
                        for col in ['bps', 'profit_dedt', 'roe_dt', 'extra_item']:
                            if col not in f:
                                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[col]
                            dset[st1[0][0]:, g] = np.round(x.iloc[i][col], 3)
                            # print('assigning from ', valid_dates[st1[0][0]], ' to ', valid_dates[-1], ' for symbol ', ts_sym, ' col ', col)
                            # if col == 'bps':
                            #     print('latest bps is ', np.round(x.iloc[i][col], 3))
                            
                        for col in ['profit_dedtQ']:
                            if col not in f:
                                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[col]
                            
                            if str(x.iloc[i].end_date)[5] == '3':
                                # print('2  ', ts_sym, col, dset[st1[0][0]:ed1[0][-1]+1, g])
                                dset[st1[0][0]:, g] = np.round(x.iloc[i]['profit_dedt'], 3) # same as profit_dedtQ
                            else:
                                idx = np.where(x.end_date < x.iloc[i].end_date)
                                if idx[0].size > 0:
                                    # print('idx is ', idx, 'idx -1 is ', idx[0], idx[0].shape, x.iloc[idx[0][-1]].end_date, x.iloc[i].end_date)
                                    dset[st1[0][0]:, g] = np.round(x.iloc[i]['profit_dedt'], 3) - np.round(x.iloc[idx[0][-1]]['profit_dedt'], 3)
                                else:
                                    dset[st1[0][0]:, g] = np.round(x.iloc[i]['profit_dedt'], 3)


                        for col in ['net_profit_fcast']:
                            if col not in f:
                                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[col]
                                
                            if col+'Q' not in f:
                                dset2 = f.create_dataset(col+'Q', (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset2 = f[col+'Q']
                            
                            lastExtra = 0
                            if str(x.iloc[i].end_date)[5] == '3':
                                lastExtra = 0
                            else:
                                idx = np.where(x.end_date < x.iloc[i].end_date)
                                if idx[0].size > 0:
                                    lastExtra = np.round(x.iloc[idx[0][-1]]['profit_dedt'], 3)
                            print(col, x.iloc[i].end_date, valid_dates[last_begin] ,valid_dates[last_end], lastExtra)
                            
                            data = fc[fc.end_date == x.iloc[i].end_date]
                            dataExp = exp[exp.end_date == x.iloc[i].end_date]
                            
                            if (data.shape[0] > 0 or dataExp.shape[0] > 0):
                                idx1 = 0
                                idx2 = 0
                                
                                while idx1 < data.shape[0] and idx2 < dataExp.shape[0]:
                                    if data.iloc[idx1].ann_date < dataExp.iloc[idx2].ann_date:
                                        this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                        dset[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4    
                                        dset2[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra  
                                        # print(valid_dates[this_one[0][0]])
                                        idx1 += 1
                                    else:
                                        this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                        # print(valid_dates[this_one[0][0]])
                                        
                                        dset[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                        dset2[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                        idx2 += 1 
                                
                                while idx1 < data.shape[0]:
                                    this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                    # print(valid_dates[this_one[0][0]])
                                    
                                    dset[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4
                                    dset2[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra
                                    idx1 += 1
                                
                                while idx2 < dataExp.shape[0]:
                                    this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                    # print(valid_dates[this_one[0][0]])
                                    
                                    dset[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                    dset2[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                    idx2 += 1                              
                            else:
                                # print(valid_dates[st1[0][0]])
                                dset[st1[0][0]:, g] = np.round(x.iloc[i].extra_item + x.iloc[i].profit_dedt, 3)
                                dset2[st1[0][0]:, g] = np.round(x.iloc[i].profit_dedt, 3) 
                            
                            data = fc[fc.end_date > x.iloc[i].end_date]
                            dataExp = exp[exp.end_date > x.iloc[i].end_date]
                            
                            
                            
                            if (data.shape[0] > 0 or dataExp.shape[0] > 0):
                                
                                lastExtra = 0
                                if data.shape[0] > 0:
                                    date = data.iloc[0].end_date
                                else:
                                    date = dataExp.iloc[0].end_date
                                    
                                if str(date)[5] == '3':
                                    lastExtra = 0
                                else:
                                    idx = np.where(x.end_date < date)
                                    if idx[0].size > 0:
                                        lastExtra = np.round(x.iloc[idx[0][-1]]['profit_dedt'], 3)
                                print(col, lastExtra, valid_dates[last_begin] ,valid_dates[last_end], lastExtra)
                                
                                idx1 = 0
                                idx2 = 0
                                
                                while idx1 < data.shape[0] and idx2 < dataExp.shape[0]:
                                    if data.iloc[idx1].ann_date < dataExp.iloc[idx2].ann_date:
                                        this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                        # print(valid_dates[this_one[0][0]])
                                        
                                        dset[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4   
                                        dset2[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra  
                                        idx1 += 1
                                    else:
                                        this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                        # print(valid_dates[this_one[0][0]])
                                        
                                        dset[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                        dset2[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                        idx2 += 1 
                                
                                while idx1 < data.shape[0]:
                                    this_one = np.where(valid_dates >= data.iloc[idx1].ann_date)
                                    # print(valid_dates[this_one[0][0]])
                                    
                                    dset[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4
                                    dset2[this_one[0][0]:, g] = np.round(data.iloc[idx1]['net_profit_min'] * 0.5 + data.iloc[idx1]['net_profit_max'] * 0.5, 3) * 1e4 - lastExtra
                                    idx1 += 1
                                
                                while idx2 < dataExp.shape[0]:
                                    this_one = np.where(valid_dates > dataExp.iloc[idx2].ann_date)
                                    # print(valid_dates[this_one[0][0]])
                                    
                                    dset[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3)
                                    dset2[this_one[0][0]:, g] = np.round(dataExp.iloc[idx2]['n_income'], 3) - lastExtra
                                    idx2 += 1                              
                            else:
                                # print(valid_dates[st1[0][0]])
                                dset[st1[0][0]:, g] = np.round(x.iloc[i].extra_item + x.iloc[i].profit_dedt, 3)
                                dset2[st1[0][0]:, g] = np.round(x.iloc[i].profit_dedt, 3)   
                                
                            # from IPython import embed
                            # embed()
                            

    def fill_balance(self):
        with h5.File(self.h5_path, 'a') as f:
            if 'dates' not in f:
                f.create_dataset('dates', (self._all_dates.shape[0],), maxshape=(None,), data=self._all_dates)
                
            if 'symbols' not in f:
                print('symbols is not created')
                return
            
            valid_symbols = np.array([x.decode('utf-8') for x in f['symbols'][()]])
            valid_dates = f['dates'][()]
            print(valid_dates)
            print(valid_symbols)
            
            for g, sym in enumerate(dg._valid_symbols):
                # if sym != '000088.SZE':
                #     continue
                ts_sym = change_ts(sym)
                print(ts_sym)

                x = dg._balance_df[dg._balance_df.ts_code == ts_sym].sort_values(by=['f_ann_date', 'end_date']).drop_duplicates()
                x = x.dropna(subset=['f_ann_date'], inplace=False)
                
                exp = self._express_df[self._express_df.ts_code == ts_sym]
                exp = exp.dropna(axis=0, subset=['total_hldr_eqy_exc_min_int'])
                # fc = self._forecast_df[self._forecast_df.ts_code == ts_sym]

                last_end = None
                last_begin = None
                for i in range(x.shape[0]):
                    if i != x.shape[0] - 1:
                        st1 = np.where(valid_dates > x.iloc[i].f_ann_date)
                        ed1 = np.where(valid_dates <= x.iloc[i+1].f_ann_date)
                        # print(x.iloc[i]['f_ann_date'], x.iloc[i]['end_date'], x.iloc[i+1]['f_ann_date'])
                        if ed1[0].shape[0] == 0:
                            continue
                        else:
                            # print(ts_sym, st1[0][0], ed1[0][-1]+1, np.round(x.iloc[i]['bps'], 3))
                            for col in ['goodwill', 'total_hldr_eqy_exc_min_int']:                                
                                if col not in f:
                                    dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[col]
                                # print(ts_sym, col)
                                dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i][col], 3)
                                
                            for col in ['total_hldr_eqy_exc_min_int_fcast']:
                                if col not in f:
                                    dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                                else:
                                    dset = f[col]
                                print(col, x.iloc[i].end_date, valid_dates[last_begin] ,valid_dates[last_end])
                                
                                data = exp[exp.end_date == x.iloc[i].end_date]
                                if data.shape[0] > 0 and last_begin != last_end:
                                    # print(data)
                                    if data.iloc[-1].ann_date <= valid_dates[last_begin]:
                                        print('error exp ann date')
                                    # assert data.ann_date.values > valid_dates[last_begin], ' error exp ann date'
                                    # assert data.ann_date.values < valid_dates[last_end], ' error exp ann date'
                                    this_one = np.where(valid_dates >= data.iloc[-1].ann_date)
                                    dset[this_one[0][0]:last_end, g] = np.round(data.iloc[-1]['total_hldr_eqy_exc_min_int'], 3)
                                    # print('this one ', valid_dates[this_one[0][0]], data.iloc[0]['total_hldr_eqy_exc_min_int']) 
                                # elif data.shape[0] > 1 and last_begin != last_end:
                                #     from IPython import embed
                                #     embed()
                                else:
                                    dset[st1[0][0]:ed1[0][-1]+1, g] = np.round(x.iloc[i]['total_hldr_eqy_exc_min_int'], 3)
                            
                            last_end = ed1[0][-1]+1
                            last_begin = st1[0][0]        
                    else:
                        # print('ann date 1 = ', x.iloc[i].f_ann_date)
                        st1 = np.where(valid_dates > x.iloc[i].f_ann_date)
                        # print('st1 , ', st1, ts_sym)
                        # print(st1[0][0])
                        for col in ['goodwill', 'total_hldr_eqy_exc_min_int']:
                            if col not in f:
                                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[col]
                            dset[st1[0][0]:, g] = np.round(x.iloc[i][col], 3)
                            
                        for col in ['total_hldr_eqy_exc_min_int_fcast']:
                            if col not in f:
                                dset = f.create_dataset(col, (valid_dates.shape[0], valid_symbols.shape[0]), maxshape=(None, None))
                            else:
                                dset = f[col]
                            # print(col, x.iloc[i].end_date, valid_dates[last_begin] ,valid_dates[last_end])
                            
                            data = exp[exp.end_date == x.iloc[i].end_date]
                            exp_remain = exp[exp.end_date > x.iloc[i].end_date]
                            if data.shape[0] > 0:
                                # assert data.ann_date.values > valid_dates[last_begin], ' error exp ann date'
                                # assert data.ann_date.values < valid_dates[last_end], ' error exp ann date'
                                this_one = np.where(valid_dates >= data.iloc[-1].ann_date)
                                dset[this_one[0][0]:last_end, g] = np.round(data.iloc[0]['total_hldr_eqy_exc_min_int'], 3)
                                # print('this one ', valid_dates[this_one[0][0]], data.iloc[0]['total_hldr_eqy_exc_min_int']) 
                            else:
                                dset[st1[0][0]:, g] = np.round(x.iloc[i]['total_hldr_eqy_exc_min_int'], 3)
                                
                            if exp_remain.shape[0] == 1:
                                this_one = np.where(valid_dates > exp_remain.iloc[-1].ann_date)
                                dset[this_one[0][0]:, g] = np.round(exp_remain.iloc[-1]['total_hldr_eqy_exc_min_int'], 3)
                            elif exp_remain.shape[0] > 1:    
                                from IPython import embed
                                embed()
                            


if __name__ == "__main__":
    # dg = DataGenerator(token, '/home/ubuntu/code/backtest/backtest/data/daily.hdf')
    dg = DataGenerator(token, 'daily.hdf')
    dg.initialize()
    
    dg.save_express()
    dg.save_forecast()
    dg.save_balancesheet()
    dg.save_fina()
        
    # dg.fill_express()
    # dg.fill_forecast()
    # dg.fill_balance()
    # dg.fill_fina()
    # final_df = pd.concat(concat_df)
    # final_df.to_csv(path, index=False)
        
    # print(dg.load_universe())
    # dg.initialize()
    # dg.fill_fina()
    # dg.fill_balance()
    # dg.save_balancesheet()
    # dg.save_fina()
    # dg.update(20070110, 20070110)
    
    # dates = dg._all_dates
    # for g, sym in enumerate(['600768.SSE']):
    #     ts_sym = change_ts(sym)
    #     print(ts_sym)

    #     x = dg._fina_df[dg._fina_df.ts_code == ts_sym].sort_values(by=['ann_date', 'end_date']).drop_duplicates()
    #     x = x.dropna(subset=['ann_date'], inplace=False)
    #     for i in range(x.shape[0]):
    #         if i != x.shape[0] - 1:
    #             st1 = np.where(dates > x.iloc[i].ann_date)
    #             ed1 = np.where(dates <= x.iloc[i+1].ann_date)
    #             print(x.iloc[i]['ann_date'], x.iloc[i]['end_date'], x.iloc[i+1]['ann_date'])
    #             if ed1[0].shape[0] == 0:
    #                 continue
    #             else:
    #                 print(st1[0][0], ed1[0][-1]+1, x)
    #         else:
    #             print('ann date 1 = ', x.iloc[i].ann_date)
    #             st1 = np.where(dg._all_dates > x.iloc[i].ann_date)
    #             print('st1 , ', st1)
    #             print(st1[0][0]) 
        
    #     break
