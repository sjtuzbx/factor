import pandas as pd
import tushare as ts
from setting import token

sym = "000001.SZ"
end_date = "20211231"

fields = (
    "ts_code,end_date,ann_date,f_ann_date,total_liab,total_assets,"
    "total_cur_liab,total_ncl,oth_eqt_tools_p_shr"
)

pro = ts.pro_api(token)
df_ts = pro.balancesheet(ts_code=sym, end_date=end_date, fields=fields)
print("tushare:")
print(df_ts.to_string(index=False))

bal = pd.read_csv("data/balance_sheet.20070101-20251231.csv")
rows = bal[(bal["ts_code"] == sym) & (bal["end_date"].astype(str) == end_date)]
print("\ncsv:")
print(rows[fields.split(",")].to_string(index=False))
