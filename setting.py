
token = "1e266a5110f1d8fd926d3af0d034458b9d5c904636c72c723ab9fa38"
DB_NAME = 'cb'
config = {
  'user': 'root',
  'password': '586084',
  'host': '192.168.18.109',
  'database': DB_NAME,
  'raise_on_warnings': True
}

def convert_symbol(symbol):
    return symbol.replace('SZ', 'SZE').replace('SH', 'SSE')