T Symbols
=========
This project excludes symbols that start with "T" from daily.hdf updates.
They should also be removed from daily.hdf (symbols and all 2D fields) to avoid
leaking into factor calculations such as SIZE or LIQUIDITY.

Note: update_vol_cache_all.py filters T* symbols in its update mask.
