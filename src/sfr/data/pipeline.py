"""fetch -> save raw -> standardise -> validate -> clean -> returns -> save processed."""
from __future__ import annotations                         

import datetime as dt                                           # load the date/time library

import pandas as pd                                        

from sfr.config import project_path                             # load the path-lookup function
from sfr.data.schema import standardize                         # load the column-standardising function
from sfr.data.validation import clean_prices, validate_prices   # load the validation and cleaning functions
from sfr.data.transforms import compute_returns                 # load the return-computing function
from sfr.data import store                                      # load the file-saving module
from sfr.registry import SOURCES                                # load the list of data sources

# Note: The values ​​below are taken from an actual run on 2026-11-08.
# Within the `for` loop, add a comment noting the value of the FIRST asset (SP500).
# Only the print statements (lines 86 or 99) actually display text in the terminal.

def make_source(cfg: dict, name: str):                     
    params = (cfg["data"].get("sources") or {}).get(name) or {}     #params = {'auto_adjust': False}
    return SOURCES.build(name, **params)                            #return = <YahooSource object> with 'auto_adjust':False


def build_dataset(cfg: dict, assets: list[str] | None = None) -> pd.DataFrame:  #assets = None
    data_cfg = cfg["data"]                                 #data_cfg = all configs/data.yaml = {'default_source':..., 'main_asset':..., 'sources':..., 'assets':..., 'validation':...}
    # {'default_source': 'yahoo',
    #  'main_asset':     'SP500',
    #  'sources': {'yahoo':     {'auto_adjust': False},
    #              'csv':       {'directory': 'data/external', 'filename': '{symbol}.csv', 'date_column': 'Date', 'column_map': {}},
    #              'synthetic': {'mu': 0.07, 'sigma': 0.18, 's0': 100.0, 'jump_intensity': 2.0, 'jump_mean': -0.03, 'jump_std': 0.04, 'seed': 0}},
    #  'assets': {'SP500':  {'symbol': '^GSPC'},    'SPY':  {'symbol': 'SPY'},
    #             'AAPL':   {'symbol': 'AAPL'},     'MSFT': {'symbol': 'MSFT'},
    #             'VIX':    {'symbol': '^VIX'},     'EUROSTOXX50': {'symbol': '^STOXX50E'}},
    #  'validation': {'max_nan_frac': 0.05,     'max_abs_log_return': 0.25,     'max_gap_days': 7,  'strict': False}}
    
    sample = cfg["settings"]["sample"]                     #sample = {'start': '2010-01-01', 'end': None, 'train_end': '2019-12-31'}
    checks = dict(data_cfg.get("validation") or {})        #checks = {'max_nan_frac': 0.05, 'max_abs_log_return': 0.25, 'max_gap_days': 7, 'strict': False}
    strict = checks.pop("strict", False)                   #strict = False; checks = {'max_nan_frac': 0.05, 'max_abs_log_return': 0.25, 'max_gap_days': 7}
    raw_root = project_path(cfg, "raw")                    #raw_root = C:\Users\ASUS\Desktop\Stochastic-Finance-Research\data\raw

    log = []                                               #log = []
    for name in assets or list(data_cfg["assets"]):        #Iterate through ['SP500', 'SPY', 'AAPL', 'MSFT', 'VIX', 'EUROSTOXX50']; first name = 'SP500'
        spec = data_cfg["assets"][name]                                         #spec = {'symbol': '^GSPC'}
        source_name = spec.get("source", data_cfg["default_source"])            #source_name = data_cfg["default_source"] = 'yahoo'
        row = {"asset": name, "symbol": spec["symbol"], "source": source_name,"built_on": dt.date.today().isoformat()}    
        #row = {'asset': 'SP500', 'symbol': '^GSPC', 'source': 'yahoo', 'built_on': '2026-11-08'}

        try:                                              
            source = make_source(cfg, source_name)         #source = <YahooSource object>
            raw = source.fetch(spec["symbol"], sample.get("start"), sample.get("end"))          #raw = table of 4238 rows x 6 columns:
            # Price        Adj Close       Close         High          Low         Open      Volume
            # 2010-01-04  1132.98999  1132.98999  1133.869995  1116.560059  1116.560059  3991400000
            # ...
            # 2026-11-06  7376.859863 7376.859863 7444.200195  7369.549805  7377.649902  5701900000
            
            raw_dir = raw_root / source_name               # raw_dir = ...\data\raw\yahoo
            raw_dir.mkdir(parents=True, exist_ok=True)     # no output (create the document if not exist)
            raw.to_csv(raw_dir / f"{name}.csv")            # save to file data\raw\yahoo\SP500.csv

            prices = standardize(raw, getattr(source, "column_map", None))                      #prices = table of 4238 rows x 6 columns, column names standardised:
            # date (index)    open         high          low       close   adj_close        volume
            # 2010-01-04  1116.560059  1133.869995  1116.560059  1132.98999  1132.98999  3.991400e+09
            # ...
            # 2026-11-06  7377.649902  7444.200195  7369.549805  7376.859863 7376.859863 5.701900e+09
            
            report = validate_prices(prices, name, **checks)                                    #report = ValidationReport(name='SP500', errors=[], warnings=[])
            row["checks"] = report.summary()               #row["checks"] = 'ok' (VIX only: '51 daily moves > 25%') (report is in class ValidationReport)
            if not report.ok:                              #not report.ok = False => skip the line below
                raise ValueError(report.summary())         # does NOT run (no asset has an error)
            
            prices = clean_prices(prices)                                                       #prices = still 4238 rows (SP500 has no missing/negative prices to drop)

            returns = compute_returns(prices)                                                   #returns = table of 4237 rows x 3 columns (first row lost because there is no previous day to compare with):
            #                   price  simple_ret   log_ret
            # 2010-01-05  1136.520020    0.003116  0.003111
            # 2010-01-06  1137.140015    0.000546  0.000545
            # ...
            # 2026-11-06  7376.859863    0.003145  0.003140

            store.save_frame(cfg, "prices", name, prices)       # prints nothing; writes file data\processed\prices\SP500.csv
            store.save_frame(cfg, "returns", name, returns)     # prints nothing; writes file data\processed\returns\SP500.csv
            
            row.update(status="ok", rows=len(returns), first=returns.index.min().date(), last=returns.index.max().date())
            #row = {'asset': 'SP500', 'symbol': '^GSPC', 'source': 'yahoo', 'built_on': '2026-11-08', 'checks': 'ok', 'status': 'ok', 'rows': 4237, 'first': date(2010, 1, 5), 'last': date(2026, 11, 6)}
            
            print(f"[OK]   {name:12s} {source_name:9s} {len(returns):5d} rows  " f"{row['first']} -> {row['last']}  ({report.summary()})")
            # PRINTED TO TERMINAL (one line per asset, all 6 times):
            # [OK]   SP500        yahoo      4237 rows  2010-01-05 -> 2026-11-06  (ok)
            # [OK]   SPY          yahoo      4237 rows  2010-01-05 -> 2026-11-06  (ok)
            # [OK]   AAPL         yahoo      4237 rows  2010-01-05 -> 2026-11-06  (ok)
            # [OK]   MSFT         yahoo      4237 rows  2010-01-05 -> 2026-11-06  (ok)
            # [OK]   VIX          yahoo      4239 rows  2010-01-05 -> 2026-11-06  (51 daily moves > 25%)
            # [OK]   EUROSTOXX50  yahoo      4226 rows  2010-01-05 -> 2026-11-06  (ok)
        
        except Exception as exc:                           # does NOT run (no error occurred)
            if strict:                                     # does NOT run
                raise                                      # does NOT run
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")   # does NOT run
            print(f"[FAIL] {name:12s} {source_name:9s} {row['error']}")         # does NOT run
        
        log.append(row)                                    #log = [SP500's row] (at the end of the loop: 6 rows)

    log_df = pd.DataFrame(log)                             #log_df = table of 6 rows x 9 columns:
    #          asset     symbol source    built_on                checks status  rows       first        last
    # 0        SP500      ^GSPC  yahoo  2026-11-08                    ok     ok  4237  2010-01-05  2026-11-06
    # 1          SPY        SPY  yahoo  2026-11-08                    ok     ok  4237  2010-01-05  2026-11-06
    # 2         AAPL       AAPL  yahoo  2026-11-08                    ok     ok  4237  2010-01-05  2026-11-06
    # 3         MSFT       MSFT  yahoo  2026-11-08                    ok     ok  4237  2010-01-05  2026-11-06
    # 4          VIX       ^VIX  yahoo  2026-11-08  51 daily moves > 25%     ok  4239  2010-01-05  2026-11-06
    # 5  EUROSTOXX50  ^STOXX50E  yahoo  2026-11-08                    ok     ok  4226  2010-01-05  2026-11-06
    
    log_path = project_path(cfg, "processed") / "build_log.csv"     # log_path = C:\Users\ASUS\Desktop\Stochastic-Finance-Research\data\processed\build_log.csv
    log_path.parent.mkdir(parents=True, exist_ok=True)              # no output (create the folder if it does not exist)
    log_df.to_csv(log_path, index=False)                            # prints nothing; writes the log_df table to build_log.csv
    return log_df                                                   #return = the log_df table above, to the run() function in stages/data.py
