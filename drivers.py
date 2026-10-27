"""Loaders for the global steel drivers (files in ./data_external, see SOURCES.md).

All series are monthly, indexed by the first day of the month:
  iron_ore_wb   World Bank Pink Sheet, 'Iron ore, cfr spot' (USD/dmtu)
  coal_wb       World Bank Pink Sheet, 'Coal, Australian' (USD/mt)
  brent_wb      World Bank Pink Sheet, 'Crude oil, Brent' (USD/bbl)
  iron_ore_imf  FRED PIORECRUSDM, IMF global price of iron ore (USD/mt; annual contract
                price repeated each month before Dec 2008)
  steel_ppi     FRED WPU1017, US PPI steel mill products (1982 = 100, NSA)
  scrap_ppi     FRED WPU1012, US PPI iron and steel scrap (1982 = 100, NSA)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import data as D

EXTERNAL_DIR = Path(__file__).resolve().parent / 'data_external'
PINK_SHEET = 'CMO-Historical-Data-Monthly.xlsx'
PINK_COLUMNS = {'iron_ore_wb': 'Iron ore, cfr spot', 'coal_wb': 'Coal, Australian',
                'brent_wb': 'Crude oil, Brent'}
FRED_SERIES = {'iron_ore_imf': 'PIORECRUSDM', 'steel_ppi': 'WPU1017', 'scrap_ppi': 'WPU1012'}


def load_pink_sheet(path: Path = EXTERNAL_DIR / PINK_SHEET, columns: dict = PINK_COLUMNS) -> pd.DataFrame:
    """Selected columns of sheet 'Monthly Prices' (header row = the one holding the
    commodity names; dates written as '2026M09')."""
    raw = pd.read_excel(path, sheet_name='Monthly Prices', header=None)
    header = raw.index[raw.apply(lambda r: r.astype(str).str.strip().isin(columns.values()).any(), axis=1)][0]
    names = raw.loc[header].astype(str).str.strip()
    out = {}
    for key, name in columns.items():
        col = names.index[names == name]
        if len(col) != 1:
            raise ValueError(f'Pink Sheet column {name!r} found {len(col)} times')
        out[key] = col[0]
    rows = raw[raw[0].astype(str).str.match(r'^\d{4}M\d{2}$')]
    idx = pd.to_datetime(rows[0].str.replace('M', '-') + '-01')
    df = pd.DataFrame({k: pd.to_numeric(rows[c], errors='coerce').to_numpy() for k, c in out.items()}, index=idx)
    updated = next((str(v) for v in raw[0].astype(str) if str(v).startswith('Updated on')), '')
    df.attrs['updated'] = re.sub(r'^Updated on\s*', '', updated)
    return df


def load_fred(series_id: str, directory: Path = EXTERNAL_DIR) -> pd.Series:
    df = pd.read_csv(directory / f'FRED_{series_id}.csv')
    return pd.Series(pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(),
                     index=pd.to_datetime(df.iloc[:, 0]), name=series_id).dropna()


def build_drivers(months: pd.DatetimeIndex, data_dir: Path = D.DEFAULT_DATA_DIR,
                  directory: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Global drivers plus THB/USD (BOT) and Bangkok retail diesel (EPPO P09), reindexed to
    `months`. Missing months (publication lags at the end of the sample) stay NaN."""
    drv = load_pink_sheet(directory / PINK_SHEET).reindex(months)
    for key, sid in FRED_SERIES.items():
        drv[key] = load_fred(sid, directory).reindex(months)
    drv['fx'] = D.load_fx(data_dir).reindex(months)
    drv['diesel_thb'] = D.load_diesel_retail(data_dir).reindex(months)
    drv['iron_ore_thb'] = drv['iron_ore_wb'] * drv['fx']
    return drv
