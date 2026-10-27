"""Data loaders for the ASEJ revision pipeline.

Every loader reads the raw files as downloaded (TPSO monthly construction
material price reports, BOT monthly exchange rates, BOT MPC policy-rate
decisions, monthly crude-oil prices) and returns monthly data indexed by the
first day of each month.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

THAI_MONTH_ABBR = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
                   'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.']
THAI_MONTH_FULL = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                   'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']

# TPSO item codes of CPAC ready-mixed concrete (cube strength, ksc)
GRADE_CODES = {
    180: '0101010108100000', 210: '0101010108200000', 240: '0101010108300000',
    280: '0101010108400000', 300: '0101010107200000', 320: '0101010108500000',
    350: '0101010108600000', 380: '0101010108700000', 400: '0101010108800000',
}
# Concrete-family items used to train the revision model: ready-mixed concrete (01),
# concrete blocks (02), precast/prestressed concrete elements (03) and concrete
# aggregates (16). Steel and other materials are left for a separate study.
CONCRETE_CATS = ['01', '02', '03', '16']


def select_concrete_family(panel: pd.DataFrame) -> pd.DataFrame:
    """Items in CONCRETE_CATS that are concrete products (or aggregates in cat 16);
    drops e.g. clay and glass bricks listed under masonry."""
    in_cat = panel['cat'].isin(CONCRETE_CATS)
    concrete = panel['item'].str.contains('คอนกรีต', na=False) | (panel['cat'] == '16')
    return panel[in_cat & concrete]

_HERE = Path(__file__).resolve().parent
if (_HERE / 'data').is_dir():  # public repository layout: all raw files in ./data
    DEFAULT_DATA_DIR = _HERE / 'data'
    DEFAULT_CPI_NATIONAL = DEFAULT_DATA_DIR / 'CPIG.xlsx'
else:  # authors' working folders
    DEFAULT_DATA_DIR = _HERE.parent / 'Contraction price' / 'รวมทุกปี'
    DEFAULT_CPI_NATIONAL = _HERE.parent / 'ดัชนีราคาผู้บริโภคเงินเฟ้อ (CPI)' / 'CPIG.xlsx'
DEFAULT_CPI_BKK = DEFAULT_DATA_DIR / 'CPIG_Bangkok.xlsx'  # Bangkok and vicinity, to Sep 2026 (main deflator)


def _be_year(y: int) -> int:
    """Buddhist-era year (2-digit 60 or 4-digit 2560) to Gregorian."""
    if y < 100:
        y += 2500
    return y - 543


def _report_month(fname: str) -> pd.Timestamp:
    """'Price_Report 3.69.xlsx' -> 2026-03-01."""
    m = re.search(r'(\d{1,2})\.(\d{2})\.xlsx$', os.path.basename(fname))
    return pd.Timestamp(year=_be_year(int(m.group(2))), month=int(m.group(1)), day=1)


def _thai_header_month(text: str) -> pd.Timestamp | None:
    """'มกราคม 2560' -> 2017-01-01."""
    for i, name in enumerate(THAI_MONTH_FULL):
        if name in str(text):
            y = re.findall(r'\d{4}', str(text))
            if y:
                return pd.Timestamp(year=_be_year(int(y[-1])), month=i + 1, day=1)
    return None


def load_price_panel(data_dir: Path, cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long panel (date, cat, catname, code, item, unit, price) of every TPSO item.

    Returns the panel and a per-file audit table (file month vs. header month,
    number of rows, duplicate codes)."""
    files = sorted(glob.glob(str(Path(data_dir) / 'Price_Report*.xlsx')))
    newest_input = max([os.path.getmtime(f) for f in files] + [os.path.getmtime(__file__)])
    if cache is not None and cache.exists() and cache.stat().st_mtime > newest_input:
        return pd.read_pickle(cache), pd.read_pickle(cache.with_suffix('.audit.pkl'))

    frames, audit = [], []
    for f in files:
        raw = pd.read_excel(f, header=None, dtype=str)
        month = _report_month(f)
        header_month = _thai_header_month(' '.join(raw.iloc[0].dropna().astype(str)))
        df = raw.iloc[5:, :6].copy()
        df.columns = ['cat', 'catname', 'code', 'item', 'unit', 'price']
        df['price'] = pd.to_numeric(df['price'].str.replace(',', '', regex=False), errors='coerce')
        df['code'] = df['code'].str.strip()
        df = df.dropna(subset=['code', 'price'])
        n_dup = int(df.duplicated('code').sum())
        df = df.drop_duplicates('code')
        # The period printed inside the report is authoritative: some 2017 file
        # names do not match their contents (e.g. 'Price_Report 2.60' holds April).
        df['date'] = header_month if header_month is not None else month
        frames.append(df)
        audit.append({'file': os.path.basename(f), 'month': month, 'header_month': header_month,
                      'header_matches': header_month == month, 'items': len(df), 'duplicate_codes': n_dup})
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(['date', 'code']).any():
        raise ValueError('Two reports resolve to the same month; check the audit table.')
    panel['cat'] = panel['cat'].str.strip()
    audit = pd.DataFrame(audit)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cache)
        audit.to_pickle(cache.with_suffix('.audit.pkl'))
    return panel, audit


def load_fx(data_dir: Path) -> pd.Series:
    """Monthly average THB/USD mid rate (mean of BOT buying-TT and selling rates)."""
    rows = []
    for f in glob.glob(str(Path(data_dir) / 'Exchange_Rate*.xlsx')):
        raw = pd.read_excel(f, header=None)
        for _, r in raw.iterrows():
            parts = str(r[0]).split()
            if len(parts) == 2 and parts[0] in THAI_MONTH_ABBR and parts[1].isdigit():
                d = pd.Timestamp(year=_be_year(int(parts[1])), month=THAI_MONTH_ABBR.index(parts[0]) + 1, day=1)
                rows.append((d, (float(r[2]) + float(r[3])) / 2))
    s = pd.Series(dict(rows)).sort_index()
    return s[~s.index.duplicated()].rename('fx')


def load_oil_usd(data_dir: Path) -> pd.Series:
    """Monthly crude-oil closing price (USD/bbl). January 2017 is missing from the
    file, so its value is taken as the February 2017 opening price."""
    raw = pd.read_excel(Path(data_dir) / 'Oil_Price.xlsx', header=None)
    rows = {}
    opens = {}
    for _, r in raw.iterrows():
        if isinstance(r[0], (pd.Timestamp,)) or re.match(r'\d{4}-\d{2}-\d{2}', str(r[0])):
            d = pd.Timestamp(r[0]).replace(day=1)
            rows[d] = float(r[1])
            opens[d] = float(r[2])
    s = pd.Series(rows).sort_index()
    first = s.index.min()
    prev = first - pd.offsets.MonthBegin(1)
    if prev == pd.Timestamp('2017-01-01'):
        s.loc[prev] = opens[first]
    return s.sort_index().rename('oil_usd')


def load_policy_rate(data_dir: Path, months: pd.DatetimeIndex) -> pd.Series:
    """BOT policy rate in force at the end of each month, from the MPC decision
    tables (all Interest*.xlsx files; later downloads extend earlier ones)."""
    dec = []
    for f in glob.glob(str(Path(data_dir) / 'Interest*.xlsx')):
        raw = pd.read_excel(f, header=None)
        for _, r in raw.iterrows():
            d = pd.to_datetime(r[2], errors='coerce')
            rate = pd.to_numeric(r[7], errors='coerce')
            if pd.notna(d) and pd.notna(rate):
                dec.append((d, float(rate)))
    dec = pd.DataFrame(dec, columns=['d', 'rate'])
    if dec.groupby('d')['rate'].nunique().max() > 1:
        raise ValueError('MPC decision files disagree on a policy rate; check Interest*.xlsx')
    dec = dec.drop_duplicates('d').sort_values('d')
    month_end = pd.DataFrame({'month': months, 'd': months + pd.offsets.MonthEnd(0)})
    out = pd.merge_asof(month_end.sort_values('d'), dec, on='d', direction='backward')
    return out.set_index('month')['rate'].rename('rate')


BRENT_FRED = 'Brent_FRED_monthly.csv'


def load_crude_usd(data_dir: Path) -> tuple[pd.Series, str]:
    """Crude oil in USD/bbl for the robustness driver: FRED Brent monthly average
    (DCOILBRENTEU) when that file is present, otherwise month-end closes from Oil_Price.xlsx."""
    path = Path(data_dir) / BRENT_FRED
    if path.exists():
        df = pd.read_csv(path)
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(),
                      index=pd.to_datetime(df.iloc[:, 0])).dropna()
        return s.rename('oil_usd'), 'Brent, monthly average (FRED DCOILBRENTEU)'
    return load_oil_usd(data_dir), 'crude oil, month-end close (Oil_Price.xlsx)'


_MONTHS_EN = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
DIESEL_COL = 8  # P09 column of HSD B7 (named HSD before 1 Oct 2020)


def load_diesel_retail(data_dir: Path) -> pd.Series:
    """Monthly weighted-average retail price of diesel (HSD B7) in Bangkok, THB/litre,
    from EPPO table P09 'Retail price of petroleum products in Bangkok'.

    The sheet lists each year as a block: a 'YYYY (MIN)' row with annual values, then
    '-MON (MIN)', '(WT.AVG)', '(MAX)' rows per month. Only the monthly WT.AVG rows are
    kept."""
    files = glob.glob(str(Path(data_dir) / '*P09*.xls*'))
    if not files:
        raise FileNotFoundError(f'EPPO table P09 not found in {data_dir}')
    raw = pd.read_excel(files[0], header=None)
    header = ' '.join(raw.iloc[3:7, DIESEL_COL].dropna().astype(str))
    if 'HSD' not in header or 'B7' not in header:
        raise ValueError(f'P09 layout changed: column {DIESEL_COL} header reads {header!r}')
    out, year, month = {}, None, None
    for _, r in raw.iterrows():
        label = str(r[0]).strip()
        y = re.match(r'^(\d{4})\s*\(MIN\)', label)
        m = re.match(r'^-\s*([A-Za-z]{3})\s*\(MIN\)', label)
        if y:
            year, month = int(y.group(1)), None
        elif m and year is not None:
            month = _MONTHS_EN.index(m.group(1).upper()) + 1  # 2023 uses 'Dec'
        elif label.startswith('(WT.AVG)') and month is not None:
            v = pd.to_numeric(r[DIESEL_COL], errors='coerce')
            if pd.notna(v):
                out[pd.Timestamp(year=year, month=month, day=1)] = float(v)
            month = None
    return pd.Series(out).sort_index().rename('diesel_thb')


ENERGY_DRIVERS = ('diesel', 'crude')


def build_macro(data_dir: Path, months: pd.DatetimeIndex, energy: str = 'diesel') -> pd.DataFrame:
    """Monthly macro table: energy price in THB (the model's cost driver), THB/USD and
    policy rate. `energy='diesel'`: Bangkok retail diesel (EPPO P09, THB/litre);
    `energy='crude'`: crude oil (USD/bbl) x THB/USD."""
    if energy not in ENERGY_DRIVERS:
        raise ValueError(f'energy must be one of {ENERGY_DRIVERS}')
    macro = pd.DataFrame(index=months)
    macro['fx'] = load_fx(data_dir).reindex(months)
    macro['rate'] = load_policy_rate(data_dir, months)
    if energy == 'diesel':
        macro['energy'] = load_diesel_retail(data_dir).reindex(months)
        source = 'Bangkok retail diesel HSD B7, monthly weighted average (EPPO P09)'
    else:
        oil, source = load_crude_usd(data_dir)
        macro['oil_usd'] = oil.reindex(months)
        macro['energy'] = macro['oil_usd'] * macro['fx']
    macro.attrs['energy'] = energy
    macro.attrs['energy_source'] = source
    missing = macro.isna().sum()
    if missing.any():
        gaps = {c: list(macro.index[macro[c].isna()].strftime('%Y-%m')) for c in missing[missing > 0].index}
        raise ValueError(f'Missing macro values: {gaps}')
    return macro


def load_cpi_index(path: Path) -> pd.Series:
    """Monthly headline CPI ('รวมทุกรายการ', base 2566 = 100) from a TPSO CPI workbook.

    The monthly index is on sheet 'Index (ของเดือน)'; the first sheet only lists the
    CPI categories."""
    raw = pd.read_excel(path, sheet_name='Index (ของเดือน)', header=None)
    header = raw.index[raw[0].astype(str).str.strip() == 'หมวด'][0]
    month_cols = [c for c in raw.columns if str(raw.at[header, c]).strip() in THAI_MONTH_ABBR]
    rows = raw[raw[0].astype(str).str.strip() == 'รวมทุกรายการ']
    bases = set(rows[1].astype(str).str.strip())
    if bases != {'2566'}:
        raise ValueError(f'Mixed CPI base years in {path}: {bases}')
    out = {}
    for _, r in rows.iterrows():
        year = _be_year(int(r[2]))
        for c in month_cols:
            v = pd.to_numeric(r[c], errors='coerce')
            if pd.notna(v):
                month = THAI_MONTH_ABBR.index(str(raw.at[header, c]).strip()) + 1
                out[pd.Timestamp(year=year, month=month, day=1)] = float(v)
    return pd.Series(out).sort_index().rename('cpi')
