"""Rebar item selection and code splicing for the TPSO price panel.

TPSO lists steel reinforcing bars in category 04 ('วัสดุชิ้นส่วนหน้าตัดรูปต่างๆ'). The item
codes of the same product change over time (e.g. SD40 bars were re-coded as
'Tempcort' in Feb 2018, 'Tempcore' in Jun 2018 and 'มอก. 24-2559' in Apr 2026), so no
single code covers Jan 2017 - Sep 2026. This module parses grade, diameter and
production process from the item text and links the codes of one product into a
continuous monthly series.

Splicing rule (documented in README.md):
  1. Product = (grade, diameter). Grade is SR24, SD30, SD40 or SD50, read from the item
     text; diameter in mm from 'ศก. d มม.' or 'ขนาดเส้นผ่านศูนย์กลาง d มม.'.
     Welded mesh ('ตะแกรง') is excluded.
  2. Main line: the codes of a product are ordered by first listing month. For SD40 the
     main line is the series TPSO lists without interruption: the plain SD40 bar to
     Jan 2018, then the quenched-and-tempered (Tempcore/'Tempcort') bar from Feb 2018.
     The plain SD40 bar that TPSO lists again from Jan 2024 is kept as a secondary
     series ('SD40 plain'), not spliced into the main line.
  3. Consecutive codes of the main line never overlap in time, so the levels are
     chained directly: the price change in the month a new code starts is kept as an
     ordinary monthly change.
  4. Each junction change is checked against a benchmark that did not change code in
     that month: SR24 bars of the same diameter class (Feb and Jun 2018) or, when every
     rebar code changed (Apr 2026), the median change of the other category-04 steel
     items whose codes continue. A junction is flagged when it differs from the
     benchmark by more than JUNCTION_TOL (log points).
  5. A gap inside a product (months with no code listed) is left as missing; it is
     never interpolated. A product enters the audit only if it covers at least
     MIN_MONTHS months.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REBAR_CAT = '04'
REBAR_PATTERN = 'เหล็กเส้น|ข้ออ้อย'
EXCLUDE_PATTERN = 'ตะแกรง'
JUNCTION_TOL = 0.02
MIN_MONTHS = 60
# Main products of the study (grade, diameters in mm)
MAIN_PRODUCTS = {'SD40': [12, 16, 20, 25, 28, 32], 'SD50': [12, 16, 20, 25, 28, 32],
                 'SR24': [6, 9, 12, 15, 19, 25]}

_GRADE = re.compile(r'S([RD])\.?\s*(\d{2})')
_DIAM = re.compile(r'(?:ศก\.|ขนาดเส้นผ่านศูนย์กลาง)\s*(\d+)\s*มม')


def select_rebar(panel: pd.DataFrame) -> pd.DataFrame:
    """Rebar items of category 04 with parsed grade, diameter and process."""
    p = panel[(panel['cat'] == REBAR_CAT)
              & panel['item'].str.contains(REBAR_PATTERN, na=False)
              & ~panel['item'].str.contains(EXCLUDE_PATTERN, na=False)].copy()
    g = p['item'].str.extract(_GRADE)
    p['grade'] = 'S' + g[0] + g[1]
    p['diam'] = pd.to_numeric(p['item'].str.extract(_DIAM)[0])
    p['tempcore'] = p['item'].str.contains('Tempco', case=False, na=False)
    if p[['grade', 'diam']].isna().any().any():
        bad = p.loc[p[['grade', 'diam']].isna().any(axis=1), 'item'].unique()
        raise ValueError(f'Cannot parse grade/diameter of: {bad}')
    p['diam'] = p['diam'].astype(int)
    return p


def code_table(rebar: pd.DataFrame) -> pd.DataFrame:
    """One row per TPSO code: product, process, listing period and months listed."""
    t = (rebar.sort_values('date').groupby('code')
         .agg(grade=('grade', 'first'), diam=('diam', 'first'), tempcore=('tempcore', 'first'),
              item=('item', 'last'), unit=('unit', 'last'), first=('date', 'min'), last=('date', 'max'),
              months=('date', 'size')))
    span = ((t['last'].dt.year - t['first'].dt.year) * 12 + t['last'].dt.month - t['first'].dt.month + 1)
    t['gaps'] = span - t['months']
    return t.sort_values(['grade', 'diam', 'first'])


def split_relisted(rebar: pd.DataFrame, min_gap: int = 12) -> pd.DataFrame:
    """A code listed, dropped for >= `min_gap` months and listed again is treated as two
    codes (suffix '#2' on the second listing). The plain SD40 codes 0401020200200000-
    0401020200700000 are listed in 2017-Jan 2018 and again from Jan 2024."""
    r = rebar.sort_values(['code', 'date']).copy()
    mi = r['date'].dt.year * 12 + r['date'].dt.month
    new_spell = (mi - mi.groupby(r['code']).shift(1)) > min_gap
    spell = new_spell.groupby(r['code']).cumsum()
    r.loc[spell > 0, 'code'] = r.loc[spell > 0, 'code'] + '#' + (spell[spell > 0] + 1).astype(str)
    return r


def splice(rebar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Continuous series per (line, diameter).

    Returns (wide prices: index month, columns '<line> <d>mm'), the code map (which
    code supplies each series in each period) and the per-code table."""
    r = split_relisted(rebar)
    codes = code_table(r)
    codes['line'] = codes['grade']
    # rule 2: the plain SD40 bar listed again from Jan 2024 (and its Apr-2026 re-coding)
    plain_relisted = (codes['grade'] == 'SD40') & ~codes['tempcore'] & (codes['first'] >= '2024-01-01')
    codes.loc[plain_relisted, 'line'] = 'SD40 plain'
    r = r.join(codes['line'], on='code')
    r['series'] = r['line'] + ' ' + r['diam'].astype(str) + 'mm'

    overlap = r.groupby(['series', 'date'])['code'].nunique()
    if (overlap > 1).any():
        raise ValueError(f'Codes overlap within a series: {overlap[overlap > 1].index.tolist()[:5]}')
    wide = r.pivot(index='date', columns='series', values='price').sort_index()
    wide = wide.reindex(pd.date_range(r['date'].min(), r['date'].max(), freq='MS'))
    cmap = (r.groupby(['series', 'code'])
            .agg(line=('line', 'first'), grade=('grade', 'first'), diam=('diam', 'first'),
                 tempcore=('tempcore', 'first'), first=('date', 'min'), last=('date', 'max'),
                 months=('date', 'size'), item=('item', 'last'))
            .reset_index().sort_values(['series', 'first']))
    cmap['link'] = cmap.groupby('series').cumcount()
    return wide, cmap, codes


def junction_checks(wide: pd.DataFrame, cmap: pd.DataFrame, panel: pd.DataFrame,
                    rebar_codes: set) -> pd.DataFrame:
    """Rule 4: the log change in each month a new code starts, against benchmarks whose
    codes do not change in that month."""
    c4 = panel[panel['cat'] == REBAR_CAT]
    other = c4[~c4['code'].isin(rebar_codes)].pivot(index='date', columns='code', values='price')
    other_dl = np.log(other).diff()
    dl = np.log(wide).diff()
    starts = cmap[cmap['link'] > 0]
    rows = []
    for _, s in starts.iterrows():
        d = s['first']
        prev = d - pd.offsets.MonthBegin(1)
        if prev not in wide.index or pd.isna(wide.at[prev, s['series']]):
            rows.append({'series': s['series'], 'month': d, 'code_in': s['code'], 'junction_dlog': np.nan,
                         'benchmark': 'none (gap before new code)', 'benchmark_dlog': np.nan, 'flag': False})
            continue
        j = dl.at[d, s['series']]
        # benchmark 1: SR24 series whose code continues through month d
        sr = cmap[(cmap['line'] == 'SR24') & (cmap['first'] < d) & (cmap['last'] >= d)]['series'].unique()
        if len(sr) and s['line'] != 'SR24':
            b, name = float(dl.loc[d, sr].median()), f'SR24 median ({len(sr)} sizes)'
        else:
            x = other_dl.loc[d].dropna() if d in other_dl.index else pd.Series(dtype=float)
            b, name = float(x.median()), f'other cat-04 steel items, median ({len(x)} items)'
        rows.append({'series': s['series'], 'month': d, 'code_in': s['code'], 'junction_dlog': j,
                     'benchmark': name, 'benchmark_dlog': b, 'flag': bool(abs(j - b) > JUNCTION_TOL)})
    return pd.DataFrame(rows)
