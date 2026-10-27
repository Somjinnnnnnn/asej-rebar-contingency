"""Coverage of the global drivers over the rebar sample and a first, descriptive look at
lead-lag correlations of monthly log changes (driver at t - k vs. rebar at t).
Not a model result; it only shows which lags to consider as features.

Writes outputs/audit_2026-10/table_driver_coverage.csv, table_driver_leadlag.csv and
data_drivers.csv."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import drivers as G

warnings.filterwarnings('ignore', message='Workbook contains no default style')
OUT = Path(__file__).resolve().parent / 'outputs' / 'audit_2026-10'
TARGETS = ['SD40 12mm', 'SD50 12mm', 'SR24 12mm']
LAGS = range(0, 7)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 200)
    rebar = pd.read_csv(OUT / 'data_rebar_series.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    months = pd.date_range('2015-01-01', rebar.index.max(), freq='MS')
    drv = G.build_drivers(months)
    drv.to_csv(OUT / 'data_drivers.csv')
    print(f'Pink Sheet updated: {G.load_pink_sheet().attrs["updated"]}')

    sample = drv.loc[rebar.index.min():]
    cov = pd.DataFrame({'first': sample.apply(lambda s: s.first_valid_index()),
                        'last': sample.apply(lambda s: s.last_valid_index()),
                        'missing_in_sample': sample.isna().sum()})
    cov.to_csv(OUT / 'table_driver_coverage.csv')
    print(cov.to_string())

    dl = np.log(drv).diff()
    rl = np.log(rebar[TARGETS]).diff()
    rows = []
    for d in dl.columns:
        for k in LAGS:
            x = dl[d].shift(k)
            for t in TARGETS:
                pair = pd.concat([x, rl[t]], axis=1).dropna()
                rows.append({'driver': d, 'lag': k, 'target': t, 'n': len(pair),
                             'corr': pair.iloc[:, 0].corr(pair.iloc[:, 1])})
    ll = pd.DataFrame(rows)
    ll.to_csv(OUT / 'table_driver_leadlag.csv', index=False)
    print(ll[ll['target'] == 'SD40 12mm'].pivot(index='driver', columns='lag', values='corr').round(2).to_string())


if __name__ == '__main__':
    main()
