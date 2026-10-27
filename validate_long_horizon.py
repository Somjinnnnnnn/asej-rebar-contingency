"""Out-of-sample check of 1-5-year bands on a longer steel price series.

The TPSO rebar series (117 months) cannot validate multi-year bands. As proxy evidence, the
same models are fitted to rolling 120-month windows of the US PPI for steel mill products
(FRED WPU1017, 2000-2026, nominal, 1982 = 100) - the same window length as the TPSO sample -
and their 12/36/60-month bands are scored at monthly origins 2010-01 .. 2021-09. Bands are
nominal by default; with --deflator CPIAUCSL the PPI is deflated by the US CPI (real terms,
as the rebar projection), so the 'zero drift' variant is the real driftless case. Overlapping
60-month outcomes make the effective sample small (about two to three independent 5-year
periods).

Writes outputs/<out-dir>/validation_detail.csv and table_validation_summary.csv.

Usage:
    python validate_long_horizon.py [--paths 2000] [--workers 12]
    python validate_long_horizon.py --deflator CPIAUCSL --out-dir validation_wpu1017_real
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import backtest_rebar as BR
import drivers as G
import models as M
import projection_rebar as P
import rebar_models as RM

WINDOW = 120
HORIZONS = [12, 36, 60]
FIRST_ORIGIN, LAST_ORIGIN = '2010-01-01', '2021-09-01'


def fit_all(lp: np.ndarray, horizon: int, n: int, key: str) -> dict:
    r = np.diff(lp, prepend=np.nan)
    price = np.exp(lp)
    rng = lambda name: B.rng_for('val', name, key)  # noqa: E731
    return {
        'Random walk (driftless)': RM.rw_paths(r, horizon, n, rng('rw'))[0],
        'GBM (fitted drift)': RM.gbm_paths(r, horizon, n, rng('gbm'))[0],
        'GARCH(1,1)': M.fit_simulate_garch(r, horizon, n, rng('garch'))[0],
        'AR(1)-GARCH(1,1)': RM.ar_paths(r, horizon, n, int(rng('argarch').integers(2 ** 31)), garch=True)[0],
        'Merton jump-diffusion': M.fit_simulate_merton(r, horizon, n, rng('merton'), price=price)[0],
        'Markov regime-switching': M.fit_simulate_regime(r, horizon, n, rng('ms'), price=price)[0],
        'Block bootstrap (demeaned)': RM.block_bootstrap_paths(r, horizon, n, rng('boot'))[0],
        'Block bootstrap (with drift)': RM.block_bootstrap_paths(r, horizon, n, rng('boot2'), demean=False)[0],
        'Mean-reverting AR(1) level': RM.mean_reverting_paths(lp, horizon, n, rng('ou'))[0],
    }


def one_origin(args):
    origin, lp_full, n = args
    warnings.filterwarnings('ignore')
    hist = lp_full.loc[:origin].iloc[-(WINDOW + 1):]
    sims = fit_all(hist.to_numpy(), max(HORIZONS), n, f'{origin:%Y-%m}')
    rows = []
    for name, fitted in sims.items():
        for drift, paths in [('as fitted', fitted), ('zero drift', P.zero_drift(fitted))]:
            for h in HORIZONS:
                y = lp_full[origin + pd.DateOffset(months=h)] - lp_full[origin]
                q = np.quantile(np.log(paths[h - 1]), RM.TAUS)
                rows.append({'origin': f'{origin:%Y-%m}', 'h': h, 'model': name, 'drift': drift, **BR.score(q, y)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out-dir', default='validation_wpu1017')
    ap.add_argument('--deflator', choices=['none', 'CPIAUCSL'], default='none')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    out = Path(__file__).resolve().parent / 'outputs' / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lp = np.log(G.load_fred('WPU1017'))
    if args.deflator != 'none':
        # months missing in the CPI (2025-10, not published) are filled by log-linear interpolation
        lc = np.log(G.load_fred(args.deflator)).reindex(pd.date_range(lp.index[0], lp.index[-1], freq='MS'))
        filled = list(lc.index[lc.isna()].strftime('%Y-%m'))
        lc = lc.interpolate(limit_area='inside')
        if lc.isna().any():
            raise ValueError(f'{args.deflator} does not cover {lp.index[0]:%Y-%m}..{lp.index[-1]:%Y-%m}')
        lp = (lp - lc + lc.iloc[-1]).rename('WPU1017_real')
        print(f'Deflated by {args.deflator} (prices of {lp.index[-1]:%b %Y}); interpolated CPI months: {filled}')
    origins = pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq='MS')
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for part in ex.map(one_origin, [(o, lp, args.paths) for o in origins]) for r in part]
    det = pd.DataFrame(rows)
    det.to_csv(out / 'validation_detail.csv', index=False)
    summ = (det.groupby(['model', 'drift', 'h'])
            .agg(n=('crps_pct', 'size'), cover_P5_P95=('in_P5_P95', 'mean'), cover_le_P80=('le_P80', 'mean'),
                 cover_le_P95=('le_P95', 'mean'), pinball95_pct=('pinball95_pct', 'mean'), crps_pct=('crps_pct', 'mean'))
            .reset_index())
    summ.to_csv(out / 'table_validation_summary.csv', index=False)
    print(f'WPU1017, rolling {WINDOW}-month windows, origins {origins[0]:%Y-%m}..{origins[-1]:%Y-%m} ({len(origins)})')
    for m in ['crps_pct', 'cover_P5_P95', 'cover_le_P80', 'cover_le_P95']:
        print(f'\n--- {m}')
        print(summ.pivot_table(index='model', columns=['drift', 'h'], values=m).round(2).to_string())


if __name__ == '__main__':
    main()
