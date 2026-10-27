"""Conformal recalibration of the backtest quantiles (post-processing of backtest_detail.csv).

For model m, horizon h and origin t, the calibration set holds the same model's earlier
forecasts whose outcome is known at t (origin o with o + h <= t). Each quantile is shifted
by a conformal quantile of the past errors e = y - q_tau (split conformal / conformalised
quantile regression, Romano, Patterson & Candes 2019):

    q'_tau(t) = q_tau(t) + e_(k),   k = ceil((n + 1) tau) for tau > 0.5 (capped at n),
                                    k = floor((n + 1) tau) for tau < 0.5 (at least 1),
                                    the empirical median of e for tau = 0.5

where e_(k) is the k-th smallest of the n calibration errors. Variants: 'expanding'
(all past errors) and 'rolling 24' (the last 24), and the symmetric CQR interval form
that only rescales the bands around an unchanged median (see VARIANTS). A forecast is recalibrated only when
n >= MIN_CAL; models are compared on the common window of origins where every horizon
has MIN_CAL calibration points. Overlapping h-month errors are not exchangeable, so the
finite-sample guarantee is only approximate.

Usage:
    python conformal_rebar.py [--run backtest_v2]
Writes conformal_detail.csv, table_conformal_summary.csv, table_conformal_dm.csv to the run folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_rebar as BR
import rebar_models as RM

MIN_CAL = 12
# name -> (calibration window, method). 'quantile': each quantile shifted by its own
# conformal error quantile (also moves the location). 'interval': CQR interval form; each
# pair (tau, 1 - tau) widened or narrowed symmetrically by the conformal quantile of
# E = max(q_tau - y, y - q_{1-tau}) at level 1 - 2 tau; the median is left unchanged.
VARIANTS = {'expanding': (None, 'quantile'), 'rolling 24': (24, 'quantile'),
            'symmetric, expanding': (None, 'interval'), 'symmetric, rolling 24': (24, 'interval')}
QCOLS = [f'q{int(round(100 * t)):02d}' for t in RM.TAUS]


def conformal_shift(errors: np.ndarray) -> np.ndarray:
    """Shift per quantile from calibration errors (n, len(TAUS))."""
    n = len(errors)
    e = np.sort(errors, axis=0)
    out = np.empty(len(RM.TAUS))
    for j, t in enumerate(RM.TAUS):
        if t > 0.5:
            k = min(int(np.ceil((n + 1) * t)), n)
        elif t < 0.5:
            k = max(int(np.floor((n + 1) * t)), 1)
        else:
            out[j] = np.median(e[:, j]); continue
        out[j] = e[k - 1, j]
    return out


def interval_adjust(q: np.ndarray, q_cal: np.ndarray, y_cal: np.ndarray) -> np.ndarray:
    """CQR interval form for every symmetric pair of TAUS; q (len(TAUS),), q_cal (n, len(TAUS))."""
    n, m = q_cal.shape
    out = q.copy()
    for j in range(m // 2):
        J = m - 1 - j
        e = np.sort(np.maximum(q_cal[:, j] - y_cal, y_cal - q_cal[:, J]))
        k = min(int(np.ceil((n + 1) * (1 - 2 * RM.TAUS[j]))), n)
        out[j] -= e[k - 1]
        out[J] += e[k - 1]
    return out


def recalibrate(g: pd.DataFrame, h: int, window: int | None, method: str = 'quantile') -> pd.DataFrame:
    """g: one model x horizon x series, sorted by origin."""
    d = pd.to_datetime(g['origin'])
    mi = (d.dt.year * 12 + d.dt.month).to_numpy()
    q = g[QCOLS].to_numpy(float)
    err = g['y_log'].to_numpy(float)[:, None] - q
    out_q, n_cal = q.copy(), np.zeros(len(g), int)
    for i, t in enumerate(mi):
        known = np.flatnonzero(mi + h <= t)  # outcome realised by origin t
        if window:
            known = known[-window:]
        n_cal[i] = len(known)
        if len(known) >= MIN_CAL:
            if method == 'quantile':
                out_q[i] = np.sort(q[i] + conformal_shift(err[known]))
            else:
                out_q[i] = np.sort(interval_adjust(q[i], q[known], g['y_log'].to_numpy(float)[known]))
    r = g.copy()
    r[QCOLS] = out_q
    r['n_cal'] = n_cal
    return r


def rescore(df: pd.DataFrame) -> pd.DataFrame:
    scores = [BR.score(row[QCOLS].to_numpy(float), row['y_log']) for _, row in df.iterrows()]
    keep = df[['series', 'origin', 'h', 'model', 'variant', 'n_cal']].reset_index(drop=True)
    return pd.concat([keep, pd.DataFrame(scores)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='backtest_v2')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    out = Path(__file__).resolve().parent / 'outputs' / args.run
    det = pd.read_csv(out / 'backtest_detail.csv')
    det = det[det[QCOLS[0]].notna()].sort_values(['series', 'model', 'h', 'origin'])

    parts = [det.assign(variant='raw', n_cal=0)]
    for name, (window, method) in VARIANTS.items():
        for (s, m, h), g in det.groupby(['series', 'model', 'h']):
            parts.append(recalibrate(g, h, window, method).assign(variant=name))
    allq = pd.concat(parts, ignore_index=True)
    sc = rescore(allq)

    # common window: origins where every horizon has MIN_CAL calibration points
    first = max(pd.Timestamp(det['origin'].min()) + pd.DateOffset(months=h + MIN_CAL - 1) for h in BR.HORIZONS)
    sc = sc[pd.to_datetime(sc['origin']) >= first]
    sc.to_csv(out / 'conformal_detail.csv', index=False)
    print(f'Common evaluation window: origins {first:%Y-%m} .. {sc["origin"].max()}')

    summ = (sc.groupby(['series', 'model', 'variant', 'h'])
            .agg(n=('crps_pct', 'size'), cover_P5_P95=('in_P5_P95', 'mean'), cover_le_P80=('le_P80', 'mean'),
                 cover_le_P95=('le_P95', 'mean'), pinball95_pct=('pinball95_pct', 'mean'), crps_pct=('crps_pct', 'mean'))
            .reset_index())
    summ.to_csv(out / 'table_conformal_summary.csv', index=False)

    rows = []
    for (s, h), g in sc.groupby(['series', 'h']):
        w = g.pivot_table(index='origin', columns=['model', 'variant'], values='crps_pct')
        rw = w[('Random walk', 'raw')]
        for (m, v) in w.columns:
            for bench_name, bench in [('same model, raw', w[(m, 'raw')]), ('Random walk, raw', rw)]:
                if v == 'raw' and bench_name == 'same model, raw':
                    continue
                pair = pd.concat([w[(m, v)], bench], axis=1).dropna()
                stat, p = BR.dm_test((pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy(), h)
                rows.append({'series': s, 'h': h, 'model': m, 'variant': v, 'benchmark': bench_name, 'n': len(pair),
                             'skill': 1 - pair.iloc[:, 0].mean() / pair.iloc[:, 1].mean(), 'dm_stat': stat, 'p_value': p})
    dm = pd.DataFrame(rows)
    dm.to_csv(out / 'table_conformal_dm.csv', index=False)

    for s in summ['series'].unique():
        t = summ[summ['series'] == s]
        for metric in ['crps_pct', 'cover_P5_P95', 'cover_le_P95']:
            print(f'\n## {s}: {metric}')
            print(t.pivot_table(index='model', columns=['variant', 'h'], values=metric).round(3).to_string())


if __name__ == '__main__':
    main()
