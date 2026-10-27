"""Analyses for the second pre-submission review (October 2026).

Parts (CSVs to outputs/revision_2026-10):
  boot_backtest  the headline block bootstrap (mean block 12 months, and the Politis-White block) in the
                 1-12-month rolling-origin backtest: CRPS, skill against the random walk with 90%
                 stationary-bootstrap intervals, coverage, Diebold-Mariano test.
  allowance      historical protection of the 24-month schedule-A allowances at several model
                 percentiles, and the scaling needed for 95% protection; purchase profiles
                 (front-loaded, even, back-loaded); an illustrative price-adjustment clause; estimation
                 uncertainty of the schedule-A expenditure P95.
  dependence     correlation of real rebar changes with Bangkok CPI inflation and THB/USD, and nominal
                 and USD bands from a joint block bootstrap of the three series against independent
                 resampling.
  threshold      distribution of the monthly difference between each 12 mm rebar series and the SR24
                 benchmark in months without a code change, as the basis of the two-log-point flag.

Usage:
    python revision2_analyses.py [--parts boot_backtest allowance dependence threshold]
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

import backtest as B
import backtest_rebar as BR
import data as D
import rebar_data as R
import rebar_models as RM
import revision_analyses as RA

warnings.filterwarnings('ignore', message='Workbook contains no default style')
OUT = RA.OUT_ROOT / 'revision_2026-10'
SCHEDULES = {'even': {3 * k: 75 for k in range(1, 9)},
             'front-loaded': dict(zip(range(3, 25, 3), [120, 110, 100, 80, 70, 50, 40, 30])),
             'back-loaded': dict(zip(range(3, 25, 3), [30, 40, 50, 70, 80, 100, 110, 120])),
             'schedule A (S-curve)': RA.SCHEDULE_A}
CLAUSE_BAND = 0.05            # illustrative clause: changes beyond +/-5% of the base price are passed on


def nominal_series():
    panel, _ = D.load_price_panel(D.DEFAULT_DATA_DIR, RA.OUT_ROOT / 'cache' / 'panel.pkl')
    wide, _, _ = R.splice(R.select_rebar(panel))
    return wide


def part_boot_backtest():
    lp = np.log(nominal_series()[RA.SERIES].dropna())
    det = RA.load_backtest()
    origins = sorted(det.origin.unique())
    rows = []
    for L, label in ((12, 'Block bootstrap (demeaned), mean block 12'), (1.4, 'Block bootstrap (demeaned), mean block 1.4')):
        for o in origins:
            t0 = pd.Timestamp(o)
            r = lp.loc[:t0].diff().to_numpy()
            paths = RM.block_bootstrap_paths(r, 12, 2000, B.rng_for('revision2', 'boot', o, L), mean_block=L)[0]
            for h in (1, 3, 6, 12):
                t1 = t0 + pd.DateOffset(months=h)
                if t1 not in lp.index:
                    continue
                q = np.quantile(np.log(paths[h - 1]), RM.TAUS)
                rows.append({'origin': o, 'h': h, 'model': label, **BR.score(q, lp[t1] - lp[t0])})
    bd = pd.DataFrame(rows)
    bd.to_csv(OUT / 'data_boot_backtest.csv', index=False)
    rng = B.rng_for('revision2', 'boot-ci')
    summ = []
    for (m, h), g in bd.groupby(['model', 'h']):
        rw = det[(det.h == h) & (det.model == 'Random walk')].set_index('origin').reindex(g.origin)
        c, b = g.crps_pct.to_numpy(), rw.crps_pct.to_numpy()
        idx = RA.stationary_indices(len(c), RA.block_length(h), RA.N_BOOT, rng)
        bs = 1 - c[idx].mean(1) / b[idx].mean(1)
        cov = g.le_P95.to_numpy()
        bc = cov[idx].mean(1)
        stat, p = BR.dm_test(c - b, h)
        summ.append({'model': m, 'h': h, 'n': len(c), 'crps_pct': c.mean(), 'skill': 1 - c.mean() / b.mean(),
                     'skill_lo': np.quantile(bs, 0.05), 'skill_hi': np.quantile(bs, 0.95), 'dm_p': p,
                     'le_P95': cov.mean(), 'le_P95_lo': np.quantile(bc, 0.05), 'le_P95_hi': np.quantile(bc, 0.95),
                     'in_P5_P95': g.in_P5_P95.mean(), 'le_P80': g.le_P80.mean()})
    pd.DataFrame(summ).to_csv(OUT / 'table_boot_backtest.csv', index=False)
    print('boot_backtest done')


def expenditure_ratio(paths, schedule, inflation=0.02):
    months = np.array(sorted(schedule)); q = np.array([schedule[m] for m in months], float)
    ratios = paths[months - 1] * ((1 + inflation) ** (months / 12))[:, None]
    return (q[:, None] * ratios).sum(0) / q.sum(), ratios, q


def scaling_factor(realized, allowance, target=0.95):
    """Smallest factor k for which realized <= k * allowance at a share of at least `target` of the origins."""
    ratio = np.sort((realized / allowance).to_numpy())
    return max(ratio[int(np.ceil(target * len(ratio))) - 1], 0.0)


def part_allowance():
    s, lp_real, origin = RA.real_series()
    r = lp_real.diff().to_numpy()
    n = 10000
    # the same 240-month paths as the projection and revision_analyses.py, so schedule A matches Table 13
    sims = {'block bootstrap': RM.block_bootstrap_paths(r, 240, n, B.rng_for('proj', 'boot', RA.SERIES))[0],
            'random walk': RM.rw_paths(r, 240, n, B.rng_for('proj', 'rw', RA.SERIES))[0]}
    prof = []
    for model, paths in sims.items():
        for name, sched in SCHEDULES.items():
            e, ratios, q = expenditure_ratio(paths, sched)
            # illustrative clause: each lot's price change beyond +/-5% of the base price is passed to the owner
            retained = (q[:, None] * (1 + np.clip(ratios - 1, -CLAUSE_BAND, CLAUSE_BAND))).sum(0) / q.sum()
            prof.append({'model': model, 'schedule': name, 'mean_month': sum(m * w for m, w in sched.items()) / sum(sched.values()),
                         'P80': 100 * (np.percentile(e, 80) - 1), 'P95': 100 * (np.percentile(e, 95) - 1),
                         'retained_P95_with_clause': 100 * (np.percentile(retained, 95) - 1),
                         'passed_on_P95_with_clause': 100 * np.percentile(e - retained, 95)})
    pd.DataFrame(prof).to_csv(OUT / 'table_procurement_profiles.csv', index=False)

    # estimation uncertainty of the schedule-A expenditure P95 (200 bootstrap histories x 2,000 paths)
    x = lp_real.diff().dropna().to_numpy()
    idx = RA.stationary_indices(len(x), 12, 200, B.rng_for('revision', 'estimation'))
    est = []
    for k in range(200):
        xs = np.r_[np.nan, x[idx[k]]]
        for model, fn in (('block bootstrap', lambda g: RM.block_bootstrap_paths(xs, 24, 2000, g)[0]),
                          ('random walk', lambda g: RM.rw_paths(xs, 24, 2000, g)[0])):
            e, ratios, _ = expenditure_ratio(fn(B.rng_for('revision2', 'est', model, k)), RA.SCHEDULE_A)
            est.append({'model': model, 'rep': k, 'P80': 100 * (np.percentile(e, 80) - 1), 'P95': 100 * (np.percentile(e, 95) - 1),
                        'endpoint24_P95': 100 * (np.percentile(ratios[-1], 95) - 1)})
    est = pd.DataFrame(est)
    est.groupby('model').agg(P95_median=('P95', 'median'), P95_lo=('P95', lambda v: v.quantile(0.05)),
                             P95_hi=('P95', lambda v: v.quantile(0.95)),
                             end24_P95_median=('endpoint24_P95', 'median'),
                             end24_P95_lo=('endpoint24_P95', lambda v: v.quantile(0.05)),
                             end24_P95_hi=('endpoint24_P95', lambda v: v.quantile(0.95)), P80_median=('P80', 'median'),
                             P80_lo=('P80', lambda v: v.quantile(0.05)), P80_hi=('P80', lambda v: v.quantile(0.95))
                             ).reset_index().to_csv(OUT / 'table_procurement_estimation.csv', index=False)

    # historical protection at several model percentiles (schedule A, nominal prices, models fitted at each origin)
    lp = np.log(s)
    months = np.array(sorted(RA.SCHEDULE_A)); q = np.array([RA.SCHEDULE_A[m] for m in months], float)
    origins = [o for o in lp.index if o >= pd.Timestamp('2019-01-01') and o + pd.DateOffset(months=24) <= lp.index[-1]]
    levels = [80, 90, 95, 97.5, 99]
    rows = []
    for o in origins:
        rr = lp.loc[:o].diff().to_numpy()
        realized = (q * np.exp(lp.reindex([o + pd.DateOffset(months=int(m)) for m in months]).to_numpy() - lp[o])).sum() / q.sum()
        row = {'origin': f'{o:%Y-%m}', 'realized_pct': 100 * (realized - 1)}
        for model, fn in (('random walk', lambda g: RM.rw_paths(rr, 24, 5000, g)[0]),
                          ('block bootstrap', lambda g: RM.block_bootstrap_paths(rr, 24, 5000, g)[0])):
            e = (q[:, None] * fn(B.rng_for('revision', 'proc', model, f'{o:%Y-%m}'))[months - 1]).sum(0) / q.sum()
            for lv in levels:
                row[f'{model}_P{lv}'] = 100 * (np.percentile(e, lv) - 1)
        rows.append(row)
    h = pd.DataFrame(rows)
    prot = []
    for model in ('random walk', 'block bootstrap'):
        for lv in levels:
            a = h[f'{model}_P{lv}']
            for period, mask in (('all origins', h.origin >= '0'), ('origins to 2022-12', h.origin <= '2022-12')):
                prot.append({'model': model, 'percentile': lv, 'period': period, 'n': int(mask.sum()),
                             'protection': float((h.realized_pct[mask] <= a[mask]).mean()),
                             'mean_allowance_pct': float(a[mask].mean())})
        # smallest scale on the P95 allowance that protects at least 95% of the past origins
        k = scaling_factor(h.realized_pct, h[f'{model}_P95'])
        prot.append({'model': model, 'percentile': 'P95 x k', 'period': 'all origins', 'n': len(h),
                     'protection': float((h.realized_pct <= k * h[f'{model}_P95']).mean()),
                     'mean_allowance_pct': float(k * h[f'{model}_P95'].mean()), 'k': float(k)})
    for fixed in (5, 10, 20, 30):
        prot.append({'model': 'fixed', 'percentile': f'+{fixed}%', 'period': 'all origins', 'n': len(h),
                     'protection': float((h.realized_pct <= fixed).mean()), 'mean_allowance_pct': float(fixed)})
    pd.DataFrame(prot).to_csv(OUT / 'table_allowance_protection.csv', index=False)

    # first evaluated origin: January 2019 (at least 24 monthly changes in every fit) against June 2020, the
    # first origin of the short-horizon backtest (at least 41 monthly changes)
    start = []
    for first in ('2019-01', '2020-06'):
        g = h[h.origin >= first]
        for model in ('random walk', 'block bootstrap'):
            a = g[f'{model}_P95']
            k = scaling_factor(g.realized_pct, a)
            start.append({'first_origin': first, 'model': model, 'n': len(g),
                          'n_to_2022_12': int((g.origin <= '2022-12').sum()),
                          'min_history_months': int((pd.Period(first, 'M') - pd.Period('2017-01', 'M')).n),
                          'mean_P95_pct': float(a.mean()), 'protection_P95': float((g.realized_pct <= a).mean()),
                          'k': float(k), 'protection_k': float((g.realized_pct <= k * a).mean())})
    pd.DataFrame(start).to_csv(OUT / 'table_allowance_start.csv', index=False)

    # the same scaling for the random-walk P95 of the h-month log change in the short-horizon backtest
    det = RA.load_backtest()
    d = det[(det.model == 'Random walk') & (det.series == RA.SERIES)]
    scal = []
    for h_, g in d.groupby('h'):
        for period, gg in (('all origins', g), ('origins to 2022-12', g[g.origin <= '2022-12']),
                           ('origins 2023-01 on', g[g.origin > '2022-12'])):
            scal.append({'h': h_, 'period': period, 'n': len(gg), 'k': float(scaling_factor(gg.y_log, gg.q95))})
    pd.DataFrame(scal).to_csv(OUT / 'table_backtest_scaling.csv', index=False)
    print('allowance done')


def part_dependence():
    s, lp_real, origin = RA.real_series()
    cpi = D.load_cpi_index(D.DEFAULT_CPI_BKK)
    fx = D.load_fx(D.DEFAULT_DATA_DIR)
    df = pd.DataFrame({'rebar': lp_real.diff(), 'infl': np.log(cpi).diff().reindex(lp_real.index),
                       'fx': np.log(fx).diff().reindex(lp_real.index)}).dropna()
    corr = {f'{a}-{b}, lag {k}': df[a].corr(df[b].shift(k)) for a, b in (('rebar', 'infl'), ('rebar', 'fx'))
            for k in (0, 1)}
    x = df.to_numpy()
    x_dm = x.copy(); x_dm[:, 0] -= x[:, 0].mean()            # zero real drift as in the headline band
    H, n = 240, 10000
    def path_indices(rng, T=len(x), mean_block=12):
        """(H, n) indices of a stationary bootstrap of length H from a sample of length T."""
        idx = np.empty((H, n), int)
        idx[0] = rng.integers(0, T, n)
        new = rng.random((H, n)) < 1 / mean_block
        starts = rng.integers(0, T, (H, n))
        for t in range(1, H):
            idx[t] = np.where(new[t], starts[t], (idx[t - 1] + 1) % T)
        return idx

    rows = []
    for label, joint in (('joint resampling', True), ('independent resampling', False)):
        rng = B.rng_for('revision2', 'dependence', label)
        idx = [path_indices(rng) for _ in range(1 if joint else 3)]
        cols = [x_dm[idx[0 if joint else j], j] for j in range(3)]
        real = np.exp(np.cumsum(cols[0], 0)); infl = np.exp(np.cumsum(cols[1], 0)); fxp = np.exp(np.cumsum(cols[2], 0))
        for y in (1, 5, 10, 20):
            t = 12 * y - 1
            rows.append({'resampling': label, 'year': y, 'real_P95': 100 * (np.percentile(real[t], 95) - 1),
                         'nominal_P95': 100 * (np.percentile(real[t] * infl[t], 95) - 1),
                         'usd_P95': 100 * (np.percentile(real[t] * infl[t] / fxp[t], 95) - 1)})
    pd.DataFrame(rows).to_csv(OUT / 'table_dependence.csv', index=False)
    pd.DataFrame([{'pair': k, 'correlation': v} for k, v in corr.items()]).to_csv(OUT / 'table_dependence_corr.csv', index=False)
    print('dependence done')


def part_threshold():
    w = nominal_series()
    lp = np.log(w)
    sr24 = [c for c in lp.columns if c.startswith('SR24')]
    bench = lp[sr24].diff().median(axis=1)
    junc = pd.to_datetime(RA.JUNCTIONS)
    rows = []
    for series in ('SD40 12mm', 'SD50 12mm', 'SD40 16mm', 'SD50 16mm'):
        d = (lp[series].diff() - bench).dropna()
        d = d[~d.index.isin(junc)].abs() * 100
        rows.append({'series': series, 'n_months': len(d), 'p50': d.quantile(0.5), 'p90': d.quantile(0.9),
                     'p95': d.quantile(0.95), 'p99': d.quantile(0.99), 'max': d.max(), 'share_gt_2': (d > 2).mean()})
    pd.DataFrame(rows).to_csv(OUT / 'table_threshold_basis.csv', index=False)
    print('threshold done')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+', default=['boot_backtest', 'allowance', 'dependence', 'threshold'])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    for p in args.parts:
        globals()[f'part_{p}']()


if __name__ == '__main__':
    main()
