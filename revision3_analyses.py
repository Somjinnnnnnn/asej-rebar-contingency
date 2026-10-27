"""Analyses for the seventh pre-submission review (October 2026).

Parts (CSVs to outputs/revision_2026-10):
  procedure    historical protection of the schedule-A P95 allowance when the models are fitted to REAL changes at
               each origin and nominal expenditure is formed at 2% inflation, the procedure used for the September
               2026 allowances, next to the nominal-fit procedure of revision2_analyses.py.
  uncertainty  90% intervals for the protection shares from a stationary bootstrap of the 69 origins (mean block
               24 months, the length of the overlapping outcomes), and protection on non-overlapping origins
               (origins 24 months apart, one subsample per starting month).
  blocks       the same intervals with mean blocks of 6, 12 and 36 months.
  scaling      a scaling rule estimated only from earlier origins: at each origin, the smallest factor that would
               have protected at least 95% of the origins whose 24-month outcomes were already known (at least 12).
  family       Holm and Benjamini-Hochberg adjustment over the complete comparison family: the 60 exploratory
               comparisons, the three primary comparisons not against the random walk, and the four block-bootstrap
               comparisons (67 tests).

Usage:
    python revision3_analyses.py [--parts procedure uncertainty scaling family]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import backtest as B
import rebar_models as RM
import revision2_analyses as R2
import revision_analyses as RA

OUT = R2.OUT
INFLATION = 0.02
MIN_PRIOR = 12  # earlier origins with known outcomes required before the out-of-sample scaling rule is applied


def schedule():
    months = np.array(sorted(RA.SCHEDULE_A))
    return months, np.array([RA.SCHEDULE_A[m] for m in months], float)


def history() -> pd.DataFrame:
    """Nominal-fit allowances and realized expenditure per origin, as written by revision2_analyses.py."""
    return pd.read_csv(OUT / 'data_procurement_history.csv', dtype={'origin': str})


def part_procedure():
    s, lp_real, _ = RA.real_series()
    lp = np.log(s)
    months, q = schedule()
    infl = (1 + INFLATION) ** (months / 12)
    origins = [o for o in lp.index if o >= pd.Timestamp('2019-01-01') and o + pd.DateOffset(months=24) <= lp.index[-1]]
    rows = []
    for o in origins:
        rr = lp_real.loc[:o].diff().to_numpy()  # real changes known at the origin (CPI up to the origin)
        realized = (q * np.exp(lp.reindex([o + pd.DateOffset(months=int(m)) for m in months]).to_numpy() - lp[o])).sum() / q.sum()
        row = {'origin': f'{o:%Y-%m}', 'realized_pct': 100 * (realized - 1)}
        for model, fn in (('random walk', lambda g: RM.rw_paths(rr, 24, 5000, g)[0]),
                          ('block bootstrap', lambda g: RM.block_bootstrap_paths(rr, 24, 5000, g)[0])):
            paths = fn(B.rng_for('revision3', 'proc-real', model, f'{o:%Y-%m}'))[months - 1] * infl[:, None]
            e = (q[:, None] * paths).sum(0) / q.sum()
            row[f'{model}_P95_pct'] = 100 * (np.percentile(e, 95) - 1)
        rows.append(row)
    h = pd.DataFrame(rows)
    h.to_csv(OUT / 'data_procurement_history_real.csv', index=False)
    nom = history()
    out = []
    for label, d in (('nominal fit', nom), ('real fit, 2% inflation', h)):
        for model in ('random walk', 'block bootstrap'):
            a = d[f'{model}_P95_pct']
            for period, mask in (('all origins', d.origin >= '0'), ('origins to 2022-12', d.origin <= '2022-12')):
                hit = d.realized_pct[mask] <= a[mask]
                out.append({'procedure': label, 'model': model, 'period': period, 'n': int(mask.sum()),
                            'protected': int(hit.sum()), 'protection': float(hit.mean()),
                            'mean_P95_pct': float(a[mask].mean())})
    pd.DataFrame(out).to_csv(OUT / 'table_procedure_protection.csv', index=False)
    print('procedure done')


def part_uncertainty():
    h = history()
    rng = B.rng_for('revision3', 'protection-ci')
    idx = RA.stationary_indices(len(h), 24, 2000, rng)
    rows = []
    for name, a in (('random walk P95', h['random walk_P95_pct']), ('block bootstrap P95', h['block bootstrap_P95_pct']),
                    ('fixed +10%', pd.Series(10.0, index=h.index))):
        hit = (h.realized_pct <= a).to_numpy(float)
        boot = hit[idx].mean(1)
        sub = [hit[j::24].mean() for j in range(24)]  # origins 24 months apart: non-overlapping outcomes
        rows.append({'allowance': name, 'n': len(hit), 'protected': int(hit.sum()), 'protection': hit.mean(),
                     'ci_lo': np.quantile(boot, 0.05), 'ci_hi': np.quantile(boot, 0.95),
                     'nonoverlap_mean': float(np.mean(sub)), 'nonoverlap_min': float(np.min(sub)),
                     'nonoverlap_max': float(np.max(sub)), 'nonoverlap_size': f'{min(len(hit[j::24]) for j in range(24))}'
                                                                             f'-{max(len(hit[j::24]) for j in range(24))}'})
    pd.DataFrame(rows).to_csv(OUT / 'table_protection_uncertainty.csv', index=False)
    print('uncertainty done')


def part_blocks():
    """Sensitivity of the 90% protection intervals to the mean block length (24 months is the main choice)."""
    h = history()
    main = pd.read_csv(OUT / 'table_protection_uncertainty.csv').set_index('allowance')
    rows = []
    for name, a in (('random walk P95', h['random walk_P95_pct']), ('block bootstrap P95', h['block bootstrap_P95_pct']),
                    ('fixed +10%', pd.Series(10.0, index=h.index))):
        hit = (h.realized_pct <= a).to_numpy(float)
        for L in (6, 12, 24, 36):
            if L == 24:
                lo, hi = main.loc[name, 'ci_lo'], main.loc[name, 'ci_hi']
            else:
                boot = hit[RA.stationary_indices(len(hit), L, 2000, B.rng_for('revision3', 'protection-ci', L))].mean(1)
                lo, hi = np.quantile(boot, 0.05), np.quantile(boot, 0.95)
            rows.append({'allowance': name, 'mean_block': L, 'protection': hit.mean(), 'ci_lo': lo, 'ci_hi': hi})
    pd.DataFrame(rows).to_csv(OUT / 'table_protection_ci_blocks.csv', index=False)
    print('blocks done')


def part_scaling():
    h = history()
    dates = pd.to_datetime(h.origin)
    rows = []
    for model in ('random walk', 'block bootstrap'):
        a = h[f'{model}_P95_pct']
        for i, o in enumerate(dates):
            prior = dates + pd.DateOffset(months=24) <= o  # outcome known at origin o
            if prior.sum() < MIN_PRIOR:
                continue
            k = R2.scaling_factor(h.realized_pct[prior], a[prior])
            rows.append({'model': model, 'origin': h.origin[i], 'n_prior': int(prior.sum()), 'k': k,
                         'allowance_pct': k * a[i], 'protected': bool(h.realized_pct[i] <= k * a[i])})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'data_scaling_oos.csv', index=False)
    d.groupby('model').agg(first_origin=('origin', 'min'), last_origin=('origin', 'max'), n=('origin', 'size'),
                           protected=('protected', 'sum'), protection=('protected', 'mean'),
                           k_min=('k', 'min'), k_max=('k', 'max'), mean_allowance_pct=('allowance_pct', 'mean')
                           ).reset_index().to_csv(OUT / 'table_scaling_oos.csv', index=False)
    print('scaling done')


def adjust(p: np.ndarray):
    """Holm and Benjamini-Hochberg adjusted p-values."""
    n = len(p); order = np.argsort(p)
    holm = np.empty(n); bh = np.empty(n)
    holm[order] = np.minimum(1, np.maximum.accumulate((n - np.arange(n)) * p[order]))
    ranked = p[order] * n / np.arange(1, n + 1)
    bh[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return holm, bh


def part_family():
    dm = pd.read_csv(OUT / 'table_dm_adjusted.csv')
    dm = dm[dm.benchmark == 'Random walk'][['model', 'h', 'p_value']].assign(family='exploratory')
    prim = pd.read_csv(OUT / 'table_primary_tests.csv')
    prim = prim[prim.benchmark != 'Random walk'].rename(columns={'p_value': 'p_value'})[['model', 'h', 'p_value']]
    prim = prim.assign(model=prim.model + ' vs ' + pd.read_csv(OUT / 'table_primary_tests.csv').loc[prim.index, 'benchmark'],
                       family='primary, not vs random walk')
    boot = pd.read_csv(OUT / 'table_boot_backtest.csv')
    boot = boot[boot.model.str.endswith('mean block 12')].rename(columns={'dm_p': 'p_value'})[['model', 'h', 'p_value']]
    boot = boot.assign(family='block bootstrap')
    f = pd.concat([dm, prim, boot], ignore_index=True)
    f['p_holm_family'], f['p_bh_family'] = adjust(f.p_value.to_numpy())
    f.to_csv(OUT / 'table_dm_full_family.csv', index=False)
    print('family done:', len(f), 'tests; min Holm', round(f.p_holm_family.min(), 3), 'min BH', round(f.p_bh_family.min(), 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+', default=['procedure', 'uncertainty', 'blocks', 'scaling', 'family'])
    for p in ap.parse_args().parts:
        globals()[f'part_{p}']()


if __name__ == '__main__':
    main()
