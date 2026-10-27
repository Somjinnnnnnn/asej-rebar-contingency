"""Additional analyses requested in the pre-submission review (October 2026).

Parts (each writes CSVs to outputs/<out-dir>/, default revision_2026-10):
  backtest      uncertainty of the backtest results: stationary-bootstrap intervals for coverage and
                for CRPS skill against the random walk, non-overlapping-origin subsamples, multiple-
                testing adjustments (Holm for the primary family, Benjamini-Hochberg for all
                model-vs-random-walk tests), economic size of the one-month Machine Learning gain,
                and the CRPS approximation error of the 19-quantile grid.
  procurement   worked procurement example: total-expenditure allowances for a 24-month purchase
                schedule and a 20-year purchase programme from September 2026, against per-date and
                end-point percentiles; historical check of the 24-month schedule at past origins.
  sensitivity   headline-band sensitivity: block length, exclusion of the 2020-2022 boom-bust and of
                the code-junction months, Monte Carlo stability, estimation uncertainty (bootstrap
                of the history), mean versus median under zero drift; block-length sensitivity of
                the US steel PPI validation.

Usage:
    python revision_analyses.py [--parts backtest procurement sensitivity] [--workers 12]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import backtest as B
import backtest_rebar as BR
import data as D
import drivers as G
import models as M
import rebar_data as R
import rebar_models as RM

warnings.filterwarnings('ignore', message='Workbook contains no default style')
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / 'outputs'
SERIES = 'SD40 12mm'
N_BOOT = 2000                      # stationary-bootstrap resamples of the forecast origins
TABLE4_MODELS = ['GBM (data-calibrated)', 'Empirical h-month changes', 'AR(1)', 'GARCH(1,1)', 'AR(1)-GARCH(1,1)',
                 'Merton jump-diffusion', 'Markov regime-switching', 'Linear QR, own history, CV-tuned',
                 'Linear QR, + scrap+ore, CV-tuned', 'Linear QR, + all drivers', 'Linear QR, + all drivers, CV-tuned',
                 'XGBoost QR, own history', 'XGBoost QR, own history, CV-tuned', 'XGBoost QR, + scrap+ore, CV-tuned',
                 'XGBoost QR, + all drivers, CV-tuned']
KEY_MODELS = ['Random walk', 'GBM (data-calibrated)', 'AR(1)-GARCH(1,1)', 'Linear QR, own history, CV-tuned',
              'XGBoost QR, own history, CV-tuned', 'XGBoost QR, + all drivers, CV-tuned',
              'Fixed +5% contingency', 'Fixed +10% contingency']
# primary (confirmatory) family: the one-month Machine Learning claims of RQ2
PRIMARY = [('XGBoost QR, own history, CV-tuned', 'Random walk', 'Machine Learning vs random walk'),
           ('XGBoost QR, own history, CV-tuned', 'Linear QR, own history, CV-tuned', 'learner: XGBoost vs linear'),
           ('XGBoost QR, + all drivers, CV-tuned', 'XGBoost QR, own history, CV-tuned', 'drivers: all vs own history'),
           ('XGBoost QR, + scrap+ore, CV-tuned', 'XGBoost QR, own history, CV-tuned', 'drivers: scrap+ore vs own history')]
BOOM = ('2020-06-01', '2022-12-01')      # boom-bust episode excluded in the regime sensitivity
JUNCTIONS = ['2018-02-01', '2018-06-01', '2026-04-01']
SCHEDULE_A = {3: 50, 6: 75, 9: 100, 12: 100, 15: 100, 18: 75, 21: 60, 24: 40}   # month after origin: tons
SCHEDULE_B = {12 * k: 30 for k in range(1, 21)}                                 # 20 annual purchases


# --------------------------------------------------------------------------- helpers

def stationary_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (reps, n) of the stationary bootstrap (Politis and Romano, 1994)."""
    idx = np.empty((reps, n), int)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < 1 / mean_block
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def block_length(h: int) -> int:
    """Mean block length for resampling forecast origins: twice the forecast horizon (overlapping
    h-month outcomes are dependent up to lag h - 1), at least 3."""
    return max(2 * h, 3)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    running = 1.0
    for rank, i in zip(range(n, 0, -1), order):
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return adj


def load_backtest() -> pd.DataFrame:
    det = pd.read_csv(OUT_ROOT / 'backtest_v2' / 'backtest_detail.csv', encoding='utf-8-sig')
    return det[det.series == SERIES]


# --------------------------------------------------------------------------- part 1: backtest

def part_backtest(out: Path):
    det = load_backtest()
    rng = B.rng_for('revision', 'backtest')
    rows, nonov = [], []
    for h in sorted(det.h.unique()):
        dh = det[det.h == h]
        sub = {k: dh.pivot(index='origin', columns='model', values=k).sort_index()
               for k in ('crps_pct', 'le_P95', 'in_P5_P95')}
        n = len(sub['crps_pct'])
        idx = stationary_indices(n, block_length(h), N_BOOT, rng)
        rw = sub['crps_pct']['Random walk'].to_numpy()
        for m in KEY_MODELS + [x for x in TABLE4_MODELS if x not in KEY_MODELS]:
            if m not in sub['crps_pct']:
                continue
            c = sub['crps_pct'][m].to_numpy()
            row = {'h': h, 'model': m, 'n': n, 'mean_block': block_length(h)}
            if not m.startswith('Fixed'):
                skill = 1 - c.mean() / rw.mean()
                bs = 1 - c[idx].mean(1) / rw[idx].mean(1)
                row.update(skill=skill, skill_lo=np.quantile(bs, 0.05), skill_hi=np.quantile(bs, 0.95))
            for k in ('le_P95', 'in_P5_P95'):
                x = sub[k][m].to_numpy()
                if np.isnan(x).all():
                    continue
                bx = np.nanmean(x[idx], 1)
                row.update({k: np.nanmean(x), f'{k}_lo': np.quantile(bx, 0.05), f'{k}_hi': np.quantile(bx, 0.95)})
            rows.append(row)
        # non-overlapping origins: every h-th origin, for each of the h possible offsets
        for m in KEY_MODELS:
            if m not in sub['crps_pct'] or m.startswith('Fixed'):
                continue
            sk, cov, ns, pv = [], [], [], []
            for off in range(h):
                sel = slice(off, None, h)
                c, b = sub['crps_pct'][m].to_numpy()[sel], rw[sel]
                ns.append(len(c)); cov.append(np.nanmean(sub['le_P95'][m].to_numpy()[sel]))
                if m != 'Random walk':
                    sk.append(1 - c.mean() / b.mean())
                    d = c - b
                    pv.append(2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d) - 1)
                    if len(d) > 2 and d.std() > 0 else np.nan)
            nonov.append({'h': h, 'model': m, 'n_per_subsample': f'{min(ns)}-{max(ns)}', 'offsets': h,
                          'skill_mean': np.mean(sk) if sk else np.nan, 'skill_min': np.min(sk) if sk else np.nan,
                          'skill_max': np.max(sk) if sk else np.nan, 'le_P95_mean': np.mean(cov),
                          'le_P95_min': np.min(cov), 'le_P95_max': np.max(cov),
                          'share_offsets_p_lt_005': np.mean(np.array(pv) < 0.05) if pv else np.nan})
    pd.DataFrame(rows).to_csv(out / 'table_backtest_uncertainty.csv', index=False)
    pd.DataFrame(nonov).to_csv(out / 'table_backtest_nonoverlap.csv', index=False)

    # multiple testing: primary family (Holm) and all model-vs-random-walk tests (BH)
    prim = []
    for m, b, label in PRIMARY:
        p = det[(det.h == 1) & det.model.isin([m, b])].pivot_table(index='origin', columns='model', values='crps_pct')
        d = (p[m] - p[b]).dropna().to_numpy()
        stat, pval = BR.dm_test(d, 1)
        prim.append({'hypothesis': label, 'model': m, 'benchmark': b, 'h': 1, 'n': len(d),
                     'skill': 1 - p[m].mean() / p[b].mean(), 'dm_stat': stat, 'p_value': pval})
    prim = pd.DataFrame(prim)
    prim['p_holm'] = holm(prim.p_value.to_numpy())
    prim.to_csv(out / 'table_primary_tests.csv', index=False)
    dm = pd.read_csv(OUT_ROOT / 'backtest_v2' / 'table_dm_tests.csv', encoding='utf-8-sig')
    dm = dm[(dm.series == SERIES) & (dm.benchmark == 'Random walk') & dm.model.isin(TABLE4_MODELS)].copy()
    dm['p_bh_all'] = bh(dm.p_value.to_numpy())
    dm['p_holm_all'] = holm(dm.p_value.to_numpy())
    dm.to_csv(out / 'table_dm_adjusted.csv', index=False)

    # economic size of the one-month gain: P95 allowance and pinball loss, RW vs tuned XGBoost
    econ = []
    for m in ['Random walk', 'XGBoost QR, own history, CV-tuned', 'XGBoost QR, + all drivers, CV-tuned']:
        s = det[(det.h == 1) & (det.model == m)]
        econ.append({'model': m, 'mean_P95_allowance_pct': 100 * (s.P95.mean() - 1),
                     'mean_abs_P95_diff_vs_rw_pct': np.nan, 'pinball95_pct': s.pinball95_pct.mean(),
                     'crps_pct': s.crps_pct.mean(), 'le_P95': s.le_P95.mean()})
    rwp = det[(det.h == 1) & (det.model == 'Random walk')].set_index('origin').P95
    for e in econ[1:]:
        mp = det[(det.h == 1) & (det.model == e['model'])].set_index('origin').P95
        e['mean_abs_P95_diff_vs_rw_pct'] = 100 * (mp - rwp).abs().mean()
    pd.DataFrame(econ).to_csv(out / 'table_ml_economic.csv', index=False)

    # CRPS approximation: 19-quantile grid vs a 999-quantile grid (random walk, GBM, empirical)
    wide_lp = R.splice(R.select_rebar(D.load_price_panel(D.DEFAULT_DATA_DIR, OUT_ROOT / 'cache' / 'panel.pkl')[0]))[0]
    lp = np.log(wide_lp[SERIES].dropna())
    taus_dense = np.arange(1, 1000) / 1000
    z19, zd = stats.norm.ppf(RM.TAUS), stats.norm.ppf(taus_dense)
    crps_rows = []
    for o in sorted(det.origin.unique()):
        t0 = pd.Timestamp(o)
        hist = lp.loc[:t0]
        r = hist.diff().to_numpy()
        s = np.nanstd(r, ddof=1)
        mu, sg = M.gbm_params_from_history(r)
        for h in (1, 3, 6, 12):
            t1 = t0 + pd.DateOffset(months=h)
            if t1 not in lp.index:
                continue
            y = np.exp(lp[t1] - lp[t0])
            emp = hist.to_numpy()
            for name, q19, qd in [
                    ('Random walk', np.sqrt(h) * s * z19, np.sqrt(h) * s * zd),
                    ('GBM (data-calibrated)', h * (mu - 0.5 * sg ** 2) + np.sqrt(h) * sg * z19,
                     h * (mu - 0.5 * sg ** 2) + np.sqrt(h) * sg * zd),
                    ('Empirical h-month changes', np.quantile(emp[h:] - emp[:-h], RM.TAUS),
                     np.quantile(emp[h:] - emp[:-h], taus_dense))]:
                c19 = 100 * 2 * np.mean([M.pinball(np.exp(q), y, t) for q, t in zip(q19, RM.TAUS)])
                cd = 100 * 2 * np.mean([M.pinball(np.exp(q), y, t) for q, t in zip(qd, taus_dense)])
                crps_rows.append({'origin': o, 'h': h, 'model': name, 'crps19': c19, 'crps999': cd})
    cr = pd.DataFrame(crps_rows)
    summ = cr.groupby(['h', 'model'])[['crps19', 'crps999']].mean().reset_index()
    rwm = summ[summ.model == 'Random walk'].set_index('h')
    summ['skill19'] = 1 - summ.crps19 / summ.h.map(rwm.crps19)
    summ['skill999'] = 1 - summ.crps999 / summ.h.map(rwm.crps999)
    summ.to_csv(out / 'table_crps_grid.csv', index=False)
    print('backtest part done')


# --------------------------------------------------------------------------- part 2: procurement

def real_series():
    panel, _ = D.load_price_panel(D.DEFAULT_DATA_DIR, OUT_ROOT / 'cache' / 'panel.pkl')
    wide, _, _ = R.splice(R.select_rebar(panel))
    s = wide[SERIES].dropna()
    cpi = D.load_cpi_index(D.DEFAULT_CPI_BKK)
    origin = s.index[-1]
    real = s / cpi.reindex(s.index) * cpi[origin]
    return s, np.log(real), origin


def expenditure_stats(paths_real: np.ndarray, schedule: dict, inflation: float) -> dict:
    """paths_real: (H, n) real price ratios. Returns allowances on the total expenditure, the
    per-date percentiles and the end-point percentile, all in % above the origin-price budget."""
    months = np.array(sorted(schedule))
    q = np.array([schedule[m] for m in months], float)
    infl = (1 + inflation) ** (months / 12)
    ratios = paths_real[months - 1] * infl[:, None]          # (purchases, n) nominal price ratios
    exp_ratio = (q[:, None] * ratios).sum(0) / q.sum()       # expenditure / budget at origin price
    per95 = np.percentile(ratios, 95, axis=1)
    return {'P50': 100 * (np.percentile(exp_ratio, 50) - 1), 'P80': 100 * (np.percentile(exp_ratio, 80) - 1),
            'P95': 100 * (np.percentile(exp_ratio, 95) - 1), 'mean': 100 * (exp_ratio.mean() - 1),
            'prob_within_5': np.mean(exp_ratio <= 1.05), 'prob_within_10': np.mean(exp_ratio <= 1.10),
            'weighted_per_date_P95': 100 * ((q * per95).sum() / q.sum() - 1),
            'endpoint_P95': 100 * (per95[-1] - 1), 'endpoint_P80': 100 * (np.percentile(ratios[-1], 80) - 1)}


def part_procurement(out: Path):
    s, lp_real, origin = real_series()
    r = lp_real.diff().to_numpy()
    n = 10000
    H = 240
    sims = {'Block bootstrap (demeaned)': RM.block_bootstrap_paths(r, H, n, B.rng_for('proj', 'boot', SERIES))[0],
            'Random walk (real, driftless)': RM.rw_paths(r, H, n, B.rng_for('proj', 'rw', SERIES))[0]}
    rows = []
    for model, paths in sims.items():
        for sched_name, sched in [('A: 24-month building, 600 t in 8 quarterly lots', SCHEDULE_A),
                                  ('B: 20-year programme, 20 annual lots of 30 t', SCHEDULE_B)]:
            for basis, infl in [('nominal THB, 2% inflation', 0.02), ('real THB', 0.0)]:
                rows.append({'model': model, 'schedule': sched_name, 'basis': basis,
                             'tons': sum(sched.values()), 'budget_mthb': sum(sched.values()) * s[origin] / 1e6,
                             **expenditure_stats(paths, sched, infl)})
    pd.DataFrame(rows).to_csv(out / 'table_procurement_example.csv', index=False)

    # historical check of schedule A with nominal prices and models fitted at each origin
    lp = np.log(s)
    months = np.array(sorted(SCHEDULE_A)); q = np.array([SCHEDULE_A[m] for m in months], float)
    hist_rows = []
    origins = [o for o in lp.index if o >= pd.Timestamp('2019-01-01') and o + pd.DateOffset(months=24) <= lp.index[-1]]
    for o in origins:
        hist = lp.loc[:o]
        rr = hist.diff().to_numpy()
        realized = (q * np.exp(lp.reindex([o + pd.DateOffset(months=int(m)) for m in months]).to_numpy() - lp[o])).sum() / q.sum()
        row = {'origin': f'{o:%Y-%m}', 'realized_pct': 100 * (realized - 1)}
        for model, fn in [('random walk', lambda g: RM.rw_paths(rr, 24, 5000, g)[0]),
                          ('block bootstrap', lambda g: RM.block_bootstrap_paths(rr, 24, 5000, g)[0])]:
            paths = fn(B.rng_for('revision', 'proc', model, f'{o:%Y-%m}'))
            e = (q[:, None] * paths[months - 1]).sum(0) / q.sum()
            row[f'{model}_P80_pct'] = 100 * (np.percentile(e, 80) - 1)
            row[f'{model}_P95_pct'] = 100 * (np.percentile(e, 95) - 1)
        hist_rows.append(row)
    hd = pd.DataFrame(hist_rows)
    hd.to_csv(out / 'data_procurement_history.csv', index=False)
    summ = []
    for label, col in [('fixed +5%', None), ('fixed +10%', None), ('random walk P80', 'random walk_P80_pct'),
                       ('random walk P95', 'random walk_P95_pct'), ('block bootstrap P80', 'block bootstrap_P80_pct'),
                       ('block bootstrap P95', 'block bootstrap_P95_pct')]:
        allow = hd[col] if col else pd.Series(5.0 if '5%' in label else 10.0, index=hd.index)
        for period, mask in [('all origins', hd.origin >= '0'), ('origins 2019-01 to 2022-12', hd.origin <= '2022-12'),
                             ('origins 2023-01 on', hd.origin >= '2023-01')]:
            summ.append({'allowance': label, 'period': period, 'n_origins': int(mask.sum()),
                         'coverage': float((hd.realized_pct[mask] <= allow[mask]).mean()),
                         'mean_allowance_pct': float(allow[mask].mean()),
                         'mean_shortfall_pct': float((hd.realized_pct[mask] - allow[mask]).clip(lower=0).mean())})
    pd.DataFrame(summ).to_csv(out / 'table_procurement_history.csv', index=False)
    print('procurement part done')


# --------------------------------------------------------------------------- part 3: sensitivity

YEARS = [1, 5, 10, 20]


def pct_rows(paths, label, extra=None):
    rows = []
    for y in YEARS:
        x = paths[12 * y - 1]
        rows.append({'variant': label, 'year': y, **(extra or {}),
                     **{f'P{p}_pct': 100 * (np.percentile(x, p) - 1) for p in (50, 80, 95)},
                     'mean_pct': 100 * (x.mean() - 1), 'prob_below_today': np.mean(x < 1)})
    return rows


def _val_origin(args):
    origin, lp_full, blocks, n = args
    warnings.filterwarnings('ignore')
    hist = lp_full.loc[:origin].iloc[-121:].to_numpy()
    r = np.diff(hist, prepend=np.nan)
    rows = []
    for L in blocks:
        # block 12 reuses the stream of validate_long_horizon.py, so it reproduces Table 10
        keys = ('val', 'boot', f'{origin:%Y-%m}') if L == 12 else ('val', 'boot', f'{origin:%Y-%m}', L)
        paths = RM.block_bootstrap_paths(r, 60, n, B.rng_for(*keys), mean_block=L)[0]
        for h in (12, 36, 60):
            y = lp_full[origin + pd.DateOffset(months=h)] - lp_full[origin]
            qv = np.quantile(np.log(paths[h - 1]), RM.TAUS)
            rows.append({'origin': f'{origin:%Y-%m}', 'h': h, 'mean_block': L, **BR.score(qv, y)})
    return rows


def part_sensitivity(out: Path, workers: int):
    from arch.bootstrap import optimal_block_length
    s, lp_real, origin = real_series()
    r_full = lp_real.diff()
    r = r_full.to_numpy()
    n, H = 10000, 240
    pw = optimal_block_length(r_full.dropna().to_numpy())
    pw_stat = float(pw['stationary'].iloc[0])
    rows = []
    for L in (3, 6, 12, 24, 36, round(pw_stat, 1)):
        paths = RM.block_bootstrap_paths(r, H, n, B.rng_for('proj', 'boot', SERIES), mean_block=L)[0]
        rows += pct_rows(paths, f'mean block {L} months' + (' (Politis-White)' if L == round(pw_stat, 1) else ''),
                         {'group': 'block length'})
    keep = r_full.dropna()
    no_boom = keep[(keep.index < BOOM[0]) | (keep.index > BOOM[1])].to_numpy()
    no_junc = keep.drop(pd.to_datetime(JUNCTIONS), errors='ignore').to_numpy()
    for label, x in [('excluding the 2020-06 to 2022-12 changes', no_boom), ('excluding the code-junction months', no_junc)]:
        paths = RM.block_bootstrap_paths(np.r_[np.nan, x], H, n, B.rng_for('proj', 'boot', SERIES))[0]
        rows += pct_rows(paths, label, {'group': 'sample'})
        paths = RM.rw_paths(np.r_[np.nan, x], H, n, B.rng_for('proj', 'rw', SERIES))[0]
        rows += pct_rows(paths, label + ' (random walk)', {'group': 'sample'})
    base = RM.block_bootstrap_paths(r, H, n, B.rng_for('proj', 'boot', SERIES))[0]
    rows += pct_rows(base, 'headline (mean block 12, full sample)', {'group': 'headline'})
    pd.DataFrame(rows).to_csv(out / 'table_sensitivity_headline.csv', index=False)

    # Monte Carlo stability (seeds) and estimation uncertainty (bootstrap of the history)
    mc = []
    for k in range(20):
        p = RM.block_bootstrap_paths(r, H, n, B.rng_for('revision', 'mc', k))[0]
        for y in YEARS:
            mc.append({'seed': k, 'year': y, 'P80_pct': 100 * (np.percentile(p[12 * y - 1], 80) - 1),
                       'P95_pct': 100 * (np.percentile(p[12 * y - 1], 95) - 1)})
    mc = pd.DataFrame(mc)
    est = []
    x = keep.to_numpy()
    rng = B.rng_for('revision', 'estimation')
    idx = stationary_indices(len(x), 12, 200, rng)
    for k in range(200):
        xs = np.r_[np.nan, x[idx[k]]]
        for model, fn in [('block bootstrap', lambda g: RM.block_bootstrap_paths(xs, H, 2000, g)[0]),
                          ('random walk', lambda g: RM.rw_paths(xs, H, 2000, g)[0])]:
            p = fn(B.rng_for('revision', 'est', model, k))
            for y in YEARS:
                est.append({'model': model, 'rep': k, 'year': y,
                            'P80_pct': 100 * (np.percentile(p[12 * y - 1], 80) - 1),
                            'P95_pct': 100 * (np.percentile(p[12 * y - 1], 95) - 1)})
    est = pd.DataFrame(est)
    unc = []
    for y in YEARS:
        m = mc[mc.year == y]
        unc.append({'source': 'Monte Carlo (20 seeds x 10,000 paths)', 'model': 'block bootstrap', 'year': y,
                    'P95_center': m.P95_pct.mean(), 'P95_lo': m.P95_pct.min(), 'P95_hi': m.P95_pct.max(),
                    'P95_sd': m.P95_pct.std(), 'P80_center': m.P80_pct.mean(), 'P80_sd': m.P80_pct.std()})
        for model in ('block bootstrap', 'random walk'):
            e = est[(est.year == y) & (est.model == model)]
            unc.append({'source': 'estimation (200 bootstrap histories x 2,000 paths)', 'model': model, 'year': y,
                        'P95_center': e.P95_pct.median(), 'P95_lo': e.P95_pct.quantile(0.05),
                        'P95_hi': e.P95_pct.quantile(0.95), 'P95_sd': e.P95_pct.std(),
                        'P80_center': e.P80_pct.median(), 'P80_sd': e.P80_pct.std()})
    pd.DataFrame(unc).to_csv(out / 'table_uncertainty_sources.csv', index=False)
    pd.DataFrame({'quantity': ['Politis-White stationary block length (months)'], 'value': [pw_stat]}).to_csv(
        out / 'table_block_length_choice.csv', index=False)

    # block-length sensitivity of the US steel PPI validation (real, demeaned bootstrap)
    lp = np.log(G.load_fred('WPU1017'))
    lc = np.log(G.load_fred('CPIAUCSL')).reindex(pd.date_range(lp.index[0], lp.index[-1], freq='MS')).interpolate(limit_area='inside')
    lp = lp - lc + lc.iloc[-1]
    origins = pd.date_range('2010-01-01', '2021-09-01', freq='MS')
    blocks = [3, 6, 12, 24, 36]
    with ProcessPoolExecutor(workers) as ex:
        vrows = [x for part in ex.map(_val_origin, [(o, lp, blocks, 2000) for o in origins]) for x in part]
    v = pd.DataFrame(vrows)
    v.groupby(['mean_block', 'h']).agg(n=('crps_pct', 'size'), le_P95=('le_P95', 'mean'), in_P5_P95=('in_P5_P95', 'mean'),
                                       crps_pct=('crps_pct', 'mean')).reset_index().to_csv(
        out / 'table_validation_block_length.csv', index=False)
    print('sensitivity part done')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', nargs='+', default=['backtest', 'procurement', 'sensitivity'])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out-dir', default='revision_2026-10')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    out = OUT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if 'backtest' in args.parts:
        part_backtest(out)
    if 'procurement' in args.parts:
        part_procurement(out)
    if 'sensitivity' in args.parts:
        part_sensitivity(out, args.workers)


if __name__ == '__main__':
    main()
