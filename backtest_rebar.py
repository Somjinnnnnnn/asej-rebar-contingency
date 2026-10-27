"""Rolling-origin backtest of 1-12-month probabilistic forecasts of TPSO rebar list prices
(RQ1: price process; RQ2: Machine Learning and global steel drivers).

Origins are every month from --first-origin to the last month that still has a
realised value; the training window expands from Jan 2017. All models are scored on the
same 19 quantiles (TAUS = 0.05..0.95) of the price ratio P(origin + h) / P(origin):
  cover_P5_P95, cover_le_P80, cover_le_P95   empirical coverage
  pinball80_pct, pinball95_pct               pinball loss (% of the origin price)
  crps_pct                                   CRPS approximated by 2 x mean pinball loss over TAUS
Diebold-Mariano tests (Newey-West variance with h-1 lags, Harvey-Leybourne-Newbold
small-sample correction) compare the CRPS of each model with the random walk, GBM and
GARCH(1,1) benchmarks.

Writes to outputs/<out-dir>/:
  backtest_detail.csv, table_backtest_summary.csv, table_backtest_by_period.csv,
  table_dm_tests.csv, table_model_fits.csv, run_log.txt (stdout)

Usage:
    python backtest_rebar.py                                  # SD40 12 mm
    python backtest_rebar.py --series "SD50 12mm" "SR24 12mm" --out-dir backtest_robust
    python backtest_rebar.py --extra-lag 1 --out-dir backtest_lag1
    python backtest_rebar.py --quick
"""
from __future__ import annotations

import argparse
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import backtest as B
import data as D
import drivers as G
import models as M
import rebar_data as R
import rebar_models as RM

HORIZONS = [1, 3, 6, 12]
H_MAX = max(HORIZONS)
MIN_TRAIN = 24
FIRST_ORIGIN = '2020-06-01'  # first month with >= MIN_TRAIN direct-model rows at h = 12
PERIOD_SPLIT = pd.Timestamp('2023-01-01')
BENCHMARKS = ['Random walk', 'GBM (data-calibrated)', 'GARCH(1,1)']

warnings.filterwarnings('ignore', message='Workbook contains no default style')


def process_models(lp: pd.Series, origin: pd.Timestamp, n: int, series: str, fits: list) -> dict:
    """RQ1 models fitted to the history up to the origin; quantiles (H_MAX, len(TAUS))."""
    hist = lp.loc[:origin]
    r = hist.diff().to_numpy()
    price = np.exp(hist.to_numpy())
    out = {'Random walk': RM.rw_quantiles(r, H_MAX),
           'GBM (data-calibrated)': RM.gbm_quantiles(r, H_MAX),
           'Empirical h-month changes': RM.empirical_quantiles(hist.to_numpy(), H_MAX)}
    for name, garch in [('AR(1)', False), ('AR(1)-GARCH(1,1)', True)]:
        seed = int(B.rng_for('ar', name, series, str(origin)).integers(2 ** 31))
        out[name], info = RM.ar_quantiles(r, H_MAX, n, seed, garch)
        fits.append({'series': series, 'origin': f'{origin:%Y-%m}', 'model': name, **info})
    for name, fit in M.ALT_MODELS.items():
        sim, info = fit(r, H_MAX, n, B.rng_for('alt', name, series, str(origin)), price=price)
        out[name] = RM.sample_quantiles(sim)
        fits.append({'series': series, 'origin': f'{origin:%Y-%m}', 'model': name, **info})
    return out


def direct_models(feat: pd.DataFrame, lp: pd.Series, origin: pd.Timestamp, h: int, series: str,
                  args, tuning: list) -> dict:
    """RQ2 models for horizon h; quantiles (len(TAUS),) or empty dict if too few rows.

    Feature sets: own history, plus each driver set in --driver-sets. Regularisation:
    fixed a priori (RM.QR_ALPHA, RM.XGB_PARAMS) and/or chosen by time-series CV (--tune).
    All variants use the same training rows (complete in every driver)."""
    y = (lp.shift(-h) - lp).rename('y')
    all_drv = RM.OWN_FEATURES + RM.driver_features(feat)
    d = feat.join(y)
    d = d[(d.index + pd.DateOffset(months=h)) <= origin].dropna(subset=all_drv + ['y'])
    x_o = feat.loc[[origin]]
    if len(d) < MIN_TRAIN or x_o[all_drv].isna().any(axis=None):
        return {}
    seed = int(B.rng_for('xgb', series, str(origin), h).integers(2 ** 31))
    sets = [('own history', RM.OWN_FEATURES)] + [
        (f'+ {name}', RM.OWN_FEATURES + RM.driver_features(feat, RM.DRIVER_SETS[name])) for name in args.driver_sets]
    variants = {'fixed': [False], 'cv': [True], 'both': [False, True]}[args.tune]
    y_tr, mi = d['y'].to_numpy(), (d.index.year * 12 + d.index.month).to_numpy()
    out = {}
    for label, cols in sets:
        X, Xo = d[cols].to_numpy(float), x_o[cols].to_numpy(float)
        for tuned in variants:
            suffix = ', CV-tuned' if tuned else ''
            alpha, prm, folds = RM.QR_ALPHA, {}, 0
            if tuned:
                alpha, folds = RM.tune_linear_qr(X, y_tr, mi, h)
                prm, _ = RM.tune_xgb_qr(X, y_tr, mi, h, seed)
                tuning.append({'series': series, 'origin': f'{origin:%Y-%m}', 'h': h, 'features': label,
                               'folds': folds, 'qr_alpha': alpha, **{f'xgb_{k}': v for k, v in prm.items()}})
            out[f'Linear QR, {label}{suffix}'] = RM.linear_qr(X, y_tr, Xo, alpha)[0]
            out[f'XGBoost QR, {label}{suffix}'] = RM.xgb_qr(X, y_tr, Xo, seed, prm)[0]
    out['_n_train'] = len(d)
    return out


def score(q_log: np.ndarray, y_log: float) -> dict:
    """Scores on the price-ratio scale, in % of the origin price."""
    q, y = np.exp(q_log), np.exp(y_log)
    pin = np.array([M.pinball(qi, y, t) for qi, t in zip(q, RM.TAUS)])
    i = {t: k for k, t in enumerate(RM.TAUS)}
    # the log-ratio quantiles are kept so that post-processing (conformal_rebar.py) can
    # recalibrate them without re-running the models
    qcols = {f'q{int(round(100 * t)):02d}': v for t, v in zip(RM.TAUS, q_log)}
    return {**qcols, 'y_log': y_log, 'actual_ratio': y, 'P50': q[i[0.5]], 'P95': q[i[0.95]],
            'in_P5_P95': float(q[i[0.05]] <= y <= q[i[0.95]]), 'le_P80': float(y <= q[i[0.8]]),
            'le_P95': float(y <= q[i[0.95]]), 'pinball80_pct': 100 * pin[i[0.8]],
            'pinball95_pct': 100 * pin[i[0.95]], 'crps_pct': 100 * 2 * pin.mean()}


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p-value for loss differences d (HLN-corrected)."""
    d = d[~np.isnan(d)]
    n = len(d)
    u = d - d.mean()
    lags = max(h - 1, 0)
    v = u @ u / n + 2 * sum((1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lags + 1))
    if v <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(v / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def run_series(series, wide, drivers, args, fits, tuning, ckpt: Path):
    """Each finished origin is saved to `ckpt` (rows and fit records); a re-run with the
    same settings loads those files and continues with the remaining origins."""
    lp = np.log(wide[series].dropna())
    if len(pd.date_range(lp.index[0], lp.index[-1], freq='MS')) != len(lp):
        raise ValueError(f'{series} has gaps; the backtest needs a contiguous series')
    feat = RM.feature_table(lp, drivers, args.extra_lag)
    end = lp.index[-1]
    origins = pd.date_range(args.first_origin, end - pd.DateOffset(months=1), freq='MS')
    if args.quick:
        origins = origins[::12]
    if args.shard:
        i, k = args.shard
        origins = origins[i::k]
    rows = []
    settings = {'paths': args.paths, 'extra_lag': args.extra_lag, 'quick': args.quick,
                'driver_sets': list(args.driver_sets), 'tune': args.tune}
    ckpt.mkdir(parents=True, exist_ok=True)
    for origin in origins:
        f = ckpt / f'{series.replace(" ", "_")}_{origin:%Y-%m}.pkl'
        if f.exists():
            saved = pd.read_pickle(f)
            if saved['settings'] != settings:
                raise ValueError(f'{f} was written with {saved["settings"]}; use another --out-dir')
            rows += saved['rows']; fits += saved['fits']; tuning += saved['tuning']
            continue
        n_rows, n_fits, n_tune = len(rows), len(fits), len(tuning)
        proc = process_models(lp, origin, args.paths, series, fits)
        for h in HORIZONS:
            target = origin + pd.DateOffset(months=h)
            if target > end:
                continue
            y = lp[target] - lp[origin]
            base = {'series': series, 'origin': f'{origin:%Y-%m}', 'h': h}
            for name, q in proc.items():
                rows.append({**base, 'model': name, **score(q[h - 1], y)})
            dm = direct_models(feat, lp, origin, h, series, args, tuning)
            n_train = dm.pop('_n_train', 0)
            for name, q in dm.items():
                rows.append({**base, 'model': name, 'n_train': n_train, **score(q, y)})
            for pct in (5, 10):
                ratio = np.exp(y)
                rows.append({**base, 'model': f'Fixed +{pct}% contingency', 'actual_ratio': ratio,
                             'le_P95': float(ratio <= 1 + pct / 100),
                             'pinball95_pct': 100 * M.pinball(1 + pct / 100, ratio, 0.95)})
        pd.to_pickle({'settings': settings, 'rows': rows[n_rows:], 'fits': fits[n_fits:],
                      'tuning': tuning[n_tune:]}, f)
        print(f'{series} origin {origin:%Y-%m} done', flush=True)
    return pd.DataFrame(rows)


def summarise(det: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (det.groupby(by + ['model', 'h'])
            .agg(n=('actual_ratio', 'size'), cover_P5_P95=('in_P5_P95', 'mean'),
                 cover_le_P80=('le_P80', 'mean'), cover_le_P95=('le_P95', 'mean'),
                 pinball80_pct=('pinball80_pct', 'mean'), pinball95_pct=('pinball95_pct', 'mean'),
                 crps_pct=('crps_pct', 'mean'))
            .reset_index())


def dm_table(det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, h), g in det.groupby(['series', 'h']):
        w = g.pivot(index='origin', columns='model', values='crps_pct')
        for bench in BENCHMARKS:
            for m in w.columns:
                if m == bench or m.startswith('Fixed') or w[m].isna().all():
                    continue
                pair = w[[m, bench]].dropna()
                stat, p = dm_test((pair[m] - pair[bench]).to_numpy(), h)
                rows.append({'series': series, 'h': h, 'model': m, 'benchmark': bench, 'n': len(pair),
                             'crps_model': pair[m].mean(), 'crps_benchmark': pair[bench].mean(),
                             'skill_vs_benchmark': 1 - pair[m].mean() / pair[bench].mean(),
                             'dm_stat': stat, 'p_value': p})
    return pd.DataFrame(rows)


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for s in self.s:
            s.write(x)

    def flush(self):
        for s in self.s:
            s.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--series', nargs='+', default=['SD40 12mm'])
    ap.add_argument('--data-dir', type=Path, default=D.DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default='backtest_main')
    ap.add_argument('--paths', type=int, default=2000)
    ap.add_argument('--first-origin', type=pd.Timestamp, default=pd.Timestamp(FIRST_ORIGIN))
    ap.add_argument('--extra-lag', type=int, default=0, help='extra publication lag (months) for every driver')
    ap.add_argument('--driver-sets', nargs='*', default=['all drivers'], choices=list(RM.DRIVER_SETS),
                    help='driver feature sets added to own history in the direct models')
    ap.add_argument('--tune', choices=['fixed', 'cv', 'both'], default='fixed',
                    help='regularisation of the direct models: fixed a priori, time-series CV, or both')
    ap.add_argument('--quick', action='store_true', help='smoke test: every 12th origin, 300 paths')
    ap.add_argument('--shard', type=lambda s: tuple(int(x) for x in s.split('/')), default=None,
                    help="i/k: compute only every k-th origin starting at i and write checkpoints, no tables; "
                         "run again without --shard to assemble all origins")
    args = ap.parse_args()
    if args.quick:
        args.paths = 300
    here = Path(__file__).resolve().parent
    out = here / 'outputs' / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    log_name = f'run_log_shard{args.shard[0]}of{args.shard[1]}.txt' if args.shard else 'run_log.txt'
    log = open(out / log_name, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, log)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    timer = B.Timer()
    print(f'Args: {vars(args)} | {platform.platform()} | Python {platform.python_version()}')

    with timer('Data'):
        panel, _ = D.load_price_panel(args.data_dir, here / 'outputs' / 'cache' / 'panel.pkl')
        wide, _, _ = R.splice(R.select_rebar(panel))
        drivers = G.build_drivers(pd.date_range('2015-01-01', wide.index.max(), freq='MS'), args.data_dir)
        print(f'Drivers: {list(drivers.columns)} | Pink Sheet {G.load_pink_sheet().attrs["updated"]}')
    fits, tuning = [], []
    with timer('Backtest'):
        det = pd.concat([run_series(s, wide, drivers, args, fits, tuning, out / 'checkpoints')
                         for s in args.series], ignore_index=True)
    if args.shard:
        print(f'Shard {args.shard[0]}/{args.shard[1]} done; checkpoints in {out / "checkpoints"}')
        sys.stdout = sys.__stdout__
        log.close()
        return
    det.to_csv(out / 'backtest_detail.csv', index=False)
    fits = pd.DataFrame(fits)
    fits.to_csv(out / 'table_model_fits.csv', index=False)
    if tuning:
        tuning = pd.DataFrame(tuning)
        tuning.to_csv(out / 'table_cv_tuning.csv', index=False)
        print('CV-chosen penalty of the linear QR (share of origins):')
        print(tuning.groupby(['features', 'h'])['qr_alpha'].value_counts(normalize=True).unstack(fill_value=0).round(2).to_string())
        print('CV-chosen XGBoost depth x trees (share of origins):')
        print(tuning.groupby(['features', 'h'])[['xgb_max_depth', 'xgb_n_estimators']].value_counts(normalize=True)
              .unstack([2, 3], fill_value=0).round(2).to_string())
    if 'reason' in fits:
        print('Fits that fell back to GBM:', fits.groupby('model')['reason'].apply(lambda s: s.notna().sum()).to_dict())

    summ = summarise(det, ['series'])
    summ.to_csv(out / 'table_backtest_summary.csv', index=False)
    det['period'] = np.where(pd.to_datetime(det['origin']) < PERIOD_SPLIT, '2020-06..2022-12', '2023-01..')
    summarise(det, ['series', 'period']).to_csv(out / 'table_backtest_by_period.csv', index=False)
    dm = dm_table(det)
    dm.to_csv(out / 'table_dm_tests.csv', index=False)

    for s in args.series:
        print(f'\n##### {s}')
        t = summ[summ['series'] == s]
        print(t.pivot(index='model', columns='h', values='crps_pct').sort_values(1).round(3).to_string())
        print('Coverage P5-P95:')
        print(t.pivot(index='model', columns='h', values='cover_P5_P95').round(2).to_string())
        print('Coverage <= P95:')
        print(t.pivot(index='model', columns='h', values='cover_le_P95').round(2).to_string())
        d = dm[(dm['series'] == s) & (dm['model'].str.contains('QR') | dm['model'].str.startswith('AR(1)'))]
        print('DM tests (CRPS skill vs benchmark, p-value):')
        print(d.assign(cell=d['skill_vs_benchmark'].round(3).astype(str) + ' (' + d['p_value'].round(3).astype(str) + ')')
              .pivot(index=['benchmark', 'model'], columns='h', values='cell').to_string())
    rt = pd.DataFrame(timer.rows)
    print(rt.to_string(index=False))
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == '__main__':
    main()
