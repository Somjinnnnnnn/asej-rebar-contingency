"""Manuscript tables, assembled from the analysis outputs (no model is re-run).

Reads outputs/audit_2026-10, backtest_v2, backtest_robust_grades, backtest_robust_lag1,
projection_2026-09, validation_wpu1017_real and validation_wpu1017, and writes
outputs/manuscript/table_<n>_<name>.csv plus manuscript_tables.xlsx (one sheet per table).

Usage:
    python manuscript_tables.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import data as D

warnings.filterwarnings('ignore', message='Workbook contains no default style')
OUT = Path(__file__).resolve().parent / 'outputs'
MS = OUT / 'manuscript'
HORIZONS = [1, 3, 6, 12]

BACKTEST_MODELS = [  # (name in the outputs, label in the manuscript)
    ('Random walk', 'Random walk'),
    ('GBM (data-calibrated)', 'GBM (data-calibrated)'),
    ('Empirical h-month changes', 'Empirical h-month changes'),
    ('AR(1)', 'AR(1)'),
    ('GARCH(1,1)', 'GARCH(1,1)'),
    ('AR(1)-GARCH(1,1)', 'AR(1)-GARCH(1,1)'),
    ('Merton jump-diffusion', 'Merton jump-diffusion'),
    ('Markov regime-switching', 'Markov regime-switching'),
    ('Linear QR, own history, CV-tuned', 'Linear QR, own history'),
    ('Linear QR, + scrap+ore, CV-tuned', 'Linear QR, + scrap and iron ore'),
    ('Linear QR, + all drivers', 'Linear QR, + all drivers (fixed penalty)'),
    ('Linear QR, + all drivers, CV-tuned', 'Linear QR, + all drivers'),
    ('XGBoost QR, own history', 'XGBoost QR, own history (fixed)'),
    ('XGBoost QR, own history, CV-tuned', 'XGBoost QR, own history'),
    ('XGBoost QR, + scrap+ore, CV-tuned', 'XGBoost QR, + scrap and iron ore'),
    ('XGBoost QR, + all drivers, CV-tuned', 'XGBoost QR, + all drivers'),
]
CONFORMAL_MODELS = ['Random walk', 'XGBoost QR, own history, CV-tuned', 'XGBoost QR, + scrap+ore, CV-tuned']
VARIANTS = ['raw', 'expanding', 'rolling 24', 'symmetric, expanding', 'symmetric, rolling 24']


def label(m: str) -> str:
    return dict(BACKTEST_MODELS).get(m, m)


def t1_audit() -> pd.DataFrame:
    """Spliced series; diameters with identical statistics are merged into one row."""
    a = pd.read_csv(OUT / 'audit_2026-10' / 'table_audit.csv', encoding='utf-8-sig')
    a = a[~a['series'].str.startswith('SD40 10mm')]  # 22-month gap, not a main product
    a['grade'] = a['series'].str.extract(r'^(S[RD]\d{2})')[0]
    a['diam'] = a['series'].str.extract(r'(\d+)mm')[0].astype(int)
    key = ['grade', 'first', 'last', 'months', 'share_months_changed', 'annual_log_drift', 'annual_volatility',
           'largest_rise_pct', 'largest_rise_month', 'largest_fall_pct', 'largest_fall_month']
    a = a.round({'share_months_changed': 3, 'annual_log_drift': 3, 'annual_volatility': 3,
                 'largest_rise_pct': 1, 'largest_fall_pct': 1})
    g = (a.sort_values('diam').groupby(key, sort=False)['diam']
         .agg(lambda d: ', '.join(str(x) for x in d)).reset_index())
    g['diameters (mm)'] = g.pop('diam')
    g = g.sort_values(['grade', 'first', 'diameters (mm)'])
    t = pd.DataFrame({
        'Grade': g['grade'], 'Diameters (mm)': g['diameters (mm)'], 'Period': g['first'] + ' to ' + g['last'],
        'Months': g['months'], 'Months with a price change (%)': (100 * g['share_months_changed']).round(1),
        'Drift (%/yr, nominal)': (100 * g['annual_log_drift']).round(1),
        'Volatility (%/yr)': (100 * g['annual_volatility']).round(1),
        'Largest rise (%)': g['largest_rise_pct'].astype(str) + ' (' + g['largest_rise_month'] + ')',
        'Largest fall (%)': g['largest_fall_pct'].astype(str) + ' (' + g['largest_fall_month'] + ')'})
    # reference row: 180 ksc ready-mixed concrete (the companion study)
    panel, _ = D.load_price_panel(D.DEFAULT_DATA_DIR, OUT / 'cache' / 'panel.pkl')
    c = panel[panel['code'] == '0101010108100000'].set_index('date')['price'].sort_index()
    r = np.log(c).diff().dropna()
    t.loc[len(t)] = ['Ready-mixed concrete 180 ksc (reference)', '–', f'{c.index[0]:%Y-%m} to {c.index[-1]:%Y-%m}', len(c),
                     round(100 * (r != 0).mean(), 1), round(1200 * r.mean(), 1), round(100 * np.sqrt(12) * r.std(), 1),
                     f'{100 * np.expm1(r.max()):.1f} ({r.idxmax():%Y-%m})', f'{100 * np.expm1(r.min()):.1f} ({r.idxmin():%Y-%m})']
    return t


def t2_junctions() -> pd.DataFrame:
    j = pd.read_csv(OUT / 'audit_2026-10' / 'table_junction_checks.csv', parse_dates=['month'])
    j = j.dropna(subset=['junction_dlog'])
    g = j.groupby(j['month'].dt.strftime('%Y-%m'))
    return pd.DataFrame({
        'Series re-coded': g.size(),
        'Change at junction, median (%)': (100 * np.expm1(g['junction_dlog'].median())).round(2),
        'Range (%)': g['junction_dlog'].agg(lambda x: f'{100 * np.expm1(x.min()):.2f} to {100 * np.expm1(x.max()):.2f}'),
        'Benchmark': g['benchmark'].first().str.replace(r' \(.*\)', '', regex=True),
        'Benchmark change (%)': (100 * np.expm1(g['benchmark_dlog'].median())).round(2),
        'Flagged (> 2 log points)': g['flag'].sum().astype(int)}).reset_index(names='Month')


def t3_diagnostics() -> pd.DataFrame:
    d = pd.read_csv(OUT / 'audit_2026-10' / 'table_process_diagnostics.csv')
    w = pd.read_csv(OUT / 'audit_2026-10' / 'data_rebar_series.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    d = d[d['series'].isin(['SD40 12mm', 'SD50 12mm', 'SR24 12mm'])].copy()
    d['acf1'] = [np.log(w[s]).diff().autocorr(1) for s in d['series']]
    return pd.DataFrame({
        'Series': d['series'].str.replace('mm', ' mm'), 'Months': d['months'],
        'Unchanged months (%)': (100 * d['share_unchanged_months']).round(1),
        'Lag-1 autocorrelation': d['acf1'].round(2), 'Jarque-Bera p': d['jarque_bera_p'].map('{:.3g}'.format),
        'Ljung-Box(12) p': d['ljung_box_12_p'].round(3), 'Ljung-Box(12) on squares p': d['ljung_box_sq_12_p'].round(3),
        'ARCH-LM(12) p': d['arch_lm_12_p'].round(3), 'ADF (log price) p': d['adf_logprice_p'].round(3)})


def _skill_cells(summ, dm, models, series='SD40 12mm'):
    s = summ[summ['series'] == series].set_index(['model', 'h'])['crps_pct']
    d = dm[(dm['series'] == series) & (dm['benchmark'] == 'Random walk')].set_index(['model', 'h'])
    rows = []
    for m, lab in models:
        row = {'Model': lab}
        for h in HORIZONS:
            if (m, h) not in s.index:
                row[f'h = {h}'] = ''
                continue
            cell = f'{s[(m, h)]:.2f}'
            if m != 'Random walk' and (m, h) in d.index:
                sk, p = d.loc[(m, h), 'skill_vs_benchmark'], d.loc[(m, h), 'p_value']
                cell += f' ({100 * sk:+.1f}%, p = {p:.3f})'
            row[f'h = {h}'] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def t4_backtest_crps() -> pd.DataFrame:
    run = OUT / 'backtest_v2'
    return _skill_cells(pd.read_csv(run / 'table_backtest_summary.csv'), pd.read_csv(run / 'table_dm_tests.csv'),
                        BACKTEST_MODELS)


def t5_coverage() -> pd.DataFrame:
    run = OUT / 'backtest_v2'
    s = pd.read_csv(run / 'table_backtest_summary.csv')
    s = s[s['series'] == 'SD40 12mm']
    keep = [m for m, _ in BACKTEST_MODELS] + ['Fixed +5% contingency', 'Fixed +10% contingency']
    s = s[s['model'].isin(keep)]
    rows = []
    for m in keep:
        g = s[s['model'] == m].set_index('h')
        row = {'Model': label(m)}
        for h in HORIZONS:
            p90 = g.at[h, 'cover_P5_P95']
            row[f'P5–P95, h = {h}'] = '' if pd.isna(p90) else round(p90, 2)
        for h in HORIZONS:
            row[f'≤ P95, h = {h}'] = round(g.at[h, 'cover_le_P95'], 2)
        rows.append(row)
    p = pd.read_csv(run / 'table_backtest_by_period.csv')
    p = p[(p['series'] == 'SD40 12mm') & (p['model'] == 'Random walk')]
    for per, g in p.groupby('period'):
        g = g.set_index('h')
        rows.append({'Model': f'Random walk, origins {per}',
                     **{f'P5–P95, h = {h}': round(g.at[h, 'cover_P5_P95'], 2) for h in HORIZONS},
                     **{f'≤ P95, h = {h}': round(g.at[h, 'cover_le_P95'], 2) for h in HORIZONS}})
    return pd.DataFrame(rows)


def t6_robustness() -> pd.DataFrame:
    """CRPS skill vs the random walk of the Machine Learning models in the round-1 runs
    (fixed penalties): other grades and one more month of driver publication lag."""
    rows = []
    runs = [('backtest_main', 'SD40 12mm', 'SD40 12 mm, main timing'),
            ('backtest_robust_lag1', 'SD40 12mm', 'SD40 12 mm, drivers lagged one more month'),
            ('backtest_robust_grades', 'SD50 12mm', 'SD50 12 mm'),
            ('backtest_robust_grades', 'SR24 12mm', 'SR24 12 mm')]
    for run, series, lab in runs:
        dm = pd.read_csv(OUT / run / 'table_dm_tests.csv')
        dm = dm[(dm['series'] == series) & (dm['benchmark'] == 'Random walk')]
        # the all-driver model is named '+ drivers' in the round-1 outputs and '+ all drivers'
        # when the round-1 commands are rerun with the current backtest_rebar.py
        dm['model'] = dm['model'].str.replace('+ all drivers', '+ drivers', regex=False)
        for m in ['XGBoost QR, own history', 'XGBoost QR, + drivers', 'AR(1)-GARCH(1,1)']:
            g = dm[dm['model'] == m].set_index('h')
            rows.append({'Run': lab, 'Model': m.replace('+ drivers', '+ all drivers'),
                         **{f'h = {h}': f'{100 * g.at[h, "skill_vs_benchmark"]:+.1f}% (p = {g.at[h, "p_value"]:.3f})'
                            for h in HORIZONS}})
    return pd.DataFrame(rows)


def t7_conformal() -> pd.DataFrame:
    run = OUT / 'backtest_v2'
    c = pd.read_csv(run / 'table_conformal_summary.csv')
    c = c[c['model'].isin(CONFORMAL_MODELS) & (c['series'] == 'SD40 12mm')]
    rows = []
    for m in CONFORMAL_MODELS:
        for v in VARIANTS:
            g = c[(c['model'] == m) & (c['variant'] == v)].set_index('h')
            rows.append({'Model': label(m), 'Recalibration': v,
                         **{f'CRPS, h = {h}': round(g.at[h, 'crps_pct'], 2) for h in HORIZONS},
                         **{f'P5–P95, h = {h}': round(g.at[h, 'cover_P5_P95'], 2) for h in HORIZONS},
                         **{f'≤ P95, h = {h}': round(g.at[h, 'cover_le_P95'], 2) for h in HORIZONS}})
    return pd.DataFrame(rows)


def t7b_median_error() -> pd.DataFrame:
    """Why conformal recalibration fails here: the random walk's median error changes sign
    between the calibration and the evaluation periods."""
    det = pd.read_csv(OUT / 'backtest_v2' / 'backtest_detail.csv')
    rw = det[(det['model'] == 'Random walk') & (det['series'] == 'SD40 12mm')].copy()
    rw['err'] = 100 * (rw['y_log'] - rw['q50'])
    rw['period'] = np.where(pd.to_datetime(rw['origin']) < '2022-05-01', 'Origins 2020-06 to 2022-04',
                            'Origins 2022-05 on')
    t = rw.pivot_table(index='period', columns='h', values='err', aggfunc='mean').round(2)
    t.columns = [f'h = {h}' for h in t.columns]
    return t.reset_index(names='Random walk: mean of (outcome − median forecast), log %')


def t8_bands() -> pd.DataFrame:
    c = pd.read_csv(OUT / 'projection_2026-09' / 'table_contingency.csv')
    pick = [('real', 'none', 'none', 'Real (THB of Sep 2026)'),
            ('nominal', '2 %', 'none', 'Nominal THB, 2 % inflation'),
            ('nominal', 'stochastic AR(1)', 'none', 'Nominal THB, stochastic inflation'),
            ('nominal', '1.34 % (2017-2026)', 'none', 'Nominal THB, 1.34 % inflation (2017–2026)'),
            ('USD', '2 %', 'stochastic AR(1)', 'USD, 2 % inflation, stochastic FX')]
    rows = []
    for role in ('headline', 'comparator'):
        for basis, infl, fx, lab in pick:
            g = c[(c['role'] == role) & (c['basis'] == basis) & (c['inflation'] == infl) & (c['fx'] == fx)].set_index('year')
            for y in [1, 5, 10, 20]:
                rows.append({'Band': f'{role}: {g["model"].iloc[0]}', 'Basis': lab, 'Years': y,
                             'P50 (%)': round(g.at[y, 'P50_pct'], 1), 'P80 (%)': round(g.at[y, 'P80_pct'], 1),
                             'P90 (%)': round(g.at[y, 'P90_pct'], 1), 'P95 (%)': round(g.at[y, 'P95_pct'], 1),
                             'P50': round(g.at[y, 'P50']), 'P80': round(g.at[y, 'P80']), 'P95': round(g.at[y, 'P95']),
                             'Unit': g.at[y, 'unit']})
    return pd.DataFrame(rows)


def t9_model_risk() -> pd.DataFrame:
    p = pd.read_csv(OUT / 'projection_2026-09' / 'table_projection_long.csv')
    p = p[(p['basis'] == 'real') & (p['year'].isin([5, 20]))]
    rows = []
    for m, g in p.groupby('model', sort=False):
        row = {'Model': m}
        for drift in ('as fitted', 'zero real drift'):
            for y in (5, 20):
                r = g[(g['drift'] == drift) & (g['year'] == y)].iloc[0]
                row[f'{drift}, {y} yr: P5 / P50 / P95 (%)'] = f'{r.P5_pct:.0f} / {r.P50_pct:.0f} / {r.P95_pct:.0f}'
        rows.append(row)
    return pd.DataFrame(rows)


def t10_validation() -> pd.DataFrame:
    rows = []
    for run, lab in [('validation_wpu1017_real', 'real (deflated by US CPI)'), ('validation_wpu1017', 'nominal')]:
        s = pd.read_csv(OUT / run / 'table_validation_summary.csv')
        for (m, drift), g in s.groupby(['model', 'drift'], sort=False):
            g = g.set_index('h')
            rows.append({'Terms': lab, 'Model': m, 'Drift': drift,
                         **{f'CRPS, {h} m': round(g.at[h, 'crps_pct'], 1) for h in (12, 36, 60)},
                         **{f'P5–P95, {h} m': round(g.at[h, 'cover_P5_P95'], 2) for h in (12, 36, 60)},
                         **{f'≤ P95, {h} m': round(g.at[h, 'cover_le_P95'], 2) for h in (12, 36, 60)}})
    return pd.DataFrame(rows)


TABLES = [('01_data_audit', t1_audit), ('02_code_junctions', t2_junctions), ('03_process_diagnostics', t3_diagnostics),
          ('04_backtest_crps', t4_backtest_crps), ('05_backtest_coverage', t5_coverage),
          ('06_robustness', t6_robustness), ('07_conformal', t7_conformal), ('07b_median_error_by_period', t7b_median_error),
          ('08_contingency_bands', t8_bands),
          ('09_model_risk_long_horizon', t9_model_risk), ('10_validation_us_steel_ppi', t10_validation)]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
    MS.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(MS / 'manuscript_tables.xlsx') as xl:
        for name, fn in TABLES:
            t = fn()
            t.to_csv(MS / f'table_{name}.csv', index=False, encoding='utf-8-sig')
            t.to_excel(xl, sheet_name=name[:31], index=False)
            print(f'\n=== Table {name} ({len(t)} rows)')
            print(t.head(12).to_string(index=False))


if __name__ == '__main__':
    main()
