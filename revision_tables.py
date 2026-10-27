"""Tables added in the October 2026 revision (main text and supplementary material).

Reads outputs/revision_2026-10 (revision_analyses.py) and the earlier manuscript tables, and writes
to outputs/manuscript:
  table_rev_primary_tests.csv        primary one-month comparisons with Holm-adjusted p-values
  table_rev_uncertainty.csv          CRPS skill and P95 coverage with 90% stationary-bootstrap intervals
  table_rev_conformal_compact.csv    conformal recalibration, CRPS only, random walk and tuned XGBoost
  table_rev_bands_compact.csv        headline and comparator P80 / P95 at 1, 5, 10 and 20 years
  table_rev_validation_compact.csv   US steel PPI validation, real terms, zero drift
  table_rev_sensitivity.csv          sensitivity and uncertainty of the headline P95
  table_rev_procurement.csv          worked procurement example (expenditure allowances)
  table_S01_code_chain.csv ... table_S10_validation_block_length.csv   supplementary tables

Usage:
    python revision_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
REV = OUT / 'revision_2026-10'
MS = OUT / 'manuscript'
SHORT = {'Random walk': 'Random walk', 'XGBoost QR, own history, CV-tuned': 'XGBoost QR, own history',
         'XGBoost QR, + all drivers, CV-tuned': 'XGBoost QR, + all drivers',
         'XGBoost QR, + scrap+ore, CV-tuned': 'XGBoost QR, + scrap and iron ore',
         'Linear QR, own history, CV-tuned': 'Linear QR, own history', 'AR(1)-GARCH(1,1)': 'AR(1)-GARCH(1,1)',
         'GBM (data-calibrated)': 'GBM (data-calibrated)', 'Fixed +5% contingency': 'Fixed +5% allowance',
         'Fixed +10% contingency': 'Fixed +10% allowance'}


def pct(x, d=1, sign=True):
    return '' if pd.isna(x) else (f'{x:+.{d}f}' if sign else f'{x:.{d}f}')


def write(df, name):
    df.to_csv(MS / f'{name}.csv', index=False, encoding='utf-8-sig')


def primary():
    p = pd.read_csv(REV / 'table_primary_tests.csv')
    t = pd.DataFrame({'Comparison (one month)': p.hypothesis.str.replace('Machine Learning vs random walk',
                                                                         'XGBoost QR, own history vs random walk')
                      .str.replace('learner: XGBoost vs linear', 'XGBoost QR vs linear QR, own history')
                      .str.replace('drivers: all vs own history', 'XGBoost QR, + all drivers vs own history')
                      .str.replace('drivers: scrap+ore vs own history', 'XGBoost QR, + scrap and iron ore vs own history'),
                      'CRPS skill (%)': (100 * p.skill).map(pct), 'DM p': p.p_value.map(lambda x: f'{x:.3f}'),
                      'Holm-adjusted p': p.p_holm.map(lambda x: f'{x:.3f}')})
    write(t, 'table_rev_primary_tests')


def uncertainty():
    u = pd.read_csv(REV / 'table_backtest_uncertainty.csv')
    models = ['Random walk', 'AR(1)-GARCH(1,1)', 'Linear QR, own history, CV-tuned', 'XGBoost QR, own history, CV-tuned',
              'XGBoost QR, + all drivers, CV-tuned', 'Fixed +5% contingency', 'Fixed +10% contingency']
    rows = []
    for m in models:
        for h in (1, 3, 6, 12):
            x = u[(u.model == m) & (u.h == h)].iloc[0]
            rows.append({'Model': SHORT[m], 'Horizon (months)': str(h),
                         'CRPS skill vs random walk, % (90% interval)': (
                             '–' if m == 'Random walk' or m.startswith('Fixed') else
                             f'{100 * x.skill:+.1f} ({100 * x.skill_lo:+.1f} to {100 * x.skill_hi:+.1f})'),
                         'Share of outcomes ≤ P95 (90% interval)': f'{x.le_P95:.2f} ({x.le_P95_lo:.2f}–{x.le_P95_hi:.2f})'})
    write(pd.DataFrame(rows), 'table_rev_uncertainty')


def conformal_compact():
    c = pd.read_csv(MS / 'table_07_conformal.csv', encoding='utf-8-sig')
    c = c[c.Model.isin(['Random walk', 'XGBoost QR, own history'])]
    keep = ['Model', 'Recalibration'] + [f'CRPS, h = {h}' for h in (1, 3, 6, 12)]
    write(c[keep], 'table_rev_conformal_compact')


def bands_compact():
    b = pd.read_csv(MS / 'table_08_contingency_bands.csv', encoding='utf-8-sig')
    b['role'] = np.where(b.Band.str.startswith('headline'), 'Headline (block bootstrap)', 'Comparator (random walk)')
    rows = []
    for role in ('Headline (block bootstrap)', 'Comparator (random walk)'):
        for basis in ('Real (THB of Sep 2026)', 'Nominal THB, 2 % inflation', 'USD, 2 % inflation, stochastic FX'):
            s = b[(b.role == role) & (b.Basis == basis)].set_index('Years')
            if s.empty:
                continue
            r = {'Band': role, 'Basis': basis.replace('2 %', '2%')}
            for q in ('P80', 'P95'):
                for y in (1, 5, 10, 20):
                    r[f'{q} (% above the current price), {y} yr'] = pct(float(s.loc[y, f'{q} (%)']))
            rows.append(r)
    write(pd.DataFrame(rows), 'table_rev_bands_compact')


def validation_compact():
    v = pd.read_csv(REV.parent / 'validation_wpu1017_real' / 'table_validation_summary.csv')
    v = v[v.drift == 'zero drift']
    order = ['Block bootstrap (demeaned)', 'Random walk (driftless)', 'AR(1)-GARCH(1,1)', 'GARCH(1,1)', 'GBM (fitted drift)',
             'Merton jump-diffusion', 'Markov regime-switching', 'Mean-reverting AR(1) level']
    rows = []
    for m in order:
        s = v[v.model == m].set_index('h')
        r = {'Model (zero drift)': m.replace(' (fitted drift)', '').replace(' (driftless)', '')}
        for h in (12, 36, 60):
            r[f'Share ≤ P95, {h} m'] = f'{s.loc[h, "cover_le_P95"]:.2f}'
        for h in (12, 36, 60):
            r[f'CRPS (%), {h} m'] = f'{s.loc[h, "crps_pct"]:.1f}'
        rows.append(r)
    write(pd.DataFrame(rows), 'table_rev_validation_compact')


def sensitivity():
    s = pd.read_csv(REV / 'table_sensitivity_headline.csv')
    u = pd.read_csv(REV / 'table_uncertainty_sources.csv')
    vb = pd.read_csv(REV / 'table_validation_block_length.csv')
    rows = []

    def add(label, sub, cov=None):
        r = {'Specification': label}
        for y in (1, 5, 10, 20):
            r[f'Real P95 (% above the current price), {y} yr'] = pct(sub[sub.year == y].P95_pct.iloc[0], 0)
        r['US PPI validation: share ≤ P95 at 60 m'] = cov if cov is not None else '–'
        rows.append(r)

    def vcov(L):
        x = vb[(vb.mean_block == L) & (vb.h == 60)]
        return f'{x.le_P95.iloc[0]:.2f}' if len(x) else '–'
    add('Headline: block bootstrap, mean block 12 months', s[s.variant.str.startswith('headline')], vcov(12))
    for L in (3, 6, 24, 36):
        add(f'Mean block {L} months', s[s.variant == f'mean block {L} months'], vcov(L))
    pw = s[s.variant.str.contains('Politis-White')]
    add(f'Mean block {pw.variant.iloc[0].split()[2]} months (Politis and White, 2004)', pw)
    add('Excluding the 2020-06 to 2022-12 changes', s[s.variant == 'excluding the 2020-06 to 2022-12 changes'])
    add('Excluding the code-junction months', s[s.variant == 'excluding the code-junction months'])
    b = pd.read_csv(MS / 'table_08_contingency_bands.csv', encoding='utf-8-sig')
    b = b[b.Band.str.startswith('comparator') & (b.Basis == 'Real (THB of Sep 2026)')].set_index('Years')
    rw = {'Specification': 'Comparator: driftless random walk'}
    for y in (1, 5, 10, 20):
        rw[f'Real P95 (% above the current price), {y} yr'] = pct(float(b.loc[y, 'P95 (%)']), 0)
    v = pd.read_csv(OUT / 'validation_wpu1017_real' / 'table_validation_summary.csv')
    v = v[(v.model == 'Random walk (driftless)') & (v.drift == 'zero drift') & (v.h == 60)]
    rw['US PPI validation: share ≤ P95 at 60 m'] = f'{v.cover_le_P95.iloc[0]:.2f}'
    rows.append(rw)
    rr = {'Specification': 'Monte Carlo range, 20 seeds x 10,000 paths'}
    for y in (1, 5, 10, 20):
        x = u[(u.source.str.startswith('Monte')) & (u.year == y)].iloc[0]
        rr[f'Real P95 (% above the current price), {y} yr'] = f'{x.P95_lo:.0f} to {x.P95_hi:.0f}'
    rr['US PPI validation: share ≤ P95 at 60 m'] = '–'
    rows.append(rr)
    for model, label in (('block bootstrap', 'Estimation, block bootstrap: median (5th–95th percentile)'),
                         ('random walk', 'Estimation, random walk: median (5th–95th percentile)')):
        rr = {'Specification': label}
        for y in (1, 5, 10, 20):
            x = u[(u.source.str.startswith('estimation')) & (u.model == model) & (u.year == y)].iloc[0]
            rr[f'Real P95 (% above the current price), {y} yr'] = f'{x.P95_center:.0f} ({x.P95_lo:.0f}–{x.P95_hi:.0f})'
        rr['US PPI validation: share ≤ P95 at 60 m'] = '–'
        rows.append(rr)
    write(pd.DataFrame(rows), 'table_rev_sensitivity')


def procurement():
    p = pd.read_csv(REV / 'table_procurement_example.csv')
    p = p[p.basis.str.startswith('nominal')]
    rows = []
    for _, x in p.iterrows():
        rows.append({'Schedule': 'A: 24 months, 8 lots, 600 t' if x.schedule.startswith('A') else 'B: 20 years, 20 lots, 600 t',
                     'Model': 'Block bootstrap' if x.model.startswith('Block') else 'Random walk',
                     'Expenditure P50 (%)': pct(x.P50), 'Expenditure P80 (%)': pct(x.P80), 'Expenditure P95 (%)': pct(x.P95),
                     'Pr(≤ +5%)': f'{x.prob_within_5:.2f}', 'Pr(≤ +10%)': f'{x.prob_within_10:.2f}',
                     'Quantity-weighted P95 of the purchase prices (%)': pct(x.weighted_per_date_P95),
                     'P95 at the last purchase (%)': pct(x.endpoint_P95)})
    t = pd.DataFrame(rows).sort_values(['Schedule', 'Model'], ascending=[True, True])
    write(t, 'table_rev_procurement')


# --------------------------------------------------------------------------- supplementary

def code_chain():
    m = pd.read_csv(OUT / 'audit_2026-10' / 'table_code_map.csv', encoding='utf-8-sig', dtype={'code': str})
    j = pd.read_csv(OUT / 'audit_2026-10' / 'table_junction_checks.csv', encoding='utf-8-sig', dtype={'code_in': str})
    rows = []
    for series in ('SD40 12mm', 'SD50 12mm', 'SR24 12mm'):
        for _, x in m[m.series == series].sort_values('first').iterrows():
            item = x['item']
            desc = ('Plain round bar' if 'ผิวเรียบ' in item else 'Deformed bar') + f' {x.grade}, {int(x.diam)} mm, 10 m'
            if x.tempcore:
                desc += ', quenched and tempered (Tempcore)'
            if 'มอก.' in item:
                desc += ', TIS ' + item.split('มอก.')[1].split()[0]
            jj = j[(j.series == series) & (j.code_in.str.lstrip('0') == x.code.lstrip('0'))]
            rows.append({'Series': series.replace('12mm', '12 mm'), 'TPSO code': x.code, 'Listing (translated)': desc,
                         'Unit': 'ton', 'First': x['first'][:7], 'Last': x['last'][:7], 'Months': int(x.months),
                         'Junction change (%)': pct(100 * jj.junction_dlog.iloc[0], 2) if len(jj) else '–',
                         'Benchmark change (%)': pct(100 * jj.benchmark_dlog.iloc[0], 2) if len(jj) else '–',
                         'Flagged': ('yes' if jj.flag.iloc[0] else 'no') if len(jj) else '–'})
    write(pd.DataFrame(rows), 'table_S01_code_chain')


def release_timing():
    rows = [
        ('TPSO construction material prices (rebar list price)', 'TPSO', 'month t', 'during month t + 1 (assumed; release '
         'dates of individual reports not recorded)', '0', 'as published'),
        ('Bangkok CPI (deflator, RQ3 only)', 'TPSO', 'month t', 'early month t + 1', 'not a forecast input', 'as published'),
        ('Iron ore, coal, Brent (Pink Sheet)', 'World Bank', 'month t', 'first working days of month t + 1', '0',
         'release of 2 October 2026; recent months can be revised'),
        ('THB/USD mid rate', 'Bank of Thailand', 'daily, averaged over month t', 'end of month t', '0', 'as published'),
        ('Bangkok retail diesel (HSD B7)', 'EPPO', 'daily, averaged over month t', 'end of month t', '0', 'as published'),
        ('US PPI steel mill products, iron and steel scrap', 'BLS via FRED', 'month t', 'mid-month t + 1', '1',
         'vintage of 22 October 2026; revised for four months after first release'),
        ('US CPI (validation deflator only)', 'BLS via FRED', 'month t', 'mid-month t + 1', 'not a forecast input',
         'vintage of 22 October 2026'),
    ]
    write(pd.DataFrame(rows, columns=['Information', 'Source', 'Reference period', 'Available', 'Lag at origin t (months)',
                                      'Vintage used']), 'table_S02_release_timing')


def supplementary_copies():
    d = pd.read_csv(REV / 'table_dm_adjusted.csv')
    t = pd.DataFrame({'Horizon': d.h.map(lambda h: f'Horizon {h} month' + ('s' if h > 1 else '')),
                      'Model': d.model.str.replace(', CV-tuned', ' (tuned)').str.replace('+ scrap+ore', '+ scrap and iron ore'),
                      'Horizon (months)': d.h, 'CRPS skill vs random walk (%)': (100 * d.skill_vs_benchmark).map(pct),
                      'DM p': d.p_value.map(lambda x: f'{x:.3f}'), 'BH-adjusted p': d.p_bh_all.map(lambda x: f'{x:.3f}'),
                      'Holm-adjusted p': d.p_holm_all.map(lambda x: f'{x:.3f}')})
    write(t.sort_values(['Horizon (months)', 'Model']).drop(columns=['Horizon (months)']), 'table_S03_dm_adjusted')
    n = pd.read_csv(REV / 'table_backtest_nonoverlap.csv')
    n = n[n.h > 1]
    t = pd.DataFrame({'Model': n.model.map(SHORT), 'Horizon (months)': n.h, 'Subsamples': n.offsets,
                      'Origins per subsample': n.n_per_subsample.str.replace('-', '–'),
                      'CRPS skill vs random walk (%): mean (range)': [
                          '–' if pd.isna(a) else f'{100 * a:+.1f} ({100 * b:+.1f} to {100 * c:+.1f})'
                          for a, b, c in zip(n.skill_mean, n.skill_min, n.skill_max)],
                      'Share ≤ P95: mean (range)': [f'{a:.2f} ({b:.2f}–{c:.2f})' for a, b, c in
                                                     zip(n.le_P95_mean, n.le_P95_min, n.le_P95_max)]})
    write(t, 'table_S04_nonoverlap')
    g = pd.read_csv(REV / 'table_crps_grid.csv')
    t = pd.DataFrame({'Model': g.model, 'Horizon (months)': g.h, 'CRPS, 19 quantiles (%)': g.crps19.map(lambda x: f'{x:.2f}'),
                      'CRPS, 999 quantiles (%)': g.crps999.map(lambda x: f'{x:.2f}'),
                      'Skill, 19 quantiles (%)': (100 * g.skill19).map(pct), 'Skill, 999 quantiles (%)': (100 * g.skill999).map(pct)})
    write(t, 'table_S05_crps_grid')
    for src, dst in (('table_05_backtest_coverage', 'table_S06_coverage_full'), ('table_07_conformal', 'table_S07_conformal_full'),
                     ('table_08_contingency_bands', 'table_S08_bands_full'), ('table_10_validation_us_steel_ppi', 'table_S09_validation_full')):
        write(pd.read_csv(MS / f'{src}.csv', encoding='utf-8-sig', dtype=str), dst)
    vb = pd.read_csv(REV / 'table_validation_block_length.csv')
    t = pd.DataFrame({'Mean block (months)': vb.mean_block, 'Horizon (months)': vb.h,
                      'Share ≤ P95': vb.le_P95.map(lambda x: f'{x:.2f}'), 'Share in P5–P95': vb.in_P5_P95.map(lambda x: f'{x:.2f}'),
                      'CRPS (%)': vb.crps_pct.map(lambda x: f'{x:.2f}')})
    write(t, 'table_S10_validation_block_length')


# --------------------------------------------------------------------------- second review (revision2_analyses.py)

BOOT = 'Block bootstrap (demeaned), mean block 12'


def crps_with_bootstrap():
    """Table 4 with the headline block bootstrap added after the process models."""
    t = pd.read_csv(MS / 'table_04_backtest_crps.csv', encoding='utf-8-sig', dtype=str)
    b = pd.read_csv(REV / 'table_boot_backtest.csv')
    b = b[b.model == BOOT].set_index('h')
    row = {'Model': 'Block bootstrap (demeaned)'}
    for h in (1, 3, 6, 12):
        x = b.loc[h]
        row[f'h = {h}'] = f'{x.crps_pct:.2f} ({100 * x.skill:+.1f}%, p = {x.dm_p:.3f})'
    i = t.index[t.Model == 'Markov regime-switching'][0] + 1
    t = pd.concat([t.iloc[:i], pd.DataFrame([row]), t.iloc[i:]], ignore_index=True)
    write(t, 'table_rev_crps')


def uncertainty_with_bootstrap():
    u = pd.read_csv(MS / 'table_rev_uncertainty.csv', encoding='utf-8-sig', dtype=str)
    b = pd.read_csv(REV / 'table_boot_backtest.csv')
    b = b[b.model == BOOT]
    rows = [{'Model': 'Block bootstrap (demeaned)', 'Horizon (months)': str(x.h),
             'CRPS skill vs random walk, % (90% interval)': f'{100 * x.skill:+.1f} ({100 * x.skill_lo:+.1f} to {100 * x.skill_hi:+.1f})',
             'Share of outcomes ≤ P95 (90% interval)': f'{x.le_P95:.2f} ({x.le_P95_lo:.2f}–{x.le_P95_hi:.2f})'}
            for x in b.itertuples()]
    i = u.index[u.Model == 'Random walk'][-1] + 1
    write(pd.concat([u.iloc[:i], pd.DataFrame(rows), u.iloc[i:]], ignore_index=True), 'table_rev_uncertainty')


def protection():
    p = pd.read_csv(REV / 'table_allowance_protection.csv')
    rows = []
    for _, x in p[p.model == 'fixed'].iterrows():
        rows.append({'Allowance': f'Fixed {x.percentile}', 'Mean allowance (%)': f'{x.mean_allowance_pct:.0f}',
                     'Protection, all 69 origins': f'{x.protection:.2f}', 'Protection, origins to 2022-12': '–'})
    for model in ('random walk', 'block bootstrap'):
        for lv in ('80', '95', '99'):
            a = p[(p.model == model) & (p.percentile.astype(str).str.replace('.0', '', regex=False) == lv)]
            rows.append({'Allowance': f'{model.capitalize()}, model P{lv}',
                         'Mean allowance (%)': f'{a[a.period == "all origins"].mean_allowance_pct.iloc[0]:.1f}',
                         'Protection, all 69 origins': f'{a[a.period == "all origins"].protection.iloc[0]:.2f}',
                         'Protection, origins to 2022-12': f'{a[a.period == "origins to 2022-12"].protection.iloc[0]:.2f}'})
        # the operational procedure of the September 2026 allowances: real fit with 2% inflation
        pr = pd.read_csv(REV / 'table_procedure_protection.csv')
        pr = pr[(pr.model == model) & (pr.procedure == 'real fit, 2% inflation')].set_index('period')
        rows.append({'Allowance': f'{model.capitalize()}, model P95, real fit and 2% inflation',
                     'Mean allowance (%)': f'{pr.loc["all origins", "mean_P95_pct"]:.1f}',
                     'Protection, all 69 origins': f'{pr.loc["all origins", "protection"]:.2f}',
                     'Protection, origins to 2022-12': f'{pr.loc["origins to 2022-12", "protection"]:.2f}'})
        k = p[(p.model == model) & (p.percentile == 'P95 x k')].iloc[0]
        rows.append({'Allowance': f'{model.capitalize()}, model P95 x {k.k:.2f} (retrospective scaling)', 'Mean allowance (%)': f'{k.mean_allowance_pct:.1f}',
                     'Protection, all 69 origins': f'{k.protection:.2f}', 'Protection, origins to 2022-12': '–'})
    write(pd.DataFrame(rows), 'table_rev_protection')
    s = pd.read_csv(REV / 'table_allowance_start.csv')
    write(pd.DataFrame({'First origin': s.first_origin, 'Minimum history (months)': s.min_history_months,
                        'Origins (to 2022-12)': [f'{a} ({b})' for a, b in zip(s.n, s.n_to_2022_12)],
                        'Model': s.model.str.capitalize(), 'Mean P95 allowance (%)': s.mean_P95_pct.map(lambda x: f'{x:.1f}'),
                        'Protection by P95': s.protection_P95.map(lambda x: f'{x:.2f}'),
                        'Scaling factor k': s.k.map(lambda x: f'{x:.2f}'),
                        'Protection by P95 x k': s.protection_k.map(lambda x: f'{x:.2f}')}), 'table_S14_allowance_start')


def profiles():
    p = pd.read_csv(REV / 'table_procurement_profiles.csv')
    order = ['front-loaded', 'schedule A (S-curve)', 'even', 'back-loaded']
    rows = []
    for model in ('block bootstrap', 'random walk'):
        for s in order:
            x = p[(p.model == model) & (p.schedule == s)].iloc[0]
            rows.append({'Model': model.capitalize(), 'Purchase profile': s.replace('schedule A (S-curve)', 'S-curve (schedule A)'),
                         'Mean purchase month': f'{x.mean_month:.1f}', 'Expenditure P80 (%)': pct(x.P80), 'Expenditure P95 (%)': pct(x.P95),
                         'P95 retained with a ±5% clause (%)': pct(x.retained_P95_with_clause),
                         'P95 passed on under the clause (%)': f'{x.passed_on_P95_with_clause:.1f}'})
    write(pd.DataFrame(rows), 'table_rev_profiles')


def round2_supplementary():
    d = pd.read_csv(REV / 'table_dependence.csv')
    c = pd.read_csv(REV / 'table_dependence_corr.csv')
    t = pd.DataFrame({'Resampling': d.resampling.str.capitalize(), 'Years': d.year,
                      'Real P95 (%)': d.real_P95.map(lambda x: pct(x)), 'Nominal P95 (%)': d.nominal_P95.map(lambda x: pct(x)),
                      'USD P95 (%)': d.usd_P95.map(lambda x: pct(x))})
    write(t, 'table_S11_dependence')
    write(pd.DataFrame({'Monthly changes': c.pair.str.replace('rebar-infl', 'real rebar price and Bangkok CPI')
                        .str.replace('rebar-fx', 'real rebar price and THB/USD'),
                        'Correlation': c.correlation.map(lambda x: f'{x:.2f}')}), 'table_S11b_dependence_corr')
    th = pd.read_csv(REV / 'table_threshold_basis.csv')
    write(pd.DataFrame({'Series': th.series.str.replace('mm', ' mm'), 'Months': th.n_months,
                        'Median': th.p50.map(lambda x: f'{x:.2f}'), 'P90': th.p90.map(lambda x: f'{x:.2f}'),
                        'P95': th.p95.map(lambda x: f'{x:.2f}'), 'P99': th.p99.map(lambda x: f'{x:.2f}'),
                        'Maximum': th['max'].map(lambda x: f'{x:.2f}'), 'Share above 2': th.share_gt_2.map(lambda x: f'{x:.3f}')}),
          'table_S12_threshold')
    b = pd.read_csv(REV / 'table_boot_backtest.csv')
    write(pd.DataFrame({'Mean block': b.model.str.extract(r'mean block ([\d.]+)')[0] + ' months', 'Horizon (months)': b.h,
                        'CRPS (%)': b.crps_pct.map(lambda x: f'{x:.2f}'),
                        'Skill vs random walk, % (90% interval)': [f'{100 * a:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})'
                                                                  for a, lo, hi in zip(b.skill, b.skill_lo, b.skill_hi)],
                        'DM p': b.dm_p.map(lambda x: f'{x:.3f}'), 'Share ≤ P95': b.le_P95.map(lambda x: f'{x:.2f}'),
                        'Share in P5–P95': b.in_P5_P95.map(lambda x: f'{x:.2f}')}), 'table_S13_boot_backtest')


LITERATURE = [  # characteristics as stated in the abstracts of the cited studies; '–' = not addressed there
    ('Ashuri and Lu (2010)', 'ENR construction cost index', 'Univariate time series', 'No', 'No', 'No', 'No'),
    ('Xu and Moon (2011)', 'ENR construction cost index', 'Cointegrated VAR', 'Stochastic forecast', 'No', 'No', 'No'),
    ('Marzouk and Amin (2013)', 'Material prices, Egypt', 'Fuzzy logic, neural networks', 'No',
     'Ordering priority of materials', 'Price-sensitive items for contract price modification', 'No'),
    ('Ning et al. (2025)', 'Threaded steel price', 'Sliding-window CatBoost', 'No', 'No', 'No', 'No'),
    ('Liu et al. (2024)', 'Material prices (review)', 'Data-driven methods', 'Monte Carlo methods reviewed', 'No', 'No', 'No'),
    ('AlTalhoni et al. (2024)', 'Cost indices (review)', 'Time series, Machine Learning, hybrid', 'No', 'No', 'No', 'No'),
    ('Thu et al. (2024)', 'Price adjustment clauses', 'Survey and interviews', 'No', 'No', 'Clause thresholds and mechanisms', 'No'),
    ('This study', 'TPSO rebar list price', 'Process models, quantile regression, XGBoost, block bootstrap',
     'Scored by coverage, pinball loss and CRPS', 'Expenditure ratio of a purchase schedule',
     'Illustrative ±5% clause', '69 past origins'),
]


def round7_supplementary():
    write(pd.DataFrame(LITERATURE, columns=['Study', 'Forecast target', 'Method', 'Probabilistic output',
                                            'Purchase schedule', 'Contractual allocation',
                                            'Retrospective allowance test']), 'table_S22_literature')
    pr = pd.read_csv(REV / 'table_procedure_protection.csv')
    un = pd.read_csv(REV / 'table_protection_uncertainty.csv').set_index('allowance')
    sc = pd.read_csv(REV / 'table_scaling_oos.csv').set_index('model')
    rows = []
    for model in ('random walk', 'block bootstrap'):
        a = pr[(pr.model == model) & (pr.period == 'all origins')].set_index('procedure')
        u = un.loc[f'{model} P95']
        s = sc.loc[model]
        rows += [
            {'Model': model.capitalize(), 'Procedure': 'Nominal fit, as in Table 6', 'Origins': '2019-01 to 2024-09',
             'Protected / origins': f'{a.loc["nominal fit", "protected"]} / {a.loc["nominal fit", "n"]}',
             'Protection': f'{a.loc["nominal fit", "protection"]:.2f}',
             '90% interval or range': f'{u.ci_lo:.2f}–{u.ci_hi:.2f}',
             'Mean allowance (%)': f'{a.loc["nominal fit", "mean_P95_pct"]:.1f}'},
            {'Model': model.capitalize(), 'Procedure': 'Real fit, 2% inflation', 'Origins': '2019-01 to 2024-09',
             'Protected / origins': f'{a.loc["real fit, 2% inflation", "protected"]} / {a.loc["real fit, 2% inflation", "n"]}',
             'Protection': f'{a.loc["real fit, 2% inflation", "protection"]:.2f}', '90% interval or range': '–',
             'Mean allowance (%)': f'{a.loc["real fit, 2% inflation", "mean_P95_pct"]:.1f}'},
            {'Model': model.capitalize(), 'Procedure': 'Nominal fit, non-overlapping origins',
             'Origins': f'24 subsamples of {u.nonoverlap_size}', 'Protected / origins': '–',
             'Protection': f'{u.nonoverlap_mean:.2f} (mean)',
             '90% interval or range': f'{u.nonoverlap_min:.2f}–{u.nonoverlap_max:.2f}', 'Mean allowance (%)': '–'},
            {'Model': model.capitalize(), 'Procedure': f'P95 × k from earlier origins (k = {s.k_min:.2f}–{s.k_max:.2f})',
             'Origins': f'{s.first_origin} to {s.last_origin}', 'Protected / origins': f'{int(s.protected)} / {int(s.n)}',
             'Protection': f'{s.protection:.2f}', '90% interval or range': '–',
             'Mean allowance (%)': f'{s.mean_allowance_pct:.1f}'},
        ]
    f = un.loc['fixed +10%']
    rows.append({'Model': 'Fixed +10%', 'Procedure': 'Fixed allowance', 'Origins': '2019-01 to 2024-09',
                 'Protected / origins': f'{int(f.protected)} / {int(f.n)}', 'Protection': f'{f.protection:.2f}',
                 '90% interval or range': f'{f.ci_lo:.2f}–{f.ci_hi:.2f}', 'Mean allowance (%)': '10.0'})
    write(pd.DataFrame(rows), 'table_S23_protection_procedures')
    b = pd.read_csv(REV / 'table_protection_ci_blocks.csv')
    b['Allowance'] = b.allowance.map({'random walk P95': 'Random walk, model P95',
                                      'block bootstrap P95': 'Block bootstrap, model P95', 'fixed +10%': 'Fixed +10%'})
    t = b.pivot(index='Allowance', columns='mean_block', values=['ci_lo', 'ci_hi'])
    out = pd.DataFrame({'Allowance': t.index,
                        'Protection': [f'{b[b.Allowance == a].protection.iloc[0]:.2f}' for a in t.index]})
    for L in (6, 12, 24, 36):
        out[f'90% interval, mean block {L} months'] = [f'{lo:.2f}–{hi:.2f}' for lo, hi in
                                                       zip(t[('ci_lo', L)], t[('ci_hi', L)])]
    order = ['Random walk, model P95', 'Block bootstrap, model P95', 'Fixed +10%']
    write(out.set_index('Allowance').loc[order].reset_index(), 'table_S15_protection_blocks')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for fn in (primary, uncertainty, conformal_compact, bands_compact, validation_compact, sensitivity, procurement,
               code_chain, release_timing, supplementary_copies, crps_with_bootstrap, uncertainty_with_bootstrap,
               protection, profiles, round2_supplementary, round7_supplementary):
        fn()
        print(f'{fn.__name__} done')


if __name__ == '__main__':
    main()
