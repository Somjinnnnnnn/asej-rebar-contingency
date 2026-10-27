"""Data audit of TPSO rebar list prices (category 04), Jan 2017 - Sep 2026.

Writes to outputs/<run>/:
  table_rebar_codes.csv        every rebar code: product, process, listing period, gaps
  table_code_map.csv           which code supplies each spliced series in each period
  table_junction_checks.csv    price change at each code change vs. a benchmark (rule 4)
  data_rebar_series.csv        spliced monthly series (THB/ton), wide
  table_audit.csv              coverage, share of months with a change, drift, volatility,
                               largest rises and falls per series
  table_audit_no_junctions.csv same statistics with the junction months left out
  table_process_diagnostics.csv  normality, autocorrelation, ARCH and unit-root tests
  fig_rebar_history.png        spliced SD40/SD50/SR24 12 mm series with code changes

Usage:
    python audit_rebar.py [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
import data as D
import rebar_data as R

warnings.filterwarnings('ignore', message='Workbook contains no default style')


def series_stats(wide: pd.DataFrame, drop_months: dict[str, list] | None = None) -> pd.DataFrame:
    rows = []
    for c in wide.columns:
        s = wide[c]
        r = np.log(s).diff()
        if drop_months and c in drop_months:
            r = r.drop(index=drop_months[c], errors='ignore')
        r = r.dropna()
        listed = s.dropna()
        span = pd.date_range(listed.index[0], listed.index[-1], freq='MS')
        rows.append({
            'series': c, 'first': f'{listed.index[0]:%Y-%m}', 'last': f'{listed.index[-1]:%Y-%m}',
            'months': len(listed), 'missing_months_inside': len(span) - len(listed),
            'monthly_changes': len(r), 'share_months_changed': (r != 0).mean(),
            'annual_log_drift': r.mean() * 12, 'annual_volatility': r.std(ddof=1) * np.sqrt(12),
            'largest_rise_pct': 100 * np.expm1(r.max()), 'largest_rise_month': f'{r.idxmax():%Y-%m}',
            'largest_fall_pct': 100 * np.expm1(r.min()), 'largest_fall_month': f'{r.idxmin():%Y-%m}',
            'months_fall_le_-5pct': int((r <= np.log(0.95)).sum()),
            'months_rise_ge_5pct': int((r >= np.log(1.05)).sum()),
            'price_first': listed.iloc[0], 'price_last': listed.iloc[-1],
            'price_min': listed.min(), 'price_max': listed.max(),
        })
    return pd.DataFrame(rows)


def plot_history(wide, cmap, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for s, col in [('SD40 12mm', '#C0392B'), ('SD50 12mm', '#2E86C1'), ('SR24 12mm', '#555555'),
                   ('SD40 plain 12mm', '#E59866')]:
        if s in wide:
            ax.step(wide.index, wide[s], where='post', color=col, lw=1.3, label=s)
    for d in sorted(cmap.loc[(cmap['link'] > 0) & cmap['series'].isin(['SD40 12mm']), 'first'].unique()):
        ax.axvline(d, color='grey', ls=':', lw=0.8)
    ax.set_ylabel('THB/ton (nominal)'); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title('TPSO rebar list prices, spliced series (dotted: code changes of SD40 12 mm)')
    fig.tight_layout(); fig.savefig(out / 'fig_rebar_history.png', dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=D.DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'audit_2026-10')
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)

    panel, audit = D.load_price_panel(args.data_dir, out.parent / 'cache' / 'panel.pkl')
    print(f'Reports: {len(audit)} | months {panel.date.min():%Y-%m}..{panel.date.max():%Y-%m} | '
          f'name != printed period: {(~audit.header_matches).sum()}')
    rebar = R.select_rebar(panel)
    wide, cmap, codes = R.splice(rebar)
    codes.to_csv(out / 'table_rebar_codes.csv', encoding='utf-8-sig')
    cmap.to_csv(out / 'table_code_map.csv', index=False, encoding='utf-8-sig')
    print(f'Rebar codes: {rebar.code.nunique()} | spliced series: {wide.shape[1]}')

    junc = R.junction_checks(wide, cmap, panel, set(rebar['code']))
    junc.to_csv(out / 'table_junction_checks.csv', index=False, encoding='utf-8-sig')
    j = junc.dropna(subset=['junction_dlog'])
    print('Junction checks (log change at code change vs benchmark), by month:')
    print(j.groupby(j['month'].dt.strftime('%Y-%m'))
          .agg(series=('series', 'size'), junction_median=('junction_dlog', 'median'),
               junction_min=('junction_dlog', 'min'), junction_max=('junction_dlog', 'max'),
               benchmark=('benchmark_dlog', 'median'), flagged=('flag', 'sum')).round(4).to_string())
    print('Junctions after a gap (not chained):', junc[junc['junction_dlog'].isna()][['series', 'month']].to_string(index=False))

    keep = [c for c in wide.columns if wide[c].notna().sum() >= R.MIN_MONTHS]
    wide = wide[keep]
    wide.to_csv(out / 'data_rebar_series.csv', encoding='utf-8-sig')
    stats = series_stats(wide)
    stats.to_csv(out / 'table_audit.csv', index=False, encoding='utf-8-sig')
    drop = {s: list(g['month']) for s, g in j.groupby('series')}
    stats_nj = series_stats(wide, drop)
    stats_nj.to_csv(out / 'table_audit_no_junctions.csv', index=False, encoding='utf-8-sig')
    cols = ['series', 'first', 'last', 'months', 'missing_months_inside', 'share_months_changed',
            'annual_log_drift', 'annual_volatility', 'largest_rise_pct', 'largest_rise_month',
            'largest_fall_pct', 'largest_fall_month', 'months_fall_le_-5pct', 'months_rise_ge_5pct']
    print(stats[cols].round(3).to_string(index=False))
    print('Without junction months:')
    print(stats_nj[cols[:1] + cols[5:]].round(3).to_string(index=False))

    diag = B.process_diagnostics(wide)
    diag.to_csv(out / 'table_process_diagnostics.csv', index=False)
    print(diag.round(4).to_string(index=False))

    # reference point from the concrete study: 180 ksc ready-mixed concrete
    c180 = panel[panel['code'] == '0101010108100000'].set_index('date')['price'].sort_index()
    r180 = np.log(c180).diff().dropna()
    print(f'Reference 180 ksc concrete: {len(c180)} months, share changed {(r180 != 0).mean():.3f}, '
          f'annual volatility {r180.std() * np.sqrt(12):.4f}')
    plot_history(wide, cmap, out)


if __name__ == '__main__':
    main()
