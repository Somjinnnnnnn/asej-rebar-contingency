"""Manuscript figures (PNG at 300 dpi and PDF), drawn from the analysis outputs.

  fig_01_research_framework  stages of the study, from the data sources to the price-escalation allowance
  fig_02_price_history       spliced SD40 12 mm list price, nominal and real, code changes; monthly changes
  fig_03_backtest_skill      CRPS skill vs the random walk by horizon (RQ1-RQ2)
  fig_04_coverage            P95 coverage by horizon incl. fixed allowances; random-walk coverage by period
  fig_05_conformal           why conformal recalibration fails: median error by period; CRPS ratio to raw
  fig_06_fan_bands           1-20-year bands: real THB, nominal THB (2 %), USD (2 %, stochastic FX)
  fig_07_model_risk          20-year real P5-P50-P95 by model (zero real drift)
  fig_08_validation          P95 coverage at 12/36/60 months on the US steel PPI (real, zero drift)

Colours follow a validated categorical order (blue, orange, aqua, yellow; adjacent and
first-three all-pairs CVD checks pass on white); de-emphasised series are gray; text stays
in neutral ink. Every figure has a table twin in outputs/manuscript (manuscript_tables.py).

Usage:
    python manuscript_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

warnings.filterwarnings('ignore', message='Workbook contains no default style')
OUT = Path(__file__).resolve().parent / 'outputs'
MS = OUT / 'manuscript'

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, RED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e34948'
GRAY = '#c3c2b7'
BLUE_FAN = ['#cde2fb', '#86b6ef']  # sequential blue steps 100 and 250 (P5-P95, P20-P80)
W2, W1 = 7.2, 3.5  # double / single column width, inches

plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.labelsize': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-', 'xtick.color': AXIS, 'ytick.color': AXIS,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'legend.fontsize': 7.5, 'text.color': INK, 'lines.linewidth': 1.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


def save(fig, name):
    fig.savefig(MS / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(MS / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def end_label(ax, x, y, text, dy=0.0, color=INK2):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', va='center', ha='left',
                fontsize=7, color=color)


def end_labels(ax, x, items, gap_pt=9.0):
    """Direct labels at the line ends, spread vertically so that no two are closer than
    `gap_pt` points. items: [(y, text)]."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    pts = [(ax.transData.transform((x, y))[1] * to_pt, y, t) for y, t in items]
    order = sorted(range(len(pts)), key=lambda i: pts[i][0])
    placed = {}
    last = -1e9
    for i in order:
        p = max(pts[i][0], last + gap_pt)
        placed[i] = p
        last = p
    shift = (sum(placed[i] - pts[i][0] for i in order) / len(order)) if order else 0
    for i in order:  # centre the block on the data so labels move both ways
        dy = placed[i] - pts[i][0] - shift
        end_label(ax, x, pts[i][1], pts[i][2], dy=dy)


# --------------------------------------------------------------------------- figure 2

def fig_price_history():
    w = pd.read_csv(OUT / 'audit_2026-10' / 'data_rebar_series.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    s = w['SD40 12mm'].dropna()
    cpi = D.load_cpi_index(D.DEFAULT_CPI_BKK)
    real = s / cpi.reindex(s.index) * cpi[s.index[-1]]
    cmap = pd.read_csv(OUT / 'audit_2026-10' / 'table_code_map.csv', parse_dates=['first'], encoding='utf-8-sig')
    junctions = sorted(cmap.loc[(cmap['series'] == 'SD40 12mm') & (cmap['link'] > 0), 'first'])

    fig, (a, b) = plt.subplots(2, 1, figsize=(W2, 4.6), height_ratios=[2.2, 1], sharex=True)
    a.step(s.index, s / 1000, where='post', color=BLUE, label='Nominal THB')
    a.step(real.index, real / 1000, where='post', color=ORANGE, label=f'Real, THB of {s.index[-1]:%b %Y}')
    for d in junctions:
        a.axvline(d, color=MUTED, lw=0.6, zorder=0)
    groups = {}  # one label per year of code changes (Feb and Jun 2018 are 4 months apart)
    for d in junctions:
        groups.setdefault(d.year, []).append(d)
    for ds in groups.values():
        txt = 'code changes\n' + ' and '.join(f'{d:%b}' for d in ds) + f' {ds[0]:%Y}' if len(ds) > 1 \
            else f'code change\n{ds[0]:%b %Y}'
        a.annotate(txt, (ds[-1], 33.4), xytext=(3, 0), textcoords='offset points', fontsize=6.5, color=INK2, va='top')
    a.set_ylabel('Thousand THB per ton')
    a.set_ylim(14, 33.5)
    a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.80))
    a.set_title('(a) TPSO list price of SD40 12 mm deformed bar, spliced series')
    i_max = s.idxmax()
    a.annotate(f'nominal peak {s.max():,.0f}\n({i_max:%b %Y})', (i_max, s.max() / 1000), xytext=(14, -14),
               textcoords='offset points', fontsize=6.5, color=INK2, va='top')
    end_label(a, s.index[-1], s.iloc[-1] / 1000, f'{s.iloc[-1]:,.0f}')

    r = 100 * np.expm1(np.log(s).diff().dropna())
    b.bar(r.index, r, width=24, color=np.where(r >= 0, BLUE, RED), linewidth=0)
    b.axhline(0, color=AXIS, lw=0.6)
    b.set_ylabel('Monthly change (%)')
    b.set_title(f'(b) Monthly change: the price moved in {100 * (r != 0).mean():.0f}% of months '
                '(blue = rise, red = fall)')
    for d in junctions:
        b.axvline(d, color=MUTED, lw=0.6, zorder=0)
    fig.tight_layout()
    save(fig, 'fig_02_price_history')


# --------------------------------------------------------------------------- figure 3

HIGHLIGHT = [('XGBoost QR, own history, CV-tuned', 'XGBoost QR, own history', BLUE),
             ('XGBoost QR, + scrap+ore, CV-tuned', 'XGBoost QR, + scrap and iron ore', ORANGE),
             ('AR(1)-GARCH(1,1)', 'AR(1)-GARCH(1,1)', AQUA),
             ('GBM (data-calibrated)', 'GBM (data-calibrated)', YELLOW)]


def fig_backtest_skill():
    dm = pd.read_csv(OUT / 'backtest_v2' / 'table_dm_tests.csv')
    dm = dm[(dm['series'] == 'SD40 12mm') & (dm['benchmark'] == 'Random walk')]
    # the linear QR with all drivers and the fixed a-priori penalty (-83 % at 12 months) is left to
    # Table 4 so that it does not set the scale
    dm = dm[dm['model'] != 'Linear QR, + all drivers']
    hl = {m for m, _, _ in HIGHLIGHT}
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    hs = [1, 3, 6, 12]
    xs = np.arange(len(hs))
    for m, g in dm.groupby('model'):
        if m in hl:
            continue
        g = g.set_index('h').reindex(hs)
        ax.plot(xs, 100 * g['skill_vs_benchmark'], color=GRAY, lw=0.8, zorder=1)
    labels = []
    for m, lab, col in HIGHLIGHT:
        g = dm[dm['model'] == m].set_index('h').reindex(hs)
        y = 100 * g['skill_vs_benchmark'].to_numpy()
        sig = (g['p_value'] < 0.05).to_numpy()
        ax.plot(xs, y, color=col, zorder=3, label=lab)
        ax.scatter(xs[sig], y[sig], s=28, color=col, edgecolor='white', linewidth=1.2, zorder=4)
        ax.scatter(xs[~sig], y[~sig], s=28, facecolor='white', edgecolor=col, linewidth=1.4, zorder=4)
        labels.append((y[-1], lab))
    # headline long-horizon model, scored in the same backtest (revision2_analyses.py)
    bt = OUT / 'revision_2026-10' / 'table_boot_backtest.csv'
    if bt.exists():
        b = pd.read_csv(bt)
        b = b[b['model'] == 'Block bootstrap (demeaned), mean block 12'].set_index('h').reindex(hs)
        y = 100 * b['skill'].to_numpy()
        ax.plot(xs, y, color=INK, lw=1.2, ls='--', zorder=3, label='block bootstrap (headline)')
        labels.append((y[-1], 'block bootstrap'))
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xticks(xs, [f'{h}' for h in hs])
    ax.set_xlim(-0.2, len(hs) - 1 + 1.6)
    end_labels(ax, xs[-1], labels + [(0.0, 'random walk')])
    ax.set_xlabel('Forecast horizon (months)')
    ax.set_ylabel('CRPS skill vs random walk (%)')
    ax.set_title('SD40 12 mm, rolling-origin backtest 2020-06 to 2026-08: positive = better than the random walk')
    ax.scatter([], [], s=28, color=INK2, label='filled: unadjusted DM p < 0.05')
    ax.scatter([], [], s=28, facecolor='white', edgecolor=INK2, label='hollow: unadjusted p ≥ 0.05')
    ax.plot([], [], color=GRAY, lw=0.8, label='other models')
    ax.text(0.99, 0.97, 'No difference is significant after\nthe false discovery rate adjustment',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=INK, fontweight='bold')
    ax.legend(loc='lower left', ncol=2)
    fig.tight_layout()
    save(fig, 'fig_03_backtest_skill')


# --------------------------------------------------------------------------- figure 4

def fig_coverage():
    s = pd.read_csv(OUT / 'backtest_v2' / 'table_backtest_summary.csv')
    s = s[s['series'] == 'SD40 12mm']
    p = pd.read_csv(OUT / 'backtest_v2' / 'table_backtest_by_period.csv')
    p = p[(p['series'] == 'SD40 12mm') & (p['model'] == 'Random walk')]
    hs = [1, 3, 6, 12]
    xs = np.arange(len(hs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 3.0), sharey=True)
    lines = [('Random walk', 'Random walk P95', BLUE), ('XGBoost QR, own history, CV-tuned', 'XGBoost QR P95', ORANGE),
             ('Fixed +10% contingency', 'Fixed +10%', AQUA), ('Fixed +5% contingency', 'Fixed +5%', YELLOW)]
    labs = []
    for m, lab, col in lines:
        y = s[s['model'] == m].set_index('h').reindex(hs)['cover_le_P95'].to_numpy()
        a.plot(xs, y, color=col, marker='o', ms=4.5, mec='white', mew=1, label=lab)
        labs.append((y[-1], lab))
    a.axhline(0.95, color=INK2, lw=0.8)
    a.annotate('target 0.95', (1.5, 0.95), xytext=(0, 3), textcoords='offset points', fontsize=7, color=INK2, ha='center')
    a.set_title('(a) Share of outcomes at or below the band')
    a.set_ylabel('Empirical coverage')
    a.set_xticks(xs, hs); a.set_xlabel('Forecast horizon (months)'); a.set_xlim(-0.2, 4.3)
    a.set_ylim(0.5, 1.04)
    blabs = []
    for per, col in [('2020-06..2022-12', BLUE), ('2023-01..', ORANGE)]:
        y = p[p['period'] == per].set_index('h').reindex(hs)['cover_P5_P95'].to_numpy()
        lab = 'origins 2020-06 to 2022-12' if per.startswith('2020') else 'origins 2023-01 on'
        b.plot(xs, y, color=col, marker='o', ms=4.5, mec='white', mew=1, label=lab)
        blabs.append((y[-1], lab))
    b.axhline(0.90, color=INK2, lw=0.8)
    b.annotate('target 0.90', (2.4, 0.90), xytext=(0, -10), textcoords='offset points', fontsize=7, color=INK2, ha='center')
    b.set_title('(b) Random walk P5–P95 band by period')
    b.set_xticks(xs, hs); b.set_xlabel('Forecast horizon (months)'); b.set_xlim(-0.2, 4.6)
    a.legend(loc='lower left'); b.legend(loc='lower left')
    fig.tight_layout()
    end_labels(a, xs[-1], labs); end_labels(b, xs[-1], blabs)
    fig.tight_layout()
    save(fig, 'fig_04_coverage')


# --------------------------------------------------------------------------- figure 5

def fig_conformal():
    det = pd.read_csv(OUT / 'backtest_v2' / 'backtest_detail.csv')
    rw = det[(det['model'] == 'Random walk') & (det['series'] == 'SD40 12mm')].copy()
    rw['err'] = 100 * (rw['y_log'] - rw['q50'])
    rw['early'] = pd.to_datetime(rw['origin']) < '2022-05-01'
    c = pd.read_csv(OUT / 'backtest_v2' / 'table_conformal_summary.csv')
    c = c[(c['series'] == 'SD40 12mm') & (c['model'] == 'Random walk')]
    hs = [1, 3, 6, 12]
    xs = np.arange(len(hs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 3.0))
    wbar = 0.36
    for k, (early, lab, col) in enumerate([(True, 'origins 2020-06 to 2022-04 (calibration)', BLUE),
                                           (False, 'origins 2022-05 on (evaluation)', ORANGE)]):
        y = rw[rw['early'] == early].groupby('h')['err'].mean().reindex(hs).to_numpy()
        bars = a.bar(xs + (k - 0.5) * (wbar + 0.03), y, width=wbar, color=col, label=lab, linewidth=0)
        for bx, by in zip(bars, y):
            a.annotate(f'{by:+.1f}', (bx.get_x() + bx.get_width() / 2, by), xytext=(0, 3 if by >= 0 else -9),
                       textcoords='offset points', ha='center', fontsize=6.5, color=INK2)
    a.axhline(0, color=AXIS, lw=0.6)
    a.set_xticks(xs, hs); a.set_xlabel('Forecast horizon (months)')
    a.set_ylabel('Mean of outcome − median forecast (log %)')
    a.set_title('(a) Random-walk median error changes sign')
    a.set_ylim(-13, 17)
    a.legend(loc='lower left')
    raw = c[c['variant'] == 'raw'].set_index('h')['crps_pct'].reindex(hs)
    blabs = []
    for v, col in [('expanding', BLUE), ('rolling 24', ORANGE), ('symmetric, expanding', AQUA), ('symmetric, rolling 24', YELLOW)]:
        y = (c[c['variant'] == v].set_index('h')['crps_pct'].reindex(hs) / raw).to_numpy()
        b.plot(xs, y, color=col, marker='o', ms=4.5, mec='white', mew=1, label=v)
        blabs.append((y[-1], v))
    b.axhline(1, color=INK2, lw=0.8)
    b.annotate('no recalibration', (2.5, 1), xytext=(0, -10), textcoords='offset points', fontsize=7, color=INK2,
               ha='center')
    b.set_xticks(xs, hs); b.set_xlabel('Forecast horizon (months)'); b.set_xlim(-0.2, 5.2)
    b.set_ylabel('CRPS relative to the raw forecast')
    b.set_title('(b) Conformal recalibration of the random walk, 2022-05 on')
    b.legend(loc='upper left')
    fig.tight_layout()
    end_labels(b, xs[-1], blabs)
    save(fig, 'fig_05_conformal')


# --------------------------------------------------------------------------- figure 6

def fig_fan_bands():
    fan = pd.read_csv(OUT / 'projection_2026-09' / 'data_fan_monthly.csv', parse_dates=['date'])
    w = pd.read_csv(OUT / 'audit_2026-10' / 'data_rebar_series.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    s = w['SD40 12mm'].dropna()
    cpi = D.load_cpi_index(D.DEFAULT_CPI_BKK)
    fx = D.load_fx(D.DEFAULT_DATA_DIR).reindex(s.index)
    hist = {'real': s / cpi.reindex(s.index) * cpi[s.index[-1]], 'nominal 2 %': s, 'USD 2 %, stochastic FX': s / fx}
    panels = [('real', '(a) Real, THB of Sep 2026 per ton', 1000, 'Thousand THB per ton'),
              ('nominal 2 %', '(b) Nominal THB per ton, 2 % inflation', 1000, 'Thousand THB per ton'),
              ('USD 2 %, stochastic FX', '(c) USD per ton, 2 % inflation, stochastic THB/USD', 1, 'USD per ton')]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.9))
    for ax, (basis, title, scale, ylab) in zip(axes, panels):
        h = fan[(fan['model'] == 'Block bootstrap (demeaned)') & (fan['basis'] == basis)]
        rw = fan[(fan['model'] == 'Random walk (real, driftless)') & (fan['basis'] == basis)]
        ax.fill_between(h['date'], h['P5'] / scale, h['P95'] / scale, color=BLUE_FAN[0], lw=0, label='P5–P95')
        ax.fill_between(h['date'], h['P20'] / scale, h['P80'] / scale, color=BLUE_FAN[1], lw=0, label='P20–P80')
        ax.plot(h['date'], h['P50'] / scale, color=BLUE, lw=1.4, label='median')
        ax.plot(rw['date'], rw['P95'] / scale, color=ORANGE, lw=1.2, label='random walk P95')
        ax.plot(hist[basis].index, hist[basis] / scale, color=INK, lw=0.9, label='TPSO 2017–2026')
        last = h.iloc[-1]
        end_label(ax, last['date'], last['P95'] / scale, f'{last["P95"] / scale:,.0f}' if scale == 1 else f'{last["P95"] / scale:.0f}k')
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, None)
        ax.set_xticks(pd.to_datetime(['2020-01-01', '2030-01-01', '2040-01-01']), ['2020', '2030', '2040'])
    axes[0].legend(loc='upper left', fontsize=6.5)
    fig.suptitle('Headline band: block bootstrap of demeaned real monthly changes; comparator: driftless random walk',
                 x=0.01, ha='left', fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, 'fig_06_fan_bands')


# --------------------------------------------------------------------------- figure 7

def fig_model_risk():
    p = pd.read_csv(OUT / 'projection_2026-09' / 'table_projection_long.csv')
    p = p[(p['basis'] == 'real') & (p['year'] == 20) & (p['drift'] == 'zero real drift')].sort_values('P95_pct')
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ys = np.arange(len(p))
    for y, (_, r) in zip(ys, p.iterrows()):
        col = BLUE if r['model'] == 'Block bootstrap (demeaned)' else ORANGE if r['model'].startswith('Random walk') else GRAY
        ax.plot([r['P5_pct'], r['P95_pct']], [y, y], color=col, lw=2.2, solid_capstyle='round')
        ax.scatter([r['P50_pct']], [y], s=30, color=col, edgecolor='white', linewidth=1.2, zorder=3)
        ax.annotate(f'{r["P95_pct"]:+.0f}%', (r['P95_pct'], y), xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=7, color=INK2)
    names = {'Block bootstrap (demeaned)': 'Block bootstrap (headline)',
             'Random walk (real, driftless)': 'Random walk (comparator)', 'GBM (real, fitted drift)': 'GBM'}
    ax.set_yticks(ys, [names.get(m, m) for m in p['model']])
    ax.axvline(0, color=AXIS, lw=0.6)
    ax.set_xlabel('Change in the real price after 20 years (%); line = P5 to P95, dot = median')
    ax.set_title('Model risk at 20 years, real terms, every model with zero real drift')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'fig_07_model_risk')


# --------------------------------------------------------------------------- figure 8

def fig_validation():
    s = pd.read_csv(OUT / 'validation_wpu1017_real' / 'table_validation_summary.csv')
    s = s[s['drift'] == 'zero drift']
    hs = [12, 36, 60]
    xs = np.arange(len(hs))
    hl = [('Block bootstrap (demeaned)', 'Block bootstrap (headline)', BLUE),
          ('Random walk (driftless)', 'Random walk (comparator)', ORANGE),
          ('AR(1)-GARCH(1,1)', 'AR(1)-GARCH(1,1)', AQUA),
          ('Mean-reverting AR(1) level', 'Mean-reverting AR(1)', YELLOW)]
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    names = {m for m, _, _ in hl}
    for m, g in s.groupby('model'):
        if m in names or m == 'Block bootstrap (with drift)':
            continue
        ax.plot(xs, g.set_index('h').reindex(hs)['cover_le_P95'], color=GRAY, lw=0.8)
    labs = []
    for m, lab, col in hl:
        y = s[s['model'] == m].set_index('h').reindex(hs)['cover_le_P95'].to_numpy()
        ax.plot(xs, y, color=col, marker='o', ms=4.5, mec='white', mew=1, label=lab)
        labs.append((y[-1], lab))
    ax.axhline(0.95, color=INK2, lw=0.8)
    ax.annotate('target 0.95', (0, 0.95), xytext=(0, 3), textcoords='offset points', fontsize=7, color=INK2)
    ax.set_xticks(xs, hs); ax.set_xlim(-0.2, 2.9)
    ax.set_xlabel('Horizon (months)'); ax.set_ylabel('Share of outcomes at or below P95')
    ax.set_title('US PPI steel mill products, real, 10-year windows')
    ax.plot([], [], color=GRAY, lw=0.8, label='other models')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    end_labels(ax, xs[-1], labs)
    save(fig, 'fig_08_validation')


# --------------------------------------------------------------------------- figure 9

def fig_procurement():
    """Worked procurement example (revision_analyses.py): allowances for the 24-month schedule, and
    the realized expenditure of that schedule at past origins against the allowances."""
    rev = OUT / 'revision_2026-10'
    p = pd.read_csv(rev / 'table_procurement_example.csv')
    p = p[p.schedule.str.startswith('A') & p.basis.str.startswith('nominal')]
    h = pd.read_csv(rev / 'data_procurement_history.csv', parse_dates=['origin'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 3.1), gridspec_kw={'width_ratios': [1, 1.25]})
    cats = [('P80', 'Expenditure P80'), ('P95', 'Expenditure P95'), ('weighted_per_date_P95', 'Weighted P95 of\npurchase prices'),
            ('endpoint_P95', 'P95 at the last\npurchase (month 24)')]
    ys = np.arange(len(cats))[::-1]
    for k, (model, col, lab) in enumerate([('Block bootstrap (demeaned)', BLUE, 'block bootstrap'),
                                            ('Random walk (real, driftless)', ORANGE, 'random walk')]):
        row = p[p.model == model].iloc[0]
        vals = [row[c] for c, _ in cats]
        a.barh(ys + (0.18 if k == 0 else -0.18), vals, height=0.34, color=col, label=lab)
        for y, v in zip(ys + (0.18 if k == 0 else -0.18), vals):
            a.text(v + 1, y, f'{v:.0f}%', va='center', fontsize=6.5, color=INK2)
    for x, lab in [(5, '+5%'), (10, '+10%')]:
        a.axvline(x, color=INK2, lw=0.8, ls='--')
        a.text(x + (-0.6 if x == 5 else 0.6), ys[-1] - 0.62, lab, ha='right' if x == 5 else 'left', fontsize=6.5, color=INK2)
    a.set_yticks(ys, [lab for _, lab in cats]); a.grid(axis='y', visible=False)
    a.set_ylim(ys[-1] - 0.85, ys[0] + 0.6)
    a.set_xlabel('% above the budget at the September 2026 price'); a.set_xlim(0, 80)
    a.set_title('(a) 24-month schedule, 2% inflation')
    a.legend(loc='upper right')
    b.plot(h.origin, h.realized_pct, color=INK, lw=1.4, label='realized expenditure')
    b.plot(h.origin, h['block bootstrap_P95_pct'], color=BLUE, lw=1.2, label='block bootstrap P95')
    b.plot(h.origin, h['random walk_P95_pct'], color=ORANGE, lw=1.2, label='random walk P95')
    b.axhline(10, color=AQUA, lw=1.2, label='fixed +10%')
    b.axhline(0, color=AXIS, lw=0.6)
    b.set_ylabel('% above the budget at the origin price'); b.set_xlabel('Origin of the 24-month schedule')
    b.set_title('(b) Same schedule at past origins, TPSO prices')
    b.set_ylim(-30, 75)
    b.legend(loc='upper right', fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, 'fig_09_procurement')


# --------------------------------------------------------------------------- figure 1

def _flow_box(ax, x, y, w, h, title, lines, fc='white', ec=AXIS, lw=0.8):
    """Rounded box with a bold title and body lines; (x, y) is the lower-left corner in inches."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=fc, ec=ec, lw=lw))
    ax.text(x + 0.12, y + h - 0.12, title, ha='left', va='top', fontsize=8, fontweight='bold', color=INK)
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.36 - 0.17 * i, line, ha='left', va='top', fontsize=7, color=INK2)


def _arrow(ax, xy_from, xy_to):
    ax.annotate('', xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=9))


def fig_framework():
    """Research framework: data sources, five stages and the allowance output."""
    width, height = W2, 6.25
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis('off')
    left, src_w, gap = 0.05, 1.95, 0.25
    mx = left + src_w + gap                 # main column
    mw = width - mx - 0.05
    half = (mw - 0.15) / 2
    # stage geometry (lower-left y, height): title + 0.17 in per body line
    s1, s23, s4, s5, out = (5.05, 1.15), (3.60, 1.15), (2.35, 0.95), (1.10, 0.95), (0.05, 0.75)

    _flow_box(ax, left, height - 0.05 - 2.05, src_w, 2.05, 'Data sources',
              ['TPSO monthly price reports,', '  January 2017 to September 2026', 'Bangkok CPI (TPSO)',
               'THB/USD rate (Bank of Thailand)', 'Bangkok diesel price (EPPO)',
               'World Bank Pink Sheet: iron ore,', '  coal, Brent', 'FRED: US steel and scrap PPI,',
               '  US CPI'], fc='#f4f3ef')
    _flow_box(ax, mx, s1[0], mw, s1[1], 'Stage 1  Data collection and preparation',
              ['Parsing of 117 monthly TPSO reports; selection of 74 rebar item codes',
               'Splicing of the changing item codes into continuous series; junction checks against benchmarks',
               'Conversion to real THB (Bangkok CPI) and USD; driver lags set by publication dates',
               'Main series: SD40 12 mm; robustness series: SD50 and SR24 12 mm'])
    _flow_box(ax, mx, s23[0], half, s23[1], 'Stage 2  Price-process models for RQ1',
              ['Random walk, GBM, empirical changes', 'AR(1), GARCH(1,1), AR(1)-GARCH(1,1)',
               'Merton jump-diffusion, Markov switching', 'Simulated predictive distributions'])
    _flow_box(ax, mx + half + 0.15, s23[0], half, s23[1], 'Stage 3  Quantile regression for RQ2',
              ['Linear and XGBoost quantile regression', 'Feature sets: own history, + scrap and',
               '  iron ore, + all global drivers', 'Fixed and time-series CV settings'])
    _flow_box(ax, mx, s4[0], mw, s4[1], 'Stage 4  Rolling-origin backtest for RQ1 and RQ2',
              ['75 monthly origins, June 2020 to August 2026; horizons of 1, 3, 6 and 12 months',
               'Coverage, pinball loss and CRPS; Diebold–Mariano tests against the random walk',
               'Conformal recalibration; comparison with fixed allowances of +5% and +10%'])
    _flow_box(ax, mx, s5[0], mw, s5[1], 'Stage 5  Long-horizon bands and external check for RQ3',
              ['1- to 20-year bands from September 2026 in real THB, nominal THB and USD',
               'Headline: block bootstrap of demeaned real changes; comparator: driftless random walk',
               'External check on rolling 10-year windows of the US steel PPI, 141 monthly origins'])
    _flow_box(ax, mx, out[0], mw, out[1], 'Stage 6  Expenditure allowances and protection for RQ4',
              ['P80 and P95 of the expenditure of a purchase schedule; comparison with fixed allowances',
               'of 5% and 10%; protection of the allowances at 69 past origins'],
              fc=BLUE_FAN[0], ec=BLUE, lw=1.1)

    mid_y1 = s1[0] + s1[1] / 2
    _arrow(ax, (left + src_w, mid_y1), (mx, mid_y1))
    c2, c3 = mx + half / 2, mx + half + 0.15 + half / 2
    for cx in (c2, c3):
        _arrow(ax, (cx, s1[0]), (cx, s23[0] + s23[1]))
        _arrow(ax, (cx, s23[0]), (cx, s4[0] + s4[1]))
    cm = mx + mw / 2
    _arrow(ax, (cm, s4[0]), (cm, s5[0] + s5[1]))
    _arrow(ax, (cm, s5[0]), (cm, out[0] + out[1]))
    save(fig, 'fig_01_research_framework')


def fig_timeline():
    """Information timing of one forecast: reference month, issuance, target month and observation."""
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    ax.set_axis_off()
    xs = {'t': 1.0, 't+1': 2.2, 'gap': 3.4, 't+h': 4.6, 't+h+1': 5.8}
    ax.plot([0.4, 6.4], [0, 0], color=AXIS, lw=1.2)
    for label, x in xs.items():
        if label == 'gap':
            ax.text(x, 0, '…', ha='center', va='center', fontsize=10, color=INK2, backgroundcolor='white')
            continue
        ax.plot([x, x], [-0.08, 0.08], color=INK2, lw=1.0)
        ax.text(x, -0.22, f'month {label}', ha='center', va='top', fontsize=7.5, color=INK2)
    above = [('t', 'Reference month\nTPSO list price P(t)'),
             ('t+1', 'TPSO report for month t\npublished; forecast for\norigin t issued'),
             ('t+h', 'Target list price P(t+h);\npurchase of the lot'),
             ('t+h+1', 'Report for month t+h\npublished; outcome\nobserved')]
    for label, text in above:
        ax.annotate(text, (xs[label], 0.1), xytext=(xs[label], 0.75), ha='center', va='bottom', fontsize=7.5,
                    color=INK, arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.9))
    ax.text(xs['t+1'], -0.6, 'Inputs at issuance: Pink Sheet, THB/USD and diesel\nfor month t (lag 0); US price indices '
            'for month t − 1 (lag 1)', ha='left', va='top', fontsize=7, color=INK2)
    ax.annotate('', (xs['t+h'], 0.42), (xs['t+1'], 0.42), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.0))
    ax.text((xs['t+1'] + xs['t+h']) / 2, 0.47, 'lead time from issuance about h − 1 months', ha='center', va='bottom',
            fontsize=7, color=ORANGE)
    ax.set_xlim(0.2, 6.6); ax.set_ylim(-1.15, 1.45)
    save(fig, 'fig_10_timeline')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    MS.mkdir(parents=True, exist_ok=True)
    for fn in [fig_framework, fig_price_history, fig_backtest_skill, fig_coverage, fig_conformal, fig_fan_bands,
               fig_model_risk, fig_validation, fig_procurement, fig_timeline]:
        fn()
        print(f'{fn.__name__} done')


if __name__ == '__main__':
    main()
