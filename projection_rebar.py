"""RQ3: 1-20-year contingency bands for TPSO rebar list prices (SD40 12 mm by default),
in real THB, nominal THB and USD.

Design
  * Headline band: stationary block bootstrap of the demeaned real monthly log changes
    (zero real drift, mean block 12 months); comparator: driftless real random walk (the best
    3-12-month model in the backtest). In the real-terms validation on the US steel PPI
    (validate_long_horizon.py --deflator CPIAUCSL) the block bootstrap had the lowest
    60-month CRPS and coverage closest to nominal, and fitted drifts did clearly worse.
  * The price models are fitted to the REAL monthly log changes (deflated by the Bangkok CPI,
    in THB of the origin month), so the long-run drift does not mix rebar and general
    inflation. Models: driftless random walk,
    GBM with fitted drift, GARCH(1,1), AR(1)-GARCH(1,1), Merton jump-diffusion, Markov
    regime-switching, stationary block bootstrap (demeaned) and a mean-reverting AR(1) in
    the log real level (Kendall bias-corrected).
  * Nominal THB = real paths x CPI paths. Inflation scenarios: the 2017-2026 Bangkok CPI
    rate, 1 %, 2 % and 3 % a year (the Bank of Thailand target range), and a stochastic
    path from an AR(1) on monthly CPI inflation fitted to 2000-2026 (residuals resampled),
    independent of the rebar paths.
  * USD = nominal THB / FX: at the BOT mid rate of the origin month, and with stochastic FX
    from an AR(1) in log THB/USD (2017-2026, AR coefficient capped at 0.99, residuals
    resampled), independent of the rebar paths (the correlation of monthly changes is
    0.02 at lag 0 and -0.07 at lag 1, table_driver_leadlag.csv).
Writes outputs/<out-dir>/table_projection_long.csv, table_projection_params.csv,
table_contingency.csv and run_log.txt.

Usage:
    python projection_rebar.py [--series "SD40 12mm"] [--paths 10000] [--origin 2026-09]
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
import models as M
import rebar_data as R
import rebar_models as RM

YEARS = [1, 2, 3, 5, 10, 15, 20]
PCTS = [5, 50, 80, 90, 95]
H = 12 * max(YEARS)
CPI_FIT_START = '2000-01-01'
MAIN_MODEL = 'Block bootstrap (demeaned)'
COMPARATOR = 'Random walk (real, driftless)'

warnings.filterwarnings('ignore', message='Workbook contains no default style')


def price_models(lp_real: pd.Series, n: int, series: str) -> dict:
    r = lp_real.diff().to_numpy()
    price = np.exp(lp_real.to_numpy())
    rng = lambda name: B.rng_for('proj', name, series)  # noqa: E731
    seed = lambda name: int(rng(name).integers(2 ** 31))  # noqa: E731
    out = {
        COMPARATOR: RM.rw_paths(r, H, n, rng('rw')),
        MAIN_MODEL: RM.block_bootstrap_paths(r, H, n, rng('boot')),
        'GBM (real, fitted drift)': RM.gbm_paths(r, H, n, rng('gbm')),
        'GARCH(1,1)': M.fit_simulate_garch(r, H, n, rng('garch')),
        'AR(1)-GARCH(1,1)': RM.ar_paths(r, H, n, seed('argarch'), garch=True),
        'Merton jump-diffusion': M.fit_simulate_merton(r, H, n, rng('merton'), price=price),
        'Markov regime-switching': M.fit_simulate_regime(r, H, n, rng('ms'), price=price),
        'Mean-reverting AR(1) level': RM.mean_reverting_paths(lp_real.to_numpy(), H, n, rng('ou')),
    }
    return out


def ar1_paths(x: np.ndarray, horizon: int, n: int, rng, log_level: bool, cap: float = 0.99) -> tuple[np.ndarray, dict]:
    """AR(1) with resampled residuals. log_level=True: x is a log level, returns level ratios
    to the last value; False: x are monthly rates, returns cumulative growth factors."""
    b, a = np.polyfit(x[:-1], x[1:], 1)
    b = min(b, cap)
    a = np.mean(x[1:] - b * x[:-1])
    e = x[1:] - (a + b * x[:-1])
    cur = np.full(n, x[-1])
    out = np.empty((horizon, n))
    for t in range(horizon):
        cur = a + b * cur + e[rng.integers(0, len(e), n)]
        out[t] = cur
    info = {'ar': b, 'long_run': a / (1 - b)}
    if log_level:
        return np.exp(out - x[-1]), info
    return np.exp(np.cumsum(out, axis=0)), info


def zero_drift(paths: np.ndarray) -> np.ndarray:
    """Remove the mean monthly log drift of a model's simulated paths (same shift for every
    path), keeping its volatility, tails and dependence. Over 20 years the fitted drift
    dominates the bands, and the QML means of the GARCH models differ from the sample mean
    (-0.06 %/month real): GARCH(1,1) mu = -0.47 %/month; the AR(1)-GARCH standardised
    residuals have mean +0.08, which adds a positive drift in filtered historical simulation."""
    lg = np.log(paths)
    m = lg[-1].mean() / len(lg)
    return np.exp(lg - m * np.arange(1, len(lg) + 1)[:, None])


def quant_rows(paths, base, model, basis, inflation, fx, unit, drift):
    rows = []
    for y in YEARS:
        q = np.percentile(paths[12 * y - 1], PCTS)
        rows.append({'model': model, 'drift': drift, 'basis': basis, 'inflation': inflation, 'fx': fx,
                     'unit': unit, 'year': y,
                     **{f'P{p}': base * v for p, v in zip(PCTS, q)},
                     **{f'P{p}_pct': 100 * (v - 1) for p, v in zip(PCTS, q)}})
    return rows


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
    ap.add_argument('--series', default='SD40 12mm')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--origin', type=pd.Timestamp, default=None, help='default: last month of the series')
    ap.add_argument('--out-dir', default='projection_2026-09')
    ap.add_argument('--data-dir', type=Path, default=D.DEFAULT_DATA_DIR)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    out = here / 'outputs' / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    log = open(out / 'run_log.txt', 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, log)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)

    panel, _ = D.load_price_panel(args.data_dir, here / 'outputs' / 'cache' / 'panel.pkl')
    wide, _, _ = R.splice(R.select_rebar(panel))
    s = wide[args.series].dropna()
    origin = args.origin or s.index[-1]
    s = s.loc[:origin]
    cpi = D.load_cpi_index(D.DEFAULT_CPI_BKK)
    fx = D.load_fx(args.data_dir)
    real = s / cpi.reindex(s.index) * cpi[origin]
    if real.isna().any():
        raise ValueError('CPI missing for some months of the series')
    lp_real = np.log(real)
    p0, fx0 = s[origin], fx[origin]
    pi_is = (cpi[origin] / cpi[s.index[0]]) ** (12 / (len(s) - 1)) - 1
    print(f'{args.series}: {s.index[0]:%Y-%m}..{origin:%Y-%m}, origin price {p0:,.0f} THB/ton, '
          f'FX {fx0:.3f} THB/USD = {p0 / fx0:,.0f} USD/ton')
    print(f'Real drift {1200 * lp_real.diff().mean():.2f} %/yr, volatility {100 * np.sqrt(12) * lp_real.diff().std():.2f} %/yr, '
          f'in-sample Bangkok CPI inflation {100 * pi_is:.2f} %/yr')

    sims = price_models(lp_real, args.paths, args.series)
    params = [{'model': k, **{a: b for a, b in v[1].items() if np.isscalar(b) or isinstance(b, (list, str))}}
              for k, v in sims.items()]
    pd.DataFrame(params).to_csv(out / 'table_projection_params.csv', index=False)
    print(pd.DataFrame(params).round(4).to_string(index=False))

    # inflation paths (cumulative CPI factors), FX paths (ratio to origin)
    infl_m = np.log(cpi.loc[CPI_FIT_START:origin]).diff().dropna().to_numpy()
    cpi_stoch, cpi_info = ar1_paths(infl_m, H, args.paths, B.rng_for('proj', 'cpi'), log_level=False)
    months = np.arange(1, H + 1)[:, None]
    infl = {f'{100 * pi_is:.2f} % (2017-2026)': (1 + pi_is) ** (months / 12),
            '1 %': 1.01 ** (months / 12), '2 %': 1.02 ** (months / 12), '3 %': 1.03 ** (months / 12),
            'stochastic AR(1)': cpi_stoch}
    fx_paths, fx_info = ar1_paths(np.log(fx.loc[:origin].to_numpy()), H, args.paths, B.rng_for('proj', 'fx'),
                                  log_level=True)
    print(f'CPI inflation AR(1): {cpi_info["ar"]:.3f}, long-run {1200 * cpi_info["long_run"]:.2f} %/yr | '
          f'log FX AR(1): {fx_info["ar"]:.3f}, long-run {np.exp(fx_info["long_run"]):.2f} THB/USD')

    rows = []
    for name, (fitted, _) in sims.items():
        for drift, paths in [('as fitted', fitted), ('zero real drift', zero_drift(fitted))]:
            q = lambda *a: quant_rows(*a, drift)  # noqa: E731
            rows += q(paths, p0, name, 'real', 'none', 'none', f'THB/ton of {origin:%b %Y}')
            for iname, ipath in infl.items():
                nom = paths * ipath
                rows += q(nom, p0, name, 'nominal', iname, 'none', 'THB/ton')
                rows += q(nom, p0 / fx0, name, 'USD', iname, f'fixed {fx0:.3f}', 'USD/ton')
                rows += q(nom / fx_paths, p0 / fx0, name, 'USD', iname, 'stochastic AR(1)', 'USD/ton')
    proj = pd.DataFrame(rows)
    proj.to_csv(out / 'table_projection_long.csv', index=False)

    # monthly quantiles of the headline band and the comparator, for fan charts
    fan = []
    for name in (MAIN_MODEL, COMPARATOR):
        paths = sims[name][0]
        for basis, series, base in [('real', paths, p0), ('nominal 2 %', paths * infl['2 %'], p0),
                                    ('USD 2 %, stochastic FX', paths * infl['2 %'] / fx_paths, p0 / fx0)]:
            q = np.percentile(series, [5, 20, 50, 80, 95], axis=1).T * base
            for m, row in enumerate(q, start=1):
                fan.append({'model': name, 'basis': basis, 'month': m,
                            'date': (origin + pd.DateOffset(months=m)).strftime('%Y-%m'),
                            **dict(zip(['P5', 'P20', 'P50', 'P80', 'P95'], row))})
    pd.DataFrame(fan).to_csv(out / 'data_fan_monthly.csv', index=False)

    def show(basis, inflation, fx_, drift, cols=('P50_pct', 'P95_pct')):
        t = proj[(proj.basis == basis) & (proj.inflation == inflation) & (proj.fx == fx_) & (proj.drift == drift)]
        for c in cols:
            print(f'\n{basis} | {drift} | inflation {inflation} | FX {fx_} | {c}')
            print(t.pivot(index='model', columns='year', values=c).round(1).to_string())
    for drift in ('as fitted', 'zero real drift'):
        show('real', 'none', 'none', drift, ('P5_pct', 'P50_pct', 'P80_pct', 'P95_pct'))
    show('nominal', '2 %', 'none', 'zero real drift')
    show('nominal', 'stochastic AR(1)', 'none', 'zero real drift')
    show('USD', '2 %', 'stochastic AR(1)', 'zero real drift')

    # contingency: % above today's price for P80/P90/P95, headline band and comparator (both
    # driftless by construction, so 'as fitted')
    cont = proj[proj.model.isin([MAIN_MODEL, COMPARATOR]) & (proj.drift == 'as fitted') & proj.year.isin([1, 5, 10, 20])]
    cont = cont.assign(role=np.where(cont.model == MAIN_MODEL, 'headline', 'comparator'))
    cont = cont[['role', 'model', 'basis', 'inflation', 'fx', 'year', 'P50_pct', 'P80_pct', 'P90_pct', 'P95_pct',
                 'P50', 'P80', 'P95', 'unit']]
    cont.to_csv(out / 'table_contingency.csv', index=False)
    for role in ('headline', 'comparator'):
        t = cont[(cont.role == role) & ((cont.basis == 'real') | (cont.inflation == '2 %'))
                 & ~cont.fx.str.startswith('fixed')]
        print(f'\nContingency, {role}: {t.model.iloc[0]} (real; nominal and USD at 2 % inflation)')
        print(t.drop(columns=['role', 'model']).round(1).to_string(index=False))
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == '__main__':
    main()
