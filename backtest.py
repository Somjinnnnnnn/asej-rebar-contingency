"""Scoring and rolling-origin backtest helpers, taken from the concrete study's
run_revision.py (stage_diagnostics, rng_for, Timer, _quantiles and the scoring loop of
stage_backtest) and made independent of the concrete item selection.

The point forecasts and bands are scored on the price ratio P(origin + h) / P(origin):
coverage of the P5-P95 band, one-sided coverage of P80/P95, pinball loss at 0.80/0.95
and the sample CRPS (models.crps_sample, models.pinball)."""
from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

import models as M

SEED = 20261022
PCTS = [5, 50, 80, 90, 95]


def rng_for(*keys) -> np.random.Generator:
    """Independent, reproducible stream per (stage, model, scenario)."""
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(np.random.SeedSequence([SEED, *ints]))


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                print(f'\n=== {name} ===', flush=True)

            def __exit__(self, *exc):
                dt = time.perf_counter() - self.t0
                timer.rows.append({'stage': name, 'seconds': round(dt, 2)})
                print(f'    ({dt:.1f} s)', flush=True)
        return _Ctx()


def quantiles(paths):  # paths (..., n) -> dict of percentiles over last axis
    return {p: np.percentile(paths, p, axis=-1) for p in PCTS}


def process_diagnostics(wide: pd.DataFrame, min_months: int = 36) -> pd.DataFrame:
    """Unchanged months, drift, volatility, normality, autocorrelation, ARCH and unit-root
    tests of each monthly log-price series (gaps are dropped, not bridged)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.stattools import adfuller
    rows = []
    for g in wide.columns:
        s = wide[g].dropna()
        if len(s) < min_months:
            continue
        r = np.log(wide[g]).diff().dropna()  # changes across a gap are NaN and dropped
        rows.append({
            'series': g, 'months': len(s), 'revisions': int((r.abs() > M.REV_TOL).sum()),
            'share_unchanged_months': round(float((r.abs() <= M.REV_TOL).mean()), 3),
            'annual_log_drift': round(float(r.mean() * 12), 4),
            'annual_volatility': round(float(r.std() * np.sqrt(12)), 4),
            'mean_revision_size': round(float(r[r.abs() > M.REV_TOL].mean()), 4),
            'jarque_bera_p': float(jarque_bera(r)[1]),
            'ljung_box_12_p': float(acorr_ljungbox(r, lags=[12])['lb_pvalue'].iloc[0]),
            'ljung_box_sq_12_p': float(acorr_ljungbox(r ** 2, lags=[12])['lb_pvalue'].iloc[0]),
            'arch_lm_12_p': float(het_arch(r, nlags=12)[1]),
            'adf_logprice_p': float(adfuller(np.log(s), autolag='AIC')[1]),
        })
    return pd.DataFrame(rows)


def score_paths(sims: dict[str, np.ndarray], actual: np.ndarray, h: int, origin: pd.Timestamp,
                series: list[str]) -> list[dict]:
    """Score simulated price ratios (horizon, n_series, n_paths) at horizon h against the
    realised ratios `actual` (n_series,)."""
    rows = []
    ok = ~np.isnan(actual)
    for name, sim in sims.items():
        q = quantiles(sim[h - 1])
        for i in np.flatnonzero(ok):
            y = actual[i]
            rows.append({'origin': f'{origin:%Y-%m}', 'h': h, 'model': name, 'series': series[i],
                         'actual_ratio': y, 'P50': q[50][i], 'P95': q[95][i],
                         'in_P5_P95': float(q[5][i] <= y <= q[95][i]), 'le_P80': float(y <= q[80][i]),
                         'le_P95': float(y <= q[95][i]),
                         'pinball80_pct': 100 * M.pinball(q[80][i], y, 0.8),
                         'pinball95_pct': 100 * M.pinball(q[95][i], y, 0.95),
                         'crps_pct': 100 * M.crps_sample(sim[h - 1, i], y)})
    for pct in (5, 10):
        for i in np.flatnonzero(ok):
            rows.append({'origin': f'{origin:%Y-%m}', 'h': h, 'model': f'Fixed +{pct}% contingency',
                         'series': series[i], 'actual_ratio': actual[i],
                         'le_P95': float(actual[i] <= 1 + pct / 100),
                         'pinball95_pct': 100 * M.pinball(1 + pct / 100, actual[i], 0.95)})
    return rows


def summarise(det: pd.DataFrame) -> pd.DataFrame:
    return (det.groupby(['model', 'h'])
            .agg(n=('actual_ratio', 'size'), cover_P5_P95=('in_P5_P95', 'mean'), cover_le_P80=('le_P80', 'mean'),
                 cover_le_P95=('le_P95', 'mean'), pinball80_pct=('pinball80_pct', 'mean'),
                 pinball95_pct=('pinball95_pct', 'mean'), crps_pct=('crps_pct', 'mean'))
            .reset_index())
