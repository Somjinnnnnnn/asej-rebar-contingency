"""Forecast models of the rebar study that are not in models.py.

Every model returns quantiles of the log price ratio log P(origin + h) / P(origin) on the
common grid TAUS, as an array (horizon, len(TAUS)). Simulation models (models.py, AR,
AR-GARCH) are reduced to the same grid, so all models are scored on identical quantiles.

Process models (RQ1), fitted to the monthly log changes up to the origin:
  random walk        driftless Gaussian random walk, volatility from history
  GBM                models.gbm_params_from_history
  empirical          historical h-month log changes (unconditional, overlapping)
  AR(1)              AR(1) mean, constant variance, bootstrapped standardised residuals
  AR(1)-GARCH(1,1)   AR(1) mean, GARCH(1,1) variance, bootstrapped standardised residuals
  GARCH(1,1), Merton jump-diffusion, Markov regime-switching   from models.py

Direct quantile models (RQ2), one model per horizon h, trained on all months s with
s + h <= origin:
  linear quantile regression (L1-penalised, standardised features)
  XGBoost multi-quantile regression
each with own-history features only or with the global steel drivers added.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import models as M

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)
_Z = norm.ppf(TAUS)
QR_ALPHA = 1e-3
XGB_PARAMS = dict(max_depth=2, learning_rate=0.05, n_estimators=200, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0)


def sample_quantiles(paths: np.ndarray) -> np.ndarray:
    """Price-ratio paths (horizon, n) -> log-ratio quantiles (horizon, len(TAUS))."""
    return np.quantile(np.log(paths), TAUS, axis=1).T


def rw_quantiles(r: np.ndarray, horizon: int) -> np.ndarray:
    s = np.nanstd(r, ddof=1)
    h = np.arange(1, horizon + 1)[:, None]
    return np.sqrt(h) * s * _Z[None, :]


def gbm_quantiles(r: np.ndarray, horizon: int) -> np.ndarray:
    mu, s = M.gbm_params_from_history(r)
    h = np.arange(1, horizon + 1)[:, None]
    return h * (mu - 0.5 * s ** 2) + np.sqrt(h) * s * _Z[None, :]


def empirical_quantiles(lp: np.ndarray, horizon: int) -> np.ndarray:
    lp = lp[~np.isnan(lp)]
    return np.stack([np.quantile(lp[h:] - lp[:-h], TAUS) for h in range(1, horizon + 1)])


def ar_quantiles(r: np.ndarray, horizon: int, n: int, seed: int, garch: bool) -> tuple[np.ndarray, dict]:
    paths, info = ar_paths(r, horizon, n, seed, garch)
    return np.quantile(np.log(paths), TAUS, axis=1).T, info


def ar_paths(r: np.ndarray, horizon: int, n: int, seed: int, garch: bool) -> tuple[np.ndarray, dict]:
    """AR(1) mean with constant or GARCH(1,1) variance, fitted by Gaussian QML to 100 x log
    changes; price-ratio paths (horizon, n) by filtered historical simulation (standardised
    residuals resampled with replacement and pushed through the fitted recursions)."""
    from arch import arch_model
    r = r[~np.isnan(r)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = arch_model(100 * r, mean='AR', lags=1, vol='GARCH' if garch else 'Constant',
                         dist='normal', rescale=False).fit(disp='off')
    p = res.params
    c, phi = p.iloc[0], p.iloc[1]
    z = res.std_resid[~np.isnan(res.std_resid)]
    rng = np.random.default_rng(seed)
    x_prev = np.full(n, 100 * r[-1])
    if garch:
        om, a, b = p['omega'], p['alpha[1]'], p['beta[1]']
        s2 = np.full(n, om + a * res.resid[-1] ** 2 + b * res.conditional_volatility[-1] ** 2)
    else:
        s2 = np.full(n, p['sigma2'])
    paths = np.empty((horizon, n))
    for t in range(horizon):
        e = np.sqrt(s2) * z[rng.integers(0, len(z), n)]
        x_prev = c + phi * x_prev + e
        paths[t] = x_prev / 100
        if garch:
            s2 = om + a * e ** 2 + b * s2
    paths = np.cumsum(paths, axis=0)  # log ratio
    info = {'ok': res.convergence_flag == 0, 'const': c / 100, 'phi': phi}
    if garch:
        info.update(alpha=a, beta=b, persistence=a + b)
    return np.exp(paths), info


# --------------------------------------------------------------------------- long-horizon paths (RQ3)
# All return price-ratio paths (horizon, n) and an info dict.

def rw_paths(r: np.ndarray, horizon: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Driftless Gaussian random walk."""
    s = np.nanstd(r, ddof=1)
    return np.exp(np.cumsum(s * rng.standard_normal((horizon, n)), axis=0)), {'sigma': s}


def gbm_paths(r: np.ndarray, horizon: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    mu, s = M.gbm_params_from_history(r)
    return M.simulate_gbm(mu, s, horizon, n, rng), {'mu': mu, 'sigma': s}


def block_bootstrap_paths(r: np.ndarray, horizon: int, n: int, rng: np.random.Generator,
                          mean_block: int = 12, demean: bool = True) -> tuple[np.ndarray, dict]:
    """Stationary bootstrap (Politis & Romano 1994) of the monthly log changes: blocks of
    geometric length with mean `mean_block` (keeps the momentum and fat tails). With
    `demean`, the in-sample mean is removed (no drift, as the random walk)."""
    r = r[~np.isnan(r)]
    x = r - r.mean() if demean else r
    T = len(x)
    idx = np.empty((horizon, n), int)
    idx[0] = rng.integers(0, T, n)
    new = rng.random((horizon, n)) < 1 / mean_block
    starts = rng.integers(0, T, (horizon, n))
    for t in range(1, horizon):
        idx[t] = np.where(new[t], starts[t], (idx[t - 1] + 1) % T)
    return np.exp(np.cumsum(x[idx], axis=0)), {'mean_block': mean_block, 'demeaned': demean}


def mean_reverting_paths(lp: np.ndarray, horizon: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """AR(1) in the log price level (discrete Ornstein-Uhlenbeck), OLS with the Kendall
    (1954) bias correction phi + (1 + 3 phi) / T, capped below 1; residuals resampled."""
    lp = lp[~np.isnan(lp)]
    x0, x1 = lp[:-1], lp[1:]
    phi, c = np.polyfit(x0, x1, 1)
    T = len(x1)
    phi_bc = min(phi + (1 + 3 * phi) / T, 0.999)
    mean = np.mean(x1 - phi * x0) / (1 - phi)  # long-run level from the OLS fit
    c_bc = mean * (1 - phi_bc)
    e = x1 - (c_bc + phi_bc * x0)
    e = e - e.mean()
    out = np.empty((horizon, n))
    x = np.full(n, lp[-1])
    for t in range(horizon):
        x = c_bc + phi_bc * x + e[rng.integers(0, len(e), n)]
        out[t] = x
    return np.exp(out - lp[-1]), {'phi_ols': phi, 'phi': phi_bc, 'half_life_months': np.log(0.5) / np.log(phi_bc),
                                  'long_run_ratio': float(np.exp(mean - lp[-1]))}


def _sorted(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=-1)  # removes quantile crossing


def linear_qr(X: np.ndarray, y: np.ndarray, Xnew: np.ndarray, alpha: float = QR_ALPHA) -> np.ndarray:
    from sklearn.linear_model import QuantileRegressor
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z, Znew = (X - mu) / sd, (Xnew - mu) / sd
    out = np.empty((len(Xnew), len(TAUS)))
    for j, t in enumerate(TAUS):
        out[:, j] = QuantileRegressor(quantile=t, alpha=alpha, solver='highs').fit(Z, y).predict(Znew)
    return _sorted(out)


def xgb_qr(X: np.ndarray, y: np.ndarray, Xnew: np.ndarray, seed: int, params: dict | None = None,
           taus: np.ndarray = TAUS) -> np.ndarray:
    import xgboost as xgb
    prm = {**XGB_PARAMS, **(params or {})}
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=taus, tree_method='hist',
                         n_jobs=1, random_state=seed, **prm).fit(X, y)
    return _sorted(np.asarray(m.predict(Xnew)).reshape(len(Xnew), len(taus)))


# --------------------------------------------------------------------------- tuning
# Regularisation is chosen at every origin by expanding-window time-series CV inside the
# training rows. A validation block starts at row a; its training rows are those whose
# targets are realised by then (month + h <= month of row a), so overlapping h-month
# targets never leak into the fit. The criterion is the mean pinball loss at CV_TAUS.

CV_TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
QR_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
XGB_GRID = [dict(max_depth=d, n_estimators=n, min_child_weight=5) for d in (1, 2) for n in (50, 150, 300)]
CV_FOLDS, CV_MIN_TRAIN = 3, 18


def ts_folds(mi: np.ndarray, h: int, k: int = CV_FOLDS, min_train: int = CV_MIN_TRAIN):
    """(train, validation) row indices; `mi` are month ordinals of the rows, ascending."""
    n = len(mi)
    v = max(4, n // 6)
    for j in range(k, 0, -1):
        a, b = n - j * v, n - (j - 1) * v
        if a <= 0:
            continue
        tr = np.flatnonzero(mi + h <= mi[a])
        if len(tr) >= min_train:
            yield tr, np.arange(a, b)


def _pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray) -> float:
    e = y[:, None] - q
    return float(np.mean(np.maximum(taus * e, (taus - 1) * e)))


def tune_linear_qr(X, y, mi, h) -> tuple[float, int]:
    """Penalty alpha with the lowest CV pinball loss (QR_ALPHA if no fold is available);
    also returns the number of folds used."""
    from sklearn.linear_model import QuantileRegressor
    folds = list(ts_folds(mi, h))
    if not folds:
        return QR_ALPHA, 0
    loss = {}
    for alpha in QR_ALPHA_GRID:
        ls = []
        for tr, va in folds:
            mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
            sd[sd == 0] = 1.0
            q = np.column_stack([QuantileRegressor(quantile=t, alpha=alpha, solver='highs')
                                 .fit((X[tr] - mu) / sd, y[tr]).predict((X[va] - mu) / sd) for t in CV_TAUS])
            ls.append(_pinball(y[va], np.sort(q, axis=1), CV_TAUS))
        loss[alpha] = np.mean(ls)
    return min(loss, key=loss.get), len(folds)


def tune_xgb_qr(X, y, mi, h, seed) -> tuple[dict, int]:
    folds = list(ts_folds(mi, h))
    if not folds:
        return {}, 0
    loss = {}
    for k, prm in enumerate(XGB_GRID):
        ls = [_pinball(y[va], xgb_qr(X[tr], y[tr], X[va], seed, prm, CV_TAUS), CV_TAUS) for tr, va in folds]
        loss[k] = np.mean(ls)
    return XGB_GRID[min(loss, key=loss.get)], len(folds)


# --------------------------------------------------------------------------- features

OWN_FEATURES = ['own_r0', 'own_r1', 'own_c3', 'own_c6']
# driver -> (publication lag in months relative to the origin, windows of the log change)
DRIVER_SPEC = {'iron_ore_wb': (0, (1, 3)), 'scrap_ppi': (1, (1, 3)), 'brent_wb': (0, (1,)),
               'coal_wb': (0, (1,)), 'steel_ppi': (1, (3,)), 'fx': (0, (3,)), 'diesel_thb': (0, (1,))}


def feature_table(lp: pd.Series, drivers: pd.DataFrame, extra_lag: int = 0) -> pd.DataFrame:
    """Features known at the end of month s (the origin).

    Own history: log changes of the target in s and s-1, cumulative 3- and 6-month change.
    Drivers: log change over the stated window, ending `lag + extra_lag` months before s.
    The World Bank Pink Sheet, BOT FX and EPPO diesel for month s are published early in
    month s+1, before or with the TPSO report; US PPIs are published mid-month s+1, so
    they enter with a one-month lag."""
    r = lp.diff()
    f = pd.DataFrame(index=lp.index)
    f['own_r0'], f['own_r1'] = r, r.shift(1)
    f['own_c3'], f['own_c6'] = lp - lp.shift(3), lp - lp.shift(6)
    ld = np.log(drivers)
    for d, (lag, windows) in DRIVER_SPEC.items():
        for w in windows:
            f[f'drv_{d}_{w}m'] = (ld[d] - ld[d].shift(w)).shift(lag + extra_lag).reindex(lp.index)
    return f


def driver_features(f: pd.DataFrame, subset: tuple[str, ...] | None = None) -> list[str]:
    """Driver columns of a feature table, optionally only those of the drivers in `subset`."""
    return [c for c in f.columns if c.startswith('drv_')
            and (subset is None or any(c.startswith(f'drv_{d}_') for d in subset))]


# feature sets of the direct models: name -> drivers used (None = all in DRIVER_SPEC)
DRIVER_SETS = {'all drivers': None, 'scrap+ore': ('scrap_ppi', 'iron_ore_wb')}
