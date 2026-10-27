"""Price-revision (jump) models, macro scenario generator and benchmark simulators.

TPSO list prices are step functions: they stay flat for months and then move by
a discrete revision. The proposed model treats the price as a marked point
process:

    log P(t+1) = log P(t) + J(t+1) * N(t+1)
    N(t+1) ~ Bernoulli(p_t),   p_t = XGBoost classifier(x_t)       (revision hazard)
    J(t+1) ~ revision size: XGBoost regressor + residual for materials with many
             revisions; the category's own observed revision sizes otherwise

where x_t holds the time since the item's last revision, the cumulative change in
the energy price (THB; Bangkok retail diesel by default, crude oil x FX as a check),
FX and policy rate since that revision, recent macro momentum, the
size of the last revision, the material category and the calendar month. Macro
drivers are simulated jointly, so XGBoost directly drives the long-horizon paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# A revision is any change in the reported price. TPSO reports prices to 0.01 THB (one
# satang), so every non-zero log change is a change of at least one satang; identical
# reported prices parse to identical floats, so unchanged months give exactly zero.
REV_TOL = 0.0
TICK = 0.01  # reporting unit of TPSO prices, THB
FEATURES = ['age', 'energy_gap', 'fx_gap', 'rate_gap', 'energy_r1', 'energy_r3', 'energy_r6',
            'fx_r3', 'rate', 'rate_ch6', 'last_jump', 'cat', 'month_next']
HIST = 6  # months of macro history needed for lagged features


def _mi(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Month ordinal (year*12 + month)."""
    d = pd.DatetimeIndex(dates)
    return (d.year * 12 + d.month).to_numpy()


def macro_feature_table(macro: pd.DataFrame) -> pd.DataFrame:
    """Macro levels and momentum keyed by month ordinal."""
    m = pd.DataFrame(index=_mi(macro.index))
    m['energy_l'] = np.log(macro['energy'].to_numpy())
    m['fx_l'] = np.log(macro['fx'].to_numpy())
    m['rate'] = macro['rate'].to_numpy()
    m['energy_r1'] = m.energy_l - m.energy_l.shift(1)
    m['energy_r3'] = m.energy_l - m.energy_l.shift(3)
    m['energy_r6'] = m.energy_l - m.energy_l.shift(6)
    m['fx_r3'] = m.fx_l - m.fx_l.shift(3)
    m['rate_ch6'] = m.rate - m.rate.shift(6)
    return m


def build_hazard_frame(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """One row per item-month t with features x_t and targets for t -> t+1."""
    p = panel[['date', 'code', 'cat', 'price']].copy()
    p['cat'] = p['cat'].astype(int)
    p['mi'] = _mi(p['date'])
    p = p.sort_values(['code', 'mi']).reset_index(drop=True)
    p['lp'] = np.log(p['price'])
    g = p.groupby('code', sort=False)
    contiguous = (p['mi'] - g['mi'].shift(1)) == 1
    p['dlp'] = np.where(contiguous, g['lp'].diff(), np.nan)
    p['rev'] = p['dlp'].abs() > REV_TOL

    first_mi = g['mi'].transform('min')
    p['anchor_mi'] = p['mi'].where(p['rev'])
    p['anchor_mi'] = p.groupby('code', sort=False)['anchor_mi'].ffill().fillna(first_mi).astype(int)
    p['last_jump'] = p['dlp'].where(p['rev'])
    p['last_jump'] = p.groupby('code', sort=False)['last_jump'].ffill().fillna(0.0)
    p['age'] = p['mi'] - p['anchor_mi']

    g = p.groupby('code', sort=False)
    p['target_ok'] = (g['mi'].shift(-1) - p['mi']) == 1
    p['rev_next'] = g['rev'].shift(-1).astype(float)
    p['size_next'] = g['dlp'].shift(-1)
    p['month_next'] = p['mi'] % 12 + 1

    mf = macro_feature_table(macro)
    p = p.join(mf, on='mi')
    anchor = mf[['energy_l', 'fx_l', 'rate']].add_suffix('_a')
    p = p.join(anchor, on='anchor_mi')
    p['energy_gap'] = p['energy_l'] - p['energy_l_a']
    p['fx_gap'] = p['fx_l'] - p['fx_l_a']
    p['rate_gap'] = p['rate'] - p['rate_a']
    return p


def _month_folds(mi: np.ndarray, n_folds: int = 3, min_train_frac: float = 0.5):
    """Expanding-window folds over calendar months (no look-ahead across items)."""
    months = np.unique(mi)
    start = int(len(months) * min_train_frac)
    edges = np.linspace(start, len(months), n_folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = mi < months[a]
        va = (mi >= months[a]) & (mi < months[b - 1] + 1)
        if tr.sum() and va.sum():
            yield tr, va


class HazardModel:
    """XGBoost classifier for P(revision next month | x_t)."""

    GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n)
            for d in (2, 3, 4) for lr in (0.03, 0.1) for n in (200, 500)]

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.params = dict(max_depth=3, learning_rate=0.05, n_estimators=300)
        self.cv_table: pd.DataFrame | None = None

    def _new(self, **kw):
        return xgb.XGBClassifier(objective='binary:logistic', subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=5, tree_method='hist', n_jobs=-1,
                                 random_state=self.seed, eval_metric='logloss', **kw)

    def fit(self, frame: pd.DataFrame, tune: bool = True) -> 'HazardModel':
        d = frame[frame['target_ok']]
        X, y, mi = d[FEATURES].to_numpy(float), d['rev_next'].to_numpy(int), d['mi'].to_numpy()
        if tune:
            rows = []
            for prm in self.GRID:
                losses = []
                for tr, va in _month_folds(mi):
                    m = self._new(**prm).fit(X[tr], y[tr])
                    pr = np.clip(m.predict_proba(X[va])[:, 1], 1e-6, 1 - 1e-6)
                    losses.append(-np.mean(y[va] * np.log(pr) + (1 - y[va]) * np.log(1 - pr)))
                rows.append({**prm, 'cv_logloss': float(np.mean(losses))})
            self.cv_table = pd.DataFrame(rows).sort_values('cv_logloss')
            best = self.cv_table.iloc[0]
            self.params = dict(max_depth=int(best.max_depth), learning_rate=float(best.learning_rate),
                               n_estimators=int(best.n_estimators))
        self.model = self._new(**self.params).fit(X, y)
        self.base_rate = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def shap_importance(self, frame: pd.DataFrame) -> pd.Series:
        d = frame[frame['target_ok']]
        contrib = self.model.get_booster().predict(xgb.DMatrix(d[FEATURES].to_numpy(float)), pred_contribs=True)
        return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=FEATURES).sort_values(ascending=False)


class SizeModel:
    """Size of a revision (log change).

    Categories with many revisions (>= `min_reg`, e.g. steel sections, precast
    elements) use an XGBoost regressor, monotone increasing in the energy cost gap,
    plus resampled out-of-fold residuals. Categories with few revisions (ready-mixed
    concrete has 7 list revisions in 2017-2026) resample their own observed revision
    sizes, so the regressor cannot impose sign patterns learned from other materials
    (e.g. steel price cuts) on them."""

    def __init__(self, seed: int = 0, min_reg: int = 500):
        self.seed, self.min_reg = seed, min_reg

    def _new(self):
        mono = tuple(1 if f == 'energy_gap' else 0 for f in FEATURES)
        return xgb.XGBRegressor(max_depth=2, learning_rate=0.05, n_estimators=300, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, tree_method='hist',
                                monotone_constraints=mono, n_jobs=-1, random_state=self.seed)

    def fit(self, frame: pd.DataFrame) -> 'SizeModel':
        d = frame[frame['target_ok'] & (frame['rev_next'] == 1)]
        X, y, cat = d[FEATURES].to_numpy(float), d['size_next'].to_numpy(float), d['cat'].to_numpy()
        counts = pd.Series(cat).value_counts()
        self.reg_cats = set(counts[counts >= self.min_reg].index)
        self.empirical = {c: y[cat == c] for c in counts.index if c not in self.reg_cats}
        oof = np.full(len(y), np.nan)
        fold = np.random.default_rng(self.seed).integers(0, 5, len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            oof[te] = self._new().fit(X[tr], y[tr]).predict(X[te])
        resid = y - oof
        self.resid = {c: resid[cat == c] for c in self.reg_cats}
        self.model = self._new().fit(X, y)
        self.n_train = len(y)
        return self

    def sample(self, X: np.ndarray, cat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(len(X))
        for c in np.unique(cat):
            idx = np.flatnonzero(cat == c)
            if c in self.reg_cats:
                r = self.resid[c]
                out[idx] = self.model.predict(X[idx]) + r[rng.integers(0, len(r), len(idx))]
            else:
                pool = self.empirical.get(c)
                if pool is None or len(pool) == 0:
                    pool = np.concatenate(list(self.empirical.values()))
                out[idx] = pool[rng.integers(0, len(pool), len(idx))]
        return out


# --------------------------------------------------------------------------- macro scenarios

def simulate_macro(macro: pd.DataFrame, horizon: int, n: int, rng: np.random.Generator,
                   energy_shift: float = 0.0, max_ar: float = 0.99) -> dict:
    """Joint scenario generator for the energy price (THB), THB/USD and the policy rate.

    log energy and log FX follow AR(1) processes fitted by OLS (AR coefficient capped
    at `max_ar` so 20-year paths stay stationary); the policy rate moves by its
    historical monthly changes. Shocks are resampled as whole historical months,
    which keeps the observed co-movement between the three drivers.
    `energy_shift` moves the long-run energy price level (e.g. +0.3 = 30% higher).
    Returned arrays have shape (HIST + 1 + horizon, n); row HIST is the origin."""
    lo, lf, r = np.log(macro['energy'].to_numpy()), np.log(macro['fx'].to_numpy()), macro['rate'].to_numpy()

    def ar1(x):
        b, a = np.polyfit(x[:-1], x[1:], 1)
        b = min(b, max_ar)
        a = np.mean(x[1:] - b * x[:-1])
        return a, b, x[1:] - (a + b * x[:-1])

    ao, bo, eo = ar1(lo)
    af, bf, ef = ar1(lf)
    ao += (1 - bo) * np.log1p(energy_shift)
    dr = np.diff(r)

    T = HIST + 1 + horizon
    energy = np.empty((T, n)); fx = np.empty((T, n)); rate = np.empty((T, n))
    energy[:HIST + 1] = lo[-(HIST + 1):, None]
    fx[:HIST + 1] = lf[-(HIST + 1):, None]
    rate[:HIST + 1] = r[-(HIST + 1):, None]
    for t in range(HIST + 1, T):
        k = rng.integers(0, len(eo), n)
        energy[t] = ao + bo * energy[t - 1] + eo[k]
        fx[t] = af + bf * fx[t - 1] + ef[k]
        rate[t] = np.clip(rate[t - 1] + dr[k], 0.0, 10.0)
    return {'energy_l': energy, 'fx_l': fx, 'rate': rate, 'energy': np.exp(energy), 'fx': np.exp(fx),
            'params': {'energy_ar': bo, 'energy_long_run_THB': float(np.exp(ao / (1 - bo))),
                       'fx_ar': bf, 'fx_long_run': float(np.exp(af / (1 - bf)))}}


# --------------------------------------------------------------------------- simulators

def simulate_hybrid(states: pd.DataFrame, mac: dict, origin_month: int, hazard: HazardModel,
                    size: SizeModel, horizon: int, rng: np.random.Generator,
                    hazard_mult: float = 1.0, size_mult: float = 1.0) -> np.ndarray:
    """Simulate price ratios P(t)/P(origin) for every item in `states` (one row per
    item, taken from build_hazard_frame at the origin). Macro path j is shared by all
    items in path j. Returns array (horizon, n_items, n_paths)."""
    m, n = len(states), mac['energy_l'].shape[1]
    lp = np.zeros((m, n))
    age = np.repeat(states['age'].to_numpy(float)[:, None], n, axis=1)
    last_jump = np.repeat(states['last_jump'].to_numpy(float)[:, None], n, axis=1)
    energy_a = np.repeat(states['energy_l_a'].to_numpy(float)[:, None], n, axis=1)
    fx_a = np.repeat(states['fx_l_a'].to_numpy(float)[:, None], n, axis=1)
    rate_a = np.repeat(states['rate_a'].to_numpy(float)[:, None], n, axis=1)
    cat = np.repeat(states['cat'].to_numpy(float)[:, None], n, axis=1)
    out = np.empty((horizon, m, n))
    O, F, R = mac['energy_l'], mac['fx_l'], mac['rate']
    for k in range(horizon):
        t = HIST + k
        month_next = (origin_month + k) % 12 + 1
        cols = {
            'age': age, 'energy_gap': O[t] - energy_a, 'fx_gap': F[t] - fx_a, 'rate_gap': R[t] - rate_a,
            'energy_r1': np.broadcast_to(O[t] - O[t - 1], (m, n)), 'energy_r3': np.broadcast_to(O[t] - O[t - 3], (m, n)),
            'energy_r6': np.broadcast_to(O[t] - O[t - 6], (m, n)), 'fx_r3': np.broadcast_to(F[t] - F[t - 3], (m, n)),
            'rate': np.broadcast_to(R[t], (m, n)), 'rate_ch6': np.broadcast_to(R[t] - R[t - 6], (m, n)),
            'last_jump': last_jump, 'cat': cat, 'month_next': np.full((m, n), month_next, float),
        }
        X = np.stack([np.asarray(cols[f], float).ravel() for f in FEATURES], axis=1)
        p = np.clip(hazard.predict(X) * hazard_mult, 0.0, 1.0)
        rev = rng.random(len(p)) < p
        jump = np.zeros(len(p))
        if rev.any():
            jump[rev] = size.sample(X[rev], X[rev, FEATURES.index('cat')], rng) * size_mult
        rev2, jump2 = rev.reshape(m, n), jump.reshape(m, n)
        lp += jump2
        age = np.where(rev2, 0.0, age + 1)
        last_jump = np.where(rev2, jump2, last_jump)
        energy_a = np.where(rev2, O[t + 1], energy_a)
        fx_a = np.where(rev2, F[t + 1], fx_a)
        rate_a = np.where(rev2, R[t + 1], rate_a)
        out[k] = lp
    return np.exp(out)


def simulate_gbm(mu: float, sigma: float, horizon: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """GBM price ratios with monthly drift mu and volatility sigma (exact discretisation)."""
    z = rng.standard_normal((horizon, n))
    return np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * z, axis=0))


def gbm_params_from_history(dlp: np.ndarray) -> tuple[float, float]:
    """Monthly GBM drift/volatility implied by historical log changes."""
    dlp = dlp[~np.isnan(dlp)]
    s = dlp.std(ddof=1)
    return dlp.mean() + 0.5 * s ** 2, s


def simulate_compound_poisson(dlp: np.ndarray, horizon: int, n: int, rng: np.random.Generator,
                              pool: np.ndarray | None = None, min_jumps: int = 3) -> np.ndarray:
    """Constant-intensity jump process: revisions arrive with the item's historical
    monthly frequency; sizes are resampled from its own past revisions (or from
    `pool` when it has fewer than `min_jumps`)."""
    dlp = dlp[~np.isnan(dlp)]
    jumps = dlp[np.abs(dlp) > REV_TOL]
    lam = len(jumps) / max(len(dlp), 1)
    if len(jumps) < min_jumps and pool is not None and len(pool):
        jumps = pool
    if len(jumps) == 0:
        return np.ones((horizon, n))
    hit = rng.random((horizon, n)) < lam
    size = jumps[rng.integers(0, len(jumps), (horizon, n))]
    return np.exp(np.cumsum(np.where(hit, size, 0.0), axis=0))


# --------------------------------------------------------------------------- alternative processes (R2-9)
# Fitted to each item's monthly log changes. A failed fit falls back to the data-calibrated
# GBM so that it can still be scored (the fallback is counted).
#
# Grouped-data likelihood. TPSO prices are reported to the nearest satang, so a reported
# change r is only known to lie in an interval [r - d, r + d] with
# d = (TICK/2)/P_t + (TICK/2)/P_{t-1} (first-order bound on the log change). The likelihood
# of an observation is therefore the probability of that interval, not the density at r
# (Sheppard 1898; Heitjan 1989; Zhao & Bai 2012). This probability is at most one, so the
# likelihood stays bounded when a standard deviation goes to zero, which is what makes the
# point-density likelihood of these mostly-unchanged series unbounded (Hathaway 1985;
# Chen, Tan & Zhang 2008). `sd_floor` is then only a numerical lower bound; its effect is
# tested in sensitivity_sd_floor.py.

SD_FLOOR = 1e-5


def _clean(dlp: np.ndarray) -> np.ndarray:
    return dlp[~np.isnan(dlp)]


def _gbm_fallback(dlp, horizon, n, rng, reason):
    return simulate_gbm(*gbm_params_from_history(dlp), horizon, n, rng), {'ok': False, 'reason': reason}


def _intervals(dlp: np.ndarray, price: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed log changes and the rounding interval each one stands for."""
    ok = ~np.isnan(dlp)
    r, p = dlp[ok], price[ok]
    d = (TICK / 2) / p + (TICK / 2) / (p * np.exp(-r))
    return r, r - d, r + d


def _interval_prob(lo, hi, m, s):
    """P(lo < X < hi) for X ~ N(m, s^2), computed on the side that avoids cancellation."""
    from scipy.special import ndtr
    a, b = (lo - m) / s, (hi - m) / s
    return np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))


def fit_simulate_merton(dlp: np.ndarray, horizon: int, n: int, rng: np.random.Generator,
                        price: np.ndarray | None = None, sd_floor: float = SD_FLOOR,
                        max_jumps: int = 3) -> tuple[np.ndarray, dict]:
    """Merton jump-diffusion for monthly log changes, fitted by grouped-data maximum likelihood:
    r = mu + sigma*z + sum_{k=1}^{N} J_k,  N ~ Poisson(lam),  J_k ~ N(mu_j, sigma_j^2)."""
    from scipy.optimize import minimize
    from scipy.stats import poisson
    r, lo, hi = _intervals(dlp, price)
    if len(r) < 24 or np.all(np.abs(r) <= REV_TOL):
        return _gbm_fallback(r, horizon, n, rng, 'no revisions')
    jumps = r[np.abs(r) > REV_TOL]
    ks = np.arange(max_jumps + 1)[:, None]

    def nll(th):
        mu, ls, lam, mj, lsj = th
        s = np.sqrt(np.exp(2 * ls) + ks * np.exp(2 * lsj))
        pr = poisson.pmf(ks, lam) * _interval_prob(lo[None, :], hi[None, :], mu + ks * mj, s)
        return -np.sum(np.log(pr.sum(axis=0) + 1e-300))

    lf = np.log(sd_floor)
    bounds = [(-0.05, 0.05), (lf, np.log(0.2)), (1e-4, 1.0), (-0.5, 0.5), (lf, np.log(0.5))]
    rate, js = len(jumps) / len(r), max(jumps.std(), 5e-3)
    # Multi-start: the mixture likelihood has local optima (e.g. lambda -> 1 with small
    # jumps); keep the start with the highest likelihood.
    starts = [(rate, js), (rate / 2, 2 * js), (min(2 * rate, 0.5), js / 2)]
    res = min((minimize(nll, [0.0, max(np.log(1e-3), lf), lam0, jumps.mean(), max(np.log(sj0), lf)],
                        method='L-BFGS-B', bounds=bounds) for lam0, sj0 in starts), key=lambda x: x.fun)
    mu, ls, lam, mj, lsj = res.x
    s, sj = np.exp(ls), np.exp(lsj)
    N = rng.poisson(lam, (horizon, n))
    steps = mu + s * rng.standard_normal((horizon, n)) + N * mj + np.sqrt(N) * sj * rng.standard_normal((horizon, n))
    info = {'ok': bool(res.success), 'mu': mu, 'sigma': s, 'lambda': lam, 'mu_j': mj, 'sigma_j': sj,
            'loglik': -res.fun, 'sigma_at_bound': bool(np.isclose(s, sd_floor, rtol=1e-3))}
    return np.exp(np.cumsum(steps, axis=0)), info


def fit_simulate_garch(dlp: np.ndarray, horizon: int, n: int, rng: np.random.Generator,
                       **_) -> tuple[np.ndarray, dict]:
    """GARCH(1,1) with constant mean and normal errors on monthly log changes (in %)."""
    import warnings
    from arch import arch_model
    r = _clean(dlp)
    if len(r) < 24 or np.all(np.abs(r) <= REV_TOL):
        return _gbm_fallback(r, horizon, n, rng, 'no revisions')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            res = arch_model(100 * r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal',
                             rescale=False).fit(disp='off')
        except Exception as e:  # noqa: BLE001
            return _gbm_fallback(r, horizon, n, rng, f'fit error: {type(e).__name__}')
    p = res.params
    mu, om, a, b = p['mu'], p['omega'], p['alpha[1]'], p['beta[1]']
    if not np.all(np.isfinite([mu, om, a, b])) or res.convergence_flag != 0:
        return _gbm_fallback(r, horizon, n, rng, 'not converged')
    s2 = np.full(n, om + a * res.resid[-1] ** 2 + b * res.conditional_volatility[-1] ** 2)
    out = np.empty((horizon, n))
    for t in range(horizon):
        e = np.sqrt(s2) * rng.standard_normal(n)
        out[t] = (mu + e) / 100
        s2 = om + a * e ** 2 + b * s2
    info = {'ok': True, 'mu': mu / 100, 'omega': om, 'alpha': a, 'beta': b, 'persistence': a + b}
    return np.exp(np.cumsum(out, axis=0)), info


def _hmm2_filter(lo, hi, mu, sd, A):
    """Hamilton filter for a two-regime Gaussian Markov-switching model with grouped
    (interval) observations. Returns the log-likelihood and the last filtered regime
    probabilities. A[i, j] = Pr(next regime = j | current = i)."""
    B = np.stack([_interval_prob(lo, hi, mu[k], sd[k]) for k in (0, 1)], axis=1) + 1e-300
    p1 = A[0, 1] / max(A[0, 1] + A[1, 0], 1e-12)  # stationary probability of regime 1
    a = np.array([1 - p1, p1]) * B[0]
    ll = np.log(a.sum()); a /= a.sum()
    for t in range(1, len(lo)):
        a = (a @ A) * B[t]
        s = a.sum(); ll += np.log(s); a /= s
    return ll, a


def fit_simulate_regime(dlp: np.ndarray, horizon: int, n: int, rng: np.random.Generator,
                        price: np.ndarray | None = None, sd_floor: float = SD_FLOOR) -> tuple[np.ndarray, dict]:
    """Two-regime Markov-switching model (regime-specific mean and variance; Hamilton 1989)
    fitted by maximising the grouped-data likelihood from the Hamilton filter."""
    from scipy.optimize import minimize
    r, lo, hi = _intervals(dlp, price)
    if len(r) < 24 or np.all(np.abs(r) <= REV_TOL):
        return _gbm_fallback(r, horizon, n, rng, 'no revisions')
    rev = np.abs(r) > REV_TOL
    lf = np.log(sd_floor)

    def unpack(th):
        mu = th[:2]; sd = np.exp(th[2:4])
        p00, p11 = 1 / (1 + np.exp(-th[4])), 1 / (1 + np.exp(-th[5]))
        return mu, sd, np.array([[p00, 1 - p00], [1 - p11, p11]])

    def nll(th):
        return -_hmm2_filter(lo, hi, *unpack(th))[0]

    sd1 = r[rev].std() if rev.sum() > 1 else 0.01
    p00 = np.clip(1 - rev.mean(), 0.05, 0.95)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (lf, np.log(0.5)), (lf, np.log(0.5)), (-15, 15), (-15, 15)]
    # Multi-start over the persistence of the revision regime (logit p11 = -6, -2, 0, 2);
    # keep the start with the highest likelihood.
    try:
        res = min((minimize(nll, [0.0, r[rev].mean(), lf + 1, max(np.log(max(sd1, 1e-3)), lf),
                                  np.log(p00 / (1 - p00)), a11], method='L-BFGS-B', bounds=bounds)
                   for a11 in (-6.0, -2.0, 0.0, 2.0)), key=lambda x: x.fun)
    except Exception as e:  # noqa: BLE001
        return _gbm_fallback(r, horizon, n, rng, f'fit error: {type(e).__name__}')
    mu, sd, A = unpack(res.x)
    ll, last = _hmm2_filter(lo, hi, mu, sd, A)
    if not (np.all(np.isfinite(res.x)) and np.isfinite(ll)):
        return _gbm_fallback(r, horizon, n, rng, 'not converged')
    state = (rng.random(n) < last[1]).astype(int)
    out = np.empty((horizon, n))
    for t in range(horizon):
        state = np.where(rng.random(n) < A[state, 1], 1, 0)
        out[t] = mu[state] + sd[state] * rng.standard_normal(n)
    info = {'ok': bool(res.success), 'mu': mu.round(5).tolist(), 'sd': sd.round(6).tolist(),
            'p00': float(A[0, 0]), 'p11': float(A[1, 1]), 'loglik': float(ll),
            'sd_at_floor': bool(np.any(np.isclose(sd, sd_floor, rtol=1e-3)))}
    return np.exp(np.cumsum(out, axis=0)), info


ALT_MODELS = {'Merton jump-diffusion': fit_simulate_merton,
              'GARCH(1,1)': fit_simulate_garch,
              'Markov regime-switching': fit_simulate_regime}


# --------------------------------------------------------------------------- scoring

def crps_sample(samples: np.ndarray, y: float) -> float:
    """Sample CRPS: E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(samples)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n ** 2)


def pinball(q: float, y: float, tau: float) -> float:
    return float((y - q) * tau if y >= q else (q - y) * (1 - tau))
