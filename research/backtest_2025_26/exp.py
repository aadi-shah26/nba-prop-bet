"""Out-of-sample improvement experiments: fit everything on 2024-25, report on 2025-26."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from analyze import F, bets, load, loglik, p_over, pit_cover

pd.set_option('display.width', 220)

tr, te = load(2024), load(2025)
STATS = list(te['stat'].unique())


def add_feats(d):
    d = d.copy()
    d['po_flag'] = (d['season_type'] == 'Playoffs').astype(float)
    d['b2b'] = (d['rest'] == 1).astype(float)
    d['miss_share'] = d['miss_min'] * d['minutes_proj'] / 240.0     # vacated minutes x player's share
    d['ret_share'] = d['ret_min_oth'] * d['minutes_proj'] / 240.0
    d['self_ret'] = d['self_return'].astype(float)
    d['min_gap'] = d['season_min_avg'] - d['minutes_proj']
    d['blow_rest'] = 0.0
    return d.fillna({'miss_share': 0, 'ret_share': 0, 'miss_min': 0, 'ret_min_oth': 0})


tr, te = add_feats(tr), add_feats(te)

# ---------------- minutes model (one row per player-game) ----------------
MIN_FEATS = ['minutes_proj', 'min_gap', 'miss_share', 'ret_share', 'b2b', 'home', 'po_flag', 'self_ret']


def fit_ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def pred_ols(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


g_tr = tr[tr['stat'] == 'PTS']
X = g_tr[MIN_FEATS].to_numpy(float)
X = np.column_stack([X, X[:, 0] * X[:, 6]])          # playoffs x minutes
beta_min = fit_ols(X, g_tr['minutes'].to_numpy(float))
print('minutes model coefs:', dict(zip(['const'] + MIN_FEATS + ['po_x_min'], np.round(beta_min, 3))))


def min_hat(d):
    X = d[MIN_FEATS].to_numpy(float)
    X = np.column_stack([X, X[:, 0] * X[:, 6]])
    return np.clip(pred_ols(beta_min, X), 1, 48)


for d in (tr, te):
    d['min_hat'] = min_hat(d)
g_te = te[te['stat'] == 'PTS']
print(f"minutes MAE 2025-26: current {np.abs(g_te.minutes - g_te.minutes_proj).mean():.3f}  "
      f"new {np.abs(g_te.minutes - g_te.min_hat).mean():.3f}   bias current "
      f"{(g_te.minutes_proj - g_te.minutes).mean():+.3f} new {(g_te.min_hat - g_te.minutes).mean():+.3f}")

# ---------------- usage bump: rate multiplier from teammates' missing production ----------------
USAGE_COL = {'PTS': 'miss_pts', 'AST': 'miss_ast'}


def fit_usage(d, stat):
    s = d[d['stat'] == stat]
    if stat not in USAGE_COL:
        return 0.0
    x = s[USAGE_COL[stat]].fillna(0).to_numpy() / 100.0
    base = s['min_hat'] * s['rate_x_opp']

    def loss(g):
        return np.abs(s['actual'] - base * (1 + g * x)).mean()
    return minimize_scalar(loss, bounds=(-1, 3), method='bounded').x


usage = {st: fit_usage(tr, st) for st in ['PTS', 'AST']}
print('usage multipliers (per 100 missing pts/ast):', {k: round(v, 3) for k, v in usage.items()})


def new_mean(d):
    m = d['min_hat'] * d['rate_x_opp']
    for st, col in USAGE_COL.items():
        mask = d['stat'] == st
        m = np.where(mask, m * (1 + usage[st] * d[col].fillna(0) / 100.0), m)
    # combos: rebuild from component means is not available row-wise; approximate with the
    # component-weighted PTS/AST bump folded in via the pts share of the combo
    return m


for d in (tr, te):
    d['mean_v2'] = new_mean(d)
    d['var_v2'] = d['mean_v2'] + d['alpha'] * d['mean_v2'] ** 2

# ---------------- per-stat linear recalibration of the mean ----------------
recal = {}
for st in STATS:
    s = tr[tr['stat'] == st]
    recal[st] = fit_ols(s[['mean_v2']].to_numpy(float), s['actual'].to_numpy(float))
for d in (tr, te):
    d['mean_v3'] = d['mean_v2'].copy()
    for st in STATS:
        m = d['stat'] == st
        d.loc[m, 'mean_v3'] = np.clip(pred_ols(recal[st], d.loc[m, ['mean_v2']].to_numpy(float)), 0.05, None)
print('recalibration a + b*mean:', {k: tuple(np.round(v, 3)) for k, v in recal.items()})

# ---------------- per-stat dispersion scale tuned on log-likelihood ----------------
vs = {}
for st in STATS:
    s = tr[tr['stat'] == st]

    def nll(k):
        v = s['mean_v3'] + k * s['alpha'] * s['mean_v3'] ** 2
        return -loglik(s['mean_v3'].to_numpy(), v.to_numpy(), s['actual'].to_numpy()).mean()
    vs[st] = minimize_scalar(nll, bounds=(0.3, 3), method='bounded').x
print('dispersion multipliers on current alpha:', {k: round(v, 2) for k, v in vs.items()})
for d in (tr, te):
    d['var_v3'] = d['mean_v3'] + d['stat'].map(vs) * d['alpha'] * d['mean_v3'] ** 2


# ---------------- report ----------------
def report(d, mcol, vcol):
    out = []
    for st, s in d.groupby('stat', sort=False):
        y, mu, v = s['actual'].to_numpy(), s[mcol].to_numpy(), s[vcol].to_numpy()
        po = p_over(mu, v, s['line'].to_numpy())
        hit = (y > s['line'].to_numpy()).astype(float)
        r = bets(s, 'line', mcol, vcol)
        out.append({'stat': st, 'mae': np.abs(y - mu).mean(), 'bias': (mu - y).mean(),
                    'll': loglik(mu, v, y).mean(), 'brier': ((po - hit) ** 2).mean(),
                    'cover80': pit_cover(mu, v, y)[0], 'bets': len(r), 'win': r['win'].mean(),
                    'roi': r['pnl'].mean()})
    return pd.DataFrame(out).set_index('stat')


variants = {'current model': ('mean', 'var'),
            '+ minutes model & usage': ('mean_v2', 'var_v2'),
            '+ mean recalibration & dispersion': ('mean_v3', 'var_v3')}
res = {k: report(te, *v) for k, v in variants.items()}
for k, r in res.items():
    print(f'\n--- 2025-26 test: {k} ---')
    print(r.to_string(float_format=F))
comp = pd.DataFrame({k: r['mae'] for k, r in res.items()})
comp['mae_change_%'] = 100 * (comp.iloc[:, -1] / comp.iloc[:, 0] - 1)
print('\nMAE by variant:\n', comp.to_string(float_format=F))
te.to_parquet(os.path.join(os.environ.get('BT_DIR', '.'), 'te_v3.parquet'))

# ---------------- ceiling check: gradient boosting on residual ratio ----------------
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = ['mean', 'minutes_proj', 'season_min_avg', 'season_avg', 'last10', 'last5', 'miss_min', 'miss_pts',
             'ret_min_oth', 'b2b', 'home', 'po_flag', 'self_ret', 'season_games', 'opp_rest', 'rest']
    print('\n--- GBM residual corrector (fit 2024-25, test 2025-26), target = actual - mean ---')
    for st in ['PTS', 'REB', 'AST', 'FG3M', 'PRA']:
        a, b = tr[tr['stat'] == st], te[te['stat'] == st]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=200, loss='absolute_error', random_state=0)
        m.fit(a[feats], a['actual'] - a['mean'])
        pred = np.clip(b['mean'] + m.predict(b[feats]), 0, None)
        print(f"{st:5} MAE current {np.abs(b.actual - b['mean']).mean():.3f}  v3 {np.abs(b.actual - b.mean_v3).mean():.3f}"
              f"  GBM {np.abs(b.actual - pred).mean():.3f}")
except ImportError:
    pass
