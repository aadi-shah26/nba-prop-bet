"""Diagnostics + out-of-sample improvement experiments. Fit on 2024-25, test on 2025-26."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

S = os.environ.get('BT_DIR', os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
F = lambda x: f'{x:.3f}'  # noqa: E731
BE = 110 / 210            # break-even at -110
P_BET = (1.03) / (1 + 100 / 110)   # p needed for +3% EV at -110 (the app's default MIN_EV)


def nb_params(mu, var):
    mu = np.maximum(mu, 1e-6)
    var = np.maximum(var, mu * (1 + 1e-6))
    n = mu ** 2 / (var - mu)
    return n, n / (n + mu)


def p_over(mu, var, line):
    n, p = nb_params(mu, var)
    return sps.nbinom.sf(np.floor(line), n, p)


def loglik(mu, var, y):
    n, p = nb_params(mu, var)
    return sps.nbinom.logpmf(y, n, p)


def pit_cover(mu, var, y, seed=0):
    n, p = nb_params(mu, var)
    u = sps.nbinom.cdf(y - 1, n, p) + np.random.default_rng(seed).uniform(size=len(y)) * sps.nbinom.pmf(y, n, p)
    return ((u >= .1) & (u <= .9)).mean(), u


def load(season):
    d = pd.read_parquet(f'{S}/pred_{season}.parquet')
    tc = pd.read_parquet(f'{S}/team_ctx.parquet')
    st = pd.read_parquet(f'{S}/starters.parquet')
    tc['game_id'] = tc['game_id'].astype(str)
    d['game_id'] = d['game_id'].astype(str)
    prev = tc.sort_values('gd').groupby(['team', 'season'])['game_date'].shift()
    tc['team_prev_date'] = prev
    d = d.merge(tc[['team', 'game_id', 'rest', 'opp_rest', 'miss_min', 'miss_pts', 'miss_ast', 'ret_min',
                    'ret_pts', 'plus_minus', 'team_prev_date', 'team_idx']], on=['team', 'game_id'], how='left')
    d = d.merge(st.astype({'player_id': int}), on=['game_id', 'player_id'], how='left')
    d['self_return'] = (d['prev_date'] != d['team_prev_date']) & d['team_prev_date'].notna()
    d['ret_min_oth'] = np.where(d['self_return'], np.maximum(d['ret_min'] - d['minutes_proj'], 0), d['ret_min'])
    d['alpha'] = (d['var'] - d['mean']) / d['mean'].clip(lower=1e-6) ** 2
    d['line'] = np.floor(d['season_avg']) + 0.5
    d['line_l10'] = np.floor(d['last10']) + 0.5
    d['month'] = d['game_date'].str[:7]
    d['b2b'] = d['rest'] == 1
    return d


def summarize(d, mean_col='mean', var_col='var', label='model'):
    out = []
    for stat, s in d.groupby('stat', sort=False):
        y = s['actual'].to_numpy()
        mu, var = s[mean_col].to_numpy(), s[var_col].to_numpy()
        po = p_over(mu, var, s['line'].to_numpy())
        hit = (y > s['line'].to_numpy()).astype(float)
        cov, _ = pit_cover(mu, var, y)
        out.append({'stat': stat, 'n': len(s), f'mae_{label}': np.abs(y - mu).mean(),
                    'mae_season': np.abs(y - s['season_avg']).mean(), 'mae_l10': np.abs(y - s['last10']).mean(),
                    'bias': (mu - y).mean(), 'brier': ((po - hit) ** 2).mean(),
                    'brier_coin': ((0.5 - hit) ** 2).mean(), 'll': loglik(mu, var, y).mean(), 'cover80': cov})
    return pd.DataFrame(out)


def bets(d, line_col='line', mean_col='mean', var_col='var', thr=P_BET):
    """Flat 1u bets at -110 on the model's side vs a proxy line when P(side) >= thr."""
    po = p_over(d[mean_col].to_numpy(), d[var_col].to_numpy(), d[line_col].to_numpy())
    y, L = d['actual'].to_numpy(), d[line_col].to_numpy()
    side = np.where(po >= thr, 1, np.where(1 - po >= thr, -1, 0))
    win = np.where(side == 1, y > L, y < L)
    r = pd.DataFrame({'stat': d['stat'].to_numpy(), 'side': side, 'win': win, 'p': np.where(side == 1, po, 1 - po)})
    r = r[r['side'] != 0]
    r['pnl'] = np.where(r['win'], 100 / 110, -1.0)
    return r


def bet_table(r):
    g = r.groupby('stat').agg(bets=('win', 'size'), win=('win', 'mean'), roi=('pnl', 'mean'),
                              share_under=('side', lambda s: (s == -1).mean()), p_avg=('p', 'mean'))
    tot = pd.DataFrame([{'bets': len(r), 'win': r['win'].mean(), 'roi': r['pnl'].mean(),
                         'share_under': (r['side'] == -1).mean(), 'p_avg': r['p'].mean()}], index=['ALL'])
    return pd.concat([g, tot])


def seg(d, col, bins=None, labels=None):
    x = pd.cut(d[col], bins, labels=labels) if bins is not None else d[col]
    e = d['actual'] - d['mean']
    return pd.DataFrame({'n': d.groupby(x, observed=True).size(),
                         'mean_proj': d.groupby(x, observed=True)['mean'].mean(),
                         'mean_act': d.groupby(x, observed=True)['actual'].mean(),
                         'bias_pct': e.groupby(x, observed=True).sum() / d.groupby(x, observed=True)['mean'].sum(),
                         'mae': e.abs().groupby(x, observed=True).mean(),
                         'min_err': (d['minutes'] - d['minutes_proj']).groupby(x, observed=True).mean()})


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    d = load(2025)
    print('rows', len(d), 'player-games', d[['player_id', 'game_id']].drop_duplicates().shape[0],
          'players', d['player_id'].nunique(), d['game_date'].min(), d['game_date'].max())
    reg = d[d['season_type'] != 'Playoffs']

    print('\n=== 1. ACCURACY, 2025-26 (all games) ===')
    print(summarize(d).to_string(index=False, float_format=F))
    print('\n=== 1b. playoffs only ===')
    print(summarize(d[d['season_type'] == 'Playoffs']).to_string(index=False, float_format=F))

    print('\n=== 2. BETS vs proxy line floor(season avg)+.5 at -110, +3% EV rule ===')
    r = bets(d)
    print(bet_table(r).to_string(float_format=F))
    print('\n always-UNDER baseline at the same proxy line:')
    u = d.assign(win=d['actual'] < d['line']).groupby('stat')['win'].mean()
    print(u.to_string(float_format=F), ' ALL', F(d['actual'].lt(d['line']).mean()))
    print('\n by model-confidence bucket (all stats):')
    r['pb'] = pd.cut(r['p'], [P_BET, .58, .62, .66, .70, 1])
    print(r.groupby('pb', observed=True).agg(n=('win', 'size'), p=('p', 'mean'), win=('win', 'mean'),
                                             roi=('pnl', 'mean')).to_string(float_format=F))
    print('\n vs alternative proxy line floor(last10)+.5:')
    print(bet_table(bets(d, 'line_l10')).to_string(float_format=F))

    print('\n=== 3. Error decomposition: what if minutes were known exactly? ===')
    d['mean_kmin'] = d['minutes'] * d['rate_x_opp']
    for stat, s in d.groupby('stat', sort=False):
        print(f"{stat:5} MAE {np.abs(s.actual - s['mean']).mean():.3f} -> known-minutes {np.abs(s.actual - s.mean_kmin).mean():.3f}"
              f"   minutes MAE {np.abs(s.minutes - s.minutes_proj).mean():.2f}")

    p = d[d['stat'] == 'PTS'].copy()
    print('\n=== 4. Segments (PTS) — bias_pct = (actual - proj)/proj ===')
    print('\nprojected minutes'); print(seg(p, 'minutes_proj', [0, 15, 22, 28, 33, 48]).to_string(float_format=F))
    print('\nrest days (team)'); print(seg(p, 'rest', [0, 1, 2, 3, 30], ['b2b', '1 day', '2 days', '3+']).to_string(float_format=F))
    print('\nhome'); print(seg(p, 'home').to_string(float_format=F))
    print('\nstarter'); print(seg(p, 'starter').to_string(float_format=F))
    print('\nteammate minutes freshly missing (vs team\'s previous game)')
    print(seg(p, 'miss_min', [-1, 0.1, 15, 30, 60, 300]).to_string(float_format=F))
    print('\nteammate minutes returning')
    print(seg(p, 'ret_min_oth', [-1, 0.1, 15, 30, 60, 300]).to_string(float_format=F))
    print('\nplayer returning himself (missed team\'s previous game)')
    print(seg(p, 'self_return').to_string(float_format=F))
    print('\nfinal margin (NOT pregame — shows blowout risk)')
    print(seg(p.assign(am=p.plus_minus.abs()), 'am', [-1, 5, 10, 15, 20, 80]).to_string(float_format=F))
    print('\nmonth'); print(seg(p, 'month').to_string(float_format=F))
    print('\nseason games played so far'); print(seg(p, 'season_games', [4, 10, 20, 40, 100]).to_string(float_format=F))
    print('\nprojection level'); print(seg(p, 'mean', [0, 5, 10, 15, 20, 25, 50]).to_string(float_format=F))
    print('\nplayer\'s recent form: last5 - season avg')
    print(seg(p.assign(f=p.last5 - p.season_avg), 'f', [-50, -5, -2, 2, 5, 50]).to_string(float_format=F))
    for stat in ['REB', 'AST', 'FG3M']:
        q = d[d['stat'] == stat]
        print(f'\n{stat}: miss_min'); print(seg(q, 'miss_min', [-1, 0.1, 15, 30, 60, 300]).to_string(float_format=F))
        print(f'{stat}: projection level'); print(seg(q, 'mean', np.quantile(q['mean'], [0, .2, .4, .6, .8, 1])).to_string(float_format=F))

    print('\n=== 5. Calibration of P(over) at proxy line, by stat (pred vs actual) ===')
    d['po'] = p_over(d['mean'].to_numpy(), d['var'].to_numpy(), d['line'].to_numpy())
    d['hit'] = (d['actual'] > d['line']).astype(float)
    cut = pd.cut(d['po'], [0, .3, .4, .45, .5, .55, .6, .7, 1])
    print(d.groupby(cut, observed=True).agg(n=('hit', 'size'), pred=('po', 'mean'), act=('hit', 'mean')).to_string(float_format=F))
    print(d.groupby('stat').agg(pred=('po', 'mean'), act=('hit', 'mean')).to_string(float_format=F))
