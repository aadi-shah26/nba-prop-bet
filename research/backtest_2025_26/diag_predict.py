"""Walk-forward predictions with extra context columns, parallel over players.
Usage: python diag_predict.py SEASON OUT.parquet   (NBA_DB_PATH must point at the DB)"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from backtest import load  # noqa: E402
from feature_engineering import (STATS, PropParams, league_priors, opponent_factors,  # noqa: E402
                                 project_from_history, team_games_with_allowed)
from db import connect  # noqa: E402

ALL = ['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV']
G = {}


def work(pids):
    pool, season, priors, params, opp_by_date = G['pool'], G['season'], G['priors'], G['params'], G['opp']
    rows = []
    for pid in pids:
        g = pool[pool['player_id'] == pid].reset_index(drop=True)
        idx = np.flatnonzero(g['season'].to_numpy() == season)
        for i in idx:
            game = g.iloc[i]
            hist = g.iloc[:i]
            cur = hist[hist['season'] == season]
            if len(cur) < 5:
                continue
            opp = opp_by_date[game['game_date']].get(game['opponent'])
            for stat in ALL:
                comps = list(STATS[stat])
                p = project_from_history(hist, stat, season, priors, params, opp_factor=opp)
                if p is None:
                    continue
                y_cur = cur[comps].sum(axis=1).to_numpy(dtype=float)
                rate_mean = p['mean'] / p['minutes_proj'] if p['minutes_proj'] > 0 else 0.0
                rows.append({
                    'player_id': pid, 'player_name': game['player_name'], 'game_id': game['game_id'],
                    'game_date': game['game_date'], 'season_type': game['season_type'],
                    'team': game['team'], 'opponent': game['opponent'], 'home': game['home'],
                    'stat': stat, 'actual': float(game[comps].sum()), 'minutes': game['minutes'],
                    'minutes_proj': p['minutes_proj'], 'mean': p['mean'], 'var': p['var'],
                    'rate_x_opp': rate_mean, 'season_games': len(cur),
                    'season_avg': y_cur.mean(), 'last10': y_cur[-10:].mean(), 'last5': y_cur[-5:].mean(),
                    'season_min_avg': cur['minutes'].mean(),
                    'prev_date': hist['game_date'].iloc[-1],
                })
    return rows


def main(season, out, params=None):
    params = params or PropParams()
    pg, tg = load(connect())
    tga = team_games_with_allowed(tg)
    priors = league_priors(pg[(pg['season'] < season) & (pg['season'] >= season - 2)])
    pool = pg[(pg['season'] >= season - params.lookback_seasons + 1) & (pg['season'] <= season)]
    dates = sorted(pool.loc[pool['season'] == season, 'game_date'].unique())
    opp = {d: opponent_factors(tga, d, season, params) for d in dates}
    G.update(pool=pool, season=season, priors=priors, params=params, opp=opp)
    pids = pool.loc[pool['season'] == season, 'player_id'].unique()
    chunks = [pids[i::32] for i in range(32)]
    with Pool(4) as p:
        res = p.map(work, chunks)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_parquet(out)
    print(season, len(df))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
