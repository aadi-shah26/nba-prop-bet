"""Walk-forward game predictions for 2024-25 and 2025-26 (cached), with the extra pregame
features the experiments try: rest (days, 3-in-4), lineup deltas per team, scoring-only lineup
deltas, and games played so far."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import game_model as gm  # noqa: E402

OUT = os.environ.get('BT_OUT', os.path.dirname(os.path.abspath(__file__)))


def load():
    c = sqlite3.connect(os.path.join(ROOT, 'nba_data.db'))
    return (pd.read_sql_query("SELECT * FROM team_games", c), pd.read_sql_query("SELECT * FROM player_games", c))


def extras(tg, pg, params):
    """Per (game_id, team): rest days, 3-in-4, games played this season, lineup delta and a
    points-only lineup delta (same construction as gm.lineup_history but on points)."""
    t = tg[['game_id', 'team', 'game_date', 'season']].drop_duplicates().copy()
    t['dt'] = pd.to_datetime(t['game_date'])
    t = t.sort_values('dt')
    g = t.groupby('team')['dt']
    t['rest'] = g.diff().dt.days.clip(upper=4).fillna(4)
    t['three_in_four'] = ((t['dt'] - g.shift(2)).dt.days <= 3).astype(int)
    t['gp'] = t.groupby(['team', 'season']).cumcount()
    vals = gm.player_values(pg, params)
    lh = gm.lineup_history(vals, params)
    # points version: player's EWM points instead of game score
    p = pg[pg['minutes'] > 0].sort_values(['game_date', 'game_id']).reset_index(drop=True)
    p = p[['player_id', 'team', 'game_id', 'game_date', 'pts']].copy()
    post = p.groupby('player_id')['pts'].transform(
        lambda x: x.ewm(halflife=params.player_half_life, min_periods=params.player_min_games).mean())
    p['val_pre'] = post.groupby(p['player_id']).shift().fillna(0.0)
    lp = gm.lineup_history(p.assign(player_name=''), params)
    t = t.merge(lh[['game_id', 'team', 'delta']], on=['game_id', 'team'], how='left') \
         .merge(lp[['game_id', 'team', 'delta']].rename(columns={'delta': 'pts_delta'}), on=['game_id', 'team'], how='left')
    return t.fillna({'delta': 0.0, 'pts_delta': 0.0})


def build(params=None, seasons=(2024, 2025), tag='base'):
    path = os.path.join(OUT, f'preds_{tag}.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    params = params or gm.GameParams()
    tg, pg = load()
    df = gm.backtest_games(tg, list(seasons), params, player_games=pg)
    x = extras(tg, pg, params)
    for side, team in (('h', 'home'), ('a', 'away')):
        df = df.merge(x.drop(columns=['dt', 'game_date', 'season']).add_suffix('_' + side)
                      .rename(columns={f'game_id_{side}': 'game_id', f'team_{side}': team}),
                      on=['game_id', team], how='left')
    df.to_pickle(path)
    return df


if __name__ == '__main__':
    import time
    t = time.time()
    df = build()
    print(len(df), 'games', f'{time.time() - t:.0f}s')
    print(df.groupby('season')[['pred_total', 'act_total']].mean())
