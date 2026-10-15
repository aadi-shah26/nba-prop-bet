"""Pregame context features per (player, game): rest, teammate absences / returns, starter."""
import os
import sqlite3

import numpy as np
import pandas as pd

S = os.environ.get('BT_DIR', os.path.dirname(os.path.abspath(__file__)))


def team_context(db=f'{S}/espn.db'):
    c = sqlite3.connect(db)
    pg = pd.read_sql_query("SELECT player_id, team, game_id, game_date, season, minutes, pts, ast, reb "
                           "FROM player_games ORDER BY game_date", c)
    tg = pd.read_sql_query("SELECT team, game_id, game_date, season, pts, plus_minus, opponent "
                           "FROM team_games ORDER BY game_date", c)
    tg['gd'] = pd.to_datetime(tg['game_date'])
    tg = tg.sort_values(['team', 'gd'])
    tg['rest'] = tg.groupby(['team', 'season'])['gd'].diff().dt.days
    tg['team_idx'] = tg.groupby(['team', 'season']).cumcount()
    # opponent rest
    o = tg[['game_id', 'team', 'rest']].rename(columns={'team': 'opponent', 'rest': 'opp_rest'})
    tg = tg.merge(o, on=['game_id', 'opponent'])

    # per player-team: recent minutes / pts per game when playing (strictly before), by team game index
    pg = pg.merge(tg[['team', 'game_id', 'team_idx']], on=['team', 'game_id'])
    pg = pg.sort_values(['player_id', 'season', 'team_idx'])
    grp = pg.groupby(['player_id', 'team', 'season'])
    for col in ('minutes', 'pts', 'ast'):
        pg[f'rec_{col}'] = grp[col].transform(lambda s: s.shift().ewm(halflife=5, min_periods=1).mean())

    # absences: for each team game t, players who played in team game t-1 (same team/season)
    # and do NOT play in t => "fresh absences"; players playing in t who did not play in t-1 but
    # played in >= 1 of t-2..t-10 => "returns". Weight by the player's recent minutes/pts.
    key = pg.set_index(['team', 'season', 'team_idx'])
    played = pg.groupby(['team', 'season', 'team_idx'])['player_id'].apply(set).to_dict()
    # latest recent averages for a player as of before team game t: use rec_ cols of the
    # player's most recent appearance + that game itself -> approximate with value at appearance
    last_vals = {}
    rows = []
    for (team, season), sub in tg.groupby(['team', 'season']):
        n = sub['team_idx'].max() + 1
        hist = {}  # pid -> list of (minutes, pts, ast) for appearances
        for t in range(n):
            cur = played.get((team, season, t), set())
            prev = played.get((team, season, t - 1), set())
            prev_window = set().union(*[played.get((team, season, k), set()) for k in range(max(0, t - 10), t - 1)])

            def exp_(pid, j):
                h = hist.get(pid)
                if not h:
                    return 0.0
                arr = np.array(h[-10:], dtype=float)
                w = 0.5 ** (np.arange(len(arr))[::-1] / 5)
                return float((w * arr[:, j]).sum() / w.sum())
            missing = prev - cur
            returning = (cur - prev) & prev_window
            rows.append({'team': team, 'season': season, 'team_idx': t,
                         'miss_min': sum(exp_(p, 0) for p in missing),
                         'miss_pts': sum(exp_(p, 1) for p in missing),
                         'miss_ast': sum(exp_(p, 2) for p in missing),
                         'ret_min': sum(exp_(p, 0) for p in returning),
                         'ret_pts': sum(exp_(p, 1) for p in returning),
                         'missing_ids': missing, 'returning_ids': returning})
            g = key.loc[(team, season, t)] if (team, season, t) in key.index else None
            if g is not None:
                g = g if isinstance(g, pd.DataFrame) else g.to_frame().T
                for r in g.itertuples():
                    hist.setdefault(r.player_id, []).append((r.minutes, r.pts, r.ast))
    ab = pd.DataFrame(rows)
    tg = tg.merge(ab, on=['team', 'season', 'team_idx'])
    return tg


def starters(raw_dir=f'{S}/raw'):
    out = []
    for y in (2024, 2025, 2026):
        p = pd.read_parquet(f'{raw_dir}/player_box_{y}.parquet', columns=['game_id', 'athlete_id', 'starter'])
        out.append(p)
    p = pd.concat(out)
    p['game_id'] = p['game_id'].astype(str)
    p['player_id'] = p['athlete_id'].astype('Int64')
    return p[['game_id', 'player_id', 'starter']].dropna().drop_duplicates(['game_id', 'player_id'])


if __name__ == '__main__':
    tg = team_context()
    tg.drop(columns=['missing_ids', 'returning_ids']).to_parquet(f'{S}/team_ctx.parquet')
    starters().to_parquet(f'{S}/starters.parquet')
    print(tg[['rest', 'miss_min', 'ret_min']].describe())
