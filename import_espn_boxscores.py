#!/usr/bin/env python3
"""
Fallback data source: ESPN box scores published by sportsdataverse on GitHub.

stats.nba.com blocks many cloud / data-center IPs. The sportsdataverse project republishes
every ESPN NBA player and team box score as one parquet file per season on GitHub, which is
reachable almost everywhere. This script loads those files into the same player_games /
team_games tables the rest of the repo uses, so the backtest (and the app) can run on them.

Differences from stats.nba.com data (small for modelling purposes):
  * player_id / team_id are ESPN ids, not NBA ids -> keep this in its own database
    (NBA_DB_PATH or --db) rather than mixing it with an NBA-sourced one
  * minutes are whole minutes
  * team turnovers include team turnovers (like stats.nba.com's team TOV)

Usage:
    python import_espn_boxscores.py --seasons 2023 2024 2025 --db espn.db
    NBA_DB_PATH=espn.db python backtest.py props --tune 2024 --seasons 2025
    python import_espn_boxscores.py --seasons 2025 --db espn.db --dir ./parquet   # local files
"""

import argparse
import io
from pathlib import Path

import pandas as pd

from db import connect, init_db, season_str
from fetch_game_logs import PLAYER_COLS, TEAM_COLS, upsert

BASE_URL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
PLAYER_URL = BASE_URL + '/espn_nba_player_boxscores/player_box_{y}.parquet'
TEAM_URL = BASE_URL + '/espn_nba_team_boxscores/team_box_{y}.parquet'

# ESPN abbreviation -> stats.nba.com abbreviation (the rest are identical)
ESPN_TEAM = {'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}
# ESPN season_type: 2 regular season, 3 playoffs, 5 play-in (1 preseason, 4 all-star: dropped)
SEASON_TYPE = {2: 'Regular Season', 3: 'Playoffs', 5: 'PlayIn'}

_STAT_MAP = {'points': 'pts', 'assists': 'ast', 'steals': 'stl', 'blocks': 'blk',
             'three_point_field_goals_made': 'fg3m', 'field_goals_made': 'fgm',
             'field_goals_attempted': 'fga', 'free_throws_made': 'ftm',
             'free_throws_attempted': 'fta', 'offensive_rebounds': 'oreb',
             'defensive_rebounds': 'dreb', 'fouls': 'pf'}


def _read(src):
    if str(src).startswith('http'):
        import requests
        r = requests.get(src, timeout=120)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    return pd.read_parquet(src)


def _team(abbr):
    return abbr.map(lambda a: ESPN_TEAM.get(a, a))


def _keep(df):
    known = set(ESPN_TEAM.values()) | {'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN',
                                       'DET', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL',
                                       'MIN', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'TOR'}
    team = _team(df['team_abbreviation'])
    opp = _team(df['opponent_team_abbreviation'])
    return df[df['season_type'].isin(SEASON_TYPE) & team.isin(known) & opp.isin(known)]


def _common(df, season):
    out = pd.DataFrame(index=df.index)
    out['team'] = _team(df['team_abbreviation'])
    out['opponent'] = _team(df['opponent_team_abbreviation'])
    out['team_id'] = df['team_id'].astype(int)
    out['game_id'] = df['game_id'].astype(str)
    out['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    out['season'] = int(season)
    out['season_type'] = df['season_type'].map(SEASON_TYPE)
    out['wl'] = df['team_winner'].map({True: 'W', False: 'L'})
    return out


def normalize_players(raw, season):
    df = _keep(raw)
    df = df[df['minutes'].fillna(0) > 0]
    out = _common(df, season)
    home = df['home_away'].eq('home').astype(int)
    out['home'] = home
    out['player_id'] = df['athlete_id'].astype(int)
    out['player_name'] = df['athlete_display_name'].astype(str)
    out['minutes'] = df['minutes'].astype(float)
    out['reb'] = df['rebounds'].fillna(0).astype(int)
    out['tov'] = df['turnovers'].fillna(0).astype(int)
    for src, dst in _STAT_MAP.items():
        out[dst] = df[src].fillna(0).astype(int)
    out['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce')
    return out.drop_duplicates(['player_id', 'game_id'], keep='last')[PLAYER_COLS]


def normalize_teams(raw, season, players):
    df = _keep(raw)
    out = _common(df, season)
    out['team_name'] = df['team_display_name'].astype(str)
    out['home'] = df['team_home_away'].eq('home').astype(int)
    out['pts'] = df['team_score'].astype(int)
    out['reb'] = df['total_rebounds'].fillna(0).astype(int)
    out['tov'] = df['total_turnovers'].fillna(df['turnovers']).fillna(0).astype(int)
    out['fg3a'] = df['three_point_field_goals_attempted'].fillna(0).astype(int)
    for src, dst in _STAT_MAP.items():
        if dst != 'pts':
            out[dst] = df[src].fillna(0).astype(int)
    out['plus_minus'] = (df['team_score'] - df['opponent_team_score']).astype(float)
    mins = players.groupby(['game_id', 'team'])['minutes'].sum()
    out['minutes'] = [mins.get((g, t)) for g, t in zip(out['game_id'], out['team'])]
    out = out.drop_duplicates(['team_id', 'game_id'], keep='last')
    # keep only complete games with one home and one away row (same rule as fetch_game_logs)
    per_game = out.groupby('game_id')['home'].agg(['size', 'sum'])
    neutral = per_game.index[(per_game['size'] == 2) & (per_game['sum'] != 1)]
    out.loc[out['game_id'].isin(neutral), 'home'] = 0
    out = out[out['game_id'].isin(per_game.index[per_game['size'] == 2])]
    return out[TEAM_COLS], neutral


def import_season(conn, season, src_dir=None, log=print):
    """season = start year (2025 = 2025-26). ESPN labels seasons by end year."""
    y = season + 1
    if src_dir:
        p_src, t_src = Path(src_dir) / f'player_box_{y}.parquet', Path(src_dir) / f'team_box_{y}.parquet'
    else:
        p_src, t_src = PLAYER_URL.format(y=y), TEAM_URL.format(y=y)
    players = normalize_players(_read(p_src), season)
    teams, neutral = normalize_teams(_read(t_src), season, players)
    players.loc[players['game_id'].isin(neutral), 'home'] = 0
    n_p = upsert(conn, 'player_games', players, PLAYER_COLS)
    n_t = upsert(conn, 'team_games', teams, TEAM_COLS)
    log(f"✅ {season_str(season)}: {n_p} player-games, {n_t} team-games "
        f"({players['game_date'].min()} .. {players['game_date'].max()})")
    return n_p, n_t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seasons', type=int, nargs='+', required=True, help='season start years')
    ap.add_argument('--db', help='database path (default: NBA_DB_PATH or nba_data.db)')
    ap.add_argument('--dir', help='read player_box_YYYY / team_box_YYYY parquet files from here')
    a = ap.parse_args()
    conn = connect(a.db)
    init_db(conn)
    for s in a.seasons:
        import_season(conn, s, a.dir)


if __name__ == '__main__':
    main()
