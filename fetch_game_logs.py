#!/usr/bin/env python3
"""
NBA game log ingestion.

Uses stats.nba.com's LeagueGameLog endpoint, which returns EVERY player-game
(or team-game) for a season in a single request. That means:
  * no hand-maintained player list (the old list had wrong/duplicate IDs),
  * every player in the league is covered,
  * team box scores come along for the opponent adjustment and game model,
  * a full season is ~2 requests instead of ~100.

Usage:
    python fetch_game_logs.py                 # last 3 seasons incl. current
    python fetch_game_logs.py --seasons 2023 2024 2025
    python fetch_game_logs.py --current       # just refresh the current season
"""

import argparse
import time

import pandas as pd

from db import SEASON_TYPES, connect, init_db, season_for_date, season_str

PLAYER_COLS = ['player_id', 'player_name', 'team_id', 'team', 'game_id', 'game_date', 'season',
               'season_type', 'opponent', 'home', 'wl', 'minutes', 'pts', 'reb', 'ast', 'stl',
               'blk', 'tov', 'fg3m', 'fgm', 'fga', 'ftm', 'fta', 'oreb', 'dreb', 'pf', 'plus_minus']
TEAM_COLS = ['team_id', 'team', 'team_name', 'game_id', 'game_date', 'season', 'season_type',
             'opponent', 'home', 'wl', 'minutes', 'pts', 'reb', 'ast', 'stl', 'blk', 'tov',
             'fg3m', 'fg3a', 'fgm', 'fga', 'ftm', 'fta', 'oreb', 'dreb', 'pf', 'plus_minus']

# stats.nba.com column -> our column (shared by player and team frames)
_STAT_MAP = {'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'STL': 'stl', 'BLK': 'blk', 'TOV': 'tov',
             'FG3M': 'fg3m', 'FG3A': 'fg3a', 'FGM': 'fgm', 'FGA': 'fga', 'FTM': 'ftm', 'FTA': 'fta',
             'OREB': 'oreb', 'DREB': 'dreb', 'PF': 'pf'}


def fetch_league_log(season, season_type, kind, retries=4, timeout=60, log=print):
    """
    kind: 'P' (player rows) or 'T' (team rows). Returns a DataFrame (empty if the
    season/season type has no games yet). Raises after `retries` failed attempts so a
    network problem is never silently mistaken for "no games".
    """
    from nba_api.stats.endpoints import leaguegamelog

    last_err = None
    for attempt in range(retries):
        try:
            return leaguegamelog.LeagueGameLog(
                season=season_str(season),
                season_type_all_star=season_type,
                player_or_team_abbreviation=kind,
                timeout=timeout,
            ).get_data_frames()[0]
        except Exception as e:  # network errors, JSON decode errors from rate limiting, ...
            last_err = e
            wait = 2 ** (attempt + 1)
            log(f"⚠️ {season_str(season)} {season_type} {kind}: {type(e).__name__}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {season_str(season)} {season_type} {kind}: {last_err}")


def parse_matchup(matchup):
    """'LAL vs. HOU' -> ('LAL', 'HOU', 1); 'LAL @ DAL' -> ('LAL', 'DAL', 0)."""
    parts = str(matchup).split()
    if len(parts) != 3 or parts[1] not in ('vs.', 'vs', '@'):
        raise ValueError(f"Unrecognized MATCHUP: {matchup!r}")
    return parts[0], parts[2], 0 if parts[1] == '@' else 1


def _minutes(v):
    """Numeric minutes, or 'MM:SS' strings -> float minutes."""
    if isinstance(v, str):
        if ':' in v:
            m, s = v.split(':')[:2]
            return int(m) + int(s) / 60
        return float(v) if v.strip() else 0.0
    return 0.0 if pd.isna(v) else float(v)


def _common(df, season, season_type):
    out = pd.DataFrame(index=df.index)
    m = df['MATCHUP'].map(parse_matchup)
    out['team'] = m.str[0]
    out['opponent'] = m.str[1]
    out['home'] = m.str[2].astype(int)
    if (out['team'] != df['TEAM_ABBREVIATION']).any():
        raise ValueError("MATCHUP team does not match TEAM_ABBREVIATION")
    out['team_id'] = df['TEAM_ID'].astype(int)
    out['game_id'] = df['GAME_ID'].astype(str).str.zfill(10)
    out['game_date'] = pd.to_datetime(df['GAME_DATE'], format='mixed').dt.strftime('%Y-%m-%d')
    out['season'] = int(season)
    out['season_type'] = season_type
    out['wl'] = df['WL'].where(df['WL'].notna(), None)
    out['minutes'] = df['MIN'].map(_minutes)
    for src, dst in _STAT_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors='coerce').fillna(0).astype(int)
    out['plus_minus'] = pd.to_numeric(df['PLUS_MINUS'], errors='coerce')
    return out


def normalize_player_frame(df, season, season_type):
    if df.empty:
        return pd.DataFrame(columns=PLAYER_COLS)
    out = _common(df, season, season_type)
    out['player_id'] = df['PLAYER_ID'].astype(int)
    out['player_name'] = df['PLAYER_NAME'].astype(str)
    # Rows with 0 minutes carry no information about a player's production.
    out = out[out['minutes'] > 0].drop_duplicates(['player_id', 'game_id'], keep='last')
    return out[PLAYER_COLS]


def normalize_team_frame(df, season, season_type):
    if df.empty:
        return pd.DataFrame(columns=TEAM_COLS)
    out = _common(df, season, season_type)
    out['team_name'] = df['TEAM_NAME'].astype(str)
    return out.drop_duplicates(['team_id', 'game_id'], keep='last')[TEAM_COLS]


def upsert(conn, table, frame, cols):
    if frame.empty:
        return 0
    rows = [tuple(None if pd.isna(v) else v for v in r)
            for r in frame[cols].itertuples(index=False, name=None)]
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        rows)
    conn.commit()
    return len(rows)


def ingest_frames(conn, season, season_type, player_raw, team_raw):
    """Normalize + store raw LeagueGameLog frames. Split out so it can be tested offline."""
    p = normalize_player_frame(player_raw, season, season_type)
    t = normalize_team_frame(team_raw, season, season_type)
    if not t.empty:
        # Each game must have exactly two team rows (home + away).
        per_game = t.groupby('game_id')['home'].agg(['size', 'sum'])
        bad = per_game[(per_game['size'] != 2) | (per_game['sum'] != 1)]
        if len(bad):
            raise ValueError(f"{len(bad)} games without exactly one home and one away row")
    return upsert(conn, 'player_games', p, PLAYER_COLS), upsert(conn, 'team_games', t, TEAM_COLS)


def ingest_season(conn, season, season_types=SEASON_TYPES, pause=1.0, log=print):
    total_p = total_t = 0
    for st in season_types:
        try:
            p_raw = fetch_league_log(season, st, 'P', log=log)
            time.sleep(pause)
            t_raw = fetch_league_log(season, st, 'T', log=log)
            time.sleep(pause)
        except RuntimeError as e:
            if st == 'PlayIn':  # a handful of games; not worth failing the whole run over
                log(f"⚠️ Skipping play-in games: {e}")
                continue
            raise
        n_p, n_t = ingest_frames(conn, season, st, p_raw, t_raw)
        log(f"✅ {season_str(season)} {st:<14} {n_p:>6} player-games, {n_t:>5} team-games")
        total_p += n_p
        total_t += n_t
    return total_p, total_t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seasons', type=int, nargs='+', help='season start years, e.g. 2024 2025')
    ap.add_argument('--current', action='store_true', help='only refresh the current season')
    args = ap.parse_args()

    current = season_for_date(None)
    if args.current:
        seasons = [current]
    else:
        seasons = args.seasons or [current - 2, current - 1, current]

    conn = connect()
    init_db(conn)
    for s in seasons:
        print(f"🔄 {season_str(s)}")
        ingest_season(conn, s)
    n_p = conn.execute('SELECT COUNT(*) FROM player_games').fetchone()[0]
    n_t = conn.execute('SELECT COUNT(*) FROM team_games').fetchone()[0]
    last = conn.execute('SELECT MAX(game_date) FROM team_games').fetchone()[0]
    conn.close()
    print(f"\n📊 DB: {n_p} player-games, {n_t} team-games, latest game {last}")


if __name__ == '__main__':
    main()
