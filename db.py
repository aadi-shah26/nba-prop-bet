"""
Shared database + lookup helpers.

Everything resolves paths relative to this file, so scripts work from any
working directory and on any machine.
"""

import os
import re
import sqlite3
import unicodedata
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv('NBA_DB_PATH', ROOT / 'nba_data.db'))
SCHEMA_PATH = ROOT / 'schema.sql'

SEASON_TYPES = ('Regular Season', 'PlayIn', 'Playoffs')


def connect(db_path=None):
    conn = sqlite3.connect(str(db_path or DB_PATH))
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db(conn):
    """Create tables. Drops the legacy `game_logs` table (it had text dates that
    sorted alphabetically and rows filed under the wrong player names)."""
    conn.execute('DROP TABLE IF EXISTS game_logs')
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

def season_for_date(d):
    """Season start year for a date. Aug-Dec -> that year, Jan-Jul -> previous year."""
    d = to_date(d)
    return d.year if d.month >= 8 else d.year - 1


def season_str(start_year):
    """2025 -> '2025-26' (the format stats.nba.com expects)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def to_date(d):
    if d is None:
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def normalize_name(name):
    """'Nikola Jokić' / 'nikola jokic' -> 'nikola jokic'; drops punctuation and Jr./III."""
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    s = re.sub(r"[^a-z0-9 ]", ' ', s.lower().replace("'", '').replace('.', ''))
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return ' '.join(parts)


def recent_players(conn, min_season=None):
    """DataFrame of players with their most recent team/game, newest first."""
    q = """
        SELECT pg.player_id, pg.player_name, pg.team, pg.game_date AS last_game, n.games
        FROM player_games pg
        JOIN (SELECT player_id, MAX(game_date) AS last_date, COUNT(*) AS games
              FROM player_games {where} GROUP BY player_id) n
          ON n.player_id = pg.player_id AND n.last_date = pg.game_date
        ORDER BY pg.game_date DESC
    """
    where = 'WHERE season >= ?' if min_season is not None else ''
    params = (min_season,) if min_season is not None else ()
    df = pd.read_sql_query(q.format(where=where), conn, params=params)
    df = df.drop_duplicates('player_id')
    df['norm'] = df['player_name'].map(normalize_name)
    return df


def find_players(conn, query, teams=None, min_season=None):
    """
    Resolve a typed/bookmaker player name to candidate players.
    Returns a DataFrame (possibly empty) ordered best-first:
    exact normalized match > all query tokens present > substring.
    `teams` (iterable of abbreviations) restricts to players whose latest team is in it.
    """
    df = recent_players(conn, min_season)
    if teams:
        teams = set(teams)
        df = df[df['team'].isin(teams)]
    q = normalize_name(query)
    if not q:
        return df.iloc[0:0]
    exact = df[df['norm'] == q]
    if len(exact):
        return exact
    tokens = q.split()
    all_tok = df[df['norm'].map(lambda n: all(any(w.startswith(t) for w in n.split()) for t in tokens))]
    if len(all_tok):
        return all_tok
    return df[df['norm'].str.contains(q, regex=False)]


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def _team_index():
    from nba_api.stats.static import teams as static_teams
    idx = {}
    for t in static_teams.get_teams():
        abbr = t['abbreviation']
        for key in (abbr, t['full_name'], t['nickname'], f"{t['city']} {t['nickname']}"):
            idx[normalize_name(key)] = abbr
    # Common alternate spellings used by sportsbooks / feeds.
    idx[normalize_name('LA Clippers')] = 'LAC'
    idx[normalize_name('LA Lakers')] = 'LAL'
    for alias, abbr in {'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS',
                        'PHO': 'PHX', 'BRK': 'BKN', 'UTAH': 'UTA', 'WSH': 'WAS',
                        'CHO': 'CHA'}.items():
        idx[normalize_name(alias)] = abbr
    return idx


_TEAM_IDX = None


def team_abbr(name):
    """'Los Angeles Lakers' / 'lakers' / 'lal' -> 'LAL'. Returns None if unknown."""
    global _TEAM_IDX
    if _TEAM_IDX is None:
        _TEAM_IDX = _team_index()
    return _TEAM_IDX.get(normalize_name(name))
