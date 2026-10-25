"""
Current team rosters from stats.nba.com's CommonTeamRoster (one request per team). Stored in
the `rosters` table so the game model knows who is on each team now, including offseason
trades and signings that the game logs can't show until those players play for their new team.
Refreshed by the daily update.

    python nba_rosters.py               # refresh all 30 teams for the current season
    python nba_rosters.py --team BOS    # then print one team
"""

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from db import connect, init_db, season_for_date, season_str

COLS = ['season', 'team', 'player_id', 'player_name', 'position', 'how_acquired']


def nba_teams():
    """{abbreviation: team_id} for the 30 teams."""
    from nba_api.stats.static import teams
    return {t['abbreviation']: t['id'] for t in teams.get_teams()}


def fetch_roster(team_id, season, retries=3, timeout=30):
    from nba_api.stats.endpoints import commonteamroster
    last_err = None
    for attempt in range(retries):
        try:
            return commonteamroster.CommonTeamRoster(team_id=team_id, season=season_str(season),
                                                     timeout=timeout).get_data_frames()[0]
        except Exception as e:  # network errors, JSON decode errors from rate limiting, ...
            last_err = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Failed to fetch roster for team {team_id}: {last_err}")


def normalize_roster(raw, team, season):
    return pd.DataFrame({
        'season': season,
        'team': team,
        'player_id': raw['PLAYER_ID'].astype(int),
        'player_name': raw['PLAYER'],
        'position': raw['POSITION'].fillna(''),
        'how_acquired': raw['HOW_ACQUIRED'].fillna('') if 'HOW_ACQUIRED' in raw else '',
    })[COLS]


def refresh_rosters(conn, season, pause=0.6, log=print):
    """Fetch all 30 rosters, then replace the stored ones for `season` in one go (if any team
    fails, the old rosters are kept). Returns the number of players stored."""
    frames = []
    for team, team_id in sorted(nba_teams().items()):
        frames.append(normalize_roster(fetch_roster(team_id, season), team, season))
        time.sleep(pause)
    df = pd.concat(frames, ignore_index=True).drop_duplicates('player_id', keep='last')
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with conn:
        conn.execute("DELETE FROM rosters WHERE season = ?", (season,))
        conn.executemany(f"INSERT INTO rosters ({', '.join(COLS)}, fetched_at) "
                         f"VALUES ({', '.join('?' * (len(COLS) + 1))})",
                         [(*r, now) for r in df.itertuples(index=False)])
    log(f"👥 Rosters {season_str(season)}: {len(df)} players on {df['team'].nunique()} teams")
    return len(df)


def stored_rosters(conn, season):
    """(DataFrame of COLS, fetched_at ISO string) for `season`, or (empty, None)."""
    df = pd.read_sql_query(f"SELECT {', '.join(COLS)}, fetched_at FROM rosters WHERE season = ?",
                           conn, params=(season,))
    if df.empty:
        return df[COLS], None
    return df[COLS], df['fetched_at'].max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=season_for_date(None), help='season start year')
    ap.add_argument('--team', help='print this team after refreshing')
    a = ap.parse_args()
    conn = connect()
    init_db(conn)
    refresh_rosters(conn, a.season)
    if a.team:
        df, _ = stored_rosters(conn, a.season)
        print(df[df['team'] == a.team.upper()][['player_name', 'position', 'how_acquired']].to_string(index=False))


if __name__ == '__main__':
    main()
