"""
NBA schedule (past and future games) from stats.nba.com's ScheduleLeagueV2, which returns a
whole season in one request. Stored in the `schedule` table so the app can list each day's
games instead of making you pick the teams. Refreshed by the daily update, and by the app
when the stored copy is more than STALE_HOURS old.

    python nba_schedule.py                  # refresh the current season and print today's games
    python nba_schedule.py --date 2026-10-21
"""

import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from db import connect, init_db, season_for_date, season_str, to_date

# Game id prefix -> kept: regular season, playoffs, play-in, NBA Cup final (no preseason / All-Star)
GAME_TYPES = ('002', '004', '005', '006')
STALE_HOURS = 6
RETRY_SECONDS = 1800     # after a failed fetch, don't make the page wait on another one for this long
_failed_at = {}
COLS = ['game_id', 'game_date', 'season', 'tip_utc', 'status', 'home', 'away', 'label']


def fetch_schedule(season, timeout=15):
    """Raw ScheduleLeagueV2 frame for a season (start year)."""
    from nba_api.stats.endpoints import scheduleleaguev2
    return scheduleleaguev2.ScheduleLeagueV2(season=season_str(season), timeout=timeout).get_data_frames()[0]


def normalize_schedule(raw, season):
    """Raw frame -> COLS. Drops preseason / All-Star games and playoff games whose teams aren't set yet."""
    df = raw[raw['gameId'].astype(str).str[:3].isin(GAME_TYPES)]
    df = df[df['homeTeam_teamTricode'].fillna('').ne('') & df['awayTeam_teamTricode'].fillna('').ne('')]
    label = df['gameLabel'].fillna('').str.strip()
    sub = df['gameSubLabel'].fillna('').str.strip()
    return pd.DataFrame({
        'game_id': df['gameId'].astype(str),
        # gameDate is the US Eastern date, e.g. '10/20/2026 00:00:00'
        'game_date': pd.to_datetime(df['gameDate'].str[:10], format='%m/%d/%Y').dt.strftime('%Y-%m-%d'),
        'season': season,
        'tip_utc': df['gameDateTimeUTC'],
        'status': df['gameStatusText'].fillna('').str.strip(),
        'home': df['homeTeam_teamTricode'],
        'away': df['awayTeam_teamTricode'],
        'label': [' · '.join(x for x in (a, b) if x) for a, b in zip(label, sub)],
    })[COLS].reset_index(drop=True)


def refresh_schedule(conn, season, log=print):
    """Replace the stored schedule for `season` (so moved or postponed games don't linger)."""
    df = normalize_schedule(fetch_schedule(season), season)
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    conn.execute("DELETE FROM schedule WHERE season = ?", (season,))
    conn.executemany(f"INSERT OR REPLACE INTO schedule ({', '.join(COLS)}, fetched_at) "
                     f"VALUES ({', '.join('?' * (len(COLS) + 1))})",
                     [(*r, now) for r in df.itertuples(index=False)])
    conn.commit()
    log(f"📅 Schedule {season_str(season)}: {len(df)} games")
    return len(df)


def age_hours(conn, season):
    """Hours since `season`'s schedule was fetched, or None if it never was."""
    t = conn.execute("SELECT MAX(fetched_at) FROM schedule WHERE season = ?", (season,)).fetchone()[0]
    if t is None:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds() / 3600


def ensure_schedule(conn, game_date, refresh_stale=True):
    """Fetch the schedule for `game_date`'s season if we don't have it, or (refresh_stale) if it's
    the current season and older than STALE_HOURS (tip times and postponements change). Returns an
    error message if the fetch failed (the stored copy, if any, is still used), else None. After a
    failure it doesn't try again for RETRY_SECONDS, so a blocked stats.nba.com can't stall every
    page load."""
    season = season_for_date(game_date)
    age = age_hours(conn, season)
    if age is not None and (not refresh_stale or season != season_for_date(None) or age < STALE_HOURS):
        return None
    if time.time() - _failed_at.get(season, (0, ''))[0] < RETRY_SECONDS:
        return _failed_at[season][1]
    try:
        refresh_schedule(conn, season, log=lambda *_: None)
    except Exception as e:  # stats.nba.com blocks / times out now and then
        _failed_at[season] = (time.time(), f"{type(e).__name__}: {e}")
        return _failed_at[season][1]
    _failed_at.pop(season, None)
    return None


def games_on(conn, game_date):
    """DataFrame of the games on `game_date`, in tip-off order."""
    return pd.read_sql_query(f"SELECT {', '.join(COLS)} FROM schedule WHERE game_date = ? ORDER BY tip_utc, game_id",
                             conn, params=(to_date(game_date).isoformat(),))


def next_game_date(conn, after):
    """First date after `after` with a game, or None."""
    return conn.execute("SELECT MIN(game_date) FROM schedule WHERE game_date > ?",
                        (to_date(after).isoformat(),)).fetchone()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', help='YYYY-MM-DD (default today)')
    a = ap.parse_args()
    d = to_date(a.date)
    conn = connect()
    init_db(conn)
    refresh_schedule(conn, season_for_date(d))
    games = games_on(conn, d)
    if games.empty:
        print(f"No games on {d}. Next game day: {next_game_date(conn, d)}")
    for g in games.itertuples():
        print(f"{g.away} @ {g.home}  {g.status}  {g.label}")


if __name__ == '__main__':
    main()
