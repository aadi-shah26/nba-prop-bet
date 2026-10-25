#!/usr/bin/env python3
"""
Daily maintenance (run once a day after the previous night's games, or press
"Update data" in the app):
  1. refresh the current season's player and team game logs (first run: last 3 seasons)
  2. settle logged props and game bets against actual results
  3. re-calibrate the game model's margin adjustments and margin/total SDs
  4. refresh the season schedule (upcoming games, so the app can list them) and team rosters

The app runs a quick version of this in the background when it's opened and something is
due (update_due): it only downloads what can have changed.

    python daily_update.py
"""

from datetime import date, datetime, timedelta, timezone

from bet_signals import settle_props
from db import SEASON_TYPES, connect, init_db, season_for_date, season_str, set_meta
from fetch_game_logs import ingest_season
from game_model import calibrate, settle_games
from nba_rosters import refresh_rosters
from nba_schedule import STALE_HOURS, age_hours, refresh_schedule

GAME_HOURS = 3             # a game counts as finished this long after tip-off
ROSTER_STALE_HOURS = 20    # rosters are 30 requests: the quick update refreshes them about daily


def season_types_due(season, today=None):
    """Season types that can have games by `today`: no play-in/playoff requests before April."""
    today = today or date.today()
    return SEASON_TYPES if today >= date(season + 1, 4, 1) else ('Regular Season',)


def finished_games_missing(conn, now=None):
    """Scheduled games that should be over by now but aren't in team_games yet."""
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    if last is None:
        return 0
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(hours=GAME_HOURS)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return conn.execute("SELECT COUNT(*) FROM schedule WHERE game_date > ? AND tip_utc <= ? "
                        "AND status NOT LIKE '%PPD%' AND status NOT LIKE '%ostpone%'", (last, cutoff)).fetchone()[0]


def roster_age_hours(conn, season):
    t = conn.execute("SELECT MAX(fetched_at) FROM rosters WHERE season = ?", (season,)).fetchone()[0]
    return None if t is None else (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds() / 3600


def update_due(conn, now=None):
    """Why an automatic update is worth running, or None when nothing can have changed."""
    if conn.execute("SELECT COUNT(*) FROM team_games").fetchone()[0] == 0:
        return "no data yet"
    n = finished_games_missing(conn, now)
    if n:
        return f"{n} finished game{'s' * (n != 1)} to load"
    season = season_for_date(None)
    age = age_hours(conn, season)
    if age is None or age >= STALE_HOURS:
        return "schedule is out of date"
    age = roster_age_hours(conn, season)
    if age is None or age >= ROSTER_STALE_HOURS:
        return "rosters are out of date"
    return None


def update(conn, log=print, retries=4, quick=False):
    """Refresh data, settle bets, recalibrate. Builds the last 3 seasons if the DB is empty.
    retries: attempts per stats.nba.com request (the app's automatic update uses fewer).
    quick: only download what's due: box scores when a scheduled game has finished since the
    last load, the schedule when it's STALE_HOURS old, rosters when ROSTER_STALE_HOURS old."""
    init_db(conn)
    season = season_for_date(None)
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    if quick and last is not None and not finished_games_missing(conn):
        log("No finished games to load")
        seasons = []
    elif last is None:
        seasons = [season - 2, season - 1, season]
        log("Empty database: downloading the last 3 seasons (about a minute)...")
    else:
        # every season from the one holding our latest game through today, so e.g. the rest of
        # last season's playoffs still loads when the first update happens in a new season
        seasons = list(range(min(season_for_date(last), season), season + 1))
    for s in seasons:
        log(f"🔄 Refreshing {season_str(s)}")
        ingest_season(conn, s, season_types=season_types_due(s), log=log, retries=retries)

    n_p, n_g = settle_props(conn), settle_games(conn)
    log(f"🧾 Settled {n_p} props, {n_g} game bets")

    cal = calibrate(conn) if seasons else None
    if cal:
        c = cal[3]
        log(f"📐 Game model SDs: margin {cal[0]:.2f}, total {cal[1]:.2f} (from {cal[2]} games); "
            f"margin scale {c['scale']:.2f}, back-to-back {c['b2b']:.1f} pts, lineup {c['lineup']:.3f}")
    age = age_hours(conn, season)
    if not quick or age is None or age >= STALE_HOURS or seasons:
        try:
            refresh_schedule(conn, season, log=log)
        except Exception as e:  # not worth failing the update over; the app retries on its own
            log(f"⚠️ Couldn't refresh the schedule: {type(e).__name__}: {e}")
    age = roster_age_hours(conn, season)
    if not quick or age is None or age >= ROSTER_STALE_HOURS:
        try:
            refresh_rosters(conn, season, log=log)
        except Exception as e:  # the stored rosters (or the game logs) are used instead
            log(f"⚠️ Couldn't refresh rosters: {type(e).__name__}: {e}")
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    set_meta(conn, 'last_update', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    log(f"📊 Latest game in database: {last}")
    return last


def main():
    print(f"\n{'=' * 50}\nNBA daily update — {datetime.now().isoformat(timespec='seconds')}\n{'=' * 50}")
    update(connect())


if __name__ == '__main__':
    main()
