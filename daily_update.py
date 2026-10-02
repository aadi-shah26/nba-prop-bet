#!/usr/bin/env python3
"""
Daily maintenance (run once a day after the previous night's games, or press
"Update data" in the app):
  1. refresh the current season's player and team game logs (first run: last 3 seasons)
  2. settle logged props and game bets against actual results
  3. re-calibrate the game model's margin/total SDs

    python daily_update.py
"""

from datetime import datetime

from bet_signals import settle_props
from db import connect, init_db, season_for_date, season_str
from fetch_game_logs import ingest_season
from game_model import calibrate, settle_games


def update(conn, log=print):
    """Refresh data, settle bets, recalibrate. Builds the last 3 seasons if the DB is empty."""
    init_db(conn)
    season = season_for_date(None)
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    if last is None:
        seasons = [season - 2, season - 1, season]
        log("Empty database: downloading the last 3 seasons (about a minute)...")
    else:
        # every season from the one holding our latest game through today, so e.g. the rest of
        # last season's playoffs still loads when the first update happens in a new season
        seasons = list(range(min(season_for_date(last), season), season + 1))
    for s in seasons:
        log(f"🔄 Refreshing {season_str(s)}")
        ingest_season(conn, s, log=log)

    n_p, n_g = settle_props(conn), settle_games(conn)
    log(f"🧾 Settled {n_p} props, {n_g} game bets")

    cal = calibrate(conn)
    if cal:
        log(f"📐 Game model SDs: margin {cal[0]:.2f}, total {cal[1]:.2f} (from {cal[2]} games)")
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    log(f"📊 Latest game in database: {last}")
    return last


def main():
    print(f"\n{'=' * 50}\nNBA daily update — {datetime.now().isoformat(timespec='seconds')}\n{'=' * 50}")
    update(connect())


if __name__ == '__main__':
    main()
