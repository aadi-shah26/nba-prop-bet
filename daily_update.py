#!/usr/bin/env python3
"""
Daily maintenance (run once a day, e.g. via cron, after the previous night's games):
  1. refresh the current season's player and team game logs (4-6 API requests)
  2. settle logged props and game bets against actual results
  3. re-calibrate the game model's margin/total SDs

    python daily_update.py
"""

from datetime import datetime

from bet_signals import settle_props
from db import connect, init_db, season_for_date, season_str
from fetch_game_logs import ingest_season
from game_model import calibrate, settle_games


def main():
    print(f"\n{'=' * 50}\nNBA daily update — {datetime.now().isoformat(timespec='seconds')}\n{'=' * 50}")
    conn = connect()
    init_db(conn)
    if conn.execute("SELECT COUNT(*) FROM team_games").fetchone()[0] == 0:
        print("Database is empty — run python fetch_game_logs.py first (loads 3 seasons).")
        return

    season = season_for_date(None)
    print(f"🔄 Refreshing {season_str(season)}")
    ingest_season(conn, season)

    n_p, n_g = settle_props(conn), settle_games(conn)
    print(f"🧾 Settled {n_p} props, {n_g} game bets")

    cal = calibrate(conn)
    if cal:
        print(f"📐 Game model SDs: margin {cal[0]:.2f}, total {cal[1]:.2f} (from {cal[2]} games)")

    n = conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    days = (datetime.now().date() - datetime.fromisoformat(last).date()).days
    print(f"\n📊 {n} player-games; latest game {last} ({days} days ago)\n")


if __name__ == '__main__':
    main()
