#!/usr/bin/env python3
"""
Kalshi's past NBA game contracts, for testing the model against the prices it would really
have bet into (market_backtest.py). Public data, no login, nothing touches an account.

For every game of a season it reads the settled winner, spread and total contracts from
/historical/markets, and from each contract's minute candlesticks the yes bid/ask 60 minutes
before tip-off (when the bot bets) and at tip-off (the closing price, for CLV). Minute candles
are only sent when something changed, so each price is the latest candle at or before its time.
Kalshi's expected_expiration_time is tip-off + 3 hours. Contracts listed after tip-off (in-game ladders)
have no pre-game price and are skipped.

Kalshi allows a few requests a second, so a season (~30,000 contracts) takes about two hours.
It saves as it goes and a rerun skips finished games.

    python kalshi_history.py                  # last season (2025-26)
    python kalshi_history.py --season 2025 --limit 20
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from db import connect, init_db
from kalshi_client import BASE_URL, SERIES, match_game, parse_event_ticker, parse_market

TIP_BEFORE_EXPIRY = timedelta(hours=3)
BET_MINUTES = 60
LOOKBACK = timedelta(hours=6)  # how far back to look for the last candle before BET_MINUTES
RATE = 5                     # requests per second, under Kalshi's basic limit


class HistoryClient:
    """Throttled, retrying GETs against Kalshi's public API (shared by worker threads)."""

    def __init__(self, base_url=BASE_URL, session=None, rate=RATE, timeout=30):
        if session is None:
            import requests
            session = requests.Session()
        self.base_url, self.session, self.timeout = base_url.rstrip('/'), session, timeout
        self.gap, self.next_at, self.lock = 1 / rate, 0.0, threading.Lock()

    def get(self, path, **params):
        for attempt in range(8):
            with self.lock:
                wait = self.next_at - time.monotonic()
                self.next_at = max(self.next_at, time.monotonic()) + self.gap
            if wait > 0:
                time.sleep(wait)
            r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
            if r.status_code == 429 or r.status_code >= 500:
                with self.lock:   # back off every thread, not just this one
                    self.next_at = max(self.next_at, time.monotonic() + min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def markets(self, **params):
        """Every settled market matching params, following the pagination cursor."""
        out, cursor = [], None
        while True:
            d = self.get('/historical/markets', limit=1000, **params, **({'cursor': cursor} if cursor else {}))
            out += d.get('markets') or []
            cursor = d.get('cursor')
            if not cursor or not d.get('markets'):
                return out

    def candles(self, ticker, start, end):
        return self.get(f'/historical/markets/{ticker}/candlesticks', start_ts=int(start.timestamp()),
                        end_ts=int(end.timestamp()), period_interval=1).get('candlesticks') or []


def _ts(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00'))


def _quote(candles, at):
    """(yes_bid, yes_ask) at the close of the latest candle ending at or before `at`, or
    (None, None) without one or without a two-sided book."""
    before = [c for c in candles if c['end_period_ts'] <= at.timestamp()]
    if not before:
        return None, None
    candle = max(before, key=lambda c: c['end_period_ts'])
    bid, ask = (candle.get(k, {}).get('close') for k in ('yes_bid', 'yes_ask'))
    bid, ask = (None if v is None else float(v) for v in (bid, ask))
    return (bid, ask) if bid is not None and ask is not None and 0 < bid < ask < 1 else (None, None)


def season_games(conn, season):
    """The season's games as schedule-like dicts (game_date, home, away) from the box scores."""
    g = pd.read_sql_query("SELECT DISTINCT game_date, team AS home, opponent AS away FROM team_games "
                          "WHERE season = ? AND home = 1", conn, params=(season,))
    return g.to_dict('records')


def season_events(client, games):
    """{event code: game} for the season's games Kalshi listed, from the winner series."""
    out = {}
    for m in client.markets(series_ticker=SERIES['WINNER']):
        parsed = parse_event_ticker(m.get('event_ticker'))
        game = parsed and match_game(*parsed, games)
        if game:
            out[m['event_ticker'].split('-', 1)[1]] = game
    return out


def game_rows(client, code, game):
    """kalshi_history rows for one game: every pre-game winner/spread/total contract."""
    rows = []
    for kind, series in SERIES.items():
        for m in client.markets(event_ticker=f'{series}-{code}'):
            c = parse_market(kind, m, game, settled=True)
            if c is None or m.get('result') not in ('yes', 'no') or not m.get('expected_expiration_time'):
                continue
            tip = _ts(m['expected_expiration_time']) - TIP_BEFORE_EXPIRY
            if m.get('created_time') and _ts(m['created_time']) >= tip:
                continue                                     # listed in-game
            candles = client.candles(m['ticker'], tip - LOOKBACK, tip)
            bet, close = _quote(candles, tip - timedelta(minutes=BET_MINUTES)), _quote(candles, tip)
            if bet[0] is None and close[0] is None:
                continue
            rows.append((c['ticker'], game['game_date'], game['home'], game['away'],
                         tip.isoformat(timespec='seconds'), kind, c['team'], c['line'], *bet, *close,
                         c['volume'], int(m['result'] == 'yes')))
    return rows


def fetch(conn, season, client=None, workers=4, limit=None, log=print):
    """Fetch and store the season's contracts; returns how many games were added."""
    client = client or HistoryClient()
    done = {r[0] for r in conn.execute("SELECT event FROM kalshi_history_games")}
    events = season_events(client, season_games(conn, season))
    todo = sorted((c, g) for c, g in events.items() if c not in done)[:limit]
    log(f"{len(events)} games on Kalshi, {len(todo)} to fetch")
    def one(x):
        try:
            return game_rows(client, *x)
        except Exception as e:  # leave the game for the next run
            log(f"{x[0]}: {type(e).__name__} {e}")
            return None

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for (code, _), rows in zip(todo, ex.map(one, todo)):
            if rows is None:
                continue
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            conn.executemany("INSERT OR REPLACE INTO kalshi_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             [r + (now,) for r in rows])
            conn.execute("INSERT OR REPLACE INTO kalshi_history_games VALUES (?, ?)", (code, now))
            conn.commit()
            n += 1
            if n % 50 == 0:
                log(f"{n}/{len(todo)} games")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2025, help="season start year (2025 = 2025-26)")
    ap.add_argument('--limit', type=int, help="fetch at most this many games")
    args = ap.parse_args()
    conn = connect()
    init_db(conn)
    n = fetch(conn, args.season, limit=args.limit)
    total = conn.execute("SELECT COUNT(*), COUNT(DISTINCT game_date || home) FROM kalshi_history").fetchone()
    print(f"Added {n} games. kalshi_history: {total[0]} contracts on {total[1]} games.")


if __name__ == '__main__':
    main()
