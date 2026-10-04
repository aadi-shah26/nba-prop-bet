#!/usr/bin/env python3
"""
Optional live lines from The Odds API (v4) — https://the-odds-api.com

Manual entry in the app always works; this only saves typing. It is switched on by an API
key in the environment or in a `.env` file next to this script (git-ignored):

    ODDS_API_KEY=your_key_here

Credit cost (usage is reported after every call):
  * /events                     — free; list of games
  * /odds  (game lines)         — 1 credit per market per region for the WHOLE slate
                                  (spreads + totals + moneyline = 3 credits)
  * /events/{id}/odds (props)   — 1 credit per market per region, PER GAME
Player props are only served by the per-event endpoint.

Usage:
    python odds_fetcher.py events                          # today's games
    python odds_fetcher.py games --book draftkings         # today's spreads/totals/moneylines
    python odds_fetcher.py props --markets player_points --max-events 1
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from db import find_players, team_abbr, to_date

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
ET = ZoneInfo("America/New_York")
ENV_FILE = Path(__file__).resolve().parent / '.env'

PROP_MARKETS = {
    'player_points': 'PTS',
    'player_rebounds': 'REB',
    'player_assists': 'AST',
    'player_threes': 'FG3M',
    'player_blocks': 'BLK',
    'player_steals': 'STL',
    'player_turnovers': 'TOV',
    'player_points_rebounds_assists': 'PRA',
    'player_points_rebounds': 'PR',
    'player_points_assists': 'PA',
    'player_rebounds_assists': 'RA',
    'player_blocks_steals': 'SB',
}

BOOKS = {
    'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM', 'williamhill_us': 'Caesars',
    'espnbet': 'ESPN BET', 'betrivers': 'BetRivers', 'fanatics': 'Fanatics', 'bovada': 'Bovada',
}


class OddsAPIError(RuntimeError):
    pass


def api_key():
    """ODDS_API_KEY from the environment, else from .env. None when not configured."""
    key = os.getenv('ODDS_API_KEY', '').strip()
    if not key and ENV_FILE.exists():
        for raw in ENV_FILE.read_text().splitlines():
            name, sep, value = raw.strip().partition('=')
            if sep and name.strip() == 'ODDS_API_KEY':
                key = value.strip().strip('"\'')
    return key or None


def enabled():
    return api_key() is not None


def _get(path, **params):
    """GET an endpoint. Returns (json, usage) where usage = {'last', 'used', 'remaining'} credits."""
    key = api_key()
    if not key:
        raise OddsAPIError("ODDS_API_KEY not set. Put ODDS_API_KEY=... in a .env file.")
    try:
        r = requests.get(f"{BASE_URL}{path}", params={**params, 'apiKey': key}, timeout=20)
    except requests.RequestException as e:
        raise OddsAPIError(f"Network error calling The Odds API: {type(e).__name__}") from e
    usage = {k: r.headers.get(f'x-requests-{k}') for k in ('last', 'used', 'remaining')}
    if r.status_code != 200:
        raise OddsAPIError(f"{r.status_code} from {path}: {r.text[:300]}")
    return r.json(), usage


def et_date(commence_time):
    """ISO UTC timestamp -> US/Eastern calendar date string (NBA schedule dates are ET)."""
    dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
    return dt.astimezone(ET).date().isoformat()


def _target(date):
    return to_date(date).isoformat() if date else datetime.now(ET).date().isoformat()


def fetch_events(date=None):
    """Games on `date` (ET, default today) with home/away abbreviations. Returns (events, usage)."""
    target = _target(date)
    data, usage = _get(f"/sports/{SPORT}/events")
    out = [{'event_id': e['id'], 'game_date': target, 'commence_time': e['commence_time'],
            'home': team_abbr(e['home_team']), 'away': team_abbr(e['away_team']),
            'home_name': e['home_team'], 'away_name': e['away_team']}
           for e in data if et_date(e['commence_time']) == target]
    return out, usage


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def parse_event_props(data, game_date=None):
    """
    Parse one /events/{id}/odds response into paired over/under rows:
    {book, player, stat, market, line, over_odds, under_odds, home, away, game_date, event_id}
    Rows where the book only offers one side are dropped (no way to remove the vig).
    """
    home, away = team_abbr(data.get('home_team', '')), team_abbr(data.get('away_team', ''))
    gd = game_date or (et_date(data['commence_time']) if data.get('commence_time') else None)
    pairs = {}
    for bk in data.get('bookmakers', []):
        for m in bk.get('markets', []):
            stat = PROP_MARKETS.get(m.get('key'))
            if stat is None:
                continue
            for o in m.get('outcomes', []):
                side = str(o.get('name', '')).lower()
                player = o.get('description')
                line, price = o.get('point'), o.get('price')
                if side not in ('over', 'under') or not player or line is None or price is None:
                    continue
                key = (bk['key'], player, m['key'], float(line))
                pairs.setdefault(key, {})[side] = int(round(float(price)))
    rows = []
    for (book, player, market, line), sides in pairs.items():
        if 'over' in sides and 'under' in sides:
            rows.append({'book': book, 'player': player, 'stat': PROP_MARKETS[market], 'market': market,
                         'line': line, 'over_odds': sides['over'], 'under_odds': sides['under'],
                         'home': home, 'away': away, 'game_date': gd, 'event_id': data.get('id')})
    return rows


def fetch_event_props(event_id, game_date, markets=('player_points',), book='draftkings'):
    """Props for one game at one book. Costs len(markets) credits. Returns (rows, usage)."""
    unknown = [m for m in markets if m not in PROP_MARKETS]
    if unknown:
        raise ValueError(f"Unknown prop markets {unknown}. Options: {', '.join(PROP_MARKETS)}")
    data, usage = _get(f"/sports/{SPORT}/events/{event_id}/odds", markets=','.join(markets),
                       bookmakers=book, oddsFormat='american')
    return parse_event_props(data, game_date), usage


def match_players(conn, rows):
    """Attach player_id to each prop row: the bookmaker's name resolved against the two teams
    in the game first, then the whole league (recent trades). Unresolved rows get None."""
    cache = {}
    for r in rows:
        k = (r['player'], r['home'], r['away'])
        if k not in cache:
            pid = None
            for teams in ([t for t in (r['home'], r['away']) if t], None):
                c = find_players(conn, r['player'], teams=teams or None)
                if len(c) and c['player_id'].nunique() == 1:
                    pid = int(c['player_id'].iloc[0])
                    break
            cache[k] = pid
        r['player_id'] = cache[k]
    return rows


# ---------------------------------------------------------------------------
# Game lines
# ---------------------------------------------------------------------------

def parse_game_lines(events, book, date=None):
    """Parse an /odds response (h2h, spreads, totals) for one bookmaker. Spread is the HOME line."""
    out = []
    for e in events:
        gd = et_date(e['commence_time'])
        if date and gd != date:
            continue
        bk = next((b for b in e.get('bookmakers', []) if b['key'] == book), None)
        if bk is None:
            continue
        hn, an = e['home_team'], e['away_team']
        g = {'game_date': gd, 'home': team_abbr(hn), 'away': team_abbr(an), 'book': book}
        for m in bk.get('markets', []):
            oc = {o['name']: o for o in m.get('outcomes', [])}
            if m['key'] == 'h2h' and hn in oc and an in oc:
                g['ml'] = (int(oc[hn]['price']), int(oc[an]['price']))
            elif m['key'] == 'spreads' and hn in oc and an in oc:
                g['spread'] = float(oc[hn]['point'])
                g['spread_odds'] = (int(oc[hn]['price']), int(oc[an]['price']))
            elif m['key'] == 'totals' and 'Over' in oc and 'Under' in oc:
                g['total'] = float(oc['Over']['point'])
                g['total_odds'] = (int(oc['Over']['price']), int(oc['Under']['price']))
        out.append(g)
    return out


def fetch_game_lines(book='draftkings', date=None, markets='h2h,spreads,totals'):
    """Spreads/totals/moneylines for every game on `date` at one book. Returns (games, usage)."""
    data, usage = _get(f"/sports/{SPORT}/odds", bookmakers=book, markets=markets, oddsFormat='american')
    return parse_game_lines(data, book, _target(date)), usage


def usage_text(usage):
    if not usage or usage.get('remaining') is None:
        return ''
    return f"Odds API credits: this call {usage['last']}, used {usage['used']}, remaining {usage['remaining']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['events', 'games', 'props'])
    ap.add_argument('--date')
    ap.add_argument('--book', default='draftkings')
    ap.add_argument('--markets', nargs='+', default=['player_points'])
    ap.add_argument('--max-events', type=int, default=1, help='props: games to fetch (credits per game)')
    a = ap.parse_args()
    if a.what == 'events':
        events, usage = fetch_events(a.date)
        for e in events:
            print(f"{e['game_date']}  {e['away']} @ {e['home']}  {e['commence_time']}  {e['event_id']}")
    elif a.what == 'games':
        games, usage = fetch_game_lines(a.book, a.date)
        for g in games:
            print(g)
    else:
        events, usage = fetch_events(a.date)
        for e in events[:a.max_events]:
            rows, usage = fetch_event_props(e['event_id'], e['game_date'], a.markets, a.book)
            for r in rows:
                print(f"{r['away']}@{r['home']}  {r['player']:<26} {r['stat']:<5} {r['line']:>5}  "
                      f"O {r['over_odds']:+d} / U {r['under_odds']:+d}")
    print(usage_text(usage))


if __name__ == '__main__':
    main()
