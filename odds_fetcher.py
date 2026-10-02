#!/usr/bin/env python3
"""
The Odds API (v4) client.

Credits (check your plan; usage is printed after every call):
  * /events                       — list of games, no market data
  * /odds  (game lines)           — 1 credit per market per region, for ALL games at once.
                                    h2h + spreads + totals for the whole slate = 3 credits.
  * /events/{id}/odds (props)     — 1 credit per market per region, PER GAME.
                                    1 market x 10 games = 10 credits a day.
Player props are only served by the per-event endpoint; asking /odds for player markets
fails. One region ('us') covers all the main US books, so line-shopping across books costs
nothing extra.

Usage:
    python odds_fetcher.py events                 # list today's games
    python odds_fetcher.py games                  # today's spreads/totals/moneylines
    python odds_fetcher.py props --markets player_points --max-events 2
"""

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from db import team_abbr, to_date

load_dotenv()
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
ET = ZoneInfo("America/New_York")

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


class OddsAPIError(RuntimeError):
    pass


def _api_key():
    key = os.getenv('ODDS_API_KEY')
    if not key:
        raise OddsAPIError("ODDS_API_KEY not set. Put ODDS_API_KEY=... in a .env file.")
    return key


def _get(path, **params):
    params['apiKey'] = _api_key()
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    used, remaining, last = (r.headers.get(h) for h in
                             ('x-requests-used', 'x-requests-remaining', 'x-requests-last'))
    if remaining is not None:
        print(f"   (Odds API: this call {last} credits, used {used}, remaining {remaining})")
    if r.status_code != 200:
        raise OddsAPIError(f"{r.status_code} from {path}: {r.text[:300]}")
    return r.json()


def et_date(commence_time):
    """ISO UTC timestamp -> US/Eastern calendar date string (NBA schedule dates are ET)."""
    dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
    return dt.astimezone(ET).date().isoformat()


def fetch_events(date=None):
    """Games on `date` (ET, default today) with home/away abbreviations."""
    target = to_date(date).isoformat() if date else datetime.now(ET).date().isoformat()
    out = []
    for e in _get(f"/sports/{SPORT}/events"):
        if et_date(e['commence_time']) != target:
            continue
        out.append({'event_id': e['id'], 'game_date': target, 'commence_time': e['commence_time'],
                    'home': team_abbr(e['home_team']), 'away': team_abbr(e['away_team']),
                    'home_name': e['home_team'], 'away_name': e['away_team']})
    return out


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


def fetch_props(date=None, markets=('player_points',), regions='us', bookmakers=None, max_events=None):
    unknown = [m for m in markets if m not in PROP_MARKETS]
    if unknown:
        raise ValueError(f"Unknown prop markets {unknown}. Options: {', '.join(PROP_MARKETS)}")
    events = fetch_events(date)
    if max_events:
        events = events[:max_events]
    n_regions = 1 if bookmakers else len(regions.split(','))
    print(f"🔄 {len(events)} games; estimated cost {len(events) * len(markets) * n_regions} credits")
    rows = []
    for e in events:
        params = {'markets': ','.join(markets), 'oddsFormat': 'american'}
        if bookmakers:
            params['bookmakers'] = bookmakers
        else:
            params['regions'] = regions
        data = _get(f"/sports/{SPORT}/events/{e['event_id']}/odds", **params)
        rows.extend(parse_event_props(data, e['game_date']))
    return rows


# ---------------------------------------------------------------------------
# Game lines
# ---------------------------------------------------------------------------

def parse_game_lines(events, book, date=None):
    """Parse an /odds response (h2h, spreads, totals) for one bookmaker."""
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
    target = to_date(date).isoformat() if date else datetime.now(ET).date().isoformat()
    events = _get(f"/sports/{SPORT}/odds", bookmakers=book, markets=markets, oddsFormat='american')
    return parse_game_lines(events, book, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['events', 'games', 'props'])
    ap.add_argument('--date')
    ap.add_argument('--book', default='draftkings')
    ap.add_argument('--markets', nargs='+', default=['player_points'])
    ap.add_argument('--max-events', type=int, default=1)
    a = ap.parse_args()
    if a.what == 'events':
        for e in fetch_events(a.date):
            print(f"{e['game_date']}  {e['away']} @ {e['home']}  {e['commence_time']}  {e['event_id']}")
    elif a.what == 'games':
        for g in fetch_game_lines(a.book, a.date):
            print(g)
    else:
        for r in fetch_props(a.date, a.markets, max_events=a.max_events)[:40]:
            print(f"{r['book']:<12} {r['player']:<26} {r['stat']:<5} {r['line']:>5}  "
                  f"O {r['over_odds']:+d} / U {r['under_odds']:+d}")


if __name__ == '__main__':
    main()
