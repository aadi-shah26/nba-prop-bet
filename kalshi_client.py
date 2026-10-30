"""
Kalshi's public market data for NBA games: winner, spread and total contracts with their
prices, matched to the games in the `schedule` table. Read-only and unauthenticated: there is
no order or account code here, so nothing in this module can touch a Kalshi account.

Kalshi contracts pay $1 if they win and are priced in dollars (0.57 = 57 cents), so a price is
roughly the market's probability. Each market is one yes/no contract:
    WINNER  KXNBAGAME-26OCT26WASCHA-CHA    "Charlotte wins"
    SPREAD  KXNBASPREAD-26OCT26WASCHA-CHA7 "Charlotte wins by over 6.5"   (team + strike)
    TOTAL   KXNBATOTAL-26OCT26WASCHA-219   "Over 218.5 points"            (strike)
The event part (26OCT26WASCHA) is the US Eastern date and the two teams' codes.

    python kalshi_client.py                 # today's NBA contracts, matched to the schedule
    python kalshi_client.py --date 2026-10-27
"""

import argparse
import re
from datetime import date

from db import team_abbr

BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
DEMO_URL = 'https://external-api.demo.kalshi.co/trade-api/v2'
SERIES = {'WINNER': 'KXNBAGAME', 'SPREAD': 'KXNBASPREAD', 'TOTAL': 'KXNBATOTAL'}
_MONTHS = {m: i for i, m in enumerate(
    ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'), 1)}


class KalshiClient:
    def __init__(self, base_url=BASE_URL, session=None, timeout=10):
        if session is None:
            import requests
            session = requests.Session()
        self.base_url, self.session, self.timeout = base_url.rstrip('/'), session, timeout

    def get(self, path, **params):
        r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def events(self, series_ticker, status='open'):
        """Every event in a series with its markets nested, following the pagination cursor."""
        out, cursor = [], None
        while True:
            params = dict(series_ticker=series_ticker, status=status, with_nested_markets='true', limit=200)
            if cursor:
                params['cursor'] = cursor
            d = self.get('/events', **params)
            out += d.get('events') or []
            cursor = d.get('cursor')
            if not cursor or not d.get('events'):
                return out

    def orderbook(self, ticker, depth=20):
        return parse_orderbook(self.get(f'/markets/{ticker}/orderbook', depth=depth))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def price(m, field):
    """A market price in dollars. Kalshi sends `<field>_dollars` ('0.5700'); older responses
    only have `<field>` in cents (57). None when missing."""
    v = _num(m.get(f'{field}_dollars'))
    if v is None:
        v = _num(m.get(field))
        v = None if v is None else v / 100
    return None if v is None else round(v, 4)


def parse_event_ticker(event_ticker):
    """'KXNBAGAME-26OCT26WASCHA' -> ('2026-10-26', 'WASCHA'), or None if it doesn't look like one."""
    m = re.fullmatch(r'[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)', event_ticker or '')
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        d = date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None
    return d.isoformat(), m.group(4)


def split_teams(codes):
    """'WASCHA' -> [('WAS', 'CHA')]: every way to read the string as two team codes (2-4 letters
    each, any spelling db.team_abbr knows, e.g. GS or NY). Usually exactly one."""
    out = []
    for i in range(2, len(codes) - 1):
        a, b = team_abbr(codes[:i]), team_abbr(codes[i:])
        if a and b and a != b and (a, b) not in out:
            out.append((a, b))
    return out


def match_game(game_date, codes, games):
    """The schedule game (dict with home/away/tip_utc/status) an event's team codes refer to.
    `games` is the schedule for that date as a list of dicts; None when no single game matches."""
    hits = [g for g in games if g['game_date'] == game_date
            and any({a, b} == {g['home'], g['away']} for a, b in split_teams(codes))]
    return hits[0] if len(hits) == 1 else None


def _strike(m, suffix_number):
    """The line a SPREAD/TOTAL contract is over ("wins by over 6.5" -> 6.5). Kalshi's
    floor_strike when it's sent, else the ticker's number - 0.5 (CHA7 = over 6.5). Tickers from
    2025-26 used the number below instead (DET9 = over 9.5), which floor_strike confirms. None when
    the two disagree or the contract isn't an "over" (fail closed rather than guess)."""
    fs = _num(m.get('floor_strike'))
    st = m.get('strike_type')
    if st not in (None, '', 'greater', 'greater_or_equal'):
        return None
    if fs is not None and st == 'greater_or_equal':
        fs = fs - 0.5 if float(fs).is_integer() else None
    from_ticker = None if suffix_number is None else suffix_number - 0.5
    if fs is None:
        return from_ticker
    if from_ticker is not None and fs not in (from_ticker, suffix_number, suffix_number + 0.5):
        return None
    return fs


def parse_market(kind, m, game, settled=False):
    """One Kalshi market -> a flat contract dict for `game`, or None if it can't be read safely.
    team: the team the YES side backs (WINNER/SPREAD); line: what it must win by / the total
    must exceed (SPREAD/TOTAL). yes_ask/no_ask are what buying each side costs right now.
    settled=True also reads finished markets (for the history backtest)."""
    if not settled and m.get('status') not in (None, 'active', 'open'):   # closed, settled, not yet trading
        return None
    suffix = (m.get('ticker') or '').rsplit('-', 1)[-1]
    sm = re.fullmatch(r'([A-Z]*)(\d*)', suffix)
    if not sm:
        return None
    letters, digits = sm.group(1), (int(sm.group(2)) if sm.group(2) else None)
    team = line = None
    if kind in ('WINNER', 'SPREAD'):
        team = team_abbr(letters) if letters else None
        if team not in (game['home'], game['away']):
            return None
    if kind == 'WINNER' and digits is not None:
        return None
    if kind in ('SPREAD', 'TOTAL'):
        line = _strike(m, digits)
        if line is None or (kind == 'TOTAL' and letters):
            return None
    yes_bid, yes_ask = price(m, 'yes_bid'), price(m, 'yes_ask')
    no_bid, no_ask = price(m, 'no_bid'), price(m, 'no_ask')
    # one order book: buying NO at x is selling YES at 1 - x
    if no_ask is None and yes_bid is not None:
        no_ask = round(1 - yes_bid, 4)
    if no_bid is None and yes_ask is not None:
        no_bid = round(1 - yes_ask, 4)
    return {
        'ticker': m['ticker'], 'market': kind, 'game_date': game['game_date'],
        'home': game['home'], 'away': game['away'], 'tip_utc': game.get('tip_utc'),
        'status': m.get('status'), 'team': team, 'line': line,
        'yes_bid': yes_bid, 'yes_ask': yes_ask, 'no_bid': no_bid, 'no_ask': no_ask,
        'volume': _num(m.get('volume_fp', m.get('volume'))),
        'title': m.get('yes_sub_title') or m.get('title'),
    }


def parse_events(kind, events, games):
    """Kalshi events of one series -> contract dicts for the schedule games they match."""
    out = []
    for e in events:
        parsed = parse_event_ticker(e.get('event_ticker'))
        if not parsed:
            continue
        game = match_game(*parsed, games)
        if game is None:
            continue
        for m in e.get('markets') or []:
            c = parse_market(kind, m, game)
            if c is not None:
                out.append(c)
    return out


def nba_contracts(client, games, markets=tuple(SERIES)):
    """Every open Kalshi contract on `games` (schedule rows as dicts) for the given market kinds."""
    out = []
    for kind in markets:
        out += parse_events(kind, client.events(SERIES[kind]), games)
    return out


def parse_orderbook(d):
    """Order book response -> {'yes': [(price, qty)], 'no': [(price, qty)]}: resting BIDS on each
    side, in dollars. Kalshi lists only bids; buying YES takes NO bids (at 1 - their price)."""
    book = d.get('orderbook_fp') or d.get('orderbook') or {}
    out = {}
    for side in ('yes', 'no'):
        rows, scale = book.get(f'{side}_dollars'), 1
        if rows is None:
            rows, scale = book.get(side), 100
        out[side] = [(round(float(p) / scale, 4), float(q)) for p, q in (rows or [])]
    return out


def depth_at(book, side, max_price):
    """Contracts available to buy on `side` at `max_price` or better."""
    other = 'no' if side == 'yes' else 'yes'
    return sum(q for p, q in book.get(other, []) if round(1 - p, 4) <= max_price + 1e-9)


def main():
    from db import connect
    import nba_schedule
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', help='YYYY-MM-DD (default today, US Eastern)')
    ap.add_argument('--demo', action='store_true', help="Kalshi's demo exchange instead of the real one")
    a = ap.parse_args()
    from kalshi_bot import et_today
    d = a.date or et_today()
    games = nba_schedule.games_on(connect(), d).to_dict('records')
    if not games:
        print(f"No games on {d} in the schedule. Run: python nba_schedule.py")
        return
    contracts = nba_contracts(KalshiClient(DEMO_URL if a.demo else BASE_URL), games)
    print(f"{len(contracts)} Kalshi contracts on {len(games)} games, {d}\n")
    print_contracts(contracts)


def print_contracts(contracts):
    """One line per contract, by tip-off, winners first, then the spread and total ladders."""
    from kalshi_bot import describe
    fmt = lambda x: '  - ' if x is None else f"{x * 100:3.0f}¢"
    order = {kind: i for i, kind in enumerate(SERIES)}
    for c in sorted(contracts, key=lambda c: (c['tip_utc'] or '', c['home'], order[c['market']], c['line'] or 0)):
        print(f"{c['away']} @ {c['home']}  {describe(c, 'yes'):<20} yes {fmt(c['yes_bid'])}/{fmt(c['yes_ask'])}  "
              f"no {fmt(c['no_bid'])}/{fmt(c['no_ask'])}  {c['ticker']}")

if __name__ == '__main__':
    main()
