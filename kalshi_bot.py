#!/usr/bin/env python3
"""
Paper-trading bot for Kalshi's NBA game contracts (KALSHI_PLAN.md, Phase 1). It reads Kalshi's
live prices, prices every contract with the game model, and "buys" the ones whose edge clears
the fee and a safety margin, sized by quarter Kelly within hard limits. Bets are recorded in
the `kalshi_orders` table against a paper bankroll and settled from the box scores. Nothing is
sent to Kalshi: there is no order code in this project yet.

Per contract bought at price `a` (dollars, pays $1), with the model blended toward the market:
    p   = w * p_model + (1 - w) * p_market         p_market = Kalshi's mid price
    EV  = p - a - fee(a)                           fee(a) = 0.07 * a * (1 - a), rounded up per order
    stake = bankroll * min(kelly_mult * (p - c) / (1 - c), max_bet)        c = a + fee(a)
A payout multiplier is just 1 / a: a 58-cent contract returns 1.72x if it wins.

Timing: each game is only bet between window_minutes (90) and cutoff_minutes (30) before its
tip-off, when most injury statuses are settled. Prices are still recorded on every run for
closing-line value. A game also waits while a player who would be in either team's top
lineup is listed Questionable or Doubtful: the model would assume he plays, so the bot waits
for the news instead of guessing. Day-To-Day doesn't hold a game: ESPN rarely updates it
before tip-off, those players usually play, and the model already assumes they do.

    python kalshi_bot.py scan               # today's +EV contracts, records nothing
    python kalshi_bot.py run                # same, and records paper bets (run it every ~15 min)
    python kalshi_bot.py run --update --game-days-only    # what cron runs: quiet when no games are left
    python kalshi_bot.py run --update       # refresh the NBA data first (like the app's update)
    python kalshi_bot.py settle             # settle finished bets (daily_update.py does this too)
    python kalshi_bot.py report             # paper record, profit, closing-line value, Brier
"""

import argparse
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats as sps

from db import connect, get_meta, init_db, set_meta
from game_model import GameContext, margin_pmf, probs_vs_line, stale_warning, total_pmf

ET = ZoneInfo('America/New_York')
FEE_RATE = 0.07
MODE = 'PAPER'
BANKROLL_KEY = 'kalshi_paper_bankroll'
HOLD_STATUSES = ('Questionable', 'Doubtful')   # injury statuses a game waits on
AUTO_MINUTES = 15          # the app runs the bot at most this often


@dataclass
class BotConfig:
    # Defaults for paper trading. The plan's 3-cent minimum edge with a 10-point disagreement cap
    # leaves almost no bets at w = 0.5 (EV = w * gap - fee needs a gap above 9.4 points), so the
    # minimum is 2 cents and big disagreements are flagged from 15 points.
    blend: float = 0.5           # weight on the model vs Kalshi's price; fit it in the Phase 0 backtest
    min_ev: float = 0.02         # dollars per contract, after fees
    min_roi: float = 0.04        # EV / cost per contract
    max_disagree: float = 0.15   # |model - market| above this: flag for a human, don't bet
    kelly_mult: float = 0.25
    max_bet: float = 0.02        # fractions of the bankroll
    max_game: float = 0.03
    max_day: float = 0.10
    max_spread: float = 0.04     # skip contracts whose bid-ask spread is wider (thin ladders)
    min_price: float = 0.05      # skip longshots and near-certainties
    max_price: float = 0.95
    cutoff_minutes: int = 30     # no bets this close to tip-off
    window_minutes: int = 90     # ... or earlier than this before it (injury news still coming)
    hold_unresolved: bool = True  # wait while a top-lineup player is Questionable or Doubtful
    markets: tuple = ('WINNER', 'SPREAD', 'TOTAL')


def et_today():
    return datetime.now(ET).date().isoformat()


def et_clock(iso):
    """'7:30 pm' (US Eastern) for a UTC ISO time, or None."""
    if not iso:
        return None
    t = pd.Timestamp(iso).tz_convert(ET)
    return f"{t.hour % 12 or 12}:{t:%M} {'am' if t.hour < 12 else 'pm'}"


def now_utc():
    return datetime.now(timezone.utc)


def fee(count, p, rate=FEE_RATE):
    """Kalshi's taker fee for an order: 7% of count * P * (1 - P), rounded up to the cent."""
    return math.ceil(round(rate * count * p * (1 - p) * 100, 6)) / 100


def fee_each(p, rate=FEE_RATE):
    """Fee per contract before rounding (for screening; orders pay fee())."""
    return rate * p * (1 - p)


def kelly(p, cost):
    """Full-Kelly fraction for a $1 contract costing `cost` (price + fee) that wins with prob p."""
    return max((p - cost) / (1 - cost), 0.0) if cost < 1 else 0.0


# ---------------------------------------------------------------------------
# Model probabilities for each contract
# ---------------------------------------------------------------------------

def model_prob(c, pred, sd_margin, sd_total):
    """The model's probability that contract c's YES side wins."""
    if c['market'] == 'WINNER':
        return pred['p_home_win'] if c['team'] == c['home'] else 1 - pred['p_home_win']
    if c['market'] == 'SPREAD':
        k, p = margin_pmf(pred['margin'], sd_margin)      # home margin
        return probs_vs_line(k, p, c['line'])[0] if c['team'] == c['home'] else probs_vs_line(k, p, -c['line'])[1]
    k, p = total_pmf(pred['total'], sd_total)
    return probs_vs_line(k, p, c['line'])[0]


def describe(c, side):
    """'CHA to win', 'CHA by over 6.5', 'Over 218.5'; the NO side reads as its opposite."""
    other = c['away'] if c['team'] == c['home'] else c['home']
    if c['market'] == 'WINNER':
        return f"{c['team'] if side == 'yes' else other} to win"
    if c['market'] == 'SPREAD':
        return f"{c['team']} by over {c['line']:g}" if side == 'yes' else f"{other} +{c['line']:g}"
    return f"{'Over' if side == 'yes' else 'Under'} {c['line']:g}"


def candidates(c, p_yes, cfg):
    """Both sides of contract c as bet candidates, each with why it would be skipped (or None)."""
    out = []
    yb, ya = c['yes_bid'], c['yes_ask']
    mid = (yb + ya) / 2 if yb is not None and ya is not None and 0 < yb < ya < 1 else None
    for side, ask, p_model in (('yes', ya, p_yes), ('no', c['no_ask'], 1 - p_yes)):
        x = {**c, 'side': side, 'ask': ask, 'p_model': p_model, 'desc': describe(c, side),
             'p_market': None, 'p_blend': None, 'ev': None, 'skip': None}
        out.append(x)
        if mid is None or ask is None:
            x['skip'] = 'no two-sided market'
            continue
        x['p_market'] = mid if side == 'yes' else 1 - mid
        x['p_blend'] = cfg.blend * p_model + (1 - cfg.blend) * x['p_market']
        x['ev'] = x['p_blend'] - ask - fee_each(ask)
        x['roi'] = x['ev'] / (ask + fee_each(ask))
        if ya - yb > cfg.max_spread + 1e-9:
            x['skip'] = f"bid-ask spread {(ya - yb) * 100:.0f}¢"
        elif not cfg.min_price <= ask <= cfg.max_price:
            x['skip'] = f"price {ask * 100:.0f}¢ out of range"
        elif abs(p_model - x['p_market']) > cfg.max_disagree:
            x['skip'] = f"FLAG: model and Kalshi differ by {abs(p_model - x['p_market']):.0%}, check news"
        elif x['ev'] < cfg.min_ev or x['roi'] < cfg.min_roi:
            x['skip'] = f"EV {x['ev'] * 100:+.1f}¢ ({x['roi']:+.1%}) below the minimum"
    return out


def market_centers(contracts, sd_margin, sd_total):
    """Kalshi's implied home margin and total for each game, in points: every two-sided rung
    priced 15-85 cents implies a center (t + sd * z(p) for "over t" at probability p, the
    model's own normal), and the median of those is the market's line.
    {(home, away): {'margin': x or None, 'total': y or None}}"""
    est = {}
    for c in contracts:
        yb, ya = c['yes_bid'], c['yes_ask']
        if yb is None or ya is None or not 0 < yb < ya < 1:
            continue
        p = (yb + ya) / 2
        if c['market'] == 'TOTAL':
            kind, t, sd = 'total', c['line'], sd_total
        else:   # "home by over t" is P(margin > t); "away by over t" is P(margin > -t) = 1 - p
            kind, sd = 'margin', sd_margin
            t = 0.0 if c['market'] == 'WINNER' else c['line']
            if c['team'] != c['home']:
                t, p = -t, 1 - p
        if 0.15 <= p <= 0.85:
            est.setdefault((c['home'], c['away']), {}).setdefault(kind, []).append(t + sd * sps.norm.ppf(p))
    return {g: {k: float(np.median(v[k])) if v.get(k) else None for k in ('margin', 'total')}
            for g, v in est.items()}


def game_flag(pred, centers, market, sd_margin, sd_total, cfg):
    """Why the model and Kalshi disagree too much about this game to bet any of its rungs, or
    None. Measured in points, so a far-out ladder rung (where probabilities are compressed) can't
    sneak past a disagreement the winner market flags. The limit is max_disagree converted to
    points at a 50/50 line: 15% is about 4.8 points of margin or 7.1 points of total."""
    kind, sd = ('total', sd_total) if market == 'TOTAL' else ('margin', sd_margin)
    mkt = (centers or {}).get(kind)
    if mkt is None:
        return None
    gap = pred[kind] - mkt
    if abs(gap) <= cfg.max_disagree * sd * math.sqrt(2 * math.pi):
        return None
    what = 'home margin' if kind == 'margin' else 'total'
    return f"FLAG: model {what} {pred[kind]:+.1f} vs Kalshi {mkt:+.1f}, check news"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def unloaded_past_games(conn, game_date):
    """Scheduled games before `game_date` whose box scores aren't in yet: the model would be
    predicting tonight without them."""
    last = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0] or ''
    return conn.execute("SELECT COUNT(*) FROM schedule WHERE game_date > ? AND game_date < ? "
                        "AND status NOT LIKE '%PPD%' AND status NOT LIKE '%ostpone%'",
                        (last, game_date)).fetchone()[0]


def injury_outs(ctx, games, rep=None):
    """From today's ESPN injury report: ({team: [player ids listed Out]}, as the app applies it,
    {team: [(name, status)]} for players listed Questionable or Doubtful whose value would put them in tonight's
    top lineup, so whether they play moves the prediction)."""
    import nba_injuries
    if rep is None:
        rep = nba_injuries.normalize_injuries(nba_injuries.fetch_injuries())
    out, unresolved = {}, {}
    for g in games:
        for team in (g['home'], g['away']):
            r = ctx.roster(team)
            inj = nba_injuries.match_roster(rep, r, team)
            out[team] = [i for i, (status, _) in inj.items() if status == nba_injuries.OUT]
            playing = r.loc[r['recent'] & ~r['player_id'].isin(out[team]), 'value']
            top = playing.sort_values(ascending=False).head(ctx.params.lineup_size)
            bar = top.min() if len(top) == ctx.params.lineup_size else -math.inf
            names = dict(zip(r['player_id'], r['player_name']))
            unresolved[team] = [(names[i], status) for i, (status, _) in inj.items()
                                if status in HOLD_STATUSES and r.loc[r['player_id'] == i, 'value'].iloc[0] >= bar]
    return out, unresolved


def injury_holds(games, unresolved):
    """{(home, away): why} for games waiting on a top-lineup player's status."""
    holds = {}
    for g in games:
        who = [f"{name} ({team}, {status})" for team in (g['away'], g['home'])
               for name, status in unresolved.get(team, ())]
        if who:
            holds[(g['home'], g['away'])] = "waiting on injury news: " + ", ".join(who)
    return holds


def predictions(ctx, games, outs):
    preds = {}
    for g in games:
        preds[(g['home'], g['away'])] = ctx.evaluate(g['home'], g['away'], home_out=outs.get(g['home'], ()),
                                                     away_out=outs.get(g['away'], ()))
    return preds


# ---------------------------------------------------------------------------
# Decisions and sizing
# ---------------------------------------------------------------------------

def bankroll(conn):
    """Paper bankroll: the starting amount plus settled profit."""
    start = float(get_meta(conn, BANKROLL_KEY) or 0)
    won = conn.execute("SELECT COALESCE(SUM(profit), 0) FROM kalshi_orders WHERE mode = ? AND result IS NOT NULL",
                       (MODE,)).fetchone()[0]
    return start + won


def exposure(conn, game_date, home=None, away=None):
    q, args = "SELECT COALESCE(SUM(cost), 0) FROM kalshi_orders WHERE mode = ? AND game_date = ?", [MODE, game_date]
    if home:
        q += " AND home = ? AND away = ?"
        args += [home, away]
    return conn.execute(q, args).fetchone()[0]


def held_markets(conn, game_date):
    return {(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT home, away, market FROM kalshi_orders WHERE mode = ? AND game_date = ?", (MODE, game_date))}


def size(x, bank, room_game, room_day, depth, cfg):
    """Contracts to buy for candidate x, and the order's fee; (0, why) when it's too small."""
    cost_each = x['ask'] + fee_each(x['ask'])
    x['kelly'] = kelly(x['p_blend'], cost_each)
    stake = min(cfg.kelly_mult * x['kelly'], cfg.max_bet) * bank
    stake = min(stake, room_game, room_day)
    n = int(stake // cost_each)
    if depth is not None:
        n = min(n, int(depth))
    if n <= 0:
        return 0, 'stake below one contract' if depth != 0 else 'nothing offered at that price'
    f = fee(n, x['ask'])
    if (n * (x['p_blend'] - x['ask']) - f) / n < cfg.min_ev:     # fee rounding on a small order
        return 0, 'EV gone after the rounded-up fee'
    return n, f


def decide(conn, contracts, preds, sd_margin, sd_total, cfg, game_date, now=None, orderbook=None, holds=None):
    """Best candidate per game and market, sized. Returns (bets, rows) where rows describes the
    best candidate of every game-market (bet or skipped) for printing and the snapshot log.
    holds: {(home, away): why} for games not to bet yet (injury_holds)."""
    now = now or now_utc()
    holds = holds or {}
    bank = bankroll(conn)
    room_day = max(cfg.max_day * bank - exposure(conn, game_date), 0)
    held = held_markets(conn, game_date)
    groups = {}
    for c in contracts:
        if c['market'] not in cfg.markets or (c['home'], c['away']) not in preds:
            continue
        c['p_yes'] = model_prob(c, preds[(c['home'], c['away'])], sd_margin, sd_total)
        for x in candidates(c, c['p_yes'], cfg):
            groups.setdefault((c['home'], c['away'], c['market']), []).append(x)
    centers = market_centers([c for c in contracts if 'p_yes' in c], sd_margin, sd_total)
    for (home, away, market), xs in groups.items():
        flag = game_flag(preds[(home, away)], centers.get((home, away)), market, sd_margin, sd_total, cfg)
        for x in xs:
            if flag and (x['skip'] is None or not x['skip'].startswith('FLAG')):
                x['skip'] = flag
    bets, rows = [], []
    for (home, away, market), xs in sorted(groups.items(), key=lambda kv: (kv[1][0]['tip_utc'] or '', kv[0])):
        ok = [x for x in xs if x['skip'] is None]
        best = max(ok or [x for x in xs if x['ev'] is not None] or xs,
                   key=lambda x: -1 if x['ev'] is None else x['ev'])
        rows.append(best)
        best['hold'] = holds.get((home, away))
        if not ok:
            continue
        tip = pd.Timestamp(best['tip_utc']).to_pydatetime() if best['tip_utc'] else None
        if tip is None or now > tip - timedelta(minutes=cfg.cutoff_minutes):
            best['skip'] = 'too close to tip-off'
            continue
        if now < tip - timedelta(minutes=cfg.window_minutes):
            opens = (tip - timedelta(minutes=cfg.window_minutes)).astimezone(ET)
            best['skip'] = f"betting opens {opens:%-I:%M %p} ET".replace('AM', 'am').replace('PM', 'pm')
            continue
        if cfg.hold_unresolved and (home, away) in holds:
            best['skip'] = holds[(home, away)]
            continue
        if (home, away, market) in held:
            best['skip'] = 'already holding this game and market'
            continue
        room_game = max(cfg.max_game * bank - exposure(conn, game_date, home, away)
                        - sum(b['cost'] for b in bets if (b['home'], b['away']) == (home, away)), 0)
        depth = None
        if orderbook is not None:
            try:
                depth = orderbook(best)
            except Exception as e:  # fail closed: no book, no bet
                best['skip'] = f"order book unavailable ({type(e).__name__})"
                continue
        n, f = size(best, bank, room_game, room_day, depth, cfg)
        if n == 0:
            best['skip'] = f
            continue
        best.update(count=n, fee=f, cost=round(n * best['ask'] + f, 2))
        room_day -= best['cost']
        bets.append(best)
    return bets, rows


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def client_order_id(x):
    """One id per game + market + side + date, so a retried run can never record a bet twice
    (Kalshi rejects a repeated client_order_id the same way once orders are live)."""
    return f"nba-{x['game_date']}-{x['away']}{x['home']}-{x['market']}-{x['side']}".lower()


def record_bets(conn, bets, now=None):
    now = (now or now_utc()).isoformat(timespec='seconds')
    n = 0
    for b in bets:
        cur = conn.execute(
            "INSERT OR IGNORE INTO kalshi_orders (created_at, mode, client_order_id, game_date, home, away, "
            "tip_utc, market, ticker, side, team, line, description, p_model, p_market, p_blend, bid, ask, ev, "
            "kelly, count, price, fee, cost) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, MODE, client_order_id(b), b['game_date'], b['home'], b['away'], b['tip_utc'], b['market'],
             b['ticker'], b['side'], b['team'], b['line'], b['desc'], b['p_model'], b['p_market'], b['p_blend'],
             b['yes_bid'] if b['side'] == 'yes' else b['no_bid'], b['ask'], b['ev'], b['kelly'], b['count'],
             b['ask'], b['fee'], b['cost']))
        n += cur.rowcount
    conn.commit()
    return n


def record_snapshots(conn, contracts, rows, now=None):
    """Kalshi's price and the model's probability for every contract, kept until tip-off: the
    first and last prices seen (the last is the closing price for CLV) and why each game-market
    was bet or skipped. Calibration (model vs Kalshi Brier) uses all of it, not just the bets."""
    now = now or now_utc()
    ts = now.isoformat(timespec='seconds')
    why = {r['ticker']: ('BET' if r['skip'] is None else r['skip']) for r in rows}
    for c in contracts:
        if 'p_yes' not in c or not c['tip_utc'] or now >= pd.Timestamp(c['tip_utc']).to_pydatetime():
            continue
        yb, ya = c['yes_bid'], c['yes_ask']
        mid = (yb + ya) / 2 if yb is not None and ya is not None and 0 < yb < ya < 1 else None
        conn.execute(
            "INSERT INTO kalshi_snapshots (ticker, game_date, home, away, tip_utc, market, team, line, p_model, "
            "yes_bid, yes_ask, first_seen, first_mid, last_seen, last_mid, decision) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET "
            "p_model = excluded.p_model, yes_bid = excluded.yes_bid, yes_ask = excluded.yes_ask, "
            "last_seen = excluded.last_seen, first_mid = COALESCE(first_mid, excluded.last_mid), "
            "last_mid = COALESCE(excluded.last_mid, last_mid), decision = COALESCE(excluded.decision, decision)",
            (c['ticker'], c['game_date'], c['home'], c['away'], c['tip_utc'], c['market'], c['team'], c['line'],
             c['p_yes'], yb, ya, ts, mid, ts, mid, why.get(c['ticker'])))
    conn.commit()


# ---------------------------------------------------------------------------
# Settling and reporting
# ---------------------------------------------------------------------------

def yes_won(market, team, line, home, home_pts, away_pts):
    if market == 'TOTAL':
        return home_pts + away_pts > line
    margin = (home_pts - away_pts) * (1 if team == home else -1)
    return margin > (0 if market == 'WINNER' else line)


def final_score(conn, game_date, home, away):
    r = conn.execute("SELECT pts FROM team_games WHERE game_date = ? AND team = ? AND opponent = ? AND home = 1",
                     (game_date, home, away)).fetchone()
    a = conn.execute("SELECT pts FROM team_games WHERE game_date = ? AND team = ? AND opponent = ? AND home = 0",
                     (game_date, away, home)).fetchone()
    return (r[0], a[0]) if r and a else None


def settle(conn, now=None):
    """Settle paper bets and snapshots whose games are in team_games. Returns bets settled."""
    ts = (now or now_utc()).isoformat(timespec='seconds')
    n = 0
    for r in conn.execute("SELECT id, game_date, home, away, market, team, line, side, count, cost "
                          "FROM kalshi_orders WHERE result IS NULL").fetchall():
        oid, gd, home, away, market, team, line, side, count, cost = r
        score = final_score(conn, gd, home, away)
        if score is None:
            continue
        won = yes_won(market, team, line, home, *score) == (side == 'yes')
        payout = float(count) if won else 0.0
        conn.execute("UPDATE kalshi_orders SET result = ?, payout = ?, profit = ?, settled_at = ? WHERE id = ?",
                     ('WIN' if won else 'LOSS', payout, round(payout - cost, 2), ts, oid))
        n += 1
    for r in conn.execute("SELECT ticker, game_date, home, away, market, team, line FROM kalshi_snapshots "
                          "WHERE result IS NULL").fetchall():
        score = final_score(conn, r[1], r[2], r[3])
        if score is not None:
            conn.execute("UPDATE kalshi_snapshots SET result = ? WHERE ticker = ?",
                         (int(yes_won(r[4], r[5], r[6], r[2], *score)), r[0]))
    conn.commit()
    return n


def report(conn):
    """Paper record and the numbers that decide whether to keep going (KALSHI_PLAN.md)."""
    o = pd.read_sql_query(
        "SELECT o.*, s.last_mid FROM kalshi_orders o LEFT JOIN kalshi_snapshots s ON s.ticker = o.ticker "
        "WHERE o.mode = ?", conn, params=(MODE,))
    s = pd.read_sql_query("SELECT * FROM kalshi_snapshots WHERE result IS NOT NULL AND last_mid IS NOT NULL", conn)
    out = {'bankroll': bankroll(conn), 'bets': len(o)}
    done = o[o['result'].notna()]
    out.update(settled=len(done), wins=int((done['result'] == 'WIN').sum()),
               profit=float(done['profit'].sum()), staked=float(done['cost'].sum()))
    out['roi'] = out['profit'] / out['staked'] if out['staked'] else None
    # closing-line value: our side's last price before tip minus what we paid (games under way)
    close = o['last_mid'].where(o['side'] == 'yes', 1 - o['last_mid'])
    started = pd.to_datetime(o['tip_utc'], utc=True) <= now_utc()
    clv = (close - o['price'])[started].dropna()
    out['clv'] = float(clv.mean()) if len(clv) else None
    out['clv_n'] = len(clv)
    w = s[s['market'] == 'WINNER']
    if len(w):
        out['brier_model'] = float(((w['p_model'] - w['result']) ** 2).mean())
        out['brier_kalshi'] = float(((w['last_mid'] - w['result']) ** 2).mean())
        out['brier_n'] = len(w)
    return out


def paper_bets(conn, show='All', day=None):
    """Paper bets for display, newest first. show: 'Today' | 'Pending' | 'Finished' | 'All'; day (ISO game
    date) narrows that to one day. CLV (cents per contract) is our side's last pre-tip mid minus what we paid,
    once the game has tipped off."""
    where = {'Pending': ' AND o.result IS NULL', 'Finished': ' AND o.result IS NOT NULL',
             'Today': f" AND o.game_date = '{et_today()}'"}.get(show, '')
    params = (MODE,)
    if day:
        where, params = where + ' AND o.game_date = ?', (MODE, day)
    o = pd.read_sql_query(
        "SELECT o.*, s.last_mid FROM kalshi_orders o LEFT JOIN kalshi_snapshots s ON s.ticker = o.ticker "
        f"WHERE o.mode = ?{where} ORDER BY o.game_date DESC, o.tip_utc, o.id", conn, params=params)
    close = o['last_mid'].where(o['side'] == 'yes', 1 - o['last_mid'])
    started = pd.to_datetime(o['tip_utc'], utc=True) <= now_utc()
    return pd.DataFrame({
        'Date': pd.to_datetime(o['game_date']).dt.strftime('%b %-d'), 'Tip-off': o['tip_utc'].map(et_clock),
        'Game': o['away'] + ' @ ' + o['home'],
        'Bet': o['description'],
        'Price': o['price'] * 100, 'Pays': 1 / o['price'], 'Qty': o['count'], 'Cost': o['cost'],
        'Model %': o['p_model'] * 100, 'Kalshi %': o['p_market'] * 100, 'EV': o['ev'] * 100,
        'CLV': ((close - o['price']) * 100).where(started),
        'Result': o['result'].fillna('Pending'), 'Profit': o['profit'],
    })


def bet_days(conn):
    """ISO game dates the bot has paper bets on, newest first."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM kalshi_orders WHERE mode = ? ORDER BY game_date DESC", (MODE,))]


def daily_summary(conn):
    """Per game date: bets, staked, settled profit and ROI."""
    d = pd.read_sql_query(
        "SELECT game_date AS Date, COUNT(*) AS Bets, SUM(cost) AS Staked, "
        "SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS Won, "
        "SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) AS Settled, SUM(profit) AS Profit, "
        "SUM(profit) / NULLIF(SUM(CASE WHEN result IS NOT NULL THEN cost END), 0) * 100 AS ROI "
        "FROM kalshi_orders WHERE mode = ? GROUP BY game_date ORDER BY game_date DESC", conn, params=(MODE,))
    d['Date'] = pd.to_datetime(d['Date']).dt.strftime('%b %-d')
    return d


def flagged(conn, game_date):
    """Game-markets the last run on `game_date` flagged for a human (big model-vs-Kalshi gaps)."""
    return pd.read_sql_query(
        "SELECT away || ' @ ' || home AS Game, market AS Market, decision AS Why FROM kalshi_snapshots "
        "WHERE game_date = ? AND decision LIKE 'FLAG%' GROUP BY home, away, market ORDER BY tip_utc",
        conn, params=(game_date,)).assign(Market=lambda d: d['Market'].str.title(),
                                          Why=lambda d: d['Why'].str.removeprefix('FLAG: '))


def last_run(conn):
    """When `run` last recorded Kalshi prices (UTC ISO), or None."""
    return conn.execute("SELECT MAX(last_seen) FROM kalshi_snapshots").fetchone()[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def scan(conn, game_date, cfg, client=None, use_injuries=True, allow_stale=False, log=print):
    """Price today's Kalshi contracts with the model. Returns (bets, rows, contracts); raises
    RuntimeError when an input is missing or stale (fail closed: no data, no bets)."""
    import kalshi_client
    import nba_schedule
    games = nba_schedule.games_on(conn, game_date).to_dict('records')
    if not games:
        raise RuntimeError(f"No games on {game_date} in the schedule")
    missing = unloaded_past_games(conn, game_date)
    if missing and not allow_stale:
        raise RuntimeError(f"{missing} earlier games aren't loaded yet. Run with --update (or daily_update.py)")
    ctx = GameContext(conn, game_date)
    if (w := stale_warning(ctx)) and not allow_stale:
        raise RuntimeError(w)
    outs, holds = {}, {}
    if use_injuries:
        try:
            outs, unresolved = injury_outs(ctx, games)
        except Exception as e:
            raise RuntimeError(f"Injury report unavailable ({type(e).__name__}: {e}); "
                               "use --no-injuries to bet without it") from e
        holds = injury_holds(games, unresolved)
    preds = predictions(ctx, games, outs)
    client = client or kalshi_client.KalshiClient()
    try:
        contracts = kalshi_client.nba_contracts(client, games, cfg.markets)
    except Exception as e:
        raise RuntimeError(f"Kalshi prices unavailable ({type(e).__name__}: {e})") from e
    log(f"{len(contracts)} Kalshi contracts on {len(games)} games, {game_date}")

    def book_depth(x):
        return kalshi_client.depth_at(client.orderbook(x['ticker']), x['side'], x['ask'])
    bets, rows = decide(conn, contracts, preds, ctx.sd_margin, ctx.sd_total, cfg, game_date, orderbook=book_depth,
                        holds=holds)
    return bets, rows, contracts


def run(conn, game_date=None, cfg=None, use_injuries=True, allow_stale=False, client=None, log=print):
    """Scan and record: paper bets plus the price snapshots. Returns (new bets recorded, bets, rows).
    Raises RuntimeError like scan() when an input is missing (no bets then)."""
    cfg = cfg or BotConfig()
    if get_meta(conn, BANKROLL_KEY) is None:
        set_meta(conn, BANKROLL_KEY, 1000)
    bets, rows, contracts = scan(conn, game_date or et_today(), cfg, client=client, use_injuries=use_injuries,
                                 allow_stale=allow_stale, log=log)
    record_snapshots(conn, contracts, rows)
    n = record_bets(conn, bets)
    set_status(conn, run_summary(n, rows))
    save_outlook(conn, game_date or et_today(), game_reasons(rows))
    return n, bets, rows


BET_PLACED = ('Bet placed', '')


def past_cutoff(hold):
    if hold:
        return 'Missed the cutoff', f"Had an edge, but waited on injury news for {hold} until 30 minutes before tip-off"
    return 'Missed the cutoff', "Had an edge, but only once it was within 30 minutes of tip-off"


def game_reason(rows):
    """(status, details) for one game from its game-markets' rows in the last run, e.g.
    ('Opens 9:30 pm ET', 'Edge on 2 of 3 markets'), or BET_PLACED."""
    if any(r['skip'] is None for r in rows):
        return BET_PLACED
    skips = [r['skip'] or '' for r in rows]
    first = lambda prefix: next((s for s in skips if s.startswith(prefix)), None)
    hold = (rows[0].get('hold') or '').removeprefix('waiting on injury news: ')
    if first('already holding'):
        return BET_PLACED
    if s := first('betting opens'):
        n = sum(s.startswith('betting opens') for s in skips)
        return (s.replace('betting opens', 'Opens'),
                f"Edge on {n} of {len(rows)} markets" + (f"; waiting on injury news: {hold}" if hold else ""))
    if first('waiting on injury news'):
        return 'Waiting on injury news', f"{hold}. Bets once the status is settled"
    if first('too close'):
        return past_cutoff(hold)
    if first('FLAG'):
        status, details = 'Flagged', "Model and Kalshi disagree too much to bet; check the news"
    elif first('EV'):
        status, details = 'No edge', "Kalshi's prices are close to the model's"
    else:
        status, details = 'Skipped', skips[0][:1].upper() + skips[0][1:]
    return status, details + (f"; also waiting on injury news: {hold}" if hold else "")


def game_reasons(rows):
    """[{game, tip_utc, status, details, hold}] per game in tip-off order (see game_reason)."""
    games = {}
    for r in rows:
        games.setdefault((r['tip_utc'] or '', r['away'] + ' @ ' + r['home']), []).append(r)
    return [{'game': g, 'tip_utc': tip or None, **dict(zip(('status', 'details'), game_reason(rs))),
             'hold': (rs[0].get('hold') or '').removeprefix('waiting on injury news: ')}
            for (tip, g), rs in sorted(games.items())]


def save_outlook(conn, game_date, games):
    """Store this run's game_reasons. A game past the cutoff keeps the injury wait that held it through its
    window: ESPN often drops a player's status once the game starts, which would hide why it wasn't bet."""
    prev = {g['game']: g.get('hold') for g in outlook(conn, game_date)[1]}
    for g in games:
        if (g['status'], g['details']) == past_cutoff('') and prev.get(g['game']):
            g.update(hold=prev[g['game']], details=past_cutoff(prev[g['game']])[1])
    set_meta(conn, 'kalshi_outlook', json.dumps({'date': game_date, 'games': games}))


def outlook(conn, game_date):
    """(paper bets on game_date, the last run's game_reasons for it or []), with games bet today as BET_PLACED."""
    n = conn.execute("SELECT COUNT(*) FROM kalshi_orders WHERE mode = ? AND game_date = ?",
                     (MODE, game_date)).fetchone()[0]
    v = json.loads(get_meta(conn, 'kalshi_outlook') or '{}')
    bet = {r[0] for r in conn.execute("SELECT DISTINCT away || ' @ ' || home FROM kalshi_orders "
                                      "WHERE mode = ? AND game_date = ?", (MODE, game_date))}
    games = [g for g in v.get('games', []) if 'status' in g] if v.get('date') == game_date else []
    return n, [{**g, **dict(zip(('status', 'details'), BET_PLACED))} if g['game'] in bet else g for g in games]


def run_summary(n, rows):
    """'1 new paper bet; waiting on injury news for WAS @ CHA; next game opens for betting 7:00 pm ET'"""
    parts = [f"{n} new paper bet{'s' * (n != 1)}" if n else "no new bets"]
    waiting = list(dict.fromkeys(f"{r['away']} @ {r['home']}" for r in rows
                                 if (r['skip'] or '').startswith('waiting on injury news')))
    if waiting:
        parts.append("waiting on injury news for " + ", ".join(waiting))
    opens = sorted((r['tip_utc'], r['skip']) for r in rows if (r['skip'] or '').startswith('betting opens'))
    if opens:
        parts.append("next game " + opens[0][1].replace('betting opens', 'opens for betting'))
    return "; ".join(parts)


def set_status(conn, text):
    set_meta(conn, 'kalshi_last_status', f"{now_utc().isoformat(timespec='seconds')}|{text}")


def last_status(conn):
    """(when, what) the last run did, e.g. ('2026-10-26T20:15:00+00:00', '3 new paper bets'), or None."""
    v = get_meta(conn, 'kalshi_last_status')
    return tuple(v.split('|', 1)) if v else None


def due(conn, now=None, cfg=None):
    """Whether the app should run the bot now: a game today is still before the betting cutoff,
    and the last attempt was at least AUTO_MINUTES ago."""
    now = now or now_utc()
    cfg = cfg or BotConfig()
    last = get_meta(conn, 'kalshi_last_attempt')
    if last and now - datetime.fromisoformat(last) < timedelta(minutes=AUTO_MINUTES):
        return False
    return games_left(conn, now, cfg.cutoff_minutes) > 0


def games_left(conn, now=None, minutes=0):
    """Today's games (US Eastern) tipping off more than `minutes` from now."""
    now = now or now_utc()
    after = (now + timedelta(minutes=minutes)).strftime('%Y-%m-%dT%H:%M:%SZ')
    return conn.execute("SELECT COUNT(*) FROM schedule WHERE game_date = ? AND tip_utc > ?",
                        (now.astimezone(ET).date().isoformat(), after)).fetchone()[0]


def attempt(conn):
    """One run for the app's background job: never raises, records what happened. Returns it."""
    set_meta(conn, 'kalshi_last_attempt', now_utc().isoformat(timespec='seconds'))
    try:
        return f"{run(conn, log=lambda *_: None)[0]} new paper bets"
    except Exception as e:  # missing data, Kalshi or ESPN down: no bets, say why
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        set_status(conn, f"no bets: {msg}")
        return f"no bets ({msg})"


def print_rows(rows, bets, bank):
    pct = lambda v: '   -' if v is None else f"{v * 100:3.0f}%"
    print(f"\nPaper bankroll ${bank:,.2f}\n")
    print(f"{'Game':<11} {'Bet':<26} {'Price':>5} {'Pays':>6} {'Model':>5} {'Kalshi':>6} {'Blend':>5} "
          f"{'EV':>6}  Decision")
    for r in rows:
        ev = '     -' if r['ev'] is None else f"{r['ev'] * 100:+5.1f}¢"
        price = '    -' if r['ask'] is None else f"{r['ask'] * 100:4.0f}¢"
        pays = '     -' if not r['ask'] else f"{1 / r['ask']:5.2f}x"
        what = (f"BET {r['count']} for ${r['cost']:.2f}" if r['skip'] is None else r['skip'])
        print(f"{r['away'] + ' @ ' + r['home']:<11} {r['desc']:<26} {price} {pays} {pct(r['p_model'])} "
              f"{pct(r['p_market']):>6} {pct(r['p_blend'])} {ev}  {what}")
    total = sum(b['cost'] for b in bets)
    print(f"\n{len(bets)} bet{'s' * (len(bets) != 1)}, ${total:,.2f} staked ({total / bank:.1%} of the bankroll)"
          if bank else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['scan', 'run', 'settle', 'report'])
    ap.add_argument('--date', help='game date YYYY-MM-DD (default today, US Eastern)')
    ap.add_argument('--bankroll', type=float, help='set the paper bankroll (first run: default $1,000)')
    ap.add_argument('--markets', default='winner,spread,total', help='comma list of winner,spread,total')
    ap.add_argument('--update', action='store_true', help='refresh NBA data before scanning')
    ap.add_argument('--no-injuries', action='store_true', help="don't apply ESPN's injury report")
    ap.add_argument('--allow-stale', action='store_true', help='bet even if the model data is behind')
    ap.add_argument('--window-minutes', type=int, default=BotConfig.window_minutes,
                    help='bet a game only this many minutes before tip-off or later (default 90)')
    ap.add_argument('--bet-through-injuries', action='store_true',
                    help="don't wait on top-lineup players listed Questionable/Doubtful (assume they play)")
    ap.add_argument('--game-days-only', action='store_true',
                    help='exit quietly unless a game today has yet to tip off (for cron)')
    for f in fields(BotConfig):
        if f.type is float:
            ap.add_argument('--' + f.name.replace('_', '-'), type=float, default=f.default)
    a = ap.parse_args()
    conn = connect()
    init_db(conn)
    if a.bankroll is not None:
        set_meta(conn, BANKROLL_KEY, a.bankroll)
    elif get_meta(conn, BANKROLL_KEY) is None:
        set_meta(conn, BANKROLL_KEY, 1000)

    if a.command == 'settle':
        print(f"Settled {settle(conn)} paper bets")
        return
    if a.command == 'report':
        settle(conn)
        r = report(conn)
        print(f"Paper bankroll ${r['bankroll']:,.2f}  ·  {r['bets']} bets, {r['settled']} settled "
              f"({r['wins']}-{r['settled'] - r['wins']})")
        if r['staked']:
            print(f"Profit ${r['profit']:+,.2f} on ${r['staked']:,.2f} staked (ROI {r['roi']:+.1%})")
        if r['clv'] is not None:
            print(f"Closing-line value {r['clv'] * 100:+.2f}¢ per contract over {r['clv_n']} bets "
                  "(positive = Kalshi's price moved our way)")
        if 'brier_n' in r:
            print(f"Winner Brier over {r['brier_n']} contracts: model {r['brier_model']:.4f}, "
                  f"Kalshi {r['brier_kalshi']:.4f} (lower is better)")
        return

    if a.game_days_only and not a.date and not games_left(conn):
        return
    if a.update:
        from daily_update import update
        update(conn, quick=True)
    cfg = BotConfig(**{f.name: getattr(a, f.name) for f in fields(BotConfig) if f.type is float},
                    markets=tuple(m.strip().upper() for m in a.markets.split(',') if m.strip()),
                    window_minutes=a.window_minutes, hold_unresolved=not a.bet_through_injuries)
    game_date = a.date or et_today()
    try:
        if a.command == 'run':
            n, bets, rows = run(conn, game_date, cfg, use_injuries=not a.no_injuries, allow_stale=a.allow_stale)
        else:
            bets, rows, _ = scan(conn, game_date, cfg, use_injuries=not a.no_injuries, allow_stale=a.allow_stale)
    except RuntimeError as e:
        if a.command == 'run':
            set_status(conn, f"no bets: {e}")
        raise SystemExit(f"No bets: {e}")
    print_rows(rows, bets, bankroll(conn))
    if a.command == 'run':
        print(f"Recorded {n} new paper bets")


if __name__ == '__main__':
    main()
