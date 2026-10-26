from datetime import datetime, timezone

import pytest

import kalshi_bot
import kalshi_client
from kalshi_bot import BotConfig, decide, fee, kelly, model_prob, record_bets, record_snapshots, report, settle
from kalshi_client import (depth_at, match_game, parse_event_ticker, parse_events, parse_market, parse_orderbook,
                           split_teams)
from fetch_game_logs import ingest_frames
from game_model import margin_pmf
from tests.conftest import make_league

GAME = {'game_date': '2026-10-26', 'home': 'CHA', 'away': 'WAS', 'tip_utc': '2026-10-26T23:00:00Z'}
NOON = datetime(2026, 10, 26, 16, 0, tzinfo=timezone.utc)
PRED = {'p_home_win': 0.70, 'margin': 7.0, 'total': 225.0}


def mkt(ticker, yes_bid, yes_ask, **kw):
    """A Kalshi market as the API sends it (prices as dollar strings)."""
    return {'ticker': ticker, 'status': 'active', 'yes_bid_dollars': f"{yes_bid:.4f}",
            'yes_ask_dollars': f"{yes_ask:.4f}", 'no_bid_dollars': f"{1 - yes_ask:.4f}",
            'no_ask_dollars': f"{1 - yes_bid:.4f}", **kw}


def contract(kind, ticker, yes_bid, yes_ask, **kw):
    return parse_market(kind, mkt(ticker, yes_bid, yes_ask, **kw), GAME)


# ---------------------------------------------------------------------------
# Fees and Kelly
# ---------------------------------------------------------------------------

def test_fee_rounds_up_per_order():
    assert fee(100, 0.50) == pytest.approx(1.75)         # 1.75 cents a contract at 50 cents
    assert fee(1, 0.58) == pytest.approx(0.02)           # 1.7 cents, rounded up to 2
    assert fee(10, 0.58) == pytest.approx(0.18)          # 17.05 cents -> 18


def test_kelly_for_a_dollar_contract():
    # 60% to win, costs 50 cents: b = 1, full Kelly = 2p - 1
    assert kelly(0.60, 0.50) == pytest.approx(0.20)
    assert kelly(0.40, 0.50) == 0


# ---------------------------------------------------------------------------
# Parsing Kalshi's markets
# ---------------------------------------------------------------------------

def test_parse_event_ticker():
    assert parse_event_ticker('KXNBAGAME-26OCT26WASCHA') == ('2026-10-26', 'WASCHA')
    assert parse_event_ticker('KXNBAGAME-26XYZ26WASCHA') is None
    assert parse_event_ticker('KXNBAGAME') is None


def test_split_teams_handles_short_codes():
    assert split_teams('WASCHA') == [('WAS', 'CHA')]
    assert split_teams('GSDEN') == [('GSW', 'DEN')]
    assert split_teams('NYBOS') == [('NYK', 'BOS')]


def test_match_game_uses_the_schedule_for_home_and_away():
    games = [GAME, {**GAME, 'home': 'DEN', 'away': 'GSW'}]
    assert match_game('2026-10-26', 'WASCHA', games) is GAME
    assert match_game('2026-10-26', 'CHAWAS', games) is GAME    # order in the ticker doesn't matter
    assert match_game('2026-10-27', 'WASCHA', games) is None
    assert match_game('2026-10-26', 'BOSLAL', games) is None


def test_parse_winner_spread_total():
    w = contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.57, 0.58)
    assert (w['team'], w['line'], w['yes_ask'], w['no_ask']) == ('CHA', None, 0.58, 0.43)
    s = contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-WAS7', 0.11, 0.12)
    assert (s['team'], s['line']) == ('WAS', 6.5)
    t = contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.77, 0.79, floor_strike=218.5, strike_type='greater')
    assert (t['team'], t['line']) == (None, 218.5)


def test_parse_market_fails_closed():
    # floor_strike disagrees with the ticker's number
    assert contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.5, 0.52, floor_strike=210.5) is None
    # an "under" style contract we don't price
    assert contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.5, 0.52, strike_type='less') is None
    # a team that isn't in this game
    assert contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-BOS', 0.5, 0.52) is None
    # closed market
    assert contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.5, 0.52, status='closed') is None


def test_parse_market_reads_cents_and_derives_the_no_side():
    m = {'ticker': 'KXNBAGAME-26OCT26WASCHA-CHA', 'status': 'active', 'yes_bid': 57, 'yes_ask': 58}
    c = parse_market('WINNER', m, GAME)
    assert (c['yes_bid'], c['yes_ask'], c['no_bid'], c['no_ask']) == (0.57, 0.58, 0.42, 0.43)


def test_parse_events_skips_unmatched_games():
    events = [{'event_ticker': 'KXNBAGAME-26OCT26WASCHA', 'markets': [
                  mkt('KXNBAGAME-26OCT26WASCHA-CHA', 0.57, 0.58), mkt('KXNBAGAME-26OCT26WASCHA-WAS', 0.42, 0.43)]},
              {'event_ticker': 'KXNBAGAME-26OCT27BOSLAL', 'markets': [mkt('KXNBAGAME-26OCT27BOSLAL-BOS', .5, .51)]}]
    out = parse_events('WINNER', events, [GAME])
    assert [c['team'] for c in out] == ['CHA', 'WAS']


def test_print_contracts_lists_every_market(capsys):
    cs = [contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.70, 0.71),
          contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-CHA2', 0.56, 0.57),
          contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.57, 0.58)]
    kalshi_client.print_contracts(cs)
    lines = capsys.readouterr().out.splitlines()
    assert ['CHA to win' in lines[0], 'CHA by over 1.5' in lines[1], 'Over 218.5' in lines[2]] == [True] * 3


def test_orderbook_depth():
    book = parse_orderbook({'orderbook_fp': {'yes_dollars': [['0.5600', '30.00'], ['0.5700', '20.00']],
                                             'no_dollars': [['0.4100', '50.00'], ['0.4200', '10.00']]}})
    assert depth_at(book, 'yes', 0.58) == 10        # NO bid at 42 = YES offered at 58
    assert depth_at(book, 'yes', 0.59) == 60
    assert depth_at(book, 'no', 0.43) == 20
    legacy = parse_orderbook({'orderbook': {'yes': [[56, 30]], 'no': None}})
    assert legacy == {'yes': [(0.56, 30.0)], 'no': []}


# ---------------------------------------------------------------------------
# Model probabilities and decisions
# ---------------------------------------------------------------------------

def test_model_prob_matches_the_game_model():
    w = contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-WAS', 0.3, 0.31)
    assert model_prob(w, PRED, 12.9, 19.0) == pytest.approx(0.30)
    k, p = margin_pmf(7.0, 12.9)
    home_win = float(p[k > 0].sum())
    # no ties, so "home by over 0.5" is "home wins" and "away by over 0.5" is "away wins"
    s_home = contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-CHA1', 0.5, 0.51)
    s_away = contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-WAS1', 0.5, 0.51)
    assert model_prob(s_home, PRED, 12.9, 19.0) == pytest.approx(home_win)
    assert model_prob(s_away, PRED, 12.9, 19.0) == pytest.approx(1 - home_win)
    s7 = contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-CHA8', 0.5, 0.51)     # over 7.5
    assert model_prob(s7, PRED, 12.9, 19.0) < 0.5


def bank(conn, amount=1000):
    kalshi_bot.set_meta(conn, kalshi_bot.BANKROLL_KEY, amount)


def run_decide(conn, contracts, cfg=None, preds=None, now=NOON, depth=1000):
    return decide(conn, contracts, preds or {('CHA', 'WAS'): PRED}, 12.9, 19.0, cfg or BotConfig(),
                  '2026-10-26', now=now, orderbook=None if depth is None else (lambda x: depth))


def test_bets_the_edge_and_sizes_by_kelly(conn):
    bank(conn)
    # model 70% on CHA, Kalshi 59.5: blend 64.75, EV = 64.75 - 60 - 1.68 = +3.1 cents (5% of the cost)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60),
          contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-WAS', 0.40, 0.41)]
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',)))
    assert len(bets) == 1 and len(rows) == 1
    b = bets[0]
    assert (b['team'], b['side'], b['ask']) in {('CHA', 'yes', 0.60), ('WAS', 'no', 0.60)}
    assert b['ev'] == pytest.approx(0.6475 - 0.60 - 0.07 * 0.60 * 0.40)
    cost_each = 0.60 + 0.07 * 0.60 * 0.40
    stake = 0.25 * (0.6475 - cost_each) / (1 - cost_each) * 1000
    assert b['count'] == int(stake // cost_each)
    assert b['cost'] == pytest.approx(b['count'] * 0.60 + fee(b['count'], 0.60))


def test_no_bet_without_enough_edge(conn):
    bank(conn)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.67, 0.68)]
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',)))
    assert bets == [] and 'below the minimum' in rows[0]['skip']


def test_big_disagreement_is_flagged_not_bet(conn):
    bank(conn)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.45, 0.46)]     # model 70 vs 45.5
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',)))
    assert bets == [] and rows[0]['skip'].startswith('FLAG')


def test_far_ladder_rung_cant_dodge_a_flagged_game(conn):
    bank(conn)
    # model 70% CHA (margin +7) vs Kalshi 45.5%: the winner contract is flagged. A far rung,
    # "CHA by over 12.5" at 20.5 cents vs the model's 33.5%, is only 13 points apart, which the
    # per-contract check alone would let through, but it's the same disagreement about the game.
    rung = contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-CHA13', 0.20, 0.21)
    win = contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.45, 0.46)
    bets, rows = run_decide(conn, [rung, win])
    assert bets == []
    assert all(r['skip'].startswith('FLAG') for r in rows)
    assert 'home margin' in [r for r in rows if r['market'] == 'SPREAD'][0]['skip']


def test_market_centers():
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.49, 0.51),
          contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-WAS', 0.49, 0.51),
          contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-WAS3', 0.38, 0.40),      # tail-ish but in range
          contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-221', 0.49, 0.51),
          contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-251', 0.02, 0.04)]          # tail: ignored
    c = kalshi_bot.market_centers(cs, 12.9, 19.0)[('CHA', 'WAS')]
    assert c['total'] == pytest.approx(220.5)
    assert abs(c['margin']) < 1.5


def test_wide_spread_and_tip_off_are_skipped(conn):
    bank(conn)
    wide = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.55, 0.62)]
    assert run_decide(conn, wide)[0] == []
    ok = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60)]
    assert len(run_decide(conn, ok)[0]) == 1
    late = datetime(2026, 10, 26, 22, 45, tzinfo=timezone.utc)                # 15 min before tip
    bets, rows = run_decide(conn, ok, now=late)
    assert bets == [] and rows[0]['skip'] == 'too close to tip-off'


def test_one_bet_per_game_market_and_caps(conn):
    bank(conn)
    # a ladder of spread rungs all +EV: only the best one is bet
    cs = [contract('SPREAD', f'KXNBASPREAD-26OCT26WASCHA-CHA{n}', p - 0.01, p)
          for n, p in ((2, 0.58), (4, 0.52), (6, 0.46))]
    cs.append(contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60))
    cfg = BotConfig(kelly_mult=1.0, max_disagree=0.5)
    bets, rows = run_decide(conn, cs, cfg)
    assert sorted(b['market'] for b in bets) == ['SPREAD', 'WINNER']
    assert all(b['cost'] <= 0.02 * 1000 + 1e-9 for b in bets)                  # per bet
    assert sum(b['cost'] for b in bets) <= 0.03 * 1000 + 1e-9                  # per game
    assert record_bets(conn, bets, now=NOON) == 2
    # a second run holds both markets already: nothing new, and re-recording is a no-op
    again, rows = run_decide(conn, cs, cfg)
    assert again == [] and all(r['skip'] == 'already holding this game and market' for r in rows)
    assert record_bets(conn, bets, now=NOON) == 0


def test_order_book_depth_limits_the_size(conn):
    bank(conn)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60)]
    bets, _ = run_decide(conn, cs, BotConfig(markets=('WINNER',)), depth=3)
    assert bets[0]['count'] == 3
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',)), depth=0)
    assert bets == [] and rows[0]['skip'] == 'nothing offered at that price'


def test_fee_rounding_can_kill_a_tiny_order(conn):
    bank(conn, 40)          # 2% of $40 buys a single contract, whose fee rounds 1.68 -> 2 cents
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60)]
    # EV is 3.07 cents before rounding, 2.75 after: a 2.9-cent minimum passes one, not the other
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',), min_ev=0.029))
    assert bets == [] and rows[0]['skip'] == 'EV gone after the rounded-up fee'


# ---------------------------------------------------------------------------
# Snapshots, settling, report
# ---------------------------------------------------------------------------

def add_final(conn, home_pts, away_pts):
    for team, opp, h, pts in (('CHA', 'WAS', 1, home_pts), ('WAS', 'CHA', 0, away_pts)):
        conn.execute("INSERT INTO team_games (team_id, team, game_id, game_date, season, season_type, opponent, "
                     "home, pts, reb, ast, stl, blk, tov, fg3m, fg3a, fgm, fga, ftm, fta, oreb, dreb, pf) "
                     "VALUES (?, ?, 'g1', '2026-10-26', 2026, 'Regular Season', ?, ?, ?, "
                     "0,0,0,0,0,0,0,0,0,0,0,0,0,0)", (h, team, opp, h, pts))
    conn.commit()


def test_settle_report_and_clv(conn, monkeypatch):
    bank(conn)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60),
          contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.40, 0.41)]          # model: over 225 > 218.5
    bets, rows = run_decide(conn, cs, BotConfig(max_disagree=0.5))
    record_snapshots(conn, cs, rows, now=NOON)
    assert record_bets(conn, bets, now=NOON) == 2
    # the price moves our way before tip-off; the last pre-tip snapshot is the close
    later = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.65, 0.66),
             contract('TOTAL', 'KXNBATOTAL-26OCT26WASCHA-219', 0.45, 0.46)]
    for c in later:
        c['p_yes'] = model_prob(c, PRED, 12.9, 19.0)
    record_snapshots(conn, later, [], now=datetime(2026, 10, 26, 22, 0, tzinfo=timezone.utc))
    after_tip = later[0] | {'yes_bid': 0.9, 'yes_ask': 0.91}
    record_snapshots(conn, [after_tip], [], now=datetime(2026, 10, 26, 23, 30, tzinfo=timezone.utc))
    snap = conn.execute("SELECT first_mid, last_mid, decision FROM kalshi_snapshots WHERE ticker LIKE '%-CHA'").fetchone()
    assert snap[0] == pytest.approx(0.595) and snap[1] == pytest.approx(0.655) and snap[2] == 'BET'

    assert settle(conn) == 0                    # no box score yet
    add_final(conn, 110, 100)                   # CHA wins, total 210 is under
    assert settle(conn) == 2
    got = {m: (side, res) for m, side, res in conn.execute("SELECT market, side, result FROM kalshi_orders")}
    assert got == {'WINNER': ('yes', 'WIN'), 'TOTAL': ('yes', 'LOSS')}      # CHA won; the over missed

    monkeypatch.setattr(kalshi_bot, 'now_utc', lambda: datetime(2026, 10, 27, 12, tzinfo=timezone.utc))
    r = report(conn)
    assert r['settled'] == 2 and r['clv_n'] == 2
    assert r['clv'] > 0                         # both closes moved toward our side
    profit = sum(p for (p,) in conn.execute("SELECT profit FROM kalshi_orders"))
    assert r['bankroll'] == pytest.approx(1000 + profit)
    assert r['brier_n'] == 1


def test_yes_won():
    assert kalshi_bot.yes_won('WINNER', 'CHA', None, 'CHA', 110, 100)
    assert not kalshi_bot.yes_won('WINNER', 'WAS', None, 'CHA', 110, 100)
    assert kalshi_bot.yes_won('SPREAD', 'CHA', 9.5, 'CHA', 110, 100)
    assert not kalshi_bot.yes_won('SPREAD', 'CHA', 10.5, 'CHA', 110, 100)
    assert kalshi_bot.yes_won('SPREAD', 'WAS', 9.5, 'CHA', 100, 110)
    assert kalshi_bot.yes_won('TOTAL', None, 209.5, 'CHA', 110, 100)


# ---------------------------------------------------------------------------
# End to end against a fake Kalshi API
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeKalshi:
    """Serves /events by series (two pages for the winner series) and /orderbook."""
    def __init__(self, events):
        self.events, self.calls = events, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith('/orderbook'):
            bids = [[f"{p:.4f}", '500'] for p in (0.30, 0.40, 0.50, 0.60, 0.70)]
            return FakeResponse({'orderbook_fp': {'yes_dollars': bids, 'no_dollars': bids}})
        evs = self.events.get(params['series_ticker'], [])
        if params.get('cursor'):
            return FakeResponse({'events': evs[1:], 'cursor': ''})
        return FakeResponse({'events': evs[:1], 'cursor': 'next' if len(evs) > 1 else ''})


def test_scan_end_to_end(conn):
    for season, (p, t) in make_league(games_per_pair=6).items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    gd = '2026-01-10'
    conn.execute("INSERT INTO schedule VALUES ('0022500999', ?, 2025, '2026-01-11T00:30:00Z', '7:30 pm ET', "
                 "'BOS', 'LAL', '', '2026-01-10T00:00:00+00:00')", (gd,))
    conn.execute("INSERT INTO schedule VALUES ('0022500998', ?, 2025, '2026-01-11T00:30:00Z', '7:30 pm ET', "
                 "'MIA', 'NYK', '', '2026-01-10T00:00:00+00:00')", (gd,))
    conn.commit()
    bank(conn)
    from game_model import GameContext
    ctx = GameContext(conn, gd)
    p = ctx.evaluate('BOS', 'LAL')['p_home_win']
    q = round(ctx.evaluate('MIA', 'NYK')['p_home_win'], 2)    # MIA priced right at the model: no bet
    # Kalshi prices BOS 8 points below the model: +EV on BOS, inside the disagreement cap
    yb = round(p - 0.085, 2)
    events = {'KXNBAGAME': [
        {'event_ticker': 'KXNBAGAME-26JAN10LALBOS', 'markets': [
            mkt('KXNBAGAME-26JAN10LALBOS-BOS', yb, yb + 0.01), mkt('KXNBAGAME-26JAN10LALBOS-LAL', 0.99 - yb - 0.01, 0.99 - yb)]},
        {'event_ticker': 'KXNBAGAME-26JAN10NYMIA', 'markets': [mkt('KXNBAGAME-26JAN10NYMIA-MIA', q - 0.01, q)]},
        {'event_ticker': 'KXNBAGAME-26JAN11LALBOS', 'markets': [mkt('KXNBAGAME-26JAN11LALBOS-BOS', .3, .31)]},
    ]}
    fake = FakeKalshi(events)
    client = kalshi_client.KalshiClient(session=fake)
    cfg = BotConfig(markets=('WINNER',))
    now = datetime(2026, 1, 10, 17, tzinfo=timezone.utc)
    orig = kalshi_bot.now_utc
    kalshi_bot.now_utc = lambda: now
    try:
        bets, rows, contracts = kalshi_bot.scan(conn, gd, cfg, client=client, use_injuries=False,
                                                allow_stale=True, log=lambda *_: None)
    finally:
        kalshi_bot.now_utc = orig
    assert {c['home'] for c in contracts} == {'BOS', 'MIA'}       # the Jan 11 event isn't on the slate
    assert len(rows) == 2
    assert len(bets) == 1 and bets[0]['home'] == 'BOS' and bets[0]['p_model'] > bets[0]['p_market']
    assert any(u.endswith('/orderbook') for u, _ in fake.calls)
    assert all(c[1].get('with_nested_markets') == 'true' for c in fake.calls if c[0].endswith('/events'))


def test_scan_fails_closed_on_stale_data(conn):
    for season, (p, t) in make_league(games_per_pair=2).items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    conn.execute("INSERT INTO schedule VALUES ('0022500999', '2026-03-01', 2025, '2026-03-02T00:30:00Z', "
                 "'7:30 pm ET', 'BOS', 'LAL', '', '2026-03-01T00:00:00+00:00')")
    conn.commit()
    with pytest.raises(RuntimeError, match='Ratings use games'):
        kalshi_bot.scan(conn, '2026-03-01', BotConfig(), client=kalshi_client.KalshiClient(session=FakeKalshi({})),
                        use_injuries=False)


def test_summary_helpers(conn, monkeypatch):
    bank(conn)
    cs = [contract('WINNER', 'KXNBAGAME-26OCT26WASCHA-CHA', 0.59, 0.60)]
    bets, rows = run_decide(conn, cs, BotConfig(markets=('WINNER',)))
    record_snapshots(conn, cs, rows, now=NOON)
    record_bets(conn, bets, now=NOON)
    flag = [contract('SPREAD', 'KXNBASPREAD-26OCT26WASCHA-CHA13', 0.20, 0.21)]   # far off the model
    _, frows = run_decide(conn, flag)
    record_snapshots(conn, flag, frows, now=NOON)
    assert kalshi_bot.last_run(conn) == NOON.isoformat(timespec='seconds')
    f = kalshi_bot.flagged(conn, '2026-10-26')
    assert list(f['Game']) == ['WAS @ CHA'] and list(f['Market']) == ['Spread']
    assert not f['Why'].iloc[0].startswith('FLAG')

    monkeypatch.setattr(kalshi_bot, 'now_utc', lambda: NOON)                    # before tip: no CLV yet
    b = kalshi_bot.paper_bets(conn)
    assert list(b['Bet']) == ['CHA to win'] and b['CLV'].isna().all() and list(b['Result']) == ['Pending']
    assert b['Pays'].iloc[0] == pytest.approx(1 / 0.60)
    assert kalshi_bot.paper_bets(conn, 'Finished').empty
    add_final(conn, 110, 100)
    settle(conn)
    d = kalshi_bot.daily_summary(conn)
    assert list(d['Date']) == ['Oct 26'] and d['Won'].iloc[0] == 1
    assert d['ROI'].iloc[0] == pytest.approx(d['Profit'].iloc[0] / d['Staked'].iloc[0] * 100)
