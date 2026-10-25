import math

from bet_signals import evaluate_prop, log_prop, prop_log_id, settle_props, untrack_prop
from db import connect, init_db
from game_model import GameContext, evaluate_markets, game_log_id, log_game_evals, settle_games, untrack_game_bet
from projection_model import PropContext
from tracking import (HIT, MISS, PENDING, VOID, game_summary, prop_summary, tracked_games,
                      tracked_props)
from tests.test_settle import _game


def _prop(ctx, pid, line, pick, game_date=None, p_over=None):
    r = evaluate_prop(ctx, pid, 'PTS', line, -110, -110, min_ev=-1.0)
    r['pick'] = pick
    if p_over is not None:
        r['p_over'], r['p_under'] = p_over, 1 - p_over
    if game_date:
        r['game_date'] = game_date
    return r


def test_prop_tracking_statuses_and_record(league_conn):
    d, home, _ = _game(league_conn)
    pid, actual = league_conn.execute("SELECT player_id, pts FROM player_games WHERE game_date=? AND team=? "
                                      "ORDER BY minutes DESC", (d, home)).fetchone()
    latest = league_conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    ctx = PropContext(league_conn, d)
    hit = _prop(ctx, pid, actual - 0.5, 'OVER')
    miss = _prop(ctx, pid, actual + 0.5, 'OVER')
    lean = _prop(ctx, pid, actual - 1.5, 'PASS', p_over=0.6)        # PASS, model leaned OVER -> hit
    dnp = _prop(ctx, pid, 20.5, 'UNDER', game_date='2025-10-01')    # before latest, no game -> void
    future = _prop(ctx, pid, 20.5, 'UNDER', game_date='2099-01-01')
    untracked = _prop(ctx, pid, actual + 3.5, 'UNDER')
    for r in (hit, miss, lean, dnp, future):
        log_prop(league_conn, r, tracked=True)
    log_prop(league_conn, untracked)
    log_prop(league_conn, hit)                                      # re-logging never un-tracks
    assert prop_log_id(league_conn, hit, tracked_only=True)
    assert prop_log_id(league_conn, untracked, tracked_only=True) is None
    settle_props(league_conn)

    s = prop_summary(league_conn)
    assert (s['logged'], s['tracked'], s['pending']) == (6, 5, 1)
    assert s['accuracy'] == {'hits': 2, 'misses': 1, 'rate': 2 / 3}
    rec = s['record']                                               # the PASS isn't a bet
    assert (rec['wins'], rec['losses'], rec['pushes']) == (1, 1, 0)
    assert math.isclose(rec['profit'], 100 / 110 - 1)

    t = tracked_props(league_conn).set_index('Bet')
    assert t.loc[f"PTS Over {actual - 0.5:g} (-110)", 'Result'] == HIT
    assert t.loc[f"PTS Over {actual + 0.5:g} (-110)", 'Profit (u)'] == -1
    lean_row = t.loc[f"PTS Over {actual - 1.5:g} (-110)"]
    assert lean_row['Type'] == 'Lean' and lean_row['Result'] == HIT and math.isnan(lean_row['Profit (u)'])
    assert sorted(t['Result']) == sorted([HIT, MISS, HIT, VOID, PENDING])
    assert list(tracked_props(league_conn, 'Pending')['Date']) == ['2099-01-01']
    assert len(tracked_props(league_conn, 'Finished')) == 4
    assert d < latest

    assert untrack_prop(league_conn, prop_log_id(league_conn, miss))
    assert prop_summary(league_conn)['tracked'] == 4
    assert prop_summary(league_conn)['logged'] == 6                 # still there for calibration


def test_game_tracking(league_conn):
    d, home, away = _game(league_conn)
    hp, ap = league_conn.execute("""SELECT h.pts, a.pts FROM team_games h JOIN team_games a
        ON a.game_id=h.game_id AND a.team=h.opponent WHERE h.game_date=? AND h.home=1""", (d,)).fetchone()
    spread, total = -(hp - ap) + 0.5, float(hp + ap) + 10.5       # home covers by half a point
    pred = GameContext(league_conn, d).evaluate(home, away, spread=spread, total=total)
    evals = evaluate_markets(pred, spread, (-110, -110), total, (-110, -110), min_ev=-1.0)
    sp, tot = (next(e for e in evals if e['market'] == m) for m in ('SPREAD', 'TOTAL'))
    sp['pick'], tot['pick'] = 'A', 'A'                               # home spread hits, over misses
    log_game_evals(league_conn, d, pred, evals)                     # silent calibration log
    assert game_summary(league_conn)['tracked'] == 0
    log_game_evals(league_conn, d, pred, [sp], tracked=True)
    log_game_evals(league_conn, d, pred, [tot], tracked=True)
    assert league_conn.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 2
    settle_games(league_conn)

    t = tracked_games(league_conn).set_index('Result')
    assert t.loc[HIT, 'Bet'] == f"{home} {spread:+g} (-110)"
    assert t.loc[MISS, 'Bet'].startswith('Over ')
    assert t.loc[HIT, 'Final'] == f"{ap} – {hp}"
    s = game_summary(league_conn)
    assert s['accuracy']['hits'] == 1 and s['record']['losses'] == 1
    assert untrack_game_bet(league_conn, game_log_id(league_conn, d, pred, tot))
    assert game_summary(league_conn)['tracked'] == 1


def test_empty_tables(conn):
    assert tracked_props(conn).empty and tracked_games(conn).empty
    assert prop_summary(conn)['accuracy']['rate'] is None and game_summary(conn)['record'] is None


def test_old_logs_get_tracked_column(tmp_path):
    c = connect(tmp_path / 'old.db')
    init_db(c)
    c.execute("ALTER TABLE prop_log DROP COLUMN tracked")            # a log from before tracking existed
    for pick, line in (('OVER', 1.5), ('PASS', 2.5)):
        c.execute("INSERT INTO prop_log (logged_at, game_date, player_id, player_name, stat, line, over_odds, "
                  "under_odds, book, proj_mean, proj_sd, p_over, p_under, p_push, fair_over, ev_over, ev_under, "
                  "pick) VALUES ('x', '2025-11-01', 1, 'A', 'PTS', ?, -110, -110, '', 1, 1, .5, .5, 0, .5, 0, 0, ?)",
                  (line, pick))
    init_db(c)
    assert c.execute("SELECT pick, tracked FROM prop_log ORDER BY line").fetchall() == [('OVER', 1), ('PASS', 0)]
    init_db(c)                                                      # idempotent
    c.close()
