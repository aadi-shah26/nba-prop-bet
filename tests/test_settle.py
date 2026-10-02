import pytest

from bet_signals import evaluate_prop, log_prop, report, settle_props
from game_model import GameContext, evaluate_markets, log_game_evals, settle_games
from projection_model import PropContext


def _game(conn, season=2025, idx=40):
    d = [r[0] for r in conn.execute("SELECT DISTINCT game_date FROM team_games WHERE season=? "
                                    "ORDER BY game_date", (season,))][idx]
    home, away = conn.execute("SELECT team, opponent FROM team_games WHERE game_date=? AND home=1",
                              (d,)).fetchone()
    return d, home, away


def test_prop_settlement(league_conn, capsys):
    d, home, away = _game(league_conn)
    pid, actual = league_conn.execute(
        "SELECT player_id, pts FROM player_games WHERE game_date=? AND team=? ORDER BY minutes DESC",
        (d, home)).fetchone()
    ctx = PropContext(league_conn, d)
    for line in (actual - 0.5, actual + 0.5, float(actual)):
        r = evaluate_prop(ctx, pid, 'PTS', line, -110, -110, opponent=away, min_ev=-1.0)  # force a pick
        assert r['pick'] in ('OVER', 'UNDER')
        log_prop(league_conn, r, 'testbook')
    # a prop for a game the player didn't play (date with games after it in DB) -> DNP
    r = evaluate_prop(ctx, pid, 'PTS', 20.5, -110, -110, min_ev=1.0)
    r['game_date'] = '2025-10-01'
    log_prop(league_conn, r, 'testbook')
    assert settle_props(league_conn) == 4
    rows = league_conn.execute("SELECT line, pick, actual, result FROM prop_log ORDER BY id").fetchall()
    for line, pick, act, res in rows[:3]:
        assert act == actual
        side = 'OVER' if act > line else 'UNDER' if act < line else 'PUSH'
        assert res == ('PUSH' if side == 'PUSH' else 'WIN' if side == pick else 'LOSS')
    assert rows[3][3] == 'DNP'
    assert settle_props(league_conn) == 0  # idempotent
    report(league_conn)
    assert 'Picks: 3' in capsys.readouterr().out


def test_prop_log_dedup(league_conn):
    d, home, away = _game(league_conn)
    pid = league_conn.execute("SELECT player_id FROM player_games WHERE game_date=? LIMIT 1", (d,)).fetchone()[0]
    r = evaluate_prop(PropContext(league_conn, d), pid, 'REB', 4.5)
    log_prop(league_conn, r, 'x')
    log_prop(league_conn, r, 'x')
    assert league_conn.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 1


def test_game_settlement(league_conn):
    d, home, away = _game(league_conn)
    hp, ap = league_conn.execute("""SELECT h.pts, a.pts FROM team_games h JOIN team_games a
        ON a.game_id=h.game_id AND a.team=h.opponent WHERE h.game_date=? AND h.home=1""", (d,)).fetchone()
    ctx = GameContext(league_conn, d)
    margin, total = hp - ap, hp + ap
    pred = ctx.evaluate(home, away, spread=-margin, total=float(total))      # both exact -> pushes
    evals = evaluate_markets(pred, -margin, (-110, -110), float(total), (-110, -110), (-110, -110),
                             min_ev=-1.0)
    log_game_evals(league_conn, d, pred, evals, 'book')
    assert settle_games(league_conn) == 3
    res = dict(league_conn.execute("SELECT market, result FROM game_log").fetchall())
    assert res['SPREAD'] == 'PUSH' and res['TOTAL'] == 'PUSH'
    ml_pick = league_conn.execute("SELECT pick FROM game_log WHERE market='MONEYLINE'").fetchone()[0]
    assert res['MONEYLINE'] == ('WIN' if (ml_pick == 'A') == (hp > ap) else 'LOSS')


def test_game_log_dedup_without_book(league_conn):
    d, home, away = _game(league_conn)
    ctx = GameContext(league_conn, d)
    pred = ctx.evaluate(home, away, spread=-2.5)
    evals = evaluate_markets(pred, -2.5, (-110, -110), ml=(-150, 130))   # moneyline has line NULL
    log_game_evals(league_conn, d, pred, evals)
    log_game_evals(league_conn, d, pred, evals)
    assert league_conn.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 2


def test_neutral_site_game_settles(league_conn):
    d, home, away = _game(league_conn)
    league_conn.execute("UPDATE team_games SET home = 0 WHERE game_date = ?", (d,))
    ctx = GameContext(league_conn, d)
    pred = ctx.evaluate(home, away, total=200.5)
    log_game_evals(league_conn, d, pred, evaluate_markets(pred, total=200.5, total_odds=(-110, -110),
                                                         min_ev=-1.0))
    assert settle_games(league_conn) == 1


def test_manual_prop_settle_correct_void_clear_delete(league_conn):
    from bet_signals import delete_prop, settle_prop_manual
    d, home, away = _game(league_conn)
    pid = league_conn.execute("SELECT player_id FROM player_games WHERE game_date=? AND team=? "
                              "ORDER BY minutes DESC", (d, home)).fetchone()[0]
    r = evaluate_prop(PropContext(league_conn, d), pid, 'PTS', 20.5, -110, -110, min_ev=-1.0)
    log_prop(league_conn, r)
    row_id = league_conn.execute("SELECT id FROM prop_log").fetchone()[0]
    pick = r['pick']
    win_val, lose_val = (25, 15) if pick == 'OVER' else (15, 25)
    assert settle_prop_manual(league_conn, row_id, actual=win_val) == 'WIN'
    assert settle_prop_manual(league_conn, row_id, actual=lose_val) == 'LOSS'     # correction
    # automatic settling never overwrites a manual result
    settle_props(league_conn)
    assert league_conn.execute("SELECT actual FROM prop_log").fetchone()[0] == lose_val
    assert settle_prop_manual(league_conn, row_id, dnp=True) == 'DNP'
    assert settle_prop_manual(league_conn, row_id) is None                        # clear
    assert league_conn.execute("SELECT result FROM prop_log").fetchone()[0] is None
    assert delete_prop(league_conn, row_id)
    assert league_conn.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 0
    assert not delete_prop(league_conn, row_id)


def test_manual_game_settle_and_delete(league_conn):
    from game_model import delete_game_bets, settle_game_manual
    d, home, away = _game(league_conn)
    pred = GameContext(league_conn, d).evaluate(home, away, spread=-3.5, total=220.5)
    evals = evaluate_markets(pred, -3.5, (-110, -110), 220.5, (-110, -110), (-150, 130), min_ev=-1.0)
    log_game_evals(league_conn, d, pred, evals)
    assert settle_game_manual(league_conn, d, home, away, 115, 108) == 3
    res = dict(league_conn.execute("SELECT market, result FROM game_log").fetchall())
    picks = dict(league_conn.execute("SELECT market, pick FROM game_log").fetchall())
    # home won by 7: covers -3.5 (A), total 223 > 220.5 (A), moneyline home (A)
    for m in ('SPREAD', 'TOTAL', 'MONEYLINE'):
        assert res[m] == ('WIN' if picks[m] == 'A' else 'LOSS')
    with pytest.raises(ValueError):
        settle_game_manual(league_conn, d, home, away, 100, 100)
    assert delete_game_bets(league_conn, d, home, away) == 3
    assert league_conn.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 0
