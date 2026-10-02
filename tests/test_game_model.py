import pandas as pd
import pytest

from game_model import (GameContext, GameParams, backtest_games, build_ratings, evaluate_markets,
                        game_rows, margin_pmf, market_probs, predict_game, probs_vs_line)
from fetch_game_logs import ingest_frames
from tests.conftest import make_league


def test_margin_pmf():
    k, p = margin_pmf(3.0, 13.0)
    assert p.sum() == pytest.approx(1)
    assert p[k == 0][0] == 0
    o, u, push = probs_vs_line(k, p, 5)
    assert o + u + push == pytest.approx(1)
    assert push > 0


def test_market_probs_symmetry():
    pred = {'margin': 0.0, 'total': 220.0}
    m = market_probs(pred, 13.0, 18.0, spread=0.5, total=220.5)
    assert m['p_home_win'] == pytest.approx(0.5, abs=1e-9)
    # home +0.5 covers whenever home wins (no ties possible)
    assert m['p_home_cover'] == pytest.approx(m['p_home_win'])
    assert m['p_over'] + m['p_under'] == pytest.approx(1)
    m2 = market_probs({'margin': 6.0, 'total': 220.0}, 13.0, 18.0, spread=-6.5)
    assert m2['p_home_cover'] < 0.5 < m2['p_home_win']


def test_ratings_recover_strength(conn):
    strength = {'BOS': 6, 'LAL': 3, 'GSW': 0, 'DEN': 0, 'MIA': -3, 'NYK': -6}
    for season, (p, t) in make_league(games_per_pair=10, team_strength=strength, seed=3).items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    tg = pd.read_sql_query('SELECT * FROM team_games', conn)
    r = build_ratings(game_rows(tg), '2099-01-01', 2025, GameParams())
    net = {t: v['off'] - v['def'] for t, v in r['teams'].items()}
    order = sorted(net, key=net.get, reverse=True)
    assert order[0] == 'BOS' and order[-1] == 'NYK'
    pred = predict_game(r, 'BOS', 'NYK')
    # true expected margin 2*(6-(-6)) = 24 plus home court; shrinkage pulls it in somewhat
    assert 12 < pred['margin'] < 32
    assert pred['pred_home'] + pred['pred_away'] == pytest.approx(pred['total'])


def test_backtest_no_lookahead(league_conn):
    tg = pd.read_sql_query('SELECT * FROM team_games', league_conn)
    df = backtest_games(tg, [2025])
    assert len(df) > 0
    # changing a game's result must not change its own prediction
    first = df.iloc[10]
    tg2 = tg.copy()
    mask = (tg2['game_date'] == first['game_date']) & (tg2['team'] == first['home'])
    tg2.loc[mask, 'pts'] += 50
    df2 = backtest_games(tg2, [2025])
    row = df2[(df2['game_date'] == first['game_date']) & (df2['home'] == first['home'])].iloc[0]
    assert row['pred_margin'] == pytest.approx(first['pred_margin'])


def test_context_and_markets(league_conn):
    ctx = GameContext(league_conn, '2026-02-01')
    pred = ctx.evaluate('Boston Celtics', 'lal', spread=-3.5, total=220.5)
    ev = evaluate_markets(pred, -3.5, (-110, -110), 220.5, (-110, -110), (-150, 130))
    assert [e['market'] for e in ev] == ['SPREAD', 'TOTAL', 'MONEYLINE']
    for e in ev:
        assert e['pick'] in ('A', 'B', 'PASS')
        assert 0 <= e['fair_a'] <= 1
