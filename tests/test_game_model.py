import numpy as np
import pandas as pd
import pytest

from game_model import (COEF_BOUNDS, GameContext, GameParams, backtest_games, build_ratings, calibrate,
                        evaluate_markets, fit_coefs, game_rows, margin_pmf, market_probs, player_values,
                        predict_game, probs_vs_line, resolve_out, rest_flags, stored_coefs)
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


def _frames(conn):
    return (pd.read_sql_query('SELECT * FROM team_games', conn),
            pd.read_sql_query('SELECT * FROM player_games', conn))


def test_rest_flags():
    rows = pd.DataFrame({'game_id': ['1', '2', '3', '3'], 'team': ['BOS', 'BOS', 'BOS', 'LAL'],
                         'game_date': ['2025-11-01', '2025-11-03', '2025-11-04', '2025-11-04']})
    f = rest_flags(rows)
    assert f[('1', 'BOS')] == 0 and f[('2', 'BOS')] == 0 and f[('3', 'BOS')] == 1 and f[('3', 'LAL')] == 0


def test_player_values_use_only_earlier_games(league_conn):
    _, pg = _frames(league_conn)
    v = player_values(pg, GameParams())
    one = v[v['player_id'] == v['player_id'].iloc[0]].reset_index(drop=True)
    assert (one['val_pre'].iloc[:5] == 0).all()           # fewer than player_min_games before
    assert one['val_pre'].iloc[6] == pytest.approx(one['val_post'].iloc[5])


def test_backtest_lineup_no_lookahead(league_conn):
    tg, pg = _frames(league_conn)
    df = backtest_games(tg, [2025], player_games=pg)
    assert df['lineup_diff'].abs().sum() > 0
    first = df.iloc[20]
    pg2 = pg.copy()
    pg2.loc[pg2['game_id'] == first['game_id'], 'pts'] += 40   # this game's own box score
    df2 = backtest_games(tg, [2025], player_games=pg2)
    row = df2[df2['game_id'] == first['game_id']].iloc[0]
    assert row['pred_margin'] == pytest.approx(first['pred_margin'])


def test_lineup_delta_skipped_in_playoffs(league_conn):
    tg, pg = _frames(league_conn)
    tg.loc[tg['game_id'] == tg['game_id'].iloc[-1], 'season_type'] = 'Playoffs'
    df = backtest_games(tg, [2025], player_games=pg)
    assert df.loc[df['season_type'] == 'Playoffs', 'lineup_diff'].eq(0).all()


def test_star_out_lowers_margin(league_conn):
    ctx = GameContext(league_conn, '2026-02-01')
    roster = ctx.roster('BOS')
    assert len(roster) == 2 and roster['value'].is_monotonic_decreasing
    star = int(roster['player_id'].iloc[0])
    full = ctx.evaluate('BOS', 'LAL')
    hurt = ctx.evaluate('BOS', 'LAL', home_out=[star])
    assert hurt['margin'] < full['margin'] - 1
    assert hurt['p_home_win'] < full['p_home_win']
    assert hurt['total'] == pytest.approx(full['total'])
    assert hurt['adjustments']['lineup'] < 0
    assert resolve_out(ctx, 'BOS', [roster['player_name'].iloc[0]]) == [star]


def test_fit_coefs_recovers_known_values():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({'rating_margin': rng.normal(0, 6, n), 'b2b_diff': rng.integers(-1, 2, n),
                       'lineup_diff': rng.normal(0, 10, n)})
    df['act_margin'] = 1.2 * df['rating_margin'] + 3 * df['b2b_diff'] + 0.15 * df['lineup_diff'] \
        + rng.normal(0, 12, n)
    coefs, sd = fit_coefs(df)
    assert coefs['scale'] == pytest.approx(1.2, abs=0.1)
    assert coefs['b2b'] == pytest.approx(3, abs=1)
    assert coefs['lineup'] == pytest.approx(0.15, abs=0.04)
    assert sd == pytest.approx(12, abs=0.6)
    # out-of-range fits are clipped; no player data -> no lineup term
    df['act_margin'] = -df['rating_margin']
    df['lineup_diff'] = 0.0
    coefs, _ = fit_coefs(df)
    assert coefs['scale'] == COEF_BOUNDS['scale'][0] and coefs['lineup'] == 0


def test_calibrate_stores_coefs(league_conn):
    sd_m, sd_t, n, coefs = calibrate(league_conn, min_games=10)
    assert n > 10 and sd_m > 0 and set(coefs) == {'scale', 'b2b', 'lineup'}
    assert stored_coefs(league_conn) == pytest.approx(coefs)
