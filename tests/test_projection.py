import numpy as np
import pandas as pd
import pytest

from feature_engineering import (PropParams, league_priors, opponent_factors, parse_stat,
                                 project_from_history, recency_weights, team_games_with_allowed)
from projection_model import PropContext


def test_parse_stat():
    assert parse_stat('pts') == 'PTS'
    assert parse_stat('3PM') == 'FG3M'
    assert parse_stat('p+r+a') == 'PRA'
    assert parse_stat('AR') == 'RA'
    with pytest.raises(ValueError):
        parse_stat('dunks')


def test_recency_weights():
    w = recency_weights(np.array([2025] * 3), 2025, half_life=1, prev_season_weight=0.5)
    assert w.tolist() == [0.25, 0.5, 1.0]
    w = recency_weights(np.array([2024, 2025]), 2025, half_life=1e9, prev_season_weight=0.5)
    assert w == pytest.approx([0.5, 1.0])


def _hist(n, pts, minutes=30.0, season=2025):
    return pd.DataFrame({'season': season, 'minutes': minutes, 'pts': pts, 'reb': 5, 'ast': 4,
                         'stl': 1, 'blk': 0, 'tov': 2, 'fg3m': 2}, index=range(n))


PRIORS = {'rate': {'pts': 0.5, 'reb': 0.2, 'ast': 0.12, 'stl': 0.03, 'blk': 0.02, 'tov': 0.05,
                   'fg3m': 0.04}, 'alpha': {k: 0.02 for k in
                                            ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M',
                                             'PRA', 'PR', 'PA', 'RA', 'SB']}}


def test_projection_scales_with_minutes_and_opponent():
    rng = np.random.default_rng(1)
    h = _hist(40, rng.poisson(24, 40))
    params = PropParams(rate_prior_minutes=0)
    base = project_from_history(h, 'PTS', 2025, PRIORS, params)
    assert base['mean'] == pytest.approx(h['pts'].mean(), rel=0.08)
    half = project_from_history(h, 'PTS', 2025, PRIORS, params, minutes=15)
    assert half['mean'] == pytest.approx(base['mean'] / 2)
    soft = project_from_history(h, 'PTS', 2025, PRIORS, params, opp_factor={'pts': 1.1})
    assert soft['mean'] == pytest.approx(base['mean'] * 1.1)
    assert base['var'] >= base['mean']   # never under-dispersed
    combo = project_from_history(h, 'PRA', 2025, PRIORS, params)
    assert combo['mean'] == pytest.approx(base['mean'] + 5 + 4, rel=1e-6)


def test_recent_games_matter_more():
    h = _hist(40, [10] * 30 + [30] * 10)
    p = project_from_history(h, 'PTS', 2025, PRIORS, PropParams(rate_prior_minutes=0))
    assert p['mean'] > (10 * 30 + 30 * 10) / 40   # above the flat average


def test_too_few_games():
    assert project_from_history(_hist(3, [20] * 3), 'PTS', 2025, PRIORS, PropParams()) is None


def test_opponent_factor_direction(league_conn):
    tg = pd.read_sql_query('SELECT * FROM team_games', league_conn)
    tga = team_games_with_allowed(tg)
    # make BOS concede 20 more points in every game
    tga.loc[tga['team'] == 'BOS', 'allowed_pts'] += 20
    f = opponent_factors(tga, '2099-01-01', 2025, PropParams())
    assert f['BOS']['pts'] > 1.05
    assert all(f[t]['pts'] < 1.0 for t in f if t != 'BOS')


def test_context_has_no_lookahead(league_conn):
    dates = [r[0] for r in league_conn.execute(
        "SELECT DISTINCT game_date FROM player_games WHERE season = 2025 ORDER BY game_date")]
    d = dates[30]
    ctx = PropContext(league_conn, d)
    assert ctx.player_games['game_date'].max() < d
    assert (ctx.tga['game_date'] < d).all() or ctx.tga['game_date'].max() < d
    pid = int(ctx.player_games['player_id'].iloc[0])
    p = ctx.project(pid, 'PTS', opponent='LAL')
    assert p is not None and p['last_game'] < d
    # adding a monster game ON the game date must not change the projection
    league_conn.execute("UPDATE player_games SET pts = 99 WHERE game_date = ?", (d,))
    p2 = PropContext(league_conn, d).project(pid, 'PTS', opponent='LAL')
    assert p2['mean'] == pytest.approx(p['mean'])


def test_league_priors(league_conn):
    pg = pd.read_sql_query('SELECT * FROM player_games', league_conn)
    pr = league_priors(pg)
    assert pr['rate']['pts'] == pytest.approx(pg['pts'].sum() / pg['minutes'].sum())
    assert all(a >= 0 for a in pr['alpha'].values())
