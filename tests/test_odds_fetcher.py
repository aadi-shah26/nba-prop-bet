import os

os.environ.setdefault('ODDS_API_KEY', 'test')

from odds_fetcher import et_date, parse_event_props, parse_game_lines  # noqa: E402

EVENT = {
    'id': 'abc', 'commence_time': '2026-11-02T00:30:00Z',
    'home_team': 'Los Angeles Lakers', 'away_team': 'Boston Celtics',
    'bookmakers': [
        {'key': 'draftkings', 'markets': [
            {'key': 'player_points', 'outcomes': [
                {'name': 'Over', 'description': 'Jayson Tatum', 'price': -115, 'point': 27.5},
                {'name': 'Under', 'description': 'Jayson Tatum', 'price': -105, 'point': 27.5},
                {'name': 'Over', 'description': 'LeBron James', 'price': -110, 'point': 24.5},
            ]},
            {'key': 'player_threes', 'outcomes': [
                {'name': 'Over', 'description': 'Jayson Tatum', 'price': 120, 'point': 3.5},
                {'name': 'Under', 'description': 'Jayson Tatum', 'price': -150, 'point': 3.5},
            ]},
        ]},
    ],
}


def test_et_date():
    # 00:30 UTC Nov 2 is the evening of Nov 1 in New York
    assert et_date('2026-11-02T00:30:00Z') == '2026-11-01'


def test_parse_event_props_pairs_sides():
    rows = parse_event_props(EVENT)
    assert len(rows) == 2  # LeBron only has one side -> dropped
    tatum = {r['stat']: r for r in rows}
    assert tatum['PTS']['over_odds'] == -115 and tatum['PTS']['under_odds'] == -105
    assert tatum['FG3M']['line'] == 3.5
    assert tatum['PTS']['home'] == 'LAL' and tatum['PTS']['away'] == 'BOS'
    assert tatum['PTS']['game_date'] == '2026-11-01'


def test_parse_game_lines():
    ev = [{'commence_time': '2026-11-02T00:30:00Z', 'home_team': 'Los Angeles Lakers',
           'away_team': 'Boston Celtics', 'bookmakers': [{'key': 'draftkings', 'markets': [
               {'key': 'h2h', 'outcomes': [{'name': 'Los Angeles Lakers', 'price': 140},
                                           {'name': 'Boston Celtics', 'price': -165}]},
               {'key': 'spreads', 'outcomes': [{'name': 'Los Angeles Lakers', 'price': -110, 'point': 3.5},
                                               {'name': 'Boston Celtics', 'price': -110, 'point': -3.5}]},
               {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -108, 'point': 228.5},
                                              {'name': 'Under', 'price': -112, 'point': 228.5}]}]}]}]
    g = parse_game_lines(ev, 'draftkings', '2026-11-01')[0]
    assert g['home'] == 'LAL' and g['spread'] == 3.5 and g['spread_odds'] == (-110, -110)
    assert g['ml'] == (140, -165) and g['total'] == 228.5 and g['total_odds'] == (-108, -112)
    assert parse_game_lines(ev, 'fanduel', '2026-11-01') == []
