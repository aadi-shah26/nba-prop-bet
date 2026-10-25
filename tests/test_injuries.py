import pandas as pd

import nba_injuries


def _raw():
    def p(name, status, comment=''):
        return {'athlete': {'displayName': name}, 'status': status, 'shortComment': comment,
                'details': {'returnDate': '2026-11-20'}}
    return {'injuries': [
        {'displayName': 'Boston Celtics', 'injuries': [p('Jayson Tatum', 'Out', 'Achilles'),
                                                       p('Derrick White', 'Day-To-Day', 'Ankle')]},
        {'displayName': 'Nowhere Nobodies', 'injuries': [p('Some One', 'Out')]},   # unknown team: skipped
    ]}


def test_normalize_and_match():
    df = nba_injuries.normalize_injuries(_raw())
    assert list(df['team']) == ['BOS', 'BOS'] and list(df['status']) == ['Out', 'Day-To-Day']
    roster = pd.DataFrame({'player_id': [1, 2, 3], 'player_name': ['Jayson Tatum', 'Derrick White', 'Jrue Holiday']})
    assert nba_injuries.match_roster(df, roster, 'BOS') == {1: ('Out', 'Achilles'), 2: ('Day-To-Day', 'Ankle')}
    assert nba_injuries.match_roster(df, roster, 'NYK') == {}


def test_empty_feed():
    assert nba_injuries.normalize_injuries({}).empty
