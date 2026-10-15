import pandas as pd

from db import connect, init_db
from fetch_game_logs import PLAYER_COLS, TEAM_COLS, upsert
from import_espn_boxscores import normalize_players, normalize_teams


def _player(gid, aid, team, opp, ha, mins, pts=10, season_type=2):
    return {'game_id': gid, 'season_type': season_type, 'game_date': pd.Timestamp('2025-11-01'),
            'athlete_id': aid, 'athlete_display_name': f'P{aid}', 'team_id': 1 if team == 'NY' else 2,
            'team_abbreviation': team, 'opponent_team_abbreviation': opp, 'home_away': ha,
            'team_winner': ha == 'home', 'minutes': mins, 'points': pts, 'rebounds': 5, 'assists': 3,
            'steals': 1, 'blocks': 0, 'turnovers': 2, 'three_point_field_goals_made': 1,
            'field_goals_made': 4, 'field_goals_attempted': 9, 'free_throws_made': 1,
            'free_throws_attempted': 2, 'offensive_rebounds': 1, 'defensive_rebounds': 4,
            'fouls': 2, 'plus_minus': '+3'}


def _team(gid, team, opp, ha, score, opp_score, season_type=2):
    return {'game_id': gid, 'season_type': season_type, 'game_date': pd.Timestamp('2025-11-01'),
            'team_id': 1 if team == 'NY' else 2, 'team_abbreviation': team,
            'opponent_team_abbreviation': opp, 'team_display_name': team, 'team_home_away': ha,
            'team_winner': score > opp_score, 'team_score': score, 'opponent_team_score': opp_score,
            'total_rebounds': 45, 'total_turnovers': 14, 'turnovers': 12,
            'three_point_field_goals_attempted': 35, 'assists': 25, 'steals': 7, 'blocks': 5,
            'three_point_field_goals_made': 12, 'field_goals_made': 40, 'field_goals_attempted': 88,
            'free_throws_made': 15, 'free_throws_attempted': 20, 'offensive_rebounds': 10,
            'defensive_rebounds': 35, 'fouls': 20}


def test_espn_frames_map_to_schema(tmp_path):
    p_raw = pd.DataFrame([
        _player(401, 11, 'NY', 'SA', 'home', 34.0, pts=30),
        _player(401, 12, 'NY', 'SA', 'home', None),           # DNP -> dropped
        _player(401, 21, 'SA', 'NY', 'away', 30.0),
        _player(402, 99, 'STARS', 'WORLD', 'home', 20.0, season_type=4),  # all-star -> dropped
    ])
    t_raw = pd.DataFrame([_team(401, 'NY', 'SA', 'home', 110, 100), _team(401, 'SA', 'NY', 'away', 100, 110)])
    players = normalize_players(p_raw, 2025)
    teams, neutral = normalize_teams(t_raw, 2025, players)

    assert set(players['player_id']) == {11, 21}
    assert set(players['team']) == {'NYK', 'SAS'}
    assert players.set_index('player_id').loc[11, 'pts'] == 30
    assert players['season_type'].eq('Regular Season').all()
    assert len(neutral) == 0
    t = teams.set_index('team')
    assert t.loc['NYK', 'home'] == 1 and t.loc['SAS', 'home'] == 0
    assert t.loc['NYK', 'tov'] == 14 and t.loc['NYK', 'pts'] == 110 and t.loc['NYK', 'plus_minus'] == 10

    conn = connect(tmp_path / 't.db')
    init_db(conn)
    assert upsert(conn, 'player_games', players, PLAYER_COLS) == 2
    assert upsert(conn, 'team_games', teams, TEAM_COLS) == 2
