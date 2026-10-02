import pandas as pd
import pytest

from db import find_players, normalize_name, season_for_date, season_str, team_abbr
from fetch_game_logs import ingest_frames, normalize_team_frame, parse_matchup
from tests.conftest import make_league


def test_parse_matchup():
    assert parse_matchup('LAL vs. HOU') == ('LAL', 'HOU', 1)
    assert parse_matchup('LAL @ DAL') == ('LAL', 'DAL', 0)
    with pytest.raises(ValueError):
        parse_matchup('LAL - DAL')


def test_seasons():
    assert season_str(2025) == '2025-26'
    assert season_for_date('2026-10-02') == 2026
    assert season_for_date('2026-04-15') == 2025


def test_names_and_teams():
    assert normalize_name('Nikola Jokić') == 'nikola jokic'
    assert normalize_name("De'Aaron Fox") == 'deaaron fox'
    assert normalize_name('Jaren Jackson Jr.') == 'jaren jackson'
    assert normalize_name('P.J. Washington') == 'pj washington'
    assert team_abbr('Los Angeles Lakers') == 'LAL'
    assert team_abbr('LA Clippers') == 'LAC'
    assert team_abbr('celtics') == 'BOS'
    assert team_abbr('nope') is None


def test_ingest_iso_dates_and_counts(conn):
    lg = make_league(seasons=(2025,))
    p, t = lg[2025]
    # stats.nba.com also sometimes sends 'MMM DD, YYYY'; both must become ISO
    p = p.copy()
    p.loc[0, 'GAME_DATE'] = pd.Timestamp(p.loc[0, 'GAME_DATE']).strftime('%b %d, %Y').upper()
    n_p, n_t = ingest_frames(conn, 2025, 'Regular Season', p, t)
    assert n_p == len(p) and n_t == len(t)
    dates = [r[0] for r in conn.execute('SELECT game_date FROM player_games')]
    assert all(len(d) == 10 and d[4] == '-' for d in dates)
    # chronological ordering really is chronological
    ordered = [r[0] for r in conn.execute('SELECT game_date FROM team_games ORDER BY game_date')]
    assert ordered == sorted(ordered, key=pd.Timestamp)
    # re-ingesting is idempotent
    ingest_frames(conn, 2025, 'Regular Season', p, t)
    assert conn.execute('SELECT COUNT(*) FROM player_games').fetchone()[0] == len(p)


def test_ingest_rejects_broken_game(conn):
    _, t = make_league(seasons=(2025,))[2025]
    broken = t.iloc[1:]  # first game now has only one team row
    with pytest.raises(ValueError):
        ingest_frames(conn, 2025, 'Regular Season', pd.DataFrame(), broken)


def test_duplicates_dropped():
    _, t = make_league(seasons=(2025,))[2025]
    doubled = pd.concat([t, t.iloc[:4]])
    assert len(normalize_team_frame(doubled, 2025, 'Regular Season')) == len(t)


def test_find_players(league_conn):
    assert find_players(league_conn, 'bos player0')['player_name'].tolist() == ['BOS Player0']
    assert len(find_players(league_conn, 'player0')) == 6
    assert len(find_players(league_conn, 'player0', teams=['LAL'])) == 1
