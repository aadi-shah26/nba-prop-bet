from datetime import date, datetime, timedelta, timezone

import daily_update
from daily_update import finished_games_missing, season_types_due, update, update_due


def _schedule(conn, game_date, tip, status='7:30 pm ET', gid='1'):
    conn.execute("INSERT INTO schedule (game_id, game_date, season, tip_utc, status, home, away, label, fetched_at) "
                 "VALUES (?,?,?,?,?,'BOS','LAL','',?)", (gid, game_date, 2026, tip, status,
                                                         datetime.now(timezone.utc).isoformat()))


def test_season_types_due():
    assert season_types_due(2026, date(2026, 10, 25)) == ('Regular Season',)
    assert season_types_due(2026, date(2027, 4, 2)) == ('Regular Season', 'PlayIn', 'Playoffs')
    assert season_types_due(2024, date(2026, 10, 25)) == ('Regular Season', 'PlayIn', 'Playoffs')


def test_finished_games_missing(league_conn):
    last = league_conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    after = (date.fromisoformat(last) + timedelta(days=1)).isoformat()
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _schedule(league_conn, last, f'{last}T23:00:00Z', 'Final', 'a')            # already loaded
    _schedule(league_conn, after, f'{after}T23:00:00Z', gid='b')               # finished, not loaded
    _schedule(league_conn, after, f'{after}T23:00:00Z', 'PPD', 'c')            # postponed: never comes
    assert finished_games_missing(league_conn, now) == 1
    just_tipped = datetime.fromisoformat(f'{after}T23:30:00+00:00')
    assert finished_games_missing(league_conn, just_tipped) == 0               # still being played


def test_update_due(league_conn, monkeypatch):
    monkeypatch.setattr(daily_update, 'season_for_date', lambda d: 2026)
    assert update_due(league_conn) == "schedule is out of date"
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    _schedule(league_conn, tomorrow[:10], tomorrow)
    assert update_due(league_conn) == "rosters are out of date"
    league_conn.execute("INSERT INTO rosters (season, team, player_id, player_name, fetched_at) "
                        "VALUES (2026, 'BOS', 1, 'X', ?)", (datetime.now(timezone.utc).isoformat(),))
    assert update_due(league_conn) is None                                     # nothing can have changed
    _schedule(league_conn, '2099-01-01', '2000-01-01T00:00:00Z', gid='late')   # a finished game to load
    assert update_due(league_conn) == "1 finished game to load"


def test_quick_update_skips_what_isnt_due(league_conn, monkeypatch):
    calls = []
    monkeypatch.setattr(daily_update, 'ingest_season', lambda *a, **k: calls.append('games'))
    monkeypatch.setattr(daily_update, 'refresh_schedule', lambda *a, **k: calls.append('schedule'))
    monkeypatch.setattr(daily_update, 'refresh_rosters', lambda *a, **k: calls.append('rosters'))
    monkeypatch.setattr(daily_update, 'season_for_date', lambda d: 2026)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    _schedule(league_conn, tomorrow[:10], tomorrow)
    league_conn.execute("INSERT INTO rosters (season, team, player_id, player_name, fetched_at) "
                        "VALUES (2026, 'BOS', 1, 'X', ?)", (datetime.now(timezone.utc).isoformat(),))
    update(league_conn, log=lambda *a: None, quick=True)
    assert calls == []                                                         # no requests at all
    update(league_conn, log=lambda *a: None)
    assert calls == ['games', 'schedule', 'rosters']                           # the button does everything
