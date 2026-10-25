from datetime import datetime, timedelta, timezone

import pandas as pd

import nba_schedule
from db import connect, init_db


def _raw():
    def g(gid, d, utc, home, away, status='7:00 pm ET', label='', sub=''):
        return {'gameId': gid, 'gameDate': d, 'gameDateTimeUTC': utc, 'gameStatusText': status,
                'homeTeam_teamTricode': home, 'awayTeam_teamTricode': away, 'gameLabel': label,
                'gameSubLabel': sub}
    return pd.DataFrame([
        g('0012600001', '10/16/2026 00:00:00', '2026-10-16T23:00:00Z', 'TOR', 'MIA', label='Preseason'),
        g('0022600002', '10/20/2026 00:00:00', '2026-10-21T01:30:00Z', 'SAS', 'OKC', status='9:30 pm ET'),
        g('0022600001', '10/20/2026 00:00:00', '2026-10-20T23:00:00Z', 'NYK', 'PHI'),
        g('0022600003', '10/21/2026 00:00:00', '2026-10-21T23:30:00Z', 'MIA', 'MIN'),
        g('0042600101', '04/18/2027 00:00:00', '2027-04-18T17:00:00Z', 'BOS', 'NYK',
          label='East First Round', sub='Game 1'),
        g('0042600201', '05/02/2027 00:00:00', '2027-05-02T17:00:00Z', None, None),  # teams TBD
    ])


def _conn(tmp_path):
    conn = connect(tmp_path / 's.db')
    init_db(conn)
    return conn


def test_normalize_drops_preseason_and_tbd():
    df = nba_schedule.normalize_schedule(_raw(), 2026)
    assert list(df['game_id']) == ['0022600002', '0022600001', '0022600003', '0042600101']
    assert df.loc[3, 'game_date'] == '2027-04-18' and df.loc[3, 'label'] == 'East First Round · Game 1'
    assert df.loc[0, 'label'] == ''


def test_refresh_and_query(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', lambda season, timeout=30: _raw())
    assert nba_schedule.refresh_schedule(conn, 2026, log=lambda *_: None) == 4
    games = nba_schedule.games_on(conn, '2026-10-20')
    assert list(zip(games['away'], games['home'])) == [('PHI', 'NYK'), ('OKC', 'SAS')]  # tip-off order
    assert nba_schedule.games_on(conn, '2026-10-16').empty
    assert nba_schedule.next_game_date(conn, '2026-10-16') == '2026-10-20'
    assert nba_schedule.next_game_date(conn, '2027-04-18') is None
    # a refresh replaces the season, so a game that moved doesn't stay on its old date
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', lambda season, timeout=30: _raw().iloc[:3])
    nba_schedule.refresh_schedule(conn, 2026, log=lambda *_: None)
    assert nba_schedule.games_on(conn, '2026-10-21').empty


def test_ensure_schedule_refetches_only_when_stale(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    calls = []

    def fetch(season, timeout=30):
        calls.append(season)
        return _raw()
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', fetch)
    monkeypatch.setattr(nba_schedule, 'season_for_date', lambda d: 2026)   # "today" is in 2026-27
    assert nba_schedule.ensure_schedule(conn, '2026-10-20') is None and calls == [2026]
    nba_schedule.ensure_schedule(conn, '2026-10-21')
    assert calls == [2026]                                     # fresh: no refetch
    old = (datetime.now(timezone.utc) - timedelta(hours=nba_schedule.STALE_HOURS + 1)).isoformat()
    conn.execute("UPDATE schedule SET fetched_at=?", (old,))
    nba_schedule.ensure_schedule(conn, '2026-10-21', refresh_stale=False)
    assert calls == [2026]                                     # stale, but left to the background update
    nba_schedule.ensure_schedule(conn, '2026-10-21')
    assert calls == [2026, 2026]


def test_ensure_schedule_reports_fetch_errors(tmp_path, monkeypatch):
    conn = _conn(tmp_path)

    def fail(season, timeout=30):
        raise TimeoutError('read timed out')
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', fail)
    assert 'TimeoutError' in nba_schedule.ensure_schedule(conn, '2026-10-20')
    assert nba_schedule.games_on(conn, '2026-10-20').empty
    calls = []
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', lambda season, timeout=30: calls.append(season))
    assert 'TimeoutError' in nba_schedule.ensure_schedule(conn, '2026-10-20')   # backing off: no new fetch
    assert calls == []
