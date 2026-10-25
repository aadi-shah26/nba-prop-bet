"""Drives the Streamlit app headlessly against a synthetic league database."""
from datetime import date
from pathlib import Path

import pytest

import db
import odds_fetcher
from fetch_game_logs import ingest_frames
from tests.conftest import make_league

APP = str(Path(__file__).resolve().parents[1] / 'app.py')
st_testing = pytest.importorskip('streamlit.testing.v1')


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    # The Odds API panels are off unless a key is configured; keep them off by default.
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr(odds_fetcher, 'ENV_FILE', tmp_path / 'no.env')
    path = tmp_path / 'app.db'
    conn = db.connect(path)
    db.init_db(conn)
    for season, (p, t) in make_league().items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    monkeypatch.setattr(db, 'DB_PATH', path)
    yield conn
    conn.close()


def _w(widgets, label):
    return next(w for w in widgets if w.label == label)


def _game_date(conn):
    d = [r[0] for r in conn.execute("SELECT DISTINCT game_date FROM team_games WHERE season = 2025 "
                                    "ORDER BY game_date")][40]
    home, away = conn.execute("SELECT team, opponent FROM team_games WHERE game_date=? AND home=1",
                              (d,)).fetchone()
    return d, home, away


def test_prop_and_game_flow(app_db):
    d, home, away = _game_date(app_db)
    pid = app_db.execute("SELECT player_id FROM player_games WHERE team=? ORDER BY minutes DESC",
                         (home,)).fetchone()[0]
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    _w(at.selectbox, 'Player').set_value(pid)
    _w(at.selectbox, 'Stat').set_value('PTS')
    _w(at.number_input, 'Line').set_value(18.5)
    _w(at.selectbox, 'Opponent').set_value(away)
    at.date_input[0].set_value(date.fromisoformat(d))
    _w(at.button, 'Evaluate prop').click()
    at.run()
    assert not at.exception and not at.error
    assert any('BET OVER' in s.value for s in at.success)       # ~25 ppg scorer vs 18.5

    _w(at.selectbox, 'Away team').set_value(away)
    _w(at.selectbox, 'Home team').set_value(home)
    at.date_input[1].set_value(date.fromisoformat(d))
    _w(at.number_input, 'Home spread').set_value(-1.5)
    _w(at.number_input, 'Total points').set_value(220.5)
    _w(at.button, 'Evaluate game').click()
    at.run()
    assert not at.exception and not at.error
    assert any(f'{home} win chance' == m.label for m in at.metric)
    assert app_db.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 1
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 2


def test_validation_messages(app_db):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    _w(at.button, 'Evaluate prop').click()
    at.run()
    msgs = [e.value for e in at.error]
    assert 'Pick a player.' in msgs and 'Enter a line.' in msgs
    _w(at.selectbox, 'Away team').set_value('BOS')
    _w(at.selectbox, 'Home team').set_value('BOS')
    _w(at.number_input, 'Home ML').set_value(-150)
    _w(at.button, 'Evaluate game').click()
    at.run()
    msgs = [e.value for e in at.error]
    assert 'Home and away must be different teams.' in msgs
    assert 'Enter both moneyline prices (or neither).' in msgs
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 0


def test_empty_database_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'empty.db')
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any('Update data' in i.value for i in at.info)


def test_manual_settle_and_delete_in_app(app_db):
    from bet_signals import evaluate_prop, log_prop
    from projection_model import PropContext
    d, home, away = _game_date(app_db)
    pid = app_db.execute("SELECT player_id FROM player_games WHERE team=? ORDER BY minutes DESC",
                         (home,)).fetchone()[0]
    r = evaluate_prop(PropContext(app_db, d), pid, 'PTS', 20.5, -110, -110, min_ev=-1.0)
    log_prop(app_db, r, tracked=True)
    row_id = app_db.execute("SELECT id FROM prop_log").fetchone()[0]

    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    _w(at.number_input, 'Actual stat').set_value(40.0)
    _w(at.button, 'Save').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT actual, result FROM prop_log").fetchone() == \
        (40.0, 'WIN' if r['pick'] == 'OVER' else 'LOSS')
    assert any('Saved' in s.value for s in at.success)

    # settled entries are hidden unless the box is ticked; then delete needs confirmation
    _w(at.checkbox, 'Include already-settled entries (to correct or delete them)').check()
    at.run()
    _w(at.radio, 'Action').set_value('Delete entry')
    _w(at.button, 'Save').click()
    at.run()
    assert any('confirm' in e.value for e in at.error)
    assert app_db.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 1
    _w(at.radio, 'Action').set_value('Delete entry')
    _w(at.checkbox, 'Yes, delete it (only needed for Delete entry)').check()
    _w(at.button, 'Save').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT COUNT(*) FROM prop_log WHERE id=?", (row_id,)).fetchone()[0] == 0


def test_odds_api_panels_hidden_without_key(app_db):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert not any('Odds API' in e.label for e in at.expander)


def test_odds_api_import_flow(app_db, monkeypatch):
    d, home, away = _game_date(app_db)
    names = {t: f"{t} Name" for t in (home, away)}
    monkeypatch.setenv('ODDS_API_KEY', 'test')
    monkeypatch.setattr(odds_fetcher, 'team_abbr', lambda n: {v: k for k, v in names.items()}.get(n, n))
    event = {'id': 'ev1', 'commence_time': f'{d}T23:00:00Z', 'home_team': names[home], 'away_team': names[away]}
    outcomes = [{'name': 'Over', 'description': f'{home} Player0', 'price': -110, 'point': 18.5},
                {'name': 'Under', 'description': f'{home} Player0', 'price': -110, 'point': 18.5}]
    calls = []

    def fake_get(path, **params):
        calls.append(path)
        usage = {'last': '1', 'used': '1', 'remaining': '499'}
        if path.endswith('/events'):
            return [event], usage
        if '/events/ev1/odds' in path:
            return {**event, 'bookmakers': [{'key': 'draftkings', 'markets': [
                {'key': 'player_points', 'outcomes': outcomes}]}]}, usage
        return [{**event, 'bookmakers': [{'key': 'draftkings', 'markets': [
            {'key': 'spreads', 'outcomes': [{'name': names[home], 'price': -110, 'point': -1.5},
                                            {'name': names[away], 'price': -110, 'point': 1.5}]},
            {'key': 'totals', 'outcomes': [{'name': 'Over', 'price': -110, 'point': 220.5},
                                           {'name': 'Under', 'price': -110, 'point': 220.5}]}]}]}], usage

    monkeypatch.setattr(odds_fetcher, '_get', fake_get)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    assert calls and calls[0].endswith('/events')        # today's games load by themselves
    at.date_input(key='oa_pdate').set_value(date.fromisoformat(d))
    at.run()
    _w(at.button, 'Load props (1 credit)').click()
    at.run()
    _w(at.button, 'Evaluate imported prop').click()
    at.run()
    assert not at.exception and not at.error
    assert any('BET OVER' in s.value for s in at.success)       # ~25 ppg scorer vs 18.5
    assert app_db.execute("SELECT book, opponent, line FROM prop_log").fetchone() == ('draftkings', away, 18.5)

    at.date_input(key='oa_gdate').set_value(date.fromisoformat(d))
    _w(at.button, 'Load lines (3 credits)').click()
    at.run()
    _w(at.button, 'Evaluate imported game').click()
    at.run()
    assert not at.exception and not at.error
    assert app_db.execute("SELECT market, line, book FROM game_log ORDER BY market").fetchall() == \
        [('SPREAD', -1.5, 'draftkings'), ('TOTAL', 220.5, 'draftkings')]
    assert any('remaining 499' in c.value for c in at.caption)


def test_game_lineup_checklist(app_db):
    d, home, away = _game_date(app_db)
    star = app_db.execute("SELECT player_id FROM player_games WHERE team=? ORDER BY minutes DESC",
                          (home,)).fetchone()[0]
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.date_input[1].set_value(date.fromisoformat(d))
    _w(at.selectbox, 'Away team').set_value(away)
    _w(at.selectbox, 'Home team').set_value(home)
    at.run()
    box = at.checkbox(key=f"play_{home}_{star}_{d}")
    assert box.value is True                     # recent players start as playing
    box.uncheck()
    _w(at.button, 'Evaluate game').click()
    at.run()
    assert not at.exception and not at.error
    assert any('lineups -' in c.value for c in at.caption)


def test_track_this_and_tracked_tab(app_db):
    d, home, away = _game_date(app_db)
    pid = app_db.execute("SELECT player_id FROM player_games WHERE team=? ORDER BY minutes DESC",
                         (home,)).fetchone()[0]
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    _w(at.selectbox, 'Player').set_value(pid)
    _w(at.number_input, 'Line').set_value(18.5)
    at.date_input[0].set_value(date.fromisoformat(d))
    _w(at.button, 'Evaluate prop').click()
    at.run()
    assert app_db.execute("SELECT tracked FROM prop_log").fetchone() == (0,)   # saved, not tracked
    at.button(key='track_prop').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT tracked FROM prop_log").fetchone() == (1,)
    assert any('Tracked' in s.value for s in at.success)
    table = next(t.value for t in at.dataframe if 'Player' in t.value)   # Tracked bets tab
    assert list(table['Player']) == [f"{home} Player0"] and table['Result'][0] == '⌛ Awaiting result'
    at.button(key='untrack_prop').click()
    at.run()
    assert app_db.execute("SELECT tracked FROM prop_log").fetchone() == (0,)


def test_auto_update_on_open(app_db, monkeypatch):
    import daily_update
    calls = []

    def fake_update(conn, log=print, retries=4):
        calls.append(retries)
    monkeypatch.setenv('NBA_AUTO_UPDATE', '1')
    monkeypatch.setattr(daily_update, 'update', fake_update)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and calls == [2]
    at.run()                                                        # reruns in the same session: no
    st_testing.AppTest.from_file(APP, default_timeout=120).run()    # new session within 30 min: no
    assert calls == [2]
    db.set_meta(app_db, 'last_update_attempt', '2000-01-01T00:00:00+00:00')

    def offline(conn, log=print, retries=4):
        raise ConnectionError("offline")
    monkeypatch.setattr(daily_update, 'update', offline)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception                                         # falls back to saved data
    assert any('offline' in e.value for e in at.error)
