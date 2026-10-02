"""Drives the Streamlit app headlessly against a synthetic league database."""
from datetime import date
from pathlib import Path

import pytest

import db
from fetch_game_logs import ingest_frames
from tests.conftest import make_league

APP = str(Path(__file__).resolve().parents[1] / 'app.py')
st_testing = pytest.importorskip('streamlit.testing.v1')


@pytest.fixture
def app_db(tmp_path, monkeypatch):
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
    log_prop(app_db, r)
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
