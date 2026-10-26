"""Drives the Streamlit app headlessly against a synthetic league database."""
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import db
import nba_schedule
import odds_fetcher
from fetch_game_logs import ingest_frames
from tests.conftest import make_league

APP = str(Path(__file__).resolve().parents[1] / 'app.py')
st_testing = pytest.importorskip('streamlit.testing.v1')


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    # The Odds API panels are off unless a key is configured; keep them off by default.
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setenv('NBA_INJURIES', '0')   # no live injury feed in tests
    monkeypatch.setattr(odds_fetcher, 'ENV_FILE', tmp_path / 'no.env')
    path = tmp_path / 'app.db'
    conn = db.connect(path)
    db.init_db(conn)
    for season, (p, t) in make_league().items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    monkeypatch.setattr(db, 'DB_PATH', path)
    # the schedule comes from the synthetic league instead of stats.nba.com
    # (own connection: the app runs in another thread)
    monkeypatch.setattr(nba_schedule, 'fetch_schedule', lambda season, timeout=30: fake_schedule(db.connect(path), season))
    import streamlit as st
    st.cache_data.clear()
    yield conn
    conn.close()


def fake_schedule(conn, season):
    """A ScheduleLeagueV2-shaped frame built from the league's home team-games."""
    rows = conn.execute("SELECT game_id, game_date, team, opponent FROM team_games WHERE season=? AND home=1",
                        (season,)).fetchall()
    return pd.DataFrame([{
        'gameId': gid, 'gameDate': f"{d[5:7]}/{d[8:10]}/{d[:4]} 00:00:00", 'gameDateTimeUTC': f"{d}T23:00:00Z",
        'gameStatusText': 'Final', 'homeTeam_teamTricode': h, 'awayTeam_teamTricode': a,
        'gameLabel': '', 'gameSubLabel': ''} for gid, d, h, a in rows],
        columns=['gameId', 'gameDate', 'gameDateTimeUTC', 'gameStatusText', 'homeTeam_teamTricode',
                 'awayTeam_teamTricode', 'gameLabel', 'gameSubLabel'])


def _w(widgets, label):
    return next(w for w in widgets if w.label == label)


def _bet_verdict(at, text):
    return any('class="verdict bet"' in m.value and text in m.value for m in at.markdown)


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
    assert _bet_verdict(at, 'Over 18.5')                         # ~25 ppg scorer vs 18.5

    at.date_input[1].set_value(date.fromisoformat(d))
    _w(at.selectbox, 'Away team').set_value(away)
    _w(at.selectbox, 'Home team').set_value(home)
    _w(at.button, 'Load matchup').click()
    at.run()
    _w(at.number_input, 'Home spread').set_value(-1.5)
    _w(at.number_input, 'Total points').set_value(220.5)
    _w(at.button, 'Evaluate game').click()
    at.run()
    assert not at.exception and not at.error
    assert any(f'{home} win chance' in m.value for m in at.markdown)
    assert app_db.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 1
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 2


def test_slate_opens_game(app_db):
    d, home, away = _game_date(app_db)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any('No games on' in c.value for c in at.caption)    # today isn't in the synthetic league
    at.date_input[1].set_value(date.fromisoformat(d))
    at.run()
    assert any(f'<span>{away}</span>' in m.value and f'<span>{home}</span>' in m.value for m in at.markdown)
    assert any(m.value.startswith('Model: **') for m in at.markdown)
    assert any(e.label.startswith('Why ') for e in at.expander)
    assert any('**Team strength**' in m.value for m in at.markdown)
    gid = app_db.execute("SELECT game_id FROM schedule WHERE game_date=? AND home=?", (d, home)).fetchone()[0]
    at.button(key=f'slate_{gid}').click()
    at.run()
    assert not at.exception and not at.error
    assert any(b.label == 'All games' for b in at.button)            # the opened game's section
    assert not any(b.label == 'Open' for b in at.button)             # ...replaces the list
    assert any(b.label == 'Evaluate game' for b in at.button)
    assert any(f'{home} win chance' in m.value for m in at.markdown)
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 0   # no lines, nothing logged
    _w(at.button, 'All games').click()
    at.run()
    assert not any(b.label == 'Evaluate game' for b in at.button)
    assert not any(f'{home} win chance' in m.value for m in at.markdown)
    assert any(b.label == 'Open' for b in at.button)


def test_track_pick_from_slate(app_db):
    d, home, away = _game_date(app_db)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.date_input[1].set_value(date.fromisoformat(d))
    at.run()
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 0   # nothing until you track
    gid = app_db.execute("SELECT game_id FROM schedule WHERE game_date=? AND home=?", (d, home)).fetchone()[0]
    at.button(key=f'slate_track_{gid}').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT market, tracked, home, away FROM game_log").fetchall() == \
        [('WINNER', 1, home, away)]
    assert at.session_state['main_tab'] == 'Tracked bets'           # jumps to the tracked list
    games = next(df.value for df in at.dataframe if 'Home logo' in df.value.columns)
    assert list(games['Type']) == ['Pick'] and games['Bet'][0].endswith(' to win')
    at.button(key=f'slate_untrack_{gid}').click()                   # the card now says Tracked
    at.run()
    assert app_db.execute("SELECT COUNT(*) FROM game_log").fetchone()[0] == 0


def test_add_to_tracked(app_db):
    d, home, away = _game_date(app_db)
    pid = app_db.execute("SELECT player_id FROM player_games WHERE team=? ORDER BY minutes DESC",
                         (home,)).fetchone()[0]
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.date_input(key='add_gdate').set_value(date.fromisoformat(d))
    at.run()
    at.radio(key='add_market').set_value('Total')
    at.run()
    at.button(key='add_game_go').click()
    at.run()
    assert any('Enter the line' in e.value for e in at.error)
    at.number_input(key='add_line').set_value(220.5)
    at.button(key='add_game_go').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT market, line, tracked FROM game_log").fetchall() == [('TOTAL', 220.5, 1)]

    at.segmented_control(key='add_kind').set_value('Prop')
    at.run()
    at.selectbox(key='add_player').set_value(pid)
    at.number_input(key='add_pline').set_value(18.5)
    at.date_input(key='add_pdate').set_value(date.fromisoformat(d))
    at.button(key='FormSubmitter:add_prop-Track').click()
    at.run()
    assert not at.exception
    assert app_db.execute("SELECT stat, line, tracked FROM prop_log").fetchall() == [('PTS', 18.5, 1)]
    props = next(df.value for df in at.dataframe if 'Player' in df.value.columns)
    assert f"{home} Player0" in list(props['Player'])


def test_slate_jumps_to_next_game_day(app_db):
    first = app_db.execute("SELECT MIN(game_date) FROM team_games WHERE season=2025").fetchone()[0]
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    at.date_input[1].set_value(date(2025, 10, 1))
    at.run()
    _w(at.button, 'Go to next game day').click()
    at.run()
    assert not at.exception
    assert at.date_input[1].value == date.fromisoformat(first)


def test_validation_messages(app_db):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    _w(at.button, 'Evaluate prop').click()
    at.run()
    msgs = [e.value for e in at.error]
    assert 'Pick a player.' in msgs and 'Enter a line.' in msgs
    _w(at.button, 'Load matchup').click()
    at.run()
    assert 'Pick both teams.' in [e.value for e in at.error]
    _w(at.selectbox, 'Away team').set_value('BOS')
    _w(at.selectbox, 'Home team').set_value('BOS')
    _w(at.button, 'Load matchup').click()
    at.run()
    assert 'Home and away must be different teams.' in [e.value for e in at.error]
    assert not any(b.label == 'Evaluate game' for b in at.button)    # nothing opened
    _w(at.selectbox, 'Home team').set_value('LAL')
    _w(at.button, 'Load matchup').click()
    at.run()
    _w(at.number_input, 'Home ML').set_value(-150)
    _w(at.button, 'Evaluate game').click()
    at.run()
    assert 'Enter both moneyline prices (or neither).' in [e.value for e in at.error]
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
    assert _bet_verdict(at, 'Over 18.5')                         # ~25 ppg scorer vs 18.5
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
    _w(at.button, 'Load matchup').click()
    at.run()
    box = at.checkbox(key=f"play_{home}_{star}_{d}")
    assert box.value is True                     # recent players start as playing
    box.uncheck()
    _w(at.button, 'Evaluate game').click()
    at.run()
    assert not at.exception and not at.error
    assert any('lineups -' in c.value for c in at.caption)
    # home star out: the explanation credits the lineup change to the away team
    assert any(m.value.startswith('**Why:**') and f'lineup changes favor {away}' in m.value and 'out for' in m.value
               for m in at.markdown)


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
    assert any('pill pos">Tracked' in m.value for m in at.markdown)
    assert at.session_state['main_tab'] == 'Tracked bets'
    props = next(df.value for df in at.dataframe if 'Player' in df.value.columns)   # Tracked bets tab
    assert list(props['Player']) == [f"{home} Player0"] and list(props['Result']) == ['Awaiting result']
    at.button(key='untrack_prop').click()
    at.run()
    assert app_db.execute("SELECT tracked FROM prop_log").fetchone() == (0,)


def test_auto_update_runs_in_background(app_db, monkeypatch):
    import background
    import daily_update
    calls, gate = [], threading.Event()

    def fake_update(conn, log=print, retries=4, quick=False):
        gate.wait(10)                                               # still running during the first render
        calls.append((retries, quick))
    monkeypatch.setenv('NBA_AUTO_UPDATE', '1')
    monkeypatch.setattr(daily_update, 'update', fake_update)
    monkeypatch.setattr(daily_update, 'update_due', lambda conn: "test")
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and calls == []                         # the page didn't wait for it
    assert any('in the background' in c.value for c in at.caption)
    gate.set()
    background.wait(10)
    assert calls == [(2, True)]
    at.run()
    assert not any('in the background' in c.value for c in at.caption)
    st_testing.AppTest.from_file(APP, default_timeout=120).run()    # new session within 30 min: no
    background.wait(10)
    assert calls == [(2, True)]

    db.set_meta(app_db, 'last_update_attempt', '2000-01-01T00:00:00+00:00')
    monkeypatch.setattr(daily_update, 'update_due', lambda conn: None)   # nothing new: no update
    st_testing.AppTest.from_file(APP, default_timeout=120).run()
    background.wait(10)
    assert calls == [(2, True)]

    def offline(conn, log=print, retries=4, quick=False):
        raise ConnectionError("offline")
    monkeypatch.setattr(daily_update, 'update_due', lambda conn: "test")
    monkeypatch.setattr(daily_update, 'update', offline)
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    background.wait(10)
    assert background.status()['error'] == "ConnectionError: offline"
    at.run()
    assert not at.exception                                         # falls back to saved data


def test_kalshi_tab(app_db):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any('No paper bets yet' in i.value for i in at.info)

    import kalshi_bot
    db.set_meta(app_db, kalshi_bot.BANKROLL_KEY, 1000)
    bets = [dict(ticker='KXNBAGAME-26JAN10LALBOS-BOS', market='WINNER', game_date='2026-01-10', home='BOS',
                 away='LAL', tip_utc='2026-01-11T00:30:00Z', team='BOS', line=None, side='yes', desc='BOS to win',
                 p_model=0.66, p_market=0.575, p_blend=0.6175, yes_bid=0.57, no_bid=0.42, ask=0.58, ev=0.02,
                 kelly=0.08, count=30, fee=0.52, cost=17.92),
            dict(ticker='KXNBATOTAL-26JAN10LALBOS-221', market='TOTAL', game_date='2026-01-10', home='BOS',
                 away='LAL', tip_utc='2026-01-11T00:30:00Z', team=None, line=220.5, side='no', desc='Under 220.5',
                 p_model=0.55, p_market=0.45, p_blend=0.50, yes_bid=0.54, no_bid=0.45, ask=0.46, ev=0.023,
                 kelly=0.05, count=20, fee=0.35, cost=9.55)]
    kalshi_bot.record_bets(app_db, bets)
    app_db.execute("INSERT INTO kalshi_snapshots (ticker, game_date, home, away, tip_utc, market, team, p_model, "
                   "first_seen, first_mid, last_seen, last_mid, decision) VALUES "
                   "('KXNBAGAME-26JAN10LALBOS-BOS', '2026-01-10', 'BOS', 'LAL', '2026-01-11T00:30:00Z', 'WINNER', "
                   "'BOS', 0.66, '2026-01-10T20:00:00+00:00', 0.575, '2026-01-11T00:15:00+00:00', 0.61, 'BET')")
    app_db.execute("UPDATE kalshi_orders SET result = 'WIN', payout = 30, profit = 12.08 WHERE market = 'WINNER'")
    app_db.commit()

    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    table = next(df.value for df in at.dataframe if 'CLV' in df.value.columns)
    assert list(table['Bet']) == ['BOS to win', 'Under 220.5']
    assert table['CLV'].iloc[0] == pytest.approx(3.0)            # closed at 61, paid 58
    assert list(table['Result']) == ['WIN', 'Pending']
    metrics = {m.label: m.value for m in at.metric}
    assert metrics['Paper bankroll'] == '$1,012' and metrics['Record (W-L)'] == '1-0'
    assert any('Last run' in m.value for m in at.markdown)
    at.segmented_control(key='kalshi_show').set_value('Pending')
    at.run()
    table = next(df.value for df in at.dataframe if 'CLV' in df.value.columns)
    assert list(table['Bet']) == ['Under 220.5']
