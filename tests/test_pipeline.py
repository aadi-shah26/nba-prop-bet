import csv

import generate_daily_bets as gdb


def test_daily_pipeline(league_conn, tmp_path, monkeypatch):
    d = [r[0] for r in league_conn.execute(
        "SELECT DISTINCT game_date FROM team_games WHERE season=2025 ORDER BY game_date")][40]
    home, away = league_conn.execute(
        "SELECT team, opponent FROM team_games WHERE game_date=? AND home=1", (d,)).fetchone()
    base = {'home': home, 'away': away, 'game_date': d, 'event_id': 'e1', 'market': 'player_points'}
    props = [
        {**base, 'book': 'bookA', 'player': f'{home} Player0', 'stat': 'PTS', 'line': 18.5,
         'over_odds': -110, 'under_odds': -110},
        {**base, 'book': 'bookB', 'player': f'{home} Player0', 'stat': 'PTS', 'line': 18.5,
         'over_odds': +105, 'under_odds': -125},   # better over price -> should be kept
        {**base, 'book': 'bookA', 'player': f'{away} Player1', 'stat': 'PTS', 'line': 9.5,
         'over_odds': -110, 'under_odds': -110},
        {**base, 'book': 'bookA', 'player': 'Nobody Atall', 'stat': 'PTS', 'line': 9.5,
         'over_odds': -110, 'under_odds': -110},
    ]
    monkeypatch.setattr(gdb, 'fetch_props', lambda *a, **k: props)
    monkeypatch.setattr(gdb, 'connect', lambda: league_conn)
    monkeypatch.setattr(gdb, 'OUT_DIR', tmp_path)
    rows = gdb.run(date=d)
    assert len(rows) == 2
    star = next(r for r in rows if r['player_name'] == f'{home} Player0')
    assert star['book'] == 'bookB' and star['pick'] == 'OVER'   # 25-ppg scorer vs 18.5
    assert star['opponent'] == away
    role = next(r for r in rows if r['player_name'] == f'{away} Player1')
    assert role['opponent'] == home
    with open(tmp_path / f'props_{d}.csv') as f:
        assert len(list(csv.DictReader(f))) == 2
    assert league_conn.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0] == 2
