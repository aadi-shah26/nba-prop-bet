import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import connect, init_db  # noqa: E402
from fetch_game_logs import ingest_frames  # noqa: E402

TEAMS = ['BOS', 'LAL', 'GSW', 'DEN', 'MIA', 'NYK']
TEAM_IDS = {t: 1610612700 + i for i, t in enumerate(TEAMS)}


def make_league(seasons=(2024, 2025), games_per_pair=4, seed=0, team_strength=None):
    """
    Synthetic LeagueGameLog-shaped frames (player 'P' and team 'T') for a 6-team league.
    Each team has 2 players: a scorer (~25 pts in 34 min) and a role player (~10 in 24).
    team_strength: {team: s} -> expected score 110 + s_team - s_opp.
    """
    rng = np.random.default_rng(seed)
    team_strength = team_strength or {}
    out = {}
    gid = 0
    for season in seasons:
        d0 = date(season, 10, 22)
        prows, trows = [], []
        day = 0
        for rep in range(games_per_pair):
            for i, h in enumerate(TEAMS):
                for a in TEAMS[i + 1:]:
                    home, away = (h, a) if rep % 2 == 0 else (a, h)
                    gid += 1
                    game_id = f"002{season % 100:02d}{gid:05d}"
                    gdate = (d0 + timedelta(days=day)).isoformat()
                    day += 1
                    team_box = {}
                    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                        mu = 110 + team_strength.get(team, 0) - team_strength.get(opp, 0)
                        pts = int(rng.normal(mu, 10))
                        team_box[team] = dict(PTS=pts, REB=int(rng.normal(44, 4)), AST=int(rng.normal(25, 3)),
                                              STL=7, BLK=5, TOV=13, FG3M=12, FG3A=34, FGM=40, FGA=88,
                                              FTM=18, FTA=23, OREB=10, DREB=34, PF=19)
                        matchup = f"{team} vs. {opp}" if is_home else f"{team} @ {opp}"
                        for pid_off, (m, pts_mu) in enumerate(((34, 25), (24, 10))):
                            pid = TEAM_IDS[team] * 10 + pid_off
                            prows.append(dict(PLAYER_ID=pid, PLAYER_NAME=f"{team} Player{pid_off}",
                                              TEAM_ID=TEAM_IDS[team], TEAM_ABBREVIATION=team, GAME_ID=game_id,
                                              GAME_DATE=gdate, MATCHUP=matchup, WL=None, MIN=m,
                                              PTS=int(rng.poisson(pts_mu)), REB=int(rng.poisson(5)),
                                              AST=int(rng.poisson(4)), STL=1, BLK=0, TOV=2, FG3M=2,
                                              FGM=8, FGA=17, FTM=4, FTA=5, OREB=1, DREB=4, PF=2,
                                              PLUS_MINUS=0))
                    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                        b = team_box[team]
                        wl = 'W' if b['PTS'] > team_box[opp]['PTS'] else 'L'
                        trows.append(dict(TEAM_ID=TEAM_IDS[team], TEAM_ABBREVIATION=team, TEAM_NAME=team,
                                          GAME_ID=game_id, GAME_DATE=gdate,
                                          MATCHUP=f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                                          WL=wl, MIN=240, PLUS_MINUS=b['PTS'] - team_box[opp]['PTS'], **b))
        out[season] = (pd.DataFrame(prows), pd.DataFrame(trows))
    return out


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / 'test.db')
    init_db(c)
    yield c
    c.close()


@pytest.fixture
def league_conn(conn):
    for season, (p, t) in make_league().items():
        ingest_frames(conn, season, 'Regular Season', p, t)
    return conn
