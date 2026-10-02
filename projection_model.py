#!/usr/bin/env python3
"""
Player stat projections -> full probability distribution -> P(over/under/push) for any line.

Stats are counts, so the outcome is modelled as a negative binomial with the projected
mean and the player's own variance (see feature_engineering.py). That handles low-count
props (assists 4.5, threes 2.5, blocks 0.5) far better than a normal curve, and gives
exact push probabilities for whole-number lines.

Usage:
    python projection_model.py "shai" PTS                    # projection + fair line + ladder
    python projection_model.py "shai" PTS --line 30.5 --opp LAL
    python projection_model.py "jokic" PRA --opp MIN --minutes 30 --date 2026-11-01
"""

import argparse
import math

import pandas as pd
from scipy import stats as sps

from db import connect, find_players, season_for_date, team_abbr, to_date
from feature_engineering import (PropParams, STATS, league_priors, opponent_factors,
                                 parse_stat, project_from_history, team_games_with_allowed)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def count_distribution(mean, var):
    """Negative binomial with the given mean/variance (Poisson if var <= mean)."""
    if var <= mean:
        return sps.poisson(mean)
    n = mean ** 2 / (var - mean)
    return sps.nbinom(n, n / (n + mean))


def line_probs(mean, var, line):
    """(p_over, p_under, p_push) for an integer-valued stat against `line`."""
    if mean <= 0:
        return (0.0, 0.0, 1.0) if line == 0 else (0.0, 1.0, 0.0) if line > 0 else (1.0, 0.0, 0.0)
    dist = count_distribution(mean, var)
    fl = math.floor(line)
    p_over = float(dist.sf(fl))                    # P(X >= floor(line)+1) == P(X > line)
    if float(line).is_integer():
        p_push = float(dist.pmf(int(line)))
        p_under = float(dist.cdf(int(line) - 1))
    else:
        p_push = 0.0
        p_under = float(dist.cdf(fl))
    return p_over, p_under, p_push


def fair_line(mean, var):
    """The x.5 line closest to a 50/50 split (what the 'right' book line would be)."""
    if mean <= 0:
        return 0.5
    m = int(count_distribution(mean, var).ppf(0.5))
    cands = [c for c in (m - 0.5, m + 0.5) if c > 0]
    return min(cands, key=lambda c: abs(line_probs(mean, var, c)[0] - 0.5))


# ---------------------------------------------------------------------------
# Context: loads data once, projects many props
# ---------------------------------------------------------------------------

class PropContext:
    """
    Everything needed to project props for games on `game_date`, using only data from
    before that date. Build once per run; call .project() for each prop.
    """

    def __init__(self, conn, game_date=None, params=None):
        self.conn = conn
        self.params = params or PropParams()
        self.game_date = to_date(game_date).isoformat()
        self.season = season_for_date(self.game_date)
        lo = self.season - self.params.lookback_seasons + 1
        self.player_games = pd.read_sql_query(
            "SELECT * FROM player_games WHERE season >= ? AND game_date < ? "
            "ORDER BY game_date, game_id", conn, params=(lo, self.game_date))
        if self.player_games.empty:
            raise RuntimeError("No player games in the database before "
                               f"{self.game_date}. Run: python fetch_game_logs.py")
        team_games = pd.read_sql_query(
            "SELECT * FROM team_games WHERE season >= ? AND game_date < ?",
            conn, params=(lo, self.game_date))
        self.priors = league_priors(self.player_games)
        self.tga = team_games_with_allowed(team_games)
        self.opp = opponent_factors(self.tga, self.game_date, self.season, self.params)
        self._by_player = {pid: g for pid, g in self.player_games.groupby('player_id', sort=False)}
        self.last_data_date = self.player_games['game_date'].max()

    def player_history(self, player_id):
        return self._by_player.get(int(player_id))

    def project(self, player_id, stat, opponent=None, minutes=None):
        hist = self.player_history(player_id)
        if hist is None:
            return None
        opp = None
        if opponent:
            abbr = team_abbr(opponent)
            if abbr is None:
                raise ValueError(f"Unknown opponent {opponent!r}")
            opp = self.opp.get(abbr)
        res = project_from_history(hist, stat, self.season, self.priors, self.params,
                                   opp_factor=opp, minutes=minutes)
        if res is not None:
            last = hist.iloc[-1]
            res.update(player_id=int(player_id), player_name=last['player_name'],
                       team=last['team'], last_game=last['game_date'],
                       opponent=team_abbr(opponent) if opponent else None,
                       fair_line=fair_line(res['mean'], res['var']))
        return res


def resolve_player(conn, query, teams=None):
    """Returns (player_id, player_name, team) or raises LookupError listing candidates."""
    cands = find_players(conn, query, teams=teams)
    if cands.empty:
        raise LookupError(f"No player matches {query!r}")
    if len(cands) > 1 and cands['norm'].nunique() > 1:
        names = ', '.join(f"{r.player_name} ({r.team})" for r in cands.head(8).itertuples())
        raise LookupError(f"{query!r} is ambiguous: {names}")
    r = cands.iloc[0]
    return int(r['player_id']), r['player_name'], r['team']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_projection(p, line=None):
    stat = p['stat']
    print(f"\n{'=' * 64}")
    print(f"📊 {p['player_name']} ({p['team']}) — {stat}"
          + (f" vs {p['opponent']}" if p['opponent'] else ''))
    print(f"{'=' * 64}")
    print(f"Projection (mean):  {p['mean']:.2f}   sd {p['sd']:.2f}")
    print(f"Fair line:          {p['fair_line']}")
    print(f"Minutes:            {p['minutes_proj']:.1f}"
          + ('' if abs(p['minutes_proj'] - p['minutes_hist']) < 1e-9
             else f"  (override; recent avg {p['minutes_hist']:.1f})"))
    opp = ', '.join(f"{c} x{f:.3f}" for c, f in p['opp_factors'].items())
    print(f"Opponent factors:   {opp}")
    sa = f"{p['season_avg']:.2f} ({p['season_games']} g)" if p['season_avg'] is not None else 'n/a'
    print(f"Season avg:         {sa}   last10 {p['last10_avg']:.2f}   last5 {p['last5_avg']:.2f}")
    print(f"Data through:       {p['last_game']}  ({p['games_used']} games used)")
    lines = [line] if line is not None else \
        [p['fair_line'] + d for d in (-3, -2, -1, 0, 1, 2, 3) if p['fair_line'] + d > 0]
    print(f"\n{'Line':>7} {'P(over)':>9} {'P(under)':>9} {'P(push)':>8}")
    for L in lines:
        o, u, ps = line_probs(p['mean'], p['var'], L)
        print(f"{L:>7} {o:>9.1%} {u:>9.1%} {ps:>8.1%}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('player')
    ap.add_argument('stat', help=', '.join(STATS))
    ap.add_argument('--line', type=float)
    ap.add_argument('--opp', help='opponent team (abbr or name)')
    ap.add_argument('--minutes', type=float, help='override projected minutes')
    ap.add_argument('--date', help='game date YYYY-MM-DD (default today)')
    a = ap.parse_args()

    conn = connect()
    pid, name, team = resolve_player(conn, a.player)
    ctx = PropContext(conn, a.date)
    p = ctx.project(pid, parse_stat(a.stat), opponent=a.opp, minutes=a.minutes)
    if p is None:
        print(f"❌ Not enough recent games for {name}")
        return
    print_projection(p, a.line)


if __name__ == '__main__':
    main()
