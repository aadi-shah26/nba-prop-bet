"""
Current NBA injury report from ESPN's public feed. The app uses it to start players listed as
Out unticked in each game's playing/out checklist, and to apply the same lineup adjustment to
the predictions on the Games tab, so you don't have to look up the injury report yourself.
It is only used for today's and upcoming games: it says who is hurt now, not on past dates.

    python nba_injuries.py              # print every team's report
    python nba_injuries.py --team BOS
"""

import argparse

import pandas as pd

from db import normalize_name, team_abbr

URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'
OUT = 'Out'
COLS = ['team', 'player_name', 'status', 'comment', 'return_date']


def fetch_injuries(timeout=10):
    """Raw ESPN injuries JSON."""
    import requests
    r = requests.get(URL, timeout=timeout)
    r.raise_for_status()
    return r.json()


def normalize_injuries(raw):
    """ESPN JSON -> COLS, one row per listed player. Teams ESPN names differently are skipped."""
    rows = []
    for t in raw.get('injuries', []):
        team = team_abbr(t.get('displayName', ''))
        if not team:
            continue
        for i in t.get('injuries', []):
            rows.append((team, i.get('athlete', {}).get('displayName', ''), i.get('status', ''),
                         i.get('shortComment', ''), (i.get('details') or {}).get('returnDate')))
    return pd.DataFrame(rows, columns=COLS)


def match_roster(injuries, roster, team):
    """{player_id: (status, comment)} for `team`'s roster players on the report (by name)."""
    rep = injuries[injuries['team'] == team]
    by_name = {normalize_name(r.player_name): (r.status, r.comment) for r in rep.itertuples()}
    return {int(p.player_id): by_name[n] for p in roster.itertuples()
            if (n := normalize_name(p.player_name)) in by_name}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', help='team abbreviation, e.g. BOS')
    a = ap.parse_args()
    df = normalize_injuries(fetch_injuries())
    if a.team:
        df = df[df['team'] == a.team.upper()]
    for team, g in df.groupby('team'):
        print(team)
        for r in g.itertuples():
            print(f"  {r.player_name:<26} {r.status:<11} {r.comment[:70]}")


if __name__ == '__main__':
    main()
