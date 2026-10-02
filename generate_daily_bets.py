#!/usr/bin/env python3
"""
Daily prop pipeline: Odds API props -> projections -> both sides evaluated -> CSV + log.

Costs Odds API credits: (#games) x (#markets) per run for the 'us' region, which covers
all major US books — so every prop is line-shopped across books for free. The best-EV
book/side per player+stat is kept.

Usage:
    python generate_daily_bets.py                                  # points props, today
    python generate_daily_bets.py --markets player_points player_rebounds --max-events 3
    python generate_daily_bets.py --date 2026-11-01 --min-ev 0.04
"""

import argparse
import csv

from bet_signals import MIN_EV, evaluate_prop, log_prop
from db import ROOT, connect, find_players, init_db
from odds_fetcher import PROP_MARKETS, fetch_props
from projection_model import PropContext

OUT_DIR = ROOT / 'daily_bets'
CSV_COLS = ['game_date', 'player_name', 'team', 'opponent', 'stat', 'line', 'book', 'over_odds',
            'under_odds', 'proj_mean', 'proj_sd', 'fair_line', 'p_over', 'p_under', 'p_push',
            'fair_over', 'ev_over', 'ev_under', 'pick', 'best_ev', 'kelly', 'warnings']


def resolve(conn, name, home, away):
    """Match a bookmaker player name to a DB player on one of the two teams."""
    cands = find_players(conn, name, teams=[home, away])
    if cands.empty:                   # traded since last game in DB, etc.
        cands = find_players(conn, name)
    if cands.empty or cands['norm'].nunique() > 1:
        return None
    return cands.iloc[0]


def run(date=None, markets=('player_points',), min_ev=MIN_EV, max_events=None, log=True):
    conn = connect()
    init_db(conn)
    props = fetch_props(date, markets, max_events=max_events)
    if not props:
        print("No props returned (no games, or markets not posted yet).")
        return []
    ctx = PropContext(conn, props[0]['game_date'])
    print(f"📊 {len(props)} book/prop rows; projecting with data through {ctx.last_data_date}")

    best = {}
    unmatched = set()
    for pr in props:
        pl = resolve(conn, pr['player'], pr['home'], pr['away'])
        if pl is None:
            unmatched.add(pr['player'])
            continue
        team = pl['team']
        opp = pr['away'] if team == pr['home'] else pr['home'] if team == pr['away'] else None
        r = evaluate_prop(ctx, pl['player_id'], pr['stat'], pr['line'], pr['over_odds'],
                          pr['under_odds'], opponent=opp, min_ev=min_ev)
        if r is None:
            continue
        r['book'] = pr['book']
        r['best_ev'] = max(r['ev_over'], r['ev_under'])
        if opp is None:
            r['warnings'].append("Player's DB team isn't in this game (trade?) — opponent not applied.")
        key = (r['player_id'], r['stat'])
        if key not in best or r['best_ev'] > best[key]['best_ev']:
            best[key] = r

    rows = sorted(best.values(), key=lambda r: r['best_ev'], reverse=True)
    if unmatched:
        print(f"⚠️  {len(unmatched)} names not matched to the DB: {', '.join(sorted(unmatched)[:10])}")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"props_{ctx.game_date}.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'proj_mean': round(r['mean'], 2), 'proj_sd': round(r['sd'], 2),
                        **{k: round(r[k], 4) for k in ('p_over', 'p_under', 'p_push', 'fair_over',
                                                       'ev_over', 'ev_under', 'best_ev', 'kelly')},
                        'warnings': ' | '.join(r['warnings'])})
            if log:
                log_prop(conn, r, r['book'])

    picks = [r for r in rows if r['pick'] != 'PASS']
    print(f"\n✅ {len(rows)} props evaluated -> {path}")
    print(f"🎯 {len(picks)} with EV >= {min_ev:.0%}:")
    for r in picks:
        odds = r['over_odds'] if r['pick'] == 'OVER' else r['under_odds']
        flag = ' ⚠️' if r['warnings'] else ''
        print(f"   {r['player_name']:<26} {r['pick']:<5} {r['line']:>5g} {r['stat']:<4} {odds:+d} @ {r['book']:<12}"
              f" proj {r['mean']:5.1f}  EV {r['best_ev']:+.1%}  ¼K {r['kelly'] / 4:.1%}{flag}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date')
    ap.add_argument('--markets', nargs='+', default=['player_points'], choices=list(PROP_MARKETS))
    ap.add_argument('--min-ev', type=float, default=MIN_EV)
    ap.add_argument('--max-events', type=int)
    ap.add_argument('--no-log', action='store_true')
    a = ap.parse_args()
    run(a.date, a.markets, a.min_ev, a.max_events, not a.no_log)


if __name__ == '__main__':
    main()
