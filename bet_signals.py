#!/usr/bin/env python3
"""
Prop evaluator: you give a line and the prices, it tells you OVER / UNDER / PASS.

For each prop it shows:
  * model P(over), P(under), P(push) from the player's projected distribution
  * the market's no-vig probability (both prices with the bookmaker margin removed)
  * EV per $1 for each side (pushes refund the stake) and a ¼-Kelly stake suggestion
  * warnings when the model is likely missing information
Every evaluation is logged to prop_log and settled automatically by daily_update.py, so
`python bet_signals.py --report` shows how your picks actually performed.

Usage:
    python bet_signals.py                                     # interactive
    python bet_signals.py "shai" PTS 30.5 --over -115 --under -105 --opp LAL
    python bet_signals.py --report
"""

import argparse
from datetime import datetime, timezone


from db import connect, find_players, init_db, to_date
from feature_engineering import STATS, parse_stat
from odds import decide, kelly_fraction, no_vig_probs
from projection_model import PropContext, line_probs

MIN_EV = 0.03            # don't bet below +3% EV: model error eats thinner edges
DISAGREE_WARN = 0.12     # model vs market gap that usually means missing news


def evaluate_prop(ctx, player_id, stat, line, over_odds=-110, under_odds=-110,
                  opponent=None, minutes=None, min_ev=MIN_EV):
    stat = parse_stat(stat)
    p = ctx.project(player_id, stat, opponent=opponent, minutes=minutes)
    if p is None:
        return None
    p_over, p_under, p_push = line_probs(p['mean'], p['var'], line)
    fair_over, fair_under, vig = no_vig_probs(over_odds, under_odds)
    pick, ev_over, ev_under = decide(p_over, p_under, over_odds, under_odds, min_ev)
    pick = {'A': 'OVER', 'B': 'UNDER'}.get(pick, 'PASS')
    kelly = (kelly_fraction(p_over, p_under, over_odds) if pick == 'OVER' else
             kelly_fraction(p_under, p_over, under_odds) if pick == 'UNDER' else 0.0)

    # model's over probability excluding pushes, comparable to the two-way market price
    p_over_np = p_over / (p_over + p_under) if p_over + p_under > 0 else 0.5
    warnings = []
    if abs(p_over_np - fair_over) > DISAGREE_WARN:
        warnings.append(f"Model and market disagree by {abs(p_over_np - fair_over):.0%}. That usually "
                        "means news the model can't see (injury, minutes limit, role change, rest). "
                        "Check before betting.")
    days_old = (to_date(ctx.game_date) - to_date(ctx.last_data_date)).days
    if days_old > 4 and p['season_games'] > 0:
        warnings.append(f"Latest game in DB is {ctx.last_data_date} ({days_old} days before this game). "
                        "Run python daily_update.py.")
    if p['season_games'] < 10:
        warnings.append(f"Only {p['season_games']} games this season — leaning on last season's data.")
    if p['minutes_hist'] < 18 and minutes is None:
        warnings.append(f"Low-minutes player ({p['minutes_hist']:.1f} mpg): outcome depends heavily on "
                        "playing time; consider --minutes.")

    return {**p, 'line': line, 'over_odds': over_odds, 'under_odds': under_odds,
            'p_over': p_over, 'p_under': p_under, 'p_push': p_push,
            'fair_over': fair_over, 'fair_under': fair_under, 'vig': vig,
            'ev_over': ev_over, 'ev_under': ev_under, 'pick': pick, 'kelly': kelly,
            'game_date': ctx.game_date, 'min_ev': min_ev, 'warnings': warnings}


def print_eval(r):
    print(f"\n{'=' * 70}")
    print(f"🎯 {r['player_name']} ({r['team']}) {r['stat']} {r['line']:g}"
          + (f" vs {r['opponent']}" if r['opponent'] else '') + f"   [{r['game_date']}]")
    print(f"{'=' * 70}")
    print(f"Projection:   {r['mean']:.2f} ± {r['sd']:.2f}    fair line {r['fair_line']:g}    "
          f"minutes {r['minutes_proj']:.1f}")
    sa = f"{r['season_avg']:.2f} ({r['season_games']} g)" if r['season_avg'] is not None else 'n/a'
    print(f"Season avg:   {sa}   last10 {r['last10_avg']:.2f}   last5 {r['last5_avg']:.2f}")
    print(f"\n{'':8}{'odds':>6} {'model':>8} {'market':>8} {'EV':>8}")
    print(f"{'OVER':<8}{r['over_odds']:>+6} {r['p_over']:>8.1%} {r['fair_over']:>8.1%} {r['ev_over']:>+8.1%}")
    print(f"{'UNDER':<8}{r['under_odds']:>+6} {r['p_under']:>8.1%} {r['fair_under']:>8.1%} {r['ev_under']:>+8.1%}")
    if r['p_push'] > 0:
        print(f"{'PUSH':<8}{'':>6} {r['p_push']:>8.1%}")
    print(f"(market = no-vig probability; bookmaker margin on this prop {r['vig']:.1%})")
    if r['pick'] == 'PASS':
        print(f"\n➜ PASS — no side clears +{r['min_ev']:.0%} EV")
    else:
        print(f"\n➜ BET {r['pick']}   ¼-Kelly stake: {r['kelly'] / 4:.1%} of bankroll")
    for w in r['warnings']:
        print(f"⚠️  {w}")
    print()


# ---------------------------------------------------------------------------
# Logging / settling / report
# ---------------------------------------------------------------------------

def log_prop(conn, r, book=None):
    conn.execute("""INSERT OR IGNORE INTO prop_log (logged_at, game_date, player_id, player_name, stat,
                    line, over_odds, under_odds, book, opponent, proj_mean, proj_sd, p_over, p_under,
                    p_push, fair_over, ev_over, ev_under, pick)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (datetime.now(timezone.utc).isoformat(timespec='seconds'), r['game_date'],
                  r['player_id'], r['player_name'], r['stat'], r['line'], int(r['over_odds']),
                  int(r['under_odds']), book or '', r['opponent'], r['mean'], r['sd'], r['p_over'],
                  r['p_under'], r['p_push'], r['fair_over'], r['ev_over'], r['ev_under'], r['pick']))
    conn.commit()


def settle_props(conn):
    """Fill actual results for logged props whose games are now in player_games.
    A player with no row once that date's games are loaded = DNP (books void these)."""
    latest = conn.execute("SELECT MAX(game_date) FROM team_games").fetchone()[0]
    rows = conn.execute("SELECT id, game_date, player_id, stat, line, pick FROM prop_log "
                        "WHERE result IS NULL").fetchall()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    n = 0
    for pid_row, d, pid, stat, line, pick in rows:
        comps = STATS[stat]
        g = conn.execute(f"SELECT {' + '.join(comps)} FROM player_games WHERE player_id = ? AND game_date = ?",
                         (pid, d)).fetchone()
        if g is None:
            if latest and d < latest:
                conn.execute("UPDATE prop_log SET result='DNP', settled_at=? WHERE id=?", (now, pid_row))
                n += 1
            continue
        actual = float(g[0])
        side = 'OVER' if actual > line else 'UNDER' if actual < line else 'PUSH'
        if pick == 'PASS':
            result = f'PASS ({side})'
        else:
            result = 'PUSH' if side == 'PUSH' else 'WIN' if side == pick else 'LOSS'
        conn.execute("UPDATE prop_log SET actual=?, result=?, settled_at=? WHERE id=?",
                     (actual, result, now, pid_row))
        n += 1
    conn.commit()
    return n


def report(conn):
    from tracking import game_summary, prop_summary
    print(f"\n{'=' * 70}\n📒 PERFORMANCE REPORT\n{'=' * 70}")
    for label, summ in (('Props', prop_summary(conn)), ('Game lines', game_summary(conn))):
        print(f"\n{label}: logged {summ['logged']}, settled {summ['settled']}")
        r = summ['record']
        if r:
            print(f"  Picks: {r['picks']}   {r['wins']}-{r['losses']}-{r['pushes']}   profit {r['profit']:+.2f}u"
                  f"   ROI {r['roi']:+.1%}   (model claimed avg EV {r['claimed_ev']:+.1%})")
        cal = summ.get('calibration')
        if cal and cal['n'] >= 20:
            better = 'model better' if cal['brier_model'] < cal['brier_market'] else 'market better'
            print(f"  Calibration on {cal['n']} settled props: Brier model {cal['brier_model']:.4f} vs "
                  f"market {cal['brier_market']:.4f} ({better})")
            print(cal['table'].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("\nA real edge needs a few hundred picks to show up; under ~100 this is mostly noise.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _ask(prompt, default=None, cast=str):
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw:
            if default is not None:
                return default
            continue
        try:
            return cast(raw)
        except ValueError as e:
            print(f"  ❌ {e}")


def _pick_player(conn, query):
    cands = find_players(conn, query)
    if cands.empty:
        print(f"  ❌ No player matches {query!r}")
        return None
    if len(cands) == 1:
        return cands.iloc[0]
    print("  Multiple matches:")
    for i, r in enumerate(cands.head(10).itertuples(), 1):
        print(f"   {i}. {r.player_name} ({r.team}, last game {r.last_game})")
    i = _ask("  Choose #", 1, int)
    return cands.iloc[min(max(i, 1), min(len(cands), 10)) - 1]


def interactive(conn, game_date=None, book=None, log=True):
    ctx = PropContext(conn, game_date)
    print(f"\n💰 PROP EVALUATOR — game date {ctx.game_date} (data through {ctx.last_data_date})")
    print(f"Stats: {', '.join(STATS)}.  Blank odds = -110.  'quit' to exit.\n")
    while True:
        q = input("Player: ").strip()
        if q.lower() in ('quit', 'q', 'exit'):
            break
        if not q:
            continue
        pl = _pick_player(conn, q)
        if pl is None:
            continue
        stat = _ask("Stat", 'PTS', parse_stat)
        line = _ask("Line", cast=float)
        over = _ask("Over odds", -110, int)
        under = _ask("Under odds", -110, int)
        opp = _ask("Opponent (blank = neutral)", '', str) or None
        mins = _ask("Projected minutes (blank = model)", '', str)
        try:
            r = evaluate_prop(ctx, pl['player_id'], stat, line, over, under, opp,
                              float(mins) if mins else None)
        except ValueError as e:
            print(f"  ❌ {e}\n")
            continue
        if r is None:
            print("  ❌ Not enough recent games to project this player.\n")
            continue
        print_eval(r)
        if log:
            log_prop(conn, r, book)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('player', nargs='?')
    ap.add_argument('stat', nargs='?')
    ap.add_argument('line', nargs='?', type=float)
    ap.add_argument('--over', type=int, default=-110, help='over odds (American)')
    ap.add_argument('--under', type=int, default=-110, help='under odds (American)')
    ap.add_argument('--opp', help='opponent team')
    ap.add_argument('--minutes', type=float, help='override projected minutes')
    ap.add_argument('--date', help='game date YYYY-MM-DD (default today)')
    ap.add_argument('--book', help='bookmaker name, stored in the log')
    ap.add_argument('--min-ev', type=float, default=MIN_EV)
    ap.add_argument('--no-log', action='store_true')
    ap.add_argument('--report', action='store_true', help='show performance of logged props')
    a = ap.parse_args()

    conn = connect()
    init_db(conn)
    if a.report:
        report(conn)
        return
    if a.player is None:
        interactive(conn, a.date, a.book, not a.no_log)
        return
    if a.stat is None or a.line is None:
        ap.error('give PLAYER STAT LINE, or no arguments for interactive mode')
    cands = find_players(conn, a.player)
    if cands.empty:
        raise SystemExit(f"No player matches {a.player!r}")
    if cands['norm'].nunique() > 1:
        raise SystemExit("Ambiguous: " + ', '.join(f"{r.player_name} ({r.team})" for r in cands.head(8).itertuples()))
    ctx = PropContext(conn, a.date)
    r = evaluate_prop(ctx, cands.iloc[0]['player_id'], a.stat, a.line, a.over, a.under,
                      a.opp, a.minutes, a.min_ev)
    if r is None:
        raise SystemExit("Not enough recent games to project this player.")
    print_eval(r)
    if not a.no_log:
        log_prop(conn, r, a.book)


if __name__ == '__main__':
    main()
