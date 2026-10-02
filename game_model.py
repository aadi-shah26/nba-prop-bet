#!/usr/bin/env python3
"""
Game score model: predicted score, win probability, spread and total probabilities.

Ratings (all from games before the game date):
    possessions  = FGA - OREB + TOV + 0.44*FTA, averaged over both teams in a game
    offense/defense = points scored/allowed per 100 possessions
    pace         = possessions per game
Each is a recency-weighted average, adjusted for opponent strength (iterated like SRS)
and shrunk toward league average. Prediction:
    pace        = pace_home + pace_away - league_pace
    eff_home    = off_home + def_away - league_eff + hca/2   (per 100 possessions)
    eff_away    = off_away + def_home - league_eff - hca/2
    score       = eff * pace / 100
Margin and total are modelled as discretized normals; their SDs are calibrated from
walk-forward residuals (python daily_update.py stores them in model_params).

Usage:
    python game_model.py --home BOS --away LAL
    python game_model.py --home BOS --away LAL --spread -5.5 --spread-odds -110 -110 \
                         --total 221.5 --total-odds -110 -110 --ml -230 +190
    python game_model.py --slate            # today's games + free Odds API lines
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

from db import connect, season_for_date, team_abbr, to_date

DEFAULT_SD_MARGIN = 13.8   # walk-forward residual SDs 2021-24; recalibrated from your data
DEFAULT_SD_TOTAL = 18.6
DISAGREE_WARN = 0.10


@dataclass
class GameParams:
    # Tuned walk-forward on 2021-22/2022-23, confirmed on 2023-24: margins are best predicted
    # by slower-moving ratings, totals by faster ones (scoring environment drifts in-season).
    half_life_margin: float = 20.0   # games per team
    half_life_total: float = 10.0
    prev_season_weight: float = 0.3
    lookback_seasons: int = 2
    prior_games: float = 6.0         # pseudo-games of league average (shrinkage)
    adjust_iters: int = 25
    min_team_games: int = 1


def game_rows(team_games):
    """One row per team-game with possessions and per-100 efficiencies."""
    t = team_games.copy()
    t['poss'] = t['fga'] - t['oreb'] + t['tov'] + 0.44 * t['fta']
    o = t[['game_id', 'team', 'pts', 'poss']].rename(
        columns={'team': 'opponent', 'pts': 'opp_pts', 'poss': 'opp_poss'})
    m = t.merge(o, on=['game_id', 'opponent'], how='inner')
    m['pace'] = (m['poss'] + m['opp_poss']) / 2
    m['off'] = 100 * m['pts'] / m['pace']
    m['def'] = 100 * m['opp_pts'] / m['pace']
    return m.sort_values(['game_date', 'game_id']).reset_index(drop=True)


def team_ratings(rows, as_of, target_season, params, half_life):
    """Ratings from rows with game_date < as_of. Returns dict or None if no data."""
    h = rows[(rows['game_date'] < as_of) &
             (rows['season'] > target_season - params.lookback_seasons)]
    if h.empty:
        return None
    teams = sorted(set(h['team']) | set(h['opponent']))
    ix = {t: i for i, t in enumerate(teams)}
    ti = h['team'].map(ix).to_numpy()
    oi = h['opponent'].map(ix).to_numpy()
    games_ago = h.groupby('team').cumcount(ascending=False).to_numpy(dtype=float)
    back = np.clip(target_season - h['season'].to_numpy(dtype=float), 0, None)
    w = 0.5 ** (games_ago / half_life) * params.prev_season_weight ** back
    off, dfn, pace = (h[c].to_numpy(dtype=float) for c in ('off', 'def', 'pace'))

    n = len(teams)
    sw = np.bincount(ti, w, n)
    lg_eff = (w * off).sum() / w.sum()
    lg_pace = (w * pace).sum() / w.sum()

    def wmean(v):
        return np.bincount(ti, w * v, n) / sw

    O, D, P = wmean(off), wmean(dfn), wmean(pace)
    for _ in range(params.adjust_iters):   # opponent adjustment (SRS-style fixed point)
        O_new = wmean(off - (D[oi] - lg_eff))
        D_new = wmean(dfn - (O[oi] - lg_eff))
        P_new = wmean(pace - (P[oi] - lg_pace))
        # re-center so league averages are preserved
        O_new += lg_eff - np.average(O_new, weights=sw)
        D_new += lg_eff - np.average(D_new, weights=sw)
        P_new += lg_pace - np.average(P_new, weights=sw)
        done = max(abs(O_new - O).max(), abs(D_new - D).max(), abs(P_new - P).max()) < 1e-6
        O, D, P = O_new, D_new, P_new
        if done:
            break

    k = params.prior_games
    shrink = sw / (sw + k)
    O = lg_eff + shrink * (O - lg_eff)
    D = lg_eff + shrink * (D - lg_eff)
    P = lg_pace + shrink * (P - lg_pace)

    home = h[h['home'] == 1]
    hca_pts = float((home['pts'] - home['opp_pts']).mean()) if len(home) else 2.5
    return {
        'teams': {t: {'off': O[i], 'def': D[i], 'pace': P[i], 'games_w': sw[i]}
                  for t, i in ix.items()},
        'lg_eff': lg_eff, 'lg_pace': lg_pace, 'hca_pts': hca_pts,
        'as_of': as_of, 'last_game': h['game_date'].max(),
    }


def build_ratings(rows, as_of, target_season, params):
    """Two rating sets: slow (for the margin) and fast (for the total)."""
    rm = team_ratings(rows, as_of, target_season, params, params.half_life_margin)
    if rm is None:
        return None
    rt = team_ratings(rows, as_of, target_season, params, params.half_life_total)
    return {'margin': rm, 'total': rt, 'teams': rm['teams'], 'last_game': rm['last_game']}


def predict_game(ratings, home, away):
    """Margin from the slow ratings, total from the fast ratings."""
    pm = _predict_one(ratings['margin'], home, away)
    pt = _predict_one(ratings['total'], home, away)
    margin, total = pm['margin'], pt['total']
    return {'home': home, 'away': away, 'pred_home': (total + margin) / 2,
            'pred_away': (total - margin) / 2, 'margin': margin, 'total': total,
            'pace': pt['pace']}


def _predict_one(ratings, home, away):
    T = ratings['teams']
    if home not in T or away not in T:
        missing = [t for t in (home, away) if t not in T]
        raise KeyError(f"No rating for {missing}")
    H, A = T[home], T[away]
    lg, lp = ratings['lg_eff'], ratings['lg_pace']
    pace = H['pace'] + A['pace'] - lp
    hca_eff = 100 * ratings['hca_pts'] / pace
    eff_h = H['off'] + A['def'] - lg + hca_eff / 2
    eff_a = A['off'] + H['def'] - lg - hca_eff / 2
    ph, pa = eff_h * pace / 100, eff_a * pace / 100
    return {'home': home, 'away': away, 'pred_home': ph, 'pred_away': pa,
            'margin': ph - pa, 'total': ph + pa, 'pace': pace}


# ---------------------------------------------------------------------------
# Probabilities (discretized normal on integers)
# ---------------------------------------------------------------------------

def _int_pmf(mu, sd, lo, hi):
    k = np.arange(lo, hi + 1)
    cdf_hi = sps.norm.cdf((k + 0.5 - mu) / sd)
    cdf_lo = sps.norm.cdf((k - 0.5 - mu) / sd)
    pmf = cdf_hi - cdf_lo
    return k, pmf / pmf.sum()


def margin_pmf(mu, sd):
    """Home margin pmf. NBA games can't end tied: the 0 mass (overtime) is split 50/50
    onto +1/-1 (OT margins are small)."""
    k, p = _int_pmf(mu, sd, -90, 90)
    z = p[k == 0][0]
    p = p.copy()
    p[k == 0] = 0
    p[k == 1] += z / 2
    p[k == -1] += z / 2
    return k, p


def total_pmf(mu, sd):
    return _int_pmf(mu, sd, max(int(mu - 10 * sd), 0), int(mu + 10 * sd) + 1)


def probs_vs_line(k, p, x):
    """(P(X > x), P(X < x), P(X == x)) for integer support k."""
    return float(p[k > x].sum()), float(p[k < x].sum()), float(p[k == x].sum())


def market_probs(pred, sd_margin, sd_total, spread=None, total=None):
    """
    spread: the HOME team's line (e.g. -5.5 = home favored by 5.5). Home covers if margin + spread > 0.
    Returns dict with p_home_win, and if given p_home_cover/p_away_cover/p_spread_push,
    p_over/p_under/p_total_push.
    """
    km, pm = margin_pmf(pred['margin'], sd_margin)
    out = {'p_home_win': float(pm[km > 0].sum())}
    if spread is not None:
        a, b, push = probs_vs_line(km, pm, -spread)
        out.update(p_home_cover=a, p_away_cover=b, p_spread_push=push)
    if total is not None:
        kt, pt = total_pmf(pred['total'], sd_total)
        o, u, push = probs_vs_line(kt, pt, total)
        out.update(p_over=o, p_under=u, p_total_push=push)
    return out


# ---------------------------------------------------------------------------
# Walk-forward backtest + calibration
# ---------------------------------------------------------------------------

def backtest_games(team_games, seasons, params=None, sd_margin=DEFAULT_SD_MARGIN,
                   sd_total=DEFAULT_SD_TOTAL):
    params = params or GameParams()
    rows = game_rows(team_games)
    out = []
    for season in seasons:
        s_rows = rows[(rows['season'] == season) & (rows['home'] == 1)]
        for d, games in s_rows.groupby('game_date'):
            r = build_ratings(rows, d, season, params)
            if r is None:
                continue
            # naive baseline: each team's season-to-date points for/against (prev season if none)
            prior = rows[(rows['game_date'] < d) & (rows['season'] >= season - 1)]
            cur = prior[prior['season'] == season]
            for g in games.itertuples():
                if g.team not in r['teams'] or g.opponent not in r['teams']:
                    continue
                p = predict_game(r, g.team, g.opponent)
                pr = market_probs(p, sd_margin, sd_total)

                def avg(team, col):
                    src = cur if (cur['team'] == team).sum() >= 3 else prior
                    return src.loc[src['team'] == team, col].mean()
                bh = (avg(g.team, 'pts') + avg(g.opponent, 'opp_pts')) / 2
                ba = (avg(g.opponent, 'pts') + avg(g.team, 'opp_pts')) / 2
                out.append({'season': season, 'game_date': d, 'home': g.team, 'away': g.opponent,
                            'pred_home': p['pred_home'], 'pred_away': p['pred_away'],
                            'pred_margin': p['margin'], 'pred_total': p['total'],
                            'p_home_win': pr['p_home_win'],
                            'act_margin': g.pts - g.opp_pts, 'act_total': g.pts + g.opp_pts,
                            'base_margin': bh - ba, 'base_total': bh + ba,
                            'season_type': g.season_type})
    return pd.DataFrame(out)


def calibrate(conn, params=None, min_games=600):
    """Estimate margin/total residual SDs from the most recent walk-forward predictions
    and store them in model_params. Returns (sd_margin, sd_total, n) or None."""
    tg = pd.read_sql_query("SELECT * FROM team_games", conn)
    if tg.empty:
        return None
    seasons = sorted(tg['season'].unique())
    # need a prior season for ratings: evaluate the last two seasons that have one
    eval_seasons = [s for s in seasons if s - 1 in seasons][-2:]
    if not eval_seasons:
        return None
    df = backtest_games(tg, eval_seasons, params)
    df = df.sort_values('game_date').tail(1500)
    if len(df) < min_games:
        return None
    sd_m = float((df['act_margin'] - df['pred_margin']).std())
    sd_t = float((df['act_total'] - df['pred_total']).std())
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    conn.executemany("INSERT OR REPLACE INTO model_params (key, value, updated_at) VALUES (?, ?, ?)",
                     [('game_sd_margin', sd_m, now), ('game_sd_total', sd_t, now)])
    conn.commit()
    return sd_m, sd_t, len(df)


def stored_sds(conn):
    vals = dict(conn.execute("SELECT key, value FROM model_params WHERE key LIKE 'game_sd_%'").fetchall())
    return vals.get('game_sd_margin', DEFAULT_SD_MARGIN), vals.get('game_sd_total', DEFAULT_SD_TOTAL), \
        bool(vals)


# ---------------------------------------------------------------------------
# Live use
# ---------------------------------------------------------------------------

class GameContext:
    def __init__(self, conn, game_date=None, params=None):
        self.conn = conn
        self.params = params or GameParams()
        self.game_date = to_date(game_date).isoformat()
        self.season = season_for_date(self.game_date)
        tg = pd.read_sql_query("SELECT * FROM team_games WHERE season >= ? AND game_date < ?",
                               conn, params=(self.season - self.params.lookback_seasons + 1,
                                             self.game_date))
        if tg.empty:
            raise RuntimeError("No team games in DB. Run: python fetch_game_logs.py")
        self.ratings = build_ratings(game_rows(tg), self.game_date, self.season, self.params)
        self.sd_margin, self.sd_total, self.calibrated = stored_sds(conn)

    def evaluate(self, home, away, spread=None, total=None):
        home, away = team_abbr(home), team_abbr(away)
        if not home or not away:
            raise ValueError("Unknown team")
        pred = predict_game(self.ratings, home, away)
        pred.update(market_probs(pred, self.sd_margin, self.sd_total, spread, total))
        return pred


def evaluate_markets(pred, spread=None, spread_odds=None, total=None, total_odds=None,
                     ml=None, min_ev=0.03):
    """Turn model probabilities + prices into EV and picks for each market given."""
    from odds import decide, kelly_fraction, no_vig_probs
    res = []
    if spread is not None and spread_odds:
        oa, ob = spread_odds
        pick, ea, eb = decide(pred['p_home_cover'], pred['p_away_cover'], oa, ob, min_ev)
        res.append(('SPREAD', spread, oa, ob, pred['p_home_cover'], pred['p_away_cover'],
                    no_vig_probs(oa, ob)[0], ea, eb, pick))
    if total is not None and total_odds:
        oa, ob = total_odds
        pick, ea, eb = decide(pred['p_over'], pred['p_under'], oa, ob, min_ev)
        res.append(('TOTAL', total, oa, ob, pred['p_over'], pred['p_under'],
                    no_vig_probs(oa, ob)[0], ea, eb, pick))
    if ml:
        oa, ob = ml
        p = pred['p_home_win']
        pick, ea, eb = decide(p, 1 - p, oa, ob, min_ev)
        res.append(('MONEYLINE', None, oa, ob, p, 1 - p, no_vig_probs(oa, ob)[0], ea, eb, pick))
    out = []
    for market, line, oa, ob, pa, pb, fair, ea, eb, pick in res:
        k = kelly_fraction(pa, pb, oa) if pick == 'A' else kelly_fraction(pb, pa, ob) if pick == 'B' else 0
        out.append({'market': market, 'line': line, 'odds_a': oa, 'odds_b': ob, 'p_a': pa, 'p_b': pb,
                    'fair_a': fair, 'ev_a': ea, 'ev_b': eb, 'pick': pick, 'kelly': k})
    return out


def log_game_evals(conn, game_date, pred, evals, book=None):
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for e in evals:
        conn.execute("""INSERT OR IGNORE INTO game_log (logged_at, game_date, home, away, market, line,
                        odds_a, odds_b, book, pred_home, pred_away, p_a, p_b, fair_a, ev_a, ev_b, pick)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (now, game_date, pred['home'], pred['away'], e['market'], e['line'],
                      int(e['odds_a']), int(e['odds_b']), book, pred['pred_home'], pred['pred_away'],
                      e['p_a'], e['p_b'], e['fair_a'], e['ev_a'], e['ev_b'], e['pick']))
    conn.commit()


def settle_games(conn):
    """Fill results for logged game bets whose games are now in team_games."""
    rows = conn.execute("SELECT id, game_date, home, away, market, line, pick FROM game_log "
                        "WHERE result IS NULL").fetchall()
    n = 0
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for gid, d, home, away, market, line, pick in rows:
        r = conn.execute("""SELECT h.pts, a.pts FROM team_games h JOIN team_games a
                            ON a.game_id = h.game_id AND a.team = h.opponent
                            WHERE h.team = ? AND h.opponent = ? AND h.game_date = ? AND h.home = 1""",
                         (home, away, d)).fetchone()
        if not r:
            continue
        hp, ap = r
        if market == 'SPREAD':
            v = hp - ap + line
            side = 'A' if v > 0 else 'B' if v < 0 else 'PUSH'
        elif market == 'TOTAL':
            v = hp + ap - line
            side = 'A' if v > 0 else 'B' if v < 0 else 'PUSH'
        else:
            side = 'A' if hp > ap else 'B'
        if pick == 'PASS':
            result = f'PASS ({side})'
        else:
            result = 'PUSH' if side == 'PUSH' else 'WIN' if side == pick else 'LOSS'
        conn.execute("UPDATE game_log SET home_pts=?, away_pts=?, result=?, settled_at=? WHERE id=?",
                     (hp, ap, result, now, gid))
        n += 1
    conn.commit()
    return n


def print_game(pred, ctx, evals):
    print(f"\n{'=' * 66}")
    print(f"🏀 {pred['away']} @ {pred['home']}   ({ctx.game_date}, ratings through "
          f"{ctx.ratings['last_game']})")
    print(f"{'=' * 66}")
    print(f"Predicted score:  {pred['away']} {pred['pred_away']:.1f} — {pred['home']} {pred['pred_home']:.1f}")
    print(f"Home margin:      {pred['margin']:+.1f}  (fair spread {pred['home']} {-pred['margin']:+.1f})")
    print(f"Total:            {pred['total']:.1f}   pace {pred['pace']:.1f}")
    print(f"Home win prob:    {pred['p_home_win']:.1%}   (fair ML from model: "
          f"{_fair_american(pred['p_home_win'])} / {_fair_american(1 - pred['p_home_win'])})")
    if not ctx.calibrated:
        print("⚠️  SDs not calibrated yet (run python daily_update.py); using defaults "
              f"{DEFAULT_SD_MARGIN}/{DEFAULT_SD_TOTAL}")
    names = {'SPREAD': (f"{pred['home']}", f"{pred['away']}"), 'TOTAL': ('OVER', 'UNDER'),
             'MONEYLINE': (pred['home'], pred['away'])}
    for e in evals:
        a, b = names[e['market']]
        line = '' if e['line'] is None else (f" {e['line']:+g}" if e['market'] == 'SPREAD' else f" {e['line']:g}")
        print(f"\n  {e['market']}{line}")
        print(f"    {a:<6} {e['odds_a']:>+5}  model {e['p_a']:.1%}  market(no-vig) {e['fair_a']:.1%}  EV {e['ev_a']:+.1%}")
        print(f"    {b:<6} {e['odds_b']:>+5}  model {e['p_b']:.1%}  market(no-vig) {1 - e['fair_a']:.1%}  EV {e['ev_b']:+.1%}")
        pick = 'PASS' if e['pick'] == 'PASS' else (a if e['pick'] == 'A' else b)
        extra = f"  (¼-Kelly stake {e['kelly'] / 4:.1%} of bankroll)" if e['pick'] != 'PASS' else ''
        print(f"    ➜ {pick}{extra}")
        p_np = e['p_a'] / (e['p_a'] + e['p_b']) if e['p_a'] + e['p_b'] > 0 else 0.5
        if abs(p_np - e['fair_a']) > DISAGREE_WARN:
            print(f"    ⚠️  Model and market differ by {abs(p_np - e['fair_a']):.0%}. Game lines are sharp: "
                  "this almost always means injuries/rest the model can't see. Check news first.")
    days_old = (to_date(ctx.game_date) - to_date(ctx.ratings['last_game'])).days
    if days_old > 4:
        print(f"⚠️  Ratings use games through {ctx.ratings['last_game']} ({days_old} days old). "
              "Run python daily_update.py.")
    print()


def _fair_american(p):
    if p <= 0 or p >= 1:
        return 'n/a'
    return f"{-100 * p / (1 - p):+.0f}" if p >= 0.5 else f"{100 * (1 - p) / p:+.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--home')
    ap.add_argument('--away')
    ap.add_argument('--date', help='game date YYYY-MM-DD (default today)')
    ap.add_argument('--spread', type=float, help="HOME team's spread, e.g. -5.5")
    ap.add_argument('--spread-odds', type=int, nargs=2, default=[-110, -110], metavar=('HOME', 'AWAY'))
    ap.add_argument('--total', type=float)
    ap.add_argument('--total-odds', type=int, nargs=2, default=[-110, -110], metavar=('OVER', 'UNDER'))
    ap.add_argument('--ml', type=int, nargs=2, metavar=('HOME', 'AWAY'), help='moneyline odds')
    ap.add_argument('--min-ev', type=float, default=0.03)
    ap.add_argument('--slate', action='store_true', help="evaluate today's games with Odds API lines")
    ap.add_argument('--book', default='draftkings', help='bookmaker key for --slate')
    ap.add_argument('--no-log', action='store_true', help="don't write evaluations to game_log")
    a = ap.parse_args()

    conn = connect()
    if a.slate:
        from odds_fetcher import fetch_game_lines
        lines = fetch_game_lines(book=a.book, date=a.date)
        if not lines:
            print("No games found for that date/book.")
            return
        ctx = GameContext(conn, lines[0]['game_date'])
        for g in lines:
            pred = ctx.evaluate(g['home'], g['away'], g.get('spread'), g.get('total'))
            evals = evaluate_markets(pred, g.get('spread'), g.get('spread_odds'), g.get('total'),
                                     g.get('total_odds'), g.get('ml'), a.min_ev)
            print_game(pred, ctx, evals)
            if not a.no_log:
                log_game_evals(conn, ctx.game_date, pred, evals, a.book)
        return

    if not (a.home and a.away):
        ap.error('--home and --away are required (or use --slate)')
    ctx = GameContext(conn, a.date)
    pred = ctx.evaluate(a.home, a.away, a.spread, a.total)
    evals = evaluate_markets(pred, a.spread, a.spread_odds if a.spread is not None else None,
                             a.total, a.total_odds if a.total is not None else None, a.ml, a.min_ev)
    print_game(pred, ctx, evals)
    if evals and not a.no_log:
        log_game_evals(conn, ctx.game_date, pred, evals)


if __name__ == '__main__':
    main()
