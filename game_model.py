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
The margin is then adjusted for two things the team ratings can't see:
    margin = scale * rating margin
             + b2b_pts * (away on a back-to-back - home on a back-to-back)
             + lineup_coef * (home lineup change - away lineup change)
    lineup change = strength of tonight's top players - the team's recent norm, where a
                    player's strength is his recency-weighted Game Score (Hollinger) and the
                    norm is weighted like the margin ratings, so a missing star (or a star
                    back from injury) moves the margin.
The three coefficients are fitted walk-forward (python daily_update.py stores them in
model_params); scale > 1 because shrunk ratings understate real gaps between teams.
Margin and total are modelled as discretized normals; their SDs are calibrated from
walk-forward residuals the same way.

Usage:
    python game_model.py --home BOS --away LAL
    python game_model.py --home BOS --away LAL --home-out "Jayson Tatum"   # players out tonight
    python game_model.py --home BOS --away LAL --spread -5.5 --spread-odds -110 -110 \
                         --total 221.5 --total-odds -110 -110 --ml -230 +190
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

from db import connect, season_for_date, team_abbr, to_date

DEFAULT_SD_MARGIN = 14.0   # walk-forward residual SDs; recalibrated from your data
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
    # Lineup strength: a player's value is his Game Score, EWM over his own appearances;
    # a team's strength is the sum over its top `lineup_size` players who play. The norm it is
    # compared with uses the same half-life as the margin ratings. Chosen on 2024-25.
    player_half_life: float = 40.0
    player_min_games: int = 5
    lineup_size: int = 8
    lineup_ref_half_life: float = 20.0
    roster_recent_games: int = 10   # live: players who appeared in the team's last N games


# Margin adjustment coefficients, fitted on 2024-25 (no intercept: home court stays in the
# ratings). calibrate() refits them on your most recent games.
DEFAULT_COEFS = {'scale': 1.15, 'b2b': 2.7, 'lineup': 0.14}
# The lineup adjustment made playoff picks worse in both 2024-25 (60.0% -> 58.9%) and 2025-26
# (64.8% -> 57.1%): the regular-season norm it compares against doesn't fit playoff rotations.
LINEUP_SEASON_TYPES = ('Regular Season',)
COEF_BOUNDS = {'scale': (0.8, 1.5), 'b2b': (0.0, 6.0), 'lineup': (0.0, 0.4)}


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


# ---------------------------------------------------------------------------
# Lineup strength and rest
# ---------------------------------------------------------------------------

def game_score(pg):
    """Hollinger Game Score per player-game."""
    return (pg['pts'] + 0.4 * pg['fgm'] - 0.7 * pg['fga'] - 0.4 * (pg['fta'] - pg['ftm'])
            + 0.7 * pg['oreb'] + 0.3 * pg['dreb'] + pg['stl'] + 0.7 * pg['ast'] + 0.7 * pg['blk']
            - 0.4 * pg['pf'] - pg['tov'])


def player_values(player_games, params):
    """One row per player-game played, with the player's value before (val_pre) and after
    (val_post) that game. val_pre only uses the player's earlier games."""
    p = player_games[player_games['minutes'] > 0]
    p = p.sort_values(['game_date', 'game_id']).reset_index(drop=True)
    p = p[['player_id', 'player_name', 'team', 'game_id', 'game_date']].assign(gs=game_score(p))
    post = p.groupby('player_id')['gs'].transform(
        lambda x: x.ewm(halflife=params.player_half_life, min_periods=params.player_min_games).mean())
    p['val_post'] = post.fillna(0.0)
    p['val_pre'] = post.groupby(p['player_id']).shift().fillna(0.0)
    return p


def lineup_history(values, params):
    """One row per team-game: strength of the players who played (pre-game values), the
    team's norm before the game (ref) and after it (ref_post), and delta = strength - ref."""
    top = values.sort_values('val_pre', ascending=False).groupby(['game_id', 'team']).head(params.lineup_size)
    s = (top.groupby(['game_id', 'team'])
         .agg(strength=('val_pre', 'sum'), game_date=('game_date', 'first')).reset_index()
         .sort_values(['game_date', 'game_id']).reset_index(drop=True))
    post = s.groupby('team')['strength'].transform(
        lambda x: x.ewm(halflife=params.lineup_ref_half_life, min_periods=3).mean())
    s['ref_post'] = post
    s['ref'] = post.groupby(s['team']).shift()
    s['delta'] = (s['strength'] - s['ref']).fillna(0.0)
    return s


def rest_flags(rows):
    """{(game_id, team): 1 if the team also played the previous day}."""
    t = rows[['game_id', 'team', 'game_date']].drop_duplicates().copy()
    t['dt'] = pd.to_datetime(t['game_date'])
    t = t.sort_values('dt')
    gap = t.groupby('team')['dt'].diff().dt.days
    return dict(zip(zip(t['game_id'], t['team']), (gap == 1).astype(int)))


def adjust_prediction(pred, coefs, b2b_home=0, b2b_away=0, lineup_home=0.0, lineup_away=0.0):
    """Apply the margin adjustments; the total is unchanged."""
    parts = {
        'scale': (coefs['scale'] - 1) * pred['margin'],
        'b2b': coefs['b2b'] * (b2b_away - b2b_home),
        'lineup': coefs['lineup'] * (lineup_home - lineup_away),
    }
    margin = pred['margin'] + sum(parts.values())
    return {**pred, 'margin': margin, 'rating_margin': pred['margin'],
            'pred_home': (pred['total'] + margin) / 2, 'pred_away': (pred['total'] - margin) / 2,
            'adjustments': parts, 'b2b_home': b2b_home, 'b2b_away': b2b_away,
            'lineup_home': lineup_home, 'lineup_away': lineup_away}


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
                   sd_total=DEFAULT_SD_TOTAL, player_games=None, coefs=None):
    """Walk-forward: every game is predicted from games before its date only. Without
    player_games the lineup adjustment is 0. Rows carry the components (rating_margin,
    b2b_diff, lineup_diff) so calibrate() can refit the coefficients."""
    params = params or GameParams()
    coefs = coefs or DEFAULT_COEFS
    rows = game_rows(team_games)
    b2b = rest_flags(rows)
    lineup = {}
    if player_games is not None and len(player_games):
        lh = lineup_history(player_values(player_games, params), params)
        lineup = dict(zip(zip(lh['game_id'], lh['team']), lh['delta']))
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
                bh, ba = b2b.get((g.game_id, g.team), 0), b2b.get((g.game_id, g.opponent), 0)
                lh_ = la_ = 0.0
                if g.season_type in LINEUP_SEASON_TYPES:
                    lh_, la_ = lineup.get((g.game_id, g.team), 0.0), lineup.get((g.game_id, g.opponent), 0.0)
                p = adjust_prediction(predict_game(r, g.team, g.opponent), coefs, bh, ba, lh_, la_)
                pr = market_probs(p, sd_margin, sd_total)

                def avg(team, col):
                    src = cur if (cur['team'] == team).sum() >= 3 else prior
                    return src.loc[src['team'] == team, col].mean()
                bhm = (avg(g.team, 'pts') + avg(g.opponent, 'opp_pts')) / 2
                bam = (avg(g.opponent, 'pts') + avg(g.team, 'opp_pts')) / 2
                out.append({'season': season, 'game_date': d, 'game_id': g.game_id,
                            'home': g.team, 'away': g.opponent,
                            'pred_home': p['pred_home'], 'pred_away': p['pred_away'],
                            'pred_margin': p['margin'], 'pred_total': p['total'],
                            'rating_margin': p['rating_margin'], 'b2b_diff': ba - bh,
                            'lineup_diff': lh_ - la_,
                            'p_home_win': pr['p_home_win'],
                            'act_margin': g.pts - g.opp_pts, 'act_total': g.pts + g.opp_pts,
                            'base_margin': bhm - bam, 'base_total': bhm + bam,
                            'season_type': g.season_type})
    return pd.DataFrame(out)


def fit_coefs(df):
    """Least-squares fit (no intercept) of the actual margin on the model's components,
    clipped to sane bounds. Returns (coefs, residual SD of the margin)."""
    X = df[['rating_margin', 'b2b_diff', 'lineup_diff']].to_numpy(dtype=float)
    y = df['act_margin'].to_numpy(dtype=float)
    use = [0, 1] + ([2] if np.abs(X[:, 2]).sum() > 0 else [])   # no player data -> no lineup term
    beta = np.linalg.lstsq(X[:, use], y, rcond=None)[0]
    coefs = dict(DEFAULT_COEFS, lineup=0.0)
    for name, b in zip(['scale', 'b2b', 'lineup'], beta):
        coefs[name] = float(np.clip(b, *COEF_BOUNDS[name]))
    pred = X @ np.array([coefs['scale'], coefs['b2b'], coefs['lineup']])
    return coefs, float((y - pred).std())


def calibrate(conn, params=None, min_games=600):
    """Fit the margin adjustment coefficients and the margin/total residual SDs from the most
    recent walk-forward predictions and store them in model_params.
    Returns (sd_margin, sd_total, n, coefs) or None."""
    tg = pd.read_sql_query("SELECT * FROM team_games", conn)
    if tg.empty:
        return None
    pg = pd.read_sql_query("SELECT * FROM player_games", conn)
    seasons = sorted(tg['season'].unique())
    # need a prior season for ratings: evaluate the last two seasons that have one
    eval_seasons = [s for s in seasons if s - 1 in seasons][-2:]
    if not eval_seasons:
        return None
    df = backtest_games(tg, eval_seasons, params, player_games=pg)
    df = df.sort_values('game_date').tail(1500)
    if len(df) < min_games:
        return None
    coefs, sd_m = fit_coefs(df)
    sd_t = float((df['act_total'] - df['pred_total']).std())
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    vals = [('game_sd_margin', sd_m), ('game_sd_total', sd_t)] + \
        [(f'game_coef_{k}', v) for k, v in coefs.items()]
    conn.executemany("INSERT OR REPLACE INTO model_params (key, value, updated_at) VALUES (?, ?, ?)",
                     [(k, v, now) for k, v in vals])
    conn.commit()
    return sd_m, sd_t, len(df), coefs


def stored_sds(conn):
    vals = dict(conn.execute("SELECT key, value FROM model_params WHERE key LIKE 'game_sd_%'").fetchall())
    return vals.get('game_sd_margin', DEFAULT_SD_MARGIN), vals.get('game_sd_total', DEFAULT_SD_TOTAL), \
        bool(vals)


def stored_coefs(conn):
    vals = dict(conn.execute("SELECT key, value FROM model_params WHERE key LIKE 'game_coef_%'").fetchall())
    return {k: vals.get(f'game_coef_{k}', v) for k, v in DEFAULT_COEFS.items()}


# ---------------------------------------------------------------------------
# Live use
# ---------------------------------------------------------------------------

class GameContext:
    def __init__(self, conn, game_date=None, params=None):
        self.conn = conn
        self.params = params or GameParams()
        self.game_date = to_date(game_date).isoformat()
        self.season = season_for_date(self.game_date)
        lo = self.season - self.params.lookback_seasons + 1
        tg = pd.read_sql_query("SELECT * FROM team_games WHERE season >= ? AND game_date < ?",
                               conn, params=(lo, self.game_date))
        if tg.empty:
            raise RuntimeError("No team games in DB. Run: python fetch_game_logs.py")
        self.ratings = build_ratings(game_rows(tg), self.game_date, self.season, self.params)
        self.sd_margin, self.sd_total, self.calibrated = stored_sds(conn)
        self.coefs = stored_coefs(conn)
        yesterday = (pd.Timestamp(self.game_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.played_yesterday = set(tg.loc[tg['game_date'] == yesterday, 'team'])
        # tonight's game is postseason once the latest game in the data is (play-in or playoffs)
        latest_type = tg.sort_values('game_date')['season_type'].iloc[-1]
        self.use_lineups = not (tg['season'].max() == self.season and latest_type not in LINEUP_SEASON_TYPES)
        pg = pd.read_sql_query("SELECT * FROM player_games WHERE season >= ? AND game_date < ?",
                               conn, params=(lo, self.game_date))
        self._rosters, self._lineup_ref = {}, {}
        if not pg.empty:
            vals = player_values(pg, self.params)
            lh = lineup_history(vals, self.params)
            self._lineup_ref = lh.groupby('team')['ref_post'].last().dropna().to_dict()
            latest = vals.groupby('player_id').tail(1).copy()
            d = pd.to_datetime(latest['game_date'])
            latest['season'] = np.where(d.dt.month >= 8, d.dt.year, d.dt.year - 1)
            for team, g in vals.groupby('team'):
                recent = g['game_id'].drop_duplicates().iloc[-self.params.roster_recent_games:]
                ids = set(g.loc[g['game_id'].isin(recent), 'player_id'])
                mine = latest[latest['team'] == team]
                # also this season's players who haven't played lately (injured, out of the
                # rotation) so they can be marked as playing when they return
                r = mine[mine['player_id'].isin(ids) | (mine['season'] == mine['season'].max())]
                self._rosters[team] = (r[['player_id', 'player_name', 'val_post', 'game_date']]
                                       .rename(columns={'val_post': 'value', 'game_date': 'last_game'})
                                       .assign(recent=r['player_id'].isin(ids).to_numpy())
                                       .sort_values(['recent', 'value'], ascending=False)
                                       .reset_index(drop=True))

    def roster(self, team):
        """The team's players, best first: player_id, player_name, value, last_game, and recent
        (True = appeared in the team's last few games, so assumed to play unless marked out).
        Empty if there is no player data."""
        r = self._rosters.get(team_abbr(team) or team)
        return r.copy() if r is not None else pd.DataFrame(
            columns=['player_id', 'player_name', 'value', 'last_game', 'recent'])

    def lineup_delta(self, team, out=(), extra=()):
        """Tonight's lineup strength vs the team's recent norm. Recent players play unless in
        `out`; players in `extra` (ids, e.g. back from injury) play too."""
        ref = self._lineup_ref.get(team)
        r = self._rosters.get(team)
        if ref is None or r is None or not self.use_lineups:
            return 0.0
        playing = (r['recent'] | r['player_id'].isin(set(extra))) & ~r['player_id'].isin(set(out))
        vals = r.loc[playing, 'value'].sort_values(ascending=False)
        return float(vals.head(self.params.lineup_size).sum() - ref)

    def evaluate(self, home, away, spread=None, total=None, home_out=(), away_out=(),
                 home_in=(), away_in=()):
        """home_out / away_out: player ids not playing tonight (everyone else who played
        recently is assumed to play). home_in / away_in: players who haven't played lately
        but will tonight."""
        home, away = team_abbr(home), team_abbr(away)
        if not home or not away:
            raise ValueError("Unknown team")
        pred = adjust_prediction(predict_game(self.ratings, home, away), self.coefs,
                                 int(home in self.played_yesterday), int(away in self.played_yesterday),
                                 self.lineup_delta(home, home_out, home_in),
                                 self.lineup_delta(away, away_out, away_in))
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
        p_np = pa / (pa + pb) if pa + pb > 0 else 0.5
        warning = None
        if abs(p_np - fair) > DISAGREE_WARN:
            warning = (f"Model and market differ by {abs(p_np - fair):.0%}. Game lines are sharp: this "
                       "almost always means injuries/rest the model can't see. Check news first.")
        out.append({'market': market, 'line': line, 'odds_a': oa, 'odds_b': ob, 'p_a': pa, 'p_b': pb,
                    'fair_a': fair, 'ev_a': ea, 'ev_b': eb, 'pick': pick, 'kelly': k,
                    'warning': warning})
    return out


def stale_warning(ctx):
    days_old = (to_date(ctx.game_date) - to_date(ctx.ratings['last_game'])).days
    if days_old > 4:
        return (f"Ratings use games through {ctx.ratings['last_game']} ({days_old} days before this game). "
                "Update the data.")
    return None


def log_game_evals(conn, game_date, pred, evals, book=None, tracked=False):
    """Save evaluated markets. tracked=True marks them as bets you're following ("Track this");
    re-logging never un-tracks a market."""
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for e in evals:
        conn.execute("""INSERT OR IGNORE INTO game_log (logged_at, game_date, home, away, market, line,
                        odds_a, odds_b, book, pred_home, pred_away, p_a, p_b, fair_a, ev_a, ev_b, pick)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (now, game_date, pred['home'], pred['away'], e['market'], e['line'],
                      int(e['odds_a']), int(e['odds_b']), book or '', pred['pred_home'], pred['pred_away'],
                      e['p_a'], e['p_b'], e['fair_a'], e['ev_a'], e['ev_b'], e['pick']))
        if tracked:
            conn.execute("UPDATE game_log SET tracked = 1 WHERE id = ?",
                         (game_log_id(conn, game_date, pred, e, book),))
    conn.commit()


def game_log_id(conn, game_date, pred, e, book=None, tracked_only=False):
    """Row id of one evaluated market in game_log (None if it isn't there / isn't tracked)."""
    row = conn.execute("SELECT id FROM game_log WHERE game_date=? AND home=? AND away=? AND market=? "
                       "AND line IS ? AND odds_a=? AND odds_b=? AND book=?"
                       + (" AND tracked=1" if tracked_only else ""),
                       (game_date, pred['home'], pred['away'], e['market'], e['line'], int(e['odds_a']),
                        int(e['odds_b']), book or '')).fetchone()
    return row[0] if row else None


def untrack_game_bet(conn, row_id):
    """Stop following one game market. The evaluation stays as calibration data."""
    n = conn.execute("UPDATE game_log SET tracked = 0 WHERE id = ?", (row_id,)).rowcount
    conn.commit()
    return n > 0


def game_result(market, line, pick, home_pts, away_pts, home, away):
    """'WIN' | 'LOSS' | 'PUSH' for a bet, or 'PASS (<side that won>)' for a pass."""
    if market == 'SPREAD':
        v = home_pts - away_pts + line
    elif market == 'TOTAL':
        v = home_pts + away_pts - line
    else:
        v = home_pts - away_pts
    side = 'A' if v > 0 else 'B' if v < 0 else 'PUSH'
    if pick == 'PASS':
        names = {'A': 'OVER' if market == 'TOTAL' else home, 'B': 'UNDER' if market == 'TOTAL' else away,
                 'PUSH': 'PUSH'}
        return f'PASS ({names[side]})'
    return 'PUSH' if side == 'PUSH' else 'WIN' if side == pick else 'LOSS'


def _write_game_result(conn, row_id, market, line, pick, hp, ap, home, away, now):
    conn.execute("UPDATE game_log SET home_pts=?, away_pts=?, result=?, settled_at=? WHERE id=?",
                 (hp, ap, game_result(market, line, pick, hp, ap, home, away), now, row_id))


def settle_games(conn):
    """Fill results for logged game bets whose games are now in team_games.
    Only unsettled rows are touched, so manual results are never overwritten."""
    rows = conn.execute("SELECT id, game_date, home, away, market, line, pick FROM game_log "
                        "WHERE result IS NULL").fetchall()
    n = 0
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for gid, d, home, away, market, line, pick in rows:
        r = conn.execute("""SELECT h.pts, a.pts FROM team_games h JOIN team_games a
                            ON a.game_id = h.game_id AND a.team = h.opponent
                            WHERE h.team = ? AND h.opponent = ? AND h.game_date = ?""",
                         (home, away, d)).fetchone()
        if not r:
            continue
        _write_game_result(conn, gid, market, line, pick, r[0], r[1], home, away, now)
        n += 1
    conn.commit()
    return n


def settle_game_manual(conn, game_date, home, away, home_pts, away_pts):
    """Enter a final score by hand: settles (or corrects) every logged market for that game."""
    if home_pts < 0 or away_pts < 0:
        raise ValueError("Scores can't be negative")
    if home_pts == away_pts:
        raise ValueError("NBA games can't end tied — enter the final score including overtime")
    rows = conn.execute("SELECT id, market, line, pick FROM game_log WHERE game_date=? AND home=? AND away=?",
                        (game_date, home, away)).fetchall()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for row_id, market, line, pick in rows:
        _write_game_result(conn, row_id, market, line, pick, int(home_pts), int(away_pts), home, away, now)
    conn.commit()
    return len(rows)


def delete_game_bets(conn, game_date, home, away):
    """Remove every logged market for one game. Returns the number of rows deleted."""
    n = conn.execute("DELETE FROM game_log WHERE game_date=? AND home=? AND away=?",
                     (game_date, home, away)).rowcount
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
    adj = adjustment_text(pred)
    if adj:
        print(f"Margin adjustments: {adj}")
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
        if e['warning']:
            print(f"    ⚠️  {e['warning']}")
    if stale_warning(ctx):
        print(f"⚠️  {stale_warning(ctx)}")
    print()


def adjustment_text(pred):
    """'BOS lineup -3.1, LAL on back-to-back +2.7' style summary of the non-rating parts of the
    home margin (positive = helps the home team). Empty when nothing notable."""
    h, a = pred['home'], pred['away']
    parts = []
    lineup = pred['adjustments']['lineup']
    if abs(lineup) >= 0.5:
        parts.append(f"lineups {lineup:+.1f} for {h}")
    if pred['b2b_home']:
        parts.append(f"{h} on a back-to-back")
    if pred['b2b_away']:
        parts.append(f"{a} on a back-to-back")
    if pred['b2b_home'] or pred['b2b_away']:
        parts[-1] += f" ({pred['adjustments']['b2b']:+.1f} for {h})"
    return ', '.join(parts)


def resolve_out(ctx, team, names):
    """Player names (as typed) -> player ids on that team's roster."""
    from db import normalize_name
    roster = ctx.roster(team)
    ids = []
    for name in names or []:
        q = normalize_name(name)
        norm = roster['player_name'].map(normalize_name)
        hit = roster[norm == q]
        if hit.empty:
            hit = roster[norm.str.contains(q, regex=False)]
        if len(hit) != 1:
            raise ValueError(f"'{name}' matches {len(hit)} players on {team_abbr(team)}'s roster")
        ids.append(int(hit['player_id'].iloc[0]))
    return ids


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
    ap.add_argument('--no-log', action='store_true', help="don't write evaluations to game_log")
    ap.add_argument('--home-out', nargs='+', default=[], metavar='PLAYER', help="home players out tonight")
    ap.add_argument('--away-out', nargs='+', default=[], metavar='PLAYER', help="away players out tonight")
    ap.add_argument('--home-in', nargs='+', default=[], metavar='PLAYER',
                    help="home players back tonight who haven't played lately")
    ap.add_argument('--away-in', nargs='+', default=[], metavar='PLAYER',
                    help="away players back tonight who haven't played lately")
    a = ap.parse_args()

    conn = connect()
    if not (a.home and a.away):
        ap.error('--home and --away are required')
    ctx = GameContext(conn, a.date)
    pred = ctx.evaluate(a.home, a.away, a.spread, a.total,
                        home_out=resolve_out(ctx, a.home, a.home_out),
                        away_out=resolve_out(ctx, a.away, a.away_out),
                        home_in=resolve_out(ctx, a.home, a.home_in),
                        away_in=resolve_out(ctx, a.away, a.away_in))
    evals = evaluate_markets(pred, a.spread, a.spread_odds if a.spread is not None else None,
                             a.total, a.total_odds if a.total is not None else None, a.ml, a.min_ev)
    print_game(pred, ctx, evals)
    if evals and not a.no_log:
        log_game_evals(conn, ctx.game_date, pred, evals)
        log_game_evals(conn, ctx.game_date, pred, [e for e in evals if e['pick'] != 'PASS'], tracked=True)


if __name__ == '__main__':
    main()
