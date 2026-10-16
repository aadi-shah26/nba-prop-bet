#!/usr/bin/env python3
"""
Walk-forward backtests. Every prediction uses ONLY games played before that game's date
(projections, opponent factors and league priors), so there is no look-ahead.

Without historical sportsbook lines we cannot measure profit, so we measure the two things
that have to be true for the model to have any chance of beating a book:
  1. Accuracy of the mean   — MAE vs simple baselines (season avg, last-10, the old 50/30/20 formula).
  2. Calibration of P(over) — at a proxy line (season-to-date average rounded to x.5), is
     P(over)=60% right 60% of the time?  Brier score / log loss / reliability table, plus
     coverage of the 80% interval and mean log-likelihood of the whole distribution.
Your logged props (prop_log) are what eventually measure real edge vs real lines.

Usage:
    python backtest.py props --seasons 2025                 # default stats
    python backtest.py props --seasons 2024 2025 --stats PTS REB AST FG3M PRA
    python backtest.py props --tune 2024 --seasons 2025     # tune on 2024, report on 2025
    python backtest.py games --seasons 2024 2025
    python backtest.py calibrate --season 2025              # refit per-stat dispersion + P(over) fix
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize, minimize_scalar

from db import connect
from feature_engineering import (STATS, PropParams, league_priors, opponent_factors,
                                 project_from_history, team_games_with_allowed)
from projection_model import count_distribution, line_probs

DEFAULT_STATS = ['PTS', 'REB', 'AST', 'FG3M', 'PRA']

# What the repo used before: fixed SD per stat, normal curve.
OLD_SD = {'PTS': 5.5, 'AST': 2.2, 'REB': 2.8, 'BLK': 1.1, 'STL': 0.9, 'PRA': 8.0,
          'PA': 6.5, 'PR': 7.0, 'RA': 3.5}


def load(conn):
    pg = pd.read_sql_query("SELECT * FROM player_games ORDER BY game_date, game_id", conn)
    tg = pd.read_sql_query("SELECT * FROM team_games ORDER BY game_date, game_id", conn)
    return pg, tg


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------

def season_priors(pg, season):
    prior_data = pg[pg['season'] < season]
    if prior_data.empty:
        raise ValueError(f"Need at least one season before {season} for league priors")
    return league_priors(prior_data[prior_data['season'] >= season - 2])


def predict_props(pg, tga, season, stats, params, min_season_games=5, priors=None, player_ids=None):
    """One row per (player-game in `season`, stat) with model + baseline predictions.
    `player_ids` restricts to some players (for parallel runs; pass the full-league `priors`)."""
    if priors is None:
        priors = season_priors(pg, season)
    lo = season - params.lookback_seasons + 1
    pool = pg[(pg['season'] >= lo) & (pg['season'] <= season)]
    if player_ids is not None:
        pool = pool[pool['player_id'].isin(player_ids)]
    opp_cache = {}
    rows = []
    for pid, g in pool.groupby('player_id', sort=False):
        g = g.reset_index(drop=True)
        idx = np.flatnonzero(g['season'].to_numpy() == season)
        for i in idx:
            game = g.iloc[i]
            hist = g.iloc[:i]
            cur = hist[hist['season'] == season]
            if len(cur) < min_season_games:
                continue  # baselines need some current-season games; keeps comparison fair
            d = game['game_date']
            if d not in opp_cache:
                opp_cache[d] = opponent_factors(tga, d, season, params)
            opp = opp_cache[d].get(game['opponent'])
            for stat in stats:
                comps = list(STATS[stat])
                p = project_from_history(hist, stat, season, priors, params, opp_factor=opp)
                if p is None:
                    continue
                y_cur = cur[comps].sum(axis=1).to_numpy(dtype=float)
                s_avg, l10, l5 = y_cur.mean(), y_cur[-10:].mean(), y_cur[-5:].mean()
                rows.append({
                    'player_id': pid, 'game_date': d, 'stat': stat,
                    'actual': float(game[comps].sum()), 'minutes': game['minutes'],
                    'mean': p['mean'], 'var': p['var'],
                    'season_avg': s_avg, 'last10': l10,
                    'old': 0.5 * s_avg + 0.3 * l10 + 0.2 * l5,
                })
    return pd.DataFrame(rows)


def evaluate_props(df):
    out = []
    for stat, s in df.groupby('stat', sort=False):
        y = s['actual'].to_numpy()
        mu, var = s['mean'].to_numpy(), s['var'].to_numpy()
        line = np.floor(s['season_avg'].to_numpy()) + 0.5          # proxy book line
        hit = (y > line).astype(float)
        p_over = np.array([line_probs(m, v, L, stat)[0] for m, v, L in zip(mu, var, line)])
        old_sd = OLD_SD.get(stat, 5.5)
        p_old = sps.norm.cdf((s['old'].to_numpy() - line) / old_sd)
        # distribution-level checks
        # randomized PIT: uniform on [0,1] iff the predictive distribution is calibrated,
        # so exactly 80% should land in [0.1, 0.9] (exact even for discrete counts).
        rng = np.random.default_rng(0)
        ll, inside = [], []
        for m, v, a in zip(mu, var, y):
            if m <= 0:
                continue
            dist = count_distribution(m, v)
            ll.append(dist.logpmf(a))
            u = dist.cdf(a - 1) + rng.uniform() * dist.pmf(a)
            inside.append(0.1 <= u <= 0.9)
        eps = 1e-6
        out.append({
            'stat': stat, 'n': len(s),
            'mae_model': np.mean(np.abs(y - mu)),
            'mae_season': np.mean(np.abs(y - s['season_avg'])),
            'mae_last10': np.mean(np.abs(y - s['last10'])),
            'mae_old': np.mean(np.abs(y - s['old'])),
            'bias': np.mean(mu - y),
            'brier_model': np.mean((p_over - hit) ** 2),
            'brier_old': np.mean((p_old - hit) ** 2),
            'brier_coin': np.mean((0.5 - hit) ** 2),
            'logloss_model': -np.mean(hit * np.log(np.clip(p_over, eps, 1)) +
                                      (1 - hit) * np.log(np.clip(1 - p_over, eps, 1))),
            'cover80': np.mean(inside),
            'mean_ll': np.mean(ll),
        })
        df.loc[s.index, 'p_over'] = p_over
        df.loc[s.index, 'hit'] = hit
    return pd.DataFrame(out)


def reliability(df, bins=(0, .2, .3, .4, .45, .5, .55, .6, .7, .8, 1.0)):
    cut = pd.cut(df['p_over'], bins, include_lowest=True)
    return df.groupby(cut, observed=True).agg(n=('hit', 'size'), predicted=('p_over', 'mean'),
                                              actual=('hit', 'mean'))


TUNE_GRID = {
    'half_life_minutes': [3, 5, 8],
    'half_life_rate': [15, 25, 40],
    'prev_season_weight': [0.4, 0.6, 0.8],
    'opp_strength': [0.0, 0.5, 1.0],
    'disp_prior_games': [5, 10, 20],
}

_W = {}


def _tune_init(season, stats):
    pg, tg = load(connect())
    _W.update(pg=pg, tga=team_games_with_allowed(tg), season=season, stats=stats)


def _tune_score(params):
    ev = evaluate_props(predict_props(_W['pg'], _W['tga'], _W['season'], _W['stats'], params))
    return (ev['mean_ll'] * ev['n']).sum() / ev['n'].sum()


def tune_props(season, stats, jobs=4):
    """Coordinate-wise grid search scored on mean log-likelihood (a proper score for the
    whole distribution). Run it on a season you will NOT report results on."""
    from multiprocessing import Pool
    best = PropParams()
    with Pool(jobs, initializer=_tune_init, initargs=(season, stats)) as pool:
        for key, values in TUNE_GRID.items():
            cands = [replace(best, **{key: v}) for v in values]
            scores = pool.map(_tune_score, cands)
            for v, sc in zip(values, scores):
                print(f"  {key}={v:<5} mean_ll={sc:.5f}")
            best = cands[int(np.argmax(scores))]
            print(f"  -> {key} = {getattr(best, key)}")
    return best


# ---------------------------------------------------------------------------
# Calibration: per-stat dispersion (VAR_SCALE) and P(over) recalibration (P_OVER_CAL)
# ---------------------------------------------------------------------------

def _nb(mu, var):
    mu = np.maximum(mu, 1e-6)
    var = np.maximum(var, mu * (1 + 1e-6))
    n = mu ** 2 / (var - mu)
    return n, n / (n + mu)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _cal_predict(pids):
    return predict_props(_W['pg'], _W['tga'], _W['season'], _W['stats'], PropParams(var_scale=1.0),
                         priors=_W['priors'], player_ids=pids)


def fit_calibration(df):
    """df: predict_props rows made with var_scale=1.0. Returns ({stat: var_scale}, {stat: (a, b)}).
    Dispersion is fit first (log-likelihood of the whole distribution), then the logistic
    P(over) fix on top of it at the proxy line (season-to-date average rounded to x.5)."""
    scales, cal = {}, {}
    for stat, s in df.groupby('stat', sort=False):
        mu, y = s['mean'].to_numpy(), s['actual'].to_numpy()
        alpha1 = (s['var'].to_numpy() - mu) / np.maximum(mu, 1e-6) ** 2

        def nll(k):
            return -sps.nbinom.logpmf(y, *_nb(mu, mu + k * alpha1 * mu ** 2)).mean()
        k = float(minimize_scalar(nll, bounds=(0.1, 3.0), method='bounded').x)
        line = np.floor(s['season_avg'].to_numpy()) + 0.5
        x = _logit(sps.nbinom.sf(np.floor(line), *_nb(mu, mu + k * alpha1 * mu ** 2)))
        hit = (y > line).astype(float)

        def logloss(w):
            z = w[0] + w[1] * x
            return np.mean(np.logaddexp(0, z) - hit * z)
        a, b = minimize(logloss, [0.0, 1.0]).x
        scales[stat], cal[stat] = round(k, 3), (round(float(a), 3), round(float(b), 3))
    return scales, cal


def calibration_check(df, scales, cal):
    """Mean P(over) raw vs recalibrated vs actual, and Brier, at the proxy line."""
    out = []
    for stat, s in df.groupby('stat', sort=False):
        mu, y = s['mean'].to_numpy(), s['actual'].to_numpy()
        var = mu + scales[stat] * (s['var'].to_numpy() - mu)
        line = np.floor(s['season_avg'].to_numpy()) + 0.5
        p_raw = sps.nbinom.sf(np.floor(line), *_nb(mu, var))
        a, b = cal[stat]
        p_cal = 1 / (1 + np.exp(-(a + b * _logit(p_raw))))
        hit = (y > line).astype(float)
        out.append({'stat': stat, 'n': len(s), 'actual_over': hit.mean(), 'p_raw': p_raw.mean(),
                    'p_cal': p_cal.mean(), 'brier_raw': np.mean((p_raw - hit) ** 2),
                    'brier_cal': np.mean((p_cal - hit) ** 2)})
    return pd.DataFrame(out)


def calibrate_props(season, stats, jobs=4):
    """Walk-forward predictions for `season`, then fit VAR_SCALE and P_OVER_CAL on them.
    Sanity check first: fit on the first half of the season, score the second half."""
    from multiprocessing import Pool
    pg, _ = load(connect())
    pids = pg.loc[pg['season'] == season, 'player_id'].unique()
    chunks = [pids[i::jobs * 4] for i in range(jobs * 4)]
    with Pool(jobs, initializer=_tune_init_cal, initargs=(season, stats)) as pool:
        df = pd.concat(pool.map(_cal_predict, chunks), ignore_index=True)

    mid = df['game_date'].sort_values().iloc[len(df) // 2]
    first, second = df[df['game_date'] < mid], df[df['game_date'] >= mid]
    print(f"\nHold-out check: fit on games before {mid}, scored on games from {mid} on")
    print(calibration_check(second, *fit_calibration(first)).to_string(index=False,
                                                                       float_format=lambda x: f'{x:.3f}'))
    scales, cal = fit_calibration(df)
    print(f"\nFitted on all of {season}. Paste into the code:\n")
    print(f"# feature_engineering.py\nVAR_SCALE = {scales}\n")
    print(f"# projection_model.py\nP_OVER_CAL = {cal}")
    return scales, cal


def _tune_init_cal(season, stats):
    _tune_init(season, stats)
    _W['priors'] = season_priors(_W['pg'], season)


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def record_pick(tg, df, prior_games=10):
    """Baseline: pick the team with the better record before the game (regular-season games
    this season, plus `prior_games` games at last season's win% so October isn't a coin flip).
    Ties go to the home team. Returns a boolean Series: True = home picked."""
    t = tg[tg['season_type'] == 'Regular Season']
    wins = t.assign(w=(t['wl'] == 'W').astype(int)).sort_values('game_date')
    last = wins.groupby(['season', 'team'])['w'].mean().to_dict()

    def wp(team, season, d):
        cur = wins[(wins['team'] == team) & (wins['season'] == season) & (wins['game_date'] < d)]['w']
        prior = last.get((season - 1, team), 0.5)
        return (cur.sum() + prior_games * prior) / (len(cur) + prior_games)
    return pd.Series([wp(r.home, r.season, r.game_date) >= wp(r.away, r.season, r.game_date)
                      for r in df.itertuples()], index=df.index)


def run_games(tg, seasons, params=None, pg=None):
    from game_model import GameParams, backtest_games
    params = params or GameParams()
    df = backtest_games(tg, seasons, params, player_games=pg)
    df['record_pick'] = record_pick(tg, df)
    print(f"\n{'=' * 72}\nGAME MODEL — walk-forward, {len(df)} games\n{'=' * 72}")
    for name, sub in [('all', df)] + list(df.groupby('season')):
        e_m = sub['act_margin'] - sub['pred_margin']
        e_t = sub['act_total'] - sub['pred_total']
        b_m = sub['act_margin'] - sub['base_margin']
        b_t = sub['act_total'] - sub['base_total']
        brier = ((sub['p_home_win'] - (sub['act_margin'] > 0)) ** 2).mean()
        print(f"[{name}] n={len(sub)}  margin MAE {e_m.abs().mean():.2f} (baseline {b_m.abs().mean():.2f})"
              f"  sd {e_m.std():.2f} | total MAE {e_t.abs().mean():.2f} (baseline {b_t.abs().mean():.2f})"
              f"  sd {e_t.std():.2f} bias {-e_t.mean():+.2f} | Brier {brier:.4f}")
        won = sub['act_margin'] > 0
        print(f"      winner picked: model {((sub['pred_margin'] > 0) == won).mean():.1%} | "
              f"ratings only {((sub['rating_margin'] > 0) == won).mean():.1%} | "
              f"better record {(sub['record_pick'] == won).mean():.1%} | home team {won.mean():.1%}")
    cut = pd.cut(df['p_home_win'], [0, .2, .35, .5, .65, .8, 1.0])
    print("\nWin-probability calibration:")
    print(df.groupby(cut, observed=True).agg(n=('p_home_win', 'size'),
                                             predicted=('p_home_win', 'mean'),
                                             actual=('act_margin', lambda m: (m > 0).mean())
                                             ).to_string(float_format=lambda x: f'{x:.3f}'))
    if pg is None or pg.empty:
        print("\n(no player data: lineup adjustment off)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['props', 'games', 'calibrate'])
    ap.add_argument('--seasons', type=int, nargs='+')
    ap.add_argument('--season', type=int, help='season to fit on (calibrate only)')
    ap.add_argument('--stats', nargs='+', default=DEFAULT_STATS)
    ap.add_argument('--tune', type=int, help='season to tune parameters on (props only)')
    ap.add_argument('--jobs', type=int, default=4, help='parallel workers for --tune')
    ap.add_argument('--csv', help='write per-prediction rows to this CSV')
    a = ap.parse_args()
    if a.mode == 'calibrate':
        if a.season is None:
            ap.error('calibrate needs --season')
        stats = list(STATS) if a.stats == DEFAULT_STATS else a.stats
        calibrate_props(a.season, stats, jobs=a.jobs)
        return
    if not a.seasons:
        ap.error('--seasons is required')

    conn = connect()
    pg, tg = load(conn)
    if a.mode == 'games':
        df = run_games(tg, a.seasons, pg=pg)
        if a.csv:
            df.to_csv(a.csv, index=False)
        return

    tga = team_games_with_allowed(tg)
    params = PropParams()
    if a.tune is not None:
        if a.tune in a.seasons:
            raise SystemExit("Tune season must differ from evaluation seasons (no peeking).")
        print(f"🔧 Tuning on {a.tune}...")
        params = tune_props(a.tune, a.stats, jobs=a.jobs)
        print(f"Best params: {params}")

    frames = []
    for s in a.seasons:
        print(f"🔄 Backtesting {s}...")
        frames.append(predict_props(pg, tga, s, a.stats, params))
    df = pd.concat(frames, ignore_index=True)
    ev = evaluate_props(df)
    pd.set_option('display.width', 200)
    print(f"\n{'=' * 100}\nPROP MODEL — walk-forward on seasons {a.seasons}\n{'=' * 100}")
    print(ev.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("\nReliability of P(over) at proxy lines (all stats):")
    print(reliability(df).to_string(float_format=lambda x: f'{x:.3f}'))
    print("\nHow to read: mae_model should beat mae_season/last10/old; brier_model should beat "
          "brier_old and brier_coin; cover80 should be ~0.80; predicted ≈ actual in each bucket.")
    if a.csv:
        df.to_csv(a.csv, index=False)


if __name__ == '__main__':
    main()
