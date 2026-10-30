#!/usr/bin/env python3
"""
Does the game model know anything Kalshi doesn't? Walk-forward test of the model against the
prices Kalshi really charged (kalshi_history.py fetches them).

Every game of the season is predicted from earlier games only, with the SDs and margin
coefficients fit on the season before (nothing is tuned on the season it is graded on). Each
Kalshi contract then gets three probabilities: Kalshi's mid an hour before tip-off, the model's,
and the bot's blend of the two. Reported per market:
    Brier            lower is better; the model has to beat Kalshi's here to be worth anything
    model weight     logistic fit of the result on logit(Kalshi) and logit(model): the model's
                     coefficient is ~0 when it adds nothing beyond Kalshi's price, negative
                     when leaning on it makes things worse
    best blend       the blend weight with the lowest Brier
and a simulation of the bot's rules (kalshi_bot.candidates, the game flag, the best contract
per game and market) at the pre-game ask, one flat $10 stake per bet: bets, ROI with a 90%
bootstrap range over games, and closing-line value.

Caveat: the backtest's lineup term uses who actually played, which the live bot only knows from
the injury report (late scratches are missed live), so the model looks a little better here
than it can be live.

    python market_backtest.py                       # 2025-26, current bot settings
    python market_backtest.py --blend 0.25 --one-per-game
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

import game_model as gm
from db import connect
from kalshi_bot import BotConfig, candidates, game_flag, market_centers, model_prob

STAKE = 10.0
MARKETS = ('WINNER', 'SPREAD', 'TOTAL')


def load_history(conn, season):
    lo, hi = f'{season}-08-01', f'{season + 1}-08-01'
    return pd.read_sql_query("SELECT * FROM kalshi_history WHERE game_date >= ? AND game_date < ?",
                             conn, params=(lo, hi))


def walk_forward(conn, season, params=None, coefs=None):
    """(predictions for `season`, sd_margin, sd_total). SDs come from the season before, and so do
    the margin coefficients unless given."""
    params = params or gm.GameParams()
    tg = pd.read_sql_query("SELECT * FROM team_games", conn)
    pg = pd.read_sql_query("SELECT * FROM player_games", conn)
    prev = gm.backtest_games(tg, [season - 1], params, player_games=pg, coefs=coefs)
    prev = prev[prev['season_type'] == 'Regular Season']
    if coefs is None:
        coefs, _ = gm.fit_coefs(prev)
        prev = gm.backtest_games(tg, [season - 1], params, player_games=pg, coefs=coefs)
    sd_m = gm.fit_win_sd(prev['pred_margin'].to_numpy(), prev['act_margin'].to_numpy() > 0)
    sd_t = float((prev['act_total'] - prev['pred_total']).std())
    return gm.backtest_games(tg, [season], params, sd_m, sd_t, player_games=pg, coefs=coefs), sd_m, sd_t


def price_contracts(hist, preds, sd_margin, sd_total):
    """hist rows with p_market (pre-game mid), p_close (closing mid) and p_model for YES."""
    by_game = {(p['game_date'], p['home'], p['away']): p for p in preds.to_dict('records')}
    h = hist[hist['bet_bid'].notna()].copy()
    keys = list(zip(h['game_date'], h['home'], h['away']))
    h = h[[k in by_game for k in keys]].copy()
    pred = [by_game[k] for k in zip(h['game_date'], h['home'], h['away'])]
    h['pred_margin'] = [p['pred_margin'] for p in pred]
    h['pred_total'] = [p['pred_total'] for p in pred]
    h['p_model'] = [model_prob(c, {'margin': p['pred_margin'], 'total': p['pred_total'],
                                   'p_home_win': p['p_home_win']}, sd_margin, sd_total)
                    for c, p in zip(h.to_dict('records'), pred)]
    h['p_market'] = (h['bet_bid'] + h['bet_ask']) / 2
    h['p_close'] = (h['close_bid'] + h['close_ask']) / 2
    return h.reset_index(drop=True)


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def model_weight(y, p_market, p_model):
    """Coefficients (intercept, Kalshi, model) of a logistic fit of y on the two logits."""
    from scipy.optimize import minimize
    X = np.c_[np.ones(len(y)), _logit(p_market), _logit(p_model)]
    nll = lambda b: np.sum(np.logaddexp(0, X @ b) - y * (X @ b))
    return minimize(nll, [0.0, 1.0, 0.0], method='BFGS').x


def accuracy(df, blend):
    """Brier of Kalshi / model / blend, the model's logistic weight and the best blend, by market.
    Contracts priced 3-97 cents (where the bot can bet)."""
    out = []
    d = df[df['p_market'].between(0.03, 0.97)]
    for market in MARKETS + ('ALL',):
        s = d if market == 'ALL' else d[d['market'] == market]
        if s.empty:
            continue
        y, pk, pm = s['result'].to_numpy(float), s['p_market'].to_numpy(), s['p_model'].to_numpy()
        ws = np.linspace(0, 1, 21)
        briers = [np.mean((w * pm + (1 - w) * pk - y) ** 2) for w in ws]
        out.append({'market': market, 'contracts': len(s), 'games': s[['game_date', 'home']].drop_duplicates().shape[0],
                    'brier_kalshi': np.mean((pk - y) ** 2), 'brier_model': np.mean((pm - y) ** 2),
                    'brier_blend': np.mean((blend * pm + (1 - blend) * pk - y) ** 2),
                    'model_weight': model_weight(y, pk, pm)[2], 'best_blend': ws[int(np.argmin(briers))]})
    return pd.DataFrame(out)


def simulate(df, sd_margin, sd_total, cfg, one_per_game=False):
    """The bot's picks at the pre-game prices: the best unflagged candidate per game and market
    (or per game). One row per bet with profit on a flat STAKE and CLV in cents."""
    picks = []
    for (d, home, away), g in df.groupby(['game_date', 'home', 'away']):
        cs = [{**c, 'yes_bid': c['bet_bid'], 'yes_ask': c['bet_ask'], 'no_ask': round(1 - c['bet_bid'], 4)}
              for c in g.to_dict('records') if c['market'] in cfg.markets]
        centers = market_centers(cs, sd_margin, sd_total).get((home, away))
        pred = {'margin': g['pred_margin'].iloc[0], 'total': g['pred_total'].iloc[0]}
        best = {}
        for c in cs:
            flag = game_flag(pred, centers, c['market'], sd_margin, sd_total, cfg)
            for x in candidates(c, c['p_model'], cfg):
                if x['skip'] is None and not flag:
                    k = (d, home, away) if one_per_game else (d, home, away, c['market'])
                    if k not in best or x['ev'] > best[k]['ev']:
                        best[k] = x
        picks += best.values()
    if not picks:
        return pd.DataFrame()
    b = pd.DataFrame(picks)
    won = np.where(b['side'] == 'yes', b['result'] == 1, b['result'] == 0)
    cost_each = b['ask'] + 0.07 * b['ask'] * (1 - b['ask'])
    b['won'] = won
    b['profit'] = STAKE * (won / cost_each - 1)
    close = np.where(b['side'] == 'yes', b['p_close'], 1 - b['p_close'])
    b['clv'] = (close - b['ask']) * 100
    return b


def bootstrap_roi(bets, n=2000, seed=0):
    """90% range of ROI, resampling games."""
    rng = np.random.default_rng(seed)
    g = bets.groupby(['game_date', 'home'])['profit'].agg(['sum', 'count'])
    s, c = g['sum'].to_numpy(), g['count'].to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    roi = s[idx].sum(1) / (STAKE * c[idx].sum(1))
    return np.percentile(roi, [5, 95])


def bet_summary(bets):
    rows = []
    for market, s in list(bets.groupby('market')) + [('ALL', bets)]:
        lo, hi = bootstrap_roi(s)
        rows.append({'market': market, 'bets': len(s), 'win%': s['won'].mean() * 100,
                     'ROI%': s['profit'].sum() / (STAKE * len(s)) * 100, 'ROI 90% lo': lo * 100,
                     'ROI 90% hi': hi * 100, 'CLV¢': s['clv'].mean(), 'avg EV¢': s['ev'].mean() * 100})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2025)
    ap.add_argument('--blend', type=float, default=BotConfig.blend)
    ap.add_argument('--max-disagree', type=float, default=BotConfig.max_disagree)
    ap.add_argument('--one-per-game', action='store_true', help="at most one bet per game")
    args = ap.parse_args()
    conn = connect()
    hist = load_history(conn, args.season)
    if hist.empty:
        raise SystemExit("No Kalshi history for that season. Run: python kalshi_history.py")
    preds, sd_m, sd_t = walk_forward(conn, args.season)
    df = price_contracts(hist, preds, sd_m, sd_t)
    cfg = replace(BotConfig(), blend=args.blend, max_disagree=args.max_disagree)
    pd.set_option('display.width', 160)
    print(f"{args.season}-{(args.season + 1) % 100:02d}: SD margin {sd_m:.1f}, total {sd_t:.1f} (fit on the season before)\n")
    print(accuracy(df, args.blend).round(4).to_string(index=False))
    bets = simulate(df, sd_m, sd_t, cfg, args.one_per_game)
    print(f"\nBot rules (blend {args.blend}, flag above {args.max_disagree:.0%}"
          f"{', one bet per game' if args.one_per_game else ''}), ${STAKE:.0f} per bet:")
    print(bet_summary(bets).round(2).to_string(index=False) if not bets.empty else "No bets.")


if __name__ == '__main__':
    main()
