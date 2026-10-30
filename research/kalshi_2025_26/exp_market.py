"""The model against Kalshi's real 2025-26 prices: accuracy, where the model helps or hurts,
and the bot's rules over a grid of settings. Run after kalshi_history.py has the season."""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import game_model as gm  # noqa: E402
import market_backtest as mb  # noqa: E402
from db import connect  # noqa: E402
from kalshi_bot import BotConfig  # noqa: E402

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
conn = connect()
hist = mb.load_history(conn, 2025)
print(f"{len(hist)} contracts, {hist[['game_date', 'home']].drop_duplicates().shape[0]} games, "
      f"{hist['game_date'].min()} to {hist['game_date'].max()}\n")

# A separate, lower last-season weight for the total ratings (0.15 vs 0.4) was tried here too: it cut
# early-season total error by ~0.2 points against results but changed nothing against Kalshi
# (total Brier 0.2197 -> 0.2196), so it was dropped.
variants = {'current': gm.GameParams()}
priced = {}
for name, params in variants.items():
    preds, sd_m, sd_t = mb.walk_forward(conn, 2025, params)
    priced[name] = (mb.price_contracts(hist, preds, sd_m, sd_t), sd_m, sd_t)
    print(f"== {name}: SD margin {sd_m:.2f}, total {sd_t:.2f}")
    print(mb.accuracy(priced[name][0], 0.5).round(4).to_string(index=False), '\n')

df, sd_m, sd_t = priced['current']
d = df[df['p_market'].between(0.03, 0.97)].copy()
d['month'] = d['game_date'].str[:7]

print("== Is Kalshi itself calibrated? (pre-game mid vs how often YES won)")
d['bucket'] = pd.cut(d['p_market'], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
cal = d.groupby(['market', 'bucket'], observed=True).agg(n=('result', 'size'), kalshi=('p_market', 'mean'),
                                                        model=('p_model', 'mean'), actual=('result', 'mean'))
print(cal.round(3).to_string(), '\n')

print("== Brier by month (model - Kalshi; negative = model better)")
m = d.groupby(['month', 'market']).apply(lambda s: pd.Series({
    'n': len(s), 'diff': ((s.p_model - s.result) ** 2).mean() - ((s.p_market - s.result) ** 2).mean()}),
    include_groups=False).unstack('market')
print(m.round(4).to_string(), '\n')

print("== Implied lines: model vs Kalshi center, and who was closer to the result")
cen = []
for (gd, home, away), g in df.groupby(['game_date', 'home', 'away']):
    cs = [{**c, 'yes_bid': c['bet_bid'], 'yes_ask': c['bet_ask']} for c in g.to_dict('records')]
    c = mb.market_centers(cs, sd_m, sd_t).get((home, away), {})
    cen.append({'game_date': gd, 'home': home, 'k_margin': c.get('margin'), 'k_total': c.get('total'),
                'm_margin': g['pred_margin'].iloc[0], 'm_total': g['pred_total'].iloc[0]})
cen = pd.DataFrame(cen)
res = pd.read_sql_query("SELECT h.game_date, h.team AS home, h.pts - a.pts AS act_margin, h.pts + a.pts AS act_total "
                        "FROM team_games h JOIN team_games a ON a.game_id = h.game_id AND a.home = 0 WHERE h.home = 1", conn)
cen = cen.merge(res, on=['game_date', 'home']).dropna()
for k in ('margin', 'total'):
    e_k, e_m = cen[f'act_{k}'] - cen[f'k_{k}'], cen[f'act_{k}'] - cen[f'm_{k}']
    gap = cen[f'm_{k}'] - cen[f'k_{k}']
    # when the model is above Kalshi, how much of the gap shows up in the result?
    b = np.polyfit(gap, cen[f'act_{k}'] - cen[f'k_{k}'], 1)[0]
    print(f"  {k:6}: games {len(cen)}  MAE Kalshi {e_k.abs().mean():.2f}  model {e_m.abs().mean():.2f}  "
          f"model-Kalshi mean {gap.mean():+.2f} (sd {gap.std():.2f})  share of the gap that came true {b:+.2f}")
print()

print("== Bot rules over a grid (pre-game ask, $10 a bet)")
rows = []
for blend in (0.5, 0.25, 0.1):
    for dis in (0.15, 0.10, 0.06):
        for one in (False, True):
            cfg = replace(BotConfig(), blend=blend, max_disagree=dis)
            b = mb.simulate(df, sd_m, sd_t, cfg, one)
            if b.empty:
                continue
            lo, hi = mb.bootstrap_roi(b)
            rows.append({'blend': blend, 'flag above': dis, 'one/game': one, 'bets': len(b),
                         'ROI%': b.profit.sum() / (mb.STAKE * len(b)) * 100, 'lo': lo * 100, 'hi': hi * 100,
                         'CLV¢': b.clv.mean(), 'EV¢': b.ev.mean() * 100})
print(pd.DataFrame(rows).round(2).to_string(index=False), '\n')

print("== Current settings, by market and by month")
b = mb.simulate(df, sd_m, sd_t, BotConfig())
print(mb.bet_summary(b).round(2).to_string(index=False))
b['month'] = b['game_date'].str[:7]
print(b.groupby('month').agg(bets=('profit', 'size'), roi=('profit', lambda p: p.sum() / (mb.STAKE * len(p)) * 100),
                             clv=('clv', 'mean')).round(2).to_string())
b['gap'] = (b['p_model'] - b['p_market']).abs()
print(b.groupby(pd.cut(b['gap'], [0, .06, .09, .12, .15]), observed=True).agg(
    bets=('profit', 'size'), roi=('profit', lambda p: p.sum() / (mb.STAKE * len(p)) * 100), clv=('clv', 'mean')).round(2))
