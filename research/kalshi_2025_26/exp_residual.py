"""Does anything the model knows predict Kalshi's miss (result - Kalshi's implied line)? If a
feature has no out-of-sample pull on that residual, it can't be an edge."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import market_backtest as mb  # noqa: E402
from db import connect  # noqa: E402
from preds import extras, load  # noqa: E402
import game_model as gm  # noqa: E402

conn = connect()
hist = mb.load_history(conn, 2025)
preds, sd_m, sd_t = mb.walk_forward(conn, 2025)
df = mb.price_contracts(hist, preds, sd_m, sd_t)
cen = []
for (gd, home, away), g in df.groupby(['game_date', 'home', 'away']):
    cs = [{**c, 'yes_bid': c['bet_bid'], 'yes_ask': c['bet_ask']} for c in g.to_dict('records')]
    c = mb.market_centers(cs, sd_m, sd_t).get((home, away), {})
    cen.append({'game_date': gd, 'home': home, 'k_margin': c.get('margin'), 'k_total': c.get('total')})
x = preds.merge(pd.DataFrame(cen), on=['game_date', 'home']).dropna(subset=['k_margin', 'k_total'])
tg, pg = load()
e = extras(tg, pg, gm.GameParams()).drop(columns=['dt', 'game_date', 'season'])
for side, team in (('h', 'home'), ('a', 'away')):
    x = x.merge(e.add_suffix('_' + side).rename(columns={f'game_id_{side}': 'game_id', f'team_{side}': team}),
                on=['game_id', team], how='left')
x['gap_margin'] = x['pred_margin'] - x['k_margin']
x['gap_total'] = x['pred_total'] - x['k_total']
x['res_margin'] = x['act_margin'] - x['k_margin']
x['res_total'] = x['act_total'] - x['k_total']
x['rest_diff'] = x['rest_h'] - x['rest_a']
x['t34_diff'] = x['three_in_four_a'] - x['three_in_four_h']
x['lin_sum'] = x['delta_h'] + x['delta_a']
x['b2b_sum'] = (x['rest_h'] == 1).astype(int) + (x['rest_a'] == 1).astype(int)
x['early'] = (np.minimum(x['gp_h'], x['gp_a']) < 15).astype(int)
x['gap_margin_early'] = x['gap_margin'] * x['early']
x['gap_total_early'] = x['gap_total'] * x['early']
print(f"{len(x)} games with a Kalshi line\n")


def ols(y, cols):
    X = np.c_[np.ones(len(x)), x[cols].to_numpy(float)]
    b, *_ = np.linalg.lstsq(X, x[y].to_numpy(float), rcond=None)
    r = x[y].to_numpy(float) - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var(ddof=X.shape[1]))
    return pd.DataFrame({'coef': b, 't': b / se}, index=['const'] + cols)


print("Margin: actual - Kalshi line, on what the model knows (t above ~2 would matter)")
print(ols('res_margin', ['gap_margin', 'gap_margin_early', 'b2b_diff', 'lineup_diff', 'rest_diff', 't34_diff']).round(3), '\n')
print("Total: actual - Kalshi line")
print(ols('res_total', ['gap_total', 'gap_total_early', 'lin_sum', 'b2b_sum']).round(3))
