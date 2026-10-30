"""Do lineup changes and rest explain the total / margin residuals? Fit on 2024-25, score 2025-26."""
import numpy as np
import pandas as pd
from preds import build

df = build()
df = df[df['season_type'] == 'Regular Season'].copy()
df['lin_sum'] = df['delta_h'] + df['delta_a']
df['pts_sum'] = df['pts_delta_h'] + df['pts_delta_a']
df['b2b_sum'] = df['b2b_diff'].abs() * 0 + (df['rest_h'] == 1).astype(int) + (df['rest_a'] == 1).astype(int)
df['t34_sum'] = df['three_in_four_h'] + df['three_in_four_a']
df['t34_diff'] = df['three_in_four_a'] - df['three_in_four_h']
df['rest_diff'] = df['rest_h'] - df['rest_a']
df['res_total'] = df['act_total'] - df['pred_total']
df['res_margin'] = df['act_margin'] - df['pred_margin']
tr, te = df[df['season'] == 2024], df[df['season'] == 2025]


def fit(target, cols):
    X = tr[cols].to_numpy(float); y = tr[target].to_numpy(float)
    b = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
    pred = np.c_[np.ones(len(te)), te[cols].to_numpy(float)] @ b
    r0, r1 = te[target], te[target] - pred
    return b, np.sqrt((r0 ** 2).mean()), np.sqrt((r1 ** 2).mean()), r0.abs().mean(), r1.abs().mean()


print("TOTAL residual (actual - model). Test-season RMSE / MAE before -> after")
for cols in (['lin_sum'], ['pts_sum'], ['b2b_sum'], ['t34_sum'], ['pts_sum', 'b2b_sum'], ['lin_sum', 'b2b_sum', 't34_sum']):
    b, r0, r1, m0, m1 = fit('res_total', cols)
    print(f"  {'+'.join(cols):28} coefs {np.round(b, 2)}  RMSE {r0:.2f} -> {r1:.2f}  MAE {m0:.2f} -> {m1:.2f}")
print("MARGIN residual. Test RMSE / MAE")
for cols in (['t34_diff'], ['rest_diff'], ['t34_diff', 'rest_diff']):
    b, r0, r1, m0, m1 = fit('res_margin', cols)
    print(f"  {'+'.join(cols):28} coefs {np.round(b, 2)}  RMSE {r0:.2f} -> {r1:.2f}  MAE {m0:.2f} -> {m1:.2f}")

print("\nBy games played (min of the two teams), test season:")
te = te.assign(gp=np.minimum(te['gp_h'], te['gp_a']))
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 90)):
    s = te[(te.gp >= lo) & (te.gp < hi)]
    print(f"  gp {lo:2}-{hi:2}: n={len(s):4}  total bias {s.res_total.mean():+5.1f} MAE {s.res_total.abs().mean():5.1f} | "
          f"margin bias {s.res_margin.mean():+5.1f} MAE {s.res_margin.abs().mean():5.1f}")
