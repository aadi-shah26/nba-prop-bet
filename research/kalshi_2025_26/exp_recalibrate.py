"""Recalibrate the model's probabilities (logistic on logit(p_model), per market) on Oct-Jan and
score Feb-Jun: does a calibrated model add anything beyond Kalshi?"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import market_backtest as mb  # noqa: E402
from db import connect  # noqa: E402

conn = connect()
preds, sd_m, sd_t = mb.walk_forward(conn, 2025)
df = mb.price_contracts(mb.load_history(conn, 2025), preds, sd_m, sd_t)
df = df[df['p_market'].between(0.03, 0.97)]
fit, test = df[df['game_date'] < '2026-02-01'], df[df['game_date'] >= '2026-02-01']
for market in mb.MARKETS:
    a, b = fit[fit.market == market], test[test.market == market].copy()
    y = a['result'].to_numpy(float)
    X = np.c_[np.ones(len(a)), mb._logit(a['p_model'].to_numpy())]
    w = minimize(lambda c: np.sum(np.logaddexp(0, X @ c) - y * (X @ c)), [0, 1]).x
    b['p_cal'] = 1 / (1 + np.exp(-(w[0] + w[1] * mb._logit(b['p_model'].to_numpy()))))
    yt = b['result'].to_numpy(float)
    br = lambda p: np.mean((p - yt) ** 2)
    ws = np.linspace(0, 1, 21)
    best = ws[int(np.argmin([br(x * b.p_cal + (1 - x) * b.p_market) for x in ws]))]
    print(f"{market:6} slope {w[1]:.2f}  test Brier: Kalshi {br(b.p_market):.4f}  model {br(b.p_model):.4f}  "
          f"calibrated {br(b.p_cal):.4f}  weight {mb.model_weight(yt, b.p_market.to_numpy(), b.p_cal.to_numpy())[2]:+.3f}  "
          f"best blend {best:.2f}")
