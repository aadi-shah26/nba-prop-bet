"""The 2025-26 playoffs looked better for the model than the regular season (81 games). Check
on the 2024-25 playoffs, the only other Kalshi playoffs (winner contracts only). Coefficients
and SDs are fit on the 2024-25 regular season, so the playoffs themselves are out of sample."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import game_model as gm  # noqa: E402
import market_backtest as mb  # noqa: E402
from db import connect  # noqa: E402
from kalshi_bot import BotConfig  # noqa: E402

conn = connect()
tg = pd.read_sql_query("SELECT * FROM team_games", conn)
pg = pd.read_sql_query("SELECT * FROM player_games", conn)
reg = gm.backtest_games(tg, [2024], player_games=pg)
reg = reg[reg['season_type'] == 'Regular Season']
coefs, _ = gm.fit_coefs(reg)
reg = gm.backtest_games(tg, [2024], player_games=pg, coefs=coefs)
reg = reg[reg['season_type'] == 'Regular Season']
sd_m = gm.fit_win_sd(reg['pred_margin'].to_numpy(), reg['act_margin'].to_numpy() > 0)
sd_t = float((reg['act_total'] - reg['pred_total']).std())
preds = gm.backtest_games(tg, [2024], gm.GameParams(), sd_m, sd_t, player_games=pg, coefs=coefs)
preds = preds[preds['season_type'] != 'Regular Season']
df = mb.price_contracts(mb.load_history(conn, 2024), preds, sd_m, sd_t)
print(mb.accuracy(df, 0.5).round(4).to_string(index=False))
b = mb.simulate(df, sd_m, sd_t, BotConfig())
print(mb.bet_summary(b).round(2).to_string(index=False) if not b.empty else 'no bets')
