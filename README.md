# 🏀 NBA Prop & Game Model

A local web app: pick a player or a game, type in the line and prices from your sportsbook,
and it tells you **OVER / UNDER / PASS** (or which side of a spread/total/moneyline) with the
expected value behind it. Projections come from every NBA player and team box score of the
last 3 seasons. Everything you evaluate is logged and settled automatically, so you can see
whether the model actually beats the market.


## Start it

**Mac:** double-click `start.command`. The first run installs the requirements; then the app
opens in your browser. (Closing the Terminal window it opens stops the app.)

**Any OS:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

First time: press **🔄 Update data** in the sidebar. It downloads 3 seasons from stats.nba.com
(about a minute). After that, press it once a day — it adds last night's games, settles your
logged bets, and recalibrates the game model.

## Using it

**🎯 Player prop** — choose the player (type to search), the stat, the line, both prices
(default -110 / -110), optionally the opponent and the game date. You get:

* the verdict: **BET OVER/UNDER** with EV and a ¼-Kelly stake size, or **PASS**
* **Model** probability vs **Market (no vig)** probability for each side, and EV per $1
* projection, fair line (where over/under would be 50/50), projected minutes, season average
* a chart of the full outcome distribution, coloured by over/under
* warnings when the model is probably missing something (big disagreement with the market,
  stale data, early-season sample, low-minutes player)

Use **Minutes (optional)** when you know something the model doesn't — a minutes limit, or a
starter out who will push this player's minutes up.

Stats: points, rebounds, assists, 3-pointers made, PRA, PR, PA, RA, steals, blocks,
steals+blocks, turnovers.

**🏀 Game** — choose away and home team. Leave the lines blank to just get the predicted
score and win probability, or fill in any of spread (the **home** team's line), total and
moneyline to get a pick for each.

**📒 My results** — record, profit, ROI and the model-vs-market Brier score on everything
you've logged, plus the full log. **This is the real test of edge.** Without free historical
lines nobody can backtest profit, so give it a few hundred logged props before trusting it.

The sidebar has the minimum EV to bet (default 3%) and a switch to stop saving evaluations.

### Terminal versions (optional)

```bash
python bet_signals.py "shai" PTS 31.5 --over -115 --under -105 --opp LAL
python projection_model.py "shai" PTS --opp LAL         # fair line + probability ladder
python game_model.py --home BOS --away LAL --spread -5.5 --total 221.5 --ml -230 +190
python bet_signals.py --report
python daily_update.py                                   # same as the Update data button
```

## How the models work

**Props** (`feature_engineering.py`, `projection_model.py`), using only games before the game date:

* projected minutes = recency-weighted recent minutes (half-life 5 games)
* per-minute rate for each component stat (half-life 25 games, last season down-weighted)
* opponent factor per component = what the opponent allows vs league average, shrunk toward 1
* mean = Σ minutes × rate × opponent factor (combos like PRA are built from components)
* variance = mean + α·mean² — the player's own over-dispersion, shrunk to the league value
* outcome distribution = negative binomial → exact P(over), P(under), P(push)

**Games** (`game_model.py`): possessions, offensive/defensive points per 100 and pace for each
team, recency-weighted, opponent-adjusted (SRS-style) and shrunk to league average.
Margin uses slower ratings (half-life 20 games), totals faster ones (10 games) — tuned
walk-forward. Home court is estimated from the data. Margin/total are discretized normals
whose SDs `daily_update.py` recalibrates from recent residuals.

## Backtest

```bash
python backtest.py props --seasons 2025
python backtest.py props --tune 2024 --seasons 2025       # tune on one season, report on another
python backtest.py games --seasons 2024 2025
```

Walk-forward: every prediction uses only earlier games. Results on 2023-24 (parameters tuned
on 2022-23, so this season was never used for fitting):

**Props, 2023-24** (8,064 player-games per stat; walk-forward, 10 teams' players in the test data):

| Stat | MAE model | season avg | last-10 | old 50/30/20* | Brier model | Brier old* | 80% interval hit |
|---|---|---|---|---|---|---|---|
| PTS  | **4.54** | 4.61 | 4.63 | 4.56 | **0.244** | 0.247 | 78.3% |
| REB  | **1.85** | 1.89 | 1.89 | 1.87 | **0.239** | 0.245 | 79.8% |
| AST  | **1.35** | 1.38 | 1.38 | 1.36 | **0.236** | 0.243 | 79.2% |
| 3PM  | 0.870 | 0.868 | 0.885 | 0.869 | **0.215** | 0.244 | 80.0% |
| PRA  | **6.01** | 6.17 | 6.13 | 6.06 | **0.241** | 0.244 | 80.4% |

2021-22 (also never used for tuning) looks the same: PTS MAE 4.56 vs 4.68 season average, 80% interval hit
80.0%; Brier is a tie with the old formula on PTS/PRA.
\* "old" = this repo's previous 50/30/20 formula and fixed SDs **with its date-sorting bug fixed** —
the code as it was used October games as "last 10", so it was considerably worse than shown.

Brier scores are for P(over) at a proxy line (season-to-date average rounded to x.5); 0.250 is a
coin flip. Calibration by bucket is close (e.g. predicted 0.474 → actual 0.487; 0.635 → 0.649),
with a slight under-projection of assists/PRA (−0.1 per game).

**Games, 2021-22 to 2023-24** (3,943 games):

| | Model | Season-average baseline |
|---|---|---|
| Margin MAE | **10.88** | 11.27 |
| Total MAE | **14.68** | 15.24 |
| Winner picked | 64.7% | |

Win probabilities are reasonably calibrated but slightly timid (predicted 71% → won 75%;
predicted 43% → won 40%). For reference, closing Vegas lines pick roughly 67-69% of winners
(estimate) — the market is still better than this model on sides.

**Honest read:** the projections are modestly better than simple averages and the
probabilities are well calibrated, which is a prerequisite for finding edges — not proof of
one. Books see injury news, rotations and sharp money; this model doesn't. Treat large
model/market gaps as "check the news", not "free money", and let `--report` on a few hundred
logged props tell you whether it beats the closing market.

**Known blind spots:** injuries and teammates' absences (use `--minutes`), late-season tanking
and rest (projections lag sudden role changes — April is the weakest month), trades
(opponent/team inferred from the player's latest team in the data).

## Files

| File | Purpose |
|---|---|
| `app.py`, `start.command` | The web app and its Mac launcher |
| `fetch_game_logs.py` | League-wide player + team game logs from stats.nba.com |
| `daily_update.py` | Refresh current season, settle logged bets, recalibrate game model |
| `feature_engineering.py` | Shared numeric core (rates, minutes, opponent factors, dispersion) |
| `projection_model.py` | Distributions, fair lines, projection CLI |
| `bet_signals.py` | Prop evaluator, prop log, settlement, report |
| `tracking.py` | Performance summaries for the Results tab and report |
| `game_model.py` | Team ratings, score/spread/total/moneyline model |
| `odds.py` | American odds, no-vig, EV, Kelly |
| `backtest.py` | Walk-forward backtests and parameter tuning |
| `db.py`, `schema.sql` | Database, season helpers, player/team name matching |
| `tests/` | `python -m pytest tests` |
