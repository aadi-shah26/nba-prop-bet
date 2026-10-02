# 🏀 NBA Prop & Game Model

Pulls every NBA player and team box score for the last 3 seasons, projects any player stat as a
full probability distribution, and tells you whether a line you enter is an **OVER, UNDER or
PASS** — with real expected value against the bookmaker's no-vig price. A team-rating model
does the same for **spreads, totals and moneylines**. Every evaluation is logged and settled
automatically, so you can measure whether the model actually beats the market.

No paid data is required. The Odds API is optional (free tier covers game lines and a few
prop markets per day).

## Setup

```bash
pip install -r requirements.txt
python fetch_game_logs.py          # one-time: last 3 seasons, ~6 requests to stats.nba.com
python daily_update.py             # every day: refresh, settle logged bets, recalibrate
```

Optional, for live lines: put `ODDS_API_KEY=...` in a `.env` file.

The database (`nba_data.db`) is not committed — `fetch_game_logs.py` rebuilds it in about a
minute.

## Daily use

### Props — you enter the line

```bash
python bet_signals.py                                         # interactive
python bet_signals.py "shai" PTS 31.5 --over -115 --under -105 --opp LAL
python bet_signals.py "jokic" PRA 52.5 --opp MIN --minutes 30   # known minutes limit
```

```
🎯 Shai Gilgeous-Alexander (OKC) PTS 31.5 vs LAL   [2024-03-01]
Projection:   30.92 ± 7.78    fair line 30.5    minutes 34.2
          odds    model   market       EV
OVER      -115    44.5%    51.1%   -16.7%
UNDER     -105    55.5%    48.9%    +8.3%
➜ BET UNDER   ¼-Kelly stake: 2.2% of bankroll
```

* **model** — probability from the player's projected distribution (pushes handled exactly on whole-number lines)
* **market** — the book's probability with its margin removed (needs both prices; default -110/-110)
* **EV** — expected profit per $1 at the offered price
* Bets only when EV ≥ +3% (`--min-ev`). Warnings flag likely missing information (big
  model/market disagreement, stale data, small samples, low-minute players).

Stats: `PTS REB AST STL BLK TOV FG3M PRA PR PA RA SB` (aliases like `3PM`, `P+R+A` work).

### What should the line be?

```bash
python projection_model.py "shai" PTS --opp LAL     # projection, fair line, P(over) ladder
```

### Games — score prediction, spreads, totals

```bash
python game_model.py --home BOS --away LAL                          # predicted score + win prob
python game_model.py --home BOS --away LAL --spread -5.5 --total 221.5 --ml -230 +190
python game_model.py --slate                                        # today's games, Odds API lines (3 credits)
```

`--spread` is always the **home** team's line.

### Automated props from The Odds API

```bash
python generate_daily_bets.py                                       # player_points, all US books
python generate_daily_bets.py --markets player_points player_rebounds --max-events 3
```

Props cost **1 credit per market per game** (all US books are included in that, so the best
price across books is found for free). A 10-game night with one market ≈ 10 credits. Output
goes to `daily_bets/props_<date>.csv` and the log.

### How am I doing?

```bash
python bet_signals.py --report
```

Record, ROI and — most importantly — whether the model's probabilities beat the market's
no-vig probabilities (Brier score) on everything you've logged. **This is the only real test of
edge**; the backtest below can't use historical lines because none are freely available.

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
| `fetch_game_logs.py` | League-wide player + team game logs from stats.nba.com |
| `daily_update.py` | Refresh current season, settle logged bets, recalibrate game model |
| `feature_engineering.py` | Shared numeric core (rates, minutes, opponent factors, dispersion) |
| `projection_model.py` | Distributions, fair lines, projection CLI |
| `bet_signals.py` | Prop evaluator, prop log, settlement, report |
| `game_model.py` | Team ratings, score/spread/total/moneyline model |
| `odds.py` | American odds, no-vig, EV, Kelly |
| `odds_fetcher.py` | The Odds API client (props via per-event endpoint, game lines) |
| `generate_daily_bets.py` | Automated prop slate from The Odds API |
| `backtest.py` | Walk-forward backtests and parameter tuning |
| `db.py`, `schema.sql` | Database, season helpers, player/team name matching |
| `tests/` | `python -m pytest tests` |
