# NBA Prop & Game Model

A local web app: pick a player or a game, type in the line and prices from your sportsbook,
and it tells you **OVER / UNDER / PASS** (or which side of a spread/total/moneyline) with the
expected value behind it. Projections come from every NBA player and team box score of the
last 3 seasons. Press **Track this** on any pick and it settles itself after the game, so you
can see whether the model was right and whether it actually beats the market.


## Start it

**Mac:** double-click `start.command`. The first run installs the requirements; then the app
opens in your browser. (Closing the Terminal window it opens stops the app.)

**Any OS:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Opening the app pulls the latest games from stats.nba.com by itself, in the background: the
page shows your saved data straight away and refreshes on its own once new games are in (the
sidebar says when a pull is running). It adds last night's games, settles your tracked bets and
recalibrates the game model, so the data is current to today without pressing anything. It
only pulls when something can have changed: a scheduled game has finished since the last load,
the schedule is over 6 hours old, or the rosters are over 20 hours old. It also runs at most
every 30 minutes. **Update data** in the sidebar refreshes everything on demand. The first run
downloads 3 seasons (about a minute). If stats.nba.com can't be reached, the app says so and
uses the saved data. To turn the automatic pull off, set `NBA_AUTO_UPDATE=0`.

## Using it

**Player props** — choose the player (type to search), the stat, the line, both prices
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

**Games** — the day's games are listed at the top (from the NBA schedule, refreshed
automatically), each with the model's predicted winner. **Track pick** follows that pick in
Tracked bets with no betting line needed. **Open** replaces the list with that game (**All
games** goes back), or add a custom matchup. Leave the lines blank to just get the
predicted score and win probability, or fill in any of spread (the **home** team's line), total
and moneyline to get a pick for each. Once a game is open, a **who's playing tonight** checklist
appears for each team (best players first). Everyone from the team's last 10 games starts
ticked, except players today's injury report (ESPN) lists as **Out**, who start unticked and are
also left out of the predictions on the game cards; *Day-To-Day* players are flagged but stay
ticked. Check for late scratches. Players who haven't played lately
(injured, out of the rotation) start unticked, so tick anyone returning. The result shows the
lineup and back-to-back adjustments it applied.

**Track this** — under every prop and game-line result, and under an opened game's prediction
(the model's pick, no line). Tracking a PASS follows the side the model leaned to. Tracking
anything switches to the Tracked bets tab. Press **Untrack** to undo.

**Tracked bets** — everything you tracked, newest first, filterable by *Pending* / *Finished*.
**Add a game or prop to track** at the top tracks one without opening it first. Each row shows
the bet, the model's chance and EV, the actual stat or final score, and Hit / Miss / Push / Void
(DNP) with the profit in units. Tick rows to **Remove** them (stops tracking, keeps the
evaluation for calibration) or **Delete** them for good. A winner *Pick* has no odds, so it
counts toward the hit rate but not profit. *Awaiting result* means
the game is over but its box score isn't in yet; **Check for results** pulls it. Above each
table: how often the model's side hit, plus record, profit and ROI on the actual bets (a tracked
PASS counts toward the hit rate but not the profit). The expander at the bottom lets you enter or
fix a result by hand.

The sidebar switch **Record every evaluation for calibration** (on by default) also quietly
saves what you *don't* track. The model-vs-market Brier score uses all of it, because grading
only the bets you chose to track would flatter or punish the model unfairly. **That comparison
is the real test of edge.** Without free historical lines nobody can backtest profit, so give
it a few hundred props before trusting it.

### Optional: live lines from The Odds API

Typing lines in always works. If you'd rather pull them from the books, get a key at
[the-odds-api.com](https://the-odds-api.com) and put it in a `.env` file in this folder
(git-ignored) or in your environment:

```
ODDS_API_KEY=your_key_here
```

Each tab then shows a **Import … from The Odds API** panel: pick a date and a book
(DraftKings, FanDuel, BetMGM, Caesars, …). The prop panel loads the chosen date's games by
itself (today by default, and it's free); game lines load when you press the button because
they cost credits. Then evaluate any line in one click.
Imported evaluations go through exactly the same model and are logged with the book's name.
Credits: listing games is free, a whole slate of spreads/totals/moneylines is 3 credits, and
props cost 1 credit per market per game. The panel shows your remaining credits after each call.
Without a key, nothing changes.

The sidebar has the minimum EV to bet (default 3%) and a switch to stop saving evaluations.

### Terminal versions (optional)

```bash
python bet_signals.py "shai" PTS 31.5 --over -115 --under -105 --opp LAL
python projection_model.py "shai" PTS --opp LAL         # fair line + probability ladder
python game_model.py --home BOS --away LAL --spread -5.5 --total 221.5 --ml -230 +190
python game_model.py --home BOS --away LAL --home-out "Jayson Tatum" --away-out "LeBron"
python game_model.py --home OKC --away DEN --home-in "Jalen Williams"   # back from injury
python bet_signals.py --report
python daily_update.py                                   # same as the Update data button
python odds_fetcher.py games --book fanduel              # optional: today's lines (needs ODDS_API_KEY)
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
walk-forward. Home court is estimated from the data. The rating margin is then adjusted for:

* **lineups**: each player's value is his recency-weighted Game Score; a team's strength is
  its top 8 players who play. Tonight's strength minus the team's recent norm (weighted like the
  ratings) moves the margin, so a missing star, or a star back from injury, counts. Off in the
  playoffs, where it made picks worse in testing.
* **back-to-backs**: a team that played yesterday loses ~1-3 points.
* **scale**: shrunk ratings understate real gaps between teams, so the margin is stretched ~1.15-1.2×.

`daily_update.py` refits those three coefficients and the margin/total SDs (margin and total
are discretized normals) from the most recent ~1,500 walk-forward predictions.

## Backtest

```bash
python backtest.py props --seasons 2025
python backtest.py props --tune 2024 --seasons 2025       # tune on one season, report on another
python backtest.py games --seasons 2024 2025
python backtest.py calibrate --season 2025              # each offseason: refit per-stat spread + P(over) fix, paste output into code
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

**Winner picks, 2025-26** (1,322 games incl. playoffs; adjustment coefficients fitted on 2024-25
only, so this season is a true hold-out; data from ESPN box scores):

| | All games | Regular season | Playoffs + play-in |
|---|---|---|---|
| **Model** (ratings + lineups + back-to-backs + scale) | **69.4%** | **69.7%** | 64.8% |
| Ratings only (the previous model) | 68.4% | 68.6% | 64.8% |
| Team with the better record | 66.3% | 67.1% | 54.9% |
| Home team | 55.5% | 55.5% | 56.0% |

Margin MAE 11.20 vs 11.50 for ratings only, Brier 0.203 vs 0.207. The +1-point gain in
winners is within noise for one season (95% CI roughly −1 to +3 points); it was also +1.6 on
2024-25, the fitting season. The DraftKings favourite won 68.5% of the 2025-26 games it had a
line for, and its probabilities are still sharper than the model's (log loss 0.578 vs 0.594):
the market is still better than this model on sides.

**Honest read:** the projections are modestly better than simple averages and the
probabilities are well calibrated, which is a prerequisite for finding edges — not proof of
one. Books see injury news, rotations and sharp money; this model doesn't. Treat large
model/market gaps as "check the news", not "free money", and let `--report` on a few hundred
logged props tell you whether it beats the closing market.

**Known blind spots:** injuries and teammates' absences (use `--minutes`), late-season tanking
and rest (projections lag sudden role changes — April is the weakest month), trades
(opponent/team inferred from the player's latest team in the data).

## Kalshi paper trading

`kalshi_bot.py` is Phase 1 of [KALSHI_PLAN.md](KALSHI_PLAN.md). It reads Kalshi's live prices
for tonight's NBA winner, spread and total contracts, prices each one with the game model
(applying today's injury report the way the app does) and paper-bets the ones with an edge.
**It never sends an order.** The project has no Kalshi login or order code yet, and Kalshi's
price data is public, so no account is needed.

```bash
python kalshi_client.py              # today's Kalshi contracts, matched to the schedule
python kalshi_bot.py scan            # what it would bet now; records nothing
python kalshi_bot.py run --update    # refresh NBA data, then record paper bets
python kalshi_bot.py report          # record, profit, closing-line value, model vs Kalshi Brier
```

How a contract is judged (all thresholds are flags, e.g. `--min-ev 0.03`, `--blend 0.6`):

* **Probability:** half the model and half Kalshi's mid price (`--blend 0.5`), because the market
  knows things the model doesn't. Phase 0's backtest should fit this weight.
* **EV per contract** = that probability − the price − Kalshi's fee (7% × P × (1 − P), rounded up
  to the cent per order). It needs at least 2¢ and a 4% return on the cost.
* **When:** each game only between 90 and 30 minutes before its tip-off (`--window-minutes`),
  when most injury news is out. Earlier runs still record prices, and print when betting opens.
* **Unsettled injuries:** a game waits ("waiting on injury news") while a player who would be in
  either team's top-8 lineup is on ESPN's report as Questionable or Doubtful. The model would
  assume he plays. `--bet-through-injuries` bets anyway. Day-To-Day doesn't hold a game: ESPN
  rarely updates it before tip-off, so waiting on it skipped most games with an edge.
* **Skipped:** bid-ask spreads wider than 4¢, prices under 5¢ or over 95¢, games within 30 minutes
  of tip-off, and gaps of more than 15 points between model and Kalshi. Those are flagged in the
  output to check the news, not bet.
* **One contract per game and market.** It picks the best rung of a spread or total ladder and
  never adds to a position on a later run.
* **Stake:** ¼ Kelly on the paper bankroll (`--bankroll`, $1,000 to start), capped at 2% a bet,
  3% a game and 10% a day, and no more contracts than the order book offers at that price.
  The output shows each bet's payout multiplier (1 / price).
* **Fails closed:** if the injury report, Kalshi's prices, an order book or last night's box scores
  are missing, it doesn't bet (`--no-injuries` and `--allow-stale` override the first and last).

Every contract it prices is kept in `kalshi_snapshots` with the first and last price seen before
tip-off. The last one is the closing price for closing-line value (CLV), so run `run` often on
game days. Every 15 minutes from 10 am to midnight, from cron (`--game-days-only` makes it exit
straight away when no game today has yet to tip off; use the full path to a Python that has the
requirements, since cron's PATH is minimal):

```
*/15 10-23 * * * cd /path/to/nba-prop-bet && .venv/bin/python kalshi_bot.py run --update --game-days-only >> kalshi.log 2>&1
```

**The app runs it for you:** on a game day, opening the app runs the bot in the background
(after pulling new games, at most every 15 minutes, until 30 minutes before the last tip-off),
and the tab has a **Run bot now** button. `NBA_KALSHI_AUTO=0` turns the automatic run off.

Against Kalshi's real prices for the whole 2025-26 season the model added nothing beyond Kalshi's
price, and these rules lost 8% over 1,011 bets (`research/kalshi_2025_26/REPORT.md`), so treat
the paper record as a live check, not a strategy.

**Backtest against Kalshi:** `python kalshi_history.py` downloads a past season's settled Kalshi
contracts (price an hour before tip-off and at tip-off; about two hours, resumable), then
`python market_backtest.py` scores the model against them walk-forward and simulates the bot's
rules (`--blend`, `--max-disagree`, `--one-per-game`). Cron
is what bets each game inside its window and catches late price moves (for CLV) when the app
isn't open.

The app's **Kalshi bot** tab opens on today's bets and shows what the bot did: paper bankroll, record, profit and average
CLV, every paper bet (price, payout, model vs Kalshi, EV, CLV, result), a day-by-day summary, and
the games it flagged today. Bets settle from the box scores in `daily_update.py`, which the app
runs when it pulls new games (or `kalshi_bot.py settle`). The plan's gate
for real money is 300+ paper bets with positive average CLV.

## Files

| File | Purpose |
|---|---|
| `app.py`, `start.command` | The web app and its Mac launcher |
| `kalshi_history.py` | Downloads a past season's settled Kalshi NBA contracts into `kalshi_history` |
| `market_backtest.py` | Walk-forward test of the model against Kalshi's real prices |
| `fetch_game_logs.py` | League-wide player + team game logs from stats.nba.com |
| `daily_update.py` | Refresh current season, settle logged bets and Kalshi paper bets, recalibrate game model, refresh schedule |
| `nba_injuries.py` | Today's injury report (ESPN), used for the playing/out defaults (`NBA_INJURIES=0` turns it off) |
| `nba_schedule.py` | Season schedule incl. upcoming games (for the Game tab's list of today's games) |
| `feature_engineering.py` | Shared numeric core (rates, minutes, opponent factors, dispersion) |
| `projection_model.py` | Distributions, fair lines, projection CLI |
| `bet_signals.py` | Prop evaluator, prop log, settlement, report |
| `tracking.py` | Performance summaries for the Results tab and report |
| `game_model.py` | Team ratings, score/spread/total/moneyline model |
| `odds.py` | American odds, no-vig, EV, Kelly |
| `odds_fetcher.py` | Optional The Odds API client (game lines + player props) |
| `kalshi_client.py` | Kalshi's public NBA prices, matched to the schedule (read-only) |
| `kalshi_bot.py` | Kalshi paper-trading bot: EV after fees, Kelly sizing, limits, settlement, CLV |
| `backtest.py` | Walk-forward backtests and parameter tuning |
| `db.py`, `schema.sql` | Database, season helpers, player/team name matching |
| `tests/` | `python -m pytest tests` |
