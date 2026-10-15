# Prop model backtest — 2025-26 season

**What was tested:** the current prop model (`feature_engineering.py` / `projection_model.py`, default
`PropParams`), walk-forward over the whole 2025-26 season (regular season, play-in and playoffs,
2025-10-21 → 2026-06-13). Every prediction uses only games played *before* that game.

* 25,742 player-games (every player who had ≥ 5 games that season) × 11 stats = **283,162 predictions**
* Stats: PTS, REB, AST, 3PM, PRA, PR, PA, RA, STL, BLK, TOV
* Improvements were fit on **2024-25** and scored on **2025-26**, so nothing below was tuned on the
  season it is graded on.

Confidence tags: **[Certain]** = measured directly in this backtest · **[Likely]** = strong evidence,
one inference step · **[Guessing]** = judgement, not measured.

---

## ⚠️ Read this first: there are no real sportsbook lines in this test

Historical prop lines aren't free, and this environment couldn't reach any odds source. So "line"
below means a **proxy line**: the player's season-to-date average rounded to x.5. A real book line is
much sharper, because it already prices in injury news, lineups and sharp money.

Against the proxy line the model "wins" 63.9% of bets at -110, for +22% ROI. **That number is not
real [Certain].** Simply betting the UNDER on every proxy line wins 56.9%, because stat
distributions are right-skewed (the median sits below the average). The proxy line is a soft
target. Use these backtests to judge *accuracy and calibration*. Only real lines can show *profit*
(see recommendation 2).

---

## 1. Scorecard: what the model gets right

| Stat | MAE model | MAE season avg | MAE last-10 | Bias (proj − actual) | 80% interval hit | Brier at proxy line (coin = 0.250) |
|---|---|---|---|---|---|---|
| PTS  | **4.58** | 4.71 | 4.67 | −0.09 | 79.8% | 0.241 |
| REB  | **1.88** | 1.93 | 1.93 | −0.01 | 81.3% | 0.238 |
| AST  | **1.34** | 1.37 | 1.37 | −0.04 | 80.7% | 0.234 |
| 3PM  | 0.90 | 0.90 | 0.91 | −0.00 | 80.7% | 0.208 |
| PRA  | **6.01** | 6.26 | 6.15 | −0.13 | 82.0% | 0.239 |
| PR   | **5.51** | 5.71 | 5.64 | −0.10 | 81.4% | 0.240 |
| PA   | **5.05** | 5.23 | 5.15 | −0.13 | 80.9% | 0.240 |
| RA   | **2.58** | 2.68 | 2.66 | −0.04 | 81.8% | 0.237 |
| STL  | 0.72 | 0.73 | 0.74 | −0.01 | 79.9% | 0.225 |
| BLK  | 0.52 | 0.51 | 0.52 | −0.00 | 79.8% | 0.184 |
| TOV  | 0.89 | 0.91 | 0.92 | −0.01 | 80.6% | 0.230 |

* **[Certain]** The projected *average* beats the season average and the last-10 average on every
  main stat, by about 3-4% (PTS) and 4% (PRA). On 3PM, STL and BLK it's a tie.
* **[Certain]** The *width* of the distribution is right: the 80% range catches 80-82% of outcomes.
* **[Certain]** There is no meaningful bias by projection level, early-season sample size, home/away,
  calendar month (regular season) or recent "hot/cold" streak. The model is not chasing noise.

## 2. The biggest problem: P(over) is too low, so the model leans UNDER

At the proxy line, the model's average P(over) compared with how often the over actually hit:

| Stat | Model P(over) | Actual over % | Gap |
|---|---|---|---|
| PTS | 42.5% | 46.4% | **−3.9 pts** |
| PRA | 43.9% | 47.9% | **−4.0 pts** |
| PR / PA | 43.5% / 43.3% | 47.4% / 47.4% | **−3.9 / −4.1 pts** |
| RA | 44.6% | 47.0% | −2.4 pts |
| REB / AST / STL / TOV | | | −0.9 to −1.4 pts |
| 3PM / BLK | | | ≈ 0 |

* **[Certain]** The over is underestimated in *every* probability bucket, not just a few:
  predicted 47.4% → actual 49.3%, predicted 57.3% → actual 60.2%, and so on.
* **[Certain]** As a result, **81% of the model's +3% EV picks are UNDERs**, and 85% on points.
* **[Likely]** Against real books this bias shows up as *fake* under edges. A 4-point probability
  error is bigger than your whole 3% EV threshold: at -110 the model only needs 54.0% vs 52.4%
  break-even, a 1.6-point margin. Books set lines near the median, which is exactly where this
  error sits.
* **Cause [Likely]:** the negative-binomial shape is more right-skewed than real scoring, and
  `var_scale = 1.3` adds even more skew. Points (and combos built on points) are closer to
  symmetric than a negative binomial assumes. A plain normal curve overshoots in the other direction
  (it predicts 50% over).
* **Fix, tested out-of-sample [Certain]:** a per-stat logistic recalibration of P(over), fit on
  2024-25 and applied to 2025-26, removes the gap. PTS: 42.5% → 46.4% (actual 46.4%); PRA:
  43.9% → 48.0% (actual 47.9%). Brier improves slightly, and the under share of PTS picks falls from
  85% to 72%. It holds with the alternative proxy line (last-10 average) too.

## 3. Where the error actually comes from: minutes

If the model had known each player's actual minutes, its error would drop a lot:

| Stat | MAE now | MAE if minutes were known | Reduction |
|---|---|---|---|
| PTS | 4.58 | 3.74 | −18% |
| REB | 1.88 | 1.60 | −15% |
| PRA | 6.01 | 4.38 | **−27%** |
| PR / PA | 5.51 / 5.05 | 4.17 / 3.94 | −24% / −22% |

**[Certain]** Minutes are the single largest source of error. The current minutes projection misses
by 4.9 minutes on average. Per-minute production is much more predictable than playing time. What
moves minutes, and what the model currently can't see:

| Situation (PTS, 2025-26) | Projection error | Minutes error | What it means |
|---|---|---|---|
| **≥ 60 teammate-minutes newly missing** (vs the team's last game) | actual **+23.6%** above projection | +3.8 min | the model has no idea who is out |
| 30-60 teammate-minutes newly missing | +5.3% | +1.1 min | |
| Nobody new missing | −5.5% | −1.3 min | the projection leans on games where others were out |
| ≥ 60 teammate-minutes *returning* | −6.6% | −1.8 min | |
| Player himself returning from an absence | −4.5% | −1.0 min | |
| Player starts | +4.4% | +1.0 min | |
| Player comes off the bench | −6.1% | −0.8 min | |
| Back-to-back | **+5.3%** | +0.9 min | *opposite* of the usual "fatigue" story: teammates rest |
| Close game (final margin ≤ 10) | +2.5 to +3.3% | +0.7 to +0.9 min | |
| Blowout (final margin > 20) | −2.0% | −0.9 min | |

The same teammate-absence pattern holds for rebounds (+20%), assists (+24%) and threes (+20%).

* **[Certain]** A pregame minutes regression fit on 2024-25 (recent minutes, season minutes,
  teammates out/returning × player's share, back-to-back, self-return, playoffs) cuts minutes MAE on
  2025-26 from 4.89 to **4.75**. Adding *confirmed starting lineups* cuts it to **4.61**: starting is
  worth +4.8 minutes on its own.
* **[Certain]** Learned coefficient: a player picks up about **0.59 × (vacated minutes) × (his
  minutes ÷ 240)**. For example, a 30-minute starter with 35 teammate-minutes out projects about
  +2.6 minutes.
* **[Certain]** 8.5% of games end with the player at under half his projected minutes (injury exit,
  foul trouble, blowout). Those games account for 17% of all minutes error and can't be predicted.
  Books face the same problem.
* **[Likely]** The blowout effect is measured on the *final* margin, which you can't know before tip.
  The pregame version is the game spread, which `game_model.py` already produces, so it is usable.

## 4. Playoffs

* **[Certain]** The model over-projects in the playoffs: PTS bias **+1.09** (regular season ≈ 0),
  PRA +1.7. Bench players are the problem: projected 6.3 pts on 15.4 min, actual 4.7 pts on 12.6 min.
  Starters play *more* (31.2 → 32.8 min). Rotations tighten, and the model keeps assuming
  regular-season minutes.

## 5. How much the fixes buy, tested out-of-sample (fit 2024-25 → test 2025-26)

| Stat | MAE now | + minutes model | + mean & dispersion recalibration | ML ceiling (gradient boosting, RMSE) |
|---|---|---|---|---|
| PTS | 4.581 | 4.554 | **4.544** (−0.8%) | RMSE 5.98 → 5.88 (−1.7%) |
| REB | 1.878 | 1.866 | **1.864** (−0.7%) | 2.48 → 2.44 |
| AST | 1.338 | 1.336 | **1.334** (−0.3%) | 1.82 → 1.80 |
| PRA | 6.009 | 5.954 | **5.935** (−1.2%) | 7.74 → 7.55 (−2.5%) |
| PR / PA | 5.51 / 5.05 | 5.47 / 5.02 | **5.45 / 5.00** (−1.0%) | |

* **[Certain]** With box-score data alone, the projected average is close to its ceiling. Better
  feature engineering, even a full ML model, buys only about 1-3%. Don't expect a rebuild of the
  averaging logic to create an edge.
* **[Certain]** The dispersion multiplier is wrong for small stats. Log-likelihood prefers the
  current over-dispersion × ~1.0 for PTS, but × 0.6-0.75 for REB/AST/RA, × 0.53 for 3PM and × 0.3-0.4
  for STL/BLK/TOV. The single global `var_scale = 1.3` is too wide for those.
* **[Likely]** The large gains come from **information the model doesn't have** (who's playing,
  who's starting, expected game script) and from **fixing the over/under bias**, not from fancier
  averaging.

---

## What to do, in priority order

1. **Recalibrate P(over) per stat (small change, biggest impact on picks).** Fit `p_cal =
   sigmoid(a + b·logit(p_model))` per stat on the previous season, using the backtest's proxy lines,
   and apply it in `line_probs` / `evaluate_prop`. Rough coefficients from 2024-25: PTS
   a=0.12 b=0.86, PRA a=0.15 b=0.91, PR a=0.14 b=0.91, PA a=0.13 b=0.90, REB a=0.02 b=0.83. Refit
   each offseason. Also replace the global `var_scale` with per-stat values. *Expected effect: removes
   the 4-point under bias that is currently inflating UNDER EV.* **[Certain about the bias; Likely
   about the size of the betting impact]**

2. **Get real lines, or the profit question can't be answered.** **[Certain]**
   * Starting with the first games of the 2026-27 season, log the **closing line** for every prop you
     evaluate and track **CLV** (closing line value: did your price beat the close?). It is the
     fastest reliable signal of edge, and it needs a few hundred bets, not thousands.
   * Optional: The Odds API has a paid historical-odds endpoint that includes player props. With it,
     this backtest could be re-run against real 2025-26 closing lines. **[Guessing]** on exact cost;
     check their pricing. Budget for several markets × ~1,300 games.

3. **Add a teammate-availability input.** Add a "who's OUT tonight" field (or pull the injury report)
   and redistribute vacated minutes: `+0.59 × vacated_min × player_min/240`, and the reverse when
   players return. This is the largest measured bias in the data (+24% when 60+ minutes are missing).
   It is also exactly why the README tells you to check the news when the model and market disagree.
   **[Certain about the effect; Likely about the gain, since it depends on getting the injury list
   right before the line moves]**

4. **Use confirmed starting lineups** when they are posted (~30 min before tip): +4.8 minutes for a
   starter vs the baseline projection. Minutes MAE 4.75 → 4.61. **[Certain]**

5. **Game-script adjustment from the spread.** Feed `game_model.py`'s predicted margin into minutes:
   starters lose about 1 minute in expected blowouts and gain about 1 in close games. **[Likely]**
   This needs its own backtest with the pregame spread before you trust the coefficient.

6. **Playoff mode.** Shrink bench minutes by ~15-20% and add ~1.5 minutes to starters in the
   playoffs. Otherwise skip playoff props. **[Certain about the bias; the exact factors are rough]**

7. **Leave these alone [Certain]:** home/away (< 1% effect), a back-to-back fatigue penalty (the data
   says the opposite), hot-hand / recent-form chasing (no bias), and a full ML rewrite (≤ 3% gain).

### What I'd skip betting until 1-3 are done
Points and combo UNDERs flagged by the model. That is where the calibration bias lives. **[Likely]**

---

## Reproduce

Data: ESPN box scores (via `import_espn_boxscores.py`; stats.nba.com was blocked in the environment
this ran in). ESPN minutes are whole minutes and ids differ from NBA ids. That doesn't matter for
modelling, but keep the ESPN data in its own database.

```bash
export BT_DIR=/path/to/workdir            # holds espn.db, raw/, outputs
python import_espn_boxscores.py --seasons 2023 2024 2025 --db $BT_DIR/espn.db
cd research/backtest_2025_26
NBA_DB_PATH=$BT_DIR/espn.db python diag_predict.py 2025 $BT_DIR/pred_2025.parquet   # ~9 min on 4 cores
NBA_DB_PATH=$BT_DIR/espn.db python diag_predict.py 2024 $BT_DIR/pred_2024.parquet
python context.py        # rest / teammate absences / starters (starters read from $BT_DIR/raw/player_box_*.parquet)
python analyze.py        # sections 1-4
python exp.py && python exp2.py && python exp3.py && python exp4.py   # section 5 + calibration fixes
```
`context.py` reads the ESPN parquet files from `$BT_DIR/raw/` (download them from the
sportsdataverse release URLs listed in `import_espn_boxscores.py`). `exp*.py` need `scikit-learn`
for the gradient-boosting ceiling check.
