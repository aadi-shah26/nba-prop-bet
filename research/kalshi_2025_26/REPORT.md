# Game model vs Kalshi's real prices: 2025-26 season

**What was tested:** the game model (`game_model.py`) and the Kalshi bot's rules (`kalshi_bot.py`)
against the prices Kalshi actually charged, for every game of 2025-26: 1,311 games, 33,277
winner, spread and total contracts (`kalshi_history.py`, public API). Each contract has Kalshi's
bid/ask 60 minutes before tip-off (when the bot bets), at tip-off (the closing price) and the
result. Every game is predicted walk-forward from earlier games only, with the SDs and margin
coefficients fit on 2024-25 (`market_backtest.py`), so nothing is tuned on the season it is
graded on.

Confidence tags: **[Certain]** = measured directly here · **[Likely]** = strong evidence, one
inference step · **[Guessing]** = judgement, not measured.

Run it: `python kalshi_history.py` (about two hours, resumable), then `python market_backtest.py`
or the scripts in this folder (`exp_market.py`, `exp_residual.py`, `exp_recalibrate.py`,
`exp_playoffs_2025.py`; outputs saved next to them as `.out`).

---

## Bottom line

**The model adds nothing beyond Kalshi's price, and the bot's rules lose money [Certain].** The
best weight on the model is 0 in every market. Betting where the model disagrees with Kalshi
loses the fee and more: −8.2% over 1,011 bets, with the 90% range below zero and negative
closing-line value. The paper bot keeps running with its usual settings as a live check (paper
money only); its record should be read with this in mind.

## 1. Accuracy: Kalshi beats the model in every market

Contracts priced 3-97¢, pre-game mid vs the model (Brier, lower is better):

| Market | Contracts | Kalshi | Model | 50/50 blend (the bot) | Best blend weight |
|---|---|---|---|---|---|
| Winner | 2,608 | **0.1978** | 0.2035 | 0.1993 | 0.0 |
| Spread | 16,147 | **0.1814** | 0.1852 | 0.1821 | 0.1 |
| Total | 14,387 | **0.2097** | 0.2186 | 0.2123 | 0.0 |

* **[Certain]** Worse in every regular-season month, in every market (October totals worst).
* **[Certain]** Kalshi's implied lines are closer to the result: margin 10.99 vs 11.23 points
  average error, total 14.52 vs 15.07.
* **[Certain]** When the model's line differs from Kalshi's, the result moves slightly the
  *other* way (−0.11 of the gap for the margin, −0.18 for the total): its disagreements are noise.
* **[Certain]** Kalshi itself is well calibrated across whole ladders: 25¢ contracts win 25%, 15¢
  spread contracts 16%. No favourite/longshot bias to exploit.

## 2. The bot's rules lose money

Simulated at the pre-game ask, $10 a bet, the bot's own rules (EV after fees, flags, best
contract per game and market):

| Setting | Bets | ROI | 90% range | CLV |
|---|---|---|---|---|
| **Current (blend 0.5, flag above 15%)** | 1,011 | **−8.2%** | −16.1% to −0.1% | −0.27¢ |
| One bet per game | 651 | −5.2% | −13.7% to +3.3% | −0.17¢ |
| Flag above 10% | 376 | +2.7% | −11.5% to +17.9% | −0.28¢ |
| Flag above 10%, one per game | 301 | −1.2% | −13.9% to +12.8% | −0.17¢ |
| Blend 0.25 | 1 | | | |

* **[Certain]** The bigger the disagreement, the worse the bets: 6-9 points +4.5%, 9-12 points
  −9.5%, 12-15 points −12.1%. Large gaps mean the model is missing something the market knows.
* **[Certain]** Closing-line value is negative everywhere: prices drift against the bot's side
  before tip-off. The +2.7% at a 10% flag comes with negative CLV and a range spanning zero; it's
  luck, not edge **[Likely]**.
* **[Certain]** At blend 0.25 the edge never clears the fee, so the bot stops betting. That is the
  honest answer for a model with no information beyond the market.

## 3. What was tried to make the model better (none helped against Kalshi)

| Idea | Result |
|---|---|
| Lineup changes (missing players) moving the total | Fit on 2024-25: ~0.01 pts per lineup point; 2025-26 total error unchanged. Teams replace the points **[Certain]** |
| Rest beyond back-to-backs (3 games in 4 nights, days off) | No effect on margin or total out of sample **[Certain]** |
| Less weight on last season for totals (0.4 → 0.15) | Early-season total error −0.2 pts vs results, but total Brier vs Kalshi 0.2197 → 0.2196. Dropped |
| Does anything the model knows predict Kalshi's miss? (model gap, early-season gap, back-to-backs, lineups, rest) | No coefficient with \|t\| above 1.9, on 1,310 games **[Certain]** |
| Recalibrating the model's probabilities (fit Oct-Jan, test Feb-Jun) | No better than raw; best blend 0-0.2 with a Brier gain far below the fee **[Certain]** |
| Playoffs only | 2025-26 playoffs (81 games) looked better for the model, but CLV was −0.5¢ and in the 2024-25 playoffs (80 games, winners only) the model was worse than Kalshi (0.2366 vs 0.2305). Not an edge **[Likely]** |

**Why [Likely]:** the model only sees box scores. Kalshi's price already contains everything in
them, plus injury news, lineups and sharp money, and it is set later.

## 4. Caveats

* The backtest's lineup term uses who actually played. Live, the bot only knows the injury
  report, so the live model can only be worse than this.
* The bet price is the latest minute candle at or before 60 minutes before tip-off; the bot bets
  anywhere from 90 to 30 minutes before. Depth isn't modelled: spread and total ladders are thin,
  so real fills would be worse, not better.

## 5. What could still work [Guessing]

Not more box-score modelling. An edge would need information or timing the market doesn't have:

1. **Speed on injury news.** Bet in the minutes after a status change, before Kalshi moves. Needs
   timestamped injury history to test, which this project doesn't have.
2. **Making instead of taking.** Resting limit orders pay about a quarter of the taker fee and
   earn the bid-ask spread on thin ladders, instead of paying it. Needs trade-level data to
   simulate fills.

Until one of those is tested, treat the paper bot as a live check, not a strategy, and keep
logging live prices so the check above can be repeated on 2026-27.
