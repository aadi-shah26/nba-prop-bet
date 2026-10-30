from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import market_backtest as mb
from kalshi_bot import BotConfig
from kalshi_history import _quote, game_rows

TIP = datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc)


def candle(minutes_before, bid, ask):
    return {'end_period_ts': int((TIP - timedelta(minutes=minutes_before)).timestamp()),
            'yes_bid': {'close': f'{bid:.4f}'}, 'yes_ask': {'close': f'{ask:.4f}'}}


def test_quote_takes_the_latest_candle_at_or_before_the_time():
    cs = [candle(200, 0.40, 0.42), candle(75, 0.44, 0.46), candle(10, 0.50, 0.51)]
    assert _quote(cs, TIP - timedelta(minutes=60)) == (0.44, 0.46)     # minutes with no change send no candle
    assert _quote(cs, TIP) == (0.50, 0.51)
    assert _quote(cs, TIP - timedelta(hours=5)) == (None, None)
    assert _quote([candle(10, 0.0, 0.03)], TIP) == (None, None)       # no two-sided book


class FakeClient:
    def __init__(self, markets, candles):
        self._m, self._c = markets, candles

    def markets(self, event_ticker):
        return self._m.get(event_ticker.split('-')[0], [])

    def candles(self, ticker, start, end):
        return self._c.get(ticker, [])


def test_game_rows_skips_in_game_markets_and_reads_results():
    exp = (TIP + timedelta(hours=3)).isoformat().replace('+00:00', 'Z')
    pre, live = (TIP - timedelta(days=1)).isoformat(), (TIP + timedelta(minutes=30)).isoformat()
    base = {'expected_expiration_time': exp, 'status': 'finalized'}
    markets = {'KXNBAGAME': [{**base, 'ticker': 'KXNBAGAME-26MAR30DALMIL-MIL', 'created_time': pre, 'result': 'yes'}],
               'KXNBATOTAL': [{**base, 'ticker': 'KXNBATOTAL-26MAR30DALMIL-230', 'created_time': pre, 'result': 'no',
                               'floor_strike': 230.5, 'strike_type': 'greater'},
                              {**base, 'ticker': 'KXNBATOTAL-26MAR30DALMIL-260', 'created_time': live, 'result': 'no',
                               'floor_strike': 260.5, 'strike_type': 'greater'}]}
    candles = {'KXNBAGAME-26MAR30DALMIL-MIL': [candle(90, 0.60, 0.61), candle(5, 0.62, 0.63)],
               'KXNBATOTAL-26MAR30DALMIL-230': [candle(70, 0.48, 0.50)]}
    rows = game_rows(FakeClient(markets, candles), '26MAR30DALMIL', {'game_date': '2026-03-30', 'home': 'MIL', 'away': 'DAL'})
    assert [(r[0], r[5], r[6], r[7], r[8:12], r[13]) for r in rows] == [
        ('KXNBAGAME-26MAR30DALMIL-MIL', 'WINNER', 'MIL', None, (0.60, 0.61, 0.62, 0.63), 1),
        ('KXNBATOTAL-26MAR30DALMIL-230', 'TOTAL', None, 230.5, (0.48, 0.50, 0.48, 0.50), 0)]
    assert rows[0][4] == TIP.isoformat()


def _priced(p_model, p_market, results, market='TOTAL'):
    n = len(results)
    return pd.DataFrame({'game_date': [f'2026-01-{i % 28 + 1:02d}' for i in range(n)], 'home': 'MIL', 'away': 'DAL',
                         'market': market, 'p_model': p_model, 'p_market': p_market, 'result': results})


def test_accuracy_finds_no_weight_for_a_model_that_is_noise():
    rng = np.random.default_rng(1)
    pk = rng.uniform(0.2, 0.8, 4000)
    y = (rng.uniform(size=4000) < pk).astype(int)
    a = mb.accuracy(_priced(rng.uniform(0.2, 0.8, 4000), pk, y), blend=0.5).set_index('market')
    assert a.loc['TOTAL', 'brier_kalshi'] < a.loc['TOTAL', 'brier_blend'] < a.loc['TOTAL', 'brier_model']
    assert a.loc['TOTAL', 'best_blend'] == 0 and abs(a.loc['TOTAL', 'model_weight']) < 0.2


def test_simulate_scores_a_bet_and_its_clv():
    df = pd.DataFrame([{'game_date': '2026-01-05', 'home': 'MIL', 'away': 'DAL', 'market': 'WINNER', 'team': 'MIL',
                        'line': None, 'tip_utc': TIP.isoformat(), 'bet_bid': 0.49, 'bet_ask': 0.50, 'p_close': 0.545,
                        'p_model': 0.62, 'result': 1, 'pred_margin': 3.0, 'pred_total': 225.0}])
    b = mb.simulate(df, 12.7, 18.6, BotConfig())
    assert list(b['side']) == ['yes'] and b['won'].all()
    assert b['profit'].iloc[0] == pytest.approx(mb.STAKE * (1 / (0.50 + 0.07 * 0.25) - 1))
    assert b['clv'].iloc[0] == pytest.approx(4.5)
