"""
Performance of everything you've logged (props and game lines).

Profit is in units (1u staked per pick). Calibration compares the model's probability with
the market's no-vig probability on every settled prop: if the market's Brier score is lower,
the market is the better forecaster and the model has no proven edge yet.
"""

import pandas as pd

from odds import american_to_decimal


def _record(df, odds_col, ev_col):
    """df rows: one settled pick each, with result in WIN/LOSS/PUSH."""
    dec = df[odds_col].map(american_to_decimal)
    profit = (df['result'] == 'WIN') * (dec - 1) - (df['result'] == 'LOSS') * 1.0
    return {
        'picks': len(df),
        'wins': int((df['result'] == 'WIN').sum()),
        'losses': int((df['result'] == 'LOSS').sum()),
        'pushes': int((df['result'] == 'PUSH').sum()),
        'profit': float(profit.sum()),
        'roi': float(profit.mean()) if len(df) else 0.0,
        'claimed_ev': float(df[ev_col].mean()) if len(df) else 0.0,
    }


def prop_summary(conn):
    df = pd.read_sql_query("SELECT * FROM prop_log WHERE result IS NOT NULL", conn)
    out = {'logged': conn.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0],
           'settled': len(df), 'record': None, 'calibration': None}
    bets = df[df['pick'].isin(['OVER', 'UNDER']) & df['result'].isin(['WIN', 'LOSS', 'PUSH'])].copy()
    if len(bets):
        over = bets['pick'] == 'OVER'
        bets['odds'] = bets['over_odds'].where(over, bets['under_odds'])
        bets['ev'] = bets['ev_over'].where(over, bets['ev_under'])
        out['record'] = _record(bets, 'odds', 'ev')
    c = df[df['actual'].notna() & (df['actual'] != df['line'])].copy()
    if len(c):
        c['p'] = c['p_over'] / (c['p_over'] + c['p_under'])
        c['hit'] = (c['actual'] > c['line']).astype(float)
        cut = pd.cut(c['p'], [0, .3, .4, .45, .5, .55, .6, .7, 1])
        out['calibration'] = {
            'n': len(c),
            'brier_model': float(((c['p'] - c['hit']) ** 2).mean()),
            'brier_market': float(((c['fair_over'] - c['hit']) ** 2).mean()),
            'table': c.groupby(cut, observed=True).agg(
                n=('hit', 'size'), model=('p', 'mean'), market=('fair_over', 'mean'),
                actual=('hit', 'mean')).reset_index().rename(columns={'p': 'bucket'}),
        }
    return out


def game_summary(conn):
    df = pd.read_sql_query("SELECT * FROM game_log WHERE result IS NOT NULL", conn)
    out = {'logged': conn.execute("SELECT COUNT(*) FROM game_log").fetchone()[0],
           'settled': len(df), 'record': None}
    bets = df[df['pick'].isin(['A', 'B']) & df['result'].isin(['WIN', 'LOSS', 'PUSH'])].copy()
    if len(bets):
        a = bets['pick'] == 'A'
        bets['odds'] = bets['odds_a'].where(a, bets['odds_b'])
        bets['ev'] = bets['ev_a'].where(a, bets['ev_b'])
        out['record'] = _record(bets, 'odds', 'ev')
    return out


def recent_props(conn, limit=200):
    return pd.read_sql_query(
        """SELECT game_date, player_name, stat, line, over_odds, under_odds, opponent,
                  ROUND(proj_mean, 1) AS projection, ROUND(p_over, 3) AS p_over,
                  ROUND(p_under, 3) AS p_under, ROUND(MAX(ev_over, ev_under), 3) AS best_ev,
                  pick, actual, result
           FROM prop_log ORDER BY game_date DESC, id DESC LIMIT ?""", conn, params=(limit,))


def recent_games(conn, limit=200):
    return pd.read_sql_query(
        """SELECT game_date, away || ' @ ' || home AS game, market, line, odds_a, odds_b,
                  ROUND(pred_away, 1) || '-' || ROUND(pred_home, 1) AS predicted,
                  ROUND(p_a, 3) AS p_a, ROUND(MAX(ev_a, ev_b), 3) AS best_ev,
                  CASE pick WHEN 'A' THEN CASE market WHEN 'TOTAL' THEN 'OVER' ELSE home END
                            WHEN 'B' THEN CASE market WHEN 'TOTAL' THEN 'UNDER' ELSE away END
                            ELSE 'PASS' END AS pick,
                  away_pts || '-' || home_pts AS final, result
           FROM game_log ORDER BY game_date DESC, id DESC LIMIT ?""", conn, params=(limit,))
