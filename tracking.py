"""
Performance of what you track (props and game lines), plus the model's calibration.

"Track this" marks an evaluation as a bet you're following. Tracked rows give the record:
whether the model's side hit, and the profit in units (1u staked per pick) on the ones that
were actual bets. Calibration uses every saved evaluation, tracked or not, because picking
which ones to track would bias it: it compares the model's probability with the market's
no-vig probability, and if the market's Brier score is lower, the model has no proven edge yet.
"""

from datetime import date

import numpy as np
import pandas as pd

from odds import american_to_decimal

PENDING, HIT, MISS, PUSH, VOID = 'Pending', 'Hit', 'Miss', 'Push', 'Void (DNP)'
AWAITING = 'Awaiting result'


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


def _accuracy(status):
    """Hits / (hits + misses) of the model's side over finished tracked rows."""
    hits, misses = int((status == HIT).sum()), int((status == MISS).sum())
    return {'hits': hits, 'misses': misses, 'rate': hits / (hits + misses) if hits + misses else None}


def _status(outcome, settled, void):
    """outcome: +1 the model's side won, -1 it lost, 0 push (NaN when unsettled)."""
    return np.select([void, ~settled, outcome > 0, outcome < 0], [VOID, PENDING, HIT, MISS], PUSH)


def _profit(is_bet, outcome, odds):
    dec = odds[is_bet].map(american_to_decimal).reindex(odds.index)   # winner picks have no odds
    return pd.Series(np.where(~is_bet | outcome.isna(), np.nan,
                              np.where(outcome > 0, dec - 1, np.where(outcome < 0, -1.0, 0.0))),
                     index=outcome.index)


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------

def _prop_rows(conn, where="tracked = 1"):
    df = pd.read_sql_query(f"SELECT * FROM prop_log WHERE {where}", conn)
    df['actual'] = pd.to_numeric(df['actual'])      # all-NULL columns come back as objects
    is_bet = df['pick'].isin(['OVER', 'UNDER'])
    # a tracked PASS is judged on the model's lean: whichever side it rated more likely
    df['side'] = df['pick'].where(is_bet, np.where(df['p_over'] >= df['p_under'], 'OVER', 'UNDER'))
    over = df['side'] == 'OVER'
    df['is_bet'] = is_bet
    df['odds'] = df['over_odds'].where(over, df['under_odds'])
    df['ev'] = df['ev_over'].where(over, df['ev_under'])
    df['p_side'] = df['p_over'].where(over, df['p_under'])
    df['outcome'] = np.sign(df['actual'] - df['line']) * np.where(over, 1, -1)
    df['status'] = _status(df['outcome'], df['result'].notna(), df['result'] == 'DNP')
    df['profit'] = _profit(is_bet, df['outcome'], df['odds'])
    return df


def prop_summary(conn):
    t = _prop_rows(conn)
    out = {'logged': conn.execute("SELECT COUNT(*) FROM prop_log").fetchone()[0],
           'tracked': len(t), 'pending': int((t['status'] == PENDING).sum()),
           'settled': int((t['status'] != PENDING).sum()), 'accuracy': _accuracy(t['status']),
           'record': None, 'calibration': None}
    bets = t[t['is_bet'] & t['result'].isin(['WIN', 'LOSS', 'PUSH'])]
    if len(bets):
        out['record'] = _record(bets, 'odds', 'ev')
    df = pd.read_sql_query("SELECT * FROM prop_log WHERE result IS NOT NULL", conn)
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


def tracked_props(conn, show='All', today=None):
    """Display table of tracked props, newest game first. show: 'All' | 'Pending' | 'Finished'."""
    df = _filter(_prop_rows(conn), show).sort_values(['game_date', 'id'], ascending=False)
    return pd.DataFrame([{
        'id': r.id,
        'Date': r.game_date,
        'Player': r.player_name,
        'Bet': f"{r.stat} {r.side.title()} {r.line:g} ({int(r.odds):+d})",
        'Type': 'Bet' if r.is_bet else 'Lean',
        'vs': r.opponent or '',
        'Projection': round(r.proj_mean, 1),
        'Model %': r.p_side,
        'EV': r.ev,
        'Actual': r.actual,
        'Result': _result_text(r, today),
        'Profit (u)': r.profit,
    } for r in df.itertuples()], columns=TABLE_COLUMNS['props'])


# ---------------------------------------------------------------------------
# Game lines
# ---------------------------------------------------------------------------

def _game_rows(conn, where="tracked = 1"):
    df = pd.read_sql_query(f"SELECT * FROM game_log WHERE {where}", conn)
    df[['line', 'home_pts', 'away_pts']] = df[['line', 'home_pts', 'away_pts']].apply(pd.to_numeric)
    is_bet = df['pick'].isin(['A', 'B'])
    df['side'] = df['pick'].where(is_bet, np.where(df['p_a'] >= df['p_b'], 'A', 'B'))
    a = df['side'] == 'A'
    df['is_bet'] = is_bet
    df['odds'] = df['odds_a'].where(a, df['odds_b'])
    df['ev'] = df['ev_a'].where(a, df['ev_b'])
    df['p_side'] = df['p_a'].where(a, df['p_b'])
    margin = df['home_pts'] - df['away_pts']
    v = np.select([df['market'] == 'SPREAD', df['market'] == 'TOTAL'],
                  [margin + df['line'], df['home_pts'] + df['away_pts'] - df['line']], margin)
    df['outcome'] = pd.Series(np.sign(v), index=df.index) * np.where(a, 1, -1)
    df['status'] = _status(df['outcome'], df['result'].notna(), pd.Series(False, index=df.index))
    df['profit'] = _profit(is_bet, df['outcome'], df['odds'])
    return df


def game_summary(conn):
    t = _game_rows(conn)
    out = {'logged': conn.execute("SELECT COUNT(*) FROM game_log").fetchone()[0],
           'tracked': len(t), 'pending': int((t['status'] == PENDING).sum()),
           'settled': int((t['status'] != PENDING).sum()), 'accuracy': _accuracy(t['status']),
           'record': None}
    bets = t[t['is_bet'] & t['result'].isin(['WIN', 'LOSS', 'PUSH'])]
    if len(bets):
        out['record'] = _record(bets, 'odds', 'ev')
    return out


def _game_side_label(r):
    if r.market == 'TOTAL':
        return f"{'Over' if r.side == 'A' else 'Under'} {r.line:g}"
    team = r.home if r.side == 'A' else r.away
    if r.market == 'SPREAD':
        return f"{team} {r.line if r.side == 'A' else -r.line:+g}"
    return f"{team} to win"


def tracked_games(conn, show='All', today=None):
    """Display table of tracked game lines, newest game first."""
    df = _filter(_game_rows(conn), show).sort_values(['game_date', 'id'], ascending=False)
    return pd.DataFrame([{
        'id': r.id,
        'Date': r.game_date,
        'Away': r.away,
        'Home': r.home,
        'Bet': _game_side_label(r) + ('' if r.market == 'WINNER' else f" ({int(r.odds):+d})"),
        'Type': 'Pick' if r.market == 'WINNER' else 'Bet' if r.is_bet else 'Lean',
        'Predicted': f"{r.pred_away:.1f} – {r.pred_home:.1f}",
        'Model %': r.p_side,
        'EV': np.nan if r.market == 'WINNER' else r.ev,
        'Final': f"{r.away_pts:.0f} – {r.home_pts:.0f}" if pd.notna(r.home_pts) else '',
        'Result': _result_text(r, today),
        'Profit (u)': r.profit,
    } for r in df.itertuples()], columns=TABLE_COLUMNS['games'])


# ---------------------------------------------------------------------------

def _filter(df, show):
    if show == 'Pending':
        return df[df['status'] == PENDING]
    if show == 'Finished':
        return df[df['status'] != PENDING]
    return df


TABLE_COLUMNS = {
    'props': ['id', 'Date', 'Player', 'Bet', 'Type', 'vs', 'Projection', 'Model %', 'EV', 'Actual', 'Result',
              'Profit (u)'],
    'games': ['id', 'Date', 'Away', 'Home', 'Bet', 'Type', 'Predicted', 'Model %', 'EV', 'Final', 'Result',
              'Profit (u)'],
}


def _result_text(r, today):
    """Pending rows say whether the game is still ahead or just waiting on the data update."""
    if r.status != PENDING:
        return r.status
    return AWAITING if r.game_date < (today or date.today()).isoformat() else PENDING
