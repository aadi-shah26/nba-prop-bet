"""
Feature engineering: the numeric core shared by live projections AND the backtest.

Model for a player's stat in one game (all inputs strictly before the game):

    minutes_proj  = recency-weighted mean of recent minutes             (short half-life)
    rate_c        = recency-weighted stat_c per minute, lightly shrunk  (longer half-life)
                    toward the league per-minute rate
    opp_factor_c  = how much of stat_c the opponent allows per game vs league average,
                    recency-weighted and shrunk toward 1.0
    mean          = sum over components c of  minutes_proj * rate_c * opp_factor_c ** opp_strength
    variance      = mean + alpha * mean**2   (negative binomial "NB2"), where alpha is the
                    player's own over-dispersion of the stat, shrunk toward the league value
                    and scaled by var_scale (calibrated in the backtest)

Combo stats (PRA, PR, ...) are built from their components so each component gets
its own opponent adjustment; dispersion is measured on the combo series itself so the
correlation between components is captured.
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

BASE_STATS = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg3m']

STATS = {
    'PTS': ('pts',),
    'REB': ('reb',),
    'AST': ('ast',),
    'STL': ('stl',),
    'BLK': ('blk',),
    'TOV': ('tov',),
    'FG3M': ('fg3m',),
    'PRA': ('pts', 'reb', 'ast'),
    'PR': ('pts', 'reb'),
    'PA': ('pts', 'ast'),
    'RA': ('reb', 'ast'),
    'SB': ('stl', 'blk'),
}

_ALIASES = {
    'POINTS': 'PTS', 'P': 'PTS', 'REBOUNDS': 'REB', 'R': 'REB', 'ASSISTS': 'AST', 'A': 'AST',
    'STEALS': 'STL', 'BLOCKS': 'BLK', 'TURNOVERS': 'TOV', 'TO': 'TOV',
    '3PM': 'FG3M', '3PT': 'FG3M', '3S': 'FG3M', 'THREES': 'FG3M', '3PTM': 'FG3M',
    'AR': 'RA', 'P+R+A': 'PRA', 'PTS+REB+AST': 'PRA', 'P+R': 'PR', 'PTS+REB': 'PR',
    'P+A': 'PA', 'PTS+AST': 'PA', 'R+A': 'RA', 'REB+AST': 'RA', 'S+B': 'SB', 'STL+BLK': 'SB',
}


def parse_stat(s):
    """User/bookmaker stat label -> canonical key in STATS. Raises ValueError if unknown."""
    key = str(s).strip().upper().replace(' ', '')
    key = _ALIASES.get(key, key)
    if key not in STATS:
        raise ValueError(f"Unknown stat {s!r}. Use one of: {', '.join(STATS)}")
    return key


@dataclass
class PropParams:
    half_life_minutes: float = 8.0     # games; minutes react quickly to role changes
    half_life_rate: float = 25.0       # games; per-minute production is more stable
    prev_season_weight: float = 0.6    # extra multiplier per season back
    lookback_seasons: int = 2          # current + previous season
    max_games: int = 160               # older games have negligible weight anyway
    rate_prior_minutes: float = 60.0   # pseudo-minutes of league-average production
    disp_prior_games: float = 20.0     # pseudo-games of league over-dispersion
    var_scale: float = 1.0             # multiplier on over-dispersion (calibration)
    opp_half_life: float = 30.0        # games (per team)
    opp_prior_games: float = 15.0      # pseudo-games of league average for opponent factors
    opp_strength: float = 1.0          # exponent on opponent factor (0 = ignore opponent)
    min_games: int = 5                 # refuse to project with fewer prior games

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def recency_weights(seasons, target_season, half_life, prev_season_weight):
    """Weights for games ordered oldest -> newest: 0.5**(games_ago/half_life),
    times prev_season_weight for each season before target_season."""
    n = len(seasons)
    games_ago = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (games_ago / half_life)
    back = np.clip(target_season - np.asarray(seasons, dtype=float), 0, None)
    return w * prev_season_weight ** back


def weighted_mean_var(y, w):
    """Weighted mean and (reliability-weights unbiased) variance."""
    sw = w.sum()
    mean = (w * y).sum() / sw
    denom = sw - (w ** 2).sum() / sw
    var = (w * (y - mean) ** 2).sum() / denom if denom > 0 else 0.0
    return mean, var


def effective_n(w):
    """Kish effective sample size."""
    return w.sum() ** 2 / (w ** 2).sum()


# ---------------------------------------------------------------------------
# League-level priors
# ---------------------------------------------------------------------------

def league_priors(player_games, min_games=30):
    """
    From a player_games DataFrame (only data you are allowed to use):
      rate[c]   = league stat_c per minute
      alpha[k]  = league NB2 over-dispersion of stat k: pooled (var - mean) / mean**2 over
                  players with >= min_games games and a meaningful mean
    """
    tot_min = player_games['minutes'].sum()
    rate = {c: player_games[c].sum() / tot_min for c in BASE_STATS}
    alpha = {}
    for key, comps in STATS.items():
        y = player_games[list(comps)].sum(axis=1)
        st = y.groupby(player_games['player_id']).agg(['mean', 'var', 'size'])
        st = st[(st['size'] >= min_games) & (st['mean'] >= 0.5)]
        alpha[key] = max(float((st['var'] - st['mean']).sum() / (st['mean'] ** 2).sum()), 0.0) \
            if len(st) else 0.1
    return {'rate': rate, 'alpha': alpha}


# ---------------------------------------------------------------------------
# Opponent factors
# ---------------------------------------------------------------------------

def team_games_with_allowed(team_games):
    """Attach the opponent's box score to each team row: allowed_<stat> = what this team gave up."""
    opp = team_games[['game_id', 'team'] + BASE_STATS].rename(
        columns={'team': 'opponent', **{c: f'allowed_{c}' for c in BASE_STATS}})
    out = team_games.merge(opp, on=['game_id', 'opponent'], how='inner')
    return out.sort_values(['game_date', 'game_id']).reset_index(drop=True)


def opponent_factors(tga, as_of, target_season, params):
    """
    tga: output of team_games_with_allowed. Uses only games with game_date < as_of.
    Returns {team: {stat_c: factor}} where factor 1.10 = allows 10% more than league average.
    """
    hist = tga[(tga['game_date'] < as_of) &
               (tga['season'] > target_season - params.lookback_seasons)]
    if hist.empty:
        return {}
    cols = [f'allowed_{c}' for c in BASE_STATS]
    # games-ago per team (0 = most recent)
    games_ago = hist.groupby('team').cumcount(ascending=False).to_numpy(dtype=float)
    back = np.clip(target_season - hist['season'].to_numpy(dtype=float), 0, None)
    w = 0.5 ** (games_ago / params.opp_half_life) * params.prev_season_weight ** back
    vals = hist[cols].to_numpy(dtype=float)
    league = (w[:, None] * vals).sum(axis=0) / w.sum()

    df = pd.DataFrame(w[:, None] * vals, columns=cols)
    df['team'] = hist['team'].to_numpy()
    df['w'] = w
    df['w2'] = w ** 2
    agg = df.groupby('team').sum()
    n_eff = agg['w'] ** 2 / agg['w2']
    k = params.opp_prior_games
    factors = {}
    for team, row in agg.iterrows():
        f = {}
        for i, c in enumerate(BASE_STATS):
            mean_allowed = row[cols[i]] / row['w']
            shrunk = (n_eff[team] * mean_allowed + k * league[i]) / (n_eff[team] + k)
            f[c] = shrunk / league[i] if league[i] > 0 else 1.0
        factors[team] = f
    return factors


# ---------------------------------------------------------------------------
# Player projection (pure function of prior games)
# ---------------------------------------------------------------------------

def project_from_history(hist, stat, target_season, priors, params,
                         opp_factor=None, minutes=None):
    """
    hist: the player's games strictly before the target game, sorted oldest -> newest,
          with columns season, minutes and BASE_STATS.
    opp_factor: {stat_c: factor} for the opponent, or None for a neutral opponent.
    minutes: optional projected-minutes override (e.g. known role change / minutes cap).
    Returns dict or None when there is not enough history.
    """
    stat = parse_stat(stat)
    comps = STATS[stat]
    hist = hist[hist['season'] > target_season - params.lookback_seasons]
    if len(hist) > params.max_games:
        hist = hist.iloc[-params.max_games:]
    if len(hist) < params.min_games:
        return None

    seasons = hist['season'].to_numpy()
    mins = hist['minutes'].to_numpy(dtype=float)
    w_min = recency_weights(seasons, target_season, params.half_life_minutes, params.prev_season_weight)
    w_rate = recency_weights(seasons, target_season, params.half_life_rate, params.prev_season_weight)

    minutes_hist = float((w_min * mins).sum() / w_min.sum())
    minutes_proj = float(minutes) if minutes is not None else minutes_hist

    m0 = params.rate_prior_minutes
    denom = (w_rate * mins).sum() + m0
    mean = 0.0
    rates, factors = {}, {}
    for c in comps:
        x = hist[c].to_numpy(dtype=float)
        rates[c] = ((w_rate * x).sum() + m0 * priors['rate'][c]) / denom
        factors[c] = (opp_factor or {}).get(c, 1.0) ** params.opp_strength
        mean += minutes_proj * rates[c] * factors[c]

    # Over-dispersion of the (combo) series itself, shrunk toward the league value.
    y = hist[list(comps)].sum(axis=1).to_numpy(dtype=float)
    y_mean, y_var = weighted_mean_var(y, w_rate)
    n_eff = effective_n(w_rate)
    a_prior = priors['alpha'][stat]
    a_player = (y_var - y_mean) / y_mean ** 2 if y_mean > 0 else a_prior
    n0 = params.disp_prior_games
    alpha = max((n_eff * a_player + n0 * a_prior) / (n_eff + n0), 0.0) * params.var_scale
    var = mean + alpha * mean ** 2

    cur = hist[hist['season'] == target_season]
    return {
        'stat': stat,
        'mean': mean,
        'var': var,
        'sd': float(np.sqrt(var)),
        'alpha': alpha,
        'minutes_proj': minutes_proj,
        'minutes_hist': minutes_hist,
        'rates': rates,
        'opp_factors': factors,
        'games_used': len(hist),
        'n_eff': n_eff,
        'season_avg': float(cur[list(comps)].sum(axis=1).mean()) if len(cur) else None,
        'season_games': len(cur),
        'last10_avg': float(y[-10:].mean()),
        'last5_avg': float(y[-5:].mean()),
    }
