-- NBA prop/game model database.
-- Dates are ISO 'YYYY-MM-DD' so ORDER BY game_date is chronological.
-- season = start year of the season (2025 = 2025-26).

-- One row per player per game (league-wide, from LeagueGameLog P).
CREATE TABLE IF NOT EXISTS player_games (
    player_id   INTEGER NOT NULL,
    player_name TEXT    NOT NULL,
    team_id     INTEGER NOT NULL,
    team        TEXT    NOT NULL,
    game_id     TEXT    NOT NULL,
    game_date   TEXT    NOT NULL,
    season      INTEGER NOT NULL,
    season_type TEXT    NOT NULL,   -- 'Regular Season' | 'PlayIn' | 'Playoffs'
    opponent    TEXT    NOT NULL,
    home        INTEGER NOT NULL,   -- 1 = home, 0 = away
    wl          TEXT,
    minutes     REAL    NOT NULL,
    pts  INTEGER NOT NULL, reb  INTEGER NOT NULL, ast  INTEGER NOT NULL,
    stl  INTEGER NOT NULL, blk  INTEGER NOT NULL, tov  INTEGER NOT NULL,
    fg3m INTEGER NOT NULL, fgm  INTEGER NOT NULL, fga  INTEGER NOT NULL,
    ftm  INTEGER NOT NULL, fta  INTEGER NOT NULL, oreb INTEGER NOT NULL,
    dreb INTEGER NOT NULL, pf   INTEGER NOT NULL,
    plus_minus REAL,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_pg_player_date ON player_games(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_pg_name ON player_games(player_name);
CREATE INDEX IF NOT EXISTS idx_pg_date ON player_games(game_date);

-- One row per team per game (from LeagueGameLog T).
CREATE TABLE IF NOT EXISTS team_games (
    team_id     INTEGER NOT NULL,
    team        TEXT    NOT NULL,
    team_name   TEXT,
    game_id     TEXT    NOT NULL,
    game_date   TEXT    NOT NULL,
    season      INTEGER NOT NULL,
    season_type TEXT    NOT NULL,
    opponent    TEXT    NOT NULL,
    home        INTEGER NOT NULL,
    wl          TEXT,
    minutes     REAL,
    pts  INTEGER NOT NULL, reb  INTEGER NOT NULL, ast  INTEGER NOT NULL,
    stl  INTEGER NOT NULL, blk  INTEGER NOT NULL, tov  INTEGER NOT NULL,
    fg3m INTEGER NOT NULL, fg3a INTEGER NOT NULL, fgm  INTEGER NOT NULL,
    fga  INTEGER NOT NULL, ftm  INTEGER NOT NULL, fta  INTEGER NOT NULL,
    oreb INTEGER NOT NULL, dreb INTEGER NOT NULL, pf   INTEGER NOT NULL,
    plus_minus REAL,
    PRIMARY KEY (team_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_tg_date ON team_games(game_date);
CREATE INDEX IF NOT EXISTS idx_tg_team_date ON team_games(team, game_date);

-- Every prop you evaluate is logged here, then settled from player_games.
-- Over time this becomes the dataset that tells you whether the model beats the market.
CREATE TABLE IF NOT EXISTS prop_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at    TEXT NOT NULL,
    game_date    TEXT NOT NULL,          -- date of the game the prop is for
    player_id    INTEGER NOT NULL,
    player_name  TEXT NOT NULL,
    stat         TEXT NOT NULL,
    line         REAL NOT NULL,
    over_odds    INTEGER NOT NULL,
    under_odds   INTEGER NOT NULL,
    book         TEXT,
    opponent     TEXT,
    proj_mean    REAL NOT NULL,
    proj_sd      REAL NOT NULL,
    p_over       REAL NOT NULL,
    p_under      REAL NOT NULL,
    p_push       REAL NOT NULL,
    fair_over    REAL NOT NULL,          -- market no-vig probability of over
    ev_over      REAL NOT NULL,
    ev_under     REAL NOT NULL,
    pick         TEXT NOT NULL,          -- 'OVER' | 'UNDER' | 'PASS'
    actual       REAL,                   -- filled in by settle
    result       TEXT,                   -- 'WIN' | 'LOSS' | 'PUSH' | 'DNP' (pick side; for PASS: which side hit)
    settled_at   TEXT,
    UNIQUE (game_date, player_id, stat, line, over_odds, under_odds, book)
);

-- Every game line you evaluate is logged here, then settled from team_games.
CREATE TABLE IF NOT EXISTS game_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at   TEXT NOT NULL,
    game_date   TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    market      TEXT NOT NULL,           -- 'SPREAD' (home line) | 'TOTAL' | 'MONEYLINE'
    line        REAL,                    -- home spread or total; NULL for moneyline
    odds_a      INTEGER NOT NULL,        -- home side / over
    odds_b      INTEGER NOT NULL,        -- away side / under
    book        TEXT,
    pred_home   REAL NOT NULL,
    pred_away   REAL NOT NULL,
    p_a         REAL NOT NULL,
    p_b         REAL NOT NULL,
    fair_a      REAL NOT NULL,
    ev_a        REAL NOT NULL,
    ev_b        REAL NOT NULL,
    pick        TEXT NOT NULL,           -- 'A' (home/over) | 'B' (away/under) | 'PASS'
    home_pts    INTEGER,
    away_pts    INTEGER,
    result      TEXT,
    settled_at  TEXT,
    UNIQUE (game_date, home, away, market, line, odds_a, odds_b, book)
);

-- Small key/value store for calibrated model parameters (e.g. game-model residual SDs).
CREATE TABLE IF NOT EXISTS model_params (
    key        TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    updated_at TEXT NOT NULL
);
