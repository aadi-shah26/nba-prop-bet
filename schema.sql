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
    tracked      INTEGER NOT NULL DEFAULT 0,  -- 1 = you pressed "Track this"; 0 = only kept for calibration
    UNIQUE (game_date, player_id, stat, line, over_odds, under_odds, book)
);

-- Every game line you evaluate is logged here, then settled from team_games.
CREATE TABLE IF NOT EXISTS game_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at   TEXT NOT NULL,
    game_date   TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    market      TEXT NOT NULL,           -- 'SPREAD' (home line) | 'TOTAL' | 'MONEYLINE' | 'WINNER' (pick, no line; odds 0)
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
    tracked     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (game_date, home, away, market, line, odds_a, odds_b, book)
);

-- NBA schedule incl. future games (ScheduleLeagueV2), refreshed by the daily update and by the
-- app when it's more than a few hours old. Lets the app list each day's games.
CREATE TABLE IF NOT EXISTS schedule (
    game_id    TEXT PRIMARY KEY,
    game_date  TEXT    NOT NULL,   -- US Eastern date
    season     INTEGER NOT NULL,
    tip_utc    TEXT,
    status     TEXT,               -- '7:30 pm ET' before tip-off, 'Final' after
    home       TEXT    NOT NULL,
    away       TEXT    NOT NULL,
    label      TEXT,               -- e.g. 'East First Round · Game 3', 'Emirates NBA Cup'
    fetched_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schedule_date ON schedule(game_date);

-- Current rosters (CommonTeamRoster), refreshed by the daily update. The game model uses them
-- for who is on each team, so offseason moves count before those players have played.
CREATE TABLE IF NOT EXISTS rosters (
    season       INTEGER NOT NULL,
    team         TEXT    NOT NULL,
    player_id    INTEGER NOT NULL,
    player_name  TEXT    NOT NULL,
    position     TEXT,
    how_acquired TEXT,              -- e.g. 'Traded from PHI on 07/06/26'
    fetched_at   TEXT    NOT NULL,
    PRIMARY KEY (season, player_id)
);
CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(season, team);

-- Small key/value store for calibrated model parameters (e.g. game-model residual SDs).
CREATE TABLE IF NOT EXISTS model_params (
    key        TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    updated_at TEXT NOT NULL
);

-- App bookkeeping (e.g. when the data was last refreshed), as text.
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- game_log's UNIQUE constraint can't dedupe moneylines (line IS NULL) or rows without a book,
-- because SQLite treats NULLs as distinct. This index treats them as values.
DELETE FROM game_log WHERE id NOT IN (
    SELECT MIN(id) FROM game_log
    GROUP BY game_date, home, away, market, IFNULL(line, 'ML'), odds_a, odds_b, IFNULL(book, ''));
CREATE UNIQUE INDEX IF NOT EXISTS ux_game_log_dedup ON game_log
    (game_date, home, away, market, IFNULL(line, 'ML'), odds_a, odds_b, IFNULL(book, ''));

-- Kalshi bets placed by kalshi_bot.py (mode 'PAPER' until real orders exist), one per game and
-- market, settled from team_games. client_order_id stops a retried run from betting twice.
CREATE TABLE IF NOT EXISTS kalshi_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    mode            TEXT NOT NULL,           -- 'PAPER'
    client_order_id TEXT NOT NULL UNIQUE,
    game_date       TEXT NOT NULL,
    home            TEXT NOT NULL,
    away            TEXT NOT NULL,
    tip_utc         TEXT,
    market          TEXT NOT NULL,           -- 'WINNER' | 'SPREAD' | 'TOTAL'
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,           -- 'yes' | 'no'
    team            TEXT,                    -- team the YES side backs (WINNER/SPREAD)
    line            REAL,                    -- YES wins if the team's margin / the total is over this
    description     TEXT NOT NULL,           -- e.g. 'CHA to win', 'Under 218.5'
    p_model         REAL NOT NULL,           -- probabilities of our side winning
    p_market        REAL NOT NULL,
    p_blend         REAL NOT NULL,
    bid             REAL,
    ask             REAL NOT NULL,
    ev              REAL NOT NULL,           -- dollars per contract after fees
    kelly           REAL NOT NULL,           -- full-Kelly fraction (the stake used kelly_mult of it)
    count           INTEGER NOT NULL,
    price           REAL NOT NULL,           -- fill price per contract
    fee             REAL NOT NULL,
    cost            REAL NOT NULL,           -- count * price + fee
    result          TEXT,                    -- 'WIN' | 'LOSS'
    payout          REAL,
    profit          REAL,
    settled_at      TEXT,
    UNIQUE (mode, game_date, home, away, market)
);

-- Every Kalshi contract the bot priced, with the model's probability and Kalshi's price when it
-- was first and last seen before tip-off (last = the closing price for CLV), settled like bets.
CREATE TABLE IF NOT EXISTS kalshi_snapshots (
    ticker      TEXT PRIMARY KEY,
    game_date   TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    tip_utc     TEXT,
    market      TEXT NOT NULL,
    team        TEXT,
    line        REAL,
    p_model     REAL NOT NULL,           -- model probability that YES wins (latest run)
    yes_bid     REAL,
    yes_ask     REAL,
    first_seen  TEXT NOT NULL,
    first_mid   REAL,
    last_seen   TEXT NOT NULL,
    last_mid    REAL,
    decision    TEXT,                    -- 'BET' or why the game-market's best contract was skipped
    result      INTEGER                  -- 1 = YES won
);
CREATE INDEX IF NOT EXISTS idx_kalshi_snap_date ON kalshi_snapshots(game_date);
