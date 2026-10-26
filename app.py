"""
NBA betting model — local web app.

    streamlit run app.py          (or double-click start.command on a Mac)

Pick a player or game, type the line and the prices, get OVER / UNDER / PASS with the
numbers behind it. Optionally, with an ODDS_API_KEY set, lines can be imported from The Odds
API instead of typed. Evaluations are saved to your log and settled automatically when you
press "Update data". Opening the app pulls the latest games by itself (at most every
AUTO_UPDATE_MINUTES), so the data and your tracked bets are current without pressing anything.
Press "Track this" on any evaluation to follow it in the Tracked bets tab.
"""

import os
from datetime import date, datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bet_signals import delete_prop, evaluate_prop, log_prop, prop_log_id, settle_prop_manual, untrack_prop
import background
import kalshi_bot
import nba_injuries
import nba_schedule
import odds_fetcher
from db import DB_PATH, connect, get_meta, init_db, recent_players, season_for_date, set_meta
from game_model import (WINNER, GameContext, adjustment_text, delete_game_bets, evaluate_markets, explain_game,
                        delete_game_bet, game_log_id, log_game_evals, pick_id, settle_game_manual,
                        stale_warning, track_pick, untrack_game_bet)
from projection_model import PropContext, count_distribution
from tracking import game_summary, prop_summary, tracked_games, tracked_props
import ui

st.set_page_config(page_title="NBA Betting Model", page_icon=":material/sports_basketball:", layout="wide")
ui.inject_css()

STAT_LABELS = {
    'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'FG3M': '3-pointers made',
    'PRA': 'Pts + Reb + Ast', 'PR': 'Pts + Reb', 'PA': 'Pts + Ast', 'RA': 'Reb + Ast',
    'STL': 'Steals', 'BLK': 'Blocks', 'SB': 'Steals + Blocks', 'TOV': 'Turnovers',
}
AUTO_UPDATE_MINUTES = 30    # opening the app pulls new games at most this often
TABS = ["Player props", "Games", "Tracked bets", "Kalshi bot"]


# ---------------------------------------------------------------------------
# Data access (cached on the data version so new games invalidate the cache)
# ---------------------------------------------------------------------------

def db():
    conn = connect()
    init_db(conn)
    return conn


def data_version():
    conn = db()
    try:
        return conn.execute("SELECT MAX(game_date), COUNT(*) FROM player_games").fetchone() + \
            conn.execute("SELECT COUNT(*) FROM model_params").fetchone() + \
            conn.execute("SELECT MAX(fetched_at) FROM rosters").fetchone()
    finally:
        conn.close()


@st.cache_resource(max_entries=4, show_spinner="Loading player data...")
def prop_context(game_date, version):
    conn = db()
    try:
        ctx = PropContext(conn, game_date)
    finally:
        conn.close()
    ctx.conn = None
    return ctx


@st.cache_resource(max_entries=4, show_spinner="Building team ratings...")
def game_context(game_date, version):
    conn = db()
    try:
        ctx = GameContext(conn, game_date)
    finally:
        conn.close()
    ctx.conn = None
    return ctx


@st.cache_data(show_spinner=False)
def player_options(version):
    conn = db()
    try:
        last = conn.execute("SELECT MAX(game_date) FROM player_games").fetchone()[0]
        df = recent_players(conn, min_season=season_for_date(last) - 1)  # this + last season
    finally:
        conn.close()
    df = df.sort_values('player_name')
    return {int(r.player_id): f"{r.player_name} ({r.team})" for r in df.itertuples()}


@st.cache_data(show_spinner=False)
def team_options():
    from nba_api.stats.static import teams
    return {t['abbreviation']: t['full_name'] for t in sorted(teams.get_teams(), key=lambda t: t['full_name'])}


@st.cache_data(ttl=600, show_spinner="Loading the schedule...")
def slate(game_date):
    """(games on `game_date`, next game day after it, fetch error or None). Fetches the
    schedule from stats.nba.com when the stored copy is missing or stale."""
    conn = db()
    try:
        # a stale copy is refreshed by the background update rather than making this page wait
        err = nba_schedule.ensure_schedule(conn, game_date, refresh_stale=not auto_enabled())
        return nba_schedule.games_on(conn, game_date), nba_schedule.next_game_date(conn, game_date), err
    finally:
        conn.close()


@st.cache_data(ttl=1800, show_spinner=False)
def injury_report():
    """Today's injury report (ESPN), or None if it can't be fetched. NBA_INJURIES=0 turns it off."""
    if os.getenv('NBA_INJURIES', '1') == '0':
        return None
    try:
        return nba_injuries.normalize_injuries(nba_injuries.fetch_injuries())
    except Exception:  # offline, feed changed: the checklists just start from recent games
        return None


def injuries_for(ctx, game_date):
    """For today's and upcoming games, a function team -> {player_id: (status, comment)} from
    the injury report; None for past dates (it describes today) or when it's unavailable."""
    rep = injury_report() if game_date >= date.today().isoformat() else None
    if rep is None:
        return None
    return lambda team: nba_injuries.match_roster(rep, ctx.roster(team), team)


def report_outs(ctx, game_date, home, away):
    """evaluate() kwargs marking the injury report's Out players as not playing."""
    inj = injuries_for(ctx, game_date)
    if not inj:
        return {}
    return {f'{side}_out': [i for i, (status, _) in inj(team).items() if status == nba_injuries.OUT]
            for side, team in (('home', home), ('away', away))}


def fmt_odds(o):
    return f"{int(o):+d}"


def pct(x):
    return f"{x:.1%}"


def bad_odds(prices):
    return [f"{o} isn't valid American odds (use e.g. -110 or +120)." for o in prices if -100 < o < 100]


def run_prop(player_id, stat, line, over_odds, under_odds, opponent, minutes, game_date, book=None):
    """Evaluate one prop (typed or imported), keep it for display and log it."""
    try:
        ctx = prop_context(game_date, version)
        r = evaluate_prop(ctx, player_id, stat, float(line), int(over_odds), int(under_odds),
                          opponent=opponent, minutes=minutes, min_ev=min_ev)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return
    if r is None:
        st.error("Not enough recent games to project this player.")
        return
    st.session_state['prop_result'] = {**r, 'book': book}
    if save:
        conn = db()
        log_prop(conn, r, book=book)
        conn.close()


def run_game(home, away, game_date, spread=None, spread_odds=None, total=None, total_odds=None,
             ml=None, book=None, lineups=None):
    """Evaluate one game's lines (typed or imported), keep them for display and log them."""
    try:
        ctx = game_context(game_date, version)
        pred = ctx.evaluate(home, away, spread, total, **(lineups or {}))
        pred['lineups'] = lineups or {}
        evals = evaluate_markets(pred, spread, spread_odds, total, total_odds, ml, min_ev)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return
    st.session_state['game_result'] = (pred, evals, stale_warning(ctx), ctx.calibrated, ctx.game_date, book)
    if save and evals:
        conn = db()
        log_game_evals(conn, ctx.game_date, pred, evals, book=book)
        conn.close()


def lineup_inputs(away, home, game_date):
    """A playing/out checklist per team. Players from the team's last few games start ticked
    (untick anyone out); this season's other players start unticked (tick anyone back).
    Returns {'away_out', 'away_in', 'home_out', 'home_in'}: lists of player ids."""
    res = {'away_out': [], 'away_in': [], 'home_out': [], 'home_in': []}
    if not away or not home or away == home:
        return res
    try:
        ctx = game_context(game_date, version)
    except Exception:
        return res
    if not ctx.use_lineups:
        st.caption("Playoffs: the lineup adjustment is off (it made playoff picks worse in testing).")
        return res
    inj = injuries_for(ctx, game_date)
    if inj:
        st.markdown("**Who's playing tonight?** Players on today's injury report as **Out** start unticked. "
                    "Check late scratches, and tick anyone returning who hasn't played lately. Best players "
                    "are listed first.")
    else:
        st.markdown("**Who's playing tonight?** Check the injury report. Untick anyone who's out, "
                    "and tick anyone returning who hasn't played lately. Best players are listed first.")
    if ctx.roster_date:
        st.caption(f"Rosters as of {ctx.roster_date} (trades and signings included). Players with no NBA "
                   "games yet, like rookies, aren't listed: the model has no value for them.")
    for col, team, side in zip(st.columns(2), (away, home), ('away', 'home')):
        r = ctx.roster(team)
        report = inj(team) if inj else {}
        with col.container(border=True):
            st.markdown(f"**{team}**")
            for recent, heading in ((True, None), (False, "Haven't played lately")):
                group = r[r['recent'] == recent]
                if group.empty:
                    continue
                if heading:
                    st.caption(heading)
                for p in group.itertuples():
                    status, comment = report.get(int(p.player_id), (None, ''))
                    on = st.checkbox(p.player_name + (f" ({status})" if status else ""),
                                     value=recent and status != nba_injuries.OUT,
                                     key=f"play_{team}_{p.player_id}_{game_date}",
                                     help=(f"{status}: {comment}  \n" if status else "")
                                     + f"Value {p.value:.1f} (recent Game Score) · last played {p.last_game}")
                    if recent and not on:
                        res[f'{side}_out'].append(int(p.player_id))
                    elif on and not recent:
                        res[f'{side}_in'].append(int(p.player_id))
    return res


def track_button(tracked_id, label, key, on_track, on_untrack):
    """"Track this" for an evaluation, or a "Tracked" note with an undo once it is."""
    if tracked_id:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"{ui.pill('Tracked', 'pos')} &nbsp;{ui.esc(label)} · settles by itself after the game, "
                    "see **Tracked bets**.", unsafe_allow_html=True)
        if c2.button("Untrack", key=f"un{key}", width="stretch", icon=":material/bookmark_remove:"):
            on_untrack(tracked_id)
            st.rerun()
    elif st.button(f"Track this: {label}", key=key, icon=":material/bookmark_add:"):
        on_track()
        tracked_go(f"Tracking {label}")
        st.rerun()


def tracked_go(msg):
    """After tracking something: switch to the Tracked bets tab and say what was added."""
    st.session_state.update(goto_tab=TABS[2], toast=msg)


def pull_data(retries=4):
    """Download the latest box scores, settle bets, recalibrate. Returns True on success."""
    from daily_update import update
    conn = db()
    try:
        update(conn, log=st.write, retries=retries)
    except Exception as e:  # network failures etc. — show, don't crash the app
        st.error(f"{type(e).__name__}: {e}")
        return False
    finally:
        conn.close()
    st.cache_resource.clear()
    st.cache_data.clear()
    return True


def auto_enabled():
    return os.getenv('NBA_AUTO_UPDATE', '1') != '0'


def auto_update():
    """When the app opens, pull the latest games (which also settles tracked bets) in the
    background, so the page shows the saved data straight away and refreshes once new games are
    in. Only when something is due (a scheduled game has finished since the last load, or the
    schedule/rosters are old), once per browser session, and not again within AUTO_UPDATE_MINUTES
    of the last try. NBA_AUTO_UPDATE=0 turns it off."""
    if st.session_state.get('auto_update_done') or not auto_enabled():
        return
    st.session_state['auto_update_done'] = True
    from daily_update import update_due
    conn = db()
    try:
        last = get_meta(conn, 'last_update_attempt')
        now = datetime.now(timezone.utc)
        if last and now - datetime.fromisoformat(last) < timedelta(minutes=AUTO_UPDATE_MINUTES):
            return
        if not update_due(conn):
            return
        # recorded before trying, so being offline doesn't retry on every page open
        set_meta(conn, 'last_update_attempt', now.isoformat(timespec='seconds'))
    finally:
        conn.close()
    background.start(retries=2)
    st.session_state['bg_watch'] = data_version()


@st.fragment(run_every=2)
def update_watcher():
    """Sidebar note while the background update runs; reruns the page with the new data after."""
    if background.status()['running']:
        st.caption(":material/sync: Pulling new games in the background. Everything works meanwhile.")
        return
    if 'bg_watch' not in st.session_state:     # already handled; the rerun below stops this timer
        return
    before = st.session_state.pop('bg_watch')
    err = background.status()['error']
    slate.clear()
    if err:
        st.session_state['toast'] = "Couldn't pull new games, so the saved data is in use. Update data retries."
    elif data_version() != before:
        player_options.clear()
        st.session_state['toast'] = "New games loaded and tracked bets settled."
    st.rerun()                                  # full rerun: shows the new data and ends the fragment


def _api_call(fn, *args):
    """Call The Odds API, showing failures and credit usage instead of crashing the app."""
    try:
        result, usage = fn(*args)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return None
    st.session_state['odds_api_usage'] = odds_fetcher.usage_text(usage)
    return result


# ---------------------------------------------------------------------------
# Sidebar: data status + settings
# ---------------------------------------------------------------------------

auto_update()
version = data_version()
last_game, n_player_games, *_ = version

with st.sidebar:
    ui.show('<div class="brand">NBA Betting Model<small>Props · spreads · totals · moneylines</small></div>')
    if n_player_games:
        days = (date.today() - date.fromisoformat(last_game)).days
        fresh = ui.pill("Data current", "pos") if days <= 1 else ui.pill(f"Data {days} days old", "warn")
        ui.show(f"{fresh}")
        st.caption(f"Games through **{last_game}** · {n_player_games:,} player-games")
    elif 'bg_watch' in st.session_state:
        st.info("Downloading the last 3 seasons (about a minute). The page fills in when it's done.")
    else:
        st.warning("No data yet. Press **Update data** to download the last 3 seasons (~1 minute).")
    if 'bg_watch' in st.session_state:
        update_watcher()

    conn = db()
    refreshed = get_meta(conn, 'last_update')
    conn.close()
    if refreshed:
        local = datetime.fromisoformat(refreshed).astimezone()
        st.caption(f"Last pulled {local:%b %d, %I:%M %p}. Opening the app pulls new games automatically.")
    if st.button("Update data", width="stretch", icon=":material/refresh:",
                 help="Downloads the latest box scores from stats.nba.com, settles your tracked bets "
                      "and recalibrates the game model. The app also does this by itself when you open it."):
        with st.status("Updating...", expanded=True) as status:
            background.wait()            # don't run alongside an automatic update
            ok = pull_data()
            status.update(label="Data updated" if ok else "Update failed", state="complete" if ok else "error")
        if ok:
            st.rerun()

    st.divider()
    min_ev = st.slider("Minimum EV to bet", 0.0, 10.0, 3.0, 0.5, format="%.1f%%",
                       help="Only recommend a side whose expected value per $1 is at least this. "
                            "3% is a sensible floor: model error eats thinner edges.") / 100
    save = st.toggle("Record every evaluation for calibration", value=True,
                     help="Quietly saves everything you evaluate (tracked or not) so the Tracked bets tab can "
                          "measure the model against the market without the bias of only counting what you "
                          "chose to track. Your tracked list only shows what you press Track this on.")
    st.caption(f"Database: `{DB_PATH.name}`")

if not n_player_games:
    ui.app_header("No data yet")
    st.info("Press **Update data** in the sidebar to get started.")
    st.stop()

ui.app_header(f"Model data through {last_game}")
if 'goto_tab' in st.session_state:   # set after tracking; must be applied before the tabs are drawn
    st.session_state['main_tab'] = st.session_state.pop('goto_tab')
tab_prop, tab_game, tab_results, tab_kalshi = st.tabs(TABS, key="main_tab", on_change="rerun")
if 'toast' in st.session_state:
    st.toast(st.session_state.pop('toast'), icon=":material/check_circle:")



# ---------------------------------------------------------------------------
# Optional: import lines from The Odds API (only shown when ODDS_API_KEY is set)
# ---------------------------------------------------------------------------

def odds_api_prop_panel(players):
    with st.expander("Import a prop from The Odds API (optional)", icon=":material/download:"):
        c1, c2, c3 = st.columns([1, 1, 2])
        d = c1.date_input("Game date", value=date.today(), key="oa_pdate")
        book = c2.selectbox("Book", options=list(odds_fetcher.BOOKS), format_func=odds_fetcher.BOOKS.get,
                            key="oa_pbook")
        markets = c3.multiselect("Markets", options=list(odds_fetcher.PROP_MARKETS),
                                 default=['player_points'], key="oa_pmarkets",
                                 format_func=lambda m: STAT_LABELS[odds_fetcher.PROP_MARKETS[m]])
        # the game list is free, so it loads by itself for the chosen date (today by default)
        refresh = st.button("Reload games (free)", key="oa_events", icon=":material/refresh:")
        if refresh or st.session_state.get('oa_event_date') != d.isoformat():
            st.session_state['oa_event_date'] = d.isoformat()
            st.session_state.pop('oa_prop_rows', None)
            ev = _api_call(odds_fetcher.fetch_events, d.isoformat())
            if ev is None:
                st.session_state.pop('oa_event_list', None)
            else:
                st.session_state['oa_event_list'] = ev
        events = st.session_state.get('oa_event_list')
        if events is None:
            return
        if not events:
            st.info(f"No games on {d.isoformat()}.")
            return
        ev_ix = st.selectbox("Game", options=range(len(events)), key="oa_event",
                             format_func=lambda i: f"{events[i]['away']} @ {events[i]['home']}")
        if st.button(f"Load props ({len(markets)} credit{'s' * (len(markets) != 1)})", key="oa_props",
                     disabled=not markets):
            e = events[ev_ix]
            rows = _api_call(odds_fetcher.fetch_event_props, e['event_id'], e['game_date'], tuple(markets), book)
            if rows is not None:
                conn = db()
                try:
                    st.session_state['oa_prop_rows'] = odds_fetcher.match_players(conn, rows)
                finally:
                    conn.close()
        rows = st.session_state.get('oa_prop_rows')
        if rows is not None:
            if not rows:
                st.info("That book has no two-sided props for this game yet.")
            else:
                i = st.selectbox("Prop", options=range(len(rows)), key="oa_prop", format_func=lambda i: (
                    f"{rows[i]['player']} — {STAT_LABELS[rows[i]['stat']]} {rows[i]['line']:g} · "
                    f"O {rows[i]['over_odds']:+d} / U {rows[i]['under_odds']:+d}"
                    + ("" if rows[i]['player_id'] in players else "  (not in database)")))
                if st.button("Evaluate imported prop", type="primary", key="oa_eval_prop"):
                    r = rows[i]
                    if r['player_id'] not in players:
                        st.error(f"Couldn't match {r['player']} to a player in the database.")
                    else:
                        ctx = prop_context(r['game_date'], version)
                        hist = ctx.player_history(r['player_id'])
                        team = hist.iloc[-1]['team'] if hist is not None else None
                        opp = r['away'] if team == r['home'] else r['home'] if team == r['away'] else None
                        run_prop(r['player_id'], r['stat'], r['line'], r['over_odds'], r['under_odds'],
                                 opp, None, r['game_date'], book=r['book'])
        if st.session_state.get('odds_api_usage'):
            st.caption(st.session_state['odds_api_usage'])


def odds_api_game_panel():
    with st.expander("Import game lines from The Odds API (optional)", icon=":material/download:"):
        c1, c2 = st.columns(2)
        d = c1.date_input("Game date", value=date.today(), key="oa_gdate")
        book = c2.selectbox("Book", options=list(odds_fetcher.BOOKS), format_func=odds_fetcher.BOOKS.get,
                            key="oa_gbook")
        if st.button("Load lines (3 credits)", key="oa_lines"):
            games = _api_call(odds_fetcher.fetch_game_lines, book, d.isoformat())
            if games is not None:
                st.session_state['oa_game_list'] = games
        games = st.session_state.get('oa_game_list')
        if games is not None:
            games = [g for g in games if g['home'] and g['away']]
            if not games:
                st.info("No games with lines at that book on that date.")
            else:
                def glabel(i):
                    g = games[i]
                    parts = [f"{g['away']} @ {g['home']}"]
                    if 'spread' in g:
                        parts.append(f"{g['home']} {g['spread']:+g}")
                    if 'total' in g:
                        parts.append(f"total {g['total']:g}")
                    if 'ml' in g:
                        parts.append(f"ML {g['ml'][0]:+d}/{g['ml'][1]:+d}")
                    return " · ".join(parts)
                i = st.selectbox("Game", options=range(len(games)), format_func=glabel, key="oa_game")
                if st.button("Evaluate imported game", type="primary", key="oa_eval_game"):
                    g = games[i]
                    run_game(g['home'], g['away'], g['game_date'], g.get('spread'), g.get('spread_odds'),
                             g.get('total'), g.get('total_odds'), g.get('ml'), book=g['book'])
                    # show it as the selected game, on its date (the date picker isn't drawn yet)
                    st.session_state['g_sel'] = (g['away'], g['home'], g['game_date'])
                    st.session_state['gdate'] = date.fromisoformat(g['game_date'])
        if st.session_state.get('odds_api_usage'):
            st.caption(st.session_state['odds_api_usage'])


# ---------------------------------------------------------------------------
# Player prop
# ---------------------------------------------------------------------------

def prop_chart(r):
    mean, var, line = r['mean'], r['var'], r['line']
    dist = count_distribution(mean, var)
    lo, hi = int(max(dist.ppf(0.005), 0)), int(dist.ppf(0.995)) + 1
    k = np.arange(lo, hi + 1)
    df = pd.DataFrame({'value': k, 'probability': dist.pmf(k)})
    df['side'] = np.where(df['value'] > line, 'Over', np.where(df['value'] < line, 'Under', 'Push'))
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:O', title=STAT_LABELS.get(r['stat'], r['stat'])),
        y=alt.Y('probability:Q', axis=alt.Axis(format='%'), title='Chance'),
        color=alt.Color('side:N', scale=alt.Scale(domain=['Under', 'Push', 'Over'],
                                                  range=['#DC6B6B', '#B6BFCC', '#2F9E63']),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['value', alt.Tooltip('probability:Q', format='.1%'), 'side'],
    )
    return bars.properties(height=260)


with tab_prop:
    players = player_options(version)
    teams = team_options()
    if odds_fetcher.enabled():
        odds_api_prop_panel(players)
    with st.form("prop"):
        c1, c2, c3 = st.columns([2.2, 1.3, 1])
        player_id = c1.selectbox("Player", options=list(players), format_func=players.get, index=None,
                                 placeholder="Type a name...")
        stat = c2.selectbox("Stat", options=list(STAT_LABELS), format_func=lambda s: f"{STAT_LABELS[s]} ({s})")
        line = c3.number_input("Line", min_value=0.0, value=None, step=0.5, placeholder="e.g. 24.5")

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1.6, 1.2, 1.2])
        over_odds = c1.number_input("Over odds", value=-110, step=5)
        under_odds = c2.number_input("Under odds", value=-110, step=5)
        opp = c3.selectbox("Opponent", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                           index=None, placeholder="Optional")
        game_date = c4.date_input("Game date", value=date.today())
        minutes = c5.number_input("Minutes (optional)", min_value=0.0, max_value=48.0, value=None, step=1.0,
                                  placeholder="model", help="Override projected minutes, e.g. for a known "
                                  "minutes limit or a starter being out.")
        go = st.form_submit_button("Evaluate prop", type="primary", width="stretch")

    if go:
        problems = [m for ok, m in ((player_id is not None, "Pick a player."), (line is not None, "Enter a line."))
                    if not ok]
        problems += bad_odds((over_odds, under_odds))
        if problems:
            for m in problems:
                st.error(m)
        else:
            run_prop(player_id, stat, line, over_odds, under_odds, opp, minutes, game_date.isoformat())

    r = st.session_state.get('prop_result')
    if r:
        vs = f" vs {r['opponent']}" if r['opponent'] else ''
        ui.title_row(r['player_name'], f"{r['team']} · {STAT_LABELS.get(r['stat'], r['stat'])} {r['line']:g}{vs} · "
                     f"{r['game_date']}", abbr=r['team'])
        if r['pick'] == 'PASS':
            lean = 'OVER' if r['p_over'] >= r['p_under'] else 'UNDER'
            ui.verdict(False, f"No bet on {r['line']:g}",
                       meta=f"Neither side clears <b>+{r['min_ev']:.1%}</b> EV. The model leans {lean.lower()}.")
        else:
            odds = r['over_odds'] if r['pick'] == 'OVER' else r['under_odds']
            ev = r['ev_over'] if r['pick'] == 'OVER' else r['ev_under']
            p_side = r['p_over'] if r['pick'] == 'OVER' else r['p_under']
            ui.verdict(True, f"{r['pick'].title()} {r['line']:g}", fmt_odds(odds),
                       f"EV <b>{ev:+.1%}</b> · model <b>{p_side:.0%}</b> · stake "
                       f"<b>{r['kelly'] / 4:.1%}</b> of bankroll (quarter Kelly)")
        for w in r['warnings']:
            st.warning(w)
        side = r['pick'] if r['pick'] != 'PASS' else ('OVER' if r['p_over'] >= r['p_under'] else 'UNDER')
        side_odds = r['over_odds'] if side == 'OVER' else r['under_odds']
        conn = db()
        try:
            track_button(prop_log_id(conn, r, r['book'], tracked_only=True),
                         f"{side} {r['line']:g} ({fmt_odds(side_odds)})"
                         + (" (lean, no edge)" if r['pick'] == 'PASS' else ""), "track_prop",
                         lambda: log_prop(conn, r, r['book'], tracked=True), lambda i: untrack_prop(conn, i))
        finally:
            conn.close()

        m = st.columns(4)
        m[0].metric("Projection", f"{r['mean']:.1f}", help=f"Standard deviation {r['sd']:.1f}")
        m[1].metric("Fair line", f"{r['fair_line']:g}", help="The x.5 line where over and under are closest to 50/50")
        m[2].metric("Projected minutes", f"{r['minutes_proj']:.1f}")
        sa = f"{r['season_avg']:.1f}" if r['season_avg'] is not None else "–"
        m[3].metric("Season avg", sa, help=f"{r['season_games']} games this season · last 10: "
                                           f"{r['last10_avg']:.1f} · last 5: {r['last5_avg']:.1f}")

        left, right = st.columns([1, 1.3])
        with left:
            pick = {'OVER': 0, 'UNDER': 1}.get(r['pick'])
            ui.odds_table([('Over', fmt_odds(r['over_odds']), pct(r['p_over']), pct(r['fair_over']), r['ev_over']),
                           ('Under', fmt_odds(r['under_odds']), pct(r['p_under']), pct(r['fair_under']),
                            r['ev_under'])]
                          + ([('Push', '–', pct(r['p_push']), '–', '–')] if r['p_push'] > 0 else []), pick)
            opp_txt = ', '.join(f"{c} ×{f:.3f}" for c, f in r['opp_factors'].items())
            st.caption(f"Bookmaker margin on this prop: {r['vig']:.1%}. Opponent adjustment: {opp_txt}. "
                       f"Data through {r['last_game']}.")
        with right:
            st.altair_chart(prop_chart(r), width="stretch")


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

def pick_game(away, home):
    """Select a game (from the list or a custom matchup) on the chosen date and show its
    prediction right away."""
    st.session_state.update(g_sel=(away, home, st.session_state['gdate'].isoformat()), g_auto=True)
    st.session_state.pop('game_result', None)


def load_custom():
    away, home = st.session_state.get('c_away'), st.session_state.get('c_home')
    if not away or not home:
        st.session_state['c_err'] = "Pick both teams."
    elif away == home:
        st.session_state['c_err'] = "Home and away must be different teams."
    else:
        pick_game(away, home)


def close_game():
    st.session_state.pop('g_sel', None)
    st.session_state.pop('game_result', None)


def day_label(d):
    return f"{d:%a %b} {d.day}"


def jump_to(d):
    st.session_state['gdate'] = date.fromisoformat(d)


def pick_label(pred):
    fav = pred['home'] if pred['p_home_win'] >= 0.5 else pred['away']
    return f"{fav} to win ({pred['away']} @ {pred['home']})"


def track_pick_cb(game_date, pred):
    conn = db()
    try:
        track_pick(conn, game_date, pred)
    finally:
        conn.close()
    tracked_go(f"Tracking {pick_label(pred)}")


def untrack_cb(row_id):
    conn = db()
    try:
        untrack_game_bet(conn, row_id)
    finally:
        conn.close()


def tracked_picks(game_date):
    """{(home, away): row id} of the winner picks tracked on `game_date`."""
    conn = db()
    try:
        return {(h, a): i for i, h, a in conn.execute(
            "SELECT id, home, away FROM game_log WHERE market = ? AND tracked = 1 AND game_date = ?",
            (WINNER, game_date))}
    finally:
        conn.close()


def slate_panel(gdate):
    """The day's games from the schedule, each with the model's pick; clicking one loads it below."""
    games, next_day, err = slate(gdate.isoformat())
    if games.empty:
        if err:
            st.warning(f"Couldn't load the schedule from stats.nba.com ({err}). Add a custom matchup below.")
        msg = f"No games on {day_label(gdate)}."
        if next_day:
            c1, c2 = st.columns([3, 1])
            c1.caption(msg + f" Next game day: {day_label(date.fromisoformat(next_day))}.")
            c2.button("Go to next game day", on_click=jump_to, args=(next_day,), width="stretch")
        else:
            st.caption(msg)
        return
    try:
        ctx = game_context(gdate.isoformat(), version)
    except Exception:
        ctx = None
    st.markdown(f"**{len(games)} game{'s' * (len(games) != 1)} on {day_label(gdate)}**: track the "
                "model's pick, or open a game to check who's playing and add betting lines")
    picks = tracked_picks(gdate.isoformat())
    cols = st.columns(3)
    for i, g in enumerate(games.itertuples()):
        pred = None
        with cols[i % 3].container(border=True):
            ui.show(f'<div class="title-row">{ui.team(g.away, 26)}<span class="s">at</span>'
                    f'{ui.team(g.home, 26)}</div>')
            st.caption(" · ".join(x for x in (g.status, g.label) if x))
            try:
                outs = report_outs(ctx, gdate.isoformat(), g.home, g.away)
                pred = ctx.evaluate(g.home, g.away, **outs)
                pred['lineups'] = outs
                p = pred['p_home_win']
                fav, pf = (g.home, p) if p >= 0.5 else (g.away, 1 - p)
                st.markdown(f"Model: **{fav}** {pf:.0%}")
                why = explain_game(ctx, pred, **outs)
                st.caption(why['summary'])
                with st.expander(f"Why {fav}?"):
                    st.markdown("\n".join(f"- {d}" for d in why['details']))
            except Exception:  # no ratings for a team yet, etc.
                pred = None
                st.markdown("Model: —")
            b1, b2 = st.columns(2)
            b1.button("Open", key=f"slate_{g.game_id}", on_click=pick_game, args=(g.away, g.home),
                      width="stretch", icon=":material/open_in_full:")
            tid = picks.get((g.home, g.away))
            if tid:
                b2.button("Tracked", key=f"slate_untrack_{g.game_id}", on_click=untrack_cb, args=(tid,),
                          width="stretch", icon=":material/bookmark_added:", type="primary",
                          help="You're tracking the model's pick for this game. Click to stop.")
            elif pred is not None:
                b2.button("Track pick", key=f"slate_track_{g.game_id}", on_click=track_pick_cb,
                          args=(gdate.isoformat(), pred), width="stretch", icon=":material/bookmark_add:",
                          help=f"Follow the model's pick ({pick_label(pred)}) in Tracked bets. It settles "
                               "by itself after the game.")


with tab_game:
    teams = team_options()
    if odds_fetcher.enabled():
        odds_api_game_panel()
    st.session_state.setdefault('gdate', date.today())   # set here so callbacks can move it
    gdate = st.date_input("Game date", key="gdate")
    sel = st.session_state.get('g_sel')
    if sel and sel[2] != gdate.isoformat():   # picked on another date
        close_game()
        sel = None
    if not sel:
        slate_panel(gdate)
        with st.expander("Add a custom matchup", icon=":material/add:"):
            c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
            c1.selectbox("Away team", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                         index=None, placeholder="Away", key="c_away")
            c2.selectbox("Home team", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                         index=None, placeholder="Home", key="c_home")
            c3.button("Load matchup", on_click=load_custom, width="stretch")
            if st.session_state.get('c_err'):
                st.error(st.session_state.pop('c_err'))
    else:
        # an opened game takes the list's place, so it's obvious something opened
        away, home, _ = sel
        h1, h2 = st.columns([4, 1], vertical_alignment="center")
        with h1:
            ui.show(f'<div class="title-row">{ui.team(away, 34, teams.get(away, away))}<span class="s">at</span>'
                    f'{ui.team(home, 34, teams.get(home, home))}</div>')
        h2.button("All games", icon=":material/arrow_back:", on_click=close_game, width="stretch")
        pred_area = st.container()   # the prediction sits above the checklists; filled in below
        lineups = lineup_inputs(away, home, gdate.isoformat())
        if st.session_state.pop('g_auto', False):  # just opened: show its prediction
            run_game(home, away, gdate.isoformat(), lineups=lineups)

        with st.form("game"):
            st.caption("Lines are optional. Leave them blank to just get the predicted score.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Spread** (home team's line)")
                spread = st.number_input("Home spread", value=None, step=0.5, placeholder="e.g. -5.5")
                s1, s2 = st.columns(2)
                spread_home = s1.number_input("Home odds", value=-110, step=5)
                spread_away = s2.number_input("Away odds", value=-110, step=5)
            with c2:
                st.markdown("**Total**")
                total = st.number_input("Total points", min_value=0.0, value=None, step=0.5, placeholder="e.g. 224.5")
                t1, t2 = st.columns(2)
                total_over = t1.number_input("Over odds", value=-110, step=5, key="tover")
                total_under = t2.number_input("Under odds", value=-110, step=5, key="tunder")
            with c3:
                st.markdown("**Moneyline**")
                m1, m2 = st.columns(2)
                ml_home = m1.number_input("Home ML", value=None, step=5, placeholder="e.g. -200")
                ml_away = m2.number_input("Away ML", value=None, step=5, placeholder="e.g. +170")
            go_game = st.form_submit_button("Evaluate game", type="primary", width="stretch")

        if go_game:
            problems = []
            if (ml_home is None) != (ml_away is None):
                problems.append("Enter both moneyline prices (or neither).")
            prices = [spread_home, spread_away, total_over, total_under] + \
                [x for x in (ml_home, ml_away) if x is not None]
            problems += bad_odds(prices)
            if problems:
                for m in problems:
                    st.error(m)
            else:
                run_game(home, away, gdate.isoformat(),
                         spread, (int(spread_home), int(spread_away)) if spread is not None else None,
                         total, (int(total_over), int(total_under)) if total is not None else None,
                         (int(ml_home), int(ml_away)) if ml_home is not None else None,
                         lineups=lineups)

        res = st.session_state.get('game_result')
        if res:
            pred, evals, stale, calibrated, gd, book = res
            h, a = pred['home'], pred['away']
            with pred_area:
                ui.matchup(a, h, team_options(), pred, gd)
                notes = [f"Projected pace {pred['pace']:.1f} possessions."]
                adj = adjustment_text(pred)
                if adj:
                    notes.append(f"Adjustments to the {h} margin: {adj}")
                if not calibrated:
                    notes.append("Score spreads use default values until the first data update calibrates them.")
                st.caption(" ".join(notes))
                conn = db()
                try:
                    track_button(pick_id(conn, gd, h, a), pick_label(pred), "track_pick",
                                 lambda: track_pick(conn, gd, pred), lambda i: untrack_game_bet(conn, i))
                finally:
                    conn.close()
                try:
                    why = explain_game(game_context(gd, version), pred, **pred.get('lineups', {}))
                    with st.container(border=True):
                        st.markdown(f"**Why:** {why['summary']}")
                        st.markdown("\n".join(f"- {d}" for d in why['details']))
                except Exception:  # the explanation is a nice-to-have; the numbers above still stand
                    pass
                if stale:
                    st.warning(stale)

            names = {'SPREAD': (h, a), 'TOTAL': ('Over', 'Under'), 'MONEYLINE': (h, a)}
            for col, e in zip(st.columns(max(len(evals), 1)), evals):
                sa, sb = names[e['market']]
                if e['market'] == 'SPREAD':
                    title = "Spread"
                    sa, sb = f"{h} {e['line']:+g}", f"{a} {-e['line']:+g}"
                elif e['market'] == 'TOTAL':
                    title = f"Total {e['line']:g}"
                    sa, sb = f"Over {e['line']:g}", f"Under {e['line']:g}"
                else:
                    title = "Moneyline"
                    sa, sb = f"{h} to win", f"{a} to win"
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    if e['pick'] == 'PASS':
                        lean = sa if e['p_a'] >= e['p_b'] else sb
                        ui.verdict(False, "No bet", meta=f"Neither side clears <b>+{min_ev:.1%}</b> EV. "
                                                        f"Model leans {ui.esc(lean)}.")
                    else:
                        side, odds, ev, p = ((sa, e['odds_a'], e['ev_a'], e['p_a']) if e['pick'] == 'A'
                                             else (sb, e['odds_b'], e['ev_b'], e['p_b']))
                        ui.verdict(True, side, fmt_odds(odds), f"EV <b>{ev:+.1%}</b> · model <b>{p:.0%}</b> · stake "
                                                             f"<b>{e['kelly'] / 4:.1%}</b> (quarter Kelly)",
                                   abbr=None if e['market'] == 'TOTAL' else (h if e['pick'] == 'A' else a))
                    if e['warning']:
                        st.warning(e['warning'])
                    ui.odds_table([(sa, fmt_odds(e['odds_a']), pct(e['p_a']), pct(e['fair_a']), e['ev_a']),
                                   (sb, fmt_odds(e['odds_b']), pct(e['p_b']), pct(1 - e['fair_a']), e['ev_b'])],
                                  {'A': 0, 'B': 1}.get(e['pick']))
                    lean = e['pick'] if e['pick'] != 'PASS' else ('A' if e['p_a'] >= e['p_b'] else 'B')
                    label = (f"{sa if lean == 'A' else sb} ({fmt_odds(e['odds_a'] if lean == 'A' else e['odds_b'])})"
                             + (" (lean, no edge)" if e['pick'] == 'PASS' else ""))
                    conn = db()
                    try:
                        track_button(game_log_id(conn, gd, pred, e, book, tracked_only=True), label,
                                     f"track_{e['market']}",
                                     lambda e=e: log_game_evals(conn, gd, pred, [e], book, tracked=True),
                                     lambda i: untrack_game_bet(conn, i))
                    finally:
                        conn.close()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def manual_settle_section():
    """Enter outcomes by hand (or correct automatic ones). Runs before the summaries are
    computed so they reflect a save immediately."""
    def _done(msg):
        # rerun so the dropdowns and summaries reflect the change; show the message after
        st.session_state['settle_msg'] = msg
        st.rerun()

    if 'settle_msg' in st.session_state:
        st.success(st.session_state.pop('settle_msg'))
    st.caption("Enter a result yourself instead of waiting for the data, correct one, stop tracking a bet, or "
               "delete it. Automatic settling never overwrites a result you entered.")
    show_all = st.checkbox("Include already-settled entries (to correct or delete them)", key="settle_all")
    kind = st.radio("What", ["Prop", "Game"], horizontal=True, key="settle_kind", label_visibility="collapsed")
    conn = db()
    try:
        if kind == "Prop":
            q = ("SELECT id, game_date, player_name, stat, line, pick, actual, result FROM prop_log WHERE tracked = 1 "
                 + ("" if show_all else "AND result IS NULL ") + "ORDER BY game_date DESC, id DESC")
            rows = {r[0]: r for r in conn.execute(q).fetchall()}
            if not rows:
                st.info("No pending tracked props." if not show_all else "No tracked props yet.")
                return

            def label(i):
                _, d, name, stat, line, pick, actual, result = rows[i]
                status = f"{result} (actual {actual:g})" if actual is not None else (result or "unsettled")
                return f"{d} · {name} {stat} {line:g} · pick {pick} · {status}"

            with st.form("settle_prop"):
                pid = st.selectbox("Prop", options=list(rows), format_func=label)
                c1, c2 = st.columns([1, 2])
                actual = c1.number_input("Actual stat", min_value=0.0, value=None, step=1.0,
                                         placeholder="e.g. 27")
                action = c2.radio("Action", ["Use actual stat", "Didn't play (void)", "Clear result",
                                             "Stop tracking", "Delete entry"], horizontal=True)
                confirm = st.checkbox("Yes, delete it (only needed for Delete entry)")
                if st.form_submit_button("Save", type="primary"):
                    if action == "Delete entry":
                        if not confirm:
                            st.error("Tick “Yes, delete it” to confirm.")
                        elif delete_prop(conn, pid):
                            _done(f"Deleted: {rows[pid][2]} {rows[pid][3]} {rows[pid][4]:g} ({rows[pid][1]})")
                    elif action == "Stop tracking":
                        untrack_prop(conn, pid)
                        _done(f"Stopped tracking: {rows[pid][2]} {rows[pid][3]} {rows[pid][4]:g} ({rows[pid][1]})")
                    elif action == "Use actual stat" and actual is None:
                        st.error("Enter the actual stat.")
                    else:
                        res = settle_prop_manual(conn, pid, actual=actual if action == "Use actual stat" else None,
                                                 dnp=action == "Didn't play (void)")
                        _done(f"Saved: {rows[pid][2]} {rows[pid][3]} {rows[pid][4]:g} → "
                                   f"{res or 'unsettled'}")
        else:
            q = ("SELECT game_date, home, away, COUNT(*), SUM(result IS NULL), MAX(away_pts), MAX(home_pts) "
                 "FROM game_log WHERE tracked = 1 GROUP BY game_date, home, away "
                 + ("" if show_all else "HAVING SUM(result IS NULL) > 0 ") + "ORDER BY game_date DESC")
            games = {f"{r[0]}|{r[1]}|{r[2]}": r for r in conn.execute(q).fetchall()}
            if not games:
                st.info("No pending tracked game bets." if not show_all else "No tracked game bets yet.")
                return

            def glabel(k):
                d, home, away, n, unsettled, ap, hp = games[k]
                status = "unsettled" if unsettled else f"final {away} {ap} – {home} {hp}"
                return f"{d} · {away} @ {home} · {n} market(s) · {status}"

            with st.form("settle_game"):
                key = st.selectbox("Game", options=list(games), format_func=glabel)
                d, home, away = key.split("|")
                c1, c2 = st.columns(2)
                ap = c1.number_input(f"{away} (away) final points", min_value=0, value=None, step=1)
                hp = c2.number_input(f"{home} (home) final points", min_value=0, value=None, step=1)
                action = st.radio("Action", ["Save final score", "Stop tracking this game",
                                             "Delete this game's bets"], horizontal=True)
                confirm = st.checkbox("Yes, delete them (only needed for Delete)")
                if st.form_submit_button("Save", type="primary"):
                    if action == "Delete this game's bets":
                        if not confirm:
                            st.error("Tick “Yes, delete them” to confirm.")
                        else:
                            n = delete_game_bets(conn, d, home, away)
                            _done(f"Deleted {n} logged market(s) for {away} @ {home} ({d}).")
                    elif action == "Stop tracking this game":
                        ids = [i for (i,) in conn.execute("SELECT id FROM game_log WHERE tracked = 1 AND "
                                                          "game_date=? AND home=? AND away=?", (d, home, away))]
                        for i in ids:
                            untrack_game_bet(conn, i)
                        _done(f"Stopped tracking {len(ids)} market(s) for {away} @ {home} ({d}).")
                    elif ap is None or hp is None:
                        st.error("Enter both scores.")
                    else:
                        try:
                            n = settle_game_manual(conn, d, home, away, int(hp), int(ap))
                            _done(f"Saved {away} {int(ap)} – {home} {int(hp)}: settled {n} market(s).")
                        except ValueError as e:
                            st.error(str(e))
    finally:
        conn.close()


def tracked_metrics(s):
    acc, rec = s['accuracy'], s['record']
    c = st.columns(5)
    c[0].metric("Tracked", s['tracked'], help=f"{s['pending']} waiting on the game")
    c[1].metric("Model's side hit", f"{acc['rate']:.0%}" if acc['rate'] is not None else "–",
                help=f"{acc['hits']} hit, {acc['misses']} missed (pushes and voids left out). "
                     "Includes tracked passes, judged on the side the model leaned to.")
    c[2].metric("Record (W-L-P)", f"{rec['wins']}-{rec['losses']}-{rec['pushes']}" if rec else "–",
                help="Tracked bets only (a tracked pass isn't a bet)")
    c[3].metric("Profit", f"{rec['profit']:+.2f} u" if rec else "–", help="1 unit staked per bet")
    c[4].metric("ROI", f"{rec['roi']:+.1%}" if rec else "–",
                help=f"Model's claimed average EV on these: {rec['claimed_ev']:+.1%}" if rec else None)


def add_tracked_section():
    """Track a game or prop straight from the Tracked bets tab, without opening it first."""
    kind = st.segmented_control("Add", ["Game", "Prop"], default="Game", key="add_kind",
                                label_visibility="collapsed") or "Game"
    if kind == "Game":
        c1, c2 = st.columns([1, 3])
        d = c1.date_input("Date", value=date.today(), key="add_gdate")
        games, _, _ = slate(d.isoformat())
        if games.empty:
            c2.selectbox("Game", options=[], placeholder=f"No games on {day_label(d)}", key="add_no_game")
            return
        gi = c2.selectbox("Game", options=range(len(games)), key="add_game", format_func=lambda i: " · ".join(
            x for x in (f"{games.away[i]} @ {games.home[i]}", games.status[i], games.label[i]) if x))
        if gi is None or gi >= len(games):   # e.g. a pick from a date with more games
            return
        g = games.iloc[gi]
        market = st.radio("What to track", ["Model's pick (no line)", "Spread", "Total", "Moneyline"],
                          horizontal=True, key="add_market")
        line, odds = None, (-110, -110)
        if market != "Model's pick (no line)":
            c1, c2, c3 = st.columns(3)
            if market != "Moneyline":
                line = c1.number_input(f"{g.home} spread" if market == "Spread" else "Total points",
                                       value=None, step=0.5, key="add_line",
                                       placeholder="e.g. -5.5" if market == "Spread" else "e.g. 224.5")
            a_lbl, b_lbl = (("Over odds", "Under odds") if market == "Total"
                            else (f"{g.home} odds", f"{g.away} odds"))
            odds = (c2.number_input(a_lbl, value=-110 if market != "Moneyline" else None, step=5, key="add_oa"),
                    c3.number_input(b_lbl, value=-110 if market != "Moneyline" else None, step=5, key="add_ob"))
        if st.button("Track", type="primary", icon=":material/bookmark_add:", key="add_game_go"):
            if market in ("Spread", "Total") and line is None:
                return st.error("Enter the line.")
            if None in odds:
                return st.error("Enter both prices.")
            if bad_odds(odds):
                return st.error(bad_odds(odds)[0])
            try:
                ctx = game_context(d.isoformat(), version)
                pred = ctx.evaluate(g.home, g.away, line if market == "Spread" else None,
                                    line if market == "Total" else None)
                odds = tuple(int(o) for o in odds)
                evals = evaluate_markets(pred, *((line, odds) if market == "Spread" else (None, None)),
                                         *((line, odds) if market == "Total" else (None, None)),
                                         odds if market == "Moneyline" else None, min_ev)
            except Exception as e:
                return st.error(f"{type(e).__name__}: {e}")
            conn = db()
            try:
                if market == "Model's pick (no line)":
                    track_pick(conn, ctx.game_date, pred)
                    msg = pick_label(pred)
                else:
                    log_game_evals(conn, ctx.game_date, pred, evals, tracked=True)
                    msg = f"{g.away} @ {g.home} {market.lower()}"
            finally:
                conn.close()
            st.session_state['toast'] = f"Tracking {msg}"
            st.rerun()
    else:
        players = player_options(version)
        teams = team_options()
        with st.form("add_prop"):
            c1, c2, c3 = st.columns([2.2, 1.3, 1])
            pid = c1.selectbox("Player", options=list(players), format_func=players.get, index=None,
                               placeholder="Type a name...", key="add_player")
            stat = c2.selectbox("Stat", options=list(STAT_LABELS), key="add_stat",
                                format_func=lambda s: f"{STAT_LABELS[s]} ({s})")
            line = c3.number_input("Line", min_value=0.0, value=None, step=0.5, placeholder="e.g. 24.5",
                                   key="add_pline")
            c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1.2])
            over = c1.number_input("Over odds", value=-110, step=5, key="add_over")
            under = c2.number_input("Under odds", value=-110, step=5, key="add_under")
            opp = c3.selectbox("Opponent", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                               index=None, placeholder="Optional", key="add_opp")
            d = c4.date_input("Game date", value=date.today(), key="add_pdate")
            go = st.form_submit_button("Track", type="primary", icon=":material/bookmark_add:")
        if go:
            if pid is None or line is None:
                return st.error("Pick a player and enter a line.")
            if bad_odds((over, under)):
                return st.error(bad_odds((over, under))[0])
            try:
                r = evaluate_prop(prop_context(d.isoformat(), version), pid, stat, float(line), int(over),
                                  int(under), opponent=opp, min_ev=min_ev)
            except Exception as e:
                return st.error(f"{type(e).__name__}: {e}")
            if r is None:
                return st.error("Not enough recent games to project this player.")
            conn = db()
            try:
                log_prop(conn, r, tracked=True)
            finally:
                conn.close()
            st.session_state['toast'] = f"Tracking {players[pid]} {stat} {line:g}"
            st.rerun()


TABLE_CONFIG = {
    'Date': st.column_config.TextColumn(width="small"),
    'Away logo': st.column_config.ImageColumn("", width="small"),
    'Home logo': st.column_config.ImageColumn("", width="small"),
    'Projection': st.column_config.NumberColumn("Proj", format="%.1f"),
    'Model %': st.column_config.NumberColumn("Model", format="%.0f%%"),
    'EV': st.column_config.NumberColumn(format="%+.1f%%"),
    'Actual': st.column_config.NumberColumn(format="%g"),
    'Profit (u)': st.column_config.NumberColumn("Units", format="%+.2f"),
}


def tracked_table(df, kind):
    """Tracked entries; select rows to stop tracking them or delete them."""
    view = df.drop(columns='id')
    view[['Model %', 'EV']] = view[['Model %', 'EV']].astype(float) * 100
    view['vs' if kind == 'props' else 'Final'] = view['vs' if kind == 'props' else 'Final'].replace('', None)
    if kind == 'games':
        view.insert(1, 'Away logo', df['Away'].map(ui.logo_uri))
        view.insert(3, 'Home logo', df['Home'].map(ui.logo_uri))
    ver = st.session_state.get('trk_ver', 0)   # bumped after a change so the old selection is dropped
    event = st.dataframe(view, hide_index=True, column_config=TABLE_CONFIG, on_select="rerun", placeholder="–",
                         selection_mode="multi-row", key=f"trk_{kind}_{ver}")
    ids = [int(df['id'].iloc[i]) for i in event.selection.rows]
    if not ids:
        st.caption("Select rows (the box at the left of each row) to remove them.")
        return
    n = f"{len(ids)} entr{'y' if len(ids) == 1 else 'ies'}"
    untrack, delete = (untrack_prop, delete_prop) if kind == 'props' else (untrack_game_bet, delete_game_bet)
    c1, c2, _ = st.columns([1.2, 1.2, 3])
    if c1.button(f"Remove {n}", key=f"rm_{kind}", icon=":material/bookmark_remove:", width="stretch",
                 help="Stop tracking them. Their evaluations stay saved for the calibration check."):
        done = untrack
    else:
        done = None
        with c2.popover(f"Delete {n}", icon=":material/delete:", width="stretch"):
            st.caption("Deletes them for good, including their calibration data.")
            if st.button("Yes, delete", key=f"del_{kind}", type="primary"):
                done = delete
    if done:
        conn = db()
        try:
            for i in ids:
                done(conn, i)
        finally:
            conn.close()
        st.session_state['trk_ver'] = ver + 1
        st.session_state['toast'] = f"{'Removed' if done is untrack else 'Deleted'} {n}"
        st.rerun()


with tab_results:
    with st.expander("Add a game or prop to track", icon=":material/add:"):
        add_tracked_section()
    summary_area = st.container()
    st.write("")
    with st.expander("Enter or fix a result by hand", icon=":material/edit_note:"):
        manual_settle_section()
    conn = db()
    try:
        ps, gs = prop_summary(conn), game_summary(conn)
        show = st.session_state.get('track_show') or 'All'
        props_df, games_df = tracked_props(conn, show), tracked_games(conn, show)
    finally:
        conn.close()

    with summary_area:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.caption("Press **Track pick** on a game, **Track this** on any prop or line, or add one above. Once "
                   "the game is over, the result fills in here by itself when the app pulls new games (on "
                   "opening it, or with **Update data**).")
        if c2.button("Check for results", width="stretch", key="check_results", icon=":material/refresh:"):
            with st.status("Pulling the latest games...", expanded=True) as status:
                ok = pull_data()
                status.update(label="Up to date" if ok else "Update failed", state="complete" if ok else "error")
            if ok:
                st.rerun()
        st.segmented_control("Show", ["All", "Pending", "Finished"], default="All", key="track_show",
                             label_visibility="collapsed")

        for title, summ, df, kind, noun in (("Props", ps, props_df, 'props', "props"),
                                           ("Games", gs, games_df, 'games', "games")):
            st.subheader(title)
            tracked_metrics(summ)
            if df.empty:
                st.caption("Nothing tracked yet." if show == 'All' else f"No {show.lower()} {noun}.")
            else:
                tracked_table(df, kind)

        cal = ps.get('calibration')
        if cal:
            better = cal['brier_model'] < cal['brier_market']
            st.subheader("Model vs market")
            ui.show(f"{ui.pill('Model better' if better else 'Market better', 'pos' if better else 'neg')} "
                    f"&nbsp;Brier score on <b>{cal['n']}</b> settled props you evaluated, tracked or not: model "
                    f"<b>{cal['brier_model']:.4f}</b>, market <b>{cal['brier_market']:.4f}</b> (lower is better)."
                    + (" <i>Too few to mean much yet.</i>" if cal['n'] < 100 else ""))
        st.caption("A real edge takes a few hundred bets to show up. Under ~100 the record is mostly luck.")


# ---------------------------------------------------------------------------
# Kalshi bot: the paper trades kalshi_bot.py made (it runs outside the app)
# ---------------------------------------------------------------------------

KALSHI_CONFIG = {
    'Price': st.column_config.NumberColumn(format="%.0f¢", help="Paid per contract; it pays $1 if it wins"),
    'Pays': st.column_config.NumberColumn(format="%.2fx", help="Payout multiplier: $1 / price"),
    'Qty': st.column_config.NumberColumn(help="Contracts bought"),
    'Cost': st.column_config.NumberColumn(format="$%.2f", help="Contracts × price + Kalshi's fee"),
    'Model %': st.column_config.NumberColumn("Model", format="%.0f%%"),
    'Kalshi %': st.column_config.NumberColumn("Kalshi", format="%.0f%%", help="Kalshi's mid price when it bet"),
    'EV': st.column_config.NumberColumn(format="%+.1f¢", help="Expected profit per contract after fees"),
    'CLV': st.column_config.NumberColumn(format="%+.1f¢", help="Closing-line value: our side's last Kalshi "
                                                               "price before tip-off minus the price paid. "
                                                               "Positive = the market moved our way"),
    'Profit': st.column_config.NumberColumn(format="$%+.2f"),
    'Staked': st.column_config.NumberColumn(format="$%.2f"),
    'ROI': st.column_config.NumberColumn(format="%+.1f%%"),
}


def et_time(iso):
    t = pd.Timestamp(iso).tz_convert('America/New_York')
    return f"{t:%b} {t.day}, {t.hour % 12 or 12}:{t:%M} {'am' if t.hour < 12 else 'pm'} ET"


with tab_kalshi:
    conn = db()
    try:
        rep = kalshi_bot.report(conn)
        start = float(get_meta(conn, kalshi_bot.BANKROLL_KEY) or 0)
        show = st.session_state.get('kalshi_show') or 'All'
        bets, days = kalshi_bot.paper_bets(conn, show), kalshi_bot.daily_summary(conn)
        flags, ran = kalshi_bot.flagged(conn, kalshi_bot.et_today()), kalshi_bot.last_run(conn)
    finally:
        conn.close()

    st.caption("Paper trades only: no money moves. The bot runs outside the app (`python kalshi_bot.py run`, "
               "ideally every 15 minutes on game days); this tab shows what it did. Bets settle when the app "
               "pulls new games.")
    ui.show(f"{ui.pill('Last run ' + et_time(ran), 'flat') if ran else ui.pill('Not run yet', 'neg')}")
    if rep['bets'] == 0:
        st.info("No paper bets yet. Run `python kalshi_bot.py run --update` in the project folder.",
                icon=":material/info:")
    else:
        c = st.columns([1.5, 0.8, 1, 1, 1])
        c[0].metric("Paper bankroll", f"${rep['bankroll']:,.0f}",
                    delta=f"{rep['bankroll'] - start:+,.2f}" if start and rep['settled'] else None,
                    help=f"Started at ${start:,.2f}; changes as bets settle")
        c[1].metric("Bets", rep['bets'], help=f"{rep['bets'] - rep['settled']} waiting on the game")
        c[2].metric("Record (W-L)", f"{rep['wins']}-{rep['settled'] - rep['wins']}" if rep['settled'] else "–")
        c[3].metric("Profit", f"${rep['profit']:+,.2f}" if rep['settled'] else "–",
                    help=f"ROI {rep['roi']:+.1%} on ${rep['staked']:,.2f} settled" if rep['roi'] is not None else None)
        c[4].metric("Avg CLV", f"{rep['clv'] * 100:+.2f}¢" if rep['clv'] is not None else "–",
                    help=f"Over {rep['clv_n']} bets. The gate for real money is positive CLV over 300+ bets.")
        if 'brier_n' in rep:
            better = rep['brier_model'] < rep['brier_kalshi']
            ui.show(f"{ui.pill('Model better' if better else 'Kalshi better', 'pos' if better else 'neg')} "
                    f"&nbsp;Brier score on <b>{rep['brier_n']}</b> settled winner contracts, bet or not: model "
                    f"<b>{rep['brier_model']:.4f}</b>, Kalshi <b>{rep['brier_kalshi']:.4f}</b> (lower is better)."
                    + (" <i>Too few to mean much yet.</i>" if rep['brier_n'] < 200 else ""))

    if not flags.empty:
        st.subheader("Flagged today")
        st.caption("Model and Kalshi disagree too much to bet. Usually injury news the model hasn't seen.")
        st.dataframe(flags, hide_index=True)

    if rep['bets']:
        st.subheader("Paper bets")
        st.segmented_control("Show", ["All", "Pending", "Finished"], default="All", key="kalshi_show",
                             label_visibility="collapsed")
        if bets.empty:
            st.caption(f"No {show.lower()} bets.")
        else:
            st.dataframe(bets, hide_index=True, column_config=KALSHI_CONFIG, placeholder="–")
        st.subheader("By day")
        st.dataframe(days, hide_index=True, column_config=KALSHI_CONFIG, placeholder="–")
        st.caption("One night is noise. Judge the bot on CLV after a few hundred bets, not on the record.")
