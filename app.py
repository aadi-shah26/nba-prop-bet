"""
NBA betting model — local web app.

    streamlit run app.py          (or double-click start.command on a Mac)

Pick a player or game, type the line and the prices, get OVER / UNDER / PASS with the
numbers behind it. Optionally, with an ODDS_API_KEY set, lines can be imported from The Odds
API instead of typed. Evaluations are saved to your log and settled automatically when you
press "Update data".
"""

from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bet_signals import delete_prop, evaluate_prop, log_prop, settle_prop_manual
import odds_fetcher
from db import DB_PATH, connect, init_db, recent_players, season_for_date
from game_model import (GameContext, adjustment_text, delete_game_bets, evaluate_markets, log_game_evals,
                        settle_game_manual, stale_warning)
from projection_model import PropContext, count_distribution
from tracking import game_summary, prop_summary, recent_games, recent_props

st.set_page_config(page_title="NBA Betting Model", page_icon="🏀", layout="wide")

STAT_LABELS = {
    'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'FG3M': '3-pointers made',
    'PRA': 'Pts + Reb + Ast', 'PR': 'Pts + Reb', 'PA': 'Pts + Ast', 'RA': 'Reb + Ast',
    'STL': 'Steals', 'BLK': 'Blocks', 'SB': 'Steals + Blocks', 'TOV': 'Turnovers',
}


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
            conn.execute("SELECT COUNT(*) FROM model_params").fetchone()
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
    st.session_state['prop_result'] = r
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
        evals = evaluate_markets(pred, spread, spread_odds, total, total_odds, ml, min_ev)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return
    st.session_state['game_result'] = (pred, evals, stale_warning(ctx), ctx.calibrated, ctx.game_date)
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
    st.markdown("**Who's playing tonight?** Check the injury report. Untick anyone who's out, "
                "and tick anyone returning who hasn't played lately. Best players are listed first.")
    for col, team, side in zip(st.columns(2), (away, home), ('away', 'home')):
        r = ctx.roster(team)
        with col.container(border=True):
            st.markdown(f"**{team}**")
            for recent, heading in ((True, None), (False, "Haven't played lately")):
                group = r[r['recent'] == recent]
                if group.empty:
                    continue
                if heading:
                    st.caption(heading)
                for p in group.itertuples():
                    on = st.checkbox(p.player_name, value=recent, key=f"play_{team}_{p.player_id}_{game_date}",
                                     help=f"Value {p.value:.1f} (recent Game Score) · last played {p.last_game}")
                    if recent and not on:
                        res[f'{side}_out'].append(int(p.player_id))
                    elif on and not recent:
                        res[f'{side}_in'].append(int(p.player_id))
    return res


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

version = data_version()
last_game, n_player_games, _ = version

with st.sidebar:
    st.header("🏀 NBA Model")
    if n_player_games:
        days = (date.today() - date.fromisoformat(last_game)).days
        st.caption(f"Data through **{last_game}** ({days} days ago) · {n_player_games:,} player-games")
    else:
        st.warning("No data yet. Press **Update data** to download the last 3 seasons (~1 minute).")

    if st.button("🔄 Update data", width="stretch",
                 help="Downloads the latest box scores from stats.nba.com, settles your logged bets "
                      "and recalibrates the game model. Do this once a day."):
        from daily_update import update
        with st.status("Updating...", expanded=True) as status:
            try:
                conn = db()
                update(conn, log=st.write)
                conn.close()
                status.update(label="Data updated", state="complete")
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception as e:  # network failures etc. — show, don't crash the app
                status.update(label="Update failed", state="error")
                st.error(f"{type(e).__name__}: {e}")

    st.divider()
    min_ev = st.slider("Minimum EV to bet", 0.0, 10.0, 3.0, 0.5, format="%.1f%%",
                       help="Only recommend a side whose expected value per $1 is at least this. "
                            "3% is a sensible floor: model error eats thinner edges.") / 100
    save = st.toggle("Save evaluations to my log", value=True,
                     help="Logged bets are settled automatically on the next data update, "
                          "so the Results tab can show whether the model actually beats the market.")
    st.caption(f"Database: `{DB_PATH.name}`")

if not n_player_games:
    st.title("NBA Betting Model")
    st.info("Press **🔄 Update data** in the sidebar to get started.")
    st.stop()

tab_prop, tab_game, tab_results = st.tabs(["🎯 Player prop", "🏀 Game", "📒 My results"])



# ---------------------------------------------------------------------------
# Optional: import lines from The Odds API (only shown when ODDS_API_KEY is set)
# ---------------------------------------------------------------------------

def odds_api_prop_panel(players):
    with st.expander("📡 Import a prop from The Odds API (optional)"):
        c1, c2, c3 = st.columns([1, 1, 2])
        d = c1.date_input("Game date", value=date.today(), key="oa_pdate")
        book = c2.selectbox("Book", options=list(odds_fetcher.BOOKS), format_func=odds_fetcher.BOOKS.get,
                            key="oa_pbook")
        markets = c3.multiselect("Markets", options=list(odds_fetcher.PROP_MARKETS),
                                 default=['player_points'], key="oa_pmarkets",
                                 format_func=lambda m: STAT_LABELS[odds_fetcher.PROP_MARKETS[m]])
        if st.button("Load games (free)", key="oa_events"):
            ev = _api_call(odds_fetcher.fetch_events, d.isoformat())
            if ev is not None:
                st.session_state['oa_event_list'] = ev
                st.session_state.pop('oa_prop_rows', None)
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
    with st.expander("📡 Import game lines from The Odds API (optional)"):
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
                                                  range=['#d9534f', '#999999', '#2e9e5b']),
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
        st.subheader(f"{r['player_name']} ({r['team']}) — {STAT_LABELS.get(r['stat'], r['stat'])} "
                     f"{r['line']:g}{vs}")
        if r['pick'] == 'PASS':
            st.info(f"**PASS** — neither side clears +{r['min_ev']:.1%} EV")
        else:
            odds = r['over_odds'] if r['pick'] == 'OVER' else r['under_odds']
            ev = r['ev_over'] if r['pick'] == 'OVER' else r['ev_under']
            st.success(f"### BET {r['pick']} {r['line']:g} at {fmt_odds(odds)}  ·  EV {ev:+.1%}  ·  "
                       f"stake {r['kelly'] / 4:.1%} of bankroll (¼ Kelly)")
        for w in r['warnings']:
            st.warning(w)

        m = st.columns(4)
        m[0].metric("Projection", f"{r['mean']:.1f}", help=f"Standard deviation {r['sd']:.1f}")
        m[1].metric("Fair line", f"{r['fair_line']:g}", help="The x.5 line where over and under are closest to 50/50")
        m[2].metric("Projected minutes", f"{r['minutes_proj']:.1f}")
        sa = f"{r['season_avg']:.1f}" if r['season_avg'] is not None else "—"
        m[3].metric("Season avg", sa, help=f"{r['season_games']} games this season · last 10: "
                                           f"{r['last10_avg']:.1f} · last 5: {r['last5_avg']:.1f}")

        left, right = st.columns([1, 1.3])
        with left:
            table = pd.DataFrame({
                'Side': ['Over', 'Under'] + (['Push'] if r['p_push'] > 0 else []),
                'Odds': [fmt_odds(r['over_odds']), fmt_odds(r['under_odds'])] + (['—'] if r['p_push'] > 0 else []),
                'Model': [pct(r['p_over']), pct(r['p_under'])] + ([pct(r['p_push'])] if r['p_push'] > 0 else []),
                'Market (no vig)': [pct(r['fair_over']), pct(r['fair_under'])] + (['—'] if r['p_push'] > 0 else []),
                'EV per $1': [f"{r['ev_over']:+.1%}", f"{r['ev_under']:+.1%}"] + (['—'] if r['p_push'] > 0 else []),
            })
            st.dataframe(table, hide_index=True, width="stretch")
            opp_txt = ', '.join(f"{c} ×{f:.3f}" for c, f in r['opp_factors'].items())
            st.caption(f"Bookmaker margin on this prop: {r['vig']:.1%}. Opponent adjustment: {opp_txt}. "
                       f"Data through {r['last_game']}.")
        with right:
            st.altair_chart(prop_chart(r), width="stretch")


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

with tab_game:
    teams = team_options()
    if odds_fetcher.enabled():
        odds_api_game_panel()
    # Teams and date sit outside the form so the playing/out checklists follow the chosen teams.
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    away = c1.selectbox("Away team", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                        index=None, placeholder="Away")
    home = c2.selectbox("Home team", options=list(teams), format_func=lambda a: f"{teams[a]} ({a})",
                        index=None, placeholder="Home")
    gdate = c3.date_input("Game date", value=date.today(), key="gdate")
    lineups = lineup_inputs(away, home, gdate.isoformat())

    with st.form("game"):
        st.caption("Lines are optional — leave blank to just get the predicted score.")
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
        if not home or not away:
            problems.append("Pick both teams.")
        elif home == away:
            problems.append("Home and away must be different teams.")
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
        pred, evals, stale, calibrated, gd = res
        h, a = pred['home'], pred['away']
        st.subheader(f"{a} @ {h}  ·  {gd}")
        m = st.columns(4)
        m[0].metric(f"{a} (away)", f"{pred['pred_away']:.1f}")
        m[1].metric(f"{h} (home)", f"{pred['pred_home']:.1f}")
        m[2].metric("Total", f"{pred['total']:.1f}", help=f"Projected pace {pred['pace']:.1f} possessions")
        m[3].metric(f"{h} win chance", pct(pred['p_home_win']),
                    help=f"Fair spread: {h} {-pred['margin']:+.1f}")
        adj = adjustment_text(pred)
        if adj:
            st.caption(f"Adjustments to the {h} margin: {adj}")
        if stale:
            st.warning(stale)
        if not calibrated:
            st.caption("Score spreads use default values until the first data update calibrates them.")

        names = {'SPREAD': (h, a), 'TOTAL': ('Over', 'Under'), 'MONEYLINE': (h, a)}
        for e in evals:
            sa, sb = names[e['market']]
            if e['market'] == 'SPREAD':
                title = f"Spread: {h} {e['line']:+g} / {a} {-e['line']:+g}"
                sa, sb = f"{h} {e['line']:+g}", f"{a} {-e['line']:+g}"
            elif e['market'] == 'TOTAL':
                title = f"Total {e['line']:g}"
            else:
                title = "Moneyline"
            with st.container(border=True):
                st.markdown(f"**{title}**")
                if e['pick'] == 'PASS':
                    st.info(f"PASS — neither side clears +{min_ev:.1%} EV")
                else:
                    side, odds, ev = (sa, e['odds_a'], e['ev_a']) if e['pick'] == 'A' else (sb, e['odds_b'], e['ev_b'])
                    st.success(f"**BET {side}** at {fmt_odds(odds)} · EV {ev:+.1%} · "
                               f"stake {e['kelly'] / 4:.1%} of bankroll (¼ Kelly)")
                if e['warning']:
                    st.warning(e['warning'])
                st.dataframe(pd.DataFrame({
                    'Side': [sa, sb],
                    'Odds': [fmt_odds(e['odds_a']), fmt_odds(e['odds_b'])],
                    'Model': [pct(e['p_a']), pct(e['p_b'])],
                    'Market (no vig)': [pct(e['fair_a']), pct(1 - e['fair_a'])],
                    'EV per $1': [f"{e['ev_a']:+.1%}", f"{e['ev_b']:+.1%}"],
                }), hide_index=True, width="stretch")


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

    st.markdown("#### ✍️ Settle or delete manually")
    if 'settle_msg' in st.session_state:
        st.success(st.session_state.pop('settle_msg'))
    st.caption("Enter the result yourself instead of waiting for **Update data**, correct one, or delete an "
               "entry. Automatic settling never overwrites a result you entered.")
    show_all = st.checkbox("Include already-settled entries (to correct or delete them)", key="settle_all")
    kind = st.radio("What", ["Prop", "Game"], horizontal=True, key="settle_kind", label_visibility="collapsed")
    conn = db()
    try:
        if kind == "Prop":
            q = ("SELECT id, game_date, player_name, stat, line, pick, actual, result FROM prop_log "
                 + ("" if show_all else "WHERE result IS NULL ") + "ORDER BY game_date DESC, id DESC")
            rows = {r[0]: r for r in conn.execute(q).fetchall()}
            if not rows:
                st.info("No unsettled props." if not show_all else "No logged props yet.")
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
                                             "Delete entry"], horizontal=True)
                confirm = st.checkbox("Yes, delete it (only needed for Delete entry)")
                if st.form_submit_button("Save", type="primary"):
                    if action == "Delete entry":
                        if not confirm:
                            st.error("Tick “Yes, delete it” to confirm.")
                        elif delete_prop(conn, pid):
                            _done(f"Deleted: {rows[pid][2]} {rows[pid][3]} {rows[pid][4]:g} ({rows[pid][1]})")
                    elif action == "Use actual stat" and actual is None:
                        st.error("Enter the actual stat.")
                    else:
                        res = settle_prop_manual(conn, pid, actual=actual if action == "Use actual stat" else None,
                                                 dnp=action == "Didn't play (void)")
                        _done(f"Saved: {rows[pid][2]} {rows[pid][3]} {rows[pid][4]:g} → "
                                   f"{res or 'unsettled'}")
        else:
            q = ("SELECT game_date, home, away, COUNT(*), SUM(result IS NULL), MAX(away_pts), MAX(home_pts) "
                 "FROM game_log GROUP BY game_date, home, away "
                 + ("" if show_all else "HAVING SUM(result IS NULL) > 0 ") + "ORDER BY game_date DESC")
            games = {f"{r[0]}|{r[1]}|{r[2]}": r for r in conn.execute(q).fetchall()}
            if not games:
                st.info("No unsettled game bets." if not show_all else "No logged game bets yet.")
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
                action = st.radio("Action", ["Save final score", "Delete this game's bets"], horizontal=True)
                confirm = st.checkbox("Yes, delete them (only needed for Delete)")
                if st.form_submit_button("Save", type="primary"):
                    if action == "Delete this game's bets":
                        if not confirm:
                            st.error("Tick “Yes, delete them” to confirm.")
                        else:
                            n = delete_game_bets(conn, d, home, away)
                            _done(f"Deleted {n} logged market(s) for {away} @ {home} ({d}).")
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


with tab_results:
    summary_area = st.container()
    st.divider()
    manual_settle_section()
    conn = db()
    try:
        ps, gs = prop_summary(conn), game_summary(conn)
        props_df, games_df = recent_props(conn), recent_games(conn)
    finally:
        conn.close()

    with summary_area:
        st.caption("Bets settle automatically when you press **Update data** after the games are played, "
                   "or enter results yourself below. A real edge takes a few hundred bets to show up — "
                   "under ~100 the record is mostly luck.")
        for label, s in (("Props", ps), ("Game lines", gs)):
            st.markdown(f"#### {label}")
            rec = s['record']
            c = st.columns(5)
            c[0].metric("Logged / settled", f"{s['logged']} / {s['settled']}")
            if rec:
                c[1].metric("Record (W-L-P)", f"{rec['wins']}-{rec['losses']}-{rec['pushes']}")
                c[2].metric("Profit", f"{rec['profit']:+.2f} units")
                c[3].metric("ROI", f"{rec['roi']:+.1%}")
                c[4].metric("Avg EV claimed", f"{rec['claimed_ev']:+.1%}")
            else:
                c[1].metric("Record (W-L-P)", "—")
            cal = s.get('calibration')
            if cal:
                better = cal['brier_model'] < cal['brier_market']
                st.markdown(f"Model vs market on **{cal['n']}** settled props (lower Brier score = better "
                            f"forecaster): model **{cal['brier_model']:.4f}**, market "
                            f"**{cal['brier_market']:.4f}** → {'✅ model better' if better else '❌ market better'}"
                            + (" *(too few to mean much yet)*" if cal['n'] < 100 else ""))

        st.markdown("#### Logged props")
        st.dataframe(props_df, hide_index=True, width="stretch")
        st.markdown("#### Logged game lines")
        st.dataframe(games_df, hide_index=True, width="stretch")
