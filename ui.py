"""
Presentation helpers for app.py: page styling, team logos and the small HTML pieces (verdicts,
matchup header, tracked-bets tables) that Streamlit's built-in widgets can't draw.

Logos are downloaded once from the NBA's CDN into .logo_cache/ and embedded in the page. Until
one is available (offline, first run without network) the team shows as its abbreviation on a
circle in the team's colour. NBA_LOGOS=0 skips downloading.
"""

import base64
import html
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import streamlit as st

LOGO_URL = "https://cdn.nba.com/logos/nba/{}/global/L/logo.svg"
LOGO_DIR = Path(__file__).resolve().parent / '.logo_cache'
LOGO_RETRY_SECONDS = 600     # after a failed download, don't try again (and stall the page) for this long
_logo_uris = {}
_logo_retry_at = 0.0

TEAM_COLORS = {
    'ATL': '#C8102E', 'BOS': '#007A33', 'BKN': '#111111', 'CHA': '#1D1160', 'CHI': '#CE1141',
    'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240', 'DET': '#C8102E', 'GSW': '#1D428A',
    'HOU': '#CE1141', 'IND': '#002D62', 'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9',
    'MIA': '#98002E', 'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160', 'POR': '#C8102E',
    'SAC': '#5A2D81', 'SAS': '#3A3F44', 'TOR': '#CE1141', 'UTA': '#002B5C', 'WAS': '#002B5C',
}

CSS = """
:root {
  --ink: #0F172A; --muted: #64748B; --line: #E2E8F0; --card: #FFFFFF;
  --accent: #1D4ED8; --accent-soft: #EFF4FF;
  --pos: #15803D; --pos-soft: #E8F6EC; --neg: #B91C1C; --neg-soft: #FCECEC;
  --warn: #B45309; --warn-soft: #FEF5E7; --flat: #475569; --flat-soft: #F1F5F9;
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.num { font-variant-numeric: tabular-nums; }

/* app header */
.app-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
  flex-wrap: wrap; margin-bottom: .25rem; }
.app-head h1 { font-size: 1.55rem; font-weight: 700; margin: 0; padding: 0; color: var(--ink); }
.app-head .sub { color: var(--muted); font-size: .9rem; }

/* metric cards */
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: .75rem 1rem; }
div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .8rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .04em; }
div[data-testid="stMetricValue"] { font-weight: 650; }

/* pills */
.pill { display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .78rem;
  font-weight: 600; white-space: nowrap; }
.pill.pos { background: var(--pos-soft); color: var(--pos); }
.pill.neg { background: var(--neg-soft); color: var(--neg); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }
.pill.flat { background: var(--flat-soft); color: var(--flat); }
.pill.accent { background: var(--accent-soft); color: var(--accent); }

/* team logos */
.logo { position: relative; display: inline-flex; align-items: center; justify-content: center;
  width: var(--s); height: var(--s); border-radius: 50%; background: var(--team); flex: none;
  vertical-align: middle; overflow: hidden; }
.logo .mono { color: #fff; font-weight: 700; font-size: calc(var(--s) * .3); letter-spacing: .02em; }
.logo img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
  background: #fff; padding: 8%; box-sizing: border-box; }
.team { display: inline-flex; align-items: center; gap: .5rem; }

/* verdict */
.verdict { border: 1px solid var(--line); border-left: 4px solid var(--flat); background: var(--card);
  border-radius: 10px; padding: .85rem 1.1rem; margin: .35rem 0 .75rem; }
.verdict.bet { border-left-color: var(--pos); background: linear-gradient(90deg, var(--pos-soft), var(--card) 55%); }
.verdict .kicker { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); }
.verdict.bet .kicker { color: var(--pos); }
.verdict .main { font-size: 1.35rem; font-weight: 700; color: var(--ink); margin: .1rem 0; }
.verdict .main .odds { color: var(--muted); font-weight: 600; font-size: 1.05rem; margin-left: .35rem; }
.verdict .meta { color: var(--muted); font-size: .88rem; }
.verdict .meta b { color: var(--ink); font-weight: 600; }

/* section title with logo */
.title-row { display: flex; align-items: center; gap: .7rem; margin: .4rem 0 .2rem; }
.title-row .t { font-size: 1.3rem; font-weight: 700; color: var(--ink); }
.title-row .s { color: var(--muted); font-size: .95rem; }

/* matchup */
.matchup { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.3rem; }
.matchup .row { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 1rem; }
.matchup .side { display: flex; align-items: center; gap: .85rem; }
.matchup .side.home { justify-content: flex-end; text-align: right; }
.matchup .name { font-weight: 650; color: var(--ink); line-height: 1.2; }
.matchup .tag { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }
.matchup .score { font-size: 2rem; font-weight: 700; color: var(--ink); }
.matchup .mid { text-align: center; color: var(--muted); font-size: .85rem; }
.matchup .mid b { display: block; color: var(--ink); font-size: 1rem; }
.probbar { display: flex; height: 8px; border-radius: 999px; overflow: hidden; margin: 1rem 0 .35rem; }
.probbar span { display: block; height: 100%; }
.problbl { display: flex; justify-content: space-between; font-size: .82rem; color: var(--muted); }
.problbl b { color: var(--ink); }

/* tracked tables */
.tbl-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--card); }
table.trk { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.trk th { text-align: left; font-weight: 600; color: var(--muted); font-size: .74rem;
  text-transform: uppercase; letter-spacing: .05em; padding: .6rem .75rem; border-bottom: 1px solid var(--line);
  background: #FAFBFC; white-space: nowrap; }
table.trk td { padding: .55rem .75rem; border-bottom: 1px solid var(--line); color: var(--ink);
  vertical-align: middle; white-space: nowrap; }
table.trk tr:last-child td { border-bottom: none; }
table.trk.compact { font-size: .84rem; }
table.trk.compact th, table.trk.compact td { padding: .45rem .5rem; white-space: normal; }
table.trk.compact th { letter-spacing: .02em; font-size: .68rem; }
table.trk.compact td:first-child { white-space: nowrap; }
table.trk td.r, table.trk th.r { text-align: right; }
table.trk .sub { color: var(--muted); font-size: .78rem; }
.pos-txt { color: var(--pos); font-weight: 600; }
.neg-txt { color: var(--neg); font-weight: 600; }
.dim { color: #94A3B8; }

/* sidebar */
section[data-testid="stSidebar"] .brand { font-size: 1.1rem; font-weight: 700; letter-spacing: -.01em; }
section[data-testid="stSidebar"] .brand small { display: block; font-weight: 500; font-size: .78rem; opacity: .65; }
"""


def inject_css():
    st.html(f"<style>{CSS}</style>")


def esc(x):
    return html.escape(str(x))


@lru_cache(maxsize=1)
def team_ids():
    from nba_api.stats.static import teams
    return {t['abbreviation']: t['id'] for t in teams.get_teams()}


def logo_uri(abbr):
    """data: URI of the team's logo, downloading it the first time. None if unavailable."""
    global _logo_retry_at
    if abbr in _logo_uris:
        return _logo_uris[abbr]
    path = LOGO_DIR / f"{abbr}.svg"
    if not path.exists():
        tid = team_ids().get(abbr)
        if tid is None or os.getenv('NBA_LOGOS', '1') == '0' or time.time() < _logo_retry_at:
            return None
        try:
            import requests
            r = requests.get(LOGO_URL.format(tid), timeout=4)
            r.raise_for_status()
            if b'<svg' not in r.content[:2000]:
                raise ValueError("not an SVG")
            LOGO_DIR.mkdir(exist_ok=True)
            path.write_bytes(r.content)
        except Exception:
            _logo_retry_at = time.time() + LOGO_RETRY_SECONDS
            return None
    _logo_uris[abbr] = "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()
    return _logo_uris[abbr]


def logo(abbr, size=28):
    """Team logo, or the abbreviation on a circle in the team's colour if the logo isn't available."""
    if not abbr:
        return ''
    uri = logo_uri(abbr)
    img = f'<img src="{uri}" alt="">' if uri else ''
    return (f'<span class="logo" style="--team:{TEAM_COLORS.get(abbr, "#334155")};--s:{size}px" '
            f'title="{esc(abbr)}"><span class="mono">{esc(abbr)}</span>{img}</span>')


def team(abbr, size=22, name=None):
    return f'<span class="team">{logo(abbr, size)}<span>{esc(name or abbr)}</span></span>'


def pill(text, tone='flat'):
    return f'<span class="pill {tone}">{esc(text)}</span>'


def show(markup):
    st.markdown(markup, unsafe_allow_html=True)


def app_header(subtitle):
    show(f'<div class="app-head"><h1>NBA Betting Model</h1><span class="sub">{subtitle}</span></div>')


def title_row(title, sub='', abbr=None):
    show(f'<div class="title-row">{logo(abbr, 40) if abbr else ""}<div><div class="t">{esc(title)}</div>'
         f'<div class="s">{esc(sub)}</div></div></div>')


def verdict(bet, main, odds=None, meta='', abbr=None):
    """bet: True for a recommended bet, False for a pass. meta may contain <b> markup.
    abbr: team the bet is on, to show its logo."""
    odds_html = f'<span class="odds">{esc(odds)}</span>' if odds else ''
    main_html = f'<span class="team">{logo(abbr, 26)}<span>{esc(main)}</span></span>' if abbr else esc(main)
    show(f'<div class="verdict {"bet" if bet else "pass"}"><div class="kicker">{"Bet" if bet else "Pass"}</div>'
         f'<div class="main">{main_html}{odds_html}</div><div class="meta">{meta}</div></div>')


def odds_table(rows, pick=None):
    """Side-by-side model vs market table. rows: [(side, odds, model, market, ev)];
    pick: index of the recommended row, highlighted."""
    head = [('Side', ''), ('Odds', 'r'), ('Model', 'r'), ('Market', 'r'), ('EV', 'r')]
    body = []
    for i, (side, odds, model, market, ev) in enumerate(rows):
        ev_html = (f'<span class="{"pos-txt" if ev > 0 else "neg-txt"}">{ev:+.1%}</span>'
                   if isinstance(ev, float) else esc(ev))
        body.append([f'<b>{esc(side)}</b>' if i == pick else esc(side), esc(odds), esc(model), esc(market), ev_html])
    _table(head, body, 'compact')


def matchup(away, home, names, pred, game_date):
    """Predicted score card with each team's logo and a win-probability bar in team colours."""
    p_home = pred['p_home_win']
    ca, ch = TEAM_COLORS.get(away, '#64748B'), TEAM_COLORS.get(home, '#1D4ED8')
    if ca == ch:
        ca = '#94A3B8'
    show(f"""<div class="matchup"><div class="row">
<div class="side">{logo(away, 56)}<div><div class="tag">Away</div><div class="name">{esc(names.get(away, away))}</div></div>
<div class="score">{pred['pred_away']:.1f}</div></div>
<div class="mid">{esc(game_date)}<b class="num">Total {pred['total']:.1f}</b>
Fair spread {esc(home)} {-pred['margin']:+.1f}</div>
<div class="side home"><div class="score">{pred['pred_home']:.1f}</div>
<div><div class="tag">Home</div><div class="name">{esc(names.get(home, home))}</div></div>{logo(home, 56)}</div>
</div>
<div class="probbar"><span style="width:{(1 - p_home) * 100:.1f}%;background:{ca}"></span>
<span style="width:{p_home * 100:.1f}%;background:{ch}"></span></div>
<div class="problbl"><span>{esc(away)} win chance <b class="num">{1 - p_home:.0%}</b></span>
<span>{esc(home)} win chance <b class="num">{p_home:.0%}</b></span></div></div>""")


# ---------------------------------------------------------------------------
# Tracked-bets tables
# ---------------------------------------------------------------------------

RESULT_TONES = {'Hit': 'pos', 'Miss': 'neg', 'Push': 'flat', 'Void (DNP)': 'flat',
                'Pending': 'accent', 'Awaiting result': 'warn'}


def _num(x, fmt, dim='–'):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f'<span class="dim">{dim}</span>'
    return format(x, fmt)


def _profit(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '<span class="dim">–</span>'
    return f'<span class="{"pos-txt" if x > 0 else "neg-txt" if x < 0 else ""}">{x:+.2f}</span>'


def _bet_cell(r):
    lean = ' <span class="pill flat">Lean</span>' if r['Type'] == 'Lean' else ''
    return f"{esc(r['Bet'])}{lean}"


def _table(head, rows, cls=''):
    th = ''.join(f'<th class="{c}">{esc(h)}</th>' for h, c in head)
    body = ''.join('<tr>' + ''.join(f'<td class="{c}">{cell}</td>' for cell, (_, c) in zip(row, head)) + '</tr>'
                   for row in rows)
    show(f'<div class="tbl-wrap"><table class="{f"trk {cls}".strip()}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def props_table(df):
    head = [('Date', ''), ('Player', ''), ('Bet', ''), ('Opp', ''), ('Proj', 'r'), ('Model', 'r'),
            ('EV', 'r'), ('Actual', 'r'), ('Result', ''), ('Units', 'r')]
    _table(head, [[
        esc(r['Date']), esc(r['Player']), _bet_cell(r), team(r['vs'], 20) if r['vs'] else '<span class="dim">–</span>',
        _num(r['Projection'], '.1f'), _num(r['Model %'], '.0%'), _num(r['EV'], '+.1%'), _num(r['Actual'], 'g'),
        pill(r['Result'], RESULT_TONES.get(r['Result'], 'flat')), _profit(r['Profit (u)']),
    ] for _, r in df.iterrows()])


def games_table(df):
    head = [('Date', ''), ('Game', ''), ('Bet', ''), ('Predicted', 'r'), ('Model', 'r'), ('EV', 'r'),
            ('Final', 'r'), ('Result', ''), ('Units', 'r')]
    _table(head, [[
        esc(r['Date']),
        f'<span class="team">{logo(r["Away"], 20)}<span>{esc(r["Away"])}</span>'
        f'<span class="dim">@</span>{logo(r["Home"], 20)}<span>{esc(r["Home"])}</span></span>',
        _bet_cell(r), esc(r['Predicted']), _num(r['Model %'], '.0%'), _num(r['EV'], '+.1%'),
        esc(r['Final']) if r['Final'] else '<span class="dim">–</span>',
        pill(r['Result'], RESULT_TONES.get(r['Result'], 'flat')), _profit(r['Profit (u)']),
    ] for _, r in df.iterrows()])
