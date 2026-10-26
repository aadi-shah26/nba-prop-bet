"""
Runs the app's automatic data update (and, on game days, the Kalshi paper bot) in a background
thread, so opening the app never waits on stats.nba.com or Kalshi: the page shows the saved data
straight away and refreshes when the job ends.

State lives at module level (not in a Streamlit cache) so it's shared by every browser tab and
survives the cache clears that follow an update. One update runs at a time.
"""

import threading
import time

_lock = threading.Lock()
_state = {'thread': None, 'started': None, 'finished': None, 'error': None, 'kalshi': None}


def start(retries=2, update=True, kalshi=False):
    """Start a quick update and/or a Kalshi paper-bot run (after the update, so it bets on fresh
    data) unless a job is already running. Returns its start time."""
    with _lock:
        t = _state['thread']
        if t is not None and t.is_alive():
            return _state['started']
        _state.update(started=time.time(), finished=None, error=None, kalshi=None)
        _state['thread'] = threading.Thread(target=_work, args=(retries, update, kalshi), name='nba-update',
                                            daemon=True)
        _state['thread'].start()
        return _state['started']


def _work(retries, update, kalshi):
    from db import connect
    error = result = None
    conn = connect()
    try:
        if update:
            from daily_update import update as run_update
            try:
                run_update(conn, log=lambda *a, **k: None, retries=retries, quick=True)
            except Exception as e:  # offline, stats.nba.com blocking: the saved data stays in use
                error = f"{type(e).__name__}: {e}"
        if kalshi:
            import kalshi_bot
            result = kalshi_bot.attempt(conn)
    finally:
        conn.close()
        with _lock:
            _state.update(finished=time.time(), error=error, kalshi=result)


def status():
    """{'running', 'started', 'finished', 'error', 'kalshi'} (kalshi: what the bot run did, or None)"""
    with _lock:
        t = _state['thread']
        return {'running': t is not None and t.is_alive(), 'started': _state['started'],
                'finished': _state['finished'], 'error': _state['error'], 'kalshi': _state['kalshi']}


def wait(timeout=None):
    """Block until the running update ends (tests)."""
    t = _state['thread']
    if t is not None:
        t.join(timeout)
