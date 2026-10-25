"""
Runs the app's automatic data update in a background thread, so opening the app never waits on
stats.nba.com: the page shows the saved data straight away and refreshes when the update ends.

State lives at module level (not in a Streamlit cache) so it's shared by every browser tab and
survives the cache clears that follow an update. One update runs at a time.
"""

import threading
import time

_lock = threading.Lock()
_state = {'thread': None, 'started': None, 'finished': None, 'error': None}


def start(retries=2):
    """Start a quick update unless one is already running. Returns its start time."""
    with _lock:
        t = _state['thread']
        if t is not None and t.is_alive():
            return _state['started']
        _state.update(started=time.time(), finished=None, error=None)
        _state['thread'] = threading.Thread(target=_work, args=(retries,), name='nba-update', daemon=True)
        _state['thread'].start()
        return _state['started']


def _work(retries):
    from daily_update import update
    from db import connect
    error = None
    conn = connect()
    try:
        update(conn, log=lambda *a, **k: None, retries=retries, quick=True)
    except Exception as e:  # offline, stats.nba.com blocking: the saved data stays in use
        error = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
        with _lock:
            _state.update(finished=time.time(), error=error)


def status():
    """{'running', 'started', 'finished', 'error'}"""
    with _lock:
        t = _state['thread']
        return {'running': t is not None and t.is_alive(), 'started': _state['started'],
                'finished': _state['finished'], 'error': _state['error']}


def wait(timeout=None):
    """Block until the running update ends (tests)."""
    t = _state['thread']
    if t is not None:
        t.join(timeout)
