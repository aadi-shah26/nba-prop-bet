import numpy as np, pandas as pd
from analyze import load
tr, te = [load(s).query("stat=='PTS'").copy() for s in (2024, 2025)]
for d in (tr, te):
    d['b2b']=(d.rest==1).astype(float); d['po']=(d.season_type=='Playoffs').astype(float); d['sr']=d.self_return.astype(float)
    d['miss_share']=d.miss_min.fillna(0)*d.minutes_proj/240; d['ret_share']=d.ret_min_oth.fillna(0)*d.minutes_proj/240
    d['gap']=d.season_min_avg-d.minutes_proj; d['st']=d.starter.fillna(False).astype(float)
    # starter in the player's recent games -> "lineup change" flag
    d['pom']=d.po*d.minutes_proj
base=['minutes_proj','gap','miss_share','ret_share','b2b','home','po','sr','pom']
def fit(cols):
    X=np.column_stack([np.ones(len(tr)),tr[cols]]); b=np.linalg.lstsq(X,tr.minutes,rcond=None)[0]
    p=np.column_stack([np.ones(len(te)),te[cols]])@b; return np.abs(te.minutes-p).mean(), b
print('current minutes MAE', np.abs(te.minutes-te.minutes_proj).mean().round(3))
print('pregame model     ', round(fit(base)[0],3))
m,b=fit(base+['st']); print('+ confirmed starter', round(m,3), 'starter coef', round(b[-1],2))
# minutes MAE share from early exits (injury/foul-out/blowout): minutes < 50% of projection
e=te.minutes<0.5*te.minutes_proj; print('games with minutes < half of projection:', f"{e.mean():.1%}", 'share of total abs minutes error:', f"{np.abs(te.minutes-te.minutes_proj)[e].sum()/np.abs(te.minutes-te.minutes_proj).sum():.1%}")
