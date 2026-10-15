import os
import numpy as np, pandas as pd
from scipy import stats as sps
from analyze import load, p_over, F
from sklearn.ensemble import HistGradientBoostingRegressor
te = pd.read_parquet(os.path.join(os.environ.get('BT_DIR', '.'), 'te_v3.parquet'))
tr = load(2024)
# --- calibration of P(over) at proxy line: current vs v3 vs normal-shape ---
def cal(d, m, v, shape='nb'):
    L = d['line'].to_numpy()
    if shape == 'nb': po = p_over(d[m].to_numpy(), d[v].to_numpy(), L)
    else: po = sps.norm.sf((np.floor(L) + 0.5 - d[m]) / np.sqrt(d[v]))
    return po
rows=[]
for st, s in te.groupby('stat'):
    hit = (s.actual > s.line).mean()
    r={'stat':st,'actual_over':hit}
    for name,(m,v,sh) in {'cur_nb':('mean','var','nb'),'v3_nb':('mean_v3','var_v3','nb'),'cur_normal':('mean','var','normal')}.items():
        po=cal(s,m,v,sh); r[name]=po.mean(); r[name+'_brier']=((po-(s.actual>s.line))**2).mean()
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False,float_format=F))
# where is P(over) miscalibrated: by projection vs line gap
s = te[te.stat=='PTS'].copy(); s['po']=cal(s,'mean','var'); s['gap']=s['mean']-s['line']
print(s.groupby(pd.cut(s.gap,[-20,-3,-1.5,-0.5,0.5,1.5,3,20]),observed=True).agg(n=('po','size'),pred=('po','mean'),act=('actual',lambda a:(a>s.loc[a.index,'line']).mean())).to_string(float_format=F))
# median vs mean check
print('PTS: share actual < mean', (s.actual<s['mean']).mean())
# --- GBM with squared error, compared on RMSE ---
tr['b2b']=(tr.rest==1).astype(float); te['b2b']=(te.rest==1).astype(float)
for d in (tr,te):
    d['po_flag']=(d.season_type=='Playoffs').astype(float); d['self_ret']=d.self_return.astype(float)
feats=['mean','minutes_proj','season_min_avg','season_avg','last10','last5','miss_min','miss_pts','ret_min_oth','b2b','home','po_flag','self_ret','season_games','opp_rest','rest']
for st in ['PTS','REB','AST','FG3M','PRA']:
    a,b=tr[tr.stat==st],te[te.stat==st]
    m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=200,random_state=0).fit(a[feats],a.actual-a['mean'])
    p=np.clip(b['mean']+m.predict(b[feats]),0.05,None)
    r=lambda x: np.sqrt(((b.actual-x)**2).mean())
    print(f"{st:5} RMSE current {r(b['mean']):.3f} v3 {r(b.mean_v3):.3f} GBM {r(p):.3f} | season avg {r(b.season_avg):.3f}")
# playoffs: starters vs bench bias
po=te[(te.season_type=='Playoffs')&(te.stat=='PTS')]
print(po.groupby('starter').apply(lambda g: pd.Series({'n':len(g),'proj':g['mean'].mean(),'act':g.actual.mean(),'min_proj':g.minutes_proj.mean(),'min_act':g.minutes.mean()})).to_string(float_format=F))
