import numpy as np, pandas as pd
from scipy.optimize import minimize
from analyze import load, p_over, F, P_BET
tr, te = load(2024), load(2025)
lg = lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
sig = lambda z: 1/(1+np.exp(-z))
def prep(d, line_col):
    d=d.copy(); d['L']=d[line_col]; d['po']=p_over(d['mean'].to_numpy(), d['var'].to_numpy(), d['L'].to_numpy()); d['hit']=(d.actual>d.L).astype(float); return d
out=[]
for line_col in ['line','line_l10']:
  a,b=prep(tr,line_col),prep(te,line_col)
  for st in sorted(te.stat.unique()):
    x,y=a[a.stat==st],b[b.stat==st]
    f=lambda w: -np.mean(x.hit*np.log(sig(w[0]+w[1]*lg(x.po)))+(1-x.hit)*np.log(1-sig(w[0]+w[1]*lg(x.po))))
    w=minimize(f,[0,1]).x
    pc=sig(w[0]+w[1]*lg(y.po))
    def betstats(p):
        side=np.where(p>=P_BET,1,np.where(1-p>=P_BET,-1,0)); m=side!=0
        win=np.where(side==1,y.actual>y.L,y.actual<y.L)[m]
        return m.sum(), win.mean(), (side[m]==-1).mean()
        
    n0,w0,u0=betstats(y.po.to_numpy()); n1,w1,u1=betstats(pc.to_numpy())
    out.append({'line':line_col,'stat':st,'a':w[0],'b':w[1],'act':y.hit.mean(),'pred_raw':y.po.mean(),'pred_cal':pc.mean(),
      'brier_raw':((y.po-y.hit)**2).mean(),'brier_cal':((pc-y.hit)**2).mean(),'bets_raw':n0,'win_raw':w0,'under_raw':u0,'bets_cal':n1,'win_cal':w1,'under_cal':u1})
print(pd.DataFrame(out).to_string(index=False,float_format=F))
