import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from model_improvement_experiments import *
from scipy.stats import spearmanr
def oof_gap(kind,model,relative):
    X=feats(df,kind); y=df.log_market_cap.values; yr=df.year.values; g=df.ticker.values
    if relative:
        ym=pd.Series(y).groupby(yr).transform("mean").values; X=X-X.groupby(yr).transform("mean"); yt=y-ym
    else: yt=y; ym=np.zeros(len(y))
    p=np.zeros(len(df))
    for tr,te in GroupKFold(5).split(X,yt,g): p[te]=mk(model).fit(X.iloc[tr],yt[tr]).predict(X.iloc[te])
    return (p+ym)-y   # >0 => model says undervalued
d=df[["ticker","year","log_market_cap"]].copy()
nxt=d.copy(); nxt["year"]-=1; nxt=nxt.rename(columns={"log_market_cap":"lmc_next"})
for label,args in [("E0 current RF",(0,"rf",False)),("E1 logscale GBM",(1,"gbm",False)),("E3 relative RF",(2,"rf",True))]:
    d["gap"]=oof_gap(*args)
    m=d.merge(nxt,on=["ticker","year"]); m["fwd"]=m.lmc_next-m.log_market_cap
    m["gap_c"]=m.gap-m.groupby("year").gap.transform("mean"); m["fwd_c"]=m.fwd-m.groupby("year").fwd.transform("mean")
    per=[(y,spearmanr(g.gap_c,g.fwd_c)[0],len(g)) for y,g in m.groupby("year") if len(g)>15]
    ic=np.mean([p[1] for p in per]); se=np.std([p[1] for p in per])/np.sqrt(len(per))
    # quintile spread: top 20% gap vs bottom 20% within year
    m=m[m.groupby("year").year.transform("size")>15].copy()
    m["q"]=m.groupby("year").gap_c.transform(lambda s: pd.qcut(s.rank(method="first"),5,labels=False))
    top=m[m.q==4].groupby("year").fwd_c.mean(); bot=m[m.q==0].groupby("year").fwd_c.mean()
    print(f"{label}: avg yearly rank-IC={ic:+.3f} (±{se:.3f}, {len(per)} yrs, n={len(m)}) | top-vs-bottom quintile next-yr log-cap spread={ (top-bot).mean():+.3f}, positive in {(top-bot>0).sum()}/{len(top)} yrs")
    print("   by year:",{p[0]:round(p[1],2) for p in per})
