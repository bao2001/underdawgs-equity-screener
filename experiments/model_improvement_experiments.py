import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import os
H=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")+"/"
leg=pd.read_csv(H+"valuation_training_dataset.csv"); mod=pd.read_csv(H+"modern_valuation_training_dataset.csv")
df=pd.concat([leg,mod],ignore_index=True,sort=False); df["year"]=df.year.astype(int)
df=df[df.log_market_cap.notna()].copy()
u=pd.read_csv(H+"ticker_universe.csv")[["ticker","sector"]].drop_duplicates("ticker")
df=df.merge(u,on="ticker",how="left"); df["sector"]=df.sector.fillna("Unknown")
print(len(df),df.ticker.nunique(),"tickers; sector coverage:",(df.sector!="Unknown").mean().round(2))
BASE=["log_revenue","net_income","liabilities","equity","cash","debt","operating_income","gross_profit","profit_margin","gross_margin","debt_to_assets","return_on_assets","return_on_equity","cash_to_assets","revenue_growth","net_income_growth"]
DOLLAR=["net_income","liabilities","equity","cash","debt","operating_income","gross_profit"]
slog=lambda x: np.sign(x)*np.log1p(np.abs(x))
def feats(d,kind):
    X=d[BASE].copy()
    if kind>=1:
        for c in DOLLAR: X[c]=slog(d[c])
        X["log_assets"]=d["log_assets"]
        for c in ["profit_margin","gross_margin","return_on_assets","return_on_equity","revenue_growth","net_income_growth"]:
            X[c]=X[c].clip(X[c].quantile(.02),X[c].quantile(.98))
    if kind>=2:
        s=pd.get_dummies(d["sector"],prefix="sec").astype(float); X=pd.concat([X,s],axis=1)
    return X
def mk(name):
    if name=="ridge": return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,20)))
    if name=="rf": return make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(300,min_samples_leaf=3,random_state=0,n_jobs=-1))
    if name=="gbm": return make_pipeline(SimpleImputer(strategy="median"),GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.04,subsample=0.8,random_state=0))
def metrics(y,p,yr):
    r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    # relative: remove each year's mean from actual and pred (ranking within the same year)
    d=pd.DataFrame({"y":y,"p":p,"yr":yr}); d["yd"]=d.y-d.groupby("yr").y.transform("mean"); d["pd"]=d.p-d.groupby("yr").p.transform("mean")
    r2r=1-((d.yd-d.pd)**2).sum()/((d.yd-d.yd.mean())**2).sum()
    sp=np.nanmean([spearmanr(g.y,g.p)[0] for _,g in d.groupby("yr") if len(g)>5])
    return dict(R2=r2,relR2=r2r,rank=sp,MAE=np.abs(y-p).mean())
def run(kind,model,relative=False):
    X=feats(df,kind); y=df.log_market_cap.values; yr=df.year.values; g=df.ticker.values
    if relative:
        # cross-sectional: demean target and each feature within year
        ym=pd.Series(y).groupby(yr).transform("mean").values; X=X-X.groupby(yr).transform("mean"); yt=y-ym
    else: yt=y
    out={}
    # company-level 5-fold
    p=np.zeros(len(df))
    for tr,te in GroupKFold(5).split(X,yt,g): p[te]=mk(model).fit(X.iloc[tr],yt[tr]).predict(X.iloc[te])
    out["company"]=metrics(yt,p,yr) if not relative else metrics(y,p+ym,yr)
    for nm,cut in [("time<=2016",2016),("time<=2020",2020)]:
        tr=yr<=cut; te=~tr; m=mk(model).fit(X[tr],yt[tr]); pp=m.predict(X[te])
        out[nm]=metrics(y[te],(pp+ym[te]) if relative else pp,yr[te])
    return out
if __name__=="__main__":
    rows=[]
    for label,kind,rel in [("E0 current",0,False),("E1 +logscale",1,False),("E2 +sector",2,False),("E3 +relative(yr-demeaned)",2,True)]:
        for model in ["ridge","rf","gbm"]:
            r=run(kind,model,rel)
            for sp,m in r.items(): rows.append(dict(exp=label,model=model,split=sp,**{k:round(v,3) for k,v in m.items()}))
    R=pd.DataFrame(rows); os.makedirs("experiments_out",exist_ok=True); R.to_csv("experiments_out/results.csv",index=False)
    pd.set_option("display.width",200)
    for sp in ["company","time<=2020","time<=2016"]:
        print("\n==",sp); print(R[R.split==sp].drop(columns="split").to_string(index=False))
