"""
quant/filing_risk_test.py — does the 10-K filing-risk score predict returns?

Hypothesis: higher risk score (or a rising score) -> lower subsequent returns.
Point-in-time: a 10-K only counts from its SEC filing date.

A. Event study: for each 10-K, excess return vs SPY over the next 21 / 63 / 126 trading days,
   rank-correlated with the score LEVEL and with the CHANGE vs the prior year's 10-K.
B. Monthly IC: at each month-end, rank all stocks by their latest public score (level / change)
   and correlate with next month's return (same method as quant/backtest.py).
Run: python3 quant/filing_risk_test.py   (needs quant/prices_daily.csv, quant/factors_monthly.csv)
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import sys; sys.path.insert(0, HERE)
from backtest import monthly_stats

sc = pd.read_csv(os.path.join(ROOT, "filing_risk_scores.csv"))[["ticker", "year", "report_risk_score_real"]]
lk = pd.read_csv(os.path.join(ROOT, "filing_lookup_results.csv"))
lk = lk[(lk.form_type == "10-K") & (lk.fetch_status == "found")][["ticker", "year", "filing_date"]].drop_duplicates(["ticker", "year"])
f = sc.merge(lk, on=["ticker", "year"]).dropna(subset=["report_risk_score_real", "filing_date"])
f["filing_date"] = pd.to_datetime(f.filing_date); f = f.sort_values(["ticker", "year"])
f["risk"] = f.report_risk_score_real
prev = f.groupby("ticker").risk.shift(1); gap = f.groupby("ticker").year.diff()
f["risk_chg"] = np.where(gap == 1, f.risk - prev, np.nan)
print(f"{len(f)} scored 10-Ks, {f.ticker.nunique()} tickers, filed {f.filing_date.min().date()} → {f.filing_date.max().date()}")

px = pd.read_csv(os.path.join(HERE, "prices_daily.csv"), index_col=0, parse_dates=True).sort_index()
spy = px["SPY"]

# ── A. event study ────────────────────────────────────────────────────────────
rows = []
for r in f.itertuples():
    if r.ticker not in px.columns: continue
    p = px[r.ticker].dropna(); idx = p.index.searchsorted(r.filing_date + pd.Timedelta(days=1))
    if idx >= len(p) or p.index[0] > r.filing_date - pd.Timedelta(days=5): continue
    d0 = p.index[idx]; s0 = spy.index.searchsorted(d0)
    rec = dict(ticker=r.ticker, year=r.year, date=r.filing_date, risk=r.risk, risk_chg=r.risk_chg)
    for h in (21, 63, 126):
        if idx + h < len(p) and s0 + h < len(spy):
            rec[f"xret_{h}"] = (p.iloc[idx + h] / p.iloc[idx] - 1) - (spy.iloc[s0 + h] / spy.iloc[s0] - 1)
    rows.append(rec)
E = pd.DataFrame(rows)
print(f"\nA. EVENT STUDY — {len(E)} filings with price history (excess return vs SPY after filing)")
print(f"{'':22}{'horizon':>8}{'n':>6}{'rank corr':>11}{'p-value':>9}{'top-tert minus bottom-tert':>29}")
for var in ("risk", "risk_chg"):
    for h in (21, 63, 126):
        d = E.dropna(subset=[var, f"xret_{h}"])
        rho, pv = spearmanr(d[var], d[f"xret_{h}"])
        q = pd.qcut(d[var].rank(method="first"), 3, labels=False)
        spread = d[f"xret_{h}"][q == 2].mean() - d[f"xret_{h}"][q == 0].mean()
        print(f"{('score level' if var=='risk' else 'score change vs prior yr'):22}{h:>6}d{len(d):>6}{rho:>+11.3f}{pv:>9.3f}{spread:>+27.1%}")

# ── B. monthly IC with the latest public score ────────────────────────────────
F = pd.read_csv(os.path.join(HERE, "factors_monthly.csv"), parse_dates=["date"])
F = F[F.ticker.isin(f.ticker.unique())].sort_values("date")
g = f[["ticker", "filing_date", "risk", "risk_chg"]].sort_values("filing_date")
M = pd.merge_asof(F, g, left_on="date", right_on="filing_date", by="ticker", direction="backward",
                  tolerance=pd.Timedelta(days=400)).dropna(subset=["risk"])
for c in ("risk", "risk_chg"): M[c + "_r"] = M.groupby("date")[c].rank(pct=True)
print(f"\nB. MONTHLY RANK IC — {M.date.nunique()} months, {M.ticker.nunique()} tickers (hypothesis: negative)")
for c, lab in (("risk_r", "score level"), ("risk_chg_r", "score change")):
    ic, sp, top, al = monthly_stats(M.dropna(subset=[c]), c)
    t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
    print(f"{lab:14} mean IC={ic.mean():+.4f}  t={t:+.2f}  hit(IC<0)={(ic<0).mean():.0%}  top-minus-bottom quintile={sp.mean()*12:+.1%}/yr  (n={len(ic)} months)")

# ── C. does the score predict future VOLATILITY / drawdown? ───────────────────
# A risk score should arguably forecast risk, not returns. Control for the stock's own past volatility
# (volatile stocks stay volatile), so the score has to add information beyond that.
rows = []
for r in f.itertuples():
    if r.ticker not in px.columns: continue
    p = px[r.ticker].dropna(); idx = p.index.searchsorted(r.filing_date + pd.Timedelta(days=1))
    if idx < 252 or idx + 126 >= len(p): continue
    past = p.pct_change().iloc[idx - 252:idx].std() * np.sqrt(252)
    fut_ret = p.pct_change().iloc[idx + 1:idx + 127]; fut = fut_ret.std() * np.sqrt(252)
    seg = p.iloc[idx:idx + 127]; dd = (seg / seg.cummax() - 1).min()
    rows.append(dict(ticker=r.ticker, year=r.year, risk=r.risk, risk_chg=r.risk_chg, past_vol=past, fut_vol=fut, fut_dd=dd))
V = pd.DataFrame(rows)
def partial_spearman(d, x, y, ctrl):
    rk = d[[x, y, ctrl]].rank()
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    return spearmanr(res(rk[x], rk[ctrl]), res(rk[y], rk[ctrl]))
print(f"\nC. FUTURE RISK — {len(V)} filings (next 126 trading days)")
for var, lab in (("risk", "score level"), ("risk_chg", "score change")):
    d = V.dropna(subset=[var])
    r1 = spearmanr(d[var], d.fut_vol); r2 = partial_spearman(d, var, "fut_vol", "past_vol")
    r3 = spearmanr(d[var], d.fut_dd); r4 = partial_spearman(d, var, "fut_dd", "past_vol")
    print(f"{lab:13} vs future volatility: raw rho={r1[0]:+.3f} (p={r1[1]:.3f}) | controlling for past vol rho={r2[0]:+.3f} (p={r2[1]:.3f})")
    print(f"{'':13} vs max drawdown (more negative=worse): raw rho={r3[0]:+.3f} (p={r3[1]:.3f}) | controlling for past vol rho={r4[0]:+.3f} (p={r4[1]:.3f})")
