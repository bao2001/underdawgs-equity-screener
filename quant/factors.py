"""
quant/factors.py — point-in-time factor table.

For every month-end t and ticker, computes factors using ONLY information public at t:
fundamentals are dated by their SEC `filed` date (not the period end), prices by t.

Output: quant/factors_monthly.csv  (ticker, date, factors..., fwd_ret_1m)
Run:    python3 quant/factors.py
"""
import os, sys, bisect, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import build_current_model_outputs as B   # reuse SEC concept chains + Q4-derivation

START, END_FWD = "2016-07-31", None


def _dur_pit(facts, concept):
    """Quarterly duration entries using the FIRST filing per period (original, point-in-time),
    with Q4 derived as annual - (Q1+Q2+Q3)."""
    out, annual = [], []
    for ns in ("us-gaap", "ifrs-full"):
        u = facts.get("facts", {}).get(ns, {}).get(concept, {}).get("units", {}).get("USD", [])
        for e in u:
            if not (e.get("start") and e.get("end") and e.get("val") is not None and e.get("filed")): continue
            s_, en = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]); days = (en - s_).days
            rec = dict(start=s_, end=en, val=float(e["val"]), filed=pd.Timestamp(e["filed"]))
            if 80 <= days <= 100: out.append(rec)
            elif 350 <= days <= 380: annual.append(rec)
    def first(recs):
        d = {}
        for r in sorted(recs, key=lambda x: x["filed"], reverse=True): d[r["end"]] = r   # earliest filed wins
        return d
    q, a = first(out), first(annual)
    for a_end, ar in a.items():
        if a_end in q: continue
        qs = [x for x in q.values() if x["start"] >= ar["start"] - pd.Timedelta(days=5) and x["end"] < a_end - pd.Timedelta(days=30)]
        if len(qs) == 3:
            q[a_end] = dict(start=ar["start"], end=a_end, val=ar["val"] - sum(x["val"] for x in qs),
                            filed=max(ar["filed"], max(x["filed"] for x in qs)))
    return sorted(q.values(), key=lambda x: x["end"])

def _chain_series(facts, concepts, kind):
    """Merged series [(end, filed, val)], preferring earlier concepts in the chain per period end."""
    by_end = {}
    for rank, c in reversed(list(enumerate(concepts))):         # later ranks first, earlier overwrite
        if kind == "dur":
            for e in _dur_pit(facts, c):
                by_end[e["end"]] = (e["end"], e["filed"], e["val"])
        else:
            for ns in ("us-gaap", "ifrs-full"):
                u = facts.get("facts", {}).get(ns, {}).get(c, {}).get("units", {}).get("USD", [])
                best = {}
                for e in u:
                    if e.get("start") or e.get("val") is None or not e.get("end") or not e.get("filed"): continue
                    en = pd.Timestamp(e["end"]); f = pd.Timestamp(e["filed"])
                    if en not in best or f < best[en][1]: best[en] = (en, f, float(e["val"]))   # first filing
                by_end.update(best)
    out = [v for v in by_end.values() if pd.notna(v[1])]
    return sorted(out, key=lambda x: x[0])

def _shares_series(facts):
    out = []
    dei = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    gaap = facts.get("facts", {}).get("us-gaap", {})
    srcs = [dei,
            gaap.get("CommonStockSharesOutstanding", {}).get("units", {}).get("shares", []),
            gaap.get("WeightedAverageNumberOfDilutedSharesOutstanding", {}).get("units", {}).get("shares", [])]
    for src in srcs:
        rows = [(pd.Timestamp(e["end"]), pd.Timestamp(e["filed"]), float(e["val"])) for e in src
                if e.get("val") and e.get("filed") and e.get("end")]
        if rows:
            out = sorted(rows, key=lambda x: x[1]); break
    return out

def _asof(series, t, key_idx=1):
    """Latest-filed value with filed <= t -> (end, val); series sorted by end."""
    best = None
    for end, filed, val in series:
        if filed <= t and (best is None or end > best[0]):
            best = (end, filed, val)
    return best

def _ttm_asof(series, t):
    """(ttm_value, latest_end, prior_ttm) from four consecutive quarters public at t."""
    vis = [(e, f, v) for e, f, v in series if f <= t]
    vis.sort(key=lambda x: x[0])
    def ttm(rows):
        if len(rows) < 4: return None, None
        last4 = rows[-4:]
        span = (last4[-1][0] - last4[0][0]).days
        if not (240 <= span <= 300): return None, None
        return sum(r[2] for r in last4), last4[-1][0]
    cur, end = ttm(vis)
    if cur is None: return None, None, None
    prior, _ = ttm([r for r in vis if r[0] <= end - pd.DateOffset(months=11)])
    return cur, end, prior

def build():
    px = pd.read_csv(os.path.join(HERE, "prices_daily.csv"), index_col=0, parse_dates=True).sort_index()
    tickers = [c for c in px.columns if c != "SPY"]
    cik = pd.read_csv(os.path.join(ROOT, "ticker_cik_mapping.csv")).set_index("ticker")["cik"]
    sector = pd.read_csv(os.path.join(ROOT, "ticker_universe.csv")).drop_duplicates("ticker").set_index("ticker")["sector"]
    extra_path = os.path.join(HERE, "universe_extra.csv")
    if os.path.exists(extra_path):                       # S&P 500 names downloaded by download_universe.py
        ex = pd.read_csv(extra_path).set_index("ticker")
        cik = pd.concat([cik, ex["cik"]]); sector = pd.concat([sector, ex["sector"]])
    me = px.resample("ME").last().index
    me = me[me >= pd.Timestamp(START)]
    rows = []
    for tk in tickers:
        if tk not in cik.index: continue
        facts = B._load_facts(int(cik[tk]))
        if facts is None:                                # fall back to the quant-only cache
            fp = os.path.join(HERE, "sec_facts_cache", f"CIK{int(cik[tk]):010d}.json")
            if not os.path.exists(fp): continue
            with open(fp, encoding="utf-8") as fh: facts = __import__("json").load(fh)
        S = {k: _chain_series(facts, c, "dur") for k, c in dict(
            rev=B._REVENUE_CONCEPTS, ni=B._NET_INCOME_CONCEPTS, oi=B._OP_INCOME_CONCEPTS, gp=B._GROSS_PROFIT_CONCEPTS).items()}
        I = {k: _chain_series(facts, c, "inst") for k, c in dict(
            assets=B._ASSETS_CONCEPTS, equity=B._EQUITY_CONCEPTS, debt=B._DEBT_CONCEPTS).items()}
        sh = _shares_series(facts)
        p = px[tk].dropna()
        if p.empty: continue
        daily_ret = p.pct_change()
        for t in me:
            pt = p[:t]
            if pt.empty or (t - pt.index[-1]).days > 7: continue
            price = pt.iloc[-1]
            if len(pt) < 260: continue
            rev, rend, rev_prior = _ttm_asof(S["rev"], t)
            if rev is None or rend < t - pd.Timedelta(days=550): continue     # fundamentals too stale
            ni, _, _ = _ttm_asof(S["ni"], t); oi, _, _ = _ttm_asof(S["oi"], t); gp, _, _ = _ttm_asof(S["gp"], t)
            eq = _asof(I["equity"], t); da = _asof(I["debt"], t); at = _asof(I["assets"], t)
            s = [x for x in sh if x[1] <= t]
            if not s: continue
            shares = s[-1][2]; mc = price * shares
            if mc <= 0: continue
            p12 = p[:t - pd.DateOffset(months=12)]; p1 = p[:t - pd.DateOffset(months=1)]
            mom = (p1.iloc[-1] / p12.iloc[-1] - 1) if len(p12) and len(p1) else np.nan
            vol = daily_ret[:t].tail(252).std() * np.sqrt(252)
            nxt = p[t + pd.Timedelta(days=1):t + pd.DateOffset(months=1)]
            fwd = (nxt.iloc[-1] / price - 1) if len(nxt) >= 15 else np.nan
            rows.append(dict(
                ticker=tk, date=t, sector=sector.get(tk, "Unknown"), price=price, mktcap=mc,
                earnings_yield=(ni / mc) if ni is not None else np.nan,
                sales_yield=rev / mc,
                book_yield=(eq[2] / mc) if eq else np.nan,
                op_margin=(oi / rev) if (oi is not None and rev) else np.nan,
                gross_margin=(gp / rev) if (gp is not None and rev) else np.nan,
                roe=(ni / eq[2]) if (ni is not None and eq and eq[2] > 0) else np.nan,
                low_leverage=(-da[2] / at[2]) if (da and at and at[2]) else np.nan,
                rev_growth=(rev / rev_prior - 1) if (rev_prior and rev_prior > 0) else np.nan,
                momentum_12_1=mom, low_volatility=-vol, fwd_ret_1m=fwd))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "factors_monthly.csv"), index=False)
    return df

if __name__ == "__main__":
    df = build()
    print(df.shape, df.date.min().date(), df.date.max().date(), df.ticker.nunique(), "tickers")
    print(df.drop(columns=["ticker", "date", "sector"]).describe().T[["count", "mean", "50%"]].round(3).to_string())
    # sanity: our price x shares market cap vs the app's current market cap
    last = df[df.date == df.date.max()].set_index("ticker").mktcap
    cur = pd.read_csv(os.path.join(ROOT, "model_outputs_current.csv")).set_index("ticker").current_market_cap
    last = last[last.index.isin(cur.index)]
    r = (last / cur).dropna()
    print(f"market-cap sanity (ours/yfinance) median={r.median():.2f}, within 20%: {((r>0.8)&(r<1.2)).mean():.0%} of {len(r)}")
