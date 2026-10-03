"""
quant/backtest.py — does each factor rank next-month returns?

Per month: rank stocks on a factor, compute the Spearman rank correlation (IC) with the next
month's return, and the return spread between the top and bottom quintile. Factor weights for the
composite are fixed a priori (equal weight per group) — nothing is fitted to the results.

Run: python3 quant/backtest.py     (needs quant/factors_monthly.csv from quant/factors.py)
CAVEATS: universe = today's ~120 tickers (survivorship bias), no transaction costs, 1-month horizon.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    "value":    ["earnings_yield", "sales_yield", "book_yield"],
    "quality":  ["op_margin", "gross_margin", "roe", "low_leverage"],
    "growth":   ["rev_growth"],
    "momentum": ["momentum_12_1"],
    "low_risk": ["low_volatility"],
}
FACTORS = [f for g in GROUPS.values() for f in g]

def add_scores(df):
    df = df.copy()
    for f in FACTORS:
        df[f + "_r"] = df.groupby("date")[f].rank(pct=True)
    for g, fs in GROUPS.items():
        df["grp_" + g] = df[[f + "_r" for f in fs]].mean(axis=1, skipna=True)
    for g in GROUPS: df["grp_" + g + "_r"] = df.groupby("date")["grp_" + g].rank(pct=True)
    core = ["value", "quality", "growth", "momentum"]
    df["composite4"] = df[["grp_" + g + "_r" for g in core]].mean(axis=1, skipna=False)
    df["composite5"] = df[["grp_" + g + "_r" for g in GROUPS]].mean(axis=1, skipna=False)
    return df

def monthly_stats(df, col):
    ics, spreads, tops, alls = [], [], [], []
    for d, g in df.groupby("date"):
        g = g.dropna(subset=[col, "fwd_ret_1m"])
        if len(g) < 30: continue
        ics.append((d, spearmanr(g[col], g.fwd_ret_1m)[0]))
        q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
        spreads.append((d, g.fwd_ret_1m[q == 4].mean() - g.fwd_ret_1m[q == 0].mean()))
        tops.append((d, g.fwd_ret_1m[q == 4].mean())); alls.append((d, g.fwd_ret_1m.mean()))
    s = lambda x: pd.Series(dict(x))
    return s(ics), s(spreads), s(tops), s(alls)

def summarize(df, col, label=None):
    ic, sp, top, al = monthly_stats(df, col)
    n = len(ic)
    h = n // 2
    t = lambda x: x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
    return dict(factor=label or col, months=n, mean_IC=ic.mean(), IC_t=t(ic), IC_hit=(ic > 0).mean(),
                spread_ann=sp.mean() * 12, spread_t=t(sp),
                topQ_excess_ann=(top - al).mean() * 12,
                IC_first_half=ic.iloc[:h].mean(), IC_second_half=ic.iloc[h:].mean())

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(HERE, "factors_monthly.csv"), parse_dates=["date"])
    df = add_scores(df)
    rows = [summarize(df, f + "_r", f) for f in FACTORS]
    rows += [summarize(df, "grp_" + g + "_r", "GROUP " + g) for g in GROUPS]
    rows += [summarize(df, "composite4", "COMPOSITE (value+quality+growth+momentum)"),
             summarize(df, "composite5", "COMPOSITE (+low risk)")]
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "backtest_results.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(f"{df.date.nunique()} months, {df.ticker.nunique()} tickers, {len(df)} rows\n")
    print(R.to_string(index=False))
    print("\nIC_t: |t|>2 is nominal significance; with ~12 factors tested, want |t|>3 to be convincing.")
