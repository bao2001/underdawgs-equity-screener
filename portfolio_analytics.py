"""
portfolio_analytics.py
Exposure, concentration and risk analytics for the (paper) portfolio.

Pure pandas/numpy — no Streamlit, no network. The app supplies position data and (on an explicit
user action) a price-history DataFrame. These are descriptive statistics about a portfolio's
makeup; they are not recommendations.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _positions(pos_df: pd.DataFrame) -> pd.DataFrame:
    """Open positions with ticker + market value, or an empty frame."""
    if pos_df is None or pos_df.empty or "Market Value" not in pos_df.columns:
        return pd.DataFrame(columns=["Ticker", "Market Value"])
    p = pos_df[["Ticker", "Market Value"]].copy()
    p["Market Value"] = pd.to_numeric(p["Market Value"], errors="coerce").fillna(0.0)
    return p[p["Market Value"] > 0].reset_index(drop=True)


def sector_exposure(pos_df: pd.DataFrame, sector_map: dict, cash: float = 0.0) -> pd.DataFrame:
    """Market value and weight (% of total portfolio incl. cash) by sector, plus a Cash row."""
    p = _positions(pos_df)
    total = float(p["Market Value"].sum()) + max(float(cash), 0.0)
    if total <= 0:
        return pd.DataFrame(columns=["Sector", "Market Value", "Weight %", "Positions"])
    p["Sector"] = p["Ticker"].map(lambda t: sector_map.get(t) or "Unknown")
    g = p.groupby("Sector").agg(**{"Market Value": ("Market Value", "sum"), "Positions": ("Ticker", "count")})
    if cash > 0:
        g.loc["Cash"] = [float(cash), 0]
    g["Weight %"] = g["Market Value"] / total * 100
    g = g.sort_values("Market Value", ascending=False).reset_index()
    return g[["Sector", "Market Value", "Weight %", "Positions"]]


def concentration_metrics(pos_df: pd.DataFrame) -> dict:
    """Concentration of INVESTED value (cash excluded)."""
    p = _positions(pos_df)
    if p.empty:
        return {"n_positions": 0, "top1_pct": 0.0, "top3_pct": 0.0, "top5_pct": 0.0,
                "hhi": 0.0, "effective_n": 0.0, "largest_ticker": None}
    w = (p["Market Value"] / p["Market Value"].sum()).sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {
        "n_positions": int(len(w)),
        "top1_pct": float(w.iloc[:1].sum() * 100),
        "top3_pct": float(w.iloc[:3].sum() * 100),
        "top5_pct": float(w.iloc[:5].sum() * 100),
        "hhi": hhi,                                   # 1/N when equal-weighted, 1.0 for a single stock
        "effective_n": 1.0 / hhi,                     # "equivalent number of equal-sized positions"
        "largest_ticker": p.loc[p["Market Value"].idxmax(), "Ticker"],
    }


def weight_drift(pos_df: pd.DataFrame, target: dict | None = None) -> pd.DataFrame:
    """
    Compare current weights (share of invested value) with a target (default: equal weight).
    'Value to reach target' is the hypothetical dollar change that would restore the target weights.
    """
    p = _positions(pos_df)
    if p.empty:
        return pd.DataFrame(columns=["Ticker", "Current %", "Target %", "Drift (pts)", "Value to reach target ($)"])
    invested = float(p["Market Value"].sum())
    cur = p.set_index("Ticker")["Market Value"] / invested
    tgt = pd.Series(target, dtype=float) if target else pd.Series(1.0 / len(cur), index=cur.index)
    tgt = (tgt.reindex(cur.index).fillna(0.0))
    s = tgt.sum()
    tgt = tgt / s if s > 0 else tgt
    out = pd.DataFrame({
        "Current %": cur * 100, "Target %": tgt * 100,
        "Drift (pts)": (cur - tgt) * 100,
        "Value to reach target ($)": (tgt - cur) * invested,
    })
    return out.reset_index().rename(columns={"index": "Ticker"}).sort_values("Drift (pts)", ascending=False).reset_index(drop=True)


def max_drawdown(cum: pd.Series) -> float:
    """Worst peak-to-trough decline of a cumulative-value series (negative number)."""
    if cum.empty:
        return 0.0
    return float((cum / cum.cummax() - 1.0).min())


def risk_metrics(prices: pd.DataFrame, weights: dict, benchmark: str = "SPY") -> dict:
    """
    Historical risk of the CURRENT weights applied to past daily returns (constant weights, i.e.
    rebalanced daily — an approximation, and backward-looking).

    prices : DataFrame (date x ticker) of adjusted closes, may include the benchmark column.
    weights: {ticker: weight}; normalised to sum to 1 across tickers with price data.
    """
    tickers = [t for t in weights if t in prices.columns and prices[t].notna().sum() > 30]
    if not tickers:
        return {"status": "no_price_data"}
    rets = prices[tickers + ([benchmark] if benchmark in prices.columns else [])].pct_change().dropna(how="all")
    rets = rets.dropna(subset=tickers)                       # common history only
    if len(rets) < 30:
        return {"status": "insufficient_history", "n_days": int(len(rets))}
    w = pd.Series({t: weights[t] for t in tickers}, dtype=float)
    w = w / w.sum()
    R = rets[tickers]
    port = R.mul(w, axis=1).sum(axis=1)
    cov = R.cov() * TRADING_DAYS
    ann_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    indiv = np.sqrt(np.diag(cov.values))
    weighted_avg_vol = float(w.values @ indiv)
    out = {
        "status": "ok", "n_days": int(len(R)), "tickers_used": tickers,
        "tickers_missing": [t for t in weights if t not in tickers],
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown((1 + port).cumprod()),
        "diversification_ratio": weighted_avg_vol / ann_vol if ann_vol > 0 else np.nan,   # >1 = diversification helps
        "avg_pairwise_corr": np.nan, "beta": np.nan,
    }
    if len(tickers) > 1:
        c = R.corr().values
        out["avg_pairwise_corr"] = float((c.sum() - len(tickers)) / (len(tickers) * (len(tickers) - 1)))
    if benchmark in rets.columns:
        b = rets[benchmark]
        out["beta"] = float(np.cov(port, b)[0, 1] / np.var(b, ddof=1)) if b.var() > 0 else np.nan
        out["benchmark"] = benchmark
    per = pd.DataFrame({"Ticker": tickers, "Weight %": (w * 100).values, "Volatility %": indiv * 100})
    if benchmark in rets.columns:
        b = rets[benchmark]
        per["Beta"] = [float(np.cov(R[t], b)[0, 1] / np.var(b, ddof=1)) for t in tickers]
    out["per_position"] = per.sort_values("Weight %", ascending=False).reset_index(drop=True)
    return out


def exposure_notes(sector_df: pd.DataFrame, conc: dict,
                   max_position_pct: float = 25.0, max_sector_pct: float = 40.0) -> list:
    """Plain-language observations about makeup (descriptive, not advice)."""
    notes = []
    if conc.get("n_positions", 0) == 0:
        return notes
    if conc["top1_pct"] > max_position_pct:
        notes.append(f"{conc['largest_ticker']} is {conc['top1_pct']:.0f}% of invested value "
                     f"(the portfolio is behaving like ~{conc['effective_n']:.1f} equal-sized positions).")
    if conc["n_positions"] < 5:
        notes.append(f"Only {conc['n_positions']} open position(s); results will be dominated by single-stock moves.")
    if not sector_df.empty:
        inv = sector_df[sector_df["Sector"] != "Cash"]
        if not inv.empty:
            top = inv.iloc[0]
            invested_total = inv["Market Value"].sum()
            share = top["Market Value"] / invested_total * 100 if invested_total else 0
            if share > max_sector_pct and len(inv) > 0:
                notes.append(f"{top['Sector']} makes up {share:.0f}% of invested value.")
        if "Unknown" in set(sector_df["Sector"]):
            notes.append("Some positions have no sector data and are grouped as 'Unknown'.")
    return notes
