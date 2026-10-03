"""
discovery_utils.py
Helpers for stock discovery: valuation/fundamental ratios, range filters that are inert at their
defaults, saved screener configurations, and same-sector peer tables.

Pure pandas — no Streamlit. Ratios are computed from SEC fundamentals (TTM income statement,
latest balance sheet) and a market capitalisation in dollars.
"""
import json
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_SCREENS_PATH = os.path.join(_HERE, "saved_screens.json")

# column -> (label, format) for display; "pct" values are already in percent
RATIO_COLUMNS = {
    "PE_Ratio":           ("P/E", "x"),
    "PS_Ratio":           ("P/S", "x"),
    "PB_Ratio":           ("P/B", "x"),
    "Revenue_Growth_Pct": ("Rev. Growth", "pct"),
    "Op_Margin_Pct":      ("Op. Margin", "pct"),
    "ROE_Pct":            ("ROE", "pct"),
    "Debt_Equity":        ("Debt/Equity", "x"),
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_fundamental_ratios(df: pd.DataFrame, mktcap_col: str = "current_market_cap") -> pd.DataFrame:
    """
    Add valuation and fundamental ratios. Ratios with a non-positive denominator are NaN
    (e.g. P/E for a loss-making company, ROE with negative equity) rather than misleading values.
    Expects raw columns revenue, net_income, operating_income, equity, debt, revenue_growth (fraction).
    """
    out = df.copy()
    mc = _num(out, mktcap_col)
    rev, ni, oi = _num(out, "revenue"), _num(out, "net_income"), _num(out, "operating_income")
    eq, debt = _num(out, "equity"), _num(out, "debt")
    pos = lambda s: s.where(s > 0)
    out["PE_Ratio"] = mc.where(mc > 0) / pos(ni)
    out["PS_Ratio"] = mc.where(mc > 0) / pos(rev)
    out["PB_Ratio"] = mc.where(mc > 0) / pos(eq)
    out["Revenue_Growth_Pct"] = _num(out, "revenue_growth") * 100
    out["Op_Margin_Pct"] = oi / pos(rev) * 100
    out["ROE_Pct"] = ni / pos(eq) * 100
    out["Debt_Equity"] = debt / pos(eq)
    out["Revenue_B"] = rev / 1e9
    return out


def range_filter(df: pd.DataFrame, col: str, lo=None, hi=None) -> pd.DataFrame:
    """
    Keep rows with lo <= col <= hi. A bound of None is ignored. When neither bound is set the
    filter is inert and rows with missing values are kept; when a bound is set, missing values
    are excluded (they cannot be shown to satisfy it).
    """
    if col not in df.columns or (lo is None and hi is None):
        return df
    v = pd.to_numeric(df[col], errors="coerce")
    m = v.notna()
    if lo is not None:
        m &= v >= lo
    if hi is not None:
        m &= v <= hi
    return df[m]


def fmt_ratio(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.1f}%" if kind == "pct" else f"{v:.1f}x"


# ── Saved screens ─────────────────────────────────────────────────────────────

def load_saved_screens(path: str = SAVED_SCREENS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_screen(name: str, settings: dict, path: str = SAVED_SCREENS_PATH) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Screen name is required.")
    screens = load_saved_screens(path)
    screens[name] = settings
    with open(path, "w", encoding="utf-8") as f:
        json.dump(screens, f, indent=2, default=str)
    return screens


def delete_screen(name: str, path: str = SAVED_SCREENS_PATH) -> dict:
    screens = load_saved_screens(path)
    screens.pop(name, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(screens, f, indent=2, default=str)
    return screens


def sanitize_multiselect(value, options: list):
    """Keep only saved values that are still valid options (data can change between sessions)."""
    if not isinstance(value, (list, tuple)):
        return None
    return [v for v in value if v in options]


# ── Peers ─────────────────────────────────────────────────────────────────────

def peer_table(df: pd.DataFrame, ticker: str, metrics: list, sector_col: str = "Sector",
               ticker_col: str = "Ticker", max_peers: int = 12, size_col: str | None = None) -> pd.DataFrame:
    """
    Same-sector companies (the selected company first, then the largest peers by size_col), with
    each metric's percentile rank within the full sector group.
    """
    if df.empty or ticker not in set(df[ticker_col]):
        return pd.DataFrame()
    sector = df.loc[df[ticker_col] == ticker, sector_col].iloc[0]
    if pd.isna(sector) or sector in ("", "Unknown"):
        return pd.DataFrame()
    grp = df[df[sector_col] == sector].copy()
    for m in metrics:
        if m in grp.columns:
            grp[m + "_pctile"] = pd.to_numeric(grp[m], errors="coerce").rank(pct=True) * 100
    others = grp[grp[ticker_col] != ticker]
    if size_col and size_col in others.columns:
        others = others.sort_values(size_col, ascending=False)
    return pd.concat([grp[grp[ticker_col] == ticker], others.head(max_peers)]).reset_index(drop=True)


def sector_medians(df: pd.DataFrame, sector: str, metrics: list, sector_col: str = "Sector") -> dict:
    grp = df[df[sector_col] == sector]
    return {m: float(pd.to_numeric(grp[m], errors="coerce").median()) for m in metrics if m in grp.columns}
