"""
ui_helpers.py
Small, dependency-light formatting/styling helpers shared across app.py.
"""

import pandas as pd

MARKET_FRESH_HOURS = 5


def market_freshness(ts, fresh_hours: float = MARKET_FRESH_HOURS) -> str:
    """Return 'fresh' / 'stale' for a timestamp, 'unknown' if unparseable."""
    try:
        age_h = (pd.Timestamp.now() - pd.to_datetime(ts)).total_seconds() / 3600
        return "fresh" if age_h <= fresh_hours else "stale"
    except Exception:
        return "unknown"


def fmt_billions(x) -> str:
    """'$12.3B' for a value already in billions; 'N/A' when missing."""
    return f"${x:.1f}B" if pd.notna(x) else "N/A"


def fmt_signed_pct(x) -> str:
    """'+4.2%' style; 'N/A' when missing."""
    return f"{x:+.1f}%" if pd.notna(x) else "N/A"


def fmt_usd(v) -> str:
    """'$1,234.56'; falls back to str(v) when not numeric."""
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)


def color_signed(val) -> str:
    """CSS for a green/red signed number shown as text ('$', ',', '+', '%' tolerated)."""
    try:
        v = float(str(val).replace("$", "").replace(",", "").replace("+", "").replace("%", ""))
        return "color:#27ae60;font-weight:600" if v >= 0 else "color:#c0392b;font-weight:600"
    except Exception:
        return ""


def pill_badge_html(label: str, text: str, bg: str, fg: str) -> str:
    """Rounded 'Label: value' pill used in the company summary badge row."""
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
        f'font-weight:600;background:{bg};color:{fg};border:1px solid {fg};margin:2px 4px 2px 0">'
        f'<span style="font-weight:400;opacity:0.8">{label}: </span>{text}</span>'
    )
