"""
data_utils.py
Data loading and optional live-price refresh via yfinance.
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(_DIR, "sample_data.csv")


def load_sample_data() -> pd.DataFrame:
    """Load pre-computed sample data from CSV."""
    df = pd.read_csv(SAMPLE_CSV)
    return df


def refresh_live_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Attempt to refresh Current_Price from yfinance.
    Returns (updated_df, status_message).
    Falls back silently to sample prices on any error.
    """
    try:
        import yfinance as yf
        tickers = df["Ticker"].tolist()
        raw     = yf.download(
            tickers, period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True
        )
        if raw.empty:
            return df, "⚠️ yfinance returned no data — using sample prices."

        close = raw["Close"].iloc[-1]
        updated = 0
        for ticker in tickers:
            if ticker in close.index and not pd.isna(close[ticker]):
                df.loc[df["Ticker"] == ticker, "Current_Price"] = round(float(close[ticker]), 2)
                updated += 1

        if updated == 0:
            return df, "⚠️ Could not parse live prices — using sample prices."
        return df, f"✅ Live prices refreshed for {updated}/{len(tickers)} tickers."

    except ImportError:
        return df, "⚠️ yfinance not installed — using sample prices."
    except Exception as exc:  # noqa: BLE001
        return df, f"⚠️ Live fetch failed ({exc}) — using sample prices."


def get_screener_data(use_live: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Primary entry point.  Returns (df, status_message).
    """
    df = load_sample_data()
    if use_live:
        df, msg = refresh_live_prices(df)
        return df, msg
    return df, "📋 Using sample dataset (demo mode)."
