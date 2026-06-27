"""
market_cap_utils.py
-------------------
Steps 1-3 of the market-cap model upgrade:

  Step 1  Load ticker_mapping.csv
  Step 2  Fetch or load market_cap_data.csv
  Step 3  Merge with XBRL features → valuation_training_dataset.csv

Run once as a standalone script to pre-fetch market cap data (takes ~2-3 min):

    python3 market_cap_utils.py

The app then loads the saved CSV files rather than hitting yfinance on every visit.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

TICKER_MAP_PATH    = os.path.join(_HERE, "ticker_mapping.csv")
MARKET_CAP_PATH    = os.path.join(_HERE, "market_cap_data.csv")
TRAINING_DATA_PATH = os.path.join(_HERE, "valuation_training_dataset.csv")

YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016]

# Minimum usable rows to proceed with model training
MIN_TRAINING_ROWS = 30

# ── Step 1: Ticker mapping ─────────────────────────────────────────────────────

def load_ticker_mapping() -> tuple:
    """
    Load ticker_mapping.csv.
    Returns (df, status_msg). df is None on failure.
    """
    if not os.path.exists(TICKER_MAP_PATH):
        return None, (
            f"ticker_mapping.csv not found at {TICKER_MAP_PATH}. "
            "A seed file is included with the project. Place it next to app.py."
        )
    try:
        df = pd.read_csv(TICKER_MAP_PATH, dtype={"company_id": str})
        required = {"company_id", "ticker"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            return None, f"ticker_mapping.csv is missing columns: {missing_cols}"
        df = df.dropna(subset=["company_id", "ticker"])
        df["company_id"] = df["company_id"].astype(str).str.strip()
        df["ticker"]     = df["ticker"].astype(str).str.strip().str.upper()
        return df, f"Loaded {len(df)} ticker mappings."
    except Exception as exc:
        return None, f"Error reading ticker_mapping.csv: {exc}"


# ── Share-count normalisation ──────────────────────────────────────────────────

def _normalize_shares(series: pd.Series) -> pd.Series:
    """
    Some XBRL filers switch from reporting shares as individual units to
    reporting in thousands mid-series.  Detect outlier years where the value
    is < 1/50 of the company's multi-year median and scale up by 1 000.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return series
    median = valid.median()
    if pd.isna(median) or median <= 0:
        return series
    return series.apply(
        lambda x: x * 1_000
        if pd.notna(x) and x > 0 and x < median / 50
        else x
    )


# ── Step 2: Fetch market cap via yfinance ──────────────────────────────────────

def _get_split_factor(ticker_obj, year_int: int) -> float:
    """
    Cumulative product of all stock splits that occurred AFTER year-end.
    yfinance returns split-adjusted historical prices; multiplying the
    adjusted price by this factor recovers the approximate historical price.
    """
    try:
        splits = ticker_obj.splits
        if splits.empty:
            return 1.0
        tz      = splits.index.tz
        cutoff  = pd.Timestamp(f"{year_int}-12-31", tz=tz)
        future  = splits[splits.index > cutoff]
        factor  = float(future.prod()) if len(future) > 0 else 1.0
        return factor if factor > 0 else 1.0
    except Exception:
        return 1.0


def _fetch_for_one_company(ticker_str: str, cid: str, shares_by_year: dict) -> list:
    """
    Fetch December year-end prices for YEARS, apply split correction,
    multiply by XBRL shares to estimate market cap.

    shares_by_year: {year_int: float or None}
    Returns list of row dicts.
    """
    import yfinance as yf

    def _row(yr, mc, flag):
        return {
            "company_id":        cid,
            "ticker":            ticker_str,
            "year":              yr,
            "market_cap":        mc,
            "data_quality_flag": flag,
        }

    rows = []
    try:
        t    = yf.Ticker(ticker_str)
        hist = yf.download(
            ticker_str,
            start="2009-11-01", end="2017-01-31",
            interval="1mo", auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if hist.empty:
            return [_row(yr, None, "missing_price") for yr in YEARS]

        hist.index = pd.to_datetime(hist.index)
        dec_price  = {
            yr: float(grp["Close"].iloc[-1])
            for yr, grp in hist.groupby(hist.index.year)
            if any(hist[hist.index.year == yr].index.month == 12)
            for grp in [hist[(hist.index.year == yr) & (hist.index.month == 12)]]
            if not grp.empty
        }

        for yr in YEARS:
            shares    = shares_by_year.get(yr)
            price_adj = dec_price.get(yr)

            if price_adj is None:
                rows.append(_row(yr, None, "missing_price"))
                continue
            if shares is None or pd.isna(shares) or shares <= 0:
                rows.append(_row(yr, None, "missing_shares"))
                continue

            sf         = _get_split_factor(t, yr)
            price_hist = price_adj * sf
            mc         = price_hist * shares

            flag = "ok" if mc > 1e6 else "missing_price"
            rows.append(_row(yr, mc if flag == "ok" else None, flag))

    except Exception as exc:
        rows = [_row(yr, None, "yf_error") for yr in YEARS]

    return rows


def fetch_and_save_market_cap(mapping_df: pd.DataFrame, xbrl_df: pd.DataFrame,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Fetch market cap estimates for every company in mapping_df and save
    results to market_cap_data.csv.  Skips tickers already present in an
    existing CSV so reruns are safe.

    xbrl_df must include columns: company_id, year, CommonStockSharesOutstanding
    """
    import yfinance as yf  # noqa: F401 — confirm installed before loop

    # Load existing to allow incremental reruns
    if os.path.exists(MARKET_CAP_PATH):
        existing = pd.read_csv(MARKET_CAP_PATH, dtype={"company_id": str})
        done_tickers = set(existing["ticker"].unique())
    else:
        existing     = pd.DataFrame()
        done_tickers = set()

    # Build shares lookup: company_id -> {year -> shares}
    if "CommonStockSharesOutstanding" in xbrl_df.columns:
        shares_col = "CommonStockSharesOutstanding"
    else:
        shares_col = None

    new_rows = []
    total    = len(mapping_df)

    for i, (_, mrow) in enumerate(mapping_df.iterrows(), 1):
        cid    = str(mrow["company_id"])
        ticker = str(mrow["ticker"])

        if ticker in done_tickers:
            if verbose:
                print(f"  [{i}/{total}] {ticker}: already cached, skipping")
            continue

        # Build per-year shares dict with normalisation
        if shares_col and cid in xbrl_df["company_id"].values:
            co = xbrl_df[xbrl_df["company_id"] == cid].sort_values("year")
            raw_shares = pd.Series(
                {int(r["year"]): r[shares_col]
                 for _, r in co.iterrows()
                 if pd.notna(r.get(shares_col))},
                dtype=float,
            )
            raw_shares = _normalize_shares(raw_shares)
            shares_by_year = raw_shares.to_dict()
        else:
            shares_by_year = {}

        company_rows = _fetch_for_one_company(ticker, cid, shares_by_year)
        new_rows.extend(company_rows)

        n_ok = sum(1 for r in company_rows if r["data_quality_flag"] == "ok")
        if verbose:
            print(f"  [{i}/{total}] {ticker}: {n_ok}/{len(YEARS)} years with market cap")

        time.sleep(0.3)  # polite rate limit

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    else:
        combined = existing

    combined.to_csv(MARKET_CAP_PATH, index=False)
    return combined


def load_market_cap_data() -> tuple:
    """
    Load market_cap_data.csv if it exists.
    Returns (df, summary_dict). df is None if file not found.
    """
    if not os.path.exists(MARKET_CAP_PATH):
        return None, {"status": "not_found"}

    try:
        df   = pd.read_csv(MARKET_CAP_PATH, dtype={"company_id": str})
        ok   = df[df["data_quality_flag"] == "ok"]
        return df, {
            "status":         "loaded",
            "total_rows":     len(df),
            "ok_rows":        len(ok),
            "companies":      df["ticker"].nunique(),
            "years":          sorted(df["year"].dropna().astype(int).unique().tolist()),
        }
    except Exception as exc:
        return None, {"status": "error", "error": str(exc)}


# ── Step 3: Merge XBRL features + market cap ──────────────────────────────────

def build_training_dataset(xbrl_df: pd.DataFrame,
                            market_cap_df: pd.DataFrame,
                            mapping_df: pd.DataFrame,
                            save: bool = True) -> tuple:
    """
    Join XBRL company-year features with market cap estimates.
    Rows with usable market_cap (flag == 'ok') and non-null core features
    are kept for model training.

    Returns (training_df, summary_dict).
    """
    mc_ok = market_cap_df[market_cap_df["data_quality_flag"] == "ok"][
        ["company_id", "ticker", "year", "market_cap", "data_quality_flag"]
    ].copy()
    mc_ok["year"] = mc_ok["year"].astype(int)

    # Attach company name
    if mapping_df is not None and "company_name" in mapping_df.columns:
        names = mapping_df[["company_id", "company_name"]].copy()
        mc_ok = mc_ok.merge(names, on="company_id", how="left")

    # Join to XBRL features
    xbrl_df = xbrl_df.copy()
    xbrl_df["year"] = xbrl_df["year"].astype(int)
    merged = mc_ok.merge(xbrl_df, on=["company_id", "year"], how="inner")

    # log market cap
    merged["log_market_cap"] = np.where(
        merged["market_cap"] > 0,
        np.log(merged["market_cap"]),
        np.nan,
    )

    usable = merged.dropna(subset=["log_market_cap", "assets"])
    n_total  = len(merged)
    n_usable = len(usable)

    summary = {
        "companies_mapped":     mc_ok["ticker"].nunique(),
        "total_joined_rows":    n_total,
        "usable_training_rows": n_usable,
        "years_covered":        sorted(usable["year"].unique().tolist()) if n_usable > 0 else [],
        "missing_feature_rows": n_total - n_usable,
    }

    if save and n_usable > 0:
        usable.to_csv(TRAINING_DATA_PATH, index=False)

    return usable, summary


def load_training_dataset() -> tuple:
    """
    Load valuation_training_dataset.csv if it exists.
    Returns (df, summary_dict).
    """
    if not os.path.exists(TRAINING_DATA_PATH):
        return None, {"status": "not_found"}
    try:
        df = pd.read_csv(TRAINING_DATA_PATH, dtype={"company_id": str})
        return df, {
            "status":    "loaded",
            "rows":      len(df),
            "companies": df["ticker"].nunique() if "ticker" in df.columns else "?",
            "years":     sorted(df["year"].dropna().astype(int).unique().tolist()),
        }
    except Exception as exc:
        return None, {"status": "error", "error": str(exc)}


# ── Standalone script entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Market Cap Data Collection Script")
    print("=" * 60)

    # Step 1: Ticker mapping
    mapping, msg = load_ticker_mapping()
    print(f"\nStep 1 — Ticker mapping: {msg}")
    if mapping is None:
        sys.exit(1)

    # Step 2: XBRL data (needed for shares outstanding)
    print("\nStep 2 — Loading XBRL data for shares outstanding...")
    from real_model_utils import load_xbrl_data
    xbrl_df, diag = load_xbrl_data()
    if xbrl_df is None:
        print(f"  ERROR: {diag.get('error')}")
        sys.exit(1)
    print(f"  Loaded {len(xbrl_df):,} company-year rows")

    # Re-attach raw shares column before pivot loses it
    # (load_xbrl_data renamed most columns but kept CommonStockSharesOutstanding)
    print(f"  Shares column present: {'CommonStockSharesOutstanding' in xbrl_df.columns}")

    # Step 3: Fetch / update market cap
    print(f"\nStep 3 — Fetching market cap data ({len(mapping)} companies × up to {len(YEARS)} years)...")
    mc_df = fetch_and_save_market_cap(mapping, xbrl_df, verbose=True)
    ok_rows = (mc_df["data_quality_flag"] == "ok").sum()
    print(f"\n  Saved {MARKET_CAP_PATH}")
    print(f"  Total rows: {len(mc_df)}  |  Usable (ok): {ok_rows}")

    # Step 4: Build training dataset
    print("\nStep 4 — Building training dataset...")
    train_df, summary = build_training_dataset(xbrl_df, mc_df, mapping, save=True)
    print(f"  Companies mapped:      {summary['companies_mapped']}")
    print(f"  Usable training rows:  {summary['usable_training_rows']}")
    print(f"  Years covered:         {summary['years_covered']}")
    print(f"  Saved: {TRAINING_DATA_PATH}")

    # Step 5: Train market-cap model and generate model_outputs.csv
    if summary['usable_training_rows'] >= 30:
        print("\nStep 5 — Training market-cap model and generating outputs...")
        from real_model_utils import (
            train_valuation_model, generate_model_outputs, save_model_artifacts,
            MODEL_OUTPUTS_PATH, MODEL_METRICS_PATH, FEATURE_IMP_PATH,
        )
        model_result, fitted = train_valuation_model(train_df)
        if model_result.get("status") == "success":
            print(f"  Best model: {model_result.get('best_model')}")
            for model_name, m in model_result["models"].items():
                marker = " *" if model_name == model_result.get("best_model") else "  "
                print(f" {marker} {model_name}: R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
            for base_name, bm in model_result.get("baseline", {}).items():
                print(f"    {base_name}: R²={bm['r2']:.4f}  MAE={bm['mae']:.4f}  RMSE={bm['rmse']:.4f}")
            gen_df = generate_model_outputs(train_df, fitted, save=True)
            print(f"  Saved: {MODEL_OUTPUTS_PATH}  ({len(gen_df)} rows)")
            save_model_artifacts(model_result)
            print(f"  Saved: {MODEL_METRICS_PATH}")
            print(f"  Saved: {FEATURE_IMP_PATH}")
        else:
            print(f"  WARNING: model training failed — {model_result.get('error', model_result.get('status'))}")
    else:
        print(f"\nStep 5 — Skipped (only {summary['usable_training_rows']} training rows, need >= 30).")

    # Step 6: Fetch current market data for mapped companies
    if mapping is not None and len(mapping) > 0:
        print("\nStep 6 — Fetching current market data (this may take ~1 min)...")
        from data_utils import fetch_current_market_data, CURRENT_MARKET_CSV
        tickers_to_fetch = mapping["ticker"].dropna().str.upper().unique().tolist()
        cm_df = fetch_current_market_data(tickers_to_fetch, save=True)
        ok = (cm_df["data_quality_flag"] == "ok").sum() if len(cm_df) > 0 else 0
        print(f"  {ok}/{len(tickers_to_fetch)} tickers with current price data")
        print(f"  Saved: {CURRENT_MARKET_CSV}")

    print("\nDone. Run:  python3 -m streamlit run app.py")
