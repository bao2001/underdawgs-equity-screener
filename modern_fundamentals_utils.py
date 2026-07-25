"""
modern_fundamentals_utils.py
────────────────────────────
Expands valuation model coverage from the 2010–2016 XBRL dataset to
2017–present using SEC EDGAR Company Facts API (free, official).

Usage (one-time setup, ~5–10 min):
    python3 modern_fundamentals_utils.py

Before running:
    Edit USER_AGENT below. SEC policy requires identification.
    Format: "Your Name your.email@example.com"

Files generated:
    sec_company_facts_cache/          per-company JSON cache (fetch once)
    ticker_cik_mapping.csv            ticker ↔ CIK lookup table
    modern_fundamentals.csv           annual SEC fundamentals 2017–present
    modern_market_cap_data.csv        year-end market caps via yfinance
    modern_valuation_training_dataset.csv
    modern_model_outputs.csv          model signals for 2017–present
    model_outputs_combined.csv        merged 2010–2016 XBRL + 2017–present

Notes:
    - report_risk_score = 50 (neutral) because SEC filing text extraction
      is not integrated. No filing-sentiment risk is implied.
    - All outputs are research signals only. Not investment advice.
    - Valuation gaps are model estimates, not proof of under/overvaluation.
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

SEC_CACHE_DIR              = os.path.join(_HERE, "sec_company_facts_cache")
TICKER_CIK_MAP_PATH        = os.path.join(_HERE, "ticker_cik_mapping.csv")
MODERN_FUND_PATH           = os.path.join(_HERE, "modern_fundamentals.csv")
MODERN_MC_PATH             = os.path.join(_HERE, "modern_market_cap_data.csv")
MODERN_TRAINING_PATH       = os.path.join(_HERE, "modern_valuation_training_dataset.csv")
MODERN_OUTPUTS_PATH        = os.path.join(_HERE, "modern_model_outputs.csv")
COMBINED_OUTPUTS_PATH      = os.path.join(_HERE, "model_outputs_combined.csv")

# Source tag values for combined file
SRC_LEGACY  = "legacy_xbrl_2010_2016"
SRC_MODERN  = "modern_sec_facts_2017_current"

# ── User-agent (required by SEC) ───────────────────────────────────────────────

# Set via environment variable or edit the default below.
# export SEC_USER_AGENT="Your Name your.email@domain.com"
USER_AGENT = os.getenv("SEC_USER_AGENT", "MSDSBA Practicum baoonguyen80@gmail.com")

SEC_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# Target years for modern expansion
MODERN_YEARS = list(range(2017, 2025))   # 2017–2024

# Minimum rows before skipping model training
MIN_TRAINING_ROWS = 20

# ── SEC concept fallback chains ────────────────────────────────────────────────
# Ordered lists: first match wins.

_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerNetOfTaxes",
    "NetRevenues",
    "TotalRevenues",
]
_NET_INCOME_CONCEPTS  = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
_ASSETS_CONCEPTS      = ["Assets"]
_LIAB_CONCEPTS        = ["Liabilities"]
_EQUITY_CONCEPTS      = ["StockholdersEquity", "StockholdersEquityAttributableToParent",
                          "Equity", "LiabilitiesAndStockholdersEquity"]
_CASH_CONCEPTS        = ["CashAndCashEquivalentsAtCarryingValue",
                          "CashCashEquivalentsAndShortTermInvestments",
                          "CashAndCashEquivalentsPeriodIncreaseDecrease"]
_DEBT_CONCEPTS        = ["LongTermDebt", "LongTermDebtNoncurrent",
                          "LongTermDebtAndCapitalLeaseObligations"]
_OP_INCOME_CONCEPTS   = ["OperatingIncomeLoss"]
_GROSS_PROFIT_CONCEPTS = ["GrossProfit"]
_SHARES_CONCEPTS      = [
    "EntityCommonStockSharesOutstanding",    # DEI namespace
    "CommonStockSharesOutstanding",          # us-gaap
]

# ── Ticker→CIK mapping ─────────────────────────────────────────────────────────

def create_ticker_cik_mapping() -> pd.DataFrame:
    """
    Build ticker_cik_mapping.csv from ticker_mapping.csv.
    ticker_mapping.company_id IS the SEC CIK for these XBRL-derived companies.
    """
    src_path = os.path.join(_HERE, "ticker_mapping.csv")
    if not os.path.exists(src_path):
        print(f"  Warning: {src_path} not found — cannot create ticker_cik_mapping.csv")
        return pd.DataFrame(columns=["ticker", "company_name", "cik",
                                      "source", "mapping_quality_flag"])
    raw = pd.read_csv(src_path, dtype={"company_id": str})
    df = pd.DataFrame({
        "ticker":               raw["ticker"].str.upper().str.strip(),
        "company_name":         raw.get("company_name", pd.Series("", index=raw.index)),
        "cik":                  raw["company_id"].str.strip(),
        "source":               "ticker_mapping_csv",
        "mapping_quality_flag": "ok",
    })
    df.to_csv(TICKER_CIK_MAP_PATH, index=False)
    return df


def load_ticker_cik_mapping() -> tuple:
    """
    Load ticker_cik_mapping.csv.
    If it does not exist, auto-create from ticker_mapping.csv.
    Returns (df, status_msg). df is None on hard failure.
    """
    if not os.path.exists(TICKER_CIK_MAP_PATH):
        df = create_ticker_cik_mapping()
        if df.empty:
            return None, (
                "ticker_cik_mapping.csv not found and could not be created. "
                "Ensure ticker_mapping.csv is present in the project directory."
            )
        return df, f"Created ticker_cik_mapping.csv with {len(df)} entries."

    try:
        df = pd.read_csv(TICKER_CIK_MAP_PATH, dtype={"cik": str})
        df = df.dropna(subset=["ticker", "cik"])
        df["cik"] = df["cik"].str.strip()
        return df, f"Loaded {len(df)} CIK mappings."
    except Exception as exc:
        return None, f"Error reading ticker_cik_mapping.csv: {exc}"


# ── SEC Company Facts cache ────────────────────────────────────────────────────

def _cache_path(cik: str) -> str:
    return os.path.join(SEC_CACHE_DIR, f"CIK{str(cik).zfill(10)}.json")


def cache_company_facts(cik: str, data: dict) -> None:
    os.makedirs(SEC_CACHE_DIR, exist_ok=True)
    with open(_cache_path(cik), "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_cached_company_facts(cik: str) -> dict | None:
    path = _cache_path(cik)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fetch_company_facts(cik: str, user_agent: str = USER_AGENT) -> tuple:
    """
    Fetch company facts from SEC EDGAR API.
    Tries `requests` first (handles SSL certificates reliably); falls back to
    `urllib` with a permissive SSL context for environments where system certs
    are unavailable.
    Returns (data_dict, status_msg). data_dict is None on failure.
    """
    padded  = str(cik).zfill(10)
    url     = f"{SEC_BASE_URL}/CIK{padded}.json"
    headers = {"User-Agent": user_agent}

    # Primary: requests (handles macOS SSL certificates automatically)
    _requests_err = None
    try:
        import requests as _req
        r = _req.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json(), "ok"
    except ImportError:
        pass   # requests not installed — fall through to urllib
    except Exception as exc:
        _requests_err = str(exc)
        # Fall through to urllib fallback below

    # Fallback: urllib with permissive SSL context
    try:
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        try:
            ctx.load_default_certs()
        except Exception:
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8")), "ok"
    except Exception as exc:
        parts = []
        if _requests_err:
            parts.append(f"requests: {_requests_err}")
        parts.append(f"urllib: {exc}")
        return None, " | ".join(parts)


def get_or_fetch_company_facts(cik: str, user_agent: str = USER_AGENT,
                                force_refresh: bool = False) -> tuple:
    """
    Cache-first loader. Returns (data_dict, source, status_msg).
    source is "cache" or "api".
    """
    if not force_refresh:
        cached = load_cached_company_facts(cik)
        if cached is not None:
            return cached, "cache", "ok"

    data, msg = fetch_company_facts(cik, user_agent)
    if data is not None:
        cache_company_facts(cik, data)
        return data, "api", "ok"
    return None, "api", msg


# ── Annual fact extraction ─────────────────────────────────────────────────────

def _extract_annual_concept(facts_dict: dict, concept_names: list,
                              target_years: list) -> dict:
    """
    Extract the best annual value for each target year from company facts.

    Searches both us-gaap and dei namespaces.
    Prefers 10-K/20-F annual filings with fp="FY".
    Returns dict: {year_int: float_value} for years where data was found.
    """
    ns_keys = ["us-gaap", "dei", "ifrs-full"]
    result  = {}

    for concept in concept_names:
        if len(result) == len(target_years):
            break   # all years found — no need to try more concepts
        for ns in ns_keys:
            ns_facts = facts_dict.get("facts", {}).get(ns, {})
            if concept not in ns_facts:
                continue
            units_dict = ns_facts[concept].get("units", {})
            # USD is the unit for financial statement items; shares use "shares"
            for unit_key in ("USD", "shares", "pure"):
                if unit_key not in units_dict:
                    continue
                entries = units_dict[unit_key]
                # Group by fiscal year; keep annual (10-K or FY) filings only
                by_year: dict[int, list] = {}
                for e in entries:
                    form = str(e.get("form", "")).upper()
                    fp   = str(e.get("fp",   "")).upper()
                    fy   = e.get("fy")
                    val  = e.get("val")
                    if val is None or fy is None:
                        continue
                    is_annual = (form in ("10-K", "10-K405", "20-F", "40-F")
                                 or fp == "FY")
                    if not is_annual:
                        continue
                    yr_int = int(fy)
                    if yr_int not in target_years:
                        continue
                    by_year.setdefault(yr_int, []).append(e)

                # For each year, pick the entry filed latest (most recent amendment)
                for yr, elist in by_year.items():
                    if yr in result:
                        continue    # already found from a preferred concept
                    best = max(elist, key=lambda x: x.get("filed", ""))
                    result[yr] = float(best["val"])
            if len(result) == len(target_years):
                break

    return result


def extract_annual_fundamentals(cik: str, ticker: str, company_name: str,
                                  entity_facts: dict,
                                  target_years: list = None) -> list:
    """
    Extract a list of company-year dicts from SEC entity facts.
    One row per year where at least assets or revenue is non-null.
    """
    if target_years is None:
        target_years = MODERN_YEARS

    def _get(concept_names):
        return _extract_annual_concept(entity_facts, concept_names, target_years)

    revenue         = _get(_REVENUE_CONCEPTS)
    net_income      = _get(_NET_INCOME_CONCEPTS)
    assets          = _get(_ASSETS_CONCEPTS)
    liabilities     = _get(_LIAB_CONCEPTS)
    equity          = _get(_EQUITY_CONCEPTS)
    cash            = _get(_CASH_CONCEPTS)
    debt            = _get(_DEBT_CONCEPTS)
    op_income       = _get(_OP_INCOME_CONCEPTS)
    gross_profit    = _get(_GROSS_PROFIT_CONCEPTS)
    shares_out      = _get(_SHARES_CONCEPTS)

    rows = []
    for yr in sorted(target_years):
        rev_v  = revenue.get(yr)
        ast_v  = assets.get(yr)
        if rev_v is None and ast_v is None:
            continue    # no useful data for this year

        # Quality flag
        missing = []
        if rev_v  is None: missing.append("missing_revenue")
        if ast_v  is None: missing.append("missing_assets")
        qflag = ", ".join(missing) if missing else "ok"

        rows.append({
            "ticker":              ticker,
            "company_name":        company_name,
            "cik":                 str(cik),
            "year":                yr,
            "revenue":             rev_v,
            "net_income":          net_income.get(yr),
            "assets":              ast_v,
            "liabilities":         liabilities.get(yr),
            "equity":              equity.get(yr),
            "cash":                cash.get(yr),
            "debt":                debt.get(yr),
            "operating_income":    op_income.get(yr),
            "gross_profit":        gross_profit.get(yr),
            "shares_outstanding":  shares_out.get(yr),
            "data_source":         "sec_company_facts",
            "data_quality_flag":   qflag,
        })

    return rows


# ── Fundamentals ingestion pipeline ───────────────────────────────────────────

def build_modern_fundamentals(ticker_cik_df: pd.DataFrame,
                               target_years: list = None,
                               user_agent: str = USER_AGENT,
                               verbose: bool = True) -> pd.DataFrame:
    """
    For each ticker in ticker_cik_df, fetch SEC company facts (cached)
    and extract annual fundamentals for target_years.

    Returns a DataFrame (one row per ticker-year) and saves to
    modern_fundamentals.csv.
    """
    if target_years is None:
        target_years = MODERN_YEARS

    all_rows = []
    total    = len(ticker_cik_df)

    for i, (_, row) in enumerate(ticker_cik_df.iterrows(), 1):
        ticker       = str(row["ticker"]).upper()
        cik          = str(row["cik"]).strip()
        company_name = str(row.get("company_name", ticker))

        if verbose:
            print(f"  [{i:2d}/{total}] {ticker} (CIK {cik})...", end=" ", flush=True)

        data, source, msg = get_or_fetch_company_facts(cik, user_agent)
        if data is None:
            if verbose:
                print(f"FAILED: {msg}")
            continue

        rows = extract_annual_fundamentals(cik, ticker, company_name,
                                           data, target_years)
        all_rows.extend(rows)

        n_ok = sum(1 for r in rows if r["data_quality_flag"] == "ok")
        if verbose:
            print(f"{len(rows)} year rows ({n_ok} ok)  [{source}]")

        if source == "api":
            time.sleep(0.12)    # polite: stay well under SEC 10 req/s limit

    if not all_rows:
        if verbose:
            print("  No fundamentals rows extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_csv(MODERN_FUND_PATH, index=False)
    if verbose:
        print(f"\n  Saved {len(df)} rows → modern_fundamentals.csv")
    return df


def load_modern_fundamentals() -> tuple:
    if not os.path.exists(MODERN_FUND_PATH):
        return None, "modern_fundamentals.csv not found — run modern_fundamentals_utils.py"
    try:
        df = pd.read_csv(MODERN_FUND_PATH)
        return df, f"Loaded {len(df)} rows, years {sorted(df['year'].dropna().astype(int).unique().tolist())}"
    except Exception as exc:
        return None, f"Error loading modern_fundamentals.csv: {exc}"


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the same feature set used by real_model_utils.py's FEATURE_COLS.
    Input: fundamentals DataFrame with columns from extract_annual_fundamentals.
    Returns new DataFrame with engineered columns added.
    """
    out = df.copy()

    def _div(num, den):
        return np.where(pd.Series(den).abs() > 1e-9,
                        pd.Series(num) / pd.Series(den),
                        np.nan)

    out["profit_margin"]    = _div(out["net_income"],      out["revenue"])
    out["gross_margin"]     = _div(out["gross_profit"],    out["revenue"])
    out["operating_margin"] = _div(out["operating_income"], out["revenue"])
    out["debt_to_assets"]   = _div(out["debt"],             out["assets"])
    out["return_on_assets"] = _div(out["net_income"],       out["assets"])
    out["return_on_equity"] = _div(out["net_income"],       out["equity"])
    out["cash_to_assets"]   = _div(out["cash"],             out["assets"])

    # YoY growth within each company
    out = out.sort_values(["ticker", "year"])
    out["revenue_growth"] = (
        out.groupby("ticker")["revenue"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )
    out["net_income_growth"] = (
        out.groupby("ticker")["net_income"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    # Log transforms
    out["log_assets"]  = np.where(out["assets"]  > 0, np.log(out["assets"]),  np.nan)
    out["log_revenue"] = np.where(out["revenue"] > 0, np.log(out["revenue"]), np.nan)

    # Flag rows with insufficient features
    core_features = ["log_revenue", "net_income", "assets", "liabilities",
                     "equity", "debt_to_assets", "return_on_assets"]
    non_null_count = out[core_features].notna().sum(axis=1)
    out["data_quality_flag"] = np.where(
        non_null_count < 3,
        "insufficient_features",
        out.get("data_quality_flag", "ok"),
    )

    return out


# ── Modern market cap via yfinance ─────────────────────────────────────────────

def _get_split_factor(ticker_obj, year_int: int) -> float:
    """Same split-correction logic as market_cap_utils.py."""
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


def fetch_modern_market_caps(ticker_cik_df: pd.DataFrame,
                              fundamentals_df: pd.DataFrame,
                              target_years: list = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Estimate year-end market caps for modern years using yfinance.
    Strategy:
      1. Get Dec monthly price from yfinance history
      2. Apply split correction to un-adjust to historical price
      3. Multiply by shares_outstanding from SEC facts (if available)
         or fall back to yfinance shares

    Returns DataFrame saved to modern_market_cap_data.csv.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — cannot fetch market caps")
        return pd.DataFrame()

    if target_years is None:
        target_years = MODERN_YEARS

    min_year = min(target_years)
    max_year = max(target_years)

    # Build SEC shares lookup: ticker → {year → shares_outstanding}
    shares_lookup: dict = {}
    if fundamentals_df is not None and not fundamentals_df.empty and \
            "shares_outstanding" in fundamentals_df.columns:
        for _, row in fundamentals_df.iterrows():
            t   = str(row["ticker"]).upper()
            yr  = int(row["year"])
            sh  = row.get("shares_outstanding")
            if pd.notna(sh) and sh > 0:
                shares_lookup.setdefault(t, {})[yr] = float(sh)

    rows    = []
    total   = len(ticker_cik_df)

    for i, (_, mrow) in enumerate(ticker_cik_df.iterrows(), 1):
        ticker = str(mrow["ticker"]).upper()
        if verbose:
            print(f"  [{i:2d}/{total}] {ticker}...", end=" ", flush=True)

        ok_count = 0
        try:
            t_obj = yf.Ticker(ticker)
            hist  = yf.download(
                ticker,
                start=f"{min_year - 1}-11-01",
                end=f"{max_year + 1}-01-31",
                interval="1mo",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if not hist.empty:
                hist.index = pd.to_datetime(hist.index)
                # Strip tz
                if getattr(hist.index, "tz", None) is not None:
                    hist.index = hist.index.tz_convert(None)

            for yr in target_years:
                # December close for this year
                dec_rows = hist[
                    (hist.index.year == yr) & (hist.index.month == 12)
                ] if not hist.empty else pd.DataFrame()

                dec_price_adj = (
                    float(dec_rows["Close"].iloc[-1])
                    if not dec_rows.empty and "Close" in dec_rows.columns
                    else None
                )

                if dec_price_adj is None:
                    rows.append({
                        "ticker": ticker, "year": yr,
                        "market_cap": None, "price_used": None,
                        "shares_used": None, "data_source": "yfinance",
                        "data_quality_flag": "missing_price",
                    })
                    continue

                # Split-adjust price back to historical
                sf           = _get_split_factor(t_obj, yr)
                price_hist   = dec_price_adj * sf

                # Shares: prefer SEC facts, else yfinance
                shares = shares_lookup.get(ticker, {}).get(yr)
                share_src = "sec_facts"
                if shares is None or shares <= 0:
                    fi = t_obj.fast_info
                    shares = getattr(fi, "shares", None)
                    share_src = "yfinance_fast_info"

                if shares is None or shares <= 0:
                    rows.append({
                        "ticker": ticker, "year": yr,
                        "market_cap": None, "price_used": price_hist,
                        "shares_used": None, "data_source": "yfinance",
                        "data_quality_flag": "missing_shares",
                    })
                    continue

                mc   = price_hist * shares
                flag = "ok" if mc > 1e7 else "suspicious_market_cap"
                rows.append({
                    "ticker": ticker, "year": yr,
                    "market_cap":  mc if flag == "ok" else None,
                    "price_used":  price_hist,
                    "shares_used": shares,
                    "data_source": f"yfinance+{share_src}",
                    "data_quality_flag": flag,
                })
                if flag == "ok":
                    ok_count += 1

        except Exception as exc:
            for yr in target_years:
                rows.append({
                    "ticker": ticker, "year": yr,
                    "market_cap": None, "price_used": None,
                    "shares_used": None, "data_source": "yfinance",
                    "data_quality_flag": f"yf_error",
                })

        if verbose:
            print(f"{ok_count}/{len(target_years)} years OK")
        time.sleep(0.25)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(MODERN_MC_PATH, index=False)
        if verbose:
            ok_n = (df["data_quality_flag"] == "ok").sum()
            print(f"\n  Saved {len(df)} rows → modern_market_cap_data.csv  ({ok_n} ok)")
    return df


def load_modern_market_caps() -> tuple:
    if not os.path.exists(MODERN_MC_PATH):
        return None, "modern_market_cap_data.csv not found"
    try:
        df  = pd.read_csv(MODERN_MC_PATH)
        ok  = (df["data_quality_flag"] == "ok").sum()
        return df, f"Loaded {len(df)} rows ({ok} ok)"
    except Exception as exc:
        return None, f"Error: {exc}"


# ── Modern training dataset ────────────────────────────────────────────────────

def build_modern_training_dataset(fund_df: pd.DataFrame,
                                   mc_df: pd.DataFrame,
                                   verbose: bool = True) -> tuple:
    """
    Merge fundamentals + market caps, engineer features, and save
    modern_valuation_training_dataset.csv.

    Returns (training_df, summary_dict).
    """
    mc_ok = mc_df[mc_df["data_quality_flag"] == "ok"][
        ["ticker", "year", "market_cap", "data_quality_flag"]
    ].copy()
    mc_ok["year"] = mc_ok["year"].astype(int)

    fund_fe = engineer_features(fund_df)
    fund_fe["year"] = fund_fe["year"].astype(int)

    merged = mc_ok.merge(fund_fe, on=["ticker", "year"], how="inner")
    merged["log_market_cap"] = np.where(
        merged["market_cap"] > 0,
        np.log(merged["market_cap"]),
        np.nan,
    )

    # Add placeholder company_id (use CIK if available)
    if "cik" in merged.columns:
        merged["company_id"] = merged["cik"]
    else:
        merged["company_id"] = merged["ticker"]

    usable = merged.dropna(subset=["log_market_cap", "log_assets"])
    n_total  = len(merged)
    n_usable = len(usable)

    summary = {
        "total_joined_rows":    n_total,
        "usable_training_rows": n_usable,
        "years_covered":        sorted(usable["year"].unique().tolist()) if n_usable > 0 else [],
        "companies":            usable["ticker"].nunique() if n_usable > 0 else 0,
    }

    if n_usable > 0:
        usable.to_csv(MODERN_TRAINING_PATH, index=False)
        if verbose:
            print(f"  Saved {n_usable} usable rows → modern_valuation_training_dataset.csv"
                  f"  (years: {summary['years_covered']})")

    return usable, summary


def load_modern_training_dataset() -> tuple:
    if not os.path.exists(MODERN_TRAINING_PATH):
        return None, "modern_valuation_training_dataset.csv not found"
    try:
        df = pd.read_csv(MODERN_TRAINING_PATH)
        return df, f"Loaded {len(df)} rows"
    except Exception as exc:
        return None, f"Error: {exc}"


# ── Modern model outputs ───────────────────────────────────────────────────────

def generate_modern_model_outputs(training_df: pd.DataFrame,
                                   legacy_training_df: pd.DataFrame = None,
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Train valuation model on combined (legacy + modern) training data if
    legacy is provided; otherwise train on modern-only.

    Uses real_model_utils infrastructure for training and output generation.
    Saves modern_model_outputs.csv (2017+ rows only).

    Returns the modern outputs DataFrame.
    """
    try:
        from real_model_utils import (
            train_valuation_model, generate_model_outputs,
            flag_model_outputs, FEATURE_COLS,
        )
    except ImportError as exc:
        if verbose:
            print(f"  Cannot import real_model_utils: {exc}")
        return pd.DataFrame()

    # Combine legacy + modern for training if legacy available
    if legacy_training_df is not None and not legacy_training_df.empty:
        combined_train = pd.concat([legacy_training_df, training_df],
                                   ignore_index=True)
        if verbose:
            print(f"  Training on combined dataset: {len(legacy_training_df)} legacy "
                  f"+ {len(training_df)} modern = {len(combined_train)} rows")
    else:
        combined_train = training_df
        if verbose:
            print(f"  Training on modern-only dataset: {len(training_df)} rows")

    # Check minimum size
    features = [f for f in FEATURE_COLS if f in combined_train.columns]
    usable   = combined_train.dropna(subset=["log_market_cap"] + features[:3])
    if len(usable) < MIN_TRAINING_ROWS:
        if verbose:
            print(f"  Insufficient training rows ({len(usable)} < {MIN_TRAINING_ROWS})")
        return pd.DataFrame()

    # Train
    model_result, fitted = train_valuation_model(combined_train)
    if model_result.get("status") != "success":
        if verbose:
            print(f"  Training failed: {model_result.get('error', model_result.get('status'))}")
        return pd.DataFrame()

    if verbose:
        best   = model_result.get("best_model", "?")
        models = model_result.get("models", {})
        r2     = models.get(best, {}).get("r2", "?")
        print(f"  Best model: {best}  R²={r2}")

    # Generate outputs for ALL combined data, then filter to modern years only
    all_outputs = generate_model_outputs(combined_train, fitted, save=False)
    if all_outputs.empty:
        if verbose:
            print("  generate_model_outputs returned empty DataFrame")
        return pd.DataFrame()

    modern_outputs = all_outputs[
        all_outputs["year"].astype(int) >= 2017
    ].copy()

    if modern_outputs.empty:
        if verbose:
            print("  No 2017+ rows in model outputs")
        return pd.DataFrame()

    # Add quality flags
    modern_reviewed = flag_model_outputs(modern_outputs)
    modern_reviewed.to_csv(MODERN_OUTPUTS_PATH, index=False)

    if verbose:
        ok_n = (modern_reviewed["output_quality_flag"].str.startswith("ok")).sum()
        print(f"  Saved {len(modern_reviewed)} rows → modern_model_outputs.csv"
              f"  ({ok_n} ok, years: {sorted(modern_reviewed['year'].astype(int).unique().tolist())})")

    return modern_reviewed


def load_modern_model_outputs() -> tuple:
    if not os.path.exists(MODERN_OUTPUTS_PATH):
        return None, "modern_model_outputs.csv not found"
    try:
        df = pd.read_csv(MODERN_OUTPUTS_PATH)
        yrs = sorted(df["year"].dropna().astype(int).unique().tolist())
        return df, f"Loaded {len(df)} rows, years {yrs}"
    except Exception as exc:
        return None, f"Error: {exc}"


# ── Combined outputs ───────────────────────────────────────────────────────────

def build_combined_outputs(verbose: bool = True) -> tuple:
    """
    Merge model_outputs_reviewed.csv (2010–2016) with modern_model_outputs.csv
    (2017+) into model_outputs_combined.csv.

    Adds model_output_source column.
    Returns (combined_df, status_msg). df is None if both sources are missing.
    """
    from data_utils import MODEL_OUTPUTS_REVIEWED_CSV

    legacy_df  = None
    modern_df  = None

    if os.path.exists(MODEL_OUTPUTS_REVIEWED_CSV):
        legacy_df = pd.read_csv(MODEL_OUTPUTS_REVIEWED_CSV)
        legacy_df["model_output_source"] = SRC_LEGACY

    if os.path.exists(MODERN_OUTPUTS_PATH):
        modern_df = pd.read_csv(MODERN_OUTPUTS_PATH)
        modern_df["model_output_source"] = SRC_MODERN

    if legacy_df is None and modern_df is None:
        return None, "Both model_outputs_reviewed.csv and modern_model_outputs.csv are missing."

    parts = [df for df in [legacy_df, modern_df] if df is not None]
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined.to_csv(COMBINED_OUTPUTS_PATH, index=False)

    legacy_yrs  = sorted(legacy_df["year"].astype(int).unique().tolist())  if legacy_df  is not None else []
    modern_yrs  = sorted(modern_df["year"].astype(int).unique().tolist())  if modern_df  is not None else []
    msg = (
        f"Combined {len(combined)} rows: "
        f"legacy {legacy_yrs}, modern {modern_yrs}"
    )
    if verbose:
        print(f"  {msg}")
        print(f"  Saved → model_outputs_combined.csv")
    return combined, msg


def load_combined_outputs() -> tuple:
    """
    Load model_outputs_combined.csv if present, else fall back to
    model_outputs_reviewed.csv.
    Returns (df, status_msg, source_label).
    """
    if os.path.exists(COMBINED_OUTPUTS_PATH):
        try:
            df  = pd.read_csv(COMBINED_OUTPUTS_PATH)
            yrs = sorted(df["year"].dropna().astype(int).unique().tolist())
            return df, f"Combined outputs: {len(df)} rows, years {yrs}", "combined"
        except Exception as exc:
            pass   # fall through to reviewed

    from data_utils import MODEL_OUTPUTS_REVIEWED_CSV
    if os.path.exists(MODEL_OUTPUTS_REVIEWED_CSV):
        try:
            df  = pd.read_csv(MODEL_OUTPUTS_REVIEWED_CSV)
            yrs = sorted(df["year"].dropna().astype(int).unique().tolist())
            return df, f"Legacy outputs only: {len(df)} rows, years {yrs}", "legacy"
        except Exception as exc:
            return None, f"Error loading model outputs: {exc}", "error"

    return None, "No model outputs found.", "missing"


# ── Diagnostics for app.py ─────────────────────────────────────────────────────

def get_modern_pipeline_diagnostics() -> dict:
    """
    Return a dict of coverage statistics for the Model Diagnostics page.
    All values are safe strings/ints — no exceptions propagate.
    """
    d = {
        "ticker_cik_mapping_exists": False,
        "ticker_cik_count":          0,
        "sec_cache_files":           0,
        "modern_fundamentals_exists": False,
        "modern_fundamentals_years": [],
        "modern_fundamentals_rows":  0,
        "modern_training_exists":    False,
        "modern_training_rows":      0,
        "modern_outputs_exists":     False,
        "modern_outputs_years":      [],
        "modern_outputs_rows":       0,
        "combined_outputs_exists":   False,
        "combined_outputs_rows":     0,
        "combined_outputs_years":    [],
        "missing_data_summary":      [],
    }

    try:
        if os.path.exists(TICKER_CIK_MAP_PATH):
            df = pd.read_csv(TICKER_CIK_MAP_PATH)
            d["ticker_cik_mapping_exists"] = True
            d["ticker_cik_count"]          = len(df)
    except Exception:
        pass

    try:
        if os.path.isdir(SEC_CACHE_DIR):
            d["sec_cache_files"] = len([
                f for f in os.listdir(SEC_CACHE_DIR) if f.endswith(".json")
            ])
    except Exception:
        pass

    try:
        if os.path.exists(MODERN_FUND_PATH):
            df = pd.read_csv(MODERN_FUND_PATH)
            d["modern_fundamentals_exists"] = True
            d["modern_fundamentals_rows"]   = len(df)
            d["modern_fundamentals_years"]  = sorted(df["year"].dropna().astype(int).unique().tolist())
    except Exception:
        pass

    try:
        if os.path.exists(MODERN_TRAINING_PATH):
            df = pd.read_csv(MODERN_TRAINING_PATH)
            d["modern_training_exists"] = True
            d["modern_training_rows"]   = len(df)
    except Exception:
        pass

    try:
        if os.path.exists(MODERN_OUTPUTS_PATH):
            df = pd.read_csv(MODERN_OUTPUTS_PATH)
            d["modern_outputs_exists"] = True
            d["modern_outputs_rows"]   = len(df)
            d["modern_outputs_years"]  = sorted(df["year"].dropna().astype(int).unique().tolist())
    except Exception:
        pass

    try:
        if os.path.exists(COMBINED_OUTPUTS_PATH):
            df = pd.read_csv(COMBINED_OUTPUTS_PATH)
            d["combined_outputs_exists"] = True
            d["combined_outputs_rows"]   = len(df)
            d["combined_outputs_years"]  = sorted(df["year"].dropna().astype(int).unique().tolist())
    except Exception:
        pass

    # Missing data summary
    if not d["ticker_cik_mapping_exists"]:
        d["missing_data_summary"].append("ticker_cik_mapping.csv — run modern_fundamentals_utils.py")
    if d["sec_cache_files"] == 0:
        d["missing_data_summary"].append("SEC cache empty — fetch required")
    if not d["modern_fundamentals_exists"]:
        d["missing_data_summary"].append("modern_fundamentals.csv — fetch required")
    if not d["modern_outputs_exists"]:
        d["missing_data_summary"].append("modern_model_outputs.csv — pipeline not run")
    if not d["combined_outputs_exists"]:
        d["missing_data_summary"].append("model_outputs_combined.csv — run pipeline to generate")

    return d


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("Modern Fundamentals Expansion Pipeline")
    print("SEC EDGAR Company Facts → Model Outputs 2017–present")
    print("=" * 65)
    print(f"\nUser-Agent: {USER_AGENT}")
    print(f"Target years: {MODERN_YEARS}")
    print(f"SEC cache: {SEC_CACHE_DIR}")

    # ── Step 1: Ticker/CIK mapping ─────────────────────────────────────────────
    print("\n── Step 1: Ticker/CIK mapping ──")
    tkr_cik, msg = load_ticker_cik_mapping()
    print(f"  {msg}")
    if tkr_cik is None:
        print("  Cannot continue without CIK mapping.")
        sys.exit(1)
    print(f"  Tickers: {sorted(tkr_cik['ticker'].tolist())}")

    # ── Step 2: Fetch SEC Company Facts ────────────────────────────────────────
    print("\n── Step 2: Fetch / load SEC Company Facts ──")
    os.makedirs(SEC_CACHE_DIR, exist_ok=True)
    fund_df = build_modern_fundamentals(
        tkr_cik, target_years=MODERN_YEARS, user_agent=USER_AGENT, verbose=True,
    )
    if fund_df.empty:
        print("  No fundamentals extracted. Check SEC connectivity and USER_AGENT.")
        sys.exit(1)

    # ── Step 3: Year-end market caps ───────────────────────────────────────────
    print("\n── Step 3: Year-end market caps (yfinance) ──")
    mc_df = fetch_modern_market_caps(
        tkr_cik, fund_df, target_years=MODERN_YEARS, verbose=True,
    )
    if mc_df.empty or (mc_df["data_quality_flag"] == "ok").sum() == 0:
        print("  No market caps retrieved. Check yfinance and internet connectivity.")
        sys.exit(1)

    # ── Step 4: Training dataset ───────────────────────────────────────────────
    print("\n── Step 4: Modern training dataset ──")
    modern_train, train_summary = build_modern_training_dataset(
        fund_df, mc_df, verbose=True,
    )
    if modern_train.empty:
        print("  Insufficient data to build training dataset.")
        sys.exit(1)
    print(f"  Summary: {train_summary}")

    # ── Step 5: Load legacy training data for combined model ───────────────────
    print("\n── Step 5: Load legacy training data ──")
    legacy_train = None
    legacy_path  = os.path.join(_HERE, "valuation_training_dataset.csv")
    if os.path.exists(legacy_path):
        legacy_train = pd.read_csv(legacy_path, dtype={"company_id": str})
        print(f"  Loaded {len(legacy_train)} legacy training rows.")
    else:
        print("  valuation_training_dataset.csv not found — training on modern data only.")

    # ── Step 6: Train model and generate outputs ───────────────────────────────
    print("\n── Step 6: Train model and generate modern model outputs ──")
    modern_outputs = generate_modern_model_outputs(
        modern_train, legacy_training_df=legacy_train, verbose=True,
    )
    if modern_outputs.empty:
        print("  Model output generation failed.")
        sys.exit(1)

    # ── Step 7: Build combined outputs ─────────────────────────────────────────
    print("\n── Step 7: Build model_outputs_combined.csv ──")
    combined, msg = build_combined_outputs(verbose=True)
    print(f"  {msg}")

    print("\n" + "=" * 65)
    print("Pipeline complete.")
    print(f"  modern_fundamentals.csv          : {len(fund_df)} rows")
    print(f"  modern_market_cap_data.csv       : {len(mc_df)} rows")
    print(f"  modern_valuation_training_dataset: {len(modern_train)} rows")
    print(f"  modern_model_outputs.csv         : {len(modern_outputs)} rows")
    if combined is not None:
        print(f"  model_outputs_combined.csv       : {len(combined)} rows")
    print("=" * 65)
