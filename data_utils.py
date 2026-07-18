"""
data_utils.py
Data loading, optional live-price refresh, model output loading,
and current market snapshot fetching.
"""

import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV                       = os.path.join(_DIR, "sample_data.csv")
MODEL_OUTPUTS_CSV                = os.path.join(_DIR, "model_outputs.csv")
MODEL_OUTPUTS_REVIEWED_CSV       = os.path.join(_DIR, "model_outputs_reviewed.csv")
MODEL_OUTPUT_QUALITY_SUMMARY_CSV = os.path.join(_DIR, "model_output_quality_summary.csv")
CURRENT_MARKET_CSV               = os.path.join(_DIR, "current_market_data.csv")
MODEL_OUTPUTS_COMBINED_CSV       = os.path.join(_DIR, "model_outputs_combined.csv")
MODEL_OUTPUTS_CALIBRATED_CSV     = os.path.join(_DIR, "model_outputs_combined_calibrated.csv")
MODEL_OUTPUTS_RISK_CSV           = os.path.join(_DIR, "model_outputs_combined_calibrated_risk.csv")


# ── Sample data ───────────────────────────────────────────────────────────────

def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def refresh_live_prices(df: pd.DataFrame) -> tuple:
    """
    Attempt to refresh Current_Price from yfinance.
    Returns (updated_df, status_message). Falls back to sample prices on error.
    """
    try:
        import yfinance as yf
        tickers = df["Ticker"].tolist()
        raw     = yf.download(
            tickers, period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True,
            multi_level_index=False,
        )
        if raw.empty:
            return df, "yfinance returned no data — using sample prices."

        close   = raw["Close"] if "Close" in raw.columns else pd.DataFrame()
        updated = 0
        for ticker in tickers:
            if not close.empty:
                col_data = close[ticker] if ticker in close.columns else None
                if col_data is not None:
                    val = col_data.dropna()
                    if len(val) > 0:
                        df.loc[df["Ticker"] == ticker, "Current_Price"] = round(float(val.iloc[-1]), 2)
                        updated += 1

        if updated == 0:
            return df, "Could not parse live prices — using sample prices."
        return df, f"Live prices refreshed for {updated}/{len(tickers)} tickers."

    except ImportError:
        return df, "yfinance not installed — using sample prices."
    except Exception as exc:
        return df, f"Live fetch failed ({exc}) — using sample prices."


def get_screener_data(use_live: bool = False) -> tuple:
    """Primary entry point for sample-data screener. Returns (df, status_message)."""
    df = load_sample_data()
    if use_live:
        return refresh_live_prices(df)
    return df, "Using sample dataset (demo mode)."


# ── XBRL model outputs ────────────────────────────────────────────────────────

def load_model_outputs() -> tuple:
    """
    Load the best available model output file, one row per ticker (most recent year).

    Fallback order:
      1. model_outputs_combined_calibrated.csv  (2010-2024 + calibrated columns)
      2. model_outputs_combined.csv             (2010-2024, no calibrated columns)
      3. model_outputs_reviewed.csv             (legacy 2010-2016 with review flags)
      4. model_outputs.csv                      (base legacy output)

    Returns one row per ticker (most recent year), in screener-compatible column names.
    Passes through calibrated columns (final_signal_calibrated, valuation_bucket_year, etc.)
    when available.

    Returns (df, status_msg). df is None when no file exists.
    """
    _candidates = [
        (MODEL_OUTPUTS_RISK_CSV,       "risk"),
        (MODEL_OUTPUTS_CALIBRATED_CSV, "calibrated"),
        (MODEL_OUTPUTS_COMBINED_CSV,   "combined"),
        (MODEL_OUTPUTS_REVIEWED_CSV,   "reviewed"),
        (MODEL_OUTPUTS_CSV,            "base"),
    ]
    src_path  = None
    src_label = None
    for path, label in _candidates:
        if os.path.exists(path):
            src_path  = path
            src_label = label
            break
    if src_path is None:
        return None, "model_outputs.csv not found — visit Model Diagnostics first or run market_cap_utils.py."

    try:
        raw = pd.read_csv(src_path, dtype={"company_id": str})
        if raw.empty:
            return None, f"{os.path.basename(src_path)} is empty."

        # One row per ticker — most recent year
        raw = raw.sort_values("year").groupby("ticker", as_index=False).last()

        # Quality columns (present in reviewed file; fill neutral values for base file)
        def _col(name, default=""):
            if name in raw.columns:
                return raw[name].values
            return [default] * len(raw)

        df = pd.DataFrame({
            "Ticker":               raw["ticker"].values,
            "Company_Name":         _col("company_name"),
            "Sector":               "Technology",    # placeholder; XBRL lacks sector data
            "year":                 raw["year"].values,
            "Market_Cap_B":         raw["actual_market_cap"].values   / 1e9,
            "Estimated_Fair_Value": raw["estimated_fair_value"].values / 1e9,
            "Current_Price":        np.nan,          # per-share price unavailable at mkt-cap level
            "Valuation_Gap_Pct":    raw["valuation_gap_pct"].values,
            "Quality_Score":        raw["quality_score"].values,
            "Report_Risk_Score":    raw["report_risk_score"].values,
            "Final_Signal":         raw["final_signal"].values,
            # Reviewed columns — final_signal_display overrides Final_Signal for flagged rows
            "final_signal_display": _col("final_signal_display", default=None),
            "output_quality_flag":  _col("output_quality_flag",  default="ok"),
            "output_warning_text":  _col("output_warning_text",  default=""),
            "model_error":          raw["model_error"].values,
            "best_model_used":      _col("best_model_used"),
            "explanation_text":     _col("explanation_text"),
            # Calibrated signal columns (present in combined_calibrated file; None otherwise)
            "final_signal_calibrated":        _col("final_signal_calibrated", default=None),
            "valuation_bucket_year":          _col("valuation_bucket_year",   default=None),
            "valuation_bucket_era":           _col("valuation_bucket_era",    default=None),
            "valuation_gap_percentile_year":  _col("valuation_gap_percentile_year", default=None),
            "valuation_gap_percentile_era":   _col("valuation_gap_percentile_era",  default=None),
            "era":                            _col("era", default=None),
            "calibration_warning_text":       _col("calibration_warning_text", default=""),
            # Filing risk columns (present in calibrated_risk file; None/empty otherwise)
            "report_risk_score_real":                   _col("report_risk_score_real",                   default=None),
            "report_risk_score_original":               _col("report_risk_score_original",               default=None),
            "report_risk_available":                    _col("report_risk_available",                    default=False),
            "risk_score_source":                        _col("risk_score_source",                        default=None),
            "risk_score_components":                    _col("risk_score_components",                    default=""),
            "filing_data_quality_flag":                 _col("filing_data_quality_flag",                 default=None),
            "filing_warning_text":                      _col("filing_warning_text",                      default=""),
            "final_signal_calibrated_risk_adjusted":    _col("final_signal_calibrated_risk_adjusted",    default=None),
            "risk_adjustment_reason":                   _col("risk_adjustment_reason",                   default=""),
        })

        # If final_signal_display is all None (base file), copy from Final_Signal
        if df["final_signal_display"].isna().all():
            df["final_signal_display"] = df["Final_Signal"]

        # Per-share / filing metrics unavailable from XBRL market-cap model
        for col in [
            "PE_Ratio", "PB_Ratio", "EV_EBITDA", "Revenue_B", "EPS_TTM",
            "Revenue_Growth_Pct", "Debt_Equity", "FCF_B", "ROE_Pct",
            "Risk_Word_Count", "Debt_Mentions", "Legal_Mentions", "Cost_Pressure_Mentions",
        ]:
            df[col] = np.nan

        n  = len(df)
        yr = int(raw["year"].max()) if "year" in raw.columns else "?"
        needs_review = (df["output_quality_flag"].str.contains("needs_review", na=False)).sum()
        has_calibrated    = df["final_signal_calibrated"].notna().any()
        has_risk          = df["report_risk_available"].any() if "report_risk_available" in df.columns else False
        if src_label == "risk":
            source_note = " (calibrated + filing risk, 2010–2024)"
        elif src_label == "calibrated":
            source_note = " (calibrated, 2010–2024)"
        elif src_label == "combined":
            source_note = " (combined, 2010–2024)"
        elif src_label == "reviewed":
            source_note = " (with quality review)"
        else:
            source_note = ""
        return df, (
            f"Model Output{source_note}: {n} companies "
            f"(most recent year: {yr})"
            + (f", {needs_review} flagged for review" if needs_review > 0 else "")
            + (" — calibrated signals available" if has_calibrated else "")
            + (" — filing risk scores available" if has_risk else "")
        )

    except Exception as exc:
        return None, f"Error loading model outputs: {exc}"


# ── Combined model outputs (legacy + modern) ─────────────────────────────────

def load_combined_model_outputs() -> tuple:
    """
    Load all-year model outputs (non-deduplicated).

    Fallback order:
      1. model_outputs_combined_calibrated.csv  (all years + calibrated columns)
      2. model_outputs_combined.csv             (all years, no calibrated columns)
      3. model_outputs_reviewed.csv             (legacy 2010-2016)
      4. model_outputs.csv                      (base legacy)

    Returns (df, status_msg, source_label).
    """
    _candidates = [
        (MODEL_OUTPUTS_RISK_CSV,       "calibrated"),   # risk file also contains calibrated columns
        (MODEL_OUTPUTS_CALIBRATED_CSV, "calibrated"),
        (MODEL_OUTPUTS_COMBINED_CSV,   "combined"),
        (MODEL_OUTPUTS_REVIEWED_CSV,   "legacy"),
        (MODEL_OUTPUTS_CSV,            "base"),
    ]
    for path, label in _candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            yrs = sorted(df["year"].dropna().astype(int).unique().tolist())
            label_str = {
                "calibrated": "Calibrated combined",
                "combined":   "Combined",
                "legacy":     "Legacy",
                "base":       "Base",
            }.get(label, label)
            return (
                df,
                f"{label_str} model outputs: {len(df)} rows, years {yrs}",
                label,
            )
        except Exception:
            continue

    return None, "No model output files found.", "missing"


# ── Current market snapshot ───────────────────────────────────────────────────

def fetch_current_market_data(tickers: list, save: bool = True) -> pd.DataFrame:
    """
    Fetch current market snapshot for a list of tickers using yfinance.
    Uses per-ticker history(period='5d') for prices and fast_info for market cap.
    Gracefully skips tickers that fail without crashing.

    Returns a DataFrame saved to current_market_data.csv.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    ts   = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []

    for ticker in tickers:
        row = {
            "ticker":              ticker,
            "current_price":       None,
            "current_market_cap":  None,
            "previous_close":      None,
            "daily_change_pct":    None,
            "volume":              None,
            "shares_outstanding":  None,
            "last_updated":        ts,
            "data_quality_flag":   "error",
        }
        try:
            t_obj = yf.Ticker(ticker)
            hist  = t_obj.history(period="5d")
            if not hist.empty:
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    row["current_price"]    = float(closes.iloc[-1])
                    row["previous_close"]   = float(closes.iloc[-2])
                    row["daily_change_pct"] = (
                        (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
                    )
                elif len(closes) == 1:
                    row["current_price"] = float(closes.iloc[-1])
                if "Volume" in hist.columns:
                    vols = hist["Volume"].dropna()
                    if len(vols) > 0:
                        row["volume"] = float(vols.iloc[-1])

            fi = t_obj.fast_info
            mc = getattr(fi, "market_cap", None)
            if mc and mc > 0:
                row["current_market_cap"] = float(mc)
            sh = getattr(fi, "shares", None)
            if sh and sh > 0:
                row["shares_outstanding"] = float(sh)

            if row["current_price"] is not None:
                row["data_quality_flag"] = "ok"
        except Exception:
            pass

        rows.append(row)
        time.sleep(0.15)   # polite rate limit

    df = pd.DataFrame(rows)
    if save and len(df) > 0:
        df.to_csv(CURRENT_MARKET_CSV, index=False)
    return df


def load_current_market_data() -> tuple:
    """
    Load current_market_data.csv if it exists.
    Returns (df, status_msg). df is None if file is absent.
    """
    if not os.path.exists(CURRENT_MARKET_CSV):
        return None, "current_market_data.csv not found — run market_cap_utils.py to fetch."

    try:
        df = pd.read_csv(CURRENT_MARKET_CSV)
        ok = (df.get("data_quality_flag", pd.Series()) == "ok").sum()
        last = df["last_updated"].max() if "last_updated" in df.columns else "unknown"
        return df, f"Current market data: {ok}/{len(df)} tickers OK (as of {str(last)[:10]})"
    except Exception as exc:
        return None, f"Error loading current_market_data.csv: {exc}"


TICKER_UNIVERSE_CSV    = os.path.join(_DIR, "ticker_universe.csv")
LATEST_10Q_RISK_CSV    = os.path.join(_DIR, "latest_10q_risk_scores.csv")
FILING_RISK_UPDATE_CSV = os.path.join(_DIR, "filing_risk_current_update.csv")


def load_10q_risk_scores() -> tuple:
    """Load latest 10-Q risk scores. Returns (df, status_msg). df is None if file absent."""
    if not os.path.exists(LATEST_10Q_RISK_CSV):
        return None, "latest_10q_risk_scores.csv not found — run: python3 filing_risk_utils.py --10q"
    try:
        df = pd.read_csv(LATEST_10Q_RISK_CSV)
        ok = (df.get("latest_10q_data_quality_flag", pd.Series()) == "ok").sum()
        return df, f"10-Q risk scores: {len(df)} tickers ({ok} ok)"
    except Exception as exc:
        return None, f"Error loading 10-Q risk scores: {exc}"


def load_filing_risk_update() -> tuple:
    """Load filing risk current update (10-K vs 10-Q trend). Returns (df, status_msg)."""
    if not os.path.exists(FILING_RISK_UPDATE_CSV):
        return None, "filing_risk_current_update.csv not found — run: python3 filing_risk_utils.py --10q"
    try:
        df = pd.read_csv(FILING_RISK_UPDATE_CSV)
        trend_counts = df["filing_risk_trend"].value_counts().to_dict() if "filing_risk_trend" in df.columns else {}
        summary = ", ".join(f"{k}: {v}" for k, v in trend_counts.items())
        return df, f"Filing risk update: {len(df)} tickers ({summary})"
    except Exception as exc:
        return None, f"Error loading filing risk update: {exc}"


# ── Ticker universe ───────────────────────────────────────────────────────────

_UNIVERSE_COLS = [
    "ticker", "company_name", "cik", "sector", "industry",
    "active", "universe_group", "data_status", "notes",
]
_EXTRA_UNIVERSE_COLS = [
    "validation_status",
    "company_facts_status",
    "market_cap_status",
    "model_output_status",
    "filing_10k_status",
    "filing_10q_status",
    "current_market_status",
    "last_pipeline_run",
    "failure_reason",
]
ALL_UNIVERSE_COLS = _UNIVERSE_COLS + _EXTRA_UNIVERSE_COLS
_UNIVERSE_GROUPS = ["current_demo", "expanded_tech", "sp500_sample", "custom"]


def _upgrade_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the extra pipeline-status columns exist (empty string default)."""
    for col in _EXTRA_UNIVERSE_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def load_ticker_universe() -> tuple:
    """
    Load ticker_universe.csv.  If missing, bootstrap it from existing model outputs
    (universe_group = 'current_demo', active = True, data_status = 'available').

    Returns (df, status_msg).  df always has ALL_UNIVERSE_COLS; never None.
    """
    if os.path.exists(TICKER_UNIVERSE_CSV):
        try:
            df = pd.read_csv(TICKER_UNIVERSE_CSV, dtype=str).fillna("")
            _needs_upgrade = any(c not in df.columns for c in _EXTRA_UNIVERSE_COLS)
            for col in _UNIVERSE_COLS:
                if col not in df.columns:
                    df[col] = ""
            df = _upgrade_universe_schema(df)
            if _needs_upgrade:
                try:
                    df.to_csv(TICKER_UNIVERSE_CSV, index=False)
                except Exception:
                    pass
            return df, f"Ticker universe: {len(df)} tickers loaded."
        except Exception:
            pass  # fall through to bootstrap

    # Bootstrap from best available model output file
    _cands = [
        MODEL_OUTPUTS_RISK_CSV, MODEL_OUTPUTS_CALIBRATED_CSV,
        MODEL_OUTPUTS_COMBINED_CSV, MODEL_OUTPUTS_REVIEWED_CSV, MODEL_OUTPUTS_CSV,
    ]
    rows = []
    for path in _cands:
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
            if raw.empty or "ticker" not in raw.columns:
                continue
            latest = raw.sort_values("year").groupby("ticker", as_index=False).last()
            for _, r in latest.iterrows():
                rows.append({
                    "ticker":              str(r["ticker"]).upper(),
                    "company_name":        str(r.get("company_name") or ""),
                    "cik":                 str(r.get("company_id") or ""),
                    "sector":              "Technology",
                    "industry":            "",
                    "active":              "True",
                    "universe_group":      "current_demo",
                    "data_status":         "available",
                    "notes":               "bootstrapped from model outputs",
                    "validation_status":   "",
                    "company_facts_status": "available",
                    "market_cap_status":   "available",
                    "model_output_status": "available",
                    "filing_10k_status":   "",
                    "filing_10q_status":   "",
                    "current_market_status": "",
                    "last_pipeline_run":   "",
                    "failure_reason":      "",
                })
            break
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=ALL_UNIVERSE_COLS) if rows else pd.DataFrame(columns=ALL_UNIVERSE_COLS)

    # Persist so subsequent calls are fast
    try:
        df.to_csv(TICKER_UNIVERSE_CSV, index=False)
    except Exception:
        pass

    n = len(df)
    msg = (
        f"Ticker universe bootstrapped from model outputs: {n} tickers (current_demo)."
        if n > 0
        else "No model outputs found — ticker universe is empty."
    )
    return df, msg


def expand_universe_for_tickers(tickers: list) -> tuple:
    """
    Add new tickers to ticker_universe.csv with data_status='pending'.

    Safe:
    - Validates ticker format (1–5 uppercase letters/digits).
    - Skips tickers already in the universe.
    - Looks up CIK from existing model outputs if available.
    - Does NOT fetch SEC data, yfinance, or run any model jobs.

    Returns (added_count, skipped_count, message).
    """
    import re

    if not tickers:
        return 0, 0, "No tickers provided."

    # Load existing
    universe_df, _ = load_ticker_universe()
    existing = set(universe_df["ticker"].str.upper().tolist())

    # Build CIK lookup from model outputs
    _cik_map: dict = {}
    for path in [MODEL_OUTPUTS_RISK_CSV, MODEL_OUTPUTS_CALIBRATED_CSV,
                 MODEL_OUTPUTS_COMBINED_CSV, MODEL_OUTPUTS_REVIEWED_CSV, MODEL_OUTPUTS_CSV]:
        if os.path.exists(path):
            try:
                _raw = pd.read_csv(path, usecols=lambda c: c in ("ticker", "company_id"))
                for _, r in _raw.iterrows():
                    _t = str(r.get("ticker", "")).upper()
                    _c = str(r.get("company_id", ""))
                    if _t and _c and _t not in _cik_map:
                        _cik_map[_t] = _c
                break
            except Exception:
                continue

    _ticker_re = re.compile(r"^[A-Z0-9]{1,5}$")
    new_rows   = []
    skipped    = []

    for raw_t in tickers:
        t = str(raw_t).strip().upper()
        if not _ticker_re.match(t):
            skipped.append(f"{raw_t} (invalid format)")
            continue
        if t in existing:
            skipped.append(f"{t} (already in universe)")
            continue
        new_rows.append({
            "ticker":              t,
            "company_name":        "",
            "cik":                 _cik_map.get(t, ""),
            "sector":              "",
            "industry":            "",
            "active":              "True",
            "universe_group":      "custom",
            "data_status":         "pending",
            "notes":               "added via expand_universe_for_tickers; data not yet generated",
            "validation_status":   "",
            "company_facts_status": "",
            "market_cap_status":   "",
            "model_output_status": "",
            "filing_10k_status":   "",
            "filing_10q_status":   "",
            "current_market_status": "",
            "last_pipeline_run":   "",
            "failure_reason":      "",
        })
        existing.add(t)

    if new_rows:
        combined = pd.concat(
            [universe_df, pd.DataFrame(new_rows, columns=ALL_UNIVERSE_COLS)],
            ignore_index=True,
        )
        try:
            combined.to_csv(TICKER_UNIVERSE_CSV, index=False)
        except Exception as exc:
            return 0, len(skipped), f"Failed to save universe: {exc}"

    added = len(new_rows)
    msg_parts = [f"Added {added} ticker(s) with data_status='pending'."]
    if skipped:
        msg_parts.append(f"Skipped: {', '.join(skipped[:10])}" + (" …" if len(skipped) > 10 else ""))
    msg_parts.append("Run model pipeline to generate outputs before these appear in Screener.")
    return added, len(skipped), "  ".join(msg_parts)
