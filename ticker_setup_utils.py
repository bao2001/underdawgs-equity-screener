"""
ticker_setup_utils.py
Single-ticker pipeline wrappers + universe schema management.

Provides:
  - upgrade_universe_schema()           — adds new status columns to old universe rows
  - update_universe_row()               — write one or more fields for a ticker in-place
  - check_ticker_readiness()            — cross-reference all data files for one ticker
  - promote_ticker_if_ready()           — set data_status=available when model output exists
  - validate_tickers_for_wizard()       — format check + CIK lookup for wizard input
  - prepare_single_ticker_fundamentals() — fetch SEC facts, append to modern_fundamentals.csv
  - prepare_single_ticker_market_cap()   — fetch yfinance market caps, append to CSV
  - prepare_single_ticker_model_output() — retrain and rebuild combined outputs
  - prepare_single_ticker_10k_risk()     — find+extract 10-K risk, append to CSV
  - prepare_single_ticker_10q_risk()     — fetch latest 10-Q risk, append to CSV
  - prepare_single_ticker_current_market() — refresh current market snapshot

All pipeline wrappers return (success: bool, message: str).
None of them crash the app — all errors are caught and reported.
"""

import os
import re
import time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Extended universe schema ───────────────────────────────────────────────────

_BASE_UNIVERSE_COLS = [
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
ALL_UNIVERSE_COLS = _BASE_UNIVERSE_COLS + _EXTRA_UNIVERSE_COLS

_STATUS_VALUES = ("pending", "running", "available", "failed", "skipped", "unavailable", "")


def upgrade_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing extra columns (defaulting to '') to an existing universe DataFrame."""
    for col in _EXTRA_UNIVERSE_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


# ── Universe row I/O ──────────────────────────────────────────────────────────

def _universe_path() -> str:
    from data_utils import TICKER_UNIVERSE_CSV
    return TICKER_UNIVERSE_CSV


def _load_universe_raw() -> pd.DataFrame:
    """Load (and schema-upgrade) the universe CSV; return empty DF on failure."""
    path = _universe_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=ALL_UNIVERSE_COLS)
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        return upgrade_universe_schema(df)
    except Exception:
        return pd.DataFrame(columns=ALL_UNIVERSE_COLS)


def _save_universe(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_universe_path(), index=False)
    except Exception:
        pass


def update_universe_row(ticker: str, **fields) -> None:
    """
    Update one or more fields for ticker in ticker_universe.csv.
    Creates the row if it doesn't exist (with active=True, universe_group=custom).
    Automatically stamps last_pipeline_run when any pipeline status field is written.
    """
    t   = ticker.upper()
    df  = _load_universe_raw()
    idx = df.index[df["ticker"].str.upper() == t].tolist()

    # Stamp last_pipeline_run when a status field is touched
    _pipeline_status_fields = {
        "company_facts_status", "market_cap_status", "model_output_status",
        "filing_10k_status", "filing_10q_status", "current_market_status",
    }
    if any(k in _pipeline_status_fields for k in fields):
        fields.setdefault("last_pipeline_run", pd.Timestamp.now().isoformat(timespec="seconds"))

    if not idx:
        new_row = {c: "" for c in ALL_UNIVERSE_COLS}
        new_row.update({"ticker": t, "active": "True",
                        "universe_group": "custom", "data_status": "pending"})
        new_row.update({k: str(v) for k, v in fields.items()})
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        for k, v in fields.items():
            if k in df.columns:
                df.loc[idx[0], k] = str(v)

    _save_universe(df)


# ── Readiness check ───────────────────────────────────────────────────────────

def check_ticker_readiness(ticker: str) -> dict:
    """
    Cross-reference all pipeline data files to determine what exists for ticker.
    Returns a dict with keys:
      has_model_output, has_market_cap, has_fundamentals,
      has_10k_risk, has_10q_risk, has_current_market,
      ready_for_screener, warnings (list), data_status
    """
    t   = ticker.upper()
    out = {
        "ticker":             t,
        "has_model_output":   False,
        "has_market_cap":     False,
        "has_fundamentals":   False,
        "has_10k_risk":       False,
        "has_10q_risk":       False,
        "has_current_market": False,
        "ready_for_screener": False,
        "warnings":           [],
        "data_status":        "pending",
    }

    # Model outputs (any combined/calibrated file)
    _mo_paths = [
        os.path.join(_HERE, "model_outputs_combined_calibrated_risk.csv"),
        os.path.join(_HERE, "model_outputs_combined_calibrated.csv"),
        os.path.join(_HERE, "model_outputs_combined.csv"),
        os.path.join(_HERE, "modern_model_outputs.csv"),
    ]
    for p in _mo_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, usecols=["ticker"])
                if t in df["ticker"].str.upper().values:
                    out["has_model_output"] = True
                    break
            except Exception:
                pass

    # Modern fundamentals
    _fund_path = os.path.join(_HERE, "modern_fundamentals.csv")
    if os.path.exists(_fund_path):
        try:
            df = pd.read_csv(_fund_path, usecols=["ticker"])
            out["has_fundamentals"] = t in df["ticker"].str.upper().values
        except Exception:
            pass

    # Market cap
    _mc_path = os.path.join(_HERE, "modern_market_cap_data.csv")
    if os.path.exists(_mc_path):
        try:
            df = pd.read_csv(_mc_path, usecols=["ticker", "data_quality_flag"])
            sub = df[df["ticker"].str.upper() == t]
            out["has_market_cap"] = (sub["data_quality_flag"] == "ok").any()
        except Exception:
            pass

    # 10-K risk
    _risk_path = os.path.join(_HERE, "filing_risk_scores.csv")
    if os.path.exists(_risk_path):
        try:
            df = pd.read_csv(_risk_path, usecols=["ticker"])
            out["has_10k_risk"] = t in df["ticker"].str.upper().values
        except Exception:
            pass

    # 10-Q risk
    _q10_path = os.path.join(_HERE, "latest_10q_risk_scores.csv")
    if os.path.exists(_q10_path):
        try:
            df = pd.read_csv(_q10_path)
            tk_col = "ticker" if "ticker" in df.columns else None
            if tk_col:
                out["has_10q_risk"] = t in df[tk_col].str.upper().values
        except Exception:
            pass

    # Current market
    _cm_path = os.path.join(_HERE, "current_market_data.csv")
    if os.path.exists(_cm_path):
        try:
            df = pd.read_csv(_cm_path, usecols=["ticker"])
            out["has_current_market"] = t in df["ticker"].str.upper().values
        except Exception:
            pass

    # Warnings for soft-missing data
    if out["has_model_output"] and not out["has_10q_risk"]:
        out["warnings"].append("Available but latest 10-Q risk missing")
    if out["has_model_output"] and not out["has_current_market"]:
        out["warnings"].append("Available but current market snapshot missing")
    if out["has_model_output"] and not out["has_10k_risk"]:
        out["warnings"].append("Available but filing risk (10-K) incomplete")

    out["ready_for_screener"] = out["has_model_output"]
    out["data_status"] = "available" if out["ready_for_screener"] else "pending"

    return out


def promote_ticker_if_ready(ticker: str) -> tuple:
    """
    Check readiness and promote data_status to 'available' in universe if
    model output exists.  Returns (promoted: bool, message: str).
    """
    r = check_ticker_readiness(ticker)
    if r["ready_for_screener"]:
        update_universe_row(
            ticker,
            data_status="available",
            model_output_status="available",
            failure_reason="",
        )
        warnings = r["warnings"]
        msg = f"{ticker} promoted to available."
        if warnings:
            msg += "  Warnings: " + "; ".join(warnings)
        return True, msg
    return False, f"{ticker} not ready: model output missing."


# ── Ticker validation for wizard ──────────────────────────────────────────────

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}$")


def validate_tickers_for_wizard(raw_input: str) -> list:
    """
    Parse a comma/space/newline-separated string of tickers.
    For each, check format, existing universe membership, and attempt CIK lookup.

    Returns list of dicts:
      ticker, company_name, cik, valid, already_in_universe, validation_status, warning
    """
    # Load CIK mapping
    _cik_map: dict  = {}
    _name_map: dict = {}
    try:
        from modern_fundamentals_utils import load_ticker_cik_mapping
        _cik_df, _ = load_ticker_cik_mapping()
        if _cik_df is not None and not _cik_df.empty:
            for _, r in _cik_df.iterrows():
                tk = str(r["ticker"]).upper()
                _cik_map[tk]  = str(r.get("cik", ""))
                _name_map[tk] = str(r.get("company_name", ""))
    except Exception:
        pass

    # Load existing universe
    _uni_tickers: set = set()
    try:
        _uni_df = _load_universe_raw()
        _uni_tickers = set(_uni_df["ticker"].str.upper().values)
    except Exception:
        pass

    # Parse raw input
    raw_tickers = [
        t.strip().upper()
        for t in re.split(r"[,\s\n;]+", raw_input)
        if t.strip()
    ]
    # Deduplicate preserving order
    seen: set = set()
    unique_tickers = []
    for t in raw_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    results = []
    for t in unique_tickers:
        valid = bool(_TICKER_RE.match(t))
        cik   = _cik_map.get(t, "")
        name  = _name_map.get(t, "")
        already = t in _uni_tickers
        warning = ""
        if not valid:
            status  = "invalid"
            warning = f"'{t}' is not a valid ticker format (1–5 alphanumeric chars)"
        elif already:
            status  = "duplicate"
            warning = "Already in universe"
        elif not cik:
            status  = "no_cik"
            warning = "CIK not found in mapping — will be added as pending (no CIK)"
        else:
            status  = "ok"
        results.append({
            "ticker":             t,
            "company_name":       name,
            "cik":                cik,
            "valid":              valid,
            "already_in_universe": already,
            "validation_status":  status,
            "warning":            warning,
        })
    return results


# ── Single-ticker pipeline wrappers ──────────────────────────────────────────
# Each returns (success: bool, message: str).
# Each appends/merges into existing CSV files; never overwrites other tickers.
# All errors are caught — they never crash the app.

def _read_modify_write(path: str, ticker: str, new_rows: list,
                       merge_keys: list = None) -> None:
    """
    Load CSV at path, drop existing rows for ticker, append new_rows, save.
    merge_keys: if provided, used to identify rows to drop (default: ["ticker"]).
    """
    if merge_keys is None:
        merge_keys = ["ticker"]
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            # Remove old rows for this ticker
            mask = existing[merge_keys[0]].str.upper() == ticker.upper()
            for k in merge_keys[1:]:
                mask = mask & (existing[k].astype(str) == str(ticker))
            existing = existing[~mask]
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def prepare_single_ticker_fundamentals(ticker: str, cik: str) -> tuple:
    """
    Fetch SEC company facts for one ticker and append rows to modern_fundamentals.csv.
    """
    t = ticker.upper()
    try:
        from modern_fundamentals_utils import (
            get_or_fetch_company_facts, extract_annual_fundamentals,
            load_ticker_cik_mapping, MODERN_YEARS, MODERN_FUND_PATH,
        )
        if not cik:
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    cik = str(match.iloc[0]["cik"])
        if not cik:
            return False, f"{t}: no CIK available — cannot fetch SEC facts."

        data, source, msg = get_or_fetch_company_facts(str(cik))
        if data is None:
            return False, f"{t}: SEC facts fetch failed — {msg}"

        # Get company name from mapping or data
        company_name = t
        try:
            from modern_fundamentals_utils import load_ticker_cik_mapping
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    company_name = str(match.iloc[0].get("company_name", t))
        except Exception:
            pass
        if not company_name or company_name == t:
            company_name = data.get("entityName", t)

        rows = extract_annual_fundamentals(cik, t, company_name, data, MODERN_YEARS)
        if not rows:
            return False, f"{t}: no annual rows extracted from SEC facts."

        _read_modify_write(MODERN_FUND_PATH, t, rows)
        ok_n = sum(1 for r in rows if r.get("data_quality_flag") == "ok")
        update_universe_row(t, company_facts_status="available",
                            company_name=company_name)
        return True, f"{t}: {len(rows)} fundamental rows ({ok_n} ok, source={source})."

    except Exception as exc:
        update_universe_row(t, company_facts_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: fundamentals failed — {exc}"


def prepare_single_ticker_market_cap(ticker: str, cik: str = "") -> tuple:
    """
    Fetch yfinance market caps for one ticker and append to modern_market_cap_data.csv.
    Requires modern_fundamentals.csv to have rows for this ticker (for shares).
    """
    t = ticker.upper()
    try:
        from modern_fundamentals_utils import (
            fetch_modern_market_caps, MODERN_MC_PATH, MODERN_YEARS,
            load_ticker_cik_mapping,
        )
        if not cik:
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    cik = str(match.iloc[0]["cik"])

        company_name = t
        try:
            _fund_path = os.path.join(_HERE, "modern_fundamentals.csv")
            if os.path.exists(_fund_path):
                fd = pd.read_csv(_fund_path)
                match = fd[fd["ticker"].str.upper() == t]
                if not match.empty and "company_name" in match.columns:
                    company_name = str(match.iloc[0]["company_name"])
        except Exception:
            pass

        single_row_df = pd.DataFrame([{
            "ticker":       t,
            "company_name": company_name,
            "cik":          cik or "",
        }])

        # Load existing fundamentals for shares lookup
        fund_df = pd.DataFrame()
        try:
            _fund_path = os.path.join(_HERE, "modern_fundamentals.csv")
            if os.path.exists(_fund_path):
                fund_df = pd.read_csv(_fund_path)
        except Exception:
            pass

        result_df = fetch_modern_market_caps(
            single_row_df, fund_df,
            target_years=MODERN_YEARS, verbose=False,
        )
        if result_df.empty:
            update_universe_row(t, market_cap_status="failed",
                                failure_reason="fetch returned no rows")
            return False, f"{t}: market cap fetch returned no rows."

        # Merge into existing CSV
        existing_mc = pd.DataFrame()
        _mc_path = os.path.join(_HERE, "modern_market_cap_data.csv")
        if os.path.exists(_mc_path):
            try:
                existing_mc = pd.read_csv(_mc_path)
                existing_mc = existing_mc[
                    existing_mc["ticker"].str.upper() != t
                ]
            except Exception:
                pass
        combined = pd.concat([existing_mc, result_df], ignore_index=True, sort=False)
        combined.to_csv(_mc_path, index=False)

        ok_n = (result_df["data_quality_flag"] == "ok").sum()
        update_universe_row(t, market_cap_status="available" if ok_n > 0 else "failed")
        return ok_n > 0, f"{t}: {ok_n}/{len(result_df)} market cap rows ok."

    except Exception as exc:
        update_universe_row(t, market_cap_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: market cap failed — {exc}"


def prepare_single_ticker_model_output(ticker: str) -> tuple:
    """
    Regenerate model outputs for all tickers (including this one) by running the
    full modern pipeline: training dataset → model → combined → calibrated.
    This is necessary because the model is trained on all available data together.
    """
    t = ticker.upper()
    try:
        from modern_fundamentals_utils import (
            load_modern_fundamentals, load_modern_market_caps,
            build_modern_training_dataset, generate_modern_model_outputs,
            build_combined_outputs,
        )

        fund_df, fund_msg = load_modern_fundamentals()
        if fund_df is None or fund_df.empty:
            return False, f"{t}: model output skipped — no fundamentals ({fund_msg})"

        mc_df, mc_msg = load_modern_market_caps()
        if mc_df is None or mc_df.empty:
            return False, f"{t}: model output skipped — no market cap data ({mc_msg})"

        # Check this ticker has data
        if t not in fund_df["ticker"].str.upper().values:
            return False, f"{t}: no fundamentals rows — run fundamentals step first."
        if not (mc_df["ticker"].str.upper() == t).any():
            return False, f"{t}: no market cap rows — run market cap step first."

        # Build training dataset (all tickers combined)
        training_df, summary = build_modern_training_dataset(
            fund_df, mc_df, verbose=False
        )
        if training_df is None or training_df.empty:
            return False, f"{t}: training dataset is empty after merge."

        # Load legacy training for combined model
        legacy_df = None
        try:
            from real_model_utils import load_xbrl_data
            _xbrl_df, _ = load_xbrl_data()
            if _xbrl_df is not None and not _xbrl_df.empty:
                from market_cap_utils import build_training_dataset, load_market_cap_data
                _mc_legacy, _ = load_market_cap_data()
                if _mc_legacy is not None:
                    legacy_df, _ = build_training_dataset(_xbrl_df, _mc_legacy)
        except Exception:
            pass

        outputs = generate_modern_model_outputs(
            training_df, legacy_df, verbose=False
        )
        if outputs is None or outputs.empty:
            update_universe_row(t, model_output_status="failed",
                                failure_reason="generate_modern_model_outputs returned empty")
            return False, f"{t}: model output generation returned empty DataFrame."

        # Rebuild combined outputs
        build_combined_outputs(verbose=False)

        # Run calibration if available
        try:
            from calibrate_signals import run_calibration
            run_calibration()
        except Exception:
            pass

        # Check ticker actually has outputs now
        t_in_output = t in outputs["ticker"].str.upper().values
        update_universe_row(t,
                            model_output_status="available" if t_in_output else "failed",
                            failure_reason="" if t_in_output else "ticker absent from outputs after run")
        if t_in_output:
            return True, f"{t}: model output generated ({len(outputs)} rows total)."
        return False, f"{t}: pipeline ran but ticker absent from outputs."

    except Exception as exc:
        update_universe_row(t, model_output_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: model output failed — {exc}"


def prepare_single_ticker_10k_risk(ticker: str, cik: str = "") -> tuple:
    """
    Find 10-K filings and extract risk scores for one ticker.
    Appends to filing_risk_features.csv and filing_risk_scores.csv.
    """
    t = ticker.upper()
    try:
        from filing_risk_utils import (
            build_filing_lookup, extract_all_features, build_risk_scores,
            load_ticker_cik_mapping, RISK_FEATURES_PATH, RISK_SCORES_PATH,
        )

        if not cik:
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    cik = str(match.iloc[0]["cik"])
        if not cik:
            update_universe_row(t, filing_10k_status="unavailable",
                                failure_reason="no CIK for 10-K lookup")
            return False, f"{t}: no CIK — cannot run 10-K risk pipeline."

        # Load calibrated model outputs to get ticker-year pairs
        _cal_path = os.path.join(_HERE, "model_outputs_combined_calibrated.csv")
        _comb_path = os.path.join(_HERE, "model_outputs_combined.csv")
        _mod_path  = os.path.join(_HERE, "modern_model_outputs.csv")
        ticker_years_df = None
        for p in [_cal_path, _comb_path, _mod_path]:
            if os.path.exists(p):
                try:
                    tmp = pd.read_csv(p)
                    sub = tmp[tmp["ticker"].str.upper() == t][["ticker","year"]].drop_duplicates()
                    if not sub.empty:
                        ticker_years_df = sub
                        break
                except Exception:
                    pass

        if ticker_years_df is None or ticker_years_df.empty:
            update_universe_row(t, filing_10k_status="unavailable",
                                failure_reason="no model output rows to derive ticker-year pairs")
            return False, f"{t}: no model output rows — run model output step first."

        # Build CIK df for this ticker
        cik_df = pd.DataFrame([{
            "ticker":       t,
            "company_name": "",
            "cik":          cik,
        }])

        lookup_df = build_filing_lookup(ticker_years_df, cik_df, verbose=False)
        if lookup_df.empty:
            update_universe_row(t, filing_10k_status="unavailable",
                                failure_reason="filing lookup returned no rows")
            return False, f"{t}: filing lookup returned no results."

        features_df = extract_all_features(lookup_df, verbose=False)
        if features_df.empty:
            update_universe_row(t, filing_10k_status="failed",
                                failure_reason="feature extraction returned empty")
            return False, f"{t}: feature extraction returned empty."

        scores_df = build_risk_scores(features_df, verbose=False)

        # Merge both into existing CSVs
        _read_modify_write(RISK_FEATURES_PATH, t, features_df.to_dict("records"))
        if not scores_df.empty:
            _read_modify_write(RISK_SCORES_PATH, t, scores_df.to_dict("records"))

        ok_n = (features_df.get("data_quality_flag", pd.Series()) == "ok").sum()
        update_universe_row(t, filing_10k_status="available" if ok_n > 0 else "failed")
        return ok_n > 0, f"{t}: {ok_n}/{len(features_df)} 10-K year rows ok."

    except Exception as exc:
        update_universe_row(t, filing_10k_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: 10-K risk failed — {exc}"


def prepare_single_ticker_10q_risk(ticker: str, cik: str = "") -> tuple:
    """
    Fetch latest 10-Q for one ticker and append to latest_10q_risk_scores.csv.
    """
    t = ticker.upper()
    try:
        from filing_risk_utils import (
            fetch_company_submissions, find_latest_10q_filing,
            _is_10q_cached, _save_10q_text_cache, _load_10q_text_cache,
            fetch_filing_text, extract_risk_features, compute_risk_scores,
            load_ticker_cik_mapping,
            LATEST_10Q_RISK_PATH, FILING_10Q_TEXT_CACHE_DIR, USER_AGENT,
        )

        if not cik:
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    cik = str(match.iloc[0]["cik"])
        if not cik:
            update_universe_row(t, filing_10q_status="unavailable",
                                failure_reason="no CIK for 10-Q lookup")
            return False, f"{t}: no CIK — cannot run 10-Q pipeline."

        os.makedirs(FILING_10Q_TEXT_CACHE_DIR, exist_ok=True)
        submissions, sub_msg = fetch_company_submissions(cik, force_refresh=False)
        if submissions is None:
            update_universe_row(t, filing_10q_status="failed",
                                failure_reason=f"submissions fetch failed: {sub_msg}")
            return False, f"{t}: submissions fetch failed — {sub_msg}"

        filing_meta = find_latest_10q_filing(submissions)
        if filing_meta is None:
            update_universe_row(t, filing_10q_status="unavailable",
                                failure_reason="no 10-Q found in SEC submissions")
            return False, f"{t}: no 10-Q filing found in SEC submissions."

        acc  = filing_meta.get("accession_number", "")
        pdoc = filing_meta.get("primary_doc", "")
        fdate = filing_meta.get("filing_date", "")

        # Fetch/load text
        if _is_10q_cached(t):
            text = _load_10q_text_cache(t) or ""
        else:
            text, status = fetch_filing_text(cik, t, 0, acc, pdoc, verbose=False)
            if text and len(text) > 200:
                _save_10q_text_cache(t, text)

        if not text or len(text) < 200:
            update_universe_row(t, filing_10q_status="failed",
                                failure_reason="10-Q text fetch/parse failed")
            return False, f"{t}: 10-Q text fetch failed."

        company_name = t
        try:
            _cik_df, _ = load_ticker_cik_mapping()
            if _cik_df is not None:
                match = _cik_df[_cik_df["ticker"].str.upper() == t]
                if not match.empty:
                    company_name = str(match.iloc[0].get("company_name", t))
        except Exception:
            pass

        feat = extract_risk_features(text, t, company_name, cik, 0, "10-Q", fdate)
        scores_df = compute_risk_scores(pd.DataFrame([feat]))

        q10_row = {
            "ticker":                        t,
            "company_name":                  company_name,
            "cik":                           cik,
            "latest_10q_filing_date":        fdate,
            "latest_10q_period":             filing_meta.get("report_date", ""),
            "latest_10q_form_type":          "10-Q",
            "latest_10q_risk_score":         float(scores_df["report_risk_score"].iloc[0])
                                             if not scores_df.empty else None,
            "latest_10q_data_quality_flag":  "ok",
            "latest_10q_warning_text":       "",
        }

        # Merge into existing latest_10q_risk_scores.csv
        existing = pd.DataFrame()
        if os.path.exists(LATEST_10Q_RISK_PATH):
            try:
                existing = pd.read_csv(LATEST_10Q_RISK_PATH)
                existing = existing[existing["ticker"].str.upper() != t]
            except Exception:
                pass
        combined = pd.concat([existing, pd.DataFrame([q10_row])],
                             ignore_index=True, sort=False)
        combined.to_csv(LATEST_10Q_RISK_PATH, index=False)

        update_universe_row(t, filing_10q_status="available")
        return True, f"{t}: 10-Q risk score {q10_row['latest_10q_risk_score']:.1f} (filing {fdate})."

    except Exception as exc:
        update_universe_row(t, filing_10q_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: 10-Q risk failed — {exc}"


def prepare_single_ticker_current_market(ticker: str) -> tuple:
    """Refresh current market snapshot for one ticker."""
    t = ticker.upper()
    try:
        from data_utils import fetch_current_market_data, CURRENT_MARKET_CSV
        result_df = fetch_current_market_data([t], save=False)
        if result_df.empty:
            update_universe_row(t, current_market_status="unavailable",
                                failure_reason="fetch returned no rows")
            return False, f"{t}: market snapshot returned no rows."

        existing = pd.DataFrame()
        if os.path.exists(CURRENT_MARKET_CSV):
            try:
                existing = pd.read_csv(CURRENT_MARKET_CSV)
                existing = existing[existing["ticker"].str.upper() != t]
            except Exception:
                pass
        combined = pd.concat([existing, result_df], ignore_index=True, sort=False)
        combined.to_csv(CURRENT_MARKET_CSV, index=False)

        row = result_df.iloc[0]
        ok  = str(row.get("data_quality_flag", "")) == "ok"
        update_universe_row(t, current_market_status="available" if ok else "failed")
        price = row.get("current_price")
        return ok, (f"{t}: price ${price:.2f}" if ok and price else f"{t}: snapshot status={row.get('data_quality_flag')}")

    except Exception as exc:
        update_universe_row(t, current_market_status="failed",
                            failure_reason=str(exc)[:200])
        return False, f"{t}: market snapshot failed — {exc}"
