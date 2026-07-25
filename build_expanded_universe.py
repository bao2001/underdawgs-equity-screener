#!/usr/bin/env python3
"""
build_expanded_universe.py
Offline batch pipeline — expand the app from ~29 to 100+ companies.

Run OFFLINE before deploying. Do NOT run from inside the Streamlit app.

Usage:
  python3 build_expanded_universe.py                     # all 118 seed tickers
  python3 build_expanded_universe.py --limit 10          # first 10 only (test run)
  python3 build_expanded_universe.py --limit 25          # first 25
  python3 build_expanded_universe.py --resume            # skip tickers already processed
  python3 build_expanded_universe.py --tickers AAPL MSFT NVDA
  python3 build_expanded_universe.py --skip-filings      # skip 10-K risk extraction
  python3 build_expanded_universe.py --skip-10q          # skip 10-Q risk extraction
  python3 build_expanded_universe.py --force             # re-fetch even if cached
  python3 build_expanded_universe.py --dry-run           # print plan, do nothing

The script:
  1. Loads expanded_ticker_seed.csv
  2. Resolves missing CIKs from SEC's company_tickers.json
  3. Fetches SEC company facts + extracts annual fundamentals (per ticker, incremental)
  4. Fetches yfinance market cap history (batch for new tickers, merged safely)
  5. Rebuilds full model outputs (modern training → model → combined → calibrated)
  6. Runs filing risk pipeline (incremental: skips cached ticker-year pairs)
  7. Runs 10-Q risk pipeline (incremental)
  8. Refreshes current market snapshot for all processed tickers
  9. Updates ticker_universe.csv + marks successful tickers as 'available'
 10. Saves expanded_universe_build_report.csv

Safety guarantees:
  - Never wipes existing 29-company data
  - Backs up major CSVs before overwriting
  - One ticker failure never stops the rest
  - Progress saved after every ticker (crash-safe)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── File paths ────────────────────────────────────────────────────────────────

SEED_CSV         = os.path.join(_HERE, "expanded_ticker_seed.csv")
REPORT_CSV       = os.path.join(_HERE, "expanded_universe_build_report.csv")
BACKUP_DIR       = os.path.join(_HERE, "backups")

# Paths from existing helpers (do not redefine, just reference)
from modern_fundamentals_utils import (
    MODERN_FUND_PATH, MODERN_MC_PATH, MODERN_OUTPUTS_PATH,
    COMBINED_OUTPUTS_PATH, TICKER_CIK_MAP_PATH,
    USER_AGENT, MODERN_YEARS,
    load_ticker_cik_mapping,
    get_or_fetch_company_facts,
    extract_annual_fundamentals,
    fetch_modern_market_caps,
    load_modern_fundamentals,
    load_modern_market_caps,
    build_modern_training_dataset,
    generate_modern_model_outputs,
    build_combined_outputs,
)
from calibrate_signals import run_calibration, CALIBRATED_PATH
from filing_risk_utils import (
    run_filing_risk_pipeline,
    run_10q_risk_pipeline,
    RISK_OUTPUTS_PATH,
)
from data_utils import (
    fetch_current_market_data,
    load_ticker_universe,
    TICKER_UNIVERSE_CSV,
    CURRENT_MARKET_CSV,
    ALL_UNIVERSE_COLS,
)
from ticker_setup_utils import (
    upgrade_universe_schema,
    update_universe_row,
)

# ── Report columns ────────────────────────────────────────────────────────────

REPORT_COLS = [
    "ticker", "company_name", "cik",
    "fundamentals_status", "market_cap_status", "model_output_status",
    "filing_10k_status", "filing_10q_status", "current_market_status",
    "final_data_status", "failure_reason",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _banner(msg: str) -> None:
    bar = "=" * 65
    print(f"\n{bar}\n  {msg}\n{bar}")


def _step(msg: str) -> None:
    print(f"\n── {msg} ──")


def _backup(path: str) -> str | None:
    """Copy path → backups/<stem>_YYYYMMDD_HHMMSS.csv. Returns backup path or None."""
    if not os.path.exists(path):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(BACKUP_DIR, f"{stem}_{_ts()}.csv")
    try:
        import shutil
        shutil.copy2(path, dest)
        print(f"  Backup → {os.path.relpath(dest)}")
        return dest
    except Exception as e:
        print(f"  Warning: backup failed for {path}: {e}")
        return None


def _merge_into_csv(csv_path: str, ticker: str,
                    new_df: pd.DataFrame,
                    key: str = "ticker") -> None:
    """Load CSV, drop rows for ticker, append new_df, save."""
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
            existing = existing[existing[key].astype(str).str.upper() != ticker.upper()]
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def _fetch_sec_ticker_cik_map() -> dict:
    """
    Fetch SEC's full ticker→CIK map from company_tickers.json.
    Returns {TICKER: "cik_str"} or {} on failure.
    Respects SEC rate-limit etiquette with a brief delay.
    """
    url = "https://data.sec.gov/submissions/company_tickers.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        result = {}
        for item in data.values():
            t = str(item.get("ticker", "")).upper().strip()
            c = str(item.get("cik_str", "")).strip()
            if t and c:
                result[t] = c
        print(f"  SEC company_tickers.json: {len(result)} entries loaded.")
        time.sleep(0.15)
        return result
    except Exception as e:
        print(f"  Warning: could not fetch company_tickers.json: {e}")
        return {}


def _load_report() -> dict:
    """Load existing build report as {ticker: row_dict}."""
    if not os.path.exists(REPORT_CSV):
        return {}
    try:
        df = pd.read_csv(REPORT_CSV, dtype=str).fillna("")
        return {str(r["ticker"]).upper(): r.to_dict() for _, r in df.iterrows()}
    except Exception:
        return {}


def _save_report(report: dict) -> None:
    rows = list(report.values())
    if not rows:
        return
    df = pd.DataFrame(rows, columns=REPORT_COLS).fillna("")
    df.to_csv(REPORT_CSV, index=False)


def _new_report_entry(ticker: str, company_name: str, cik: str) -> dict:
    return {
        "ticker":               ticker.upper(),
        "company_name":         company_name,
        "cik":                  cik,
        "fundamentals_status":  "pending",
        "market_cap_status":    "pending",
        "model_output_status":  "pending",
        "filing_10k_status":    "pending",
        "filing_10q_status":    "pending",
        "current_market_status":"pending",
        "final_data_status":    "pending",
        "failure_reason":       "",
    }


# ── Phase 1: Fundamentals ─────────────────────────────────────────────────────

def phase1_fundamentals(todo: list, report: dict, force: bool) -> dict:
    """
    Fetch SEC company facts + extract annual fundamentals for each ticker.
    Merges incrementally into modern_fundamentals.csv.
    Returns updated report dict.
    """
    _banner("Phase 1: SEC Company Facts + Annual Fundamentals")
    total = len(todo)

    # Pre-load existing fundamentals for skip-check
    existing_fund_tickers: set = set()
    if os.path.exists(MODERN_FUND_PATH) and not force:
        try:
            _ef = pd.read_csv(MODERN_FUND_PATH, usecols=["ticker", "data_quality_flag"])
            _ok = _ef[_ef["data_quality_flag"] == "ok"]["ticker"].str.upper()
            existing_fund_tickers = set(_ok.unique())
        except Exception:
            pass

    for i, (ticker, cik, company_name) in enumerate(todo, 1):
        entry = report.setdefault(ticker, _new_report_entry(ticker, company_name, cik))
        print(f"\n  [{i:3d}/{total}] {ticker}  CIK={cik or '?'}  ({company_name})")

        # Skip check
        if not force and ticker in existing_fund_tickers:
            print(f"    → already has ok fundamentals rows — skipping")
            entry["fundamentals_status"] = "cached"
            _save_report(report)
            continue

        if not cik:
            print(f"    → no CIK — skipping fundamentals")
            entry["fundamentals_status"] = "no_cik"
            entry["failure_reason"]      = "CIK not found"
            _save_report(report)
            continue

        data, source, msg = get_or_fetch_company_facts(cik, USER_AGENT)
        if data is None:
            print(f"    → SEC facts fetch FAILED: {msg}")
            entry["fundamentals_status"] = "failed"
            entry["failure_reason"]      = f"SEC facts: {msg}"
            _save_report(report)
            continue

        rows = extract_annual_fundamentals(cik, ticker, company_name, data, MODERN_YEARS)
        if not rows:
            print(f"    → no annual rows extracted  [{source}]")
            entry["fundamentals_status"] = "no_data"
            entry["failure_reason"]      = "extract returned 0 rows"
            _save_report(report)
            continue

        new_df = pd.DataFrame(rows)
        _merge_into_csv(MODERN_FUND_PATH, ticker, new_df)

        ok_n = sum(1 for r in rows if r.get("data_quality_flag") == "ok")
        print(f"    → {ok_n}/{len(rows)} ok rows saved  [{source}]")
        entry["fundamentals_status"] = "ok" if ok_n > 0 else "no_ok_rows"
        entry["company_name"]        = company_name
        entry["cik"]                 = cik
        _save_report(report)

        if source == "api":
            time.sleep(0.12)

    return report


# ── Phase 2: Market caps ──────────────────────────────────────────────────────

def phase2_market_caps(todo: list, report: dict, force: bool) -> dict:
    """
    Fetch yfinance market cap history for tickers that have ok fundamentals.
    Merges safely into modern_market_cap_data.csv.
    """
    _banner("Phase 2: Market Cap History (yfinance)")

    # Which tickers to actually process
    need_mc: list = []
    for ticker, cik, company_name in todo:
        e = report.get(ticker, {})
        fund_ok = e.get("fundamentals_status") in ("ok", "cached")
        if not fund_ok:
            print(f"  {ticker}: skip market caps (no ok fundamentals)")
            e["market_cap_status"] = "skipped_no_fundamentals"
            continue
        if not force and e.get("market_cap_status") == "ok":
            print(f"  {ticker}: already has ok market cap rows — skipping")
            continue
        need_mc.append({"ticker": ticker, "company_name": company_name, "cik": cik or ""})

    if not need_mc:
        print("  Nothing new to process for market caps.")
        return report

    # Backup existing MC CSV before batch call
    _backup(MODERN_MC_PATH)
    existing_mc_backup = None
    if os.path.exists(MODERN_MC_PATH):
        try:
            existing_mc_backup = pd.read_csv(MODERN_MC_PATH).copy()
        except Exception:
            pass

    # Load full fundamentals for shares lookup
    all_fund_df, fund_msg = load_modern_fundamentals()
    fund_df_arg = all_fund_df if isinstance(all_fund_df, pd.DataFrame) else pd.DataFrame()
    if fund_df_arg.empty:
        print(f"  Warning: no fundamentals available for shares lookup — "
              f"market caps will use yfinance shares only ({fund_msg})")

    new_cik_df = pd.DataFrame(need_mc)
    print(f"  Fetching market cap data for {len(new_cik_df)} tickers…")
    new_mc_df = fetch_modern_market_caps(
        new_cik_df, fund_df_arg,
        target_years=MODERN_YEARS, verbose=True,
    )

    # Merge with backup so existing ticker rows are preserved
    if existing_mc_backup is not None and not existing_mc_backup.empty and not new_mc_df.empty:
        new_set = set(new_mc_df["ticker"].str.upper())
        kept    = existing_mc_backup[~existing_mc_backup["ticker"].str.upper().isin(new_set)]
        merged  = pd.concat([kept, new_mc_df], ignore_index=True, sort=False)
        merged.to_csv(MODERN_MC_PATH, index=False)
        print(f"  Merged: {len(kept)} existing + {len(new_mc_df)} new = {len(merged)} rows total")
    elif not new_mc_df.empty:
        # No existing backup — new_mc_df already saved by fetch_modern_market_caps
        pass

    # Update report per-ticker
    if not new_mc_df.empty:
        for t_row in need_mc:
            t = t_row["ticker"]
            t_mc = new_mc_df[new_mc_df["ticker"].str.upper() == t]
            ok_n = (t_mc["data_quality_flag"] == "ok").sum() if not t_mc.empty else 0
            report[t]["market_cap_status"] = "ok" if ok_n > 0 else "no_ok_rows"

    _save_report(report)
    return report


# ── Phase 3: Model rebuild ────────────────────────────────────────────────────

def phase3_model_rebuild(report: dict) -> dict:
    """
    Rebuild full model pipeline on all available data.
    This always processes all tickers together — not per-ticker.
    Steps: training dataset → model outputs → combined → calibrated.
    """
    _banner("Phase 3: Model Rebuild (full dataset)")

    fund_df, fund_msg = load_modern_fundamentals()
    if fund_df is None or fund_df.empty:
        print(f"  ERROR: No fundamentals available — {fund_msg}")
        return report
    print(f"  Fundamentals: {fund_msg}")

    mc_df, mc_msg = load_modern_market_caps()
    if mc_df is None or mc_df.empty:
        print(f"  ERROR: No market cap data — {mc_msg}")
        return report
    print(f"  Market caps: {mc_msg}")

    # Build training dataset
    _step("Building modern training dataset")
    training_df, summary = build_modern_training_dataset(fund_df, mc_df, verbose=True)
    if training_df is None or training_df.empty:
        print("  ERROR: Training dataset is empty.")
        return report

    # Load legacy training for combined model
    legacy_df = None
    try:
        from real_model_utils import load_xbrl_data
        from market_cap_utils import build_training_dataset, load_market_cap_data
        _xbrl, _ = load_xbrl_data()
        _mc_leg, _ = load_market_cap_data()
        if _xbrl is not None and _mc_leg is not None:
            legacy_df, _ = build_training_dataset(_xbrl, _mc_leg)
            if legacy_df is not None and not legacy_df.empty:
                print(f"  Legacy training data: {len(legacy_df)} rows included")
    except Exception as e:
        print(f"  Note: legacy training data not available: {e}")

    # Backup before overwrite
    _backup(MODERN_OUTPUTS_PATH)
    _backup(COMBINED_OUTPUTS_PATH)
    _backup(CALIBRATED_PATH)

    _step("Generating model outputs")
    outputs = generate_modern_model_outputs(training_df, legacy_df, verbose=True)
    if outputs is None or outputs.empty:
        print("  ERROR: generate_modern_model_outputs returned empty.")
        return report

    _step("Building combined outputs")
    combined_df, combined_msg = build_combined_outputs(verbose=True)
    print(f"  {combined_msg}")

    _step("Running calibration")
    run_calibration()

    # Update report for tickers now in outputs
    if outputs is not None and not outputs.empty:
        out_tickers = set(outputs["ticker"].str.upper())
        for t, entry in report.items():
            entry["model_output_status"] = "ok" if t in out_tickers else "not_in_outputs"

    _save_report(report)
    return report


# ── Phase 4: Filing risk (10-K) ───────────────────────────────────────────────

def phase4_filing_risk(report: dict) -> dict:
    """Run incremental filing risk pipeline. Skips cached ticker-year pairs."""
    _banner("Phase 4: Filing Risk (10-K) — Incremental")

    if not os.path.exists(CALIBRATED_PATH):
        print(f"  ERROR: {CALIBRATED_PATH} not found. Run Phase 3 first.")
        for entry in report.values():
            if entry.get("model_output_status") == "ok":
                entry["filing_10k_status"] = "skipped_no_calibrated"
        _save_report(report)
        return report

    _backup(RISK_OUTPUTS_PATH)
    run_filing_risk_pipeline(verbose=True, run_10q=False)

    # Check which tickers now have risk scores
    risk_tickers: set = set()
    from filing_risk_utils import RISK_SCORES_PATH
    if os.path.exists(RISK_SCORES_PATH):
        try:
            rs = pd.read_csv(RISK_SCORES_PATH, usecols=["ticker"])
            risk_tickers = set(rs["ticker"].str.upper())
        except Exception:
            pass

    for t, entry in report.items():
        if entry.get("model_output_status") == "ok":
            entry["filing_10k_status"] = "ok" if t in risk_tickers else "no_data"

    _save_report(report)
    return report


# ── Phase 5: 10-Q risk ─────────────────────────────────────────────��──────────

def phase5_10q_risk(report: dict) -> dict:
    """Run incremental 10-Q risk pipeline."""
    _banner("Phase 5: Latest 10-Q Risk — Incremental")

    run_10q_risk_pipeline(verbose=True)

    from filing_risk_utils import LATEST_10Q_RISK_PATH
    q10_tickers: set = set()
    if os.path.exists(LATEST_10Q_RISK_PATH):
        try:
            q = pd.read_csv(LATEST_10Q_RISK_PATH, usecols=["ticker"])
            q10_tickers = set(q["ticker"].str.upper())
        except Exception:
            pass

    for t, entry in report.items():
        entry["filing_10q_status"] = "ok" if t in q10_tickers else "no_data"

    _save_report(report)
    return report


# ── Phase 6: Current market snapshot ─────────────────────────────────────────

def phase6_current_market(todo: list, report: dict, force: bool) -> dict:
    """Refresh current market data for processed tickers."""
    _banner("Phase 6: Current Market Snapshot")

    # Only refresh tickers with at least ok fundamentals
    candidates = [
        t for t, cik, name in todo
        if report.get(t, {}).get("fundamentals_status") in ("ok", "cached")
    ]
    if not candidates:
        print("  No tickers with ok fundamentals — skipping.")
        return report

    # Check which already have fresh market data (unless force)
    existing_cm_tickers: set = set()
    if os.path.exists(CURRENT_MARKET_CSV) and not force:
        try:
            cm = pd.read_csv(CURRENT_MARKET_CSV, usecols=["ticker"])
            existing_cm_tickers = set(cm["ticker"].str.upper())
        except Exception:
            pass

    need_refresh = [t for t in candidates if force or t not in existing_cm_tickers]
    if not need_refresh:
        print(f"  All {len(candidates)} tickers already have market data — skipping.")
        for t in candidates:
            report[t]["current_market_status"] = "cached"
        _save_report(report)
        return report

    print(f"  Refreshing {len(need_refresh)} tickers…")

    # Load existing CM data for merge
    existing_cm_df = pd.DataFrame()
    if os.path.exists(CURRENT_MARKET_CSV):
        try:
            existing_cm_df = pd.read_csv(CURRENT_MARKET_CSV)
        except Exception:
            pass

    # Batch fetch
    new_cm_df = fetch_current_market_data(need_refresh, save=False)
    if not new_cm_df.empty:
        new_set = set(new_cm_df["ticker"].str.upper())
        if not existing_cm_df.empty:
            kept = existing_cm_df[~existing_cm_df["ticker"].str.upper().isin(new_set)]
            merged_cm = pd.concat([kept, new_cm_df], ignore_index=True, sort=False)
        else:
            merged_cm = new_cm_df
        merged_cm.to_csv(CURRENT_MARKET_CSV, index=False)
        print(f"  Saved {len(merged_cm)} total rows → current_market_data.csv")

        for t in need_refresh:
            t_row = new_cm_df[new_cm_df["ticker"].str.upper() == t]
            ok = not t_row.empty and (t_row["data_quality_flag"] == "ok").any()
            report[t]["current_market_status"] = "ok" if ok else "fetch_failed"

    _save_report(report)
    return report


# ── Phase 7: Universe update ──────────────────────────────────────────────────

def phase7_universe_update(todo: list, report: dict) -> dict:
    """
    Update ticker_universe.csv:
    - Ensure all seed tickers are present
    - Set data_status=available for tickers with model outputs
    - Set data_status=failed for tickers with failure_reason
    """
    _banner("Phase 7: Ticker Universe Update")

    uni_df, _ = load_ticker_universe()

    # Seed CSV data for lookup
    seed_df = pd.read_csv(SEED_CSV, dtype=str).fillna("")
    seed_map = {str(r["ticker"]).upper(): r.to_dict() for _, r in seed_df.iterrows()}

    rows_to_add = []
    for ticker, cik, company_name in todo:
        t = ticker.upper()
        entry = report.get(t, {})
        has_model  = entry.get("model_output_status") == "ok"
        has_failed = bool(entry.get("failure_reason", "").strip())
        data_status = "available" if has_model else ("failed" if has_failed else "pending")

        seed_row = seed_map.get(t, {})

        # Check if already in universe
        if not uni_df.empty and t in uni_df["ticker"].str.upper().values:
            idx = uni_df.index[uni_df["ticker"].str.upper() == t][0]
            uni_df.loc[idx, "data_status"]          = data_status
            uni_df.loc[idx, "model_output_status"]  = entry.get("model_output_status", "")
            uni_df.loc[idx, "company_facts_status"] = "available" if entry.get("fundamentals_status") in ("ok","cached") else ""
            uni_df.loc[idx, "market_cap_status"]    = entry.get("market_cap_status", "")
            uni_df.loc[idx, "filing_10k_status"]    = entry.get("filing_10k_status", "")
            uni_df.loc[idx, "filing_10q_status"]    = entry.get("filing_10q_status", "")
            uni_df.loc[idx, "current_market_status"]= entry.get("current_market_status", "")
            uni_df.loc[idx, "failure_reason"]       = entry.get("failure_reason", "")[:200]
            if cik:
                uni_df.loc[idx, "cik"] = cik
        else:
            new_row = {c: "" for c in ALL_UNIVERSE_COLS}
            new_row.update({
                "ticker":               t,
                "company_name":         company_name or seed_row.get("company_name", ""),
                "cik":                  cik or seed_row.get("cik", ""),
                "sector":               seed_row.get("sector", ""),
                "industry":             "",
                "active":               "True",
                "universe_group":       seed_row.get("universe_group", "expanded_100"),
                "data_status":          data_status,
                "notes":                seed_row.get("notes", ""),
                "model_output_status":  entry.get("model_output_status", ""),
                "company_facts_status": "available" if entry.get("fundamentals_status") in ("ok","cached") else "",
                "market_cap_status":    entry.get("market_cap_status", ""),
                "filing_10k_status":    entry.get("filing_10k_status", ""),
                "filing_10q_status":    entry.get("filing_10q_status", ""),
                "current_market_status":entry.get("current_market_status", ""),
                "failure_reason":       entry.get("failure_reason", "")[:200],
            })
            rows_to_add.append(new_row)

    if rows_to_add:
        new_rows_df = pd.DataFrame(rows_to_add, columns=ALL_UNIVERSE_COLS)
        uni_df = pd.concat([uni_df, new_rows_df], ignore_index=True, sort=False)

    uni_df.to_csv(TICKER_UNIVERSE_CSV, index=False)

    n_avail = (uni_df["data_status"] == "available").sum()
    n_pend  = (uni_df["data_status"] == "pending").sum()
    n_fail  = (uni_df["data_status"] == "failed").sum()
    print(f"  Universe saved: {len(uni_df)} total  "
          f"({n_avail} available, {n_pend} pending, {n_fail} failed)")

    # Update final_data_status in report
    for t, entry in report.items():
        has_model = entry.get("model_output_status") == "ok"
        has_fail  = bool(entry.get("failure_reason", "").strip())
        entry["final_data_status"] = "available" if has_model else ("failed" if has_fail else "pending")

    _save_report(report)
    return report


# ── CIK resolution ────────────────────────────────────────────────────��───────

def resolve_ciks(seed_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Fill in missing CIKs from SEC company_tickers.json and/or existing ticker_cik_mapping.csv.
    Updates ticker_cik_mapping.csv with any new CIK entries found.
    Returns updated seed_df.
    """
    _step("Resolving CIKs")

    # Load existing CIK mapping
    existing_cik_map: dict = {}
    cik_map_df, _ = load_ticker_cik_mapping()
    if cik_map_df is not None:
        for _, r in cik_map_df.iterrows():
            existing_cik_map[str(r["ticker"]).upper()] = str(r["cik"]).strip()

    # Find tickers with blank CIKs in seed
    need_lookup = seed_df[seed_df["cik"].fillna("").str.strip() == ""]["ticker"].str.upper().tolist()
    # Also tickers whose seed CIK is not in existing mapping (to double-check)
    sec_map: dict = {}
    if need_lookup or verbose:
        if need_lookup:
            print(f"  {len(need_lookup)} tickers missing CIK — fetching from SEC…")
        sec_map = _fetch_sec_ticker_cik_map()

    # Resolve
    new_cik_entries = []
    for idx, row in seed_df.iterrows():
        t   = str(row["ticker"]).upper()
        cik = str(row.get("cik", "")).strip()
        if cik:
            # Trust the seed CIK; also add to mapping if not present
            if t not in existing_cik_map:
                existing_cik_map[t] = cik
                new_cik_entries.append({
                    "ticker": t,
                    "company_name": str(row.get("company_name", "")),
                    "cik": cik,
                    "source": "expanded_ticker_seed",
                    "mapping_quality_flag": "ok",
                })
        elif t in existing_cik_map:
            cik = existing_cik_map[t]
            seed_df.at[idx, "cik"] = cik
            print(f"  {t}: CIK from existing mapping → {cik}")
        elif t in sec_map:
            cik = sec_map[t]
            seed_df.at[idx, "cik"] = cik
            existing_cik_map[t] = cik
            new_cik_entries.append({
                "ticker": t,
                "company_name": str(row.get("company_name", "")),
                "cik": cik,
                "source": "sec_company_tickers_json",
                "mapping_quality_flag": "ok",
            })
            print(f"  {t}: CIK from SEC → {cik}")
        else:
            print(f"  {t}: CIK not found — will be skipped in SEC fetch phases")

    # Append new CIK entries to ticker_cik_mapping.csv
    if new_cik_entries:
        existing_map_df = cik_map_df if cik_map_df is not None else pd.DataFrame()
        new_map_df = pd.DataFrame(new_cik_entries)
        if not existing_map_df.empty:
            combined = pd.concat([existing_map_df, new_map_df], ignore_index=True)
            # Deduplicate: keep first occurrence per ticker
            combined = combined.drop_duplicates(subset="ticker", keep="first")
        else:
            combined = new_map_df
        combined.to_csv(TICKER_CIK_MAP_PATH, index=False)
        print(f"  Updated ticker_cik_mapping.csv with {len(new_cik_entries)} new entries")

    return seed_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build expanded 100+ company universe for Underdawg."
    )
    parser.add_argument("--limit",         type=int,   default=None,
                        help="Process only first N seed tickers")
    parser.add_argument("--tickers",       nargs="+",  default=None,
                        help="Process only these tickers (space-separated)")
    parser.add_argument("--resume",        action="store_true",
                        help="Skip tickers that already have ok fundamentals")
    parser.add_argument("--force",         action="store_true",
                        help="Re-fetch even if data is cached")
    parser.add_argument("--skip-filings",  action="store_true",
                        help="Skip 10-K filing risk extraction")
    parser.add_argument("--skip-10q",      action="store_true",
                        help="Skip 10-Q risk extraction")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print plan without running pipeline")
    args = parser.parse_args()

    _banner("Underdawg — Expanded Universe Build Pipeline")
    print(f"  Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  SEC User-Agent: {USER_AGENT}")

    # ── Load seed ─────────────────────────────────────────────────────────────
    if not os.path.exists(SEED_CSV):
        print(f"  ERROR: {SEED_CSV} not found.")
        sys.exit(1)
    seed_df = pd.read_csv(SEED_CSV, dtype=str).fillna("")
    seed_df["ticker"] = seed_df["ticker"].str.upper().str.strip()
    print(f"  Seed file: {len(seed_df)} tickers")

    # ── Resolve CIKs ─────────────────────────────────────────────────────────
    seed_df = resolve_ciks(seed_df, verbose=True)

    # ── Determine which tickers to process ���──────────────────────────────────
    if args.tickers:
        target_set = {t.upper().strip() for t in args.tickers}
        seed_df = seed_df[seed_df["ticker"].isin(target_set)].reset_index(drop=True)
        print(f"  --tickers filter: {len(seed_df)} tickers selected")

    if args.limit:
        seed_df = seed_df.head(args.limit).reset_index(drop=True)
        print(f"  --limit {args.limit}: processing first {len(seed_df)} tickers")

    # Build todo list: [(ticker, cik, company_name), ...]
    todo = [
        (
            str(r["ticker"]).upper(),
            str(r.get("cik", "")).strip(),
            str(r.get("company_name", r["ticker"])).strip(),
        )
        for _, r in seed_df.iterrows()
    ]

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        _banner("DRY RUN — plan only, no execution")
        for i, (t, cik, name) in enumerate(todo, 1):
            print(f"  {i:3d}. {t:8s}  CIK={cik or '?':12s}  {name}")
        print(f"\n  {len(todo)} tickers would be processed.")
        print(f"  Phases: fundamentals | market_caps | model_rebuild "
              f"| {'filing_risk' if not args.skip_filings else '(skip filings)'} "
              f"| {'10q' if not args.skip_10q else '(skip 10q)'} "
              f"| current_market | universe_update")
        sys.exit(0)

    # ── Load existing report ──────────────────────────────────────────────────
    report = _load_report()

    # --resume: remove tickers with ok fundamentals from Phase 1
    if args.resume:
        existing_fund_tickers: set = set()
        if os.path.exists(MODERN_FUND_PATH):
            try:
                _ef = pd.read_csv(MODERN_FUND_PATH,
                                  usecols=["ticker", "data_quality_flag"])
                _ok = _ef[_ef["data_quality_flag"] == "ok"]["ticker"].str.upper()
                existing_fund_tickers = set(_ok.unique())
            except Exception:
                pass
        skip_n = sum(1 for t, _, _ in todo if t in existing_fund_tickers)
        if skip_n:
            print(f"  --resume: {skip_n}/{len(todo)} tickers already have ok fundamentals "
                  f"(will skip Phase 1 for those)")

    os.makedirs(BACKUP_DIR, exist_ok=True)

    # ── Run phases ────────────────────────────────────────────────────────────
    report = phase1_fundamentals(todo, report, force=args.force)
    report = phase2_market_caps(todo, report, force=args.force)
    report = phase3_model_rebuild(report)

    if not args.skip_filings:
        report = phase4_filing_risk(report)
    else:
        print("\n  (Skipping Phase 4: --skip-filings)")
        for entry in report.values():
            if entry.get("model_output_status") == "ok":
                entry["filing_10k_status"] = "skipped"

    if not args.skip_10q:
        report = phase5_10q_risk(report)
    else:
        print("\n  (Skipping Phase 5: --skip-10q)")
        for entry in report.values():
            entry["filing_10q_status"] = "skipped"

    report = phase6_current_market(todo, report, force=args.force)
    report = phase7_universe_update(todo, report)

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("Build Complete")
    n_ok   = sum(1 for e in report.values() if e.get("final_data_status") == "available")
    n_pend = sum(1 for e in report.values() if e.get("final_data_status") == "pending")
    n_fail = sum(1 for e in report.values() if e.get("final_data_status") == "failed")
    print(f"  Processed:   {len(report)} tickers")
    print(f"  Available:   {n_ok}  (will appear in Screener)")
    print(f"  Pending:     {n_pend}  (fundamentals succeeded, model not yet rebuilt)")
    print(f"  Failed:      {n_fail}")
    print(f"  Report:      {os.path.relpath(REPORT_CSV)}")
    print(f"  Finished:    {datetime.now().isoformat(timespec='seconds')}")

    if n_fail > 0:
        print("\n  Failed tickers:")
        for t, e in sorted(report.items()):
            if e.get("final_data_status") == "failed":
                print(f"    {t:8s}  {e.get('failure_reason','')}")

    print("\n  Next steps:")
    print("   1. Commit updated CSVs (including model outputs, risk scores, ticker_universe.csv).")
    print("   2. Restart the Streamlit app — all available tickers appear in Screener.")
    print("   3. For failed tickers: check failure_reason in expanded_universe_build_report.csv.")
    print("   4. Re-run with --resume --tickers <failed_ticker> to retry a specific ticker.")


if __name__ == "__main__":
    main()
