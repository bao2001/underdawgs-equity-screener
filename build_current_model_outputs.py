"""
build_current_model_outputs.py
Builds a TRUE current signal layer for Underdawg.

Pipeline:
  historical training data (legacy 2010-2016 + modern 2017-2024)
    -> reuse real_model_utils.train_valuation_model (Ridge / Random Forest on log market cap)
    -> build a LATEST-AVAILABLE feature row per ticker from SEC Company Facts:
         * income-statement metrics: latest TTM (sum of trailing 4 quarters)
         * balance-sheet metrics:    latest reported quarter (point-in-time)
         * YoY growth:               latest TTM vs prior TTM
    -> predict estimated fair value from those latest feature rows
    -> merge current market snapshot (current_market_data.csv)
    -> merge latest 10-Q filing-risk context (filing_risk_current_update.csv, latest_10q_risk_scores.csv)
    -> current_valuation_gap_pct = (estimated_fair_value - current_market_cap) / current_market_cap * 100
    -> assign calibrated + risk-adjusted current signal (mirrors app signal rules)
    -> save model_outputs_current.csv

Honesty rules:
  * Fundamentals are labelled by their actual filing period END DATE (calendar quarter),
    NOT by SEC fy/fp fields (which mislabel off-calendar fiscal years).
  * A ticker is only "current_ttm" if its latest filing period end date is recent.
  * Tickers without usable recent Company Facts fall back to their latest historical
    model row and are explicitly labelled data_freshness = "historical_fallback".
  * Market data can be current; fundamentals update only when filings update.

Run: python3 build_current_model_outputs.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Input paths ────────────────────────────────────────────────────────────────
LEGACY_TRAIN_CSV   = os.path.join(_HERE, "valuation_training_dataset.csv")
MODERN_TRAIN_CSV   = os.path.join(_HERE, "modern_valuation_training_dataset.csv")
MODEL_RISK_CSV     = os.path.join(_HERE, "model_outputs_combined_calibrated_risk.csv")
CURRENT_MKT_CSV    = os.path.join(_HERE, "current_market_data.csv")
RISK_UPDATE_CSV    = os.path.join(_HERE, "filing_risk_current_update.csv")
TEN_Q_SCORES_CSV   = os.path.join(_HERE, "latest_10q_risk_scores.csv")
CIK_MAP_CSV        = os.path.join(_HERE, "ticker_cik_mapping.csv")
SEC_CACHE_DIR      = os.path.join(_HERE, "sec_company_facts_cache")
OUTPUT_CSV         = os.path.join(_HERE, "model_outputs_current.csv")

TODAY = pd.Timestamp.now().normalize()

# ── SEC concept fallback chains (match modern_fundamentals_utils.py) ───────────
_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerNetOfTaxes", "NetRevenues", "TotalRevenues",
]
_NET_INCOME_CONCEPTS   = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
_ASSETS_CONCEPTS       = ["Assets"]
_LIAB_CONCEPTS         = ["Liabilities"]
_EQUITY_CONCEPTS       = ["StockholdersEquity", "StockholdersEquityAttributableToParent",
                          "Equity", "LiabilitiesAndStockholdersEquity"]
_CASH_CONCEPTS         = ["CashAndCashEquivalentsAtCarryingValue",
                          "CashCashEquivalentsAndShortTermInvestments",
                          "CashAndCashEquivalentsPeriodIncreaseDecrease"]
_DEBT_CONCEPTS         = ["LongTermDebt", "LongTermDebtNoncurrent",
                          "LongTermDebtAndCapitalLeaseObligations"]
_OP_INCOME_CONCEPTS    = ["OperatingIncomeLoss"]
_GROSS_PROFIT_CONCEPTS = ["GrossProfit"]
_SHARES_CONCEPTS       = ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"]

# Freshness thresholds (based on latest filing period END DATE vs today)
_CURRENT_CUTOFF = TODAY - pd.DateOffset(months=10)   # end date newer than this -> current
_RECENT_CUTOFF  = TODAY - pd.DateOffset(months=16)   # newer than this -> recent


# ── SEC Company Facts extraction ───────────────────────────────────────────────

def _load_facts(cik: str) -> dict | None:
    p = os.path.join(SEC_CACHE_DIR, f"CIK{str(cik).zfill(10)}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _duration_entries(facts: dict, concepts: list) -> list:
    """Return quarterly (~90-day duration) USD entries for the first matching concept."""
    for ns in ("us-gaap", "ifrs-full"):
        nsf = facts.get("facts", {}).get(ns, {})
        for c in concepts:
            if c not in nsf:
                continue
            units = nsf[c].get("units", {})
            if "USD" not in units:
                continue
            out = []
            for e in units["USD"]:
                if e.get("start") and e.get("end") and e.get("val") is not None:
                    s = pd.to_datetime(e["start"]); en = pd.to_datetime(e["end"])
                    days = (en - s).days
                    if 80 <= days <= 100:       # keep only single-quarter durations
                        out.append({"end": en, "val": float(e["val"]),
                                    "form": str(e.get("form", "")), "filed": e.get("filed", "")})
            if out:
                # de-dupe by end date, keeping the latest-filed amendment
                by_end: dict = {}
                for e in sorted(out, key=lambda x: x["filed"]):
                    by_end[e["end"]] = e
                return sorted(by_end.values(), key=lambda x: x["end"])
    return []


def _instant_latest(facts: dict, concepts: list, as_of: pd.Timestamp | None = None) -> float | None:
    """Return the latest point-in-time (instant) USD value at/before as_of (or overall latest)."""
    for ns in ("us-gaap", "ifrs-full"):
        nsf = facts.get("facts", {}).get(ns, {})
        for c in concepts:
            if c not in nsf:
                continue
            units = nsf[c].get("units", {})
            if "USD" not in units:
                continue
            cands = []
            for e in units["USD"]:
                if e.get("end") and e.get("val") is not None and not e.get("start"):
                    en = pd.to_datetime(e["end"])
                    if as_of is None or en <= as_of + pd.Timedelta(days=5):
                        cands.append((en, e.get("filed", ""), float(e["val"])))
            if cands:
                cands.sort(key=lambda x: (x[0], x[1]))
                return cands[-1][2]
    return None


def _ttm(entries: list, end_before: pd.Timestamp | None = None) -> tuple:
    """Sum the trailing 4 quarterly entries. Returns (ttm_value, latest_end_date)."""
    if end_before is not None:
        entries = [e for e in entries if e["end"] <= end_before]
    if len(entries) < 4:
        return None, None
    last4 = entries[-4:]
    return sum(e["val"] for e in last4), last4[-1]["end"]


def _calendar_period(end_date: pd.Timestamp) -> str:
    """Calendar-quarter label from a period end date, e.g. 2026-Q1."""
    if end_date is None or pd.isna(end_date):
        return "unknown"
    q = (end_date.month - 1) // 3 + 1
    return f"{end_date.year}-Q{q}"


def build_latest_feature_row(cik: str) -> dict | None:
    """
    Build one latest-available feature row from SEC Company Facts.
    Income-statement items = latest TTM; balance-sheet items = latest quarter.
    Returns None if no usable quarterly income data exists.
    """
    facts = _load_facts(cik)
    if facts is None:
        return None

    rev_q = _duration_entries(facts, _REVENUE_CONCEPTS)
    ni_q  = _duration_entries(facts, _NET_INCOME_CONCEPTS)
    oi_q  = _duration_entries(facts, _OP_INCOME_CONCEPTS)
    gp_q  = _duration_entries(facts, _GROSS_PROFIT_CONCEPTS)

    ttm_rev, latest_end = _ttm(rev_q)
    if ttm_rev is None or latest_end is None:
        return None    # no usable trailing-twelve-month revenue -> cannot build current row

    ttm_ni, _ = _ttm(ni_q, end_before=latest_end)
    ttm_oi, _ = _ttm(oi_q, end_before=latest_end)
    ttm_gp, _ = _ttm(gp_q, end_before=latest_end)

    # Prior-year TTM for growth (4 quarters ending ~1 year before latest_end)
    prior_end = latest_end - pd.DateOffset(months=11)
    prior_rev, _ = _ttm([e for e in rev_q if e["end"] <= prior_end])
    prior_ni,  _ = _ttm([e for e in ni_q  if e["end"] <= prior_end])

    # Balance-sheet items: latest instant at/before latest_end
    assets = _instant_latest(facts, _ASSETS_CONCEPTS, latest_end)
    liab   = _instant_latest(facts, _LIAB_CONCEPTS,   latest_end)
    equity = _instant_latest(facts, _EQUITY_CONCEPTS, latest_end)
    cash   = _instant_latest(facts, _CASH_CONCEPTS,   latest_end)
    debt   = _instant_latest(facts, _DEBT_CONCEPTS,   latest_end)

    # Filing form for the latest quarter
    latest_form = next((e["form"] for e in reversed(rev_q) if e["end"] == latest_end), "10-Q")

    return {
        "revenue":            ttm_rev,
        "net_income":         ttm_ni,
        "operating_income":   ttm_oi,
        "gross_profit":       ttm_gp,
        "assets":             assets,
        "liabilities":        liab,
        "equity":             equity,
        "cash":               cash,
        "debt":               debt,
        "_prior_revenue":     prior_rev,
        "_prior_net_income":  prior_ni,
        "_latest_end":        latest_end,
        "latest_fundamentals_period": _calendar_period(latest_end),
        "latest_fundamentals_form":   latest_form or "10-Q",
    }


def _engineer(row: dict) -> dict:
    """Derive the model's engineered features from a raw fundamentals row."""
    def _div(n, d):
        try:
            n = float(n); d = float(d)
            return n / d if abs(d) > 1e-9 else np.nan
        except (TypeError, ValueError):
            return np.nan

    rev = row.get("revenue"); ni = row.get("net_income")
    out = dict(row)
    out["profit_margin"]    = _div(ni, rev)
    out["gross_margin"]     = _div(row.get("gross_profit"), rev)
    out["operating_margin"] = _div(row.get("operating_income"), rev)
    out["debt_to_assets"]   = _div(row.get("debt"), row.get("assets"))
    out["return_on_assets"] = _div(ni, row.get("assets"))
    out["return_on_equity"] = _div(ni, row.get("equity"))
    out["cash_to_assets"]   = _div(row.get("cash"), row.get("assets"))
    out["revenue_growth"]    = _div(rev, row.get("_prior_revenue")) - 1 \
        if row.get("_prior_revenue") else np.nan
    out["net_income_growth"] = _div(ni, row.get("_prior_net_income")) - 1 \
        if row.get("_prior_net_income") else np.nan
    out["log_revenue"] = np.log(rev) if rev and rev > 0 else np.nan
    out["log_assets"]  = np.log(row["assets"]) if row.get("assets") and row["assets"] > 0 else np.nan
    return out


# ── Signal helpers (mirror model_utils / calibrate_signals / filing_risk_utils) ─

def _compute_signal(val_gap, quality, risk) -> str:
    if pd.isna(val_gap) or pd.isna(quality):
        return "Fairly valued / neutral"
    risk_v = float(risk) if pd.notna(risk) else 50.0
    val_gap = float(val_gap); quality = float(quality)
    if val_gap > 15 and quality < 45:
        return "Possible value trap"
    if val_gap > 20 and quality >= 65 and risk_v < 40:
        return "High-priority research candidate"
    if val_gap > 10 and quality >= 50:
        return "Research candidate"
    if -10 <= val_gap <= 10:
        return "Fairly valued / neutral"
    if val_gap < -10:
        return "Potentially overvalued"
    return "Research candidate"


def _assign_bucket(pct, flag="ok") -> str:
    if "needs_review" in str(flag):
        return "Needs review"
    if pd.isna(pct):
        return "Neutral relative value"
    if pct >= 90: return "Top decile relative value"
    if pct >= 70: return "Above-average relative value"
    if pct >= 40: return "Neutral relative value"
    if pct >= 20: return "Below-average relative value"
    return "Low relative value"


def _assign_calibrated(bucket, quality, flag="ok") -> str:
    if "needs_review" in str(flag):
        return "Needs review"
    quality = float(quality) if pd.notna(quality) else 50.0
    if bucket in ("Top decile relative value", "Above-average relative value"):
        return "Possible value trap" if quality < 45 else "Research candidate"
    if bucket == "Neutral relative value":
        return "Fairly valued / neutral"
    return "Potentially overvalued"


def _risk_adjust(calib_sig, risk_score, quality, has_real) -> tuple:
    if not has_real or pd.isna(risk_score):
        return calib_sig, "no_real_filing_risk_data"
    rs = float(risk_score); q = float(quality) if pd.notna(quality) else 50.0
    if rs >= 80 and calib_sig == "Research candidate":
        if q < 45:
            return "Possible value trap", f"Risk-adjusted: very high filing risk ({rs:.0f}/100) + low quality."
        return "Fairly valued / neutral", (
            f"Risk-adjusted: very high filing risk ({rs:.0f}/100). Downgraded from Research candidate.")
    if rs >= 90 and calib_sig == "Fairly valued / neutral":
        return calib_sig, f"Caution: extreme filing risk ({rs:.0f}/100)."
    return calib_sig, ""


def _quality_scores(df: pd.DataFrame) -> pd.Series:
    """0-100 quality score, percentile-ranked across the current cohort (mirrors real_model_utils)."""
    def _pct(s, ascending=True):
        return s.rank(pct=True, ascending=ascending, na_option="keep") * 100
    rev = df["revenue"]; oi = df["operating_income"]
    op_margin = pd.Series(np.where(rev.abs() > 1e6, oi / rev, np.nan), index=df.index)
    roe = df.get("return_on_equity", pd.Series(np.nan, index=df.index))
    rg  = df.get("revenue_growth",   pd.Series(np.nan, index=df.index))
    d2a = df.get("debt_to_assets",   pd.Series(np.nan, index=df.index))
    q = (_pct(roe).fillna(50) * 0.30 + _pct(op_margin).fillna(50) * 0.25
         + _pct(rg).fillna(50) * 0.25 + _pct(d2a, ascending=False).fillna(50) * 0.20)
    return q.clip(0, 100)


# ── Main build ─────────────────────────────────────────────────────────────────

def build(verbose: bool = True) -> pd.DataFrame | None:
    print("=" * 64)
    print("build_current_model_outputs.py — TRUE current signal layer")
    print("=" * 64)

    # 1. Train the fair-value model on historical data ──────────────────────────
    from real_model_utils import train_valuation_model
    legacy = pd.read_csv(LEGACY_TRAIN_CSV) if os.path.exists(LEGACY_TRAIN_CSV) else pd.DataFrame()
    modern = pd.read_csv(MODERN_TRAIN_CSV) if os.path.exists(MODERN_TRAIN_CSV) else pd.DataFrame()
    if modern.empty:
        print("ERROR: modern_valuation_training_dataset.csv required.")
        return None
    common = [c for c in modern.columns if c in legacy.columns] if not legacy.empty else list(modern.columns)
    train_df = pd.concat([legacy[common], modern[common]], ignore_index=True) if not legacy.empty else modern
    print(f"\n  Training rows: {len(train_df)}  "
          f"(years {int(train_df['year'].min())}–{int(train_df['year'].max())})")
    res, fitted = train_valuation_model(train_df)
    if res.get("status") != "success":
        print(f"ERROR: model training failed: {res.get('status')}")
        return None
    best = fitted["best_model"]
    feats = fitted["features"]
    print(f"  Model trained: best={best}  "
          f"(Ridge R²={res['models']['Ridge Regression']['r2']}, "
          f"RF R²={res['models']['Random Forest']['r2']})")

    def _predict(feat_df: pd.DataFrame) -> np.ndarray:
        X = feat_df.reindex(columns=feats).copy()
        for c in feats:
            X[c] = X[c].fillna(fitted["impute_medians"].get(c, 0.0))
        if best == "Ridge Regression":
            return fitted["ridge"].predict(fitted["scaler"].transform(X))
        return fitted["rf"].predict(X)

    # 2. Load supporting data ───────────────────────────────────────────────────
    cik_map = pd.read_csv(CIK_MAP_CSV, dtype=str)
    cik_map["ticker"] = cik_map["ticker"].str.upper().str.strip()
    cik_by_ticker = dict(zip(cik_map["ticker"], cik_map["cik"]))

    mkt = pd.read_csv(CURRENT_MKT_CSV)
    mkt["ticker"] = mkt["ticker"].str.upper().str.strip()
    print(f"  Market snapshot: {len(mkt)} tickers "
          f"(as of {str(mkt['last_updated'].dropna().iloc[0])[:10]})")

    # Historical latest row per ticker (fallback fundamentals + names + needs_review flags)
    hist = pd.read_csv(MODEL_RISK_CSV, dtype={"company_id": str})
    hist["ticker"] = hist["ticker"].str.upper().str.strip()
    hist_latest = (hist.sort_values("year").groupby("ticker", as_index=False).last())
    hist_by_ticker = hist_latest.set_index("ticker").to_dict("index")

    # Historical fundamentals (modern training) for fallback feature rows
    modern["ticker"] = modern["ticker"].str.upper().str.strip()
    modern_latest = modern.sort_values("year").groupby("ticker", as_index=False).last()
    modern_by_ticker = modern_latest.set_index("ticker").to_dict("index")

    # 3. Build latest feature rows per ticker ────────────────────────────────────
    records = []
    n_ttm = n_recent = n_stale = n_fallback = 0

    for t in mkt["ticker"].unique():
        cik = cik_by_ticker.get(t)
        feat_row = build_latest_feature_row(cik) if cik else None

        if feat_row is not None:
            latest_end = feat_row["_latest_end"]
            eng = _engineer(feat_row)
            if latest_end >= _CURRENT_CUTOFF:
                freshness = "current_ttm"; n_ttm += 1
            elif latest_end >= _RECENT_CUTOFF:
                freshness = "recent"; n_recent += 1
            else:
                freshness = "stale_fundamentals"; n_stale += 1
            feature_source = "sec_company_facts_ttm"
            fund_period = feat_row["latest_fundamentals_period"]
            fund_form   = feat_row["latest_fundamentals_form"]
        else:
            # Fallback: latest historical fundamentals from modern training dataset
            mrow = modern_by_ticker.get(t)
            if mrow is None:
                # No historical fundamentals either — skip (cannot build a feature row)
                continue
            eng = _engineer({
                "revenue": mrow.get("revenue"), "net_income": mrow.get("net_income"),
                "operating_income": mrow.get("operating_income"),
                "gross_profit": mrow.get("gross_profit"), "assets": mrow.get("assets"),
                "liabilities": mrow.get("liabilities"), "equity": mrow.get("equity"),
                "cash": mrow.get("cash"), "debt": mrow.get("debt"),
                "_prior_revenue": None, "_prior_net_income": None,
            })
            # keep pre-computed growth from training if present
            eng["revenue_growth"] = mrow.get("revenue_growth", np.nan)
            eng["net_income_growth"] = mrow.get("net_income_growth", np.nan)
            freshness = "historical_fallback"; n_fallback += 1
            feature_source = "historical_model_row"
            fund_period = f"{int(mrow.get('year', 0))} annual"
            fund_form   = "10-K"

        hrow = hist_by_ticker.get(t, {})
        eng["ticker"] = t
        eng["company_name"] = (hrow.get("company_name")
                               or (modern_by_ticker.get(t) or {}).get("company_name") or t)
        eng["data_freshness"] = freshness
        eng["latest_model_feature_source"] = feature_source
        eng["latest_fundamentals_period"]  = fund_period
        eng["latest_fundamentals_form"]    = fund_form
        eng["output_quality_flag"]         = str(hrow.get("output_quality_flag", "ok"))
        eng["company_id"]                  = str(hrow.get("company_id", ""))
        records.append(eng)

    df = pd.DataFrame(records)
    print(f"\n  Feature rows built: {len(df)}")
    print(f"    current_ttm         : {n_ttm}")
    print(f"    recent              : {n_recent}")
    print(f"    stale_fundamentals  : {n_stale}")
    print(f"    historical_fallback : {n_fallback}")

    # 4. Predict estimated fair value ────────────────────────────────────────────
    df["estimated_fair_value"] = np.exp(_predict(df))

    # 5. Merge current market snapshot ───────────────────────────────────────────
    df = df.merge(
        mkt[["ticker", "current_price", "current_market_cap", "daily_change_pct", "last_updated"]],
        on="ticker", how="left",
    )
    df["market_data_as_of"] = pd.to_datetime(df["last_updated"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 6. Current valuation gap ───────────────────────────────────────────────────
    has_mc = df["current_market_cap"].notna() & (df["current_market_cap"] > 0) & df["estimated_fair_value"].notna()
    df["current_valuation_gap_pct"] = np.nan
    df.loc[has_mc, "current_valuation_gap_pct"] = (
        (df.loc[has_mc, "estimated_fair_value"] - df.loc[has_mc, "current_market_cap"])
        / df.loc[has_mc, "current_market_cap"] * 100
    ).round(1)

    # 7. Quality score (percentile across current cohort) ────────────────────────
    df["quality_score"] = _quality_scores(df).round(1)

    # 8. Merge latest 10-Q filing risk context ───────────────────────────────────
    if os.path.exists(RISK_UPDATE_CSV):
        upd = pd.read_csv(RISK_UPDATE_CSV)
        upd["ticker"] = upd["ticker"].str.upper().str.strip()
        df = df.merge(
            upd[["ticker", "latest_10q_risk_score", "latest_10q_filing_date",
                 "filing_risk_delta", "filing_risk_trend", "annual_10k_risk_score"]],
            on="ticker", how="left",
        )
    else:
        for c in ["latest_10q_risk_score", "latest_10q_filing_date", "filing_risk_delta",
                  "filing_risk_trend", "annual_10k_risk_score"]:
            df[c] = np.nan

    if os.path.exists(TEN_Q_SCORES_CSV):
        q10 = pd.read_csv(TEN_Q_SCORES_CSV)
        q10["ticker"] = q10["ticker"].str.upper().str.strip()
        # Convert the 10-Q period end date to a clean calendar-quarter label
        q10["latest_10q_period"] = pd.to_datetime(q10["latest_10q_period"], errors="coerce").map(
            lambda d: _calendar_period(d) if pd.notna(d) else np.nan)
        df = df.merge(
            q10[["ticker", "latest_10q_period", "latest_10q_data_quality_flag"]],
            on="ticker", how="left",
        )
    else:
        df["latest_10q_period"] = np.nan
        df["latest_10q_data_quality_flag"] = np.nan

    # Effective filing-risk score: prefer clean 10-Q, else historical 10-K real score
    df["report_risk_score_real"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("report_risk_score_real"))
    df["report_risk_available"] = df["ticker"].map(
        lambda t: bool(hist_by_ticker.get(t, {}).get("report_risk_available", False)))
    has_10q = df["latest_10q_risk_score"].notna() & (df["latest_10q_data_quality_flag"].fillna("") == "ok")
    df["filing_risk_score"] = np.where(
        has_10q, df["latest_10q_risk_score"],
        df["report_risk_score_real"].where(df["report_risk_score_real"].notna(),
                                            df["annual_10k_risk_score"]))
    df["has_latest_10q_update"] = has_10q

    # 9. Assign current signals ──────────────────────────────────────────────────
    df["report_risk_score"] = df["filing_risk_score"].fillna(50)

    # Base signal (absolute thresholds)
    df["final_signal"] = df.apply(
        lambda r: _compute_signal(r["current_valuation_gap_pct"], r["quality_score"], r["filing_risk_score"]),
        axis=1)

    # Cross-sectional calibration on current gaps
    valid = df["current_valuation_gap_pct"].dropna()
    if len(valid) >= 5:
        df["valuation_gap_percentile_year"] = (df["current_valuation_gap_pct"].rank(pct=True) * 100).round(1)
    else:
        df["valuation_gap_percentile_year"] = np.nan
    df["valuation_bucket_year"] = df.apply(
        lambda r: _assign_bucket(r["valuation_gap_percentile_year"], r["output_quality_flag"]), axis=1)
    df["final_signal_calibrated"] = df.apply(
        lambda r: _assign_calibrated(r["valuation_bucket_year"], r["quality_score"], r["output_quality_flag"]),
        axis=1)

    adj = df.apply(
        lambda r: _risk_adjust(
            r["final_signal_calibrated"], r["filing_risk_score"], r["quality_score"],
            bool(r["report_risk_available"]) or bool(r["has_latest_10q_update"])),
        axis=1, result_type="expand")
    df["final_signal_calibrated_risk_adjusted"] = adj[0]
    df["risk_adjustment_reason"]                 = adj[1]

    # final_signal_display: current risk-adjusted signal, unless needs_review preserved
    df["final_signal_display"] = df.apply(
        lambda r: r["final_signal_calibrated_risk_adjusted"]
        if "needs_review" not in str(r["output_quality_flag"]) else r["final_signal"],
        axis=1)

    # 10. Consistency + metadata columns ─────────────────────────────────────────
    df["actual_market_cap"] = df["current_market_cap"]           # gap basis is current market cap
    df["valuation_gap_pct"] = df["current_valuation_gap_pct"]     # model gap == current gap (fundamentals ARE current)
    df["year"] = pd.to_datetime(df["last_updated"], errors="coerce").dt.year.fillna(TODAY.year).astype(int)
    df["latest_model_year"] = df["latest_fundamentals_period"]
    df["model_error"] = np.nan
    df["best_model_used"] = best
    df["current_signal_as_of"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")

    _FRESH_TEXT = {
        "current_ttm":        "Latest TTM fundamentals + current market data + latest 10-Q risk",
        "recent":             "Recent-quarter fundamentals + current market data + latest 10-Q risk",
        "stale_fundamentals": "Older-quarter fundamentals + current market data + latest 10-Q risk",
        "historical_fallback": "Historical fallback fundamentals + current market data + latest 10-Q risk",
    }
    df["signal_basis"] = df["data_freshness"].map(_FRESH_TEXT)

    _WARN_STALE = ("Current fundamentals could not be built from recent filings for this ticker — "
                   "signal uses the latest historical model fundamentals. Treat as less current.")
    _WARN_OK = ("Signal uses latest available filing fundamentals plus current market data. "
                "It may not reflect filings or events after the latest available filing update.")
    df["warning_text"] = np.where(
        df["data_freshness"].isin(["historical_fallback", "stale_fundamentals"]),
        _WARN_STALE, _WARN_OK)

    # Latest filing period label used by the app banner: the actual fundamentals period
    # driving the signal (calendar quarter), which is the honest "fundamentals through" value.
    df["latest_filing_period"] = df["latest_fundamentals_period"]

    # Pass-through / neutral columns expected by data_utils.load_model_outputs
    df["report_risk_score_original"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("report_risk_score_original", 50))
    df["risk_score_source"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("risk_score_source"))
    df["risk_score_components"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("risk_score_components", ""))
    df["filing_data_quality_flag"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("filing_data_quality_flag"))
    df["filing_warning_text"] = ""
    df["output_warning_text"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("output_warning_text", ""))
    df["calibration_warning_text"] = ""
    df["era"] = "current"
    df["valuation_bucket_era"] = df["valuation_bucket_year"]
    df["valuation_gap_percentile_era"] = df["valuation_gap_percentile_year"]
    df["explanation_text"] = df["ticker"].map(
        lambda t: hist_by_ticker.get(t, {}).get("explanation_text", ""))

    # Market-cap in billions for the model_output source column parity
    df["Market_Cap_B"] = df["current_market_cap"] / 1e9

    # 11. Order + save ───────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}")
    print(f"  Rows: {len(df)}  Columns: {len(df.columns)}")

    # 12. Summary ────────────────────────────────────────────────────────────────
    print("\n── Latest fundamentals period distribution ──")
    print(df["latest_fundamentals_period"].value_counts().sort_index().to_string())

    print("\n── data_freshness distribution ──")
    print(df["data_freshness"].value_counts().to_string())

    print("\n── Current signal distribution (final_signal_display) ──")
    print(df["final_signal_display"].value_counts().to_string())

    print(f"\n  Tickers with current valuation gap: {int(has_mc.sum())}/{len(df)}")
    print(f"  Tickers using 10-Q risk score:      {int(has_10q.sum())}/{len(df)}")

    print("\n" + "=" * 64)
    print("Done.")
    return df


if __name__ == "__main__":
    out = build()
    sys.exit(0 if out is not None else 1)
