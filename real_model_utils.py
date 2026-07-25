"""
real_model_utils.py
-------------------
Loads the XBRL financial-statement dataset, engineers features, and trains a
proxy model for fundamental company size.

PROXY MODEL NOTICE
------------------
No market-capitalization data (MarketCap, MarketCapitalization, EntityPublicFloat)
is present in this dataset. The model target is log(Total Assets), which is a
"fundamental size proxy" — it captures company scale from balance-sheet data, not
market-implied fair value. All outputs are diagnostic only and must not be
presented as market-value or fair-value estimates.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
XBRL_DIR = os.path.join(_HERE, "XBRL")

# ── Indicator list ────────────────────────────────────────────────────────────
# Restricted to a focused set; we do NOT iterate over all 8 500+ indicators.

REQUESTED_INDICATORS = [
    "Assets",
    "Revenues",
    "SalesRevenueNet",          # fallback revenue when Revenues is absent
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "GrossProfit",
    "CostOfRevenue",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
    "LongTermDebtNoncurrent",   # fallback debt when LongTermDebt is absent
    "CommonStockSharesOutstanding",
]

YEAR_COLS = ["2010", "2011", "2012", "2013", "2014", "2015", "2016"]

# Indicators we know are absent (confirmed by pre-scan; listed for transparency)
KNOWN_MISSING = ["EntityPublicFloat", "MarketCapitalization", "MarketCap"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_xbrl_data() -> tuple:
    """
    Load, filter, and reshape XBRL data into a company-year feature table.

    Returns
    -------
    df : pd.DataFrame or None
    diag : dict  — data coverage statistics for the diagnostics page
    """
    diag = {
        "status":               "not_loaded",
        "error":                None,
        "companies_total":      0,
        "companies_with_data":  0,
        "company_year_obs":     0,
        "years_covered":        YEAR_COLS,
        "indicators_requested": REQUESTED_INDICATORS,
        "indicators_found":     [],
        "indicators_missing":   [],
        "market_cap_available": False,
        "target_variable":      "log_assets",
        "target_label":         "log(Total Assets) — Fundamental Size Proxy",
        "target_note": (
            "No market-cap data exists in this dataset. "
            "log(Total Assets) is used as a balance-sheet size proxy only."
        ),
    }

    comp_path = os.path.join(XBRL_DIR, "companies.csv")
    ind_path  = os.path.join(XBRL_DIR, "indicators_by_company.csv")

    if not os.path.exists(comp_path) or not os.path.exists(ind_path):
        diag["status"] = "files_not_found"
        diag["error"]  = (
            f"Dataset files not found in {XBRL_DIR}. "
            "Place companies.csv and indicators_by_company.csv there and restart."
        )
        return None, diag

    try:
        # ── Companies ─────────────────────────────────────────────────────────
        companies = pd.read_csv(comp_path, usecols=["company_id", "name_latest"],
                                dtype={"company_id": str})
        diag["companies_total"] = len(companies)

        # ── Indicators — filter FIRST to keep memory and runtime low ──────────
        ind_raw = pd.read_csv(ind_path, dtype={"company_id": str})

        all_indicators = set(ind_raw["indicator_id"].unique())
        found   = [i for i in REQUESTED_INDICATORS if i in all_indicators]
        missing = [i for i in REQUESTED_INDICATORS if i not in all_indicators]
        diag["indicators_found"]   = found
        diag["indicators_missing"] = missing + KNOWN_MISSING   # always show known-absent ones

        subset = ind_raw[ind_raw["indicator_id"].isin(found)].copy()
        del ind_raw                                             # free ~200 MB

        # ── Melt year columns → long format ───────────────────────────────────
        long = subset.melt(
            id_vars=["company_id", "indicator_id"],
            value_vars=YEAR_COLS,
            var_name="year",
            value_name="value",
        ).dropna(subset=["value"])
        long["year"] = long["year"].astype(int)
        del subset

        # ── Pivot to company-year × indicator (wide) ──────────────────────────
        wide = (
            long
            .pivot_table(
                index=["company_id", "year"],
                columns="indicator_id",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        del long

        # ── Standardise column names ───────────────────────────────────────────
        # Revenue: prefer Revenues, fall back to SalesRevenueNet
        wide["revenue"] = wide.get("Revenues", pd.Series(np.nan, index=wide.index))
        if "SalesRevenueNet" in wide.columns:
            wide["revenue"] = wide["revenue"].fillna(wide["SalesRevenueNet"])

        # Debt: prefer LongTermDebt, fall back to LongTermDebtNoncurrent
        wide["debt"] = wide.get("LongTermDebt", pd.Series(np.nan, index=wide.index))
        if "LongTermDebtNoncurrent" in wide.columns:
            wide["debt"] = wide["debt"].fillna(wide["LongTermDebtNoncurrent"])

        rename = {
            "Assets":                               "assets",
            "NetIncomeLoss":                        "net_income",
            "OperatingIncomeLoss":                  "operating_income",
            "GrossProfit":                          "gross_profit",
            "CostOfRevenue":                        "cost_of_revenue",
            "Liabilities":                          "liabilities",
            "StockholdersEquity":                   "equity",
            "CashAndCashEquivalentsAtCarryingValue":"cash",
        }
        wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})

        # Ensure all base columns exist (some may be absent for small datasets)
        for col in ["assets", "revenue", "net_income", "liabilities", "equity",
                    "cash", "debt", "operating_income", "gross_profit"]:
            if col not in wide.columns:
                wide[col] = np.nan

        # ── Derived features (safe division — no inf, no div-by-zero) ─────────
        def _div(num: pd.Series, den: pd.Series) -> pd.Series:
            return np.where(den.abs() > 1e-9, num / den, np.nan)

        wide["profit_margin"]    = _div(wide["net_income"],    wide["revenue"])
        wide["gross_margin"]     = _div(wide["gross_profit"],  wide["revenue"])
        wide["debt_to_assets"]   = _div(wide["debt"],          wide["assets"])
        wide["return_on_assets"] = _div(wide["net_income"],    wide["assets"])
        wide["return_on_equity"] = _div(wide["net_income"],    wide["equity"])
        wide["cash_to_assets"]   = _div(wide["cash"],          wide["assets"])

        # YoY growth (within each company, sorted by year)
        wide = wide.sort_values(["company_id", "year"])
        wide["revenue_growth"]    = (
            wide.groupby("company_id")["revenue"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )
        wide["net_income_growth"] = (
            wide.groupby("company_id")["net_income"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )

        # Log transforms (positive values only)
        wide["log_assets"]  = np.where(wide["assets"]  > 0, np.log(wide["assets"]),  np.nan)
        wide["log_revenue"] = np.where(wide["revenue"] > 0, np.log(wide["revenue"]), np.nan)

        # ── Coverage stats ─────────────────────────────────────────────────────
        diag["companies_with_data"] = wide["company_id"].nunique()
        diag["company_year_obs"]    = len(wide)
        diag["status"]              = "loaded"

        return wide, diag

    except Exception as exc:  # noqa: BLE001
        diag["status"] = "error"
        diag["error"]  = str(exc)
        return None, diag


# ── Model training ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "log_revenue", "net_income", "liabilities", "equity", "cash", "debt",
    "operating_income", "gross_profit",
    "profit_margin", "gross_margin", "debt_to_assets",
    "return_on_assets", "return_on_equity", "cash_to_assets",
    "revenue_growth", "net_income_growth",
]

TARGET_COL                        = "log_assets"
VALUATION_TARGET                  = "log_market_cap"
MODEL_OUTPUTS_PATH                = os.path.join(_HERE, "model_outputs.csv")
MODEL_OUTPUTS_REVIEWED_PATH       = os.path.join(_HERE, "model_outputs_reviewed.csv")
MODEL_METRICS_PATH                = os.path.join(_HERE, "model_metrics.csv")
FEATURE_IMP_PATH                  = os.path.join(_HERE, "feature_importance.csv")
MODEL_OUTPUT_QUALITY_SUMMARY      = os.path.join(_HERE, "model_output_quality_summary.csv")
MODEL_VALIDATION_COMPARISON_PATH  = os.path.join(_HERE, "model_validation_comparison.csv")

# Thresholds for output quality flagging
_MC_SUSPICIOUS_THRESHOLD   = 1e8    # $100M — below this is likely a data error for these companies
_GAP_EXTREME_THRESHOLD     = 100.0  # |valuation_gap_pct| > 100% is extreme
_ERROR_HIGH_THRESHOLD      = 1.0    # |model_error| > 1.0 log units is poor fit
_NEUTRAL_RISK_SCORE        = 50     # report_risk_score == 50 means no filing text was used


# ── Output quality flagging ───────────────────────────────────────────────────

def _classify_output_row(row: pd.Series) -> tuple:
    """
    Assign output_quality_flag and output_warning_text to a single model output row.

    Returns (flag_str, warning_str).
    flag_str is comma-separated list of applicable flags (or 'ok').
    Rows with any serious flag (suspicious_market_cap, extreme_valuation_gap,
    high_model_error) also receive the 'needs_review' flag.
    """
    flags    = []
    warnings = []

    mc  = row.get("actual_market_cap")
    gap = row.get("valuation_gap_pct")
    err = row.get("model_error")
    rsk = row.get("report_risk_score")

    # Market cap sanity check
    if pd.isna(mc) or (not pd.isna(mc) and float(mc) <= 0):
        flags.append("suspicious_market_cap")
        warnings.append("Market cap is missing or zero — possible data error.")
    elif not pd.isna(mc) and float(mc) < _MC_SUSPICIOUS_THRESHOLD:
        flags.append("suspicious_market_cap")
        warnings.append(
            f"Market cap is unusually low (${float(mc)/1e6:.2f}M) for a listed company "
            "— likely a units error in XBRL shares outstanding."
        )

    # Extreme valuation gap
    if not pd.isna(gap) and abs(float(gap)) > _GAP_EXTREME_THRESHOLD:
        flags.append("extreme_valuation_gap")
        warnings.append(
            f"Valuation gap of {float(gap):+.1f}% exceeds 100% — "
            "the model estimate is likely unreliable for this observation."
        )

    # Large model error
    if not pd.isna(err) and abs(float(err)) > _ERROR_HIGH_THRESHOLD:
        flags.append("high_model_error")
        warnings.append(
            f"Model error of {float(err):+.3f} (log scale) is large — "
            "the model fit was poor for this observation."
        )

    # Neutral report risk (informational, not a serious flag)
    if not pd.isna(rsk) and int(float(rsk)) == _NEUTRAL_RISK_SCORE:
        flags.append("neutral_report_risk")
        warnings.append(
            "Report risk is currently neutral because filing-text extraction "
            "has not yet been integrated for this dataset."
        )

    serious = {"suspicious_market_cap", "extreme_valuation_gap", "high_model_error"}
    if set(flags) & serious:
        flags.append("needs_review")

    flag_str    = ", ".join(flags) if flags else "ok"
    warning_str = " | ".join(warnings) if warnings else ""
    return flag_str, warning_str


def flag_model_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add output_quality_flag and output_warning_text columns to a model outputs
    DataFrame.  Rows with serious flags have final_signal_display overridden to
    'Needs review' so they cannot appear as high-priority candidates.

    Returns a new DataFrame (does not modify the input).
    """
    out = df.copy()
    flags, warnings = zip(*out.apply(_classify_output_row, axis=1))
    out["output_quality_flag"]  = list(flags)
    out["output_warning_text"]  = list(warnings)
    out["final_signal_display"] = out.apply(
        lambda r: "Needs review" if "needs_review" in str(r["output_quality_flag"])
        else r.get("final_signal", ""),
        axis=1,
    )
    return out


def save_reviewed_outputs(flagged_df: pd.DataFrame) -> None:
    """
    Save model_outputs_reviewed.csv (all rows + quality columns) and
    model_output_quality_summary.csv (flag counts).
    """
    flagged_df.to_csv(MODEL_OUTPUTS_REVIEWED_PATH, index=False)

    total = len(flagged_df)
    # Count by primary flag (first token)
    primary_flags = flagged_df["output_quality_flag"].apply(
        lambda s: s.split(",")[0].strip() if s else "ok"
    )
    counts = primary_flags.value_counts()
    summary_rows = [
        {
            "flag":       flag,
            "count":      int(cnt),
            "percentage": round(int(cnt) / total * 100, 1) if total > 0 else 0,
        }
        for flag, cnt in counts.items()
    ]
    pd.DataFrame(summary_rows).to_csv(MODEL_OUTPUT_QUALITY_SUMMARY, index=False)


# ── Explanation text ─────────────────────────────────────────────────────────

def _generate_explanation(signal: str, val_gap: float, quality: float,
                           report_risk: float, model_error: float) -> str:
    """
    Plain-English explanation of a model output row.
    Language constraints: no buy/sell, no investment advice claims.
    """
    # Valuation gap
    if pd.isna(val_gap):
        val_text = "the model could not estimate a valuation gap"
    elif abs(val_gap) < 5:
        val_text = "the model-estimated fair value is within 5% of actual market cap"
    elif val_gap > 0:
        val_text = (
            f"the model-estimated fair value is {val_gap:.1f}% above actual market cap"
        )
    else:
        val_text = (
            f"the model-estimated fair value is {abs(val_gap):.1f}% below actual market cap"
        )

    # Quality
    if quality >= 65:
        q_text = "fundamentals score in the upper third of the training dataset"
    elif quality >= 45:
        q_text = "fundamentals are in the middle range of the training dataset"
    else:
        q_text = "fundamentals score in the lower third of the training dataset"

    # Model error magnitude
    if pd.isna(model_error):
        e_text = "model error is unavailable for this row"
    elif abs(model_error) < 0.3:
        e_text = f"the log-scale model error is low ({model_error:+.2f})"
    elif abs(model_error) < 0.6:
        e_text = f"the log-scale model error is moderate ({model_error:+.2f})"
    else:
        e_text = (
            f"the log-scale model error is high ({model_error:+.2f}) — "
            "treat this estimate with additional caution"
        )

    return (
        f"Research signal: {signal}. "
        f"This signal was assigned because {val_text}; "
        f"{q_text}; "
        f"report risk is set to neutral because SEC filing text has not been "
        f"integrated into this dataset; and {e_text}. "
        f"The valuation gap is a research signal, not proof of undervaluation or "
        f"overvaluation. This output is for research purposes only and does not "
        f"constitute investment advice."
    )


# ── Quality scoring ───────────────────────────────────────────────────────────

def _compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute 0-100 quality score from XBRL features.
    Components are percentile-ranked across the dataset.
    Missing values receive a neutral 50 for that component.
    Components: ROE (30%), operating margin as FCF proxy (25%),
                revenue growth (25%), low debt-to-assets (20%).
    """
    def _pct(s, ascending=True):
        return s.rank(pct=True, ascending=ascending, na_option="keep") * 100

    rev = df["revenue"] if "revenue" in df.columns else pd.Series(np.nan, index=df.index)
    oi  = df["operating_income"] if "operating_income" in df.columns \
          else pd.Series(np.nan, index=df.index)
    op_margin = pd.Series(
        np.where(rev.abs() > 1e6, oi / rev, np.nan),
        index=df.index,
    )

    roe = df.get("return_on_equity", pd.Series(np.nan, index=df.index))
    rg  = df.get("revenue_growth",   pd.Series(np.nan, index=df.index))
    d2a = df.get("debt_to_assets",   pd.Series(np.nan, index=df.index))

    quality = (
        _pct(roe).fillna(50)                    * 0.30
        + _pct(op_margin).fillna(50)            * 0.25
        + _pct(rg).fillna(50)                   * 0.25
        + _pct(d2a, ascending=False).fillna(50) * 0.20
    )
    return quality.clip(0, 100)


# ── True valuation model (log market cap) ────────────────────────────────────

def _run_one_split(
    X_train, X_test, y_train, y_test,
    validation_type: str,
    train_meta: dict,
    test_meta: dict,
    features: list,
) -> list:
    """
    Train Ridge, Random Forest, and mean baseline on a single (train, test) split.
    Returns a list of row dicts suitable for model_validation_comparison.csv.
    train_meta / test_meta: dicts with keys 'companies', 'years', 'rows'.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    MIN_SPLIT_ROWS = 5
    if len(X_train) < MIN_SPLIT_ROWS or len(X_test) < MIN_SPLIT_ROWS:
        return [{
            "validation_type":  validation_type,
            "model_name":       "SKIPPED",
            "target":           VALUATION_TARGET,
            "train_rows":       len(X_train),
            "test_rows":        len(X_test),
            "train_companies":  train_meta.get("companies", ""),
            "test_companies":   test_meta.get("companies", ""),
            "train_years":      train_meta.get("years", ""),
            "test_years":       test_meta.get("years", ""),
            "mae":              None,
            "rmse":             None,
            "r2":               None,
            "is_best_for_split": False,
            "note":             f"Skipped — too few rows (train={len(X_train)}, test={len(X_test)}, need >={MIN_SPLIT_ROWS})",
        }]

    def _base_row(model_name):
        return {
            "validation_type":  validation_type,
            "model_name":       model_name,
            "target":           VALUATION_TARGET,
            "train_rows":       len(X_train),
            "test_rows":        len(X_test),
            "train_companies":  train_meta.get("companies", ""),
            "test_companies":   test_meta.get("companies", ""),
            "train_years":      train_meta.get("years", ""),
            "test_years":       test_meta.get("years", ""),
            "is_best_for_split": False,
            "note":             "",
        }

    rows = []

    # Mean baseline
    y_mean   = float(y_train.mean())
    base_hat = np.full(len(y_test), y_mean)
    r = _base_row("Mean Prediction (baseline)")
    r["mae"]  = round(float(mean_absolute_error(y_test, base_hat)), 4)
    r["rmse"] = round(float(mean_squared_error(y_test, base_hat) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(y_test, base_hat)), 4)
    rows.append(r)

    # Ridge
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_train)
    Xte_s   = scaler.transform(X_test)
    ridge   = Ridge(alpha=1.0)
    ridge.fit(Xtr_s, y_train)
    y_hat_r = ridge.predict(Xte_s)
    r = _base_row("Ridge Regression")
    r["mae"]  = round(float(mean_absolute_error(y_test, y_hat_r)), 4)
    r["rmse"] = round(float(mean_squared_error(y_test, y_hat_r) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(y_test, y_hat_r)), 4)
    rows.append(r)

    # Random Forest
    rf       = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_hat_rf = rf.predict(X_test)
    r = _base_row("Random Forest")
    r["mae"]  = round(float(mean_absolute_error(y_test, y_hat_rf)), 4)
    r["rmse"] = round(float(mean_squared_error(y_test, y_hat_rf) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(y_test, y_hat_rf)), 4)
    rows.append(r)

    # Mark best non-baseline model by RMSE
    non_base = [row for row in rows if row["model_name"] != "Mean Prediction (baseline)" and row["rmse"] is not None]
    if non_base:
        best_rmse = min(r["rmse"] for r in non_base)
        for row in non_base:
            if row["rmse"] == best_rmse:
                row["is_best_for_split"] = True
                break

    return rows


def run_validation_comparison(training_df: pd.DataFrame) -> list:
    """
    Run three validation strategies (row-based, company-level, time-based) and
    return a flat list of result dicts for model_validation_comparison.csv.

    Does not modify training_df or any fitted objects — purely diagnostic.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return []

    features = [f for f in FEATURE_COLS if f in training_df.columns]
    if not features:
        return []

    df = training_df.copy()

    # Impute with medians (same logic as train_valuation_model)
    for col in features:
        med = df[col].median()
        df[col] = df[col].fillna(float(med) if pd.notna(med) else 0.0)

    df = df.dropna(subset=[VALUATION_TARGET])
    if len(df) < 10:
        return []

    X = df[features].values
    y = df[VALUATION_TARGET].values

    # Identify company and year columns if present
    ticker_col = "ticker" if "ticker" in df.columns else ("company_id" if "company_id" in df.columns else None)
    year_col   = "year"   if "year"   in df.columns else None

    def _meta(idx_mask):
        sub = df[idx_mask]
        cos = sorted(sub[ticker_col].unique().tolist()) if ticker_col else []
        yrs = sorted(sub[year_col].astype(int).unique().tolist()) if year_col else []
        return {
            "companies": ", ".join(str(c) for c in cos),
            "years":     ", ".join(str(y) for y in yrs),
            "rows":      int(idx_mask.sum()),
        }

    all_rows = []

    # ── A. Row-based split ────────────────────────────────────────────────────
    idx_all  = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=42)
    tr_mask = pd.Series(False, index=df.index)
    tr_mask.iloc[tr_idx] = True
    te_mask = ~tr_mask
    all_rows += _run_one_split(
        X[tr_idx], X[te_idx], y[tr_idx], y[te_idx],
        validation_type="row_based",
        train_meta=_meta(tr_mask),
        test_meta=_meta(te_mask),
        features=features,
    )

    # ── B. Company-level split ────────────────────────────────────────────────
    if ticker_col:
        companies = df[ticker_col].unique()
        n_train_cos = max(1, int(len(companies) * 0.8))
        rng          = np.random.default_rng(42)
        train_cos    = set(rng.choice(companies, size=n_train_cos, replace=False).tolist())
        test_cos     = set(companies) - train_cos
        tr_mask_c    = df[ticker_col].isin(train_cos)
        te_mask_c    = df[ticker_col].isin(test_cos)
        tr_idx_c     = np.where(tr_mask_c.values)[0]
        te_idx_c     = np.where(te_mask_c.values)[0]
        all_rows += _run_one_split(
            X[tr_idx_c], X[te_idx_c], y[tr_idx_c], y[te_idx_c],
            validation_type="company_level",
            train_meta=_meta(tr_mask_c),
            test_meta=_meta(te_mask_c),
            features=features,
        )
    else:
        all_rows.append({
            "validation_type": "company_level", "model_name": "SKIPPED",
            "note": "No ticker/company_id column found.", "mae": None, "rmse": None, "r2": None,
            "train_rows": 0, "test_rows": 0, "is_best_for_split": False,
            "train_companies": "", "test_companies": "", "train_years": "", "test_years": "",
            "target": VALUATION_TARGET,
        })

    # ── C. Time-based split ───────────────────────────────────────────────────
    if year_col:
        years       = sorted(df[year_col].astype(int).unique())
        cutoff_idx  = max(1, int(len(years) * 0.75))   # ~75% of years for training
        train_years = set(years[:cutoff_idx])
        test_years  = set(years[cutoff_idx:])
        # Prefer explicit 2014/2015 boundary when available
        if 2014 in years and 2015 in years:
            train_years = {y for y in years if y <= 2014}
            test_years  = {y for y in years if y >= 2015}
        tr_mask_t   = df[year_col].astype(int).isin(train_years)
        te_mask_t   = df[year_col].astype(int).isin(test_years)
        tr_idx_t    = np.where(tr_mask_t.values)[0]
        te_idx_t    = np.where(te_mask_t.values)[0]
        all_rows += _run_one_split(
            X[tr_idx_t], X[te_idx_t], y[tr_idx_t], y[te_idx_t],
            validation_type="time_based",
            train_meta=_meta(tr_mask_t),
            test_meta=_meta(te_mask_t),
            features=features,
        )
    else:
        all_rows.append({
            "validation_type": "time_based", "model_name": "SKIPPED",
            "note": "No year column found.", "mae": None, "rmse": None, "r2": None,
            "train_rows": 0, "test_rows": 0, "is_best_for_split": False,
            "train_companies": "", "test_companies": "", "train_years": "", "test_years": "",
            "target": VALUATION_TARGET,
        })

    return all_rows


def train_valuation_model(training_df: pd.DataFrame) -> tuple:
    """
    Train Ridge Regression and Random Forest on log(market_cap).

    Returns
    -------
    result_dict : dict  — metrics only (safe for Streamlit @st.cache_data)
    fitted      : dict  — sklearn objects for generate_model_outputs(); not cached
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        empty = {
            "status": "sklearn_not_installed",
            "error":  "scikit-learn is required. Run: pip install scikit-learn",
        }
        return empty, {}

    features = [f for f in FEATURE_COLS if f in training_df.columns]
    model_df = training_df[[VALUATION_TARGET] + features].dropna(subset=[VALUATION_TARGET]).copy()

    # Median imputation — record medians for use in generate_model_outputs
    impute_medians: dict = {}
    for col in features:
        med = model_df[col].median()
        impute_medians[col] = float(med) if pd.notna(med) else 0.0
        model_df[col] = model_df[col].fillna(impute_medians[col])

    model_df = model_df.dropna()

    if len(model_df) < 30:
        return {"status": "insufficient_data", "n_obs": len(model_df)}, {}

    X = model_df[features]
    y = model_df[VALUATION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Mean-prediction baseline (predicts training mean for every test observation)
    y_mean       = float(y_train.mean())
    base_hat     = np.full(len(y_test), y_mean)
    baseline_mae  = float(mean_absolute_error(y_test, base_hat))
    baseline_rmse = float(mean_squared_error(y_test, base_hat) ** 0.5)
    baseline_r2   = float(r2_score(y_test, base_hat))

    result: dict = {
        "status":        "success",
        "model_type":    "true_market_cap",
        "target":        VALUATION_TARGET,
        "target_label":  "log(Market Cap) — True Valuation Model",
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "features_used": features,
        "models":        {},
        "baseline": {
            "Mean Prediction (baseline)": {
                "mae":  round(baseline_mae,  4),
                "rmse": round(baseline_rmse, 4),
                "r2":   round(baseline_r2,   4),
            }
        },
    }

    # Ridge Regression
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)
    ridge    = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_train)
    y_hat_r  = ridge.predict(X_te_s)

    result["models"]["Ridge Regression"] = {
        "mae":  round(float(mean_absolute_error(y_test, y_hat_r)), 4),
        "rmse": round(float(mean_squared_error(y_test, y_hat_r) ** 0.5), 4),
        "r2":   round(float(r2_score(y_test, y_hat_r)), 4),
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_hat_rf = rf.predict(X_test)

    importances = sorted(
        zip(features, rf.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    result["models"]["Random Forest"] = {
        "mae":         round(float(mean_absolute_error(y_test, y_hat_rf)), 4),
        "rmse":        round(float(mean_squared_error(y_test, y_hat_rf) ** 0.5), 4),
        "r2":          round(float(r2_score(y_test, y_hat_rf)), 4),
        "importances": importances[:10],
    }

    # Select best model by test RMSE (lower is better)
    ridge_rmse = result["models"]["Ridge Regression"]["rmse"]
    rf_rmse    = result["models"]["Random Forest"]["rmse"]
    result["best_model"] = "Ridge Regression" if ridge_rmse <= rf_rmse else "Random Forest"

    # Run all three validation strategies and stash in result
    try:
        result["validation_comparison"] = run_validation_comparison(model_df.assign(
            ticker=training_df["ticker"].values if "ticker" in training_df.columns else "",
            year=training_df["year"].values     if "year"   in training_df.columns else 0,
            company_id=training_df["company_id"].values if "company_id" in training_df.columns else "",
        ))
    except Exception as exc:
        result["validation_comparison"] = []
        result["validation_comparison_error"] = str(exc)

    fitted = {
        "rf":             rf,
        "scaler":         scaler,
        "ridge":          ridge,
        "features":       features,
        "impute_medians": impute_medians,
        "best_model":     result["best_model"],
    }

    return result, fitted


def generate_model_outputs(training_df: pd.DataFrame,
                            fitted: dict,
                            save: bool = True) -> pd.DataFrame:
    """
    Generate model_outputs.csv using the fitted Random Forest.
    Predicts on all training rows (80% will be in-sample, 20% holdout).

    Columns saved:
        company_id, company_name, ticker, year,
        actual_market_cap, estimated_fair_value,
        valuation_gap_pct, quality_score, report_risk_score,
        final_signal, model_error
    """
    from model_utils import compute_signal  # local import avoids circular dependency

    features       = fitted.get("features", [])
    impute_medians = fitted.get("impute_medians", {})
    best_model     = fitted.get("best_model", "Random Forest")

    if not features:
        return pd.DataFrame()

    # Prepare feature matrix with same imputation as training
    feat_df = training_df.copy()
    for col in features:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
        med = impute_medians.get(col, feat_df[col].median())
        feat_df[col] = feat_df[col].fillna(med if pd.notna(med) else 0.0)

    X_all = feat_df[features]

    # Use best model for predictions
    if best_model == "Ridge Regression":
        scaler = fitted.get("scaler")
        ridge  = fitted.get("ridge")
        if ridge is None:
            return pd.DataFrame()
        preds_log_mc = ridge.predict(scaler.transform(X_all))
    else:
        rf = fitted.get("rf")
        if rf is None:
            return pd.DataFrame()
        preds_log_mc = rf.predict(X_all)

    # Market cap columns
    actual_mc = (
        training_df["market_cap"].values
        if "market_cap" in training_df.columns
        else np.exp(training_df[VALUATION_TARGET].values)
    )
    est_fv = np.exp(preds_log_mc)

    # Valuation gap — clamp denominator to avoid inf
    denom = np.where(np.abs(actual_mc) > 1e6, actual_mc, np.nan)
    val_gap = (est_fv - actual_mc) / denom * 100

    quality_scores = _compute_quality_score(training_df).values

    out = pd.DataFrame({
        "company_id":          training_df["company_id"].values,
        "company_name":        training_df["company_name"].values
                               if "company_name" in training_df.columns
                               else "",
        "ticker":              training_df["ticker"].values,
        "year":                training_df["year"].values,
        "actual_market_cap":   actual_mc,
        "estimated_fair_value": est_fv,
        "valuation_gap_pct":   val_gap,
        "quality_score":       quality_scores,
        "report_risk_score":   50,   # placeholder — no filing text in XBRL dataset
        "model_error":         training_df[VALUATION_TARGET].values - preds_log_mc,
    })
    out["valuation_gap_pct"] = out["valuation_gap_pct"].replace([np.inf, -np.inf], np.nan)

    out["final_signal"] = out.apply(
        lambda r: compute_signal(
            float(r["valuation_gap_pct"]) if pd.notna(r["valuation_gap_pct"]) else 0.0,
            float(r["quality_score"])     if pd.notna(r["quality_score"])     else 50.0,
            float(r["report_risk_score"]),
        ),
        axis=1,
    )

    out["best_model_used"] = best_model

    out["explanation_text"] = out.apply(
        lambda r: _generate_explanation(
            signal     = r["final_signal"],
            val_gap    = r["valuation_gap_pct"],
            quality    = r["quality_score"],
            report_risk = r["report_risk_score"],
            model_error = r["model_error"],
        ),
        axis=1,
    )

    if save:
        out.to_csv(MODEL_OUTPUTS_PATH, index=False)
        reviewed = flag_model_outputs(out)
        save_reviewed_outputs(reviewed)

    return out


def save_model_artifacts(model_result: dict) -> None:
    """
    Save model_metrics.csv and feature_importance.csv from a successful
    train_valuation_model() result. Called by run_diagnostics() and the
    standalone script so both paths produce the same artefacts.
    """
    if model_result.get("status") != "success":
        return

    # model_metrics.csv
    rows = []
    best = model_result.get("best_model", "")
    for model_name, m in model_result.get("models", {}).items():
        rows.append({
            "model_name": model_name,
            "target":     model_result.get("target", VALUATION_TARGET),
            "n_train":    model_result.get("n_train"),
            "n_test":     model_result.get("n_test"),
            "mae":        m.get("mae"),
            "rmse":       m.get("rmse"),
            "r2":         m.get("r2"),
            "is_best":    model_name == best,
        })
    for base_name, bm in model_result.get("baseline", {}).items():
        rows.append({
            "model_name": base_name,
            "target":     model_result.get("target", VALUATION_TARGET),
            "n_train":    model_result.get("n_train"),
            "n_test":     model_result.get("n_test"),
            "mae":        bm.get("mae"),
            "rmse":       bm.get("rmse"),
            "r2":         bm.get("r2"),
            "is_best":    False,
        })
    if rows:
        pd.DataFrame(rows).to_csv(MODEL_METRICS_PATH, index=False)

    # feature_importance.csv (Random Forest only)
    rf_m = model_result.get("models", {}).get("Random Forest", {})
    imps = rf_m.get("importances", [])
    if imps:
        imp_df = pd.DataFrame(imps, columns=["feature", "importance"])
        imp_df["rank"] = range(1, len(imp_df) + 1)
        imp_df.to_csv(FEATURE_IMP_PATH, index=False)

    # model_validation_comparison.csv (all three split strategies)
    val_rows = model_result.get("validation_comparison", [])
    if val_rows:
        val_df = pd.DataFrame(val_rows)
        # Ensure consistent column order
        col_order = [
            "validation_type", "model_name", "target",
            "train_rows", "test_rows",
            "train_companies", "test_companies",
            "train_years", "test_years",
            "mae", "rmse", "r2", "is_best_for_split", "note",
        ]
        for c in col_order:
            if c not in val_df.columns:
                val_df[c] = ""
        val_df[col_order].to_csv(MODEL_VALIDATION_COMPARISON_PATH, index=False)


def train_proxy_model(df: pd.DataFrame) -> dict:
    """
    Train Ridge Regression and Random Forest on log(Total Assets).

    Returns a dict with model metrics and feature importances.
    The caller is responsible for labeling all outputs as proxy estimates.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {
            "status": "sklearn_not_installed",
            "error":  "scikit-learn is required. Run: pip install scikit-learn",
        }

    features = [f for f in FEATURE_COLS if f in df.columns]
    model_df = df[[TARGET_COL] + features].dropna(subset=[TARGET_COL]).copy()

    # Median imputation for feature NaNs
    for col in features:
        med = model_df[col].median()
        model_df[col] = model_df[col].fillna(med if pd.notna(med) else 0.0)

    model_df = model_df.dropna()

    if len(model_df) < 100:
        return {"status": "insufficient_data", "n_obs": len(model_df)}

    X = model_df[features]
    y = model_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    result = {
        "status":        "success",
        "target":        TARGET_COL,
        "target_label":  "log(Total Assets) — Fundamental Size Proxy",
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "features_used": features,
        "models":        {},
    }

    # Ridge Regression (scaled inputs)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)
    ridge    = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_train)
    y_hat_r  = ridge.predict(X_te_s)

    result["models"]["Ridge Regression"] = {
        "mae":  round(float(mean_absolute_error(y_test, y_hat_r)), 4),
        "rmse": round(float(mean_squared_error(y_test, y_hat_r) ** 0.5), 4),
        "r2":   round(float(r2_score(y_test, y_hat_r)), 4),
    }

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_hat_rf = rf.predict(X_test)

    importances = sorted(
        zip(features, rf.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    result["models"]["Random Forest"] = {
        "mae":         round(float(mean_absolute_error(y_test, y_hat_rf)), 4),
        "rmse":        round(float(mean_squared_error(y_test, y_hat_rf) ** 0.5), 4),
        "r2":          round(float(r2_score(y_test, y_hat_rf)), 4),
        "importances": importances[:10],
    }

    return result


# ── Public entry point ────────────────────────────────────────────────────────

def run_diagnostics() -> tuple:
    """
    Called by app.py.  Returns (data_diag, model_results).

    Priority:
    1. If valuation_training_dataset.csv exists with >= 30 usable rows →
       train the true log(market_cap) model and save model_outputs.csv.
    2. Otherwise → fall back to the proxy log(assets) model on XBRL data.
    """
    df, data_diag = load_xbrl_data()
    if df is None:
        return data_diag, None

    # ── Try true market-cap model ──────────────────────────────────────────────
    try:
        from market_cap_utils import (
            load_training_dataset,
            load_ticker_mapping,
            load_market_cap_data,
        )

        training_df, _  = load_training_dataset()
        mapping_df,  _  = load_ticker_mapping()
        _,           mc_info = load_market_cap_data()

        data_diag["ticker_mapping_rows"]  = len(mapping_df) if mapping_df is not None else 0
        data_diag["market_cap_ok_rows"]   = mc_info.get("ok_rows",    0)
        data_diag["market_cap_companies"] = mc_info.get("companies",   0)

        if (
            training_df is not None
            and len(training_df) >= 30
            and VALUATION_TARGET in training_df.columns
        ):
            n_co = (
                training_df["ticker"].nunique()
                if "ticker" in training_df.columns
                else "?"
            )
            data_diag["target_variable"] = VALUATION_TARGET
            data_diag["target_label"]    = "log(Market Cap) — True Valuation Model"
            data_diag["target_note"] = (
                f"Market cap estimated via yfinance (split-adjusted price × XBRL shares "
                f"outstanding, with cumulative split-factor correction). "
                f"Training dataset: {len(training_df)} company-year rows, "
                f"{n_co} companies, years 2010–2016."
            )
            data_diag["market_cap_training_rows"] = len(training_df)

            model_results, fitted = train_valuation_model(training_df)

            if model_results.get("status") == "success" and fitted:
                generate_model_outputs(training_df, fitted, save=True)
                save_model_artifacts(model_results)

            return data_diag, model_results

    except Exception as exc:  # noqa: BLE001
        data_diag["true_model_error"] = str(exc)

    # ── Fallback: proxy model ──────────────────────────────────────────────────
    model_results = train_proxy_model(df)
    return data_diag, model_results
