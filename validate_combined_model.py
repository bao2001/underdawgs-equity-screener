"""
validate_combined_model.py
Validate the combined 2010–2024 valuation model across four split strategies.

Splits:
  A. row_based_combined         — random 80/20 row split across all years
  B. company_level_combined     — split by unique ticker
  C. time_based_legacy_to_modern — train 2010–2016, test 2017–2024
  D. time_based_pre_2021_to_2021_2024 — train 2010–2020, test 2021–2024

Models evaluated per split:
  - Mean Prediction (baseline)
  - Ridge Regression
  - Random Forest Regressor

Generates: combined_model_validation_comparison.csv

Run: python3 validate_combined_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

COMBINED_CALIBRATED_PATH = os.path.join(_HERE, "model_outputs_combined_calibrated.csv")
COMBINED_PATH            = os.path.join(_HERE, "model_outputs_combined.csv")
TRAINING_MODERN_PATH     = os.path.join(_HERE, "modern_valuation_training_dataset.csv")
TRAINING_LEGACY_PATH     = os.path.join(_HERE, "valuation_training_dataset.csv")
OUTPUT_PATH              = os.path.join(_HERE, "combined_model_validation_comparison.csv")

FEATURE_COLS = [
    "log_revenue", "net_income", "liabilities", "equity", "cash", "debt",
    "operating_income", "gross_profit",
    "profit_margin", "gross_margin", "debt_to_assets",
    "return_on_assets", "return_on_equity", "cash_to_assets",
    "revenue_growth", "net_income_growth",
]
TARGET = "log_market_cap"
MIN_SPLIT_ROWS = 5


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_training_data() -> pd.DataFrame:
    """
    Load combined training dataset from modern + legacy training CSVs.
    Falls back to model_outputs_combined.csv with re-derived log_market_cap
    if training CSVs are unavailable.
    """
    parts = []

    if os.path.exists(TRAINING_LEGACY_PATH):
        leg = pd.read_csv(TRAINING_LEGACY_PATH)
        leg["year"] = leg["year"].astype(int)
        parts.append(leg)
        print(f"  Legacy training: {len(leg)} rows, years {sorted(leg['year'].unique().tolist())}")

    if os.path.exists(TRAINING_MODERN_PATH):
        mod = pd.read_csv(TRAINING_MODERN_PATH)
        mod["year"] = mod["year"].astype(int)
        parts.append(mod)
        print(f"  Modern training: {len(mod)} rows, years {sorted(mod['year'].unique().tolist())}")

    if parts:
        df = pd.concat(parts, ignore_index=True, sort=False)
        if TARGET not in df.columns and "market_cap" in df.columns:
            df[TARGET] = np.where(df["market_cap"] > 0, np.log(df["market_cap"]), np.nan)
        return df

    # Fallback: derive from model outputs
    for path in [COMBINED_CALIBRATED_PATH, COMBINED_PATH]:
        if os.path.exists(path):
            raw = pd.read_csv(path)
            raw["year"] = raw["year"].astype(int)
            if "actual_market_cap" in raw.columns:
                raw[TARGET] = np.where(
                    raw["actual_market_cap"] > 0,
                    np.log(raw["actual_market_cap"]),
                    np.nan,
                )
                print(f"  Derived training from {os.path.basename(path)}: {len(raw)} rows")
                return raw
            break

    print("  ERROR: No training data found.")
    return pd.DataFrame()


# ── One split runner ───────────────────────────────────────────────────────────

def _run_split(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list,
    validation_type: str,
) -> list:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    def _meta(sub):
        return {
            "companies": ", ".join(sorted(sub["ticker"].dropna().unique().tolist()))
                          if "ticker" in sub.columns else "",
            "years":     ", ".join(str(y) for y in sorted(sub["year"].dropna().astype(int).unique().tolist()))
                          if "year" in sub.columns else "",
            "rows":      len(sub),
        }

    # Impute with training medians
    medians = df_train[features].median()
    Xtr = df_train[features].fillna(medians).values
    ytr = df_train[TARGET].values
    Xte = df_test[features].fillna(medians).values
    yte = df_test[TARGET].values

    # Remove rows where target is NaN
    tr_mask = ~np.isnan(ytr)
    te_mask = ~np.isnan(yte)
    Xtr, ytr = Xtr[tr_mask], ytr[tr_mask]
    Xte, yte = Xte[te_mask], yte[te_mask]

    tr_meta = _meta(df_train)
    te_meta = _meta(df_test)

    if len(Xtr) < MIN_SPLIT_ROWS or len(Xte) < MIN_SPLIT_ROWS:
        return [{
            "validation_type":  validation_type,
            "model_name":       "SKIPPED",
            "target":           TARGET,
            "train_rows":       len(Xtr),
            "test_rows":        len(Xte),
            "train_companies":  tr_meta["companies"],
            "test_companies":   te_meta["companies"],
            "train_years":      tr_meta["years"],
            "test_years":       te_meta["years"],
            "mae": None, "rmse": None, "r2": None,
            "is_best_for_split": False,
            "note": f"Skipped — too few rows (train={len(Xtr)}, test={len(Xte)}, need >={MIN_SPLIT_ROWS})",
        }]

    def _base_row(name):
        return {
            "validation_type":  validation_type,
            "model_name":       name,
            "target":           TARGET,
            "train_rows":       len(Xtr),
            "test_rows":        len(Xte),
            "train_companies":  tr_meta["companies"],
            "test_companies":   te_meta["companies"],
            "train_years":      tr_meta["years"],
            "test_years":       te_meta["years"],
            "is_best_for_split": False,
            "note": "",
        }

    rows = []

    # Mean baseline
    base_hat = np.full(len(yte), float(ytr.mean()))
    r = _base_row("Mean Prediction (baseline)")
    r["mae"]  = round(float(mean_absolute_error(yte, base_hat)), 4)
    r["rmse"] = round(float(mean_squared_error(yte, base_hat) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(yte, base_hat)), 4)
    rows.append(r)

    # Ridge
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(Xtr)
    Xte_s   = scaler.transform(Xte)
    ridge   = Ridge(alpha=1.0)
    ridge.fit(Xtr_s, ytr)
    r = _base_row("Ridge Regression")
    yhat = ridge.predict(Xte_s)
    r["mae"]  = round(float(mean_absolute_error(yte, yhat)), 4)
    r["rmse"] = round(float(mean_squared_error(yte, yhat) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(yte, yhat)), 4)
    rows.append(r)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    r = _base_row("Random Forest Regressor")
    yhat = rf.predict(Xte)
    r["mae"]  = round(float(mean_absolute_error(yte, yhat)), 4)
    r["rmse"] = round(float(mean_squared_error(yte, yhat) ** 0.5), 4)
    r["r2"]   = round(float(r2_score(yte, yhat)), 4)
    rows.append(r)

    # Mark best non-baseline for this split
    non_base = [row for row in rows if "baseline" not in row["model_name"].lower()
                and row["r2"] is not None]
    if non_base:
        best = max(non_base, key=lambda x: x["r2"])
        best["is_best_for_split"] = True

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def run_combined_validation() -> None:
    print("Loading training data …")
    df = _load_training_data()
    if df.empty:
        print("ERROR: No training data. Cannot run validation.")
        return

    features = [f for f in FEATURE_COLS if f in df.columns]
    print(f"  {len(df)} rows, {len(features)} features, years "
          f"{sorted(df['year'].dropna().astype(int).unique().tolist())}")

    usable = df.dropna(subset=[TARGET]).copy()
    print(f"  {len(usable)} usable rows (have {TARGET})")

    all_rows = []

    # ── A. Row-based combined ──────────────────────────────────────────────────
    print("\nA. row_based_combined …")
    np.random.seed(42)
    idx = np.arange(len(usable))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    tr_rows = _run_split(
        usable.iloc[tr_idx], usable.iloc[te_idx],
        features, "row_based_combined",
    )
    all_rows.extend(tr_rows)
    for r in tr_rows:
        print(f"  {r['model_name']:35s}  R²={r['r2']}  MAE={r['mae']}")

    # ── B. Company-level combined ──────────────────────────────────────────────
    print("\nB. company_level_combined …")
    if "ticker" in usable.columns:
        tickers = sorted(usable["ticker"].dropna().unique().tolist())
        np.random.seed(42)
        np.random.shuffle(tickers)
        split_t = int(0.8 * len(tickers))
        train_t = set(tickers[:split_t])
        test_t  = set(tickers[split_t:])
        tr_df   = usable[usable["ticker"].isin(train_t)]
        te_df   = usable[usable["ticker"].isin(test_t)]
        tr_rows = _run_split(tr_df, te_df, features, "company_level_combined")
        all_rows.extend(tr_rows)
        for r in tr_rows:
            print(f"  {r['model_name']:35s}  R²={r['r2']}  MAE={r['mae']}")
    else:
        print("  SKIPPED — no ticker column in training data")

    # ── C. Time-based: legacy → modern ────────────────────────────────────────
    print("\nC. time_based_legacy_to_modern …")
    tr_df = usable[usable["year"] <= 2016]
    te_df = usable[usable["year"] >= 2017]
    tr_rows = _run_split(tr_df, te_df, features, "time_based_legacy_to_modern")
    all_rows.extend(tr_rows)
    for r in tr_rows:
        print(f"  {r['model_name']:35s}  R²={r['r2']}  MAE={r['mae']}")

    # ── D. Time-based: pre-2021 → 2021–2024 ──────────────────────────────────
    print("\nD. time_based_pre_2021_to_2021_2024 …")
    tr_df = usable[usable["year"] <= 2020]
    te_df = usable[usable["year"] >= 2021]
    tr_rows = _run_split(tr_df, te_df, features, "time_based_pre_2021_to_2021_2024")
    all_rows.extend(tr_rows)
    for r in tr_rows:
        print(f"  {r['model_name']:35s}  R²={r['r2']}  MAE={r['mae']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(out_df)} rows)")

    # Summary warnings
    print("\n── Summary ──")
    _summary(out_df)


def _summary(df: pd.DataFrame) -> None:
    def _best_r2(vtype):
        sub = df[
            (df["validation_type"] == vtype) &
            (~df["model_name"].str.contains("baseline|SKIPPED", case=False, na=False))
        ]
        vals = pd.to_numeric(sub["r2"], errors="coerce").dropna()
        return float(vals.max()) if len(vals) > 0 else None

    rb_r2  = _best_r2("row_based_combined")
    cl_r2  = _best_r2("company_level_combined")
    ltm_r2 = _best_r2("time_based_legacy_to_modern")
    p21_r2 = _best_r2("time_based_pre_2021_to_2021_2024")

    for label, r2 in [
        ("row_based_combined",              rb_r2),
        ("company_level_combined",          cl_r2),
        ("time_based_legacy_to_modern",     ltm_r2),
        ("time_based_pre_2021_to_2021_2024", p21_r2),
    ]:
        if r2 is None:
            print(f"  {label}: SKIPPED")
        else:
            strength = "strong" if r2 >= 0.6 else ("moderate" if r2 >= 0.3 else "weak")
            print(f"  {label}: best R²={r2:.3f}  ({strength})")

    if ltm_r2 is not None and ltm_r2 < 0.3:
        print("\n  ⚠ legacy→modern R² is weak — model does not generalize well across eras.")
        print("    Calibrated signals are particularly important for modern years.")
    elif ltm_r2 is not None:
        print(f"\n  ✓ legacy→modern R²={ltm_r2:.3f} — model shows some cross-era generalization.")


if __name__ == "__main__":
    run_combined_validation()
