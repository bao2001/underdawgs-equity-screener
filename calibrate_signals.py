"""
calibrate_signals.py
Signal calibration for model_outputs_combined.csv.

Adds year- and era-level valuation percentile ranks, relative-value buckets,
and calibrated signal labels that compare each company against its own year/era
rather than fixed absolute thresholds.

Generates:
  signal_distribution_by_year.csv
  valuation_gap_distribution_by_era.csv
  model_calibration_diagnostics.csv
  model_outputs_combined_calibrated.csv

Run:  python3 calibrate_signals.py
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

COMBINED_PATH   = os.path.join(_HERE, "model_outputs_combined.csv")
CALIBRATED_PATH = os.path.join(_HERE, "model_outputs_combined_calibrated.csv")
SIG_DIST_PATH   = os.path.join(_HERE, "signal_distribution_by_year.csv")
ERA_DIST_PATH   = os.path.join(_HERE, "valuation_gap_distribution_by_era.csv")
CALIB_DIAG_PATH = os.path.join(_HERE, "model_calibration_diagnostics.csv")

# Signal labels (original and calibrated)
_ORIG_SIGS = [
    "High-priority research candidate",
    "Research candidate",
    "Fairly valued / neutral",
    "Possible value trap",
    "Potentially overvalued",
]
_CALIB_SIGS = _ORIG_SIGS + ["Needs review"]

# Stable short slugs for column names
_SIG_SLUG = {
    "High-priority research candidate": "high_priority",
    "Research candidate":               "research_candidate",
    "Fairly valued / neutral":          "fairly_valued_neutral",
    "Possible value trap":              "possible_value_trap",
    "Potentially overvalued":           "potentially_overvalued",
    "Needs review":                     "needs_review",
}


# ── Era assignment ─────────────────────────────────────────────────────────────

def _assign_era(year: int) -> str:
    if year <= 2016:
        return "legacy_2010_2016"
    if year <= 2020:
        return "modern_2017_2020"
    return "modern_2021_2024"


# ── Valuation bucket from percentile ──────────────────────────────────────────

def _assign_bucket(pct: float, flag: str = "ok") -> str:
    """Map a valuation-gap percentile (0-100, higher = better relative value) to a bucket label."""
    if "needs_review" in str(flag):
        return "Needs review"
    if pd.isna(pct):
        return "Neutral relative value"
    if pct >= 90:
        return "Top decile relative value"
    if pct >= 70:
        return "Above-average relative value"
    if pct >= 40:
        return "Neutral relative value"
    if pct >= 20:
        return "Below-average relative value"
    return "Low relative value"


# ── Calibrated signal from bucket + quality ────────────────────────────────────

def _assign_calibrated_signal(row) -> str:
    flag    = str(row.get("output_quality_flag", "ok"))
    if "needs_review" in flag:
        return "Needs review"

    bucket  = row.get("valuation_bucket_year", "Neutral relative value")
    quality = float(row.get("quality_score", 50) or 50)

    if bucket in ("Top decile relative value", "Above-average relative value"):
        if quality < 45:
            return "Possible value trap"
        return "Research candidate"
    if bucket == "Neutral relative value":
        return "Fairly valued / neutral"
    # Below-average or Low relative value
    return "Potentially overvalued"


# ── Calibration warning text ───────────────────────────────────────────────────

def _calibration_warning(row) -> str:
    flag       = str(row.get("output_quality_flag", "ok"))
    if "needs_review" in flag:
        return ""  # output_warning_text already covers needs_review rows

    original   = str(row.get("final_signal", ""))
    calibrated = str(row.get("final_signal_calibrated", ""))
    quality    = float(row.get("quality_score", 50) or 50)
    vgap       = float(row.get("valuation_gap_pct", 0) or 0)
    src        = str(row.get("model_output_source", ""))
    year       = int(row.get("year", 0))

    parts = []

    if quality < 30:
        parts.append(
            f"Very low quality score ({quality:.0f}/100) — high valuation gap "
            "may not reflect fundamental strength."
        )

    if (original == "Potentially overvalued"
            and calibrated not in ("Potentially overvalued", "Needs review")
            and "modern" in src):
        parts.append(
            f"Original signal used absolute thresholds (trained on 2010–2016 valuations). "
            f"Calibrated signal compares against {year} peers — "
            "valuations in this era were generally higher."
        )

    if abs(vgap) > 100:
        parts.append(
            f"Extreme valuation gap ({vgap:+.0f}%) — "
            "model prediction may be unreliable for this company-year."
        )

    return " | ".join(parts)


# ── Signal distribution by year ────────────────────────────────────────────────

def _build_signal_dist(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, grp in df.groupby("year"):
        total = len(grp)
        nr    = grp["output_quality_flag"].str.contains("needs_review", na=False).sum()
        sig_v = grp["final_signal"].value_counts()
        cal_v = grp["final_signal_calibrated"].value_counts() if "final_signal_calibrated" in grp.columns else pd.Series(dtype=int)

        row = {
            "year":                         int(year),
            "era":                          _assign_era(int(year)),
            "total_rows":                   total,
            "needs_review_rows":            int(nr),
            "avg_valuation_gap_pct":        round(grp["valuation_gap_pct"].mean(), 2),
            "median_valuation_gap_pct":     round(grp["valuation_gap_pct"].median(), 2),
            "avg_quality_score":            round(grp["quality_score"].mean(), 2),
            "avg_report_risk_score":        round(grp["report_risk_score"].mean(), 2),
        }
        for sig, slug in _SIG_SLUG.items():
            n = int(sig_v.get(sig, 0))
            row[f"orig_n_{slug}"]   = n
            row[f"orig_pct_{slug}"] = round(n / total * 100, 1) if total else 0
        for sig, slug in _SIG_SLUG.items():
            n = int(cal_v.get(sig, 0))
            row[f"cal_n_{slug}"]   = n
            row[f"cal_pct_{slug}"] = round(n / total * 100, 1) if total else 0

        # Concentration warnings
        orig_max = max(row[f"orig_pct_{s}"] for s in _SIG_SLUG.values())
        cal_max  = max(row[f"cal_pct_{s}"]  for s in _SIG_SLUG.values())
        row["orig_signal_concentrated"] = "yes" if orig_max > 60 else "no"
        row["cal_signal_concentrated"]  = "yes" if cal_max  > 60 else "no"

        rows.append(row)
    return pd.DataFrame(rows)


# ── Valuation gap distribution by era ─────────────────────────────────────────

def _build_era_dist(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for era, grp in df.groupby("era"):
        total = len(grp)
        vg    = grp["valuation_gap_pct"].dropna()
        sig_v = grp["final_signal"].value_counts()
        cal_v = grp["final_signal_calibrated"].value_counts() if "final_signal_calibrated" in grp.columns else pd.Series(dtype=int)

        row = {
            "era":                       era,
            "rows":                      total,
            "years":                     str(sorted(grp["year"].astype(int).unique().tolist())),
            "mean_valuation_gap_pct":    round(vg.mean(),  2),
            "median_valuation_gap_pct":  round(vg.median(), 2),
            "std_valuation_gap_pct":     round(vg.std(),   2),
            "p10_valuation_gap_pct":     round(vg.quantile(0.10), 2),
            "p25_valuation_gap_pct":     round(vg.quantile(0.25), 2),
            "p50_valuation_gap_pct":     round(vg.quantile(0.50), 2),
            "p75_valuation_gap_pct":     round(vg.quantile(0.75), 2),
            "p90_valuation_gap_pct":     round(vg.quantile(0.90), 2),
        }
        for sig, slug in _SIG_SLUG.items():
            n = int(sig_v.get(sig, 0))
            row[f"orig_n_{slug}"]   = n
            row[f"orig_pct_{slug}"] = round(n / total * 100, 1) if total else 0
        for sig, slug in _SIG_SLUG.items():
            n = int(cal_v.get(sig, 0))
            row[f"cal_n_{slug}"]   = n
            row[f"cal_pct_{slug}"] = round(n / total * 100, 1) if total else 0

        rows.append(row)
    return pd.DataFrame(rows)


# ── Calibration diagnostics by source × era ───────────────────────────────────

def _build_calib_diag(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, era), grp in df.groupby(["model_output_source", "era"]):
        total  = len(grp)
        vg     = grp["valuation_gap_pct"].dropna()
        me     = grp["model_error"].dropna() if "model_error" in grp.columns else pd.Series(dtype=float)
        sig_v  = grp["final_signal"].value_counts()
        cal_v  = (grp["final_signal_calibrated"].value_counts()
                  if "final_signal_calibrated" in grp.columns else pd.Series(dtype=int))
        mc_col = "actual_market_cap"
        ev_col = "estimated_fair_value"

        orig_dom_n   = int(sig_v.iloc[0]) if len(sig_v) else 0
        orig_dom_sig = sig_v.idxmax() if len(sig_v) else ""
        cal_dom_n    = int(cal_v.iloc[0]) if len(cal_v) else 0
        cal_dom_sig  = cal_v.idxmax() if len(cal_v) else ""

        row = {
            "model_output_source":           src,
            "era":                           era,
            "rows":                          total,
            "mean_actual_mc_B":              round(grp[mc_col].mean() / 1e9, 2) if mc_col in grp.columns else None,
            "mean_estimated_fv_B":           round(grp[ev_col].mean() / 1e9, 2) if ev_col in grp.columns else None,
            "mean_valuation_gap_pct":        round(vg.mean(), 2),
            "median_valuation_gap_pct":      round(vg.median(), 2),
            "mean_abs_model_error":          round(me.abs().mean(), 4) if len(me) else None,
            "rmse_model_error":              round(float((me ** 2).mean() ** 0.5), 4) if len(me) else None,
            "dominant_orig_signal":          orig_dom_sig,
            "dominant_orig_signal_pct":      round(orig_dom_n / total * 100, 1) if total else 0,
            "dominant_cal_signal":           cal_dom_sig,
            "dominant_cal_signal_pct":       round(cal_dom_n / total * 100, 1) if total else 0,
            "orig_concentration_warning":    "yes" if (orig_dom_n / total * 100 > 60 if total else False) else "no",
            "cal_concentration_warning":     "yes" if (cal_dom_n  / total * 100 > 60 if total else False) else "no",
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_calibration() -> None:
    print(f"Loading: {COMBINED_PATH}")
    if not os.path.exists(COMBINED_PATH):
        print(f"ERROR: {COMBINED_PATH} not found. Run modern_fundamentals_utils.py first.")
        return

    df = pd.read_csv(COMBINED_PATH)
    years = sorted(df["year"].dropna().astype(int).unique())
    print(f"  {len(df)} rows, years {years[0]}–{years[-1]}, {df['ticker'].nunique()} tickers")

    # ── 1. Era ────────────────────────────────────────────────────────────────
    df["era"] = df["year"].astype(int).apply(_assign_era)

    # ── 2. Percentile ranks (higher = relatively better value in the same year/era) ──
    df["valuation_gap_percentile_year"] = (
        df.groupby("year")["valuation_gap_pct"]
          .rank(pct=True, na_option="keep") * 100
    ).round(1)
    df["valuation_gap_percentile_era"] = (
        df.groupby("era")["valuation_gap_pct"]
          .rank(pct=True, na_option="keep") * 100
    ).round(1)

    # ── 3. Valuation buckets ──────────────────────────────────────────────────
    df["valuation_bucket_year"] = df.apply(
        lambda r: _assign_bucket(r["valuation_gap_percentile_year"],
                                 r.get("output_quality_flag", "ok")),
        axis=1,
    )
    df["valuation_bucket_era"] = df.apply(
        lambda r: _assign_bucket(r["valuation_gap_percentile_era"],
                                 r.get("output_quality_flag", "ok")),
        axis=1,
    )

    # ── 4. Calibrated signal ──────────────────────────────────────────────────
    df["final_signal_calibrated"] = df.apply(_assign_calibrated_signal, axis=1)

    # ── 5. Calibration warning text ───────────────────────────────────────────
    df["calibration_warning_text"] = df.apply(_calibration_warning, axis=1)

    # ── 6. Signal distribution by year ───────────────────────────────────────
    print("Building signal_distribution_by_year.csv …")
    sig_dist = _build_signal_dist(df)
    sig_dist.to_csv(SIG_DIST_PATH, index=False)
    print(f"  Saved: {SIG_DIST_PATH}  ({len(sig_dist)} rows)")

    # ── 7. Era distribution ───────────────────────────────────────────────────
    print("Building valuation_gap_distribution_by_era.csv …")
    era_dist = _build_era_dist(df)
    era_dist.to_csv(ERA_DIST_PATH, index=False)
    print(f"  Saved: {ERA_DIST_PATH}  ({len(era_dist)} rows)")

    # ── 8. Calibration diagnostics ────────────────────────────────────────────
    print("Building model_calibration_diagnostics.csv …")
    calib_diag = _build_calib_diag(df)
    calib_diag.to_csv(CALIB_DIAG_PATH, index=False)
    print(f"  Saved: {CALIB_DIAG_PATH}  ({len(calib_diag)} rows)")

    # ── 9. Combined calibrated output ─────────────────────────────────────────
    print("Saving model_outputs_combined_calibrated.csv …")
    df.to_csv(CALIBRATED_PATH, index=False)
    print(f"  Saved: {CALIBRATED_PATH}  ({len(df)} rows, {len(df.columns)} columns)")

    # ── 10. Console summary ───────────────────────────────────────────────────
    print("\n── Original vs Calibrated signal distribution by era ──")
    for era in sorted(df["era"].unique()):
        sub = df[df["era"] == era]
        orig_top = sub["final_signal"].value_counts().head(3)
        cal_top  = sub["final_signal_calibrated"].value_counts().head(3)
        print(f"\n  {era} ({len(sub)} rows)")
        print("    Original:")
        for sig, n in orig_top.items():
            print(f"      {sig:35s}: {n:3d} ({n/len(sub)*100:.0f}%)")
        print("    Calibrated:")
        for sig, n in cal_top.items():
            print(f"      {sig:35s}: {n:3d} ({n/len(sub)*100:.0f}%)")

    dominated_years = sig_dist[sig_dist["orig_signal_concentrated"] == "yes"]["year"].tolist()
    if dominated_years:
        print(f"\n  ⚠ Years dominated by one original signal (>60%): {dominated_years}")
    cal_dominated = sig_dist[sig_dist["cal_signal_concentrated"] == "yes"]["year"].tolist()
    if cal_dominated:
        print(f"  ⚠ Years still dominated after calibration (>60%): {cal_dominated}")
    else:
        print("\n  ✓ No year dominated by one calibrated signal (>60%).")

    print("\nDone.\n")


if __name__ == "__main__":
    run_calibration()
