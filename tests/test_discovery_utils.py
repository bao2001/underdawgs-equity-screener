import numpy as np, pandas as pd, pytest
from discovery_utils import (add_fundamental_ratios, range_filter, fmt_ratio, load_saved_screens,
                             save_screen, delete_screen, sanitize_multiselect, peer_table, sector_medians)

RAW = pd.DataFrame({
    "Ticker": ["A", "B", "C", "D"], "Sector": ["Tech", "Tech", "Energy", "Tech"],
    "current_market_cap": [100e9, 50e9, 20e9, 10e9],
    "revenue": [50e9, 10e9, 40e9, 0.0], "net_income": [5e9, -1e9, 2e9, 1e9],
    "operating_income": [10e9, -0.5e9, 4e9, 1e9], "equity": [25e9, 10e9, -5e9, 4e9],
    "debt": [10e9, 0.0, 8e9, 2e9], "revenue_growth": [0.10, -0.05, np.nan, 0.2],
})

def test_ratios_basic_and_guarded():
    r = add_fundamental_ratios(RAW).set_index("Ticker")
    assert r.loc["A", "PE_Ratio"] == pytest.approx(20.0)
    assert r.loc["A", "PS_Ratio"] == pytest.approx(2.0)
    assert r.loc["A", "PB_Ratio"] == pytest.approx(4.0)
    assert r.loc["A", "Op_Margin_Pct"] == pytest.approx(20.0)
    assert r.loc["A", "ROE_Pct"] == pytest.approx(20.0)
    assert r.loc["A", "Debt_Equity"] == pytest.approx(0.4)
    assert r.loc["A", "Revenue_Growth_Pct"] == pytest.approx(10.0)
    assert np.isnan(r.loc["B", "PE_Ratio"])            # loss-making -> no P/E
    assert np.isnan(r.loc["C", "ROE_Pct"]) and np.isnan(r.loc["C", "Debt_Equity"])  # negative equity
    assert np.isnan(r.loc["D", "PS_Ratio"]) and np.isnan(r.loc["D", "Op_Margin_Pct"])  # zero revenue

def test_ratios_missing_columns_are_nan():
    r = add_fundamental_ratios(pd.DataFrame({"Ticker": ["X"]}))
    assert r[["PE_Ratio", "ROE_Pct"]].isna().all().all()

def test_range_filter_inert_by_default_keeps_nan():
    r = add_fundamental_ratios(RAW)
    assert len(range_filter(r, "PE_Ratio")) == 4
    assert len(range_filter(r, "nope", 0, 1)) == 4

def test_range_filter_active_excludes_nan_and_applies_bounds():
    r = add_fundamental_ratios(RAW)
    assert set(range_filter(r, "PE_Ratio", hi=15)["Ticker"]) == {"C", "D"}
    assert set(range_filter(r, "PE_Ratio", lo=15)["Ticker"]) == {"A"}
    assert set(range_filter(r, "Revenue_Growth_Pct", lo=0)["Ticker"]) == {"A", "D"}

def test_fmt_ratio():
    assert fmt_ratio(12.345, "x") == "12.3x" and fmt_ratio(5, "pct") == "+5.0%" and fmt_ratio(np.nan, "x") == "—"

def test_saved_screens_roundtrip(tmp_path):
    p = str(tmp_path / "s.json")
    assert load_saved_screens(p) == {}
    save_screen("Cheap tech", {"flt_sec": ["Tech"], "pe_max": 15}, p)
    save_screen("Growth", {"rg_min": 10}, p)
    assert set(load_saved_screens(p)) == {"Cheap tech", "Growth"}
    assert load_saved_screens(p)["Cheap tech"]["pe_max"] == 15
    delete_screen("Growth", p)
    assert set(load_saved_screens(p)) == {"Cheap tech"}
    with pytest.raises(ValueError):
        save_screen("  ", {}, p)

def test_saved_screens_corrupt_file(tmp_path):
    p = tmp_path / "s.json"; p.write_text("{not json")
    assert load_saved_screens(str(p)) == {}

def test_sanitize_multiselect():
    assert sanitize_multiselect(["Tech", "Gone"], ["Tech", "Energy"]) == ["Tech"]
    assert sanitize_multiselect("bad", ["Tech"]) is None

def test_peer_table_same_sector_selected_first_with_percentiles():
    r = add_fundamental_ratios(RAW)
    p = peer_table(r, "B", ["PE_Ratio", "ROE_Pct"], size_col="current_market_cap")
    assert list(p["Ticker"]) == ["B", "A", "D"]           # selected first, then by size; no Energy
    assert "ROE_Pct_pctile" in p.columns
    assert p.set_index("Ticker").loc["A", "ROE_Pct_pctile"] == pytest.approx(100 * 2 / 3)

def test_peer_table_unknown_sector_or_ticker():
    r = add_fundamental_ratios(RAW).assign(Sector=["Unknown"] * 4)
    assert peer_table(r, "A", ["PE_Ratio"]).empty
    assert peer_table(add_fundamental_ratios(RAW), "ZZZ", ["PE_Ratio"]).empty

def test_sector_medians():
    r = add_fundamental_ratios(RAW)
    m = sector_medians(r, "Tech", ["Debt_Equity"])
    assert m["Debt_Equity"] == pytest.approx(0.4)
