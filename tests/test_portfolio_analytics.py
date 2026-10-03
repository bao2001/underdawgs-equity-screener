import numpy as np, pandas as pd, pytest
from portfolio_analytics import (sector_exposure, concentration_metrics, weight_drift,
                                 risk_metrics, max_drawdown, exposure_notes)

POS = pd.DataFrame({"Ticker": ["AAA", "BBB", "CCC"], "Market Value": [5000.0, 3000.0, 2000.0]})
SECT = {"AAA": "Technology", "BBB": "Technology", "CCC": "Energy"}

def test_sector_weights_include_cash_and_sum_to_100():
    s = sector_exposure(POS, SECT, cash=10000.0)
    assert s["Weight %"].sum() == pytest.approx(100.0)
    assert s.set_index("Sector").loc["Technology", "Weight %"] == pytest.approx(40.0)
    assert s.set_index("Sector").loc["Cash", "Weight %"] == pytest.approx(50.0)

def test_unknown_sector_bucket():
    s = sector_exposure(POS, {"AAA": "Technology"})
    assert "Unknown" in set(s["Sector"])

def test_concentration():
    c = concentration_metrics(POS)
    assert c["top1_pct"] == pytest.approx(50.0) and c["top3_pct"] == pytest.approx(100.0)
    assert c["hhi"] == pytest.approx(0.25 + 0.09 + 0.04)
    assert c["effective_n"] == pytest.approx(1 / 0.38)
    assert c["largest_ticker"] == "AAA"

def test_concentration_equal_weight_effective_n_equals_n():
    p = pd.DataFrame({"Ticker": list("ABCD"), "Market Value": [100.0] * 4})
    assert concentration_metrics(p)["effective_n"] == pytest.approx(4.0)

def test_empty_inputs():
    for empty in (pd.DataFrame(), None):
        assert concentration_metrics(empty)["n_positions"] == 0
        assert sector_exposure(empty, SECT).empty
        assert weight_drift(empty).empty

def test_weight_drift_equal_target_nets_to_zero():
    d = weight_drift(POS)
    assert d["Drift (pts)"].sum() == pytest.approx(0.0)
    assert d["Value to reach target ($)"].sum() == pytest.approx(0.0)
    assert d.set_index("Ticker").loc["AAA", "Drift (pts)"] == pytest.approx(50 - 100 / 3)

def test_weight_drift_custom_target_is_normalised():
    d = weight_drift(POS, {"AAA": 1, "BBB": 1, "CCC": 2})
    assert d["Target %"].sum() == pytest.approx(100.0)
    assert d.set_index("Ticker").loc["CCC", "Target %"] == pytest.approx(50.0)

def test_max_drawdown():
    assert max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.1])) == pytest.approx(0.9 / 1.2 - 1)
    assert max_drawdown(pd.Series(dtype=float)) == 0.0

def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed); idx = pd.bdate_range("2025-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    a = 1.5 * mkt + rng.normal(0, 0.005, n); b = -0.5 * mkt + rng.normal(0, 0.005, n)
    return pd.DataFrame({"SPY": 100 * np.cumprod(1 + mkt), "AAA": 100 * np.cumprod(1 + a),
                         "BBB": 100 * np.cumprod(1 + b)}, index=idx)

def test_risk_metrics_beta_and_diversification():
    px = _prices()
    r = risk_metrics(px, {"AAA": 0.5, "BBB": 0.5})
    assert r["status"] == "ok" and r["n_days"] == len(px) - 1
    assert r["beta"] == pytest.approx(0.5 * 1.5 + 0.5 * -0.5, abs=0.1)   # = 0.5
    assert r["diversification_ratio"] > 1.0                                # anti-correlated names
    assert r["avg_pairwise_corr"] < 0
    assert r["max_drawdown"] <= 0

def test_risk_metrics_single_asset_matches_its_own_vol_and_beta():
    px = _prices(); r = risk_metrics(px, {"AAA": 1.0})
    own = px["AAA"].pct_change().dropna().std() * np.sqrt(252)
    assert r["ann_vol"] == pytest.approx(own, rel=1e-6)
    assert r["diversification_ratio"] == pytest.approx(1.0)

def test_risk_metrics_missing_and_short_history():
    px = _prices()
    assert risk_metrics(px, {"ZZZ": 1.0})["status"] == "no_price_data"
    r = risk_metrics(px, {"AAA": 0.5, "ZZZ": 0.5}); assert r["tickers_missing"] == ["ZZZ"]
    assert risk_metrics(px.iloc[:20], {"AAA": 1.0})["status"] == "no_price_data"

def test_exposure_notes():
    c = concentration_metrics(POS); s = sector_exposure(POS, SECT)
    n = " ".join(exposure_notes(s, c))
    assert "AAA is 50%" in n and "Technology makes up 80%" in n
