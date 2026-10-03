import time
from datetime import datetime
import pandas as pd, pytest
import market_refresh as mr
NY = mr.NY
T = lambda *a: datetime(*a, tzinfo=NY)

def test_market_open_hours():
    assert mr.is_market_open(T(2026, 10, 1, 10, 0))          # Thursday
    assert mr.is_market_open(T(2026, 10, 1, 16, 10))         # inside close buffer
    assert not mr.is_market_open(T(2026, 10, 1, 9, 0))
    assert not mr.is_market_open(T(2026, 10, 3, 12, 0))      # Saturday

def test_last_close():
    assert mr.last_close(T(2026, 10, 1, 12, 0)) == T(2026, 9, 30, 16, 15)
    assert mr.last_close(T(2026, 10, 1, 17, 0)) == T(2026, 10, 1, 16, 15)
    assert mr.last_close(T(2026, 10, 4, 12, 0)) == T(2026, 10, 2, 16, 15)   # Sunday -> Friday

def test_refresh_due_market_hours():
    now = T(2026, 10, 1, 11, 0)
    assert mr.refresh_due(None, now)
    assert not mr.refresh_due(T(2026, 10, 1, 10, 50), now)
    assert mr.refresh_due(T(2026, 10, 1, 10, 45), now)

def test_refresh_due_after_close_once():
    assert mr.refresh_due(T(2026, 10, 1, 15, 55), T(2026, 10, 1, 18, 0))      # missed the close
    assert not mr.refresh_due(T(2026, 10, 1, 16, 20), T(2026, 10, 1, 23, 0))  # already have closing prices
    assert not mr.refresh_due(T(2026, 10, 2, 16, 30), T(2026, 10, 4, 12, 0))  # weekend
    assert mr.refresh_due(T(2026, 10, 1, 16, 20), T(2026, 10, 1, 23, 0), market_hours_only=False)

def _snap(rows):
    return pd.DataFrame(rows, columns=["ticker", "current_price", "data_quality_flag", "last_updated"])

def test_merge_keeps_previous_row_for_failures():
    old = _snap([("A", 10.0, "ok", "t0"), ("B", 20.0, "ok", "t0"), ("C", 30.0, "ok", "t0")])
    new = _snap([("A", 11.0, "ok", "t1"), ("B", None, "error", "t1"), ("D", None, "error", "t1")])
    m, failed = mr.merge_snapshot(new, old)
    m = m.set_index("ticker")
    assert failed == 2
    assert m.loc["A", "current_price"] == 11.0 and m.loc["A", "last_updated"] == "t1"
    assert m.loc["B", "current_price"] == 20.0 and m.loc["B", "last_updated"] == "t0"   # kept old
    assert m.loc["C", "current_price"] == 30.0                                          # untouched ticker kept
    assert m.loc["D", "data_quality_flag"] == "error"                                    # no old row to keep
    assert len(m) == 4

def _wait():
    for _ in range(100):
        if not mr.status()["running"]:
            return
        time.sleep(0.02)

def test_background_refresh_writes_and_guards(tmp_path):
    p = str(tmp_path / "snap.csv")
    _snap([("A", 10.0, "ok", "t0"), ("B", 20.0, "ok", "t0")]).to_csv(p, index=False)
    mr._state.update(running=False, started=None)
    good = lambda t: _snap([(x, 99.0, "ok", "t1") for x in t])
    assert mr.start_background_refresh(["A", "B"], good, p, min_gap_seconds=0)
    _wait()
    assert mr.status()["last_result"]["ok"] and set(pd.read_csv(p).current_price) == {99.0}
    # mostly failed run -> previous file kept
    bad = lambda t: _snap([(x, None, "error", "t2") for x in t])
    assert mr.start_background_refresh(["A", "B"], bad, p, min_gap_seconds=0)
    _wait()
    assert not mr.status()["last_result"]["ok"] and set(pd.read_csv(p).current_price) == {99.0}
    # back-off: a second start inside the gap is refused
    assert not mr.start_background_refresh(["A"], good, p, min_gap_seconds=3600)
    # exceptions are captured, not raised
    mr._state.update(started=None)
    def boom(t): raise RuntimeError("yahoo down")
    assert mr.start_background_refresh(["A"], boom, p, min_gap_seconds=0)
    _wait()
    assert "yahoo down" in mr.status()["last_result"]["message"]

def test_snapshot_time_uses_median_row_stamp_not_mtime(tmp_path):
    p = tmp_path / "s.csv"
    _snap([("A", 1.0, "ok", "2026-08-01T10:00:00"), ("B", 1.0, "ok", "2026-08-01T10:05:00"),
           ("C", 1.0, "ok", "2026-10-03T12:00:00"), ("D", None, "error", "2026-10-03T12:00:00")]).to_csv(p, index=False)
    t = mr.snapshot_time(str(p))
    assert t.astimezone(NY).date().isoformat() == "2026-08-01"       # median of ok rows, file mtime ignored

def test_snapshot_time_missing_file():
    assert mr.snapshot_time("/nonexistent/x.csv") is None
