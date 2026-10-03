"""
market_refresh.py
Background auto-refresh of the market snapshot (current_market_data.csv).

- Due every REFRESH_MINUTES while the US market is open, plus once after each close to capture
  closing prices (set MARKET_HOURS_ONLY = False to refresh around the clock).
- Runs in a background thread so the page stays responsive; one refresh at a time per server
  process, shared by all sessions.
- Never degrades the file: tickers that fail keep their previous row, a run where most tickers
  fail is discarded, and the write is atomic.

No Streamlit imports — the app polls status and reloads data when the file changes.
"""
import os
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

REFRESH_MINUTES = 15
MARKET_HOURS_ONLY = True
MAX_FAILURE_SHARE = 0.5           # discard a run if more than this share of tickers failed
NY = ZoneInfo("America/New_York")
_OPEN, _CLOSE_BUFFER = (9, 30), (16, 15)   # close + 15 min so the final prints are captured

_lock = threading.Lock()
_state = {"running": False, "started": None, "finished": None, "last_result": None}


# ── schedule ──────────────────────────────────────────────────────────────────

def _at(d: datetime, hm: tuple) -> datetime:
    return d.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def is_market_open(now: datetime) -> bool:
    now = now.astimezone(NY)
    return now.weekday() < 5 and _at(now, _OPEN) <= now < _at(now, _CLOSE_BUFFER)


def last_close(now: datetime) -> datetime:
    """Most recent weekday close (+buffer) at or before `now` (holidays not modelled)."""
    d = now.astimezone(NY)
    cand = _at(d, _CLOSE_BUFFER)
    while cand > d or cand.weekday() >= 5:
        cand = _at(cand - timedelta(days=1), _CLOSE_BUFFER)
    return cand


def refresh_due(snapshot_time: datetime | None, now: datetime,
                interval_min: int = REFRESH_MINUTES, market_hours_only: bool = MARKET_HOURS_ONLY) -> bool:
    if snapshot_time is None:
        return True
    snapshot_time, now = snapshot_time.astimezone(NY), now.astimezone(NY)
    stale = now - snapshot_time >= timedelta(minutes=interval_min)
    if not market_hours_only or is_market_open(now):
        return stale
    return snapshot_time < last_close(now)      # closed: one refresh after the close, then wait


def snapshot_time(path: str) -> datetime | None:
    """
    When the snapshot's prices were fetched: the MEDIAN of the per-ticker `last_updated` stamps of
    successful rows (file mtime is unreliable — a partial pipeline run rewrites the file while most
    rows stay old; the median also stops one persistently failing ticker forcing constant refreshes).
    Naive stamps are local machine time. Falls back to file mtime.
    """
    try:
        df = pd.read_csv(path, usecols=["last_updated", "data_quality_flag"])
        ts = pd.to_datetime(df.loc[df["data_quality_flag"].eq("ok"), "last_updated"], errors="coerce").dropna()
        if not ts.empty:
            med = ts.sort_values().iloc[len(ts) // 2].to_pydatetime()
            return med.astimezone(NY) if med.tzinfo is None else med.astimezone(NY)
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), tz=NY)
    except OSError:
        return None


def snapshot_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# ── merge + write ─────────────────────────────────────────────────────────────

def merge_snapshot(new: pd.DataFrame, old: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Prefer new rows that succeeded; keep the previous row for tickers that failed this time."""
    ok = new["data_quality_flag"].eq("ok") & new["current_price"].notna()
    failed = int((~ok).sum())
    if old is None or old.empty:
        return new.reset_index(drop=True), failed
    keep_old = old[old["ticker"].isin(new.loc[~ok, "ticker"]) | ~old["ticker"].isin(new["ticker"])]
    merged = pd.concat([new[ok], keep_old], ignore_index=True)
    still_missing = new[~ok & ~new["ticker"].isin(old["ticker"])]
    merged = pd.concat([merged, still_missing], ignore_index=True)
    return merged.sort_values("ticker").reset_index(drop=True), failed


def write_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ── background runner ─────────────────────────────────────────────────────────

def status() -> dict:
    with _lock:
        return dict(_state)


def _run(tickers: list, fetch_fn, path: str) -> None:
    result = {}
    try:
        new = fetch_fn(tickers)
        if new is None or new.empty:
            result = {"ok": False, "message": "no data returned"}
        else:
            old = pd.read_csv(path) if os.path.exists(path) else None
            merged, failed = merge_snapshot(new, old)
            if failed > MAX_FAILURE_SHARE * len(new):
                result = {"ok": False, "message": f"{failed}/{len(new)} tickers failed; kept previous snapshot"}
            else:
                write_atomic(merged, path)
                result = {"ok": True, "message": f"refreshed {len(new) - failed}/{len(new)} tickers"}
    except Exception as exc:                       # never let the thread die silently
        result = {"ok": False, "message": f"error: {exc}"}
    finally:
        with _lock:
            _state.update(running=False, finished=time.time(), last_result=result)


def start_background_refresh(tickers: list, fetch_fn, path: str,
                             min_gap_seconds: float = REFRESH_MINUTES * 60) -> bool:
    """Start a refresh unless one is already running. fetch_fn(tickers) -> DataFrame (not saved)."""
    with _lock:
        if _state["running"]:
            return False
        if _state["started"] and time.time() - _state["started"] < min_gap_seconds:
            return False                           # back off after a failed run instead of retrying every tick
        _state.update(running=True, started=time.time())
    threading.Thread(target=_run, args=(list(tickers), fetch_fn, path), daemon=True,
                     name="market-snapshot-refresh").start()
    return True
