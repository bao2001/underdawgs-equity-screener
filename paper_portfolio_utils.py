"""
paper_portfolio_utils.py
Paper Portfolio Simulator — persistence, trade execution, and portfolio analytics.

All trades are simulated. No real trades are placed. No brokerage APIs are used.
Prices may be delayed, missing, or estimated. No tax or dividend modeling.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

PAPER_TRANSACTIONS_CSV = os.path.join(_HERE, "paper_transactions.csv")
PAPER_SNAPSHOT_CSV     = os.path.join(_HERE, "paper_portfolio_snapshot.csv")
PAPER_HISTORY_CSV      = os.path.join(_HERE, "paper_portfolio_history.csv")

DEFAULT_STARTING_CASH  = 10_000.0

ACTION_BUY  = "Add to paper portfolio"
ACTION_SELL = "Reduce position"

# 23-column schema. Backward-compatible loader handles old 15-col files.
TRANSACTION_COLS = [
    "timestamp", "ticker", "company_name", "action", "quantity",
    "reference_price", "trade_price", "price_source", "manual_override_flag",
    "trade_value", "cash_after_trade",
    "signal_at_trade", "valuation_gap_at_trade", "quality_score_at_trade",
    "report_risk_score_at_trade", "output_quality_flag_at_trade",
    "output_warning_text_at_trade",
    "realized_pl", "realized_pl_pct", "remaining_quantity", "remaining_cost_basis",
    "notes",
]

_DEFAULT_PORTFOLIO = {
    "starting_cash": DEFAULT_STARTING_CASH,
    "current_cash":  DEFAULT_STARTING_CASH,
    "created_at":    "",
    "last_updated":  "",
    "initialized":   False,
}

_MANUAL_OVERRIDE_ABS_THRESHOLD = 0.01    # $0.01 absolute difference
_MANUAL_OVERRIDE_PCT_THRESHOLD = 0.001   # 0.1% relative difference


# ── Persistence ───────────────────────────────────────────────────────────────

def load_paper_portfolio() -> dict:
    if not os.path.exists(PAPER_SNAPSHOT_CSV):
        return dict(_DEFAULT_PORTFOLIO)
    try:
        df = pd.read_csv(PAPER_SNAPSHOT_CSV)
        if df.empty:
            return dict(_DEFAULT_PORTFOLIO)
        row = df.iloc[0]
        return {
            "starting_cash": float(row.get("starting_cash", DEFAULT_STARTING_CASH)),
            "current_cash":  float(row.get("current_cash",  DEFAULT_STARTING_CASH)),
            "created_at":    str(row.get("created_at",  "")),
            "last_updated":  str(row.get("last_updated", "")),
            "initialized":   True,
        }
    except Exception:
        return dict(_DEFAULT_PORTFOLIO)


def save_paper_portfolio(portfolio: dict) -> None:
    ts = pd.Timestamp.now().isoformat(timespec="seconds")
    pd.DataFrame([{
        "starting_cash": portfolio.get("starting_cash", DEFAULT_STARTING_CASH),
        "current_cash":  portfolio.get("current_cash",  DEFAULT_STARTING_CASH),
        "created_at":    portfolio.get("created_at", ts),
        "last_updated":  ts,
    }]).to_csv(PAPER_SNAPSHOT_CSV, index=False)


def load_paper_transactions() -> pd.DataFrame:
    """Load all simulated transactions with backward-compatible column migration."""
    if not os.path.exists(PAPER_TRANSACTIONS_CSV):
        return pd.DataFrame(columns=TRANSACTION_COLS)
    try:
        df = pd.read_csv(PAPER_TRANSACTIONS_CSV)
        # Migrate old column names
        if "price" in df.columns and "trade_price" not in df.columns:
            df = df.rename(columns={"price": "trade_price"})
        if "data_source" in df.columns and "price_source" not in df.columns:
            df = df.rename(columns={"data_source": "price_source"})
        # Fill any missing columns
        for col in TRANSACTION_COLS:
            if col not in df.columns:
                df[col] = ""
        return df[TRANSACTION_COLS].copy()
    except Exception:
        return pd.DataFrame(columns=TRANSACTION_COLS)


def save_paper_transactions(df: pd.DataFrame) -> None:
    df.to_csv(PAPER_TRANSACTIONS_CSV, index=False)


def load_portfolio_history() -> pd.DataFrame:
    """Load portfolio history. Supports both old (date-only) and new (timestamp) schemas."""
    if not os.path.exists(PAPER_HISTORY_CSV):
        return pd.DataFrame(columns=["timestamp", "date", "portfolio_value",
                                     "cash", "invested_value"])
    try:
        df = pd.read_csv(PAPER_HISTORY_CSV)
        # Backward compat: old files only have "date"
        if "timestamp" not in df.columns:
            df["timestamp"] = df.get("date", "")
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "date", "portfolio_value",
                                     "cash", "invested_value"])


def save_portfolio_history(df: pd.DataFrame) -> None:
    df.to_csv(PAPER_HISTORY_CSV, index=False)


def initialize_paper_account(starting_cash: float = DEFAULT_STARTING_CASH) -> dict:
    """Initialize or reset the paper account. Clears transactions, history, and snapshot."""
    ts = pd.Timestamp.now().isoformat(timespec="seconds")
    portfolio = {
        "starting_cash": round(starting_cash, 2),
        "current_cash":  round(starting_cash, 2),
        "created_at":    ts,
        "last_updated":  ts,
        "initialized":   True,
    }
    save_paper_portfolio(portfolio)
    pd.DataFrame(columns=TRANSACTION_COLS).to_csv(PAPER_TRANSACTIONS_CSV, index=False)
    pd.DataFrame(columns=["timestamp", "date", "portfolio_value",
                           "cash", "invested_value"]).to_csv(PAPER_HISTORY_CSV, index=False)
    return portfolio


# ── Ticker and price helpers ──────────────────────────────────────────────────

def get_available_tickers(mo_df=None, sample_df=None) -> list:
    tickers = set()
    if mo_df is not None and "Ticker" in mo_df.columns:
        tickers.update(mo_df["Ticker"].dropna().str.upper().tolist())
    if sample_df is not None and "Ticker" in sample_df.columns:
        tickers.update(sample_df["Ticker"].dropna().str.upper().tolist())
    return sorted(tickers)


def get_trade_price(ticker: str, cm_df=None, sample_df=None) -> tuple:
    """
    Return (price, source_str). Priority: current_market_data.csv → sample_data.csv → (None, "manual").
    """
    ticker = ticker.upper()
    if cm_df is not None and not cm_df.empty:
        col = "ticker" if "ticker" in cm_df.columns else None
        if col:
            m = cm_df[cm_df[col].str.upper() == ticker]
            if len(m) > 0:
                p = m.iloc[0].get("current_price")
                if p is not None and not pd.isna(p) and float(p) > 0:
                    return float(p), "live (current_market_data.csv)"

    if sample_df is not None and not sample_df.empty and "Ticker" in sample_df.columns:
        m = sample_df[sample_df["Ticker"].str.upper() == ticker]
        if len(m) > 0:
            p = m.iloc[0].get("Current_Price")
            if p is not None and not pd.isna(p) and float(p) > 0:
                return float(p), "sample_data.csv (may be dated)"

    return None, "manual entry required"


def get_company_name(ticker: str, mo_df=None, sample_df=None) -> str:
    ticker = ticker.upper()
    if mo_df is not None and "Ticker" in mo_df.columns:
        m = mo_df[mo_df["Ticker"].str.upper() == ticker]
        if len(m) > 0:
            return str(m.iloc[0].get("Company_Name", ticker))
    if sample_df is not None and "Ticker" in sample_df.columns:
        m = sample_df[sample_df["Ticker"].str.upper() == ticker]
        if len(m) > 0:
            return str(m.iloc[0].get("Company_Name", ticker))
    return ticker


def get_signal_metadata(ticker: str, mo_df=None, sample_df=None) -> dict:
    ticker = ticker.upper()
    meta = {
        "signal": "", "valuation_gap": None, "quality_score": None,
        "report_risk": None, "quality_flag": "", "warning_text": "",
    }
    if mo_df is not None and "Ticker" in mo_df.columns:
        m = mo_df[mo_df["Ticker"].str.upper() == ticker]
        if len(m) > 0:
            row = m.iloc[0]
            meta["signal"]        = str(row.get("final_signal_display") or row.get("Final_Signal", ""))
            meta["valuation_gap"] = row.get("Valuation_Gap_Pct")
            meta["quality_score"] = row.get("Quality_Score")
            meta["report_risk"]   = row.get("Report_Risk_Score")
            meta["quality_flag"]  = str(row.get("output_quality_flag", ""))
            meta["warning_text"]  = str(row.get("output_warning_text", ""))
            return meta

    if sample_df is not None and "Ticker" in sample_df.columns:
        m = sample_df[sample_df["Ticker"].str.upper() == ticker]
        if len(m) > 0:
            row = m.iloc[0]
            meta["signal"]        = str(row.get("Final_Signal", ""))
            meta["valuation_gap"] = row.get("Valuation_Gap_Pct")
            meta["quality_score"] = row.get("Quality_Score")
            meta["report_risk"]   = row.get("Report_Risk_Score")
            meta["quality_flag"]  = "ok"
            return meta
    return meta


def detect_manual_override(trade_price: float, reference_price) -> bool:
    """Return True if trade_price differs from reference_price beyond thresholds."""
    if reference_price is None or pd.isna(reference_price):
        return False
    ref = float(reference_price)
    if ref <= 0:
        return False
    abs_diff = abs(trade_price - ref)
    pct_diff = abs_diff / ref
    return abs_diff > _MANUAL_OVERRIDE_ABS_THRESHOLD or pct_diff > _MANUAL_OVERRIDE_PCT_THRESHOLD


# ── Trade execution ───────────────────────────────────────────────────────────

def execute_paper_trade(
    portfolio:       dict,
    transactions:    pd.DataFrame,
    ticker:          str,
    company_name:    str,
    action:          str,
    quantity:        float,
    trade_price:     float,
    reference_price  = None,
    price_source:    str = "manual",
    notes:           str = "",
    mo_df            = None,
    sample_df        = None,
) -> tuple:
    """
    Validate and execute a simulated trade.
    Returns (updated_portfolio, updated_transactions, error_msg).
    error_msg is empty string on success.
    """
    ticker      = ticker.upper().strip()
    quantity    = float(quantity)
    trade_price = float(trade_price)
    trade_val   = round(quantity * trade_price, 4)

    if not ticker:
        return portfolio, transactions, "Ticker cannot be empty."
    if quantity <= 0:
        return portfolio, transactions, "Quantity must be positive."
    if trade_price <= 0:
        return portfolio, transactions, "Price must be positive."

    current_cash = float(portfolio.get("current_cash", 0))
    holdings     = calculate_holdings(transactions)
    open_pos     = holdings[holdings["is_open"]] if not holdings.empty else pd.DataFrame()

    realized_pl          = None
    realized_pl_pct      = None
    remaining_quantity   = None
    remaining_cost_basis = None

    if action == ACTION_BUY:
        if trade_val > current_cash + 0.01:
            return portfolio, transactions, (
                f"Insufficient cash. Trade value ${trade_val:,.2f} exceeds "
                f"available cash ${current_cash:,.2f}."
            )
        new_cash = current_cash - trade_val

    elif action == ACTION_SELL:
        held_row = open_pos[open_pos["ticker"] == ticker] if not open_pos.empty else pd.DataFrame()
        if held_row.empty:
            return portfolio, transactions, f"No open position in {ticker} to reduce."
        held_qty  = float(held_row.iloc[0]["quantity"])
        avg_cost  = float(held_row.iloc[0]["avg_cost"])
        if quantity > held_qty + 0.0001:
            return portfolio, transactions, (
                f"Cannot reduce by {quantity:.4g} shares — only {held_qty:.4g} held."
            )
        realized_pl          = round((trade_price - avg_cost) * quantity, 4)
        realized_pl_pct      = round((trade_price - avg_cost) / avg_cost * 100, 4) if avg_cost > 0 else 0.0
        remaining_quantity   = round(held_qty - quantity, 4)
        remaining_cost_basis = round(avg_cost * remaining_quantity, 4)
        new_cash             = current_cash + trade_val

    else:
        return portfolio, transactions, f"Unknown action: {action}"

    is_override  = detect_manual_override(trade_price, reference_price)
    final_source = f"{price_source} (manual override)" if is_override else price_source
    meta         = get_signal_metadata(ticker, mo_df=mo_df, sample_df=sample_df)
    ts           = pd.Timestamp.now().isoformat(timespec="seconds")

    new_row = pd.DataFrame([{
        "timestamp":                    ts,
        "ticker":                       ticker,
        "company_name":                 company_name,
        "action":                       action,
        "quantity":                     quantity,
        "reference_price":              reference_price if reference_price is not None else "",
        "trade_price":                  trade_price,
        "price_source":                 final_source,
        "manual_override_flag":         is_override,
        "trade_value":                  trade_val,
        "cash_after_trade":             round(new_cash, 4),
        "signal_at_trade":              meta["signal"],
        "valuation_gap_at_trade":       meta["valuation_gap"],
        "quality_score_at_trade":       meta["quality_score"],
        "report_risk_score_at_trade":   meta["report_risk"],
        "output_quality_flag_at_trade": meta["quality_flag"],
        "output_warning_text_at_trade": meta["warning_text"],
        "realized_pl":                  realized_pl if realized_pl is not None else "",
        "realized_pl_pct":              realized_pl_pct if realized_pl_pct is not None else "",
        "remaining_quantity":           remaining_quantity if remaining_quantity is not None else "",
        "remaining_cost_basis":         remaining_cost_basis if remaining_cost_basis is not None else "",
        "notes":                        notes,
    }])[TRANSACTION_COLS]

    updated_txns                      = pd.concat([transactions, new_row], ignore_index=True)
    updated_portfolio                 = dict(portfolio)
    updated_portfolio["current_cash"] = round(new_cash, 4)

    save_paper_transactions(updated_txns)
    save_paper_portfolio(updated_portfolio)

    return updated_portfolio, updated_txns, ""


# ── Holdings and portfolio analytics ─────────────────────────────────────────

def calculate_holdings(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker position state using sequential average-cost accounting.

    Returns ALL tickers (open + closed) with columns:
      ticker, company_name, quantity, avg_cost, total_cost_basis,
      last_trade_price, realized_pl, is_open

    Callers should filter `is_open == True` for open positions.
    """
    _empty_cols = ["ticker", "company_name", "quantity", "avg_cost",
                   "total_cost_basis", "last_trade_price", "realized_pl", "is_open"]
    if transactions.empty:
        return pd.DataFrame(columns=_empty_cols)

    # Support both old "price" column and new "trade_price"
    price_col = "trade_price" if "trade_price" in transactions.columns else "price"

    rows = []
    for ticker, grp in transactions.groupby("ticker"):
        grp          = grp.sort_values("timestamp").reset_index(drop=True)
        qty          = 0.0
        cost_basis   = 0.0
        realized_pl  = 0.0
        last_price   = 0.0
        company_name = str(grp["company_name"].dropna().iloc[-1]) if len(grp) > 0 else ticker

        for _, tx in grp.iterrows():
            tx_qty   = float(tx.get("quantity", 0) or 0)
            tx_price = float(tx.get(price_col, 0) or 0)
            last_price = tx_price

            if tx["action"] == ACTION_BUY:
                cost_basis += tx_qty * tx_price
                qty        += tx_qty

            elif tx["action"] == ACTION_SELL and qty > 0.0001:
                avg_cost     = cost_basis / qty
                pl           = (tx_price - avg_cost) * tx_qty
                realized_pl += pl
                reduce_basis = avg_cost * tx_qty
                cost_basis   = max(0.0, cost_basis - reduce_basis)
                qty          = max(0.0, qty - tx_qty)
                if qty < 0.0001:
                    qty        = 0.0
                    cost_basis = 0.0

        avg_cost = cost_basis / qty if qty > 0.0001 else 0.0
        rows.append({
            "ticker":           ticker,
            "company_name":     company_name,
            "quantity":         round(qty, 4),
            "avg_cost":         round(avg_cost, 4),
            "total_cost_basis": round(cost_basis, 4),
            "last_trade_price": round(last_price, 4),
            "realized_pl":      round(realized_pl, 4),
            "is_open":          qty > 0.0001,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_empty_cols)


def calculate_realized_summary(holdings: pd.DataFrame) -> dict:
    """Aggregate realized P/L across all tickers (open + closed)."""
    if holdings.empty or "realized_pl" not in holdings.columns:
        return {"total_realized_pl": 0.0, "closed_positions": 0}
    closed = holdings[~holdings["is_open"]] if "is_open" in holdings.columns else pd.DataFrame()
    return {
        "total_realized_pl": round(float(holdings["realized_pl"].fillna(0).sum()), 2),
        "closed_positions":  int((~holdings["is_open"]).sum()) if "is_open" in holdings.columns else 0,
    }


def calculate_portfolio_value(holdings: pd.DataFrame, cash: float,
                               cm_df=None, sample_df=None) -> tuple:
    """
    Returns (total_value, invested_value, price_map).
    price_map: {ticker: (price, source_str)}.
    """
    open_pos = holdings[holdings["is_open"]] if not holdings.empty and "is_open" in holdings.columns else holdings
    if open_pos.empty:
        return round(cash, 2), 0.0, {}

    invested  = 0.0
    price_map = {}
    for _, row in open_pos.iterrows():
        ticker   = row["ticker"]
        qty      = float(row["quantity"])
        fallback = float(row.get("last_trade_price") or row.get("avg_cost", 0))
        live_p, live_src = get_trade_price(ticker, cm_df=cm_df, sample_df=sample_df)
        price, src       = (live_p, live_src) if live_p is not None else (fallback, "last trade price (fallback)")
        invested         += qty * price
        price_map[ticker] = (price, src)

    return round(cash + invested, 2), round(invested, 2), price_map


def calculate_position_metrics(holdings: pd.DataFrame, price_map: dict,
                                mo_df=None, sample_df=None,
                                total_portfolio_value: float = 0.0) -> pd.DataFrame:
    """
    Returns display DataFrame for open positions with full P/L and signal metadata.
    """
    open_pos = holdings[holdings["is_open"]] if not holdings.empty and "is_open" in holdings.columns else holdings
    if open_pos.empty:
        return pd.DataFrame()

    rows = []
    for _, h in open_pos.iterrows():
        ticker      = h["ticker"]
        qty         = float(h["quantity"])
        avg_cost    = float(h["avg_cost"])
        cost_basis  = float(h["total_cost_basis"])
        realized_pl = float(h.get("realized_pl", 0) or 0)
        price, src  = price_map.get(ticker, (avg_cost, "last trade price"))
        market_val  = qty * price
        unreal_pl   = market_val - cost_basis
        unreal_pct  = (unreal_pl / cost_basis * 100) if cost_basis > 0 else 0.0
        total_pl    = unreal_pl + realized_pl
        weight_pct  = (market_val / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
        meta        = get_signal_metadata(ticker, mo_df=mo_df, sample_df=sample_df)

        rows.append({
            "Ticker":           ticker,
            "Company":          h["company_name"],
            "Qty":              qty,
            "Avg Cost":         avg_cost,
            "Price":            price,
            "Price Source":     src,
            "Market Value":     round(market_val, 2),
            "Cost Basis":       round(cost_basis, 2),
            "Unrealized P/L":   round(unreal_pl, 2),
            "Unrealized P/L %": round(unreal_pct, 2),
            "Realized P/L":     round(realized_pl, 2),
            "Total P/L":        round(total_pl, 2),
            "Weight %":         round(weight_pct, 1),
            "Signal":           meta["signal"],
            "Val. Gap %":       meta["valuation_gap"],
            "Data Quality":     meta["quality_flag"],
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def calculate_portfolio_metrics(total_value: float, invested_value: float,
                                 portfolio: dict,
                                 holdings: pd.DataFrame = None) -> dict:
    """Compute full summary statistics for the portfolio analytics section."""
    starting    = float(portfolio.get("starting_cash", DEFAULT_STARTING_CASH))
    cash        = float(portfolio.get("current_cash",  starting))
    unrealized  = total_value - cash - (invested_value - (total_value - cash - (total_value - cash - invested_value)))

    # Simpler: unrealized = invested_value - cost_basis_of_open_positions
    # We compute it as: total_value = cash + market_value_of_positions
    # unrealized_pl = (total_value - cash) - cost_basis_of_open_positions
    market_value = total_value - cash
    cost_basis_open = 0.0
    realized_pl     = 0.0
    num_positions   = 0
    largest_weight  = 0.0

    if holdings is not None and not holdings.empty:
        open_pos = holdings[holdings["is_open"]] if "is_open" in holdings.columns else holdings
        if not open_pos.empty:
            cost_basis_open = float(open_pos["total_cost_basis"].fillna(0).sum())
            num_positions   = len(open_pos)
            if total_value > 0:
                pos_weights = (open_pos["total_cost_basis"].fillna(0) / total_value * 100)
                largest_weight = float(pos_weights.max()) if len(pos_weights) > 0 else 0.0
        realized_pl = float(holdings["realized_pl"].fillna(0).sum())

    unrealized_pl = market_value - cost_basis_open
    total_pl      = unrealized_pl + realized_pl
    total_return  = (total_value - starting) / starting * 100 if starting > 0 else 0.0
    exposure_pct  = (market_value / total_value * 100) if total_value > 0 else 0.0

    return {
        "starting_cash":         starting,
        "current_cash":          cash,
        "invested_value":        round(market_value, 2),
        "total_value":           round(total_value, 2),
        "unrealized_pl":         round(unrealized_pl, 2),
        "realized_pl":           round(realized_pl, 2),
        "total_pl":              round(total_pl, 2),
        "total_return_pct":      round(total_return, 2),
        "exposure_pct":          round(exposure_pct, 1),
        "num_positions":         num_positions,
        "largest_position_pct":  round(largest_weight, 1),
        "concentration_warning": largest_weight > 40.0,
    }


# ── Portfolio history ─────────────────────────────────────────────────────────

def update_portfolio_history(portfolio_value: float, cash: float,
                              invested_value: float) -> pd.DataFrame:
    """Append a timestamped snapshot. Also stores date for backward compat."""
    ts   = pd.Timestamp.now().isoformat(timespec="seconds")
    date = ts[:10]
    hist = load_portfolio_history()

    new_row = pd.DataFrame([{
        "timestamp":       ts,
        "date":            date,
        "portfolio_value": round(portfolio_value, 2),
        "cash":            round(cash, 2),
        "invested_value":  round(invested_value, 2),
    }])
    updated = pd.concat([hist, new_row], ignore_index=True).reset_index(drop=True)
    save_portfolio_history(updated)
    return updated


def generate_portfolio_history(transactions: pd.DataFrame,
                                portfolio: dict) -> pd.DataFrame:
    """
    Synthetic history from transaction timestamps for charting when real history is sparse.
    Values are conservative (cash-only; invested is not re-estimated).
    """
    if transactions.empty:
        return pd.DataFrame(columns=["timestamp", "date", "portfolio_value", "estimated"])

    starting = float(portfolio.get("starting_cash", DEFAULT_STARTING_CASH))
    ts_col   = "timestamp"
    price_col = "trade_price" if "trade_price" in transactions.columns else "price"

    created_ts = portfolio.get("created_at", "")
    rows = [{
        "timestamp":       created_ts,
        "date":            created_ts[:10] if created_ts else "",
        "portfolio_value": starting,
        "estimated":       True,
    }]

    cash = starting
    for _, tx in transactions.sort_values(ts_col).iterrows():
        cash = float(tx.get("cash_after_trade", cash) or cash)
        ts_val = str(tx.get(ts_col, ""))
        rows.append({
            "timestamp":       ts_val,
            "date":            ts_val[:10],
            "portfolio_value": cash,
            "estimated":       True,
        })

    df = (pd.DataFrame(rows)
          .drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True))
    return df


# ── Benchmark comparison ──────────────────────────────────────────────────────

def compare_to_benchmark(start_date: str, starting_cash: float,
                          benchmark: str = "SPY") -> tuple:
    """
    Fetch benchmark close prices since start_date via yfinance, normalized to starting_cash.
    Returns (bm_df, status_msg). bm_df columns: [Date, benchmark_value].
    On failure returns (None, reason_string).
    """
    if not start_date or start_date == "nan":
        return None, "No account start date available."
    try:
        import yfinance as yf
        raw = yf.download(benchmark, start=start_date,
                          auto_adjust=True, progress=False)
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return None, f"No {benchmark} data available since {start_date}."
        close = raw["Close"].squeeze()
        if len(close) == 0:
            return None, f"No {benchmark} price rows returned."
        start_p = float(close.iloc[0])
        if start_p <= 0:
            return None, f"{benchmark} start price is zero or negative."
        bm_df = pd.DataFrame({
            "Date":            pd.to_datetime(close.index),
            "benchmark_value": (close.values.astype(float) / start_p * starting_cash).round(2),
        })
        total_ret = (float(bm_df["benchmark_value"].iloc[-1]) - starting_cash) / starting_cash * 100
        return bm_df, f"{benchmark}: {total_ret:+.1f}% since {start_date}"
    except ImportError:
        return None, "yfinance not installed — cannot fetch benchmark."
    except Exception as exc:
        return None, f"Could not fetch {benchmark}: {exc}"


# ── Portfolio report generation ───────────────────────────────────────────────

def generate_portfolio_report(
    portfolio:    dict,
    holdings:     pd.DataFrame,
    metrics:      dict,
    price_map:    dict,
    bm_msg:       str = "",
    mo_df         = None,
    sample_df     = None,
) -> str:
    """
    Generate a plain-text / markdown portfolio report for download.
    Returns a UTF-8 string.
    """
    ts      = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines   = [
        "# Paper Portfolio Report",
        f"Generated: {ts}",
        "",
        "> SIMULATED ONLY. No real trades. No real money. Not investment advice.",
        "> Prices may be delayed, missing, or estimated from historical data.",
        "> No tax, dividend, or slippage modeling.",
        "",
        "## Account Summary",
        f"- Starting cash:    ${metrics['starting_cash']:,.2f}",
        f"- Current cash:     ${metrics['current_cash']:,.2f}",
        f"- Invested value:   ${metrics['invested_value']:,.2f}",
        f"- Total value:      ${metrics['total_value']:,.2f}",
        f"- Total return:     {metrics['total_return_pct']:+.2f}%",
        f"- Unrealized P/L:   ${metrics['unrealized_pl']:+,.2f}",
        f"- Realized P/L:     ${metrics['realized_pl']:+,.2f}",
        f"- Total P/L:        ${metrics['total_pl']:+,.2f}",
        f"- Equity exposure:  {metrics['exposure_pct']:.1f}%",
        f"- Open positions:   {metrics['num_positions']}",
    ]
    if metrics.get("concentration_warning"):
        lines.append(
            f"- WARNING: Largest position is {metrics['largest_position_pct']:.1f}% of portfolio (>40%)."
        )
    if bm_msg:
        lines += ["", f"## Benchmark (SPY)", f"- {bm_msg}",
                  "- Benchmark comparison is context, not proof of future performance."]

    lines += ["", "## Open Positions"]
    open_pos = holdings[holdings["is_open"]] if not holdings.empty and "is_open" in holdings.columns else holdings
    if open_pos.empty:
        lines.append("No open positions.")
    else:
        for _, h in open_pos.iterrows():
            ticker      = h["ticker"]
            price, src  = price_map.get(ticker, (h.get("avg_cost", 0), "fallback"))
            market_val  = float(h["quantity"]) * price
            cost_basis  = float(h["total_cost_basis"])
            unreal_pl   = market_val - cost_basis
            meta        = get_signal_metadata(ticker, mo_df=mo_df, sample_df=sample_df)
            flag        = meta["quality_flag"]
            lines.append(
                f"  {ticker:6s}  qty={h['quantity']:.4g}  avg_cost=${h['avg_cost']:.2f}"
                f"  price=${price:.2f} ({src[:20]})"
                f"  mkt=${market_val:,.2f}"
                f"  unreal_pl=${unreal_pl:+,.2f}"
                f"  signal={meta['signal']}"
                + (f"  FLAG={flag}" if flag not in ("ok", "") else "")
            )

    closed = holdings[~holdings["is_open"]] if not holdings.empty and "is_open" in holdings.columns else pd.DataFrame()
    if not closed.empty:
        lines += ["", "## Closed Positions (Realized P/L)"]
        for _, h in closed.iterrows():
            lines.append(
                f"  {h['ticker']:6s}  realized_pl=${h.get('realized_pl', 0):+,.2f}"
            )

    lines += [
        "",
        "## Limitations",
        "- This is a simulated paper portfolio. No real trades were placed.",
        "- Prices are sourced from current_market_data.csv, sample_data.csv, or last trade prices.",
        "- No brokerage connection, no real account, no real money.",
        "- No taxes, dividends, or market slippage are modeled.",
        "- Model signals are research outputs, not investment advice.",
        "- Benchmark comparison uses normalized SPY prices, not actual portfolio performance.",
    ]
    return "\n".join(lines)


# ── Historical Replay ─────────────────────────────────────────────────────────

# Fallback symbols tried in order if the primary ticker returns no data.
_TICKER_ALIASES: dict = {
    "GOOGL": ["GOOG"],
    "BRK.B": ["BRK-B"],
}


def _strip_tz(index):
    """Return a tz-naive copy of a DatetimeIndex."""
    if getattr(index, "tz", None) is not None:
        return index.tz_convert(None)
    return index


def _get_close_series(hist: pd.DataFrame):
    """
    Extract a close-price Series from a yfinance history DataFrame.
    Prefers 'Adj Close' (split/dividend adjusted) then 'Close'.
    Returns None if neither column is present.
    """
    for col in ("Adj Close", "Close", "adj close", "close"):
        if col in hist.columns:
            return hist[col].dropna()
    return None


def _fetch_single_ticker(
    ticker: str,
    entry_start: pd.Timestamp,
    target_exit: pd.Timestamp,
    yf,
) -> dict:
    """
    Download history for one ticker and extract entry/exit prices.

    entry = first available close on or AFTER entry_start
    exit  = last  available close on or BEFORE target_exit (capped at today)

    Returns a dict with keys:
        status, reason, rows, entry_price, exit_price,
        entry_date, exit_date, exit_capped
    """
    today       = pd.Timestamp.today().normalize()
    capped_exit = min(target_exit, today)
    exit_capped = target_exit > today
    fetch_end   = capped_exit + pd.DateOffset(days=15)

    row = {
        "status":       "error",
        "reason":       "",
        "rows":         0,
        "entry_price":  None,
        "entry_date":   None,
        "exit_price":   None,
        "exit_date":    None,
        "exit_capped":  exit_capped,
    }

    try:
        hist = yf.Ticker(ticker).history(
            start=entry_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if hist.empty:
            row["reason"] = "yfinance returned empty DataFrame"
            row["status"] = "no_data"
            return row

        # Normalise tz-aware index to tz-naive (yfinance returns America/New_York)
        hist.index = _strip_tz(hist.index)

        closes = _get_close_series(hist)
        if closes is None:
            row["reason"] = f"No Close column found; columns={list(hist.columns)}"
            return row

        row["rows"] = int(len(closes))
        if row["rows"] == 0:
            row["reason"] = "All close prices are NaN"
            row["status"] = "no_data"
            return row

        # Entry: first close on or after entry_start
        entry_cands = closes[closes.index >= entry_start]
        if entry_cands.empty:
            row["reason"] = f"No data on or after {entry_start.date()}"
            row["status"] = "no_entry_price"
            return row
        row["entry_price"] = float(entry_cands.iloc[0])
        row["entry_date"]  = str(entry_cands.index[0])[:10]

        # Exit: LAST close on or before capped_exit
        exit_cands = closes[closes.index <= capped_exit]
        if exit_cands.empty:
            row["reason"] = f"No data on or before {capped_exit.date()}"
            row["status"] = "no_exit_price"
            return row
        row["exit_price"] = float(exit_cands.iloc[-1])
        row["exit_date"]  = str(exit_cands.index[-1])[:10]
        row["reason"]     = "ok" + (" (exit capped at today)" if exit_capped else "")
        row["status"]     = "ok"

    except Exception as exc:
        row["reason"] = str(exc)

    return row


def fetch_replay_prices(tickers: list, start_year: int, holding_months: int) -> dict:
    """
    Fetch entry and exit prices for a historical replay backtest.

    For each ticker, tries the primary symbol first; if that fails, tries aliases
    from _TICKER_ALIASES.  Returns a dict keyed by original ticker symbol.

    Each value is a dict with:
        status       — "ok" | "no_data" | "no_entry_price" | "no_exit_price" | "error"
        reason       — human-readable explanation
        rows         — number of price rows returned by yfinance
        entry_price  — float (or None)
        entry_date   — "YYYY-MM-DD" (or None)
        exit_price   — float (or None)
        exit_date    — "YYYY-MM-DD" (or None)
        exit_capped  — True if target exit was beyond today
        ticker_used  — actual symbol that succeeded (may differ from key on alias)
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            t: {
                "status": "error", "reason": "yfinance not installed",
                "rows": 0, "entry_price": None, "entry_date": None,
                "exit_price": None, "exit_date": None,
                "exit_capped": False, "ticker_used": t,
            }
            for t in tickers
        }

    entry_start  = pd.Timestamp(f"{start_year}-01-01")
    target_exit  = entry_start + pd.DateOffset(months=holding_months)

    result = {}
    for ticker in tickers:
        candidates = [ticker] + _TICKER_ALIASES.get(ticker, [])
        best = None
        for candidate in candidates:
            row = _fetch_single_ticker(candidate, entry_start, target_exit, yf)
            row["ticker_used"] = candidate
            if row["status"] == "ok":
                best = row
                break
            if best is None:
                best = row
        result[ticker] = best

    return result


def _fetch_single_ticker_daily(
    ticker: str,
    entry_start: pd.Timestamp,
    target_exit: pd.Timestamp,
    yf,
) -> dict:
    """
    Like _fetch_single_ticker but also returns a daily_series of close prices
    between the actual entry and exit dates (tz-naive, for chart use).
    """
    today       = pd.Timestamp.today().normalize()
    capped_exit = min(target_exit, today)
    exit_capped = target_exit > today
    fetch_end   = capped_exit + pd.DateOffset(days=15)

    row = {
        "status":       "error",
        "reason":       "",
        "rows":         0,
        "entry_price":  None,
        "entry_date":   None,
        "exit_price":   None,
        "exit_date":    None,
        "exit_capped":  exit_capped,
        "daily_series": None,
    }

    try:
        hist = yf.Ticker(ticker).history(
            start=entry_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if hist.empty:
            row["reason"] = "yfinance returned empty DataFrame"
            row["status"] = "no_data"
            return row

        hist.index = _strip_tz(hist.index)
        closes = _get_close_series(hist)
        if closes is None:
            row["reason"] = f"No Close column found; columns={list(hist.columns)}"
            return row

        row["rows"] = int(len(closes))
        if row["rows"] == 0:
            row["reason"] = "All close prices are NaN"
            row["status"] = "no_data"
            return row

        entry_cands = closes[closes.index >= entry_start]
        if entry_cands.empty:
            row["reason"] = f"No data on or after {entry_start.date()}"
            row["status"] = "no_entry_price"
            return row
        row["entry_price"] = float(entry_cands.iloc[0])
        row["entry_date"]  = str(entry_cands.index[0])[:10]
        entry_ts           = entry_cands.index[0]

        exit_cands = closes[closes.index <= capped_exit]
        if exit_cands.empty:
            row["reason"] = f"No data on or before {capped_exit.date()}"
            row["status"] = "no_exit_price"
            return row
        row["exit_price"] = float(exit_cands.iloc[-1])
        row["exit_date"]  = str(exit_cands.index[-1])[:10]
        exit_ts            = exit_cands.index[-1]

        row["daily_series"] = closes[
            (closes.index >= entry_ts) & (closes.index <= exit_ts)
        ]
        row["reason"] = "ok" + (" (exit capped at today)" if exit_capped else "")
        row["status"] = "ok"

    except Exception as exc:
        row["reason"] = str(exc)

    return row


def fetch_replay_daily_prices(
    tickers: list,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    """
    Fetch daily close prices for a historical replay given exact start/end dates.

    start_date: exact start — entry = first trading day on/after this date
    end_date:   exit cap — already capped at today by the caller

    Each value in the returned dict:
        status, reason, rows, entry_price, entry_date, exit_price, exit_date,
        exit_capped, daily_series (pd.Series or None), ticker_used
    """
    try:
        import yfinance as yf
    except ImportError:
        return {
            t: {
                "status": "error", "reason": "yfinance not installed",
                "rows": 0, "entry_price": None, "entry_date": None,
                "exit_price": None, "exit_date": None,
                "exit_capped": False, "daily_series": None, "ticker_used": t,
            }
            for t in tickers
        }

    result = {}
    for ticker in tickers:
        candidates = [ticker] + _TICKER_ALIASES.get(ticker, [])
        best = None
        for candidate in candidates:
            row = _fetch_single_ticker_daily(candidate, start_date, end_date, yf)
            row["ticker_used"] = candidate
            if row["status"] == "ok":
                best = row
                break
            if best is None:
                best = row
        result[ticker] = best

    return result
