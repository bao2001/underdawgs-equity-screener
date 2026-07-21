"""
app.py -- Equity Research Screener
Streamlit single-page application with sidebar navigation.

Run:  python3 -m streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_utils import (
    get_screener_data,
    load_model_outputs,
    load_current_market_data,
    fetch_current_market_data,
    load_ticker_universe,
    expand_universe_for_tickers,
)
from ticker_setup_utils import (
    validate_tickers_for_wizard,
    check_ticker_readiness,
    promote_ticker_if_ready,
    prepare_single_ticker_fundamentals,
    prepare_single_ticker_market_cap,
    prepare_single_ticker_model_output,
    prepare_single_ticker_10k_risk,
    prepare_single_ticker_10q_risk,
    prepare_single_ticker_current_market,
)
from paper_portfolio_utils import (
    load_paper_portfolio, save_paper_portfolio,
    load_paper_transactions, save_paper_transactions,
    load_portfolio_history, save_portfolio_history,
    initialize_paper_account, get_available_tickers,
    get_trade_price, get_company_name, get_signal_metadata,
    execute_paper_trade, calculate_holdings, calculate_portfolio_value,
    calculate_position_metrics, calculate_portfolio_metrics,
    calculate_realized_summary,
    update_portfolio_history, compare_to_benchmark,
    generate_portfolio_history, generate_portfolio_report,
    ACTION_BUY, ACTION_SELL, DEFAULT_STARTING_CASH,
    fetch_replay_prices, fetch_replay_daily_prices,
)
from model_utils import (
    SIGNAL_COLORS, SIGNAL_BG_COLORS, SIGNAL_ORDER,
    get_signal_explanation,
    COMPANY_DESCRIPTIONS, REPORT_SIGNALS,
    generate_benchmark_data, portfolio_metrics,
)
from auth_utils import (
    is_authenticated, sign_in, sign_up, sign_out,
    get_current_user, secrets_configured,
    save_auth_cookies, clear_auth_cookies, restore_session_if_possible,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Underdawg — Equity Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Remove top white bar (Streamlit header chrome) ─────────────────── */
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

/* ── Remove Streamlit's reserved top spacing for the hidden header ─── */
/* section[data-testid="stMain"] carries ~58px padding-top to clear the */
/* Streamlit toolbar chrome.  stAppViewContainer / stMainBlockContainer  */
/* may add further offsets.  Zero all three so the page starts flush.   */
section[data-testid="stMain"],
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ── Hide native Streamlit sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarResizeHandle"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Collapse invisible top elements (CSS injection wrapper, CookieManager) ── */
/* The CSS st.markdown and stx.CookieManager() both render as stElementContainer */
/* direct children of block-container > stVerticalBlock, above the nav row.     */
/* They contain no visible content but Streamlit gives them non-zero height.     */
/* :has(style)  → targets the CSS injection markdown container                  */
/* :has(iframe) → targets the CookieManager iframe-backed component             */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"]:has(style),
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"]:has(iframe) {
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
}

/* ── Zero outer stVerticalBlock gap — removes purple spacer bands ───────────── */
/* Streamlit's default gap (1rem) on the outer stVerticalBlock creates visible   */
/* spacer bands: one above the nav (between collapsed items and nav row) and one */
/* below the nav (stacking on top of the nav's intentional margin-bottom: 24px). */
/* The > direct-child combinator limits this to the outermost stVerticalBlock    */
/* only — nested stVerticalBlocks (inside columns, tabs, etc.) are unaffected.   */
.block-container > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── Top nav bar — sticky, white background ──────────────────────── */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type {
    position: sticky !important;
    top: 0 !important;
    height: 92px !important;
    min-height: 92px !important;
    z-index: 9998 !important;
    background: #ffffff !important;
    gap: 0 !important;
    padding: 0 8px !important;
    border-bottom: 1px solid #dee3ea !important;
    align-items: center !important;
    margin-bottom: 24px !important;
}

/* Column layout */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stColumn"] {
    background: transparent !important;
    height: 92px !important;
    padding: 0 4px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: stretch !important;
    min-width: 0 !important;
}

            
/* Inner vertical blocks — flex:1 + padding:0 + margin:0 ensures clean centering */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex: 1 !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

/* Element containers */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stElementContainer"] {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}

/* Markdown containers — display:flex + align-items:center centers child div vertically */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    display: flex !important;
    align-items: center !important;
}
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    display: contents !important;
}

/* ── Top nav buttons ─────────────────────────────────────────────── */
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #334e68 !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 16px !important;
    border-radius: 10px !important;
    width: 100% !important;
    letter-spacing: 0.1px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button:hover {
    background: rgba(37,99,235,0.08) !important;
    color: #1d4ed8 !important;
    border: none !important;
    box-shadow: none !important;
}
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button:focus,
.block-container > [data-testid="stVerticalBlock"]
  > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stButton"] button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

/* ── Logo box ─────────────────────────────────────────────────────── */
.ud-nav-logo-box {
    width: 72px; min-width: 72px; height: 72px;
    border-radius: 14px; overflow: hidden;
    background: #1a3060; flex-shrink: 0;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.4);
}

/* ── Active nav pill — matches inactive button height exactly ──────── */
.ud-nav-pill-active {
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #fff !important;
    font-size: 13.5px;
    font-weight: 600;
    height: 38px;
    min-height: 38px;
    padding: 0 18px;
    border-radius: 10px;
    white-space: nowrap;
    cursor: default;
    width: 100%;
    box-sizing: border-box;
    text-align: center;
    letter-spacing: 0.1px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}

/* ── Nav control offset spacer — pushes buttons/email/signout downward ───────── */
.ud-nav-control-offset {
    height: 14px;
}

/* Override Streamlit's default block-container padding (80px 80px 160px).       */
/* Two-class selector (.stMainBlockContainer.block-container) has specificity    */
/* (0,2,0) — beats Streamlit's single-class rules even when both use !important. */
.stMainBlockContainer.block-container {
    padding: 30px 20px 0 20px !important;
}

[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}

.disclaimer-box {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.6;
}

.signal-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.section-header {
    font-size: 20px;
    font-weight: 700;
    color: #1a2744;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

.mini-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.mini-card .label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
.mini-card .value { font-size: 20px; font-weight: 700; color: #1a2744; }

.hero-banner {
    background: linear-gradient(135deg, #1a2744 0%, #2563eb 100%);
    color: white;
    border-radius: 12px;
    padding: 22px 32px;
    margin-top: 80px;
    margin-bottom: 45px;
}
.hero-banner h1 { color: white; margin: 0 0 8px 0; font-size: 28px; }
.hero-banner p  { color: #c7d8f5; margin: 0; font-size: 15px; }

.data-label {
    background: #e8f4fd;
    border: 1px solid #bee3f8;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    color: #2b6cb0;
    display: inline-block;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.current-label {
    background: #e9fef0;
    border: 1px solid #9ae6b4;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    color: #22543d;
    display: inline-block;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.ud-product-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 130px;
}
.ud-product-title {
    font-size: 14px;
    font-weight: 700;
    color: #1a2744;
    margin-bottom: 8px;
}
.ud-product-desc {
    font-size: 12px;
    color: #64748b;
    line-height: 1.5;
}

.replay-summary-box {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 14px 20px;
    margin: 12px 0 16px 0;
    font-size: 14px;
    line-height: 1.6;
}
.replay-summary-box table { border-spacing: 0 4px; }
.replay-summary-box td    { padding-right: 18px; }

.product-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .mc-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.metric-card .mc-value { font-size: 22px; font-weight: 700; color: #1a2744; }
.signal-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 600; border: 1px solid currentColor;
}
.preview-card {
    background: #f0f7ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 16px 18px;
}
.preview-card .pc-title { font-size: 13px; font-weight: 700; color: #1e3a8a; margin-bottom: 10px; }
.preview-card .pc-row   { font-size: 12px; color: #334155; line-height: 1.7; }
.preview-card .pc-cap   { font-size: 11px; color: #64748b; margin-top: 8px; font-style: italic; }
.disclaimer-strip {
    background: #fff8e1; border: 1px solid #fde68a; border-radius: 6px;
    padding: 8px 14px; font-size: 12px; color: #78350f; margin: 8px 0 36px 0;
}
.about-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1a2744 100%);
    color: white; border-radius: 12px; padding: 20px 28px; margin-bottom: 16px;
}
.section-subtitle {
    font-size: 13px; color: #64748b; margin: -8px 0 14px 0; line-height: 1.5;
}
/* ── Home CTA cards ─────────────────────────────────────────────────── */
.cta-card {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    margin-bottom: 12px;
    display: flex;
    flex-direction: column;
    height: 320px;
}
.cta-card:hover {
    box-shadow: 0 8px 24px rgba(37,99,235,0.13);
    border-color: #2563eb;
}
.cta-card-body {
    padding: 14px 16px 10px 16px;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

/* ── Home CTA anchor cards (kept for query-param sidebar links, not used for card HTML) */
.home-cta-card {
    display: block;
    text-decoration: none !important;
    color: inherit !important;
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.18s ease;
    cursor: pointer;
}
.home-cta-card:hover {
    box-shadow: 0 10px 32px rgba(37,99,235,0.18);
    border-color: #2563eb;
    transform: translateY(-3px);
    text-decoration: none !important;
    color: inherit !important;
}
.home-cta-card:active {
    transform: translateY(0);
    box-shadow: 0 3px 10px rgba(37,99,235,0.1);
}
.cta-card-body { padding: 14px 16px 10px 16px; }
.cta-card-footer {
    padding: 11px 20px;
    background: #f0f7ff;
    border-top: 1px solid #e2e8f0;
    font-size: 14px;
    font-weight: 700;
    color: #2563eb;
    transition: background 0.18s ease, color 0.18s ease;
    letter-spacing: 0.1px;
}
.home-cta-card:hover .cta-card-footer {
    background: #dbeafe;
    color: #1d4ed8;
}
.cta-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #2563eb;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.cta-title {
    font-size: 18px;
    font-weight: 800;
    color: #1a2744;
    margin-bottom: 4px;
    line-height: 1.2;
}
.cta-subtitle {
    font-size: 13px;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 12px;
}
.cta-preview-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
}

/* ── Consistent page heading system ─────────────────────────────────── */
/* H2: major content sections below the page title */
.block-container [data-testid="stMarkdownContainer"] h2 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a2744 !important;
    letter-spacing: -0.3px !important;
    margin: 22px 0 8px 0 !important;
    line-height: 1.2 !important;
}
/* H3: section headings within tabs / content blocks — matches .section-header size */
.block-container [data-testid="stMarkdownContainer"] h3 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1a2744 !important;
    margin: 18px 0 8px 0 !important;
    line-height: 1.2 !important;
}
/* H4: subsection and card headings */
.block-container [data-testid="stMarkdownContainer"] h4 {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #1a2744 !important;
    margin: 14px 0 5px 0 !important;
}
/* H5: smaller labels and grouped items */
.block-container [data-testid="stMarkdownContainer"] h5 {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #334e68 !important;
    margin: 10px 0 3px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Page names ────────────────────────────────────────────────────────────────

PAGES = [
    "Home",
    "Screener",
    "Company Detail",
    "Portfolio Simulator",
    "About",
]

# ── Auth debug flag ───────────────────────────────────────────────────────────
# Terminal-only output — never shown in the Streamlit UI.  Set False when done.
AUTH_DEBUG = False

# ── Cookie manager (invisible browser-bridge component) ───────────────────────
# Rendered here, at the very top of every run, so cookies are readable before
# the auth gate.  Falls back to None when the package is not installed.
try:
    import extra_streamlit_components as stx  # type: ignore
    _cm = stx.CookieManager(key="ud_auth")
except Exception:
    _cm = None

# ── Session state ─────────────────────────────────────────────────────────────

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "detail_source" not in st.session_state:
    st.session_state.detail_source = "sample"
if "paper_ticker_prefill" not in st.session_state:
    st.session_state.paper_ticker_prefill = None
if "paper_source_prefill" not in st.session_state:
    st.session_state.paper_source_prefill = "sample"
if "supabase_user" not in st.session_state:
    st.session_state["supabase_user"] = None
if "supabase_session" not in st.session_state:
    st.session_state["supabase_session"] = None

# Resolve navigation.  Priority: pending_nav_page (button) > query param (anchor link).
if "pending_nav_page" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("pending_nav_page")
else:
    try:
        _qp_nav = st.query_params.get("nav", None)
        _qp_tab = st.query_params.get("portfolio_tab", None)
        _qp_consumed = False
        if _qp_nav and _qp_nav in PAGES:
            st.session_state.nav_page = _qp_nav
            _qp_consumed = True
        if _qp_tab:
            st.session_state["portfolio_tab"] = _qp_tab
            _qp_consumed = True
        if _qp_consumed:
            st.query_params.clear()
    except Exception:
        pass

# Handle logout via ?auth=logout query param (legacy / direct URL fallback).
try:
    if st.query_params.get("auth") == "logout":
        sign_out()
        clear_auth_cookies(_cm)
        st.session_state["_just_logged_out"] = True
        st.session_state["_cm_init_retries"] = 0
        st.query_params.clear()
        st.rerun()
except Exception:
    pass

# ── Cached data ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_sample(use_live: bool) -> tuple:
    return get_screener_data(use_live=use_live)


@st.cache_data(show_spinner=False)
def _cached_diagnostics() -> tuple:
    try:
        from real_model_utils import run_diagnostics
        return run_diagnostics()
    except Exception as exc:
        return {"status": "import_error", "error": str(exc)}, None


@st.cache_data(ttl=300)
def _load_supplementary() -> tuple:
    """Load model outputs, current market data, and 10-Q risk update (cached 5 min)."""
    mo_df, mo_msg = load_model_outputs()
    cm_df, cm_msg = load_current_market_data()
    return mo_df, mo_msg, cm_df, cm_msg


@st.cache_data(ttl=300)
def _load_10q_data() -> tuple:
    """Load 10-Q risk scores and trend update (cached 5 min). Returns (q_df, upd_df)."""
    try:
        from data_utils import load_10q_risk_scores, load_filing_risk_update
        q_df,   _ = load_10q_risk_scores()
        upd_df, _ = load_filing_risk_update()
        return q_df, upd_df
    except ImportError:
        return None, None
    except Exception:
        return None, None


@st.cache_data(ttl=300)
def _load_all_model_outputs():
    """
    Load all model-output rows (not deduplicated by ticker).
    Prefers model_outputs_combined.csv (2010–present) over
    model_outputs_reviewed.csv (2010–2016 only).
    """
    from data_utils import load_combined_model_outputs
    df, msg, src = load_combined_model_outputs()
    if df is None:
        return None, msg
    yrs = sorted(df["year"].dropna().astype(int).unique().tolist())
    _label_map = {
        "calibrated": "Calibrated combined outputs",
        "combined":   "Combined outputs",
        "legacy":     "Legacy reviewed outputs",
        "base":       "Base legacy outputs",
        "missing":    "Missing outputs",
    }
    label = _label_map.get(src, src)
    return df, f"{label}: {len(df)} rows across years {yrs}"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_ticker_universe_cached() -> tuple:
    """Cached ticker universe load (1-hour TTL — refreshed after pipeline runs)."""
    return load_ticker_universe()


@st.cache_data(ttl=3600)
def _fetch_replay_prices_cached(tickers_tuple: tuple, start_year: int, holding_months: int) -> dict:
    """Cached wrapper around fetch_replay_prices. Uses tuple key for hashability."""
    return fetch_replay_prices(list(tickers_tuple), start_year, holding_months)


@st.cache_data(ttl=3600)
def _fetch_replay_daily_cached(
    tickers_tuple: tuple,
    start_date_str: str,
    end_date_str: str,
) -> dict:
    """Cached wrapper around fetch_replay_daily_prices. Date strings are cache-key safe."""
    return fetch_replay_daily_prices(
        list(tickers_tuple),
        pd.Timestamp(start_date_str),
        pd.Timestamp(end_date_str),
    )


# ── Auth gate ─────────────────────────────────────────────────────────────────

def _page_login() -> None:
    """Login / sign-up screen shown to unauthenticated visitors."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;padding:0 0 16px 0">
          <div style="font-size:30px;font-weight:800;color:#1a2744;letter-spacing:-0.5px">Underdawg</div>
          <div style="font-size:12px;color:#64748b;margin-top:4px;letter-spacing:0.5px;text-transform:uppercase">
            Equity Research &middot; MSDSBA
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align:center;color:#475569;font-size:14px;margin-bottom:24px'>"
            "Sign in to keep your watchlist, notes, and paper portfolio separate."
            "</p>",
            unsafe_allow_html=True,
        )

        if not secrets_configured():
            st.error(
                "Supabase is not configured. "
                "Add `[supabase]` with `url` and `anon_key` to `.streamlit/secrets.toml`."
            )
            st.stop()

        # ── Sign in ──────────────────────────────────────────────────────────
        st.markdown("#### Sign in")
        login_email = st.text_input(
            "Email", key="auth_login_email", placeholder="you@example.com",
            label_visibility="collapsed",
        )
        login_pw = st.text_input(
            "Password", key="auth_login_password", type="password",
            placeholder="Password",
            label_visibility="collapsed",
        )

        if st.button("Sign in", type="primary", use_container_width=True, key="auth_signin_btn"):
            if not login_email.strip():
                st.error("Please enter your email address.")
            elif not login_pw:
                st.error("Please enter your password.")
            else:
                with st.spinner("Signing in…"):
                    _r = sign_in(login_email.strip(), login_pw)
                if _r["ok"]:
                    st.session_state["supabase_user"]    = _r["user"]
                    st.session_state["supabase_session"] = _r["session"]
                    # Defer cookie save to next render — calling cm.set() right
                    # before st.rerun() races with the component JS and cookies
                    # never get written.  The _pending_cookie_save flag is consumed
                    # in the authenticated section on the very next render.
                    st.session_state["_pending_cookie_save"] = _r.get("session") or {}
                    st.rerun()
                else:
                    st.error(_r["error"])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Create account ───────────────────────────────────────────────────
        with st.expander("Create account"):
            signup_email = st.text_input(
                "Email", key="auth_signup_email", placeholder="you@example.com",
                label_visibility="collapsed",
            )
            signup_pw = st.text_input(
                "Password", key="auth_signup_pw", type="password",
                placeholder="Password (min 6 characters)",
                label_visibility="collapsed",
                help="Minimum 6 characters.",
            )
            signup_pw2 = st.text_input(
                "Confirm password", key="auth_signup_pw2", type="password",
                placeholder="Confirm password",
                label_visibility="collapsed",
            )

            if st.button("Create account", use_container_width=True, key="auth_signup_btn"):
                if not signup_email.strip():
                    st.error("Please enter an email address.")
                elif not signup_pw:
                    st.error("Please enter a password.")
                elif len(signup_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif signup_pw != signup_pw2:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    with st.spinner("Creating account…"):
                        _r = sign_up(signup_email.strip(), signup_pw)
                    if _r["ok"]:
                        if _r.get("needs_confirmation"):
                            st.success(
                                "Account created! Check your email to confirm your address, "
                                "then sign in above."
                            )
                        elif _r.get("session"):
                            st.session_state["supabase_user"]    = _r["user"]
                            st.session_state["supabase_session"] = _r["session"]
                            st.session_state["_pending_cookie_save"] = _r.get("session") or {}
                            st.rerun()
                        else:
                            st.success("Account created. Please sign in above.")
                    else:
                        st.error(_r["error"])

        st.markdown(
            "<p style='text-align:center;font-size:11px;color:#94a3b8;margin-top:28px'>"
            "Underdawg is for educational and research support only. Not investment advice."
            "</p>",
            unsafe_allow_html=True,
        )


# ── Cookie-based auth restore ─────────────────────────────────────────────────
# CookieManager is an iframe-backed component.  On a hard browser reload, the
# component re-initialises and its JS fires asynchronously.  get_all() returns
# {} (the default) while hydrating — identical to "no cookies" — so we cannot
# distinguish the two cases on the first render.  Strategy:
#
#  1. Try restore immediately (succeeds on render 2+ when CM was already warm).
#  2. If cookies are empty AND we have retries left AND the user didn't just log
#     out: show "Restoring session…" and wait.  The CM auto-rerun (triggered by
#     setComponentValue) will deliver the real cookies on the next render.
#  3. Once CM has actual data (non-empty dict), attempt restore.
#  4. After retry window is exhausted with no cookies, or after explicit logout,
#     show the login page immediately.

_CM_MAX_RETRIES = 3   # covers CM hydration delay; 3 renders ≈ a few hundred ms

# Step 1 — immediate attempt (works on render 2+ after CM fires setComponentValue)
if not is_authenticated():
    restore_session_if_possible(_cm, debug=AUTH_DEBUG)

# Step 2 — CM hydration gate
if not is_authenticated() and _cm is not None:
    _cm_retry  = st.session_state.get("_cm_init_retries", 0)
    _just_lo   = st.session_state.get("_just_logged_out",  False)

    try:
        _all_c = _cm.get_all()
    except Exception:
        _all_c = {}

    _cm_has_data   = bool(_all_c)                                 # non-empty dict
    _at_present    = bool((_all_c or {}).get("ud_access_token"))
    _rt_present    = bool((_all_c or {}).get("ud_refresh_token"))
    _auth_ck_found = _at_present or _rt_present

    # Wait when: CM returned empty (might be default, not yet hydrated),
    #            retry budget remains, and user didn't just click Sign out.
    _should_wait = (
        not _cm_has_data
        and _cm_retry < _CM_MAX_RETRIES
        and not _just_lo
    )

    if AUTH_DEBUG:
        print(
            f"[auth] cm_data={'yes' if _cm_has_data else 'no'} | "
            f"at={'yes' if _at_present else 'no'} | "
            f"rt={'yes' if _rt_present else 'no'} | "
            f"retry={_cm_retry} | wait={'yes' if _should_wait else 'no'} | "
            f"keys={list((_all_c or {}).keys())}"
        )

    if _should_wait:
        # CM hasn't delivered cookies yet — hold the login gate, give it a render.
        if AUTH_DEBUG:
            print(
                f"[auth] restoring session… "
                f"(attempt {_cm_retry + 1}/{_CM_MAX_RETRIES})"
            )
        st.session_state["_cm_init_retries"] = _cm_retry + 1
        st.markdown(
            "<div style='text-align:center;color:#64748b;"
            "padding:60px 0;font-size:14px'>Restoring session…</div>",
            unsafe_allow_html=True,
        )
        st.rerun()

    elif _auth_ck_found:
        # Auth cookies are present — attempt restore.
        _restored = restore_session_if_possible(_cm, debug=AUTH_DEBUG)
        if not _restored:
            # Tokens are stale / invalid — clear them so the login form is clean.
            if AUTH_DEBUG:
                print("[auth] restore failed — clearing stale auth cookies")
            clear_auth_cookies(_cm, debug=AUTH_DEBUG)

    # Delivery confirmed (or retries exhausted) — reset counter for next nav click.
    st.session_state["_cm_init_retries"] = 0

if not is_authenticated():
    if AUTH_DEBUG:
        print(
            "[auth] showing login "
            f"({'post-logout' if st.session_state.get('_just_logged_out') else 'retry exhausted or no auth cookies'})"
        )
    st.session_state.pop("_just_logged_out", None)
    _page_login()
    st.stop()

# ── Deferred auth-cookie save ─────────────────────────────────────────────────
# Calling cm.set() immediately before st.rerun() (in the sign-in handler) races
# with the CookieManager component's JS — the rerun can start before the browser
# executes the cookie-write.  Instead, sign-in stores a _pending_cookie_save flag
# and we consume it here, during the first fully authenticated render cycle.
if "_pending_cookie_save" in st.session_state:
    _pcs = st.session_state.pop("_pending_cookie_save")
    if isinstance(_pcs, dict):
        _pcs_at = str(_pcs.get("access_token") or "")
        _pcs_rt = str(_pcs.get("refresh_token") or "")
        if _pcs_at and _pcs_rt:
            if AUTH_DEBUG:
                print(
                    f"[auth] deferred cookie save: "
                    f"at_len={len(_pcs_at)} rt_len={len(_pcs_rt)}"
                )
            save_auth_cookies(_cm, _pcs_at, _pcs_rt, debug=AUTH_DEBUG)

# ── Authenticated — get user for nav rail display ─────────────────────────────
_auth_user  = get_current_user() or {}
_auth_email = _auth_user.get("email", "")

# ── Sidebar navigation ────────────────────────────────────────────────────────

_LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACwAAAAsCAYAAAAehFoBAAAGA0lEQVR42u1YbWxb1Rl+"
    "zjnXcZzEid0kdtKEZA2gdSYFQYAobdogSqdMglZMdTdYgcTJbux0dWyH0oYWXXkaoNJW"
    "VOUHNFr5Kpq0ekJlo1KF6BAZjEBpl6HV62BMgPhIU6XLB4kd557z8qOO8KZqqkcSpsnP"
    "z3vP+973POc9z3POBXLIIYcc/q/A5jOZYRg8Hr+GeTxnGADE49eQx3OGotGo+p+atdd7"
    "RFzmGPatMmwYBs9kz+ePXC84q5fE3IwRMVLnAX7ml0/uOw2ALhWzaAV7vV4Ri8UkAHR0"
    "3x/QNHEbKZpSyvzQlHJYMCGJswpNiCsZmEMq+fr4yMdPxGIxmRm7KAXPfdDnD9Yzrj3O"
    "uBjijJ7qf2Lvh5carwe31ygl/UrKJiWp99n+/ae/SdFZtwEAtPlDqzu6I+90+kOr595x"
    "zgGbvdF9xXc2ebzePM45GPuak3a9p9EXiLzd5g/dmplrQYslItYeCNV1dEfeaQ+E6tKM"
    "5xGRAFCyfedDn2/b8eAJACgsca11lFddB4C1trZaAUDXgzUdgcjJtq7g1UTEsi066xky"
    "xogR20Okdj3z5P5/6LpuASAZY/LaG27afW19fWV5eZkLwIYO373HVjevrARAdrvd1HXd"
    "0t9/4BMF1ceZ2MMYowWULq8AgPv04Nr2QPiFNOMaAE5EHMBVXVuCqS3BsHz52LFUsbNc"
    "tXV2kc+nNxERn4tPx8DnDx2+Vw99PzP3vDLs8XgYAAiheUnhV0TE4vE4NTQ0CMaYWra8"
    "fv2a5lWWsfFxOXjylKW5eZVaUe+h1/8w0MAYUyMjI2kziRMRMUXqecGxEQBG0rnnHS2G"
    "ofn84aN3BwLO9BMNAJBf1Fx79ffOvTU4qKrqvitFgYNEfrHy6QHzrcFBefv6O8Occ6Tb"
    "hwGAz7fN3u4PH21t3WpdMAv/iR6qbA9EjqR3IAcAW2nFjQUl5S8tqaiRdctXqKLSSrI5"
    "XGS1l9K61tvVxMS4/OvZswTgJgagpaVFm2uBNn/413f7umuzkdjLbImLeyNvFjYiMgEA"
    "0SgVl5dfKUj9gGuaO2WafPjcCCWTSZQ6ShDp2UKbvD9Ehx5QXwwPq/sf6NtrtdubBgYG"
    "zN9c1F/GwVIWjdsAgOjy9p+Wnb/ISQ7Nkp6oUlJsBqlhcCzXhKCx0VHWEw6SsasPTqcT"
    "UpqIddutQklJe3Y/sqa5uemPO3Yab46dPx8eHv70pCJpI2aZTKvPvG46AsBqapaMEkjz"
    "9/aWARAg6QDnD2lCK5mcmCDvj73Yv+8xcjqdKplMgDGG2ppaWrasTgGY2XDHHSm3u2LV"
    "+HQCq0NDhQTkJS40Df/LMs6XShiGIaLRqCJSHxVYSm4GRCu3WLeCcEEpmdQ0jV/45xgb"
    "eONNDkDk59tgmrOYnJjgqdSMNmua1uOvvJo3ev7c4cTE6MkDTz3XKhj7OBbbJOekbh5b"
    "AgCgAIDk7AuJxHREK7Q7oGiG5GxQ8bzXrLZ8OnHitclT7767/r2hU50vHv3d5kOHnqbk"
    "TGrM5So7bc2zfvb7V159G5g5lKbzHqZod2buBTOPLZGdextW3vIXUeCYKnEtjRSWVrxh"
    "L68ip7vqWQDw3tP10zXr1hOsReMAPHPxgl9c0PZAeLPPHzmcrWlkbc2xWEwZhsE1qf3i"
    "husbxqqWVlmnphL7OGN2ADRrypXV1dU2m4X/XZoz007Hkg8YY/GDBw9awuGwTSqFdr2n"
    "kYGHUpjuNQyDx44cUYtyWgvteLTuPr3n/eXX3UyiwEEFTney2FVNBUvcewGg0On6tKi0"
    "8uVMfe3s7t3YEYj8qU0Pr/hvT2vsm9w0+voedo9Nf3nw/Q/+tuHP7w0hkZgGZ9w0VXKV"
    "QN5LBPrt1IVzXXp3byOE8EtpulOm6T/cf+CTRTsP/zvTABB9dP+djWvWHbcWl40Vu62p"
    "0Ok+ay9bSlp+8a5Q38836sHtxzt/tu2uS8UuKoiIZa5SaekVS4scFS2FTtddhU73j/Id"
    "jtqtW43izPHpmG//1vyfWCMilq0aLMp/iXQ+nunl2bhYDjnkkEMOOeSQw0LjKwbZl8H5"
    "v+eVAAAAAElFtkSuQmCC"
)

def set_nav_page(page_name: str) -> None:
    """Internal navigation (used by in-page buttons). Sets nav_page and reruns."""
    st.session_state["nav_page"] = page_name
    st.rerun()


_cur_page = st.session_state.get("nav_page", "Home")

# ── Top navigation bar ─────────────────────────────────────────────────────
(
    _tnc_brand,
    _tnc_home,
    _tnc_screener,
    _tnc_detail,
    _tnc_portfolio,
    _tnc_about,
    _tnc_gap,
    _tnc_email,
    _tnc_signout,
) = st.columns([2.5, 0.9, 1.1, 1.4, 2.0, 0.8, 0.4, 1.8, 1.0])

with _tnc_brand:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding:0 4px">'
        f'<div class="ud-nav-logo-box">'
        f'<img src="data:image/png;base64,{_LOGO_B64}" '
        f'style="width:72px;height:72px;object-fit:contain;image-rendering:auto;display:block" alt=""></div>'
        f'<div style="display:flex;flex-direction:column;gap:3px">'
        f'<div style="font-size:28px;font-weight:900;color:#0b2347;line-height:1.1;'
        f'letter-spacing:-0.6px">Underdawg</div>'
        f'<div style="font-size:13px;font-weight:500;color:#3d638a;line-height:1.2;'
        f'letter-spacing:0.4px">Equity Research</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

for _tnc_label, _tnc_col in [
    ("Home",                _tnc_home),
    ("Screener",            _tnc_screener),
    ("Company Detail",      _tnc_detail),
    ("Portfolio Simulator", _tnc_portfolio),
    ("About",               _tnc_about),
]:
    with _tnc_col:
        st.markdown('<div class="ud-nav-control-offset"></div>', unsafe_allow_html=True)
        if _cur_page == _tnc_label:
            st.markdown(
                f'<div class="ud-nav-pill-active">{_tnc_label}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(
                _tnc_label,
                key=f"nav_{_tnc_label.replace(' ', '_')}",
                use_container_width=True,
            ):
                set_nav_page(_tnc_label)

with _tnc_email:
    st.markdown('<div class="ud-nav-control-offset"></div>', unsafe_allow_html=True)
    if _auth_email:
        _disp = (_auth_email[:22] + "…") if len(_auth_email) > 22 else _auth_email
        st.markdown(
            f'<div style="font-size:11px;color:#5a7a9e;text-align:right;padding:2px 4px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'{_disp}</div>',
            unsafe_allow_html=True,
        )

with _tnc_signout:
    st.markdown('<div class="ud-nav-control-offset"></div>', unsafe_allow_html=True)
    if st.button("Sign out", key="nav_signout", use_container_width=True):
        sign_out()
        clear_auth_cookies(_cm, debug=AUTH_DEBUG)
        st.session_state["_just_logged_out"] = True
        st.session_state["_cm_init_retries"] = 0
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────

page = st.session_state.get("nav_page", "Home")

df, data_status = _load_sample(False)

# ── Supplementary data ────────────────────────────────────────────────────────

mo_df, mo_msg, cm_df, cm_msg = _load_supplementary()

# ── Styling helpers ───────────────────────────────────────────────────────────

def _safe_map(styler, fn, col_name: str):
    if col_name not in styler.data.columns:
        return styler
    try:
        return styler.map(fn, subset=[col_name])
    except AttributeError:
        return styler.applymap(fn, subset=[col_name])


def _safe_apply(styler, fn):
    try:
        return styler.apply(fn)
    except Exception:
        return styler


def _safe_format(styler, fmt_dict: dict):
    existing = {k: v for k, v in fmt_dict.items() if k in styler.data.columns}
    if not existing:
        return styler
    try:
        return styler.format(existing)
    except Exception:
        return styler


def _sfmt(val, fmt=".2f", prefix="$", suffix="", na="N/A") -> str:
    """Format a numeric value safely; return na string for NaN."""
    try:
        if pd.isna(val):
            return na
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except Exception:
        return na


def signal_badge_html(signal: str) -> str:
    bg = SIGNAL_BG_COLORS.get(signal, "#f0f0f0")
    fg = SIGNAL_COLORS.get(signal, "#333")
    return (
        f'<span class="signal-badge" '
        f'style="background:{bg}; color:{fg}; border: 1px solid {fg};">'
        f'{signal}</span>'
    )


# ── Watchlist + Research Notes helpers ───────────────────────────────────────

_WL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.csv")
_NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_notes.csv")

_WL_COLS = [
    "ticker", "company_name", "added_at", "source_page", "model_year", "signal",
    "current_valuation_gap_pct", "quality_score", "filing_risk_score",
    "latest_10q_risk_score", "filing_risk_trend", "research_status", "user_note",
]
_WL_STATUSES = ["New", "Reviewing", "Simulated", "Rejected", "Added to Paper Portfolio"]

_NOTES_COLS = ["ticker", "updated_at", "thesis", "risks", "questions", "decision", "checklist_json"]
_NOTES_DECISIONS = ["No decision", "Continue researching", "Reject", "Simulate", "Add to paper portfolio"]
_CHECKLIST_ITEMS = [
    "Reviewed business model",
    "Reviewed revenue growth",
    "Reviewed margins",
    "Reviewed debt / liquidity",
    "Reviewed annual 10-K risk",
    "Reviewed latest 10-Q update",
    "Compared peers",
    "Ran historical replay",
    "Wrote thesis",
]


def _load_watchlist() -> pd.DataFrame:
    if not os.path.exists(_WL_PATH):
        return pd.DataFrame(columns=_WL_COLS)
    try:
        df = pd.read_csv(_WL_PATH, dtype=str).fillna("")
        for c in _WL_COLS:
            if c not in df.columns:
                df[c] = ""
        return df[_WL_COLS]
    except Exception:
        return pd.DataFrame(columns=_WL_COLS)


def _save_watchlist(wl: pd.DataFrame) -> None:
    try:
        wl.to_csv(_WL_PATH, index=False)
    except Exception:
        pass


def _wl_in_watchlist(ticker: str) -> bool:
    return ticker.upper() in _load_watchlist()["ticker"].str.upper().values


def _wl_add(ticker: str, *, company_name="", source_page="", model_year="",
             signal="", val_gap=None, quality=None, risk=None,
             q10_risk=None, q10_trend="") -> bool:
    wl = _load_watchlist()
    if ticker.upper() in wl["ticker"].str.upper().values:
        return False
    def _s(v):
        return str(v) if (v is not None and pd.notna(v)) else ""
    new_row = {
        "ticker":                    ticker.upper(),
        "company_name":              company_name,
        "added_at":                  pd.Timestamp.now().isoformat(timespec="seconds"),
        "source_page":               source_page,
        "model_year":                str(model_year) if model_year else "",
        "signal":                    str(signal) if signal else "",
        "current_valuation_gap_pct": _s(val_gap),
        "quality_score":             _s(quality),
        "filing_risk_score":         _s(risk),
        "latest_10q_risk_score":     _s(q10_risk),
        "filing_risk_trend":         str(q10_trend) if q10_trend else "",
        "research_status":           "New",
        "user_note":                 "",
    }
    _save_watchlist(pd.concat([wl, pd.DataFrame([new_row])], ignore_index=True))
    return True


def _wl_remove(ticker: str) -> None:
    wl = _load_watchlist()
    _save_watchlist(wl[wl["ticker"].str.upper() != ticker.upper()])


def _wl_update(ticker: str, *, status: str = None, note: str = None) -> None:
    wl = _load_watchlist()
    mask = wl["ticker"].str.upper() == ticker.upper()
    if status is not None:
        wl.loc[mask, "research_status"] = status
    if note is not None:
        wl.loc[mask, "user_note"] = note
    _save_watchlist(wl)


def _load_notes() -> pd.DataFrame:
    if not os.path.exists(_NOTES_PATH):
        return pd.DataFrame(columns=_NOTES_COLS)
    try:
        df = pd.read_csv(_NOTES_PATH, dtype=str).fillna("")
        for c in _NOTES_COLS:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=_NOTES_COLS)


def _get_note(ticker: str) -> dict:
    notes = _load_notes()
    match = notes[notes["ticker"].str.upper() == ticker.upper()]
    return match.iloc[0].to_dict() if len(match) > 0 else {}


def _save_note(ticker: str, thesis: str, risks: str, questions: str,
               decision: str, checklist: dict) -> None:
    import json
    notes = _load_notes()
    mask = notes["ticker"].str.upper() == ticker.upper()
    row = {
        "ticker":         ticker.upper(),
        "updated_at":     pd.Timestamp.now().isoformat(timespec="seconds"),
        "thesis":         thesis or "",
        "risks":          risks or "",
        "questions":      questions or "",
        "decision":       decision or "No decision",
        "checklist_json": json.dumps(checklist),
    }
    if mask.any():
        for col, val in row.items():
            notes.loc[mask, col] = val
    else:
        notes = pd.concat([notes, pd.DataFrame([row])], ignore_index=True)
    try:
        notes.to_csv(_NOTES_PATH, index=False)
    except Exception:
        pass


def generate_company_explanation(row: dict) -> list:
    """Return plain-English research bullets for a company row. No buy/sell language."""
    bullets = []

    vgap = row.get("Valuation_Gap_Pct") or row.get("valuation_gap_pct") or row.get("current_valuation_gap_pct")
    if vgap is not None:
        try:
            vgap_f = float(vgap)
            if not pd.isna(vgap_f):
                if vgap_f > 30:
                    bullets.append(
                        f"Valuation gap is **+{vgap_f:.1f}%** — the model estimates fair value well above "
                        "the market cap in the model year. This may deserve further review, though the gap "
                        "could reflect model uncertainty or sector dynamics."
                    )
                elif vgap_f > 5:
                    bullets.append(
                        f"Valuation gap is **+{vgap_f:.1f}%** — the model estimates fair value modestly "
                        "above the market cap in the model year."
                    )
                elif vgap_f >= -10:
                    bullets.append(
                        f"Valuation gap is **{vgap_f:.1f}%** — the model estimates fair value roughly "
                        "in line with the market cap in the model year."
                    )
                else:
                    bullets.append(
                        f"Valuation gap is **{vgap_f:.1f}%** — the model estimates fair value below "
                        "the market cap in the model year."
                    )
        except (TypeError, ValueError):
            pass

    qs = row.get("Quality_Score") or row.get("quality_score")
    if qs is not None:
        try:
            qs_f = float(qs)
            if not pd.isna(qs_f):
                if qs_f >= 70:
                    bullets.append(f"Quality score is **{qs_f:.0f}/100** (strong) — the company showed strong profitability and low leverage in the model year.")
                elif qs_f >= 40:
                    bullets.append(f"Quality score is **{qs_f:.0f}/100** (moderate) — fundamentals were adequate but not exceptional in the model year.")
                else:
                    bullets.append(f"Quality score is **{qs_f:.0f}/100** (weak) — the company showed weakness in profitability or leverage in the model year.")
        except (TypeError, ValueError):
            pass

    risk = (row.get("report_risk_score_real") or row.get("Report_Risk_Score")
            or row.get("report_risk_score") or row.get("filing_risk_score"))
    if risk is not None:
        try:
            risk_f = float(risk)
            if not pd.isna(risk_f):
                if risk_f < 35:
                    bullets.append(f"Annual filing risk is **{risk_f:.0f}/100** (low) — the 10-K contained relatively few risk-escalation terms relative to peers.")
                elif risk_f < 65:
                    bullets.append(f"Annual filing risk is **{risk_f:.0f}/100** (moderate) — the 10-K contained a typical level of risk language.")
                else:
                    bullets.append(f"Annual filing risk is **{risk_f:.0f}/100** (elevated) — risk language in this 10-K should be reviewed carefully.")
        except (TypeError, ValueError):
            pass

    trend   = row.get("filing_risk_trend") or row.get("latest_10q_risk_trend")
    q_score = row.get("latest_10q_risk_score")
    if trend and str(trend) not in ("", "nan", "None", "Unavailable"):
        q_s = f" (10-Q score: {float(q_score):.0f}/100)" if (q_score and pd.notna(q_score)) else ""
        ts  = str(trend)
        if "Increasing" in ts:
            bullets.append(f"Latest 10-Q risk trend is **increasing**{q_s} — risk language has escalated since the annual filing. Supplemental context only; does not change the model signal.")
        elif "Decreasing" in ts:
            bullets.append(f"Latest 10-Q risk trend is **decreasing**{q_s} — risk language is lower than the annual baseline.")
        elif "Stable" in ts:
            bullets.append(f"Latest 10-Q risk trend is **stable**{q_s} — no meaningful change in risk language since the annual filing.")

    freshness = row.get("market_freshness")
    if freshness and str(freshness) not in ("", "nan", "None"):
        fs = str(freshness).lower()
        if "stale" in fs:
            bullets.append("Current market data is **stale** — current market context may change the valuation picture significantly.")
        elif "missing" in fs:
            bullets.append("Current market data is **missing** — the valuation gap shown uses model-year data only.")

    qflag = str(row.get("output_quality_flag") or "")
    wtext = str(row.get("output_warning_text") or "")
    if qflag not in ("ok", "", "nan", "None") and wtext not in ("", "nan", "None"):
        first_warn = wtext.split("|")[0].strip()
        if first_warn:
            bullets.append(f"**Data quality note:** {first_warn} — interpret this signal with additional caution.")

    calib   = row.get("final_signal_calibrated")
    orig    = row.get("Final_Signal") or row.get("final_signal")
    radj    = row.get("final_signal_calibrated_risk_adjusted")
    _show   = (radj if (radj and str(radj) not in ("nan", "None", ""))
               else (calib if (calib and str(calib) not in ("nan", "None", "")) else orig))
    if _show and str(_show) not in ("nan", "None", ""):
        sig = str(_show)
        if "High-priority" in sig:
            bullets.append("This company appears on the **high-priority shortlist** because valuation gap and quality score both meet the model's top-tier thresholds in the model year.")
        elif "Research candidate" in sig:
            bullets.append("This company appears as a **research candidate** — model signals meet the threshold for further investigation. Additional due diligence is needed.")
        elif "Fairly valued" in sig or "neutral" in sig.lower():
            bullets.append("The model sees this company as **fairly valued** relative to the model-year market — no strong directional signal.")
        elif "value trap" in sig.lower():
            bullets.append("The model flags a **possible value trap** — the low apparent valuation may reflect underlying fundamental weakness.")
        elif "overvalued" in sig.lower():
            bullets.append("The model signal suggests **potential overvaluation** relative to model-year fundamentals.")

    if not bullets:
        bullets.append("Key metrics are missing or incomplete — no explanation can be generated.")
    return bullets


def _render_explanation(row: dict) -> None:
    """Render the explainability panel inline."""
    st.markdown('<div class="section-header">Why this company appears here</div>', unsafe_allow_html=True)
    st.caption("Research context only. Not investment advice. Valuation gap is a model-year estimate, not a forward return prediction.")
    for b in generate_company_explanation(row):
        st.markdown(f"- {b}")


def _render_research_notes_tab(ticker: str) -> None:
    """Render Research Notes tab content for a given ticker."""
    import json
    note = _get_note(ticker)
    updated_at = note.get("updated_at", "")
    if updated_at:
        st.caption(f"Last updated: {updated_at}")

    st.markdown('<div class="section-header">Research Notes</div>', unsafe_allow_html=True)
    thesis    = st.text_area("Thesis", value=note.get("thesis", ""), height=100,
                              placeholder="What is the investment thesis for further researching this company?",
                              key=f"notes_thesis_{ticker}")
    risks     = st.text_area("Key Risks", value=note.get("risks", ""), height=80,
                              placeholder="What are the key risks to the thesis?",
                              key=f"notes_risks_{ticker}")
    questions = st.text_area("Questions to Research", value=note.get("questions", ""), height=80,
                              placeholder="What open questions need to be answered?",
                              key=f"notes_questions_{ticker}")
    decision  = st.selectbox(
        "Research Decision", _NOTES_DECISIONS,
        index=_NOTES_DECISIONS.index(note.get("decision", "No decision"))
              if note.get("decision", "") in _NOTES_DECISIONS else 0,
        key=f"notes_decision_{ticker}",
    )
    st.markdown("**Research Checklist**")
    try:
        saved_checks = json.loads(note.get("checklist_json", "{}") or "{}")
    except Exception:
        saved_checks = {}
    checks = {item: st.checkbox(item, value=bool(saved_checks.get(item, False)),
                                 key=f"chk_{ticker}_{item.replace(' ','_').replace('/','_')}")
              for item in _CHECKLIST_ITEMS}
    if st.button("Save Notes", type="primary", key=f"save_notes_{ticker}"):
        _save_note(ticker, thesis, risks, questions, decision, checks)
        st.success("Notes saved.")
        st.rerun()
    st.caption("Notes are saved locally to research_notes.csv. Educational research only — not investment advice.")


def _render_watchlist_button(ticker: str, row: dict, source_page: str) -> None:
    """Render Add-to-Watchlist or Already-in-Watchlist button."""
    already = _wl_in_watchlist(ticker)
    if already:
        st.button("✓ In Watchlist", key=f"wl_already_{ticker}_{source_page}", disabled=True)
    else:
        st.caption("Save this company to continue research later.")
        if st.button("★ Add to Watchlist", key=f"wl_add_{ticker}_{source_page}", type="primary"):
            _wl_add(
                ticker,
                company_name=str(row.get("Company_Name") or row.get("company_name") or ticker),
                source_page=source_page,
                model_year=str(row.get("year") or row.get("model_year") or ""),
                signal=str(row.get("final_signal_display") or row.get("Final_Signal") or row.get("final_signal") or ""),
                val_gap=row.get("Valuation_Gap_Pct") or row.get("valuation_gap_pct"),
                quality=row.get("Quality_Score") or row.get("quality_score"),
                risk=row.get("report_risk_score_real") or row.get("Report_Risk_Score") or row.get("report_risk_score"),
                q10_risk=row.get("latest_10q_risk_score"),
                q10_trend=str(row.get("filing_risk_trend") or ""),
            )
            st.success(f"Added {ticker} to Watchlist.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 -- HOME
# ════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    # Hero
    st.markdown("""
    <div class="hero-banner">
      <div style="font-size:11px;color:#93c5fd;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">UNDERDAWG</div>
      <h1 style="font-size:26px;margin:0 0 8px 0;color:white">Turn scattered stock research<br>into a ranked, explainable shortlist.</h1>
      <p style="font-size:15px;color:#c7d8f5;margin:0">
        Screen companies, understand why each is flagged, and test ideas without real money.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Integrated CTA preview cards ──────────────────────────────────────────
    all_mo, _ = _load_all_model_outputs()
    card1, card2 = st.columns(2)

    with card1:
        _scr_inner = ""
        if all_mo is not None:
            try:
                _prev_df = all_mo.copy()
                _prev_df = _prev_df.sort_values("year").groupby("ticker", as_index=False).last()
                _sig_pri = {
                    "High-priority research candidate": 0, "Research candidate": 0,
                    "Fairly valued / neutral": 1, "Possible value trap": 2, "Potentially overvalued": 3,
                }
                _sig_col = (
                    "final_signal_calibrated_risk_adjusted"
                    if "final_signal_calibrated_risk_adjusted" in _prev_df.columns
                    else ("final_signal_calibrated" if "final_signal_calibrated" in _prev_df.columns
                          else "final_signal")
                )
                _prev_df["_pri"] = _prev_df[_sig_col].fillna(_prev_df["final_signal"]).map(_sig_pri).fillna(99)
                _prev_df = _prev_df.sort_values(["_pri", "valuation_gap_pct"], ascending=[True, False])
                _top5 = _prev_df.head(5)
                _max_g_raw = _top5["valuation_gap_pct"].abs().max()
                _max_g = float(_max_g_raw) if pd.notna(_max_g_raw) and float(_max_g_raw) > 0 else 100.0
                _bars = ""
                for _, _row in _top5.iterrows():
                    _g = _row.get("valuation_gap_pct")
                    _t = str(_row.get("ticker", ""))[:5]
                    if pd.notna(_g):
                        _w = min(max(abs(float(_g)) / _max_g * 80, 6), 90)
                        _bc = "#2563eb" if float(_g) > 0 else "#ef4444"
                        _bars += (
                            f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">'
                            f'<span style="font-size:10px;color:#64748b;width:34px;flex-shrink:0;font-family:monospace">{_t}</span>'
                            f'<div style="background:{_bc};height:7px;border-radius:3px;width:{_w:.0f}%"></div>'
                            f'<span style="font-size:10px;color:#334155">{float(_g):+.0f}%</span>'
                            f'</div>'
                        )
                _top = _prev_df.iloc[0]
                _top_ticker = str(_top.get("ticker", ""))
                _gap  = _top.get("valuation_gap_pct")
                _qs   = _top.get("quality_score")
                _rrs  = _top.get("report_risk_score_real") if pd.notna(_top.get("report_risk_score_real", None)) else _top.get("report_risk_score")
                _top_gap_s = f"{float(_gap):+.1f}%" if pd.notna(_gap) else "N/A"
                _top_qs_s  = f"{float(_qs):.0f}/100" if pd.notna(_qs) else "N/A"
                _top_rrs_s = f"{float(_rrs):.0f}/100" if pd.notna(_rrs) else "N/A"
                _scr_inner = (
                    '<div style="font-size:10px;color:#64748b;margin-bottom:5px;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:0.5px">Valuation gap · top candidates</div>'
                    + _bars
                    + f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #e2e8f0">'
                    f'<span style="font-size:12px;font-weight:700;color:#1a2744">{_top_ticker}</span>'
                    f'<span style="font-size:11px;color:#64748b"> · Gap: <b>{_top_gap_s}</b>'
                    f' · Quality: <b>{_top_qs_s}</b> · Risk: <b>{_top_rrs_s}</b></span></div>'
                )
            except Exception:
                _scr_inner = '<div style="font-size:12px;color:#64748b">Open Screener to see ranked research candidates.</div>'
        else:
            _scr_inner = (
                '<div style="font-size:12px;color:#64748b">Run the model pipeline to see live candidates here.'
                '<br><span style="font-size:11px;color:#94a3b8">market_cap_utils.py · calibrate_signals.py</span></div>'
            )
        st.markdown(
            '<div class="cta-card"><div class="cta-card-body">'
            '<div class="cta-eyebrow">Research Screener</div>'
            '<div class="cta-title">Open Screener</div>'
            '<div class="cta-subtitle">Rank companies by valuation, quality, and filing risk.</div>'
            f'<div class="cta-preview-box">{_scr_inner}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Research Screener →", key="home_cta_screener", use_container_width=True):
            st.session_state.pending_nav_page = "Screener"
            st.rerun()

    with card2:
        _rpl_inner = ""
        if all_mo is not None:
            try:
                avail_yrs = sorted(all_mo["year"].dropna().astype(int).unique())
                n_cos = all_mo["ticker"].nunique()
                yr_range = f"{avail_yrs[0]}–{avail_yrs[-1]}" if len(avail_yrs) > 1 else str(avail_yrs[0])
                _rpl_inner = (
                    '<div style="font-size:10px;color:#64748b;margin-bottom:5px;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:0.5px">Signal coverage</div>'
                    '<svg width="100%" height="50" viewBox="0 0 240 50" '
                    'style="display:block;margin-bottom:10px;overflow:visible">'
                    '<defs><linearGradient id="sparkgrad" x1="0" y1="0" x2="1" y2="0">'
                    '<stop offset="0%" stop-color="#bfdbfe"/>'
                    '<stop offset="100%" stop-color="#2563eb"/></linearGradient>'
                    '<linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1">'
                    '<stop offset="0%" stop-color="#2563eb" stop-opacity="0.12"/>'
                    '<stop offset="100%" stop-color="#2563eb" stop-opacity="0"/></linearGradient></defs>'
                    '<polygon points="0,46 48,38 96,28 144,18 192,10 240,4 240,50 0,50" '
                    'fill="url(#sparkfill)"/>'
                    '<polyline points="0,46 48,38 96,28 144,18 192,10 240,4" fill="none" '
                    'stroke="url(#sparkgrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
                    '<circle cx="240" cy="4" r="4" fill="#2563eb"/>'
                    '</svg>'
                    f'<table style="font-size:11px;color:#334155;border-spacing:0 4px;width:100%">'
                    f'<tr><td style="color:#64748b;padding-right:10px;padding-bottom:2px">Years</td><td><b>{yr_range}</b></td>'
                    f'<td style="color:#64748b;padding-left:12px;padding-right:6px">Companies</td><td><b>{n_cos}</b></td></tr>'
                    f'<tr><td style="color:#64748b;padding-bottom:2px">Benchmarks</td><td colspan="3"><b>SPY · QQQ · Cash</b></td></tr>'
                    f'<tr><td style="color:#64748b">Holding</td><td colspan="3"><b>1 month – 10 years · exact-date entry</b></td></tr>'
                    f'</table>'
                )
            except Exception:
                _rpl_inner = '<div style="font-size:12px;color:#64748b">Replay a model shortlist over any historical date range.</div>'
        else:
            _rpl_inner = (
                '<div style="font-size:12px;color:#64748b">Replay a model shortlist over any historical date range using daily prices.'
                '<br><span style="font-size:11px;color:#94a3b8">Holding: 1 month to 10 years · exact-date entry</span></div>'
            )
        st.markdown(
            '<div class="cta-card"><div class="cta-card-body">'
            '<div class="cta-eyebrow">Portfolio Simulator</div>'
            '<div class="cta-title">Try Historical Replay</div>'
            '<div class="cta-subtitle">Pick a date and replay model-backed signals against a benchmark.</div>'
            f'<div class="cta-preview-box">{_rpl_inner}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Run Historical Replay →", key="home_cta_replay", use_container_width=True):
            st.session_state.pending_nav_page = "Portfolio Simulator"
            st.session_state["portfolio_tab"] = "Historical Replay"
            st.rerun()

    st.markdown('<div style="height:120px"></div>', unsafe_allow_html=True)

    # Disclaimer strip
    st.markdown("""
    <div class="disclaimer-strip">
    <b>Disclaimer:</b> Underdawg is for educational and research support only.
    It does not provide investment advice, buy/sell recommendations, or performance guarantees.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Full disclaimer"):
        st.markdown("""
        Signal labels (e.g., "Research candidate") indicate where the model suggests investing *research time*, not capital.
        The valuation gap is a research signal, not proof of undervaluation.
        Past simulated performance does not predict future real-world results.
        Always conduct your own due diligence and consult a qualified financial advisor.
        """)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 -- SCREENER
# ════════════════════════════════════════════════════════════════════════════

def page_screener(df: pd.DataFrame) -> None:
    st.title("Research Screener")
    st.markdown('<div class="section-subtitle">Find companies worth deeper review using valuation, quality, and filing-risk signals.</div>', unsafe_allow_html=True)

    # Determine data source — default to model outputs, sample data as hidden fallback
    use_xbrl = mo_df is not None
    active_df = mo_df.copy() if use_xbrl else df.copy()

    if use_xbrl:
        if "final_signal_display" in active_df.columns:
            active_df["Final_Signal"] = active_df["final_signal_display"].fillna(active_df["Final_Signal"])
        if cm_df is not None and len(cm_df) > 0:
            cm_slim = cm_df[["ticker", "current_price", "current_market_cap",
                             "daily_change_pct", "last_updated"]].copy()
            cm_slim = cm_slim.rename(columns={"ticker": "Ticker"})
            cm_slim["Ticker"] = cm_slim["Ticker"].str.upper()
            active_df = active_df.merge(cm_slim, on="Ticker", how="left")
        else:
            for col in ["current_price", "current_market_cap", "daily_change_pct"]:
                active_df[col] = np.nan
    else:
        st.info("Model outputs unavailable. Using demo fallback data.")

    # Merge 10-Q risk update if available (supplemental — does not alter model signals)
    _q10_df, _upd_df = _load_10q_data()
    if use_xbrl and _upd_df is not None and len(_upd_df) > 0:
        try:
            _upd_slim = _upd_df[["ticker", "latest_10q_risk_score", "filing_risk_trend",
                                   "filing_risk_delta", "latest_10q_filing_date"]].copy()
            _upd_slim["ticker"] = _upd_slim["ticker"].str.upper()
            _upd_slim = _upd_slim.rename(columns={"ticker": "Ticker"})
            active_df = active_df.merge(_upd_slim, on="Ticker", how="left")
        except Exception:
            for _c in ["latest_10q_risk_score", "filing_risk_trend", "filing_risk_delta", "latest_10q_filing_date"]:
                active_df[_c] = None

    # Merge universe_group into active_df for screener filter
    try:
        _uni_df, _ = _load_ticker_universe_cached()
        if not _uni_df.empty and "universe_group" in _uni_df.columns:
            _uni_slim = _uni_df[["ticker", "universe_group"]].copy()
            _uni_slim["Ticker"] = _uni_slim["ticker"].str.upper()
            _uni_slim = _uni_slim.drop(columns="ticker")
            active_df = active_df.merge(_uni_slim, on="Ticker", how="left")
            active_df["universe_group"] = active_df["universe_group"].fillna("current_demo")
    except Exception:
        active_df["universe_group"] = "current_demo"

    # Current valuation gap using live market cap vs model fair value
    if use_xbrl and "Estimated_Fair_Value" in active_df.columns:
        _cmask = (
            active_df.get("current_market_cap", pd.Series(dtype=float)).notna()
            & (active_df.get("current_market_cap", pd.Series(dtype=float)) > 0)
            & active_df["Estimated_Fair_Value"].notna()
        )
        active_df["current_valuation_gap_pct"] = np.nan
        if _cmask.any():
            active_df.loc[_cmask, "current_valuation_gap_pct"] = (
                (active_df.loc[_cmask, "Estimated_Fair_Value"] * 1e9
                 - active_df.loc[_cmask, "current_market_cap"])
                / active_df.loc[_cmask, "current_market_cap"] * 100
            ).round(1)
        active_df["valuation_gap_source"] = np.where(_cmask, "current_mkt", "model_yr")

    # Per-ticker market freshness flag (fresh ≤5h, stale >5h, missing)
    if "last_updated" in active_df.columns:
        def _fresh_flag(ts):
            try:
                age_h = (pd.Timestamp.now() - pd.to_datetime(ts)).total_seconds() / 3600
                return "fresh" if age_h <= 5 else "stale"
            except Exception:
                return "unknown"
        active_df["market_freshness"] = active_df["last_updated"].apply(
            lambda x: _fresh_flag(x) if pd.notna(x) else "missing"
        )
    else:
        active_df["market_freshness"] = "missing"

    # Status metric cards
    if use_xbrl:
        _all_mo, _ = _load_all_model_outputs()
        _n_companies = int(active_df["Ticker"].nunique())
        _n_rows_all  = len(_all_mo) if _all_mo is not None else _n_companies
        _yrs_all     = sorted(_all_mo["year"].dropna().astype(int).unique()) if _all_mo is not None else []
        _yr_range    = f"{_yrs_all[0]}–{_yrs_all[-1]}" if len(_yrs_all) > 1 else (str(_yrs_all[0]) if _yrs_all else "?")
        _n_flagged   = int(active_df["output_quality_flag"].str.contains("needs_review", na=False).sum()) \
                       if "output_quality_flag" in active_df.columns else 0
        _has_risk_m  = "report_risk_available" in active_df.columns and active_df["report_risk_available"].any()
        _risk_ct     = int(active_df["report_risk_available"].sum()) if _has_risk_m else 0
        _has_10q_m   = "latest_10q_risk_score" in active_df.columns and active_df["latest_10q_risk_score"].notna().any()
        _10q_ct      = int(active_df["latest_10q_risk_score"].notna().sum()) if _has_10q_m else 0
        _fresh_ct    = int((active_df.get("market_freshness", pd.Series()) == "fresh").sum())
        _cm_any      = _fresh_ct > 0 or int((active_df.get("market_freshness", pd.Series()) == "stale").sum()) > 0

        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Companies",         _n_companies)
        sc2.metric("Model year range",  _yr_range)
        sc3.metric("10-K risk scores",  _risk_ct if _has_risk_m else "—",
                   help="Annual 10-K filing risk scores used in model-backed signals.")
        sc4.metric("10-Q risk scores",  _10q_ct if _has_10q_m else "—",
                   help="Latest quarterly 10-Q risk update (supplemental, does not alter model signal).")
        sc5.metric("Mkt data fresh",    _fresh_ct if _cm_any else "—",
                   help="Tickers with current market data fetched within the last 5 hours.")
        sc6.metric("Flagged rows",      _n_flagged)

        # Market snapshot freshness banner + refresh
        _mts_label   = "Current market unavailable"
        _mts_color   = "#64748b"
        _cm_available = "last_updated" in active_df.columns and active_df["last_updated"].notna().any()
        if _cm_available:
            try:
                _mts = pd.to_datetime(active_df["last_updated"].dropna().iloc[0])
                _age_h = (pd.Timestamp.now() - _mts).total_seconds() / 3600
                _ts_short = str(_mts)[:16]
                if _age_h <= 5:
                    _mts_label = f"Market snapshot fresh — as of {_ts_short}"
                    _mts_color = "#16a34a"
                else:
                    _mts_label = f"Market snapshot stale — as of {_ts_short} ({_age_h:.0f}h ago)"
                    _mts_color = "#d97706"
            except Exception:
                pass
        _fb1, _fb2 = st.columns([4, 1])
        with _fb1:
            st.markdown(
                f'<div style="font-size:12px;color:{_mts_color};margin:6px 0 2px 0">● {_mts_label}</div>'
                '<div style="font-size:11px;color:#94a3b8">Fundamentals and filing risk update when '
                'companies file reports. Market price context updates via yfinance snapshot.</div>',
                unsafe_allow_html=True,
            )
        with _fb2:
            if st.button("Refresh market snapshot", key="scr_refresh_mkt", use_container_width=True):
                try:
                    _tkrs = active_df["Ticker"].dropna().tolist()
                    with st.spinner("Fetching current prices …"):
                        fetch_current_market_data(_tkrs, save=True)
                    _load_supplementary.clear()
                    st.rerun()
                except Exception as _re:
                    st.warning(f"Refresh failed: {_re}")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Signal toggles
    hide_flagged = False
    use_calibrated = False
    use_risk_adjusted = False

    if use_xbrl and "output_quality_flag" in active_df.columns:
        tgl1, tgl2, tgl3 = st.columns(3)
        with tgl1:
            hide_flagged = st.toggle(
                "Hide needs_review rows", value=True, key="screener_hide_flagged",
                help="Rows flagged as needs_review have suspicious model outputs.",
            )
        flagged_mask = active_df["output_quality_flag"].str.contains("needs_review", na=False)
        n_flagged_total = int(flagged_mask.sum())
        if hide_flagged and n_flagged_total > 0:
            active_df = active_df[~flagged_mask].copy()

        _has_calibrated = (
            "final_signal_calibrated" in active_df.columns
            and active_df["final_signal_calibrated"].notna().any()
        )
        if _has_calibrated:
            with tgl2:
                use_calibrated = st.toggle(
                    "Use calibrated signals", value=True, key="screener_use_calibrated",
                    help="Calibrated signals compare each company against same-year peers.",
                )
            if use_calibrated:
                active_df["Final_Signal"] = (
                    active_df["final_signal_calibrated"].fillna(active_df["Final_Signal"])
                )

        _has_risk_adjusted = (
            use_calibrated
            and "final_signal_calibrated_risk_adjusted" in active_df.columns
            and active_df["final_signal_calibrated_risk_adjusted"].notna().any()
        )
        if _has_risk_adjusted:
            with tgl3:
                use_risk_adjusted = st.toggle(
                    "Use filing-risk adjusted signals", value=True, key="screener_use_risk_adjusted",
                    help="Applies SEC 10-K risk scores. Very high risk may downgrade Research candidate.",
                )
            if use_risk_adjusted:
                active_df["Final_Signal"] = (
                    active_df["final_signal_calibrated_risk_adjusted"].fillna(active_df["Final_Signal"])
                )

    _has_current_mc = (
        "current_market_cap" in active_df.columns
        and active_df["current_market_cap"].notna().any()
        and "current_valuation_gap_pct" in active_df.columns
    )
    _has_10q_data = (
        "latest_10q_risk_score" in active_df.columns
        and active_df["latest_10q_risk_score"].notna().any()
    )
    use_current_gap = False
    show_10q_risk = False
    if use_xbrl and (_has_current_mc or _has_10q_data):
        tgl4, tgl5, _ = st.columns(3)
        if _has_current_mc:
            with tgl4:
                use_current_gap = st.toggle(
                    "Use current market cap gap",
                    value=True,
                    key="screener_use_current_gap",
                    help="Show valuation gap using live market cap vs model fair value instead of model-year cap.",
                )
        if _has_10q_data:
            with tgl5:
                show_10q_risk = st.toggle(
                    "Show 10-Q risk update",
                    value=True,
                    key="screener_show_10q",
                    help="Show latest quarterly 10-Q risk score alongside annual 10-K filing risk.",
                )

    # Filter panel — collapsed by default
    with st.expander("Filters and sorting", expanded=False):
        fp1, fp2, fp3 = st.columns(3)
        with fp1:
            ticker_search = st.text_input("Search ticker / company", value="", key="flt_search", placeholder="e.g. AAPL")
        with fp2:
            sig_opts = list(SIGNAL_ORDER)
            if use_xbrl and not hide_flagged:
                sig_opts = sig_opts + ["Needs review"]
            sel_sigs = st.multiselect("Signal", sig_opts, default=sig_opts, key="flt_sig")
        with fp3:
            sectors = sorted(active_df["Sector"].dropna().unique())
            sel_sec = st.multiselect("Sector", sectors, default=sectors, key="flt_sec")

        fp4, fp5, fp6 = st.columns(3)
        with fp4:
            q_vals = active_df["Quality_Score"].dropna()
            min_q  = int(q_vals.min()) if len(q_vals) > 0 else 0
            max_q  = int(q_vals.max()) if len(q_vals) > 0 else 100
            q_range = st.slider("Min quality score", min_q, max_q, min_q, key="flt_q")
        with fp5:
            _risk_col = ("report_risk_score_real"
                         if "report_risk_score_real" in active_df.columns
                         and active_df["report_risk_score_real"].notna().any()
                         else "Report_Risk_Score")
            if _risk_col in active_df.columns:
                rsk_max_filter = st.slider("Max filing risk score", 0, 100, 100, key="flt_rsk")
            else:
                rsk_max_filter = 100
        with fp6:
            sort_by = st.selectbox(
                "Sort by",
                ["Research priority", "Valuation gap", "Current gap", "Quality score", "Filing risk", "10-Q Risk", "Market cap", "Year"],
                key="flt_sort",
            )

        # Universe group filter — only shown when multiple groups are present
        sel_uni_groups: list = []
        _uni_groups_avail = (
            sorted(active_df["universe_group"].dropna().unique().tolist())
            if "universe_group" in active_df.columns else []
        )
        if len(_uni_groups_avail) > 1:
            _ug1, _ug2, _ = st.columns(3)
            with _ug1:
                sel_uni_groups = st.multiselect(
                    "Universe group",
                    _uni_groups_avail,
                    default=_uni_groups_avail,
                    key="flt_uni_group",
                    help="Filter by universe group (current_demo, expanded_100, custom).",
                )

        sel_freshness: list = []
        sel_trend: list = []
        _has_freshness_col = "market_freshness" in active_df.columns
        _has_trend_col = "filing_risk_trend" in active_df.columns and active_df["filing_risk_trend"].notna().any()
        if use_xbrl and (_has_freshness_col or _has_trend_col):
            fp7, fp8, _ = st.columns(3)
            if _has_freshness_col:
                with fp7:
                    _fresh_opts = sorted(active_df["market_freshness"].dropna().unique().tolist())
                    sel_freshness = st.multiselect(
                        "Market data freshness",
                        _fresh_opts,
                        default=_fresh_opts,
                        key="flt_freshness",
                        help="Filter by how recent the live market data is.",
                    )
            if _has_trend_col:
                with fp8:
                    _trend_opts = ["Increasing", "Stable", "Decreasing", "Unavailable"]
                    _trend_avail = [t for t in _trend_opts if t in active_df["filing_risk_trend"].values or t == "Unavailable"]
                    sel_trend = st.multiselect(
                        "Filing risk trend",
                        _trend_avail,
                        default=_trend_avail,
                        key="flt_trend",
                        help="Filter by 10-Q vs 10-K risk trend direction.",
                    )

    # Apply filters
    fdf = active_df[
        active_df["Final_Signal"].isin(sel_sigs) &
        active_df["Sector"].isin(sel_sec) &
        (active_df["Quality_Score"] >= q_range)
    ].copy()

    if ticker_search.strip():
        _srch = ticker_search.strip().upper()
        _mask = (
            fdf["Ticker"].str.upper().str.contains(_srch, na=False) |
            fdf["Company_Name"].str.upper().str.contains(_srch, na=False)
        )
        fdf = fdf[_mask].copy()

    if _risk_col in fdf.columns:
        fdf = fdf[fdf[_risk_col].fillna(0) <= rsk_max_filter].copy()

    if sel_uni_groups and "universe_group" in fdf.columns:
        fdf = fdf[fdf["universe_group"].isin(sel_uni_groups)].copy()

    if sel_freshness and "market_freshness" in fdf.columns:
        fdf = fdf[fdf["market_freshness"].isin(sel_freshness)].copy()

    if sel_trend and "filing_risk_trend" in fdf.columns:
        _trend_mask = fdf["filing_risk_trend"].isin([t for t in sel_trend if t != "Unavailable"])
        if "Unavailable" in sel_trend:
            _trend_mask = _trend_mask | fdf["filing_risk_trend"].isna()
        fdf = fdf[_trend_mask].copy()

    # Sort
    if sort_by == "Research priority":
        fdf["_pri"] = fdf["Final_Signal"].map({s: i for i, s in enumerate(SIGNAL_ORDER)})
        fdf = fdf.sort_values("_pri").drop(columns="_pri")
    elif sort_by == "Valuation gap":
        if use_current_gap and "current_valuation_gap_pct" in fdf.columns:
            fdf = fdf.sort_values("current_valuation_gap_pct", ascending=False)
        else:
            fdf = fdf.sort_values("Valuation_Gap_Pct", ascending=False)
    elif sort_by == "Current gap":
        _gap_sort_col = "current_valuation_gap_pct" if "current_valuation_gap_pct" in fdf.columns else "Valuation_Gap_Pct"
        fdf = fdf.sort_values(_gap_sort_col, ascending=False)
    elif sort_by == "Quality score":
        fdf = fdf.sort_values("Quality_Score", ascending=False)
    elif sort_by == "Filing risk":
        if _risk_col in fdf.columns:
            fdf = fdf.sort_values(_risk_col, ascending=False)
    elif sort_by == "10-Q Risk":
        if "latest_10q_risk_score" in fdf.columns:
            fdf = fdf.sort_values("latest_10q_risk_score", ascending=False)
    elif sort_by == "Market cap":
        fdf = fdf.sort_values("Market_Cap_B", ascending=False)
    elif sort_by == "Year":
        fdf = fdf.sort_values("year", ascending=False)
    fdf = fdf.reset_index(drop=True)

    # Signal distribution summary strip
    counts = {s: len(fdf[fdf["Final_Signal"] == s]) for s in SIGNAL_ORDER}
    summary_labels = ["High-Priority", "Research Candidate", "Fairly Valued", "Pot. Overvalued", "Value Trap"]
    scols = st.columns(5)
    for col, sig, lbl in zip(scols, SIGNAL_ORDER, summary_labels):
        with col:
            color = SIGNAL_COLORS[sig]
            st.markdown(
                f'<div class="mini-card" style="border-left: 4px solid {color}">'
                f'<div class="label">{lbl}</div>'
                f'<div class="value" style="color:{color}">{counts[sig]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Build display table
    if use_xbrl:
        desired = ["Ticker", "Company_Name", "year", "Market_Cap_B", "Estimated_Fair_Value",
                   "Valuation_Gap_Pct", "Quality_Score", "Final_Signal"]
        # Current market cap and gap: only when toggle is on and data is available
        if use_current_gap and "current_valuation_gap_pct" in fdf.columns and fdf["current_valuation_gap_pct"].notna().any():
            desired.insert(desired.index("Valuation_Gap_Pct") + 1, "current_valuation_gap_pct")
            if "current_market_cap" in fdf.columns:
                desired.insert(desired.index("Market_Cap_B") + 1, "current_market_cap")
        if use_calibrated and "valuation_bucket_year" in fdf.columns:
            desired.insert(desired.index("Final_Signal") + 1, "valuation_bucket_year")
        if _risk_col in fdf.columns:
            desired.append(_risk_col)
        if show_10q_risk and "latest_10q_risk_score" in fdf.columns and fdf["latest_10q_risk_score"].notna().any():
            desired += ["latest_10q_risk_score", "filing_risk_trend"]
        if "output_quality_flag" in fdf.columns:
            desired.append("output_quality_flag")
        if "current_price" in fdf.columns:
            desired += ["current_price", "daily_change_pct"]
        col_rename = {
            "Ticker": "Ticker", "Company_Name": "Company", "year": "Model Yr",
            "Market_Cap_B": "Model MC (B)", "current_market_cap": "Live MC (B)",
            "Estimated_Fair_Value": "Est. Value (B)",
            "Valuation_Gap_Pct": "Model Gap %", "current_valuation_gap_pct": "Current Gap %",
            "Quality_Score": "Quality",
            "Final_Signal": "Signal", "valuation_bucket_year": "Value Bucket",
            "output_quality_flag": "Data Quality",
            "report_risk_score_real": "Filing Risk", "Report_Risk_Score": "Filing Risk",
            "latest_10q_risk_score": "10-Q Risk", "filing_risk_trend": "Risk Trend",
            "current_price": "Live Price", "daily_change_pct": "Daily Chg %",
        }
    else:
        desired = ["Ticker", "Company_Name", "Sector", "Current_Price", "Estimated_Fair_Value",
                   "Valuation_Gap_Pct", "Quality_Score", "Report_Risk_Score", "Final_Signal"]
        col_rename = {
            "Ticker": "Ticker", "Company_Name": "Company", "Sector": "Sector",
            "Current_Price": "Market Price ($)", "Estimated_Fair_Value": "Est. Fair Value ($)",
            "Valuation_Gap_Pct": "Valuation Gap (%)", "Quality_Score": "Quality Score",
            "Report_Risk_Score": "Report Risk Score", "Final_Signal": "Signal",
        }

    src_cols = [c for c in desired if c in fdf.columns]
    display  = fdf[src_cols].copy().rename(columns={k: v for k, v in col_rename.items() if k in src_cols})

    def _fmt_b(x):
        return f"${x:.1f}B" if pd.notna(x) else "N/A"
    def _fmt_pct(x):
        return f"{x:+.1f}%" if pd.notna(x) else "N/A"

    if "Model MC (B)"        in display.columns: display["Model MC (B)"]        = display["Model MC (B)"].map(_fmt_b)
    if "Live MC (B)"         in display.columns: display["Live MC (B)"]         = display["Live MC (B)"].map(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A")
    if "Est. Value (B)"      in display.columns: display["Est. Value (B)"]      = display["Est. Value (B)"].map(_fmt_b)
    if "Model Gap %"         in display.columns: display["Model Gap %"]         = display["Model Gap %"].map(_fmt_pct)
    if "Current Gap %"       in display.columns: display["Current Gap %"]       = display["Current Gap %"].map(_fmt_pct)
    if "Daily Chg %"         in display.columns: display["Daily Chg %"]         = display["Daily Chg %"].map(_fmt_pct)
    if "Live Price"          in display.columns: display["Live Price"]          = display["Live Price"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    if "Market Price ($)"    in display.columns: display["Market Price ($)"]    = display["Market Price ($)"].map("${:.2f}".format)
    if "Est. Fair Value ($)" in display.columns: display["Est. Fair Value ($)"] = display["Est. Fair Value ($)"].map("${:.2f}".format)
    if "Valuation Gap (%)"   in display.columns: display["Valuation Gap (%)"]   = display["Valuation Gap (%)"].map(_fmt_pct)

    def _gap_label(v):
        try: v = float(v)
        except: return "N/A"
        s = "+" if v >= 0 else ""
        if   v > 20:   desc = "High"
        elif v > 5:    desc = "Favorable"
        elif v >= -5:  desc = "Neutral"
        elif v >= -20: desc = "Stretched"
        else:          desc = "Deep Stretched"
        return f"{s}{v:.1f}% | {desc}"

    def _risk_label(v):
        try: v = float(v)
        except: return "N/A"
        if   v < 30: level = "Low"
        elif v < 50: level = "Moderate"
        elif v < 70: level = "Elevated"
        else:        level = "High"
        return f"{v:.0f}/100 | {level}"

    def _quality_label(v):
        try: v = float(v)
        except: return "N/A"
        return f"{v:.0f}/100"

    def _trend_label(v):
        s = str(v).strip().lower() if pd.notna(v) else ""
        if   "decr" in s: return "Decreasing"
        elif "incr" in s: return "Increasing"
        elif "stab" in s: return "Stable"
        elif s:           return s.title()
        return "Unavailable"

    _dq_disp = {
        "needs_review":        "Needs Review",
        "neutral_report_risk": "Neutral Report Risk",
        "clean":               "Clean",
        "ok":                  "Clean",
    }
    _sig_disp = {
        "High-priority research candidate": "High-Priority Candidate",
        "Research candidate":               "Research Candidate",
        "Fairly valued / neutral":          "Fairly Valued",
        "Potentially overvalued":           "Potentially Overvalued",
        "Possible value trap":              "Value Trap",
    }
    _vb_disp = {
        "top decile relative value":     "Top Decile",
        "above-average relative value":  "Above Average",
        "neutral relative value":        "Neutral",
        "below-average relative value":  "Below Average",
        "low relative value":            "Low",
    }

    # Gap columns — enriched labels from raw fdf
    if "Model Gap %"   in display.columns and "Valuation_Gap_Pct"         in fdf.columns:
        display["Model Gap %"]   = fdf["Valuation_Gap_Pct"].map(_gap_label)
    if "Current Gap %" in display.columns and "current_valuation_gap_pct" in fdf.columns:
        display["Current Gap %"] = fdf["current_valuation_gap_pct"].map(
            lambda v: _gap_label(v) if pd.notna(v) else "N/A")

    # Risk / quality — text labels (no ProgressColumn)
    if "Filing Risk" in display.columns:
        display["Filing Risk"] = display["Filing Risk"].map(_risk_label)
    if "10-Q Risk"   in display.columns:
        display["10-Q Risk"]   = display["10-Q Risk"].map(_risk_label)
    if "Quality"     in display.columns:
        display["Quality"]     = display["Quality"].map(_quality_label)

    # Text columns — normalize labels
    if "Risk Trend"   in display.columns and "filing_risk_trend" in fdf.columns:
        display["Risk Trend"]   = fdf["filing_risk_trend"].map(_trend_label)
    if "Data Quality" in display.columns:
        display["Data Quality"] = display["Data Quality"].map(
            lambda v: _dq_disp.get(str(v).strip().lower(), str(v).strip().title()) if pd.notna(v) else "—")
    if "Signal"       in display.columns:
        display["Signal"]       = display["Signal"].map(
            lambda v: _sig_disp.get(str(v).strip(), str(v).strip()) if pd.notna(v) else "—")
    if "Value Bucket" in display.columns:
        display["Value Bucket"] = display["Value Bucket"].map(
            lambda v: _vb_disp.get(str(v).strip().lower(), str(v).strip().title()) if pd.notna(v) else "—")

    _col_cfg = {}
    if "Ticker"         in display.columns: _col_cfg["Ticker"]         = st.column_config.TextColumn("Ticker",       width="small")
    if "Company"        in display.columns: _col_cfg["Company"]        = st.column_config.TextColumn("Company",      width="medium")
    if "Model Yr"       in display.columns: _col_cfg["Model Yr"]       = st.column_config.NumberColumn("Yr",          format="%d",  width="small")
    if "Model MC (B)"   in display.columns: _col_cfg["Model MC (B)"]   = st.column_config.TextColumn("Model MC",     width="small")
    if "Live MC (B)"    in display.columns: _col_cfg["Live MC (B)"]    = st.column_config.TextColumn("Live MC",      width="small")
    if "Est. Value (B)" in display.columns: _col_cfg["Est. Value (B)"] = st.column_config.TextColumn("Est. Val.",    width="small")
    if "Model Gap %"    in display.columns: _col_cfg["Model Gap %"]    = st.column_config.TextColumn("Model Gap",    width="medium")
    if "Current Gap %"  in display.columns: _col_cfg["Current Gap %"]  = st.column_config.TextColumn("Current Gap",  width="medium")
    if "Quality"        in display.columns: _col_cfg["Quality"]        = st.column_config.TextColumn("Quality",      width="small")
    if "Signal"         in display.columns: _col_cfg["Signal"]         = st.column_config.TextColumn("Signal",       width="medium")
    if "Value Bucket"   in display.columns: _col_cfg["Value Bucket"]   = st.column_config.TextColumn("Val. Bucket",  width="medium")
    if "Filing Risk"    in display.columns: _col_cfg["Filing Risk"]    = st.column_config.TextColumn("10-K Risk",    width="medium")
    if "10-Q Risk"      in display.columns: _col_cfg["10-Q Risk"]      = st.column_config.TextColumn("10-Q Risk",    width="medium")
    if "Risk Trend"     in display.columns: _col_cfg["Risk Trend"]     = st.column_config.TextColumn("Trend",        width="medium")
    if "Data Quality"   in display.columns: _col_cfg["Data Quality"]   = st.column_config.TextColumn("Data Quality", width="medium")
    if "Live Price"     in display.columns: _col_cfg["Live Price"]     = st.column_config.TextColumn("Price",        width="small")
    if "Daily Chg %"    in display.columns: _col_cfg["Daily Chg %"]    = st.column_config.TextColumn("Daily Chg",    width="small")

    # ── Label maps built BEFORE table so we can sync the dropdown key atomically ──
    visible_tickers = fdf["Ticker"].dropna().astype(str).tolist()
    if visible_tickers:
        _co_map          = fdf.set_index("Ticker")["Company_Name"].to_dict()
        _ticker_to_label = {t: f"{t} — {_co_map.get(t, t)}" for t in visible_tickers}
        _label_to_ticker = {v: k for k, v in _ticker_to_label.items()}
        visible_labels   = [_ticker_to_label[t] for t in visible_tickers]

        # Canonical state — reset both keys + clear stale table selection when ticker falls out of view
        if st.session_state.get("screener_selected_ticker") not in visible_tickers:
            _reset_t = visible_tickers[0]
            st.session_state["screener_selected_ticker"]              = _reset_t
            st.session_state["screener_selected_company_dropdown"]    = _ticker_to_label[_reset_t]
            st.session_state["_screener_tbl_gen"]                     = st.session_state.get("_screener_tbl_gen", 0) + 1

        st.caption("Click a row to inspect that company below.  Display labels are research context, not investment advice.")

    # Generation-keyed table: incrementing the key resets visual selection state
    _tbl_key = f"screener_results_table_{st.session_state.get('_screener_tbl_gen', 0)}"
    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
        key=_tbl_key,
        column_config=_col_cfg,
    )

    # Row-click → update BOTH canonical keys, then rerun so dropdown rebuilds with correct index
    if visible_tickers:
        _sel_rows = getattr(event.selection, "rows", []) if event else []
        if _sel_rows and 0 <= _sel_rows[0] < len(visible_tickers):
            _clicked = visible_tickers[_sel_rows[0]]
            if st.session_state["screener_selected_ticker"] != _clicked:
                st.session_state["screener_selected_ticker"]           = _clicked
                st.session_state["screener_selected_company_dropdown"] = _ticker_to_label[_clicked]
                st.rerun()

    # CSV download
    st.download_button(
        "Download filtered results (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="screener_results.csv",
        mime="text/csv",
        key="screener_dl_csv",
    )

    if use_xbrl:
        _cap_parts = [
            "Research candidates based on latest available fundamentals, filing risk, and market snapshot.",
            "Model Yr = year of fundamentals used. Model Gap % = (Est.Value – Model MC) / Model MC.",
        ]
        if use_current_gap and "current_valuation_gap_pct" in fdf.columns and fdf["current_valuation_gap_pct"].notna().any():
            _cap_parts.append("Current Gap % = (Est.Value – Live MC) / Live MC using current market cap (research signal only).")
        if show_10q_risk and "latest_10q_risk_score" in fdf.columns and fdf["latest_10q_risk_score"].notna().any():
            _cap_parts.append("10-Q Risk = latest quarterly filing risk score (supplemental; does not alter model signal).")
        _cap_parts.append("Filing Risk 0–100 (higher = more risk language in annual 10-K).")
        _cap_parts.append("Valuation gap is a research signal, not proof of undervaluation.")
        st.caption("  ".join(_cap_parts))
    else:
        st.caption("Valuation Gap = (Est. Fair Value - Market Price) / Market Price × 100.")

    st.divider()

    # ── Company selection + preview panel ────────────────────────────────────
    if not visible_tickers:
        st.warning("No companies match the current filters.")
        # If the search matches a pending ticker in universe, show a helpful note
        if ticker_search.strip():
            _srch_up = ticker_search.strip().upper()
            try:
                _uni_s, _ = load_ticker_universe()
                if not _uni_s.empty:
                    _uni_match = _uni_s[_uni_s["ticker"].str.upper() == _srch_up]
                    if not _uni_match.empty:
                        _u_s_row  = _uni_match.iloc[0]
                        _u_s_stat = str(_u_s_row.get("data_status", ""))
                        _u_s_name = str(_u_s_row.get("company_name", "")).strip()
                        if _u_s_stat != "available":
                            st.info(
                                f"**{_srch_up}**"
                                f"{' — ' + _u_s_name if _u_s_name else ''} "
                                f"is in the ticker universe (status: **{_u_s_stat}**) "
                                "but model outputs are not yet available. "
                                "Go to **About → Ticker Universe** to run the pipeline."
                            )
            except Exception:
                pass
        st.caption(
            "Screener includes tickers with available model outputs. "
            "Pending tickers are tracked in **About → Ticker Universe**."
        )
    else:
        # Label maps already built in the pre-table block above.
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            chosen_label = st.selectbox(
                "Select a company to inspect:",
                options=visible_labels,
                key="screener_selected_company_dropdown",
            )
        _dropdown_ticker = _label_to_ticker[chosen_label]
        if _dropdown_ticker != st.session_state["screener_selected_ticker"]:
            # Dropdown changed — update canonical state and reset table visual selection
            st.session_state["screener_selected_ticker"] = _dropdown_ticker
            st.session_state["_screener_tbl_gen"]        = st.session_state.get("_screener_tbl_gen", 0) + 1
            st.rerun()
        chosen = st.session_state["screener_selected_ticker"]
        st.session_state.selected_ticker = chosen

        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("View Details", type="primary", use_container_width=True, key="screener_view_detail"):
                st.session_state.detail_source          = "xbrl" if use_xbrl else "sample"
                st.session_state.pending_nav_page        = "Company Detail"
                st.session_state["_scroll_to_top"]       = True
                st.rerun()
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add to Simulator", use_container_width=True, key="screener_add_sim"):
                st.session_state.paper_ticker_prefill = chosen
                st.session_state.paper_source_prefill = "xbrl" if use_xbrl else "sample"
                st.session_state.pending_nav_page = "Portfolio Simulator"
                st.rerun()

        # Always derive row from filtered dataset so it matches what the user sees
        row = fdf[fdf["Ticker"] == chosen].iloc[0]
        _sig_disp = str(row.get("final_signal_display") or row.get("Final_Signal") or "")
        _risk_adj = str(row.get("final_signal_calibrated_risk_adjusted") or "") if use_risk_adjusted else ""
        _show_sig = _risk_adj if (_risk_adj and _risk_adj != "nan") else _sig_disp

        st.markdown('<div class="section-header">Selected Company</div>', unsafe_allow_html=True)
        if use_xbrl:
            st.markdown('<div class="data-label">Historical Model Output</div>', unsafe_allow_html=True)
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("Actual Mkt Cap",  _sfmt(row.get("Market_Cap_B"), ".1f", suffix="B"))
            q2.metric("Est. Fair Value", _sfmt(row.get("Estimated_Fair_Value"), ".1f", suffix="B"))
            q3.metric("Valuation Gap",   _sfmt(row.get("Valuation_Gap_Pct"), "+.1f", prefix="", suffix="%"))
            q4.metric("Quality Score",   _sfmt(row.get("Quality_Score"), ".0f", prefix="", suffix="/100"))
            _rrs_disp = (row.get("report_risk_score_real")
                         if pd.notna(row.get("report_risk_score_real", None))
                         else row.get("Report_Risk_Score"))
            q5.metric("Filing Risk",     _sfmt(_rrs_disp, ".0f", prefix="", suffix="/100"))
            if cm_df is not None:
                cm_match2 = cm_df[cm_df["ticker"].str.upper() == chosen.upper()]
                if len(cm_match2) > 0:
                    cm_r = cm_match2.iloc[0]
                    st.markdown('<div class="current-label">Current Market Snapshot</div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Price",  _sfmt(cm_r.get("current_price")))
                    _cmc = cm_r.get("current_market_cap")
                    m2.metric("Current Mkt Cap", _sfmt(float(_cmc)/1e9 if pd.notna(_cmc) and float(_cmc)>0 else None, ".1f", suffix="B"))
                    m3.metric("Daily Change",   _sfmt(cm_r.get("daily_change_pct"), "+.2f", prefix="", suffix="%"))
                    last_u = str(cm_r.get("last_updated", ""))[:10]
                    m4.metric("As Of", last_u or "N/A")
        else:
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("Current Price",     _sfmt(row.get("Current_Price")))
            q2.metric("Est. Fair Value",   _sfmt(row.get("Estimated_Fair_Value")))
            q3.metric("Valuation Gap",     _sfmt(row.get("Valuation_Gap_Pct"), "+.1f", prefix="", suffix="%"))
            q4.metric("Quality Score",     _sfmt(row.get("Quality_Score"), ".0f", prefix="", suffix="/100"))
            q5.metric("Report Risk Score", _sfmt(row.get("Report_Risk_Score"), ".0f", prefix="", suffix="/100"))

        st.markdown(signal_badge_html(_show_sig), unsafe_allow_html=True)
        if _risk_adj and _risk_adj not in ("", "nan") and _risk_adj != _sig_disp:
            st.caption(f"Risk-adjusted signal. Original calibrated: {_sig_disp}")

        # Colored context badges (gap, risk, trend, data quality)
        if use_xbrl:
            _bp = []
            _gv_raw = row.get("current_valuation_gap_pct")
            if not pd.notna(_gv_raw):
                _gv_raw = row.get("Valuation_Gap_Pct")
            if pd.notna(_gv_raw):
                _gv = float(_gv_raw)
                if _gv > 20:   _gbg, _gfg, _gdesc = "#dbeafe", "#1d4ed8", "High"
                elif _gv > 5:  _gbg, _gfg, _gdesc = "#eff6ff", "#2563eb", "Favorable"
                elif _gv >= -5: _gbg, _gfg, _gdesc = "#f1f5f9", "#475569", "Neutral"
                elif _gv >= -20: _gbg, _gfg, _gdesc = "#fff7ed", "#c2410c", "Stretched"
                else:           _gbg, _gfg, _gdesc = "#fee2e2", "#b91c1c", "Deep Stretched"
                _bp.append(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
                    f'font-weight:600;background:{_gbg};color:{_gfg};border:1px solid {_gfg};margin:2px 4px 2px 0">'
                    f'<span style="font-weight:400;opacity:0.8">Gap: </span>{_gv:+.1f}% {_gdesc}</span>'
                )
            _fr_raw = row.get("report_risk_score_real")
            if not pd.notna(_fr_raw):
                _fr_raw = row.get("Report_Risk_Score")
            if pd.notna(_fr_raw):
                _frv = float(_fr_raw)
                if _frv < 30:   _frbg, _frfg, _frlvl = "#dcfce7", "#15803d", "Low"
                elif _frv < 50: _frbg, _frfg, _frlvl = "#fef9c3", "#a16207", "Moderate"
                elif _frv < 70: _frbg, _frfg, _frlvl = "#ffedd5", "#c2410c", "Elevated"
                else:           _frbg, _frfg, _frlvl = "#fee2e2", "#b91c1c", "High"
                _bp.append(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
                    f'font-weight:600;background:{_frbg};color:{_frfg};border:1px solid {_frfg};margin:2px 4px 2px 0">'
                    f'<span style="font-weight:400;opacity:0.8">10-K Risk: </span>{_frv:.0f}/100 {_frlvl}</span>'
                )
            _qr_raw = row.get("latest_10q_risk_score")
            if pd.notna(_qr_raw):
                _qrv = float(_qr_raw)
                if _qrv < 30:   _qrbg, _qrfg, _qrlvl = "#dcfce7", "#15803d", "Low"
                elif _qrv < 50: _qrbg, _qrfg, _qrlvl = "#fef9c3", "#a16207", "Moderate"
                elif _qrv < 70: _qrbg, _qrfg, _qrlvl = "#ffedd5", "#c2410c", "Elevated"
                else:           _qrbg, _qrfg, _qrlvl = "#fee2e2", "#b91c1c", "High"
                _bp.append(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
                    f'font-weight:600;background:{_qrbg};color:{_qrfg};border:1px solid {_qrfg};margin:2px 4px 2px 0">'
                    f'<span style="font-weight:400;opacity:0.8">10-Q Risk: </span>{_qrv:.0f}/100 {_qrlvl}</span>'
                )
            _tr_raw = row.get("filing_risk_trend", "")
            if pd.notna(_tr_raw) and str(_tr_raw).strip():
                _ts = str(_tr_raw).strip().lower()
                if "decr" in _ts:   _tbg, _tfg, _tlbl = "#dcfce7", "#15803d", "Decreasing"
                elif "incr" in _ts: _tbg, _tfg, _tlbl = "#fee2e2", "#b91c1c", "Increasing"
                elif "stab" in _ts: _tbg, _tfg, _tlbl = "#f1f5f9", "#475569", "Stable"
                else:               _tbg, _tfg, _tlbl = "#f1f5f9", "#475569", str(_tr_raw).strip().title()
                _bp.append(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
                    f'font-weight:600;background:{_tbg};color:{_tfg};border:1px solid {_tfg};margin:2px 4px 2px 0">'
                    f'<span style="font-weight:400;opacity:0.8">Trend: </span>{_tlbl}</span>'
                )
            _dq_raw = row.get("output_quality_flag", "")
            if pd.notna(_dq_raw) and str(_dq_raw).strip():
                _dqs = str(_dq_raw).strip().lower()
                _dql_map = {"needs_review": "Needs Review", "neutral_report_risk": "Neutral Report Risk",
                            "clean": "Clean", "ok": "Clean"}
                _dql = _dql_map.get(_dqs, str(_dq_raw).strip().title())
                if _dql == "Clean":                 _dqbg, _dqfg = "#dcfce7", "#15803d"
                elif _dql == "Neutral Report Risk": _dqbg, _dqfg = "#fef9c3", "#a16207"
                else:                               _dqbg, _dqfg = "#fee2e2", "#b91c1c"
                _bp.append(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;'
                    f'font-weight:600;background:{_dqbg};color:{_dqfg};border:1px solid {_dqfg};margin:2px 4px 2px 0">'
                    f'<span style="font-weight:400;opacity:0.8">Data Quality: </span>{_dql}</span>'
                )
            if _bp:
                st.markdown('<div style="margin:8px 0 6px 0">' + "".join(_bp) + "</div>", unsafe_allow_html=True)

        _scr_wl_col, _ = st.columns([1, 3])
        with _scr_wl_col:
            _render_watchlist_button(chosen, row.to_dict(), "Screener")

        with st.expander("Why this company appears here"):
            _render_explanation(row.to_dict())

    st.divider()

    # Charts in tabs
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(
        ["Signal Distribution", "Valuation vs Quality", "Risk vs Quality"]
    )

    with chart_tab1:
        sig_counts  = active_df["Final_Signal"].value_counts().reindex(SIGNAL_ORDER, fill_value=0)
        bar_colors  = [SIGNAL_COLORS[s] for s in sig_counts.index]
        fig = go.Figure(go.Bar(
            x=sig_counts.index, y=sig_counts.values,
            marker_color=bar_colors,
            text=sig_counts.values, textposition="auto",
        ))
        fig.update_layout(
            xaxis_title="Signal", yaxis_title="Companies",
            height=240, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=80),
        )
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab2:
        _sc_df = fdf.dropna(subset=["Valuation_Gap_Pct", "Quality_Score"]).copy()
        if not _sc_df.empty:
            _sc_colors = [SIGNAL_COLORS.get(s, "#888") for s in _sc_df["Final_Signal"]]
            _hover_t = [
                f"{r['Ticker']} ({str(r.get('Company_Name',''))[:20]})<br>"
                f"Year: {r.get('year','')}<br>Val Gap: {r.get('Valuation_Gap_Pct',0):+.1f}%<br>"
                f"Quality: {r.get('Quality_Score',0):.0f}<br>Signal: {r.get('Final_Signal','')}"
                for _, r in _sc_df.iterrows()
            ]
            fig_sc = go.Figure(go.Scatter(
                x=_sc_df["Valuation_Gap_Pct"], y=_sc_df["Quality_Score"],
                mode="markers+text",
                marker=dict(color=_sc_colors, size=9, opacity=0.8, line=dict(color="white", width=1)),
                text=_sc_df["Ticker"], textposition="top center", textfont=dict(size=9),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=_hover_t,
            ))
            fig_sc.update_layout(
                xaxis_title="Valuation Gap %", yaxis_title="Quality Score",
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=40),
            )
            fig_sc.add_vline(x=0, line_dash="dash", line_color="#cbd5e1", line_width=1)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption("Each point = one company. Color = research signal.")
        else:
            st.info("Insufficient data for scatter chart.")

    with chart_tab3:
        _rsk_sc_col = ("report_risk_score_real"
                       if "report_risk_score_real" in fdf.columns and fdf["report_risk_score_real"].notna().any()
                       else "Report_Risk_Score")
        _sc_risk = fdf.dropna(subset=[_rsk_sc_col, "Quality_Score"]).copy() if _rsk_sc_col in fdf.columns else pd.DataFrame()
        if not _sc_risk.empty:
            _sc_colors2 = [SIGNAL_COLORS.get(s, "#888") for s in _sc_risk["Final_Signal"]]
            _hover2 = [
                f"{r['Ticker']}<br>Risk: {r.get(_rsk_sc_col,0):.0f}/100<br>"
                f"Quality: {r.get('Quality_Score',0):.0f}<br>Signal: {r.get('Final_Signal','')}"
                for _, r in _sc_risk.iterrows()
            ]
            fig_rsk = go.Figure(go.Scatter(
                x=_sc_risk[_rsk_sc_col], y=_sc_risk["Quality_Score"],
                mode="markers+text",
                marker=dict(color=_sc_colors2, size=9, opacity=0.8, line=dict(color="white", width=1)),
                text=_sc_risk["Ticker"], textposition="top center", textfont=dict(size=9),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=_hover2,
            ))
            fig_rsk.update_layout(
                xaxis_title="Filing Risk Score (0–100)", yaxis_title="Quality Score",
                height=320, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_rsk, use_container_width=True)
            st.caption("Higher filing risk = more risk language in 10-K. Not a performance predictor.")
        else:
            st.info("Filing risk scores not yet available. Run python3 filing_risk_utils.py.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 -- COMPANY DETAIL  (routes between sample and XBRL views)
# ════════════════════════════════════════════════════════════════════════════

def _clean_filing_text(text: str) -> str:
    """Fix concatenated filing text: insert spaces at word merge points, replace $ with USD."""
    import re
    if not text:
        return text
    text = re.sub(r'([a-z0-9B])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d+(?:\.\d+)?[BMK])(in|against|vs|of|for|to|with|and|or)', r'\1 \2', text)
    text = re.sub(r'\$(\d)', r'USD \1', text)
    return text


def _xbrl_company_detail(mo: pd.DataFrame) -> None:
    """Full company detail page for XBRL model output data."""
    all_tickers = sorted(mo["Ticker"].unique())

    # Universe-awareness: if the requested ticker is in the universe but has no model output, say so
    _req = str(st.session_state.get("selected_ticker", "")).upper()
    if _req and _req not in [t.upper() for t in all_tickers]:
        try:
            _uni, _ = load_ticker_universe()
            if not _uni.empty and _req in _uni["ticker"].str.upper().values:
                _u_row       = _uni[_uni["ticker"].str.upper() == _req].iloc[0]
                _u_grp       = str(_u_row.get("universe_group", ""))
                _u_stat      = str(_u_row.get("data_status", ""))
                _u_company   = str(_u_row.get("company_name", "")).strip()
                _u_failure   = str(_u_row.get("failure_reason", "")).strip()
                _u_last_run  = str(_u_row.get("last_pipeline_run", "")).strip()

                _ready = check_ticker_readiness(_req)
                _missing = []
                if not _ready["has_fundamentals"]:
                    _missing.append("SEC company facts (fundamentals)")
                if not _ready["has_market_cap"]:
                    _missing.append("Market cap data")
                if not _ready["has_model_output"]:
                    _missing.append("Model output")
                if not _ready["has_10k_risk"]:
                    _missing.append("10-K filing risk")

                _lines = [
                    f"**{_req}**{' — ' + _u_company if _u_company else ''} is in the ticker "
                    f"universe (group: **{_u_grp}**, status: **{_u_stat}**), but model outputs "
                    "are not yet available for this ticker.",
                ]
                if _missing:
                    _lines.append("**Missing pipeline steps:** " + ", ".join(_missing))
                if _u_failure:
                    _lines.append(f"**Last failure:** {_u_failure}")
                if _u_last_run:
                    _lines.append(f"Last pipeline run: {_u_last_run}")
                _lines.append(
                    "To generate data for this ticker, go to **About → Ticker Universe** "
                    "and run the pipeline steps."
                )
                st.info("  \n".join(_lines))

                if st.button("Go to Ticker Universe setup",
                             key="detail_goto_uni_btn"):
                    st.session_state.pending_nav_page = "About"
                    st.rerun()
        except Exception:
            pass

    default_idx = (
        all_tickers.index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in all_tickers else 0
    )
    chosen = st.selectbox(
        "Select company:",
        all_tickers,
        index=default_idx,
        format_func=lambda t: f"{t} — {mo[mo['Ticker']==t]['Company_Name'].values[0]}",
        key="xbrl_detail_select",
    )
    st.session_state.selected_ticker = chosen
    row = mo[mo["Ticker"] == chosen].iloc[0]

    # Current market row
    cm_row = None
    if cm_df is not None and len(cm_df) > 0:
        cm_match = cm_df[cm_df["ticker"].str.upper() == chosen.upper()]
        if len(cm_match) > 0:
            cm_row = cm_match.iloc[0]

    signal_display  = row.get("final_signal_display") or row.get("Final_Signal", "")
    is_needs_review = "needs_review" in str(row.get("output_quality_flag", ""))
    qflag  = str(row.get("output_quality_flag", "ok")).strip()
    wtext  = str(row.get("output_warning_text", "")).strip()
    yr     = row.get("year", "N/A")
    bm     = row.get("best_model_used", "Ridge Regression")
    calib_sig    = row.get("final_signal_calibrated")
    risk_adj_sig = row.get("final_signal_calibrated_risk_adjusted")
    _risk_source  = str(row.get("risk_score_source", "") or "")
    _risk_score_r = row.get("report_risk_score_real")
    _has_real_risk = _risk_source == "real_sec_filing" and pd.notna(_risk_score_r)
    _show_sig = (risk_adj_sig if (risk_adj_sig and pd.notna(risk_adj_sig) and str(risk_adj_sig) != "nan")
                 else (calib_sig if (calib_sig and pd.notna(calib_sig)) else signal_display))

    # Quality warning
    _QFLAG_LABELS = {
        "neutral_report_risk":   "Neutral Report Risk",
        "needs_review":          "Needs Review",
        "extreme_valuation_gap": "Extreme Valuation Gap",
        "high_model_error":      "High Model Error",
        "suspicious_market_cap": "Suspicious Market Cap",
    }
    def _hq(raw: str) -> str:
        return ", ".join(
            _QFLAG_LABELS.get(p.strip(), p.strip().replace("_", " ").title())
            for p in raw.split(",") if p.strip()
        )
    _STALE_FILING_MSG = "filing-text extraction has not yet been integrated"
    _is_only_nrr = {p.strip() for p in qflag.split(",")} == {"neutral_report_risk"}

    def _plain_english_warnings(raw_text: str) -> list[str]:
        """Convert raw pipeline warning strings into readable sentences."""
        import re
        out = []
        for part in raw_text.split("|"):
            part = part.strip()
            if not part or _STALE_FILING_MSG in part.lower():
                continue
            # Model error magnitude
            m = re.search(r"model error of ([+\-]?\d+\.?\d*)\s*\(log scale\)", part, re.I)
            if m:
                val = m.group(1)
                out.append(
                    f"The valuation model had a large estimation error ({val} on a log scale) "
                    f"for this data point, which reduces confidence in the model output."
                )
                continue
            # Extreme valuation gap
            if "extreme" in part.lower() and "valuation" in part.lower():
                out.append(
                    "The model-estimated fair value is very far from the current market price. "
                    "This may reflect unusual business conditions or a data quality issue — "
                    "treat the signal with extra caution."
                )
                continue
            # Suspicious market cap
            if "suspicious" in part.lower() and "market cap" in part.lower():
                out.append(
                    "The market capitalisation figure for this company looks unusual. "
                    "Verify the data independently before relying on this signal."
                )
                continue
            # Fallback: surface the raw text, cleaned up
            out.append(part.rstrip(".") + ".")
        return out

    if is_needs_review and wtext:
        _plain = _plain_english_warnings(wtext)
        _msg = "**Data Quality Review**\n\nOne or more checks flagged this company's data:"
        if _plain:
            _msg += "\n\n" + "\n".join(f"- {s}" for s in _plain)
        _msg += (
            "\n\nThe research signal shown here has been marked **Needs Review** "
            "and should not be used without independent verification."
        )
        st.warning(_msg)
    elif _is_only_nrr:
        if not _has_real_risk:
            st.info(
                "**Data Quality Note:** Filing risk data is not yet available for this "
                "company. The filing risk component uses a neutral placeholder (50/100)."
            )
        # _has_real_risk True → flag is a pre-integration artifact, silent
    elif qflag not in ("ok", "") and wtext:
        _plain = _plain_english_warnings(wtext)
        if _plain:
            st.warning(
                "**Data Quality Review**\n\n"
                + "\n".join(f"- {s}" for s in _plain)
            )

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"## {row.get('Company_Name', chosen)}")
        st.markdown(f"**{chosen}** &nbsp;·&nbsp; Technology &nbsp;·&nbsp; Year: {yr}" +
                    (f" &nbsp;·&nbsp; Model: {bm}" if bm else ""))
    with h2:
        sig_color = "#c0392b" if is_needs_review else SIGNAL_COLORS.get(_show_sig, "#555")
        sig_bg    = "#fdf3f2" if is_needs_review else SIGNAL_BG_COLORS.get(_show_sig, "#f0f0f0")
        st.markdown(
            f"""<div style="border:2px solid {sig_color}; border-radius:12px;
                     background:{sig_bg}; padding:16px; text-align:center; margin-top:12px">
              <div style="font-size:12px;color:#555;margin-bottom:4px">Research Signal</div>
              <div style="font-size:13px;font-weight:700;color:{sig_color}">{_show_sig}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Action buttons
    ab1, ab2, ab3, _ = st.columns([1, 1, 2, 1])
    with ab1:
        if st.button("Open in Simulator", key="xbrl_open_simulator"):
            st.session_state.paper_ticker_prefill = chosen
            st.session_state.paper_source_prefill = "xbrl"
            st.session_state.pending_nav_page = "Portfolio Simulator"
            st.rerun()
    with ab2:
        if st.button("Back to Screener", key="xbrl_back_screener"):
            st.session_state.pending_nav_page = "Screener"
            st.rerun()
    with ab3:
        _render_watchlist_button(chosen, row.to_dict(), "Company Detail")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_ov, tab_val, tab_risk, tab_profile, tab_notes = st.tabs(
        ["Overview", "Valuation", "Filing Risk", "Financial Profile", "Research Notes"]
    )

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_ov:
        st.markdown('<div class="data-label">Historical Model Output</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual Market Cap",   _sfmt(row.get("Market_Cap_B"), ".2f", suffix="B"))
        c2.metric("Est. Fair Value",     _sfmt(row.get("Estimated_Fair_Value"), ".2f", suffix="B"))
        c3.metric("Valuation Gap",       _sfmt(row.get("Valuation_Gap_Pct"), "+.1f", prefix="", suffix="%"),
                  help="(Est. Fair Value - Actual MC) / Actual MC × 100")
        c4.metric("Model Error (log)",   _sfmt(row.get("model_error"), "+.3f", prefix=""),
                  help="Actual log(MC) - Predicted log(MC).")

        c5, c6, c7 = st.columns(3)
        rsk_val = row.get("Report_Risk_Score")
        c5.metric("Quality Score",     _sfmt(row.get("Quality_Score"), ".1f", prefix="", suffix="/100"),
                  help="Percentile-ranked composite: ROE 30%, Op. Margin 25%, Rev. Growth 25%, Low Debt 20%")
        c6.metric("Report Risk Score", _sfmt(rsk_val, ".0f", prefix="", suffix="/100"))
        c7.metric("Historical Year",   str(yr))

        expl = row.get("explanation_text", "")
        if expl and str(expl).strip():
            st.divider()
            st.markdown('<div class="section-header">Plain-English Explanation</div>', unsafe_allow_html=True)
            st.info(str(expl))

        # Current market snapshot
        st.divider()
        st.markdown('<div class="current-label">Current Market Snapshot</div>', unsafe_allow_html=True)
        if cm_row is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price",   _sfmt(cm_row.get("current_price")))
            curr_mc_raw = cm_row.get("current_market_cap")
            try:
                curr_mc_b = float(curr_mc_raw) / 1e9 if pd.notna(curr_mc_raw) and float(curr_mc_raw) > 0 else None
            except Exception:
                curr_mc_b = None
            m2.metric("Current Mkt Cap", _sfmt(curr_mc_b, ".1f", suffix="B"))
            m3.metric("Daily Change",    _sfmt(cm_row.get("daily_change_pct"), "+.2f", prefix="", suffix="%"))
            last_u = str(cm_row.get("last_updated", ""))[:10]
            m4.metric("As Of", last_u or "N/A")
            st.caption("Current market data is for context only and does not validate the historical model analysis.")
            if curr_mc_b is not None and pd.notna(row.get("Market_Cap_B")):
                hist_mc = float(row["Market_Cap_B"])
                delta_pct = (curr_mc_b - hist_mc) / hist_mc * 100 if hist_mc > 0 else None
                if delta_pct is not None:
                    st.metric(f"Market Cap Change ({yr} → today)",
                              _sfmt(curr_mc_b, ".1f", suffix="B"),
                              delta=f"{delta_pct:+.1f}%")
        else:
            st.info("Current market data not available. Run `python3 market_cap_utils.py` to fetch.")

        st.divider()
        _render_explanation(row.to_dict())
        st.divider()
        st.warning(
            "**Limitations:** Historical data only (2010–2024). "
            "In-sample predictions for training-period companies. "
            "29 tech-heavy companies. Not investment advice."
        )

    # ── Valuation ─────────────────────────────────────────────────────────────
    with tab_val:
        st.markdown('<div class="section-header">Signal Analysis</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("Original Signal",   str(signal_display))
        v2.metric("Calibrated Signal", str(calib_sig) if calib_sig and pd.notna(calib_sig) else "N/A")
        v3.metric("Risk-Adj Signal",   str(risk_adj_sig) if risk_adj_sig and pd.notna(risk_adj_sig) else "N/A")

        if calib_sig and not pd.isna(calib_sig):
            st.divider()
            st.markdown('<div class="section-header">Calibrated Signal Analysis</div>', unsafe_allow_html=True)
            st.caption(
                "The calibrated signal compares this company against other companies "
                "from the same year and market era. This helps avoid labeling most modern "
                "companies as overvalued simply because valuation levels were higher after 2017."
            )
            pct_yr  = row.get("valuation_gap_percentile_year")
            pct_era = row.get("valuation_gap_percentile_era")
            bkt_yr  = row.get("valuation_bucket_year", "")
            bkt_era = row.get("valuation_bucket_era",  "")
            era     = row.get("era", "")

            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.metric("Original Signal",    str(signal_display))
            ca2.metric("Calibrated Signal",  str(calib_sig))
            ca3.metric("Percentile in Year",
                       f"{float(pct_yr):.0f}th" if pd.notna(pct_yr) else "N/A",
                       help="What % of companies in the same year had a lower valuation gap.")
            ca4.metric("Percentile in Era",
                       f"{float(pct_era):.0f}th" if pd.notna(pct_era) else "N/A",
                       help="What % of companies in the same era had a lower valuation gap.")

            if bkt_yr or bkt_era:
                cb1, cb2, cb3 = st.columns(3)
                cb1.metric("Value Bucket (Year)", str(bkt_yr) if bkt_yr else "N/A")
                cb2.metric("Value Bucket (Era)",  str(bkt_era) if bkt_era else "N/A")
                cb3.metric("Market Era",          str(era) if era else "N/A")

            calib_warn = row.get("calibration_warning_text", "")
            if calib_warn and str(calib_warn).strip():
                for w in str(calib_warn).split("|"):
                    if w.strip():
                        st.warning(w.strip())

            if str(calib_sig) != str(signal_display):
                st.info(
                    f"The original signal is **{signal_display}** (absolute thresholds). "
                    f"The calibrated signal is **{calib_sig}** (relative to {int(row.get('year', 0))} peers). "
                    "Both are research signals, not investment advice."
                )
        else:
            st.info("Calibrated signal not available for this company.")

    # ── Filing Risk ────────────────────────────────────────────────────────────
    with tab_risk:
        st.markdown('<div class="section-header">Filing Risk Analysis</div>', unsafe_allow_html=True)
        st.caption(
            "Higher filing-risk score means this company's filing contains more "
            "risk-related language relative to other filings in the dataset. "
            "It is not a prediction of stock performance. "
            "Source: SEC EDGAR 10-K filing text."
        )

        fr1, fr2, fr3, fr4 = st.columns(4)
        fr1.metric("Report Risk Score",
                   f"{float(_risk_score_r):.0f}/100" if _has_real_risk else "50/100 (neutral)",
                   help="0–100 relative filing risk. Higher = more risk language in the filing.")
        fr2.metric("Risk Source",
                   "Real SEC 10-K" if _has_real_risk else "Neutral placeholder")

        _form_type   = str(row.get("form_type",   "") or "") if _has_real_risk else ""
        _filing_date = str(row.get("filing_date", "") or "") if _has_real_risk else ""
        fr3.metric("Form Type",   _form_type   or "N/A")
        fr4.metric("Filing Date", _filing_date or "N/A")

        if _has_real_risk:
            _components = str(row.get("risk_score_components", "") or "")
            if _components:
                st.caption(f"Score components: {_components}")
            _adj_sig    = row.get("final_signal_calibrated_risk_adjusted")
            _adj_reason = str(row.get("risk_adjustment_reason", "") or "")
            if _adj_sig and pd.notna(_adj_sig) and _adj_reason and "no_real" not in _adj_reason:
                st.warning(f"**Risk-adjusted signal: {_adj_sig}**  |  {_adj_reason}")
            _fw = str(row.get("filing_warning_text", "") or "")
            if _fw:
                st.info(_fw)
        else:
            _fdq = str(row.get("filing_data_quality_flag", "") or "")
            _fdq_note = {
                "too_short":               "filing text was too short to score",
                "filing_missing_or_error": "filing was missing or could not be parsed",
            }.get(_fdq, "")
            if _fdq_note:
                st.info(f"**Filing risk note:** Data not available for this company ({_fdq_note}).")
            else:
                st.info("Filing risk data is not available for this company.")

        # ── Latest 10-Q risk update (supplemental) ───────────────────────────
        st.markdown('<div class="section-header" style="margin-top:18px">Latest 10-Q Risk Update</div>',
                    unsafe_allow_html=True)
        _cd_ticker = str(row.get("Ticker", "")).upper()
        _q10_df_cd, _upd_df_cd = _load_10q_data()
        _q10_row = None
        if _upd_df_cd is not None and len(_upd_df_cd) > 0:
            _q_match = _upd_df_cd[_upd_df_cd["ticker"].str.upper() == _cd_ticker]
            if len(_q_match) > 0:
                _q10_row = _q_match.iloc[0]
        if _q10_row is not None:
            _q_score   = _q10_row.get("latest_10q_risk_score")
            _q_trend   = str(_q10_row.get("filing_risk_trend") or "Unavailable")
            _q_delta   = _q10_row.get("filing_risk_delta")
            _q_date    = str(_q10_row.get("latest_10q_filing_date") or "N/A")
            _ann_score = _q10_row.get("annual_10k_risk_score")
            _ann_date  = str(_q10_row.get("annual_10k_filing_date") or "N/A")
            _ann_yr    = _q10_row.get("latest_model_year")
            _q_warn    = str(_q10_row.get("current_risk_warning_text") or "")

            st.markdown("**Annual 10-K Baseline** (used in model signal)")
            ak1, ak2, ak3, _ = st.columns(4)
            ak1.metric("Annual 10-K Risk",   f"{float(_ann_score):.0f}/100" if pd.notna(_ann_score) else "—",
                       help="Annual 10-K filing risk score used in the model-backed signal.")
            ak2.metric("10-K Filed",         _ann_date[:10] if len(_ann_date) > 4 else "N/A")
            ak3.metric("Model Year",         str(int(_ann_yr)) if pd.notna(_ann_yr) else "N/A")

            st.markdown("**Latest 10-Q Update** (supplemental)")
            _trend_colors = {"Increasing": "🔴", "Decreasing": "🟢", "Stable": "🟡", "Unavailable": "⚪"}
            qq1, qq2, qq3, qq4 = st.columns(4)
            qq1.metric("10-Q Risk Score",  f"{float(_q_score):.0f}/100" if pd.notna(_q_score) else "—",
                       help="Latest quarterly 10-Q filing risk score.")
            qq2.metric("vs Annual 10-K",   f"{float(_q_delta):+.0f}" if pd.notna(_q_delta) else "—",
                       help="Difference between 10-Q risk and annual 10-K risk (positive = risk increased).")
            qq3.metric("Risk Trend",       f"{_trend_colors.get(_q_trend,'⚪')} {_q_trend}")
            qq4.metric("10-Q Filed",       _q_date[:10] if len(_q_date) > 4 else "N/A")

            if _q_warn:
                st.warning(_q_warn)
            st.caption(
                "10-K risk is the **annual baseline** used in the model-backed signal. "
                "10-Q risk is a **recent quarterly update** indicating whether risk language "
                "has increased or decreased since the annual filing. "
                "10-Q risk does not alter the model signal — it is supplemental context only."
            )
        else:
            st.info(
                "10-Q risk data not available for this company. "
                "Run `python3 filing_risk_utils.py --10q` to generate latest quarterly risk scores."
            )

    # ── Financial Profile ─────────────────────────────────────────────────────
    with tab_profile:
        st.markdown('<div class="section-header">Profile Radar</div>', unsafe_allow_html=True)
        _qs   = float(row.get("Quality_Score") or 50)
        _gap  = float(row.get("Valuation_Gap_Pct") or 0)
        _merr = float(row.get("model_error") or 0)
        _rrs2 = float(_risk_score_r) if _has_real_risk else 50.0

        gap_norm     = min(max((_gap + 50) / 100 * 100, 0), 100)
        low_risk_n   = max(100 - _rrs2, 0)
        model_conf   = max(100 - abs(_merr) * 50, 0)
        _sig_score   = {
            "High-priority research candidate": 100, "Research candidate": 75,
            "Fairly valued / neutral": 50, "Possible value trap": 25,
            "Potentially overvalued": 25,
        }.get(_show_sig, 50)

        categories = ["Quality Score", "Valuation Upside", "Low Filing Risk", "Model Confidence", "Research Signal"]
        values     = [_qs, gap_norm, low_risk_n, model_conf, _sig_score]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=SIGNAL_BG_COLORS.get(_show_sig, "#f0f0f0"),
            line=dict(color=SIGNAL_COLORS.get(_show_sig, "#555"), width=2),
            name=chosen,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=360, showlegend=False,
            margin=dict(l=50, r=50, t=30, b=30),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "Quality Score = composite fundamental quality (0–100). "
            "Valuation Upside = normalized valuation gap. "
            "Low Filing Risk = inverse of filing risk score. "
            "Model Confidence = inversely related to model prediction error. "
            "All axes normalized 0–100 for visualization only."
        )

    # ── Research Notes ────────────────────────────────────────────────────────
    with tab_notes:
        _render_research_notes_tab(chosen)


def _sample_company_detail(df: pd.DataFrame) -> None:
    """Full company detail page for sample data."""
    all_tickers = df["Ticker"].tolist()
    default_idx = (
        all_tickers.index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in all_tickers else 0
    )
    chosen = st.selectbox(
        "Select company:",
        all_tickers,
        index=default_idx,
        format_func=lambda t: f"{t} — {df[df['Ticker']==t]['Company_Name'].values[0]}",
        key="detail_select",
    )
    st.session_state.selected_ticker = chosen
    row = df[df["Ticker"] == chosen].iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"## {row.Company_Name}")
        st.markdown(f"**{chosen}** · {row.Sector}")
        st.markdown(COMPANY_DESCRIPTIONS.get(chosen, "No description available."))
    with h2:
        sig_color = SIGNAL_COLORS.get(row.Final_Signal, "#555")
        sig_bg    = SIGNAL_BG_COLORS.get(row.Final_Signal, "#f0f0f0")
        st.markdown(
            f"""<div style="border:2px solid {sig_color}; border-radius:12px;
                     background:{sig_bg}; padding:16px; text-align:center; margin-top:12px">
              <div style="font-size:13px;color:#555;margin-bottom:4px">Research Signal</div>
              <div style="font-size:15px;font-weight:700;color:{sig_color}">{row.Final_Signal}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    _btn1, _btn2 = st.columns([1, 3])
    with _btn1:
        if st.button("Open in Portfolio Simulator", key="sample_open_simulator"):
            st.session_state.paper_ticker_prefill = chosen
            st.session_state.paper_source_prefill = "sample"
            st.session_state.pending_nav_page = "Portfolio Simulator"
            st.rerun()
    with _btn2:
        _render_watchlist_button(chosen, row.to_dict(), "Company Detail (sample)")

    st.divider()

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Market Price",  _sfmt(row.get("Current_Price")))
    c2.metric("Model Est. Fair Value", _sfmt(row.get("Estimated_Fair_Value")))
    c3.metric("Valuation Gap",         _sfmt(row.get("Valuation_Gap_Pct"), "+.1f", prefix="", suffix="%"),
              help="(Est. Fair Value - Market Price) / Market Price x 100")
    c4.metric("Market Cap",            _sfmt(row.get("Market_Cap_B"), ".0f", suffix="B"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Quality Score",     _sfmt(row.get("Quality_Score"), ".0f", prefix="", suffix="/100"),
              help="Higher = stronger fundamentals")
    c6.metric("Report Risk Score", _sfmt(row.get("Report_Risk_Score"), ".0f", prefix="", suffix="/100"),
              help="Higher = more risk language in filings")
    c7.metric("P/E Ratio",         _sfmt(row.get("PE_Ratio"), ".1f", prefix="", suffix="x"))
    c8.metric("EV / EBITDA",       _sfmt(row.get("EV_EBITDA"), ".1f", prefix="", suffix="x"))

    st.divider()

    # ── Three-column detail ───────────────────────────────────────────────────
    left, mid, right = st.columns(3, gap="medium")

    def _kv(label: str, value: str) -> None:
        col_l, col_r = st.columns([2, 1])
        col_l.markdown(f"<span style='font-size:13px;color:#64748b'>{label}</span>",
                       unsafe_allow_html=True)
        col_r.markdown(f"<span style='font-size:13px;font-weight:600'>{value}</span>",
                       unsafe_allow_html=True)

    with left:
        st.markdown('<div class="section-header">Valuation Metrics</div>', unsafe_allow_html=True)
        _kv("Current Price",   _sfmt(row.get("Current_Price")))
        _kv("Est. Fair Value", _sfmt(row.get("Estimated_Fair_Value")))
        _kv("Valuation Gap",   _sfmt(row.get("Valuation_Gap_Pct"), "+.1f", prefix="", suffix="%"))
        _kv("P/E Ratio",       _sfmt(row.get("PE_Ratio"), ".1f", prefix="", suffix="x"))
        pb = row.get("PB_Ratio", np.nan)
        pb_val = (f"{pb:.2f}x" if (pd.notna(pb) and pb > 0) else
                  ("Neg. (buybacks)" if pd.notna(pb) else "N/A"))
        _kv("P/B Ratio",       pb_val)
        _kv("EV / EBITDA",     _sfmt(row.get("EV_EBITDA"), ".1f", prefix="", suffix="x"))

    with mid:
        st.markdown('<div class="section-header">Quality Indicators</div>', unsafe_allow_html=True)
        _kv("Quality Score",  _sfmt(row.get("Quality_Score"), ".0f", prefix="", suffix="/100"))
        _kv("ROE",            _sfmt(row.get("ROE_Pct"), ".1f", prefix="", suffix="%"))
        _kv("Revenue (TTM)",  _sfmt(row.get("Revenue_B"), ".1f", suffix="B"))
        _kv("Revenue Growth", _sfmt(row.get("Revenue_Growth_Pct"), "+.1f", prefix="", suffix="%"))
        _kv("EPS (TTM)",      _sfmt(row.get("EPS_TTM")))
        _kv("Free Cash Flow", _sfmt(row.get("FCF_B"), ".1f", suffix="B"))

    with right:
        st.markdown('<div class="section-header">Risk Indicators</div>', unsafe_allow_html=True)
        _kv("Report Risk Score",   _sfmt(row.get("Report_Risk_Score"), ".0f", prefix="", suffix="/100"))
        de = row.get("Debt_Equity", np.nan)
        de_val = (f"{de:.2f}x" if (pd.notna(de) and de >= 0) else
                  ("Neg. equity" if pd.notna(de) else "N/A"))
        _kv("Debt / Equity",       de_val)
        _kv("Risk Word Count",     _sfmt(row.get("Risk_Word_Count"), ".0f", prefix="", na="N/A"))
        _kv("Debt Mentions",       _sfmt(row.get("Debt_Mentions"), ".0f", prefix="", na="N/A"))
        _kv("Legal Mentions",      _sfmt(row.get("Legal_Mentions"), ".0f", prefix="", na="N/A"))
        _kv("Cost Pressure Flags", _sfmt(row.get("Cost_Pressure_Mentions"), ".0f", prefix="", na="N/A"))

    st.divider()

    # ── Signal explanation ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Why This Signal?</div>', unsafe_allow_html=True)
    st.info(get_signal_explanation(row))

    st.divider()

    # ── Report text signals ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">10-K / 10-Q Filing Signal Summary</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Derived from NLP scanning of SEC EDGAR filings. "
        "Frequency of risk-related language is one input to the Report Risk Score."
    )
    sigs = REPORT_SIGNALS.get(chosen, {})
    if sigs:
        t1, t2 = st.columns(2)
        with t1:
            with st.expander("Key Risk Keywords Detected", expanded=True):
                for kw in sigs.get("risk_keywords", []):
                    st.markdown(f"- `{kw}`")
            with st.expander("Legal / Litigation Mentions"):
                st.write(_clean_filing_text(sigs.get("legal_mentions", "None identified.")))
        with t2:
            with st.expander("Debt and Liquidity Language", expanded=True):
                st.write(_clean_filing_text(sigs.get("debt_liquidity", "None identified.")))
            with st.expander("Cost Pressure Language"):
                st.write(_clean_filing_text(sigs.get("cost_pressures", "None identified.")))

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Profile Radar</div>', unsafe_allow_html=True)

    rev_growth_norm = min(max((row.get("Revenue_Growth_Pct", 0) + 5) / 25 * 100, 0), 100)
    roe_norm        = min(max(row.get("ROE_Pct", 0) / 40 * 100, 0), 100)
    mc_b            = row.get("Market_Cap_B", 0.001) or 0.001
    fcf_yield_norm  = min(max(row.get("FCF_B", 0) / mc_b * 1000, 0), 100)
    de              = row.get("Debt_Equity", 0) or 0
    low_debt_norm   = max(100 - de * 15, 0)
    low_risk_norm   = 100 - (row.get("Report_Risk_Score", 50) or 50)
    gap_norm        = min(max((row.get("Valuation_Gap_Pct", 0) or 0) + 10, 0) / 40 * 100, 100)

    categories = ["Revenue Growth", "ROE", "FCF Yield", "Low Debt", "Low Filing Risk", "Valuation Upside"]
    values     = [rev_growth_norm, roe_norm, fcf_yield_norm, low_debt_norm, low_risk_norm, gap_norm]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=SIGNAL_BG_COLORS.get(row.Final_Signal, "#f0f0f0"),
        line=dict(color=SIGNAL_COLORS.get(row.Final_Signal, "#555"), width=2),
        name=chosen,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=350, showlegend=False,
        margin=dict(l=50, r=50, t=30, b=30),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Each axis is normalized 0-100 relative to model scale.")

    st.divider()
    _render_explanation(row.to_dict())

    st.divider()
    with st.expander("Research Notes"):
        _render_research_notes_tab(chosen)


def page_company_detail(df: pd.DataFrame) -> None:
    if st.session_state.get("_scroll_to_top", False):
        st.session_state["_scroll_to_top"] = False
        import streamlit.components.v1 as _stv1
        _stv1.html(
            """<script>
            (function(){
                function doScroll(){
                    try{
                        ['[data-testid="stAppViewContainer"]',
                         '[data-testid="stMain"]',
                         'section.main', '.main',
                         'body', 'html'
                        ].forEach(function(s){
                            var e=window.parent.document.querySelector(s);
                            if(e){e.scrollTop=0;e.scrollLeft=0;}
                        });
                        window.parent.scrollTo(0,0);
                    }catch(ex){}
                }
                doScroll();
                [50,150,350,700].forEach(function(t){setTimeout(doScroll,t);});
            })();
            </script>""",
            height=1,
        )
    st.title("Company Detail")
    if mo_df is not None:
        _xbrl_company_detail(mo_df)
    else:
        st.caption("Showing sample data company profile.")
        _sample_company_detail(df)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 -- METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════

def page_methodology() -> None:
    st.title("Methodology")
    st.caption("How the screener collects data, scores companies, and generates research signals.")

    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
    st.markdown("""
The screener follows a four-stage pipeline designed to surface companies worth deeper
research without replacing that research:

| Stage | What happens |
|---|---|
| **1. Data Collection** | Structured financial metrics from XBRL filings (2010–2016) and live prices via yfinance |
| **2. Text Signal Extraction** | 10-K / 10-Q filings scanned for risk-related language (sample data only) |
| **3. Scoring** | Three model scores: Fair Value (Ridge regression), Quality, Report Risk |
| **4. Signal Assignment** | Scores combined into a ranked research signal |
    """)

    st.divider()

    st.markdown('<div class="section-header">Data Sources</div>', unsafe_allow_html=True)
    st.markdown("""
**Underdawg Model Layer (29 companies, 2010–2024):**
- Legacy (2010–2016): XBRL financial statements from SEC EDGAR
- Modern (2017–2024): SEC EDGAR Company Facts API (annual GAAP fundamentals)
- Market cap estimated as: historical year-end price (split-adjusted) × shares outstanding
- Model target: log(market_cap)
- Features: 16 derived financial ratios and growth metrics

**Sample Data Layer (10 companies):**
- Pre-processed dataset with full valuation, quality, and risk scores
- Includes per-share metrics (P/E, P/B, EV/EBITDA, EPS) and filing risk scores
- Intended for full feature demonstration of all screener pages
    """)

    st.markdown('<div class="section-header">Fair Value Model</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        st.markdown("##### Underdawg Valuation Model (2010–2024)")
        st.markdown("""
A Ridge Regression model trained on 361 company-year observations (29 companies,
2010–2024, legacy XBRL + modern SEC). The model predicts **log(market_cap)** from 16
financial features.

Legacy-era best model: **Ridge Regression**
(RMSE = 0.549, R² = 0.80 on 28-observation legacy holdout)

**Key drivers (Random Forest importances):**
1. log(Revenue) — 49.5%
2. Gross Margin — 15.0%
3. Equity — 6.9%
4. Gross Profit — 6.3%

Estimated fair value = exp(predicted log(MC)), shown in billions.
        """)
        st.code(
            "Valuation Gap % =\n"
            "  (Est. Fair Value - Actual Market Cap)\n"
            "  / Actual Market Cap x 100",
            language="text",
        )

    with sc2:
        st.markdown("##### Quality Score (0-100)")
        st.markdown("""
Computed from XBRL features using percentile ranking within the training dataset.

| Component | Weight | Metric |
|---|---|---|
| Return on Equity | 30% | net_income / equity |
| Operating Margin | 25% | operating_income / revenue (FCF proxy) |
| Revenue Growth | 25% | YoY % change in revenue |
| Low Debt-to-Assets | 20% | Lower debt = higher score |

Missing values receive a neutral 50th-percentile score for that component.

**Note:** Report Risk Score is set to 50 (neutral) for XBRL model outputs —
SEC filing text has not been integrated into the XBRL dataset.
        """)

    st.divider()

    st.markdown('<div class="section-header">Signal Assignment Rules</div>', unsafe_allow_html=True)
    st.markdown("Signals are evaluated top-to-bottom — first match wins:")

    sig_rows = [
        ("Possible value trap",              "Gap > 15% AND Quality < 45",                "Apparent upside but low fundamentals quality may negate it."),
        ("High-priority research candidate", "Gap > 20% AND Quality >= 65 AND Risk < 40", "Clear potential upside, strong fundamentals, clean filing language."),
        ("Research candidate",               "Gap > 10% AND Quality >= 50",               "Moderate upside with acceptable quality — worth investigating."),
        ("Fairly valued / neutral",          "-10% <= Gap <= +10%",                        "Model sees no clear margin of safety or significant overvaluation."),
        ("Potentially overvalued",           "Gap < -10%",                                 "Market price appears to exceed model-estimated fair value by > 10%."),
    ]
    for sig, rule, reason in sig_rows:
        c1, c2, c3 = st.columns([2, 2, 3])
        sig_color = SIGNAL_COLORS.get(sig, "#333")
        c1.markdown(f'<span style="color:{sig_color};font-weight:600">{sig}</span>',
                    unsafe_allow_html=True)
        c2.markdown(f"`{rule}`")
        c3.markdown(f"*{reason}*")
        st.divider()

    st.markdown('<div class="section-header">Limitations and Disclaimers</div>', unsafe_allow_html=True)
    st.warning("""
**This model has important limitations:**

- **Small training set:** 137 company-year rows, 27 tech-heavy companies, 2010–2016 only
- **No filing text for XBRL companies:** Report Risk Score is neutral (50) for all XBRL outputs
- **In-sample predictions:** model_outputs.csv uses the model trained on 80% of data applied to all rows
- **Survivorship bias:** Only companies with both XBRL data and yfinance prices are included
- **Not generalized:** Model trained on tech sector — may not apply to other sectors
- **Historical only:** No out-of-sample validation beyond 2016
- **Not investment advice:** Signals indicate where to focus research time, not whether to buy or sell
    """)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5 -- BENCHMARK COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def page_benchmark_comparison(df: pd.DataFrame) -> None:
    st.title("Benchmark Comparison")
    st.caption(
        "Compares a screener-selected portfolio against alternative benchmarks. "
        "This page uses simulated demonstration data unless otherwise noted."
    )

    st.info(
        "Demonstration Data Notice: The portfolio returns below are generated using "
        "simulated data (geometric Brownian motion) to illustrate what a benchmark "
        "comparison module would look like in a production system. They are NOT historical "
        "back-test results and should NOT be interpreted as performance claims. "
        "Benchmark comparison is context, not proof of future performance."
    )

    # ── Portfolio composition ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Screener Portfolio Composition</div>', unsafe_allow_html=True)
    st.markdown(
        "The screener portfolio consists of the top-ranked companies by signal priority. "
        "In this demo, the portfolio includes companies rated as High-priority or Research candidate:"
    )

    top = df[df["Final_Signal"].isin([
        "High-priority research candidate", "Research candidate"
    ])].copy()

    port_src = {
        "Ticker":            "Ticker",
        "Company_Name":      "Company",
        "Final_Signal":      "Signal",
        "Valuation_Gap_Pct": "Valuation Gap (%)",
        "Quality_Score":     "Quality Score",
    }
    available = {k: v for k, v in port_src.items() if k in top.columns}
    port_disp = top[list(available.keys())].copy().rename(columns=available)
    if "Valuation Gap (%)" in port_disp.columns:
        port_disp["Valuation Gap (%)"] = port_disp["Valuation Gap (%)"].map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )

    def _sig_style(val):
        return f"color:{SIGNAL_COLORS.get(val,'#000')}; font-weight:600"

    try:
        styler = port_disp.style
        styler = _safe_map(styler, _sig_style, "Signal")
        st.dataframe(styler, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(port_disp, use_container_width=True, hide_index=True)

    st.divider()

    # ── Simulated growth chart ────────────────────────────────────────────────
    bm_df = generate_benchmark_data(seed=42)

    plot_series = {
        "Screener":         "#27ae60",
        "S&P 500":          "#2563eb",
        "Random Portfolio": "#f39c12",
        "Risk-Free (4.5%)": "#94a3b8",
    }
    plot_dashes = {
        "Screener":         "solid",
        "S&P 500":          "solid",
        "Random Portfolio": "dash",
        "Risk-Free (4.5%)": "dot",
    }

    fig = go.Figure()
    for series_name, color in plot_series.items():
        if series_name not in bm_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=bm_df["Date"], y=bm_df[series_name],
            name=series_name, mode="lines",
            line=dict(color=color, width=2, dash=plot_dashes[series_name]),
            hovertemplate=(
                f"<b>{series_name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Value: $%{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="SIMULATED Portfolio Growth — $10,000 Invested (1 Year)",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All values are SIMULATED using geometric Brownian motion with assumed parameters.")

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Simulated Performance Summary</div>', unsafe_allow_html=True)

    metrics_rows = []
    for series_name in plot_series:
        if series_name not in bm_df.columns:
            continue
        m = portfolio_metrics(bm_df[series_name].values)
        m["Portfolio"] = series_name
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Portfolio")

    def _highlight_col(s):
        styles = []
        numeric_vals = pd.to_numeric(s, errors="coerce")
        for val in numeric_vals:
            if pd.isna(val):
                styles.append("")
                continue
            if s.name in ("Annual Return (%)", "Sharpe Ratio", "Final Value ($)"):
                color = "#d5f5e3" if val == numeric_vals.max() else ""
            elif s.name == "Volatility (%)":
                color = "#d5f5e3" if val == numeric_vals.min() else ""
            elif s.name == "Max Drawdown (%)":
                color = "#d5f5e3" if val == numeric_vals.max() else ""
            else:
                color = ""
            styles.append(f"background-color: {color}" if color else "")
        return styles

    fmt_dict = {
        "Annual Return (%)": "{:.1f}%",
        "Volatility (%)":    "{:.1f}%",
        "Sharpe Ratio":      "{:.2f}",
        "Max Drawdown (%)":  "{:.1f}%",
        "Final Value ($)":   "${:,.0f}",
    }

    try:
        styler = metrics_df.style
        styler = _safe_apply(styler, _highlight_col)
        numeric_cols = metrics_df.apply(lambda c: pd.to_numeric(c, errors="coerce").notna().all())
        numeric_fmt  = {k: v for k, v in fmt_dict.items()
                        if k in metrics_df.columns and numeric_cols.get(k, False)}
        styler = _safe_format(styler, numeric_fmt)
        st.dataframe(styler, use_container_width=True)
    except Exception:
        st.dataframe(metrics_df, use_container_width=True)

    st.caption(
        "Green cells highlight the best value per metric. "
        "Sharpe Ratio = (Return - Risk-Free Rate) / Volatility. "
        "All figures annualized from SIMULATED daily returns."
    )

    st.divider()

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">How to Interpret This</div>', unsafe_allow_html=True)
    st.markdown("""
| Comparison | What it shows |
|---|---|
| **Screener vs. Random Portfolio** | Whether quality and risk filters add value over random selection |
| **Screener vs. S&P 500** | Whether focusing on a small screened set competes with broad diversification |
| **Screener vs. Risk-Free Rate** | The minimum bar — any equity strategy should exceed this over the long run |

**In a production system**, this module would use:
- Real historical price data via yfinance or a financial data provider
- Walk-forward back-testing to avoid look-ahead bias
- Full transaction cost and rebalancing modeling
- Statistical significance testing (is any outperformance due to skill or luck?)
    """)

    st.divider()
    st.markdown("""
    <div class="disclaimer-box">
    <strong>Benchmark Disclaimer</strong><br>
    All portfolio return figures on this page are <strong>simulated demonstration data</strong>
    generated using a random-walk model with assumed parameters. They do not represent actual
    historical performance. The screener does not claim to outperform the S&P 500 or any
    professional investment manager. Past simulated performance is not indicative of future results.
    Benchmark comparison is context, not proof of future performance.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 6 -- MODEL DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_diagnostics_result() -> tuple:
    try:
        from real_model_utils import run_diagnostics
        return run_diagnostics()
    except Exception as exc:
        return {"status": "import_error", "error": str(exc)}, None


def page_model_diagnostics() -> None:
    st.title("Model Diagnostics")
    st.caption(
        "Technical view of the real XBRL financial dataset and the trained "
        "market-cap model. Intended for internal review and academic evaluation."
    )

    with st.spinner("Loading dataset and model — runs once per session, then cached..."):
        data_diag, model_results = _cached_diagnostics_result()

    # ── Dataset status check ──────────────────────────────────────────────────
    status = data_diag.get("status", "unknown")

    if status == "files_not_found":
        st.error(
            data_diag.get("error", "Dataset files not found.") + "\n\n"
            "Place **companies.csv** and **indicators_by_company.csv** inside the "
            "`XBRL/` folder next to `app.py`, then refresh the page."
        )
        st.info("All other pages continue to work with the built-in sample dataset.")
        return

    if status in ("error", "import_error"):
        st.error(f"Diagnostics unavailable: {data_diag.get('error', 'unknown error')}")
        return

    # ── Active model badge ────────────────────────────────────────────────────
    model_type = (model_results or {}).get("model_type", "proxy")
    best_model = (model_results or {}).get("best_model", "")

    if model_type == "true_market_cap":
        st.success(
            f"**True Market Cap Model Active** — Target: log(Market Cap)  |  "
            f"Best model: {best_model}  |  "
            f"Training: {data_diag.get('market_cap_training_rows', '?')} rows, "
            f"{data_diag.get('market_cap_companies', '?')} companies"
        )
    else:
        st.warning(
            "**Proxy Model Active** — Target: log(Total Assets)  |  "
            "Run `python3 market_cap_utils.py` to activate the true market-cap model."
        )
        te = data_diag.get("true_model_error")
        if te:
            st.caption(f"True model error: {te}")

    st.divider()

    # ── Data coverage ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Data Coverage</div>', unsafe_allow_html=True)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Companies in Registry",     f"{data_diag.get('companies_total', 0):,}")
    r1c2.metric("Companies with XBRL Data",  f"{data_diag.get('companies_with_data', 0):,}")
    r1c3.metric("Company-Year Observations", f"{data_diag.get('company_year_obs', 0):,}")
    r1c4.metric("Years Covered",             ", ".join(str(y) for y in data_diag.get("years_covered", [])))

    # Market cap data layer
    tm_rows = data_diag.get("ticker_mapping_rows", 0)
    mc_ok   = data_diag.get("market_cap_ok_rows", 0)
    mc_cos  = data_diag.get("market_cap_companies", 0)
    mc_tr   = data_diag.get("market_cap_training_rows", 0)

    if tm_rows or mc_ok:
        st.markdown('<div class="section-header">Market Cap Data Layer</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Ticker Mappings",          tm_rows)
        r2c2.metric("Usable MC Rows",           mc_ok,
                    help="Company-year rows where price × shares data is available")
        r2c3.metric("Companies with MC",        mc_cos)
        r2c4.metric("Training Rows",            mc_tr,
                    help="Merged XBRL features + market cap usable for training")

    # model_outputs.csv and current market data
    from real_model_utils import (
        MODEL_OUTPUTS_PATH, MODEL_OUTPUTS_REVIEWED_PATH,
        MODEL_METRICS_PATH, FEATURE_IMP_PATH,
        MODEL_OUTPUT_QUALITY_SUMMARY,
    )
    from data_utils import CURRENT_MARKET_CSV, MODEL_OUTPUTS_REVIEWED_CSV, MODEL_OUTPUT_QUALITY_SUMMARY_CSV
    mo_rows = 0
    if os.path.exists(MODEL_OUTPUTS_PATH):
        try:
            mo_rows = len(pd.read_csv(MODEL_OUTPUTS_PATH))
        except Exception:
            pass
    cm_rows = 0
    if os.path.exists(CURRENT_MARKET_CSV):
        try:
            cm_df_local = pd.read_csv(CURRENT_MARKET_CSV)
            cm_rows = (cm_df_local.get("data_quality_flag", pd.Series()) == "ok").sum()
        except Exception:
            pass

    st.markdown('<div class="section-header">Generated Files</div>', unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("model_outputs.csv rows",      mo_rows)
    g2.metric("Current Market Tickers (ok)", cm_rows)
    g3.metric("model_metrics.csv",           "exists" if os.path.exists(MODEL_METRICS_PATH) else "missing")
    g4.metric("feature_importance.csv",      "exists" if os.path.exists(FEATURE_IMP_PATH)   else "missing")

    # ── Current Data Coverage ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Current Data Coverage</div>', unsafe_allow_html=True)
    st.caption("Live market snapshot and 10-Q filing risk coverage as of the last fetch.")

    _cd_cm_df, _cd_cm_msg = load_current_market_data()
    _cd_10q_df, _cd_upd_df = _load_10q_data()

    if _cd_cm_df is not None and len(_cd_cm_df) > 0:
        _cd_last = _cd_cm_df["last_updated"].max() if "last_updated" in _cd_cm_df.columns else None
        try:
            _cd_age_h = (pd.Timestamp.now() - pd.to_datetime(_cd_last)).total_seconds() / 3600 if _cd_last else None
        except Exception:
            _cd_age_h = None

        def _cd_freshness(ts):
            try:
                age_h = (pd.Timestamp.now() - pd.to_datetime(ts)).total_seconds() / 3600
                return "fresh" if age_h <= 5 else "stale"
            except Exception:
                return "unknown"

        _cd_cm_df["_freshness"] = _cd_cm_df["last_updated"].apply(
            lambda x: _cd_freshness(x) if pd.notna(x) else "missing"
        )
        _cd_n_fresh   = int((_cd_cm_df["_freshness"] == "fresh").sum())
        _cd_n_stale   = int((_cd_cm_df["_freshness"] == "stale").sum())
        _cd_n_missing = int((_cd_cm_df["_freshness"] == "missing").sum())
        _cd_n_ok      = int(((_cd_cm_df.get("data_quality_flag", pd.Series())) == "ok").sum())

        _cd_age_str = f"{_cd_age_h:.1f}h ago" if _cd_age_h is not None else "unknown"
        cd1, cd2, cd3, cd4 = st.columns(4)
        cd1.metric("Market Tickers (ok)",   _cd_n_ok,      help="Tickers with valid price data.")
        cd2.metric("Fresh (≤5h)",           _cd_n_fresh,   help="Tickers fetched within the last 5 hours.")
        cd3.metric("Stale (>5h)",           _cd_n_stale,   help="Tickers with data older than 5 hours.")
        cd4.metric("Last fetch",            str(_cd_last)[:16] if _cd_last else "—",
                   delta=_cd_age_str if _cd_age_h is not None else None,
                   delta_color="inverse")
    else:
        st.info(f"No current market data. {_cd_cm_msg}  Run `python3 market_cap_utils.py --current` to fetch.")

    if _cd_upd_df is not None and len(_cd_upd_df) > 0:
        _cd_trend_cts = _cd_upd_df["filing_risk_trend"].value_counts().to_dict() if "filing_risk_trend" in _cd_upd_df.columns else {}
        _cd_10q_n     = int(_cd_upd_df["latest_10q_risk_score"].notna().sum()) if "latest_10q_risk_score" in _cd_upd_df.columns else 0
        cd5, cd6, cd7, cd8 = st.columns(4)
        cd5.metric("10-Q Tickers",       len(_cd_upd_df))
        cd6.metric("10-Q Scores (ok)",   _cd_10q_n)
        cd7.metric("Increasing risk",    _cd_trend_cts.get("Increasing", 0))
        cd8.metric("Decreasing risk",    _cd_trend_cts.get("Decreasing", 0))
        st.caption(
            f"Filing risk trend: Stable={_cd_trend_cts.get('Stable',0)}, "
            f"Increasing={_cd_trend_cts.get('Increasing',0)}, "
            f"Decreasing={_cd_trend_cts.get('Decreasing',0)}, "
            f"Unavailable={_cd_trend_cts.get('Unavailable',0)}."
        )
    else:
        st.info("No 10-Q risk data. Run `python3 filing_risk_utils.py --10q` to generate quarterly risk scores.")

    st.divider()

    # ── Model Output Quality section ──────────────────────────────────────────
    reviewed_path = MODEL_OUTPUTS_REVIEWED_PATH
    quality_summary_path = MODEL_OUTPUT_QUALITY_SUMMARY

    if os.path.exists(reviewed_path):
        st.markdown('<div class="section-header">Model Output Quality</div>', unsafe_allow_html=True)
        try:
            rev_df = pd.read_csv(reviewed_path)
            total_rows   = len(rev_df)
            ok_rows      = (rev_df["output_quality_flag"] == "ok").sum()
            review_rows  = rev_df["output_quality_flag"].str.contains("needs_review", na=False).sum()

            qq1, qq2, qq3, qq4 = st.columns(4)
            qq1.metric("Total Output Rows", total_rows)
            qq2.metric("Rows: ok",          ok_rows)
            qq3.metric("Rows: needs_review", review_rows)
            qq4.metric("Flag types",        rev_df["output_quality_flag"].str.split(",").explode().str.strip().nunique())

            # Flag count breakdown
            if os.path.exists(quality_summary_path):
                qsum = pd.read_csv(quality_summary_path)
                st.markdown("**Flag breakdown:**")
                try:
                    st.dataframe(
                        qsum.style.format({"percentage": "{:.1f}%"}),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(qsum, use_container_width=True, hide_index=True)

            # Top 10 largest valuation gaps
            st.markdown("**Top 10 largest absolute valuation gaps:**")
            top_gap = (
                rev_df[["ticker", "year", "actual_market_cap", "valuation_gap_pct",
                         "model_error", "output_quality_flag"]]
                .assign(abs_gap=rev_df["valuation_gap_pct"].abs())
                .sort_values("abs_gap", ascending=False)
                .head(10)
                .drop(columns="abs_gap")
            )
            top_gap["actual_market_cap"] = top_gap["actual_market_cap"].map(
                lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A"
            )
            top_gap["valuation_gap_pct"] = top_gap["valuation_gap_pct"].map(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
            )
            top_gap["model_error"] = top_gap["model_error"].map(
                lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A"
            )
            st.dataframe(top_gap.rename(columns={
                "ticker":             "Ticker",
                "year":               "Year",
                "actual_market_cap":  "Actual MC",
                "valuation_gap_pct":  "Val. Gap %",
                "model_error":        "Model Error",
                "output_quality_flag":"Quality Flag",
            }), use_container_width=True, hide_index=True)

            # Top 10 largest absolute model errors
            st.markdown("**Top 10 largest absolute model errors:**")
            top_err = (
                rev_df[["ticker", "year", "actual_market_cap", "model_error",
                         "valuation_gap_pct", "output_quality_flag"]]
                .assign(abs_err=rev_df["model_error"].abs())
                .sort_values("abs_err", ascending=False)
                .head(10)
                .drop(columns="abs_err")
            )
            top_err["actual_market_cap"] = top_err["actual_market_cap"].map(
                lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A"
            )
            top_err["model_error"] = top_err["model_error"].map(
                lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A"
            )
            top_err["valuation_gap_pct"] = top_err["valuation_gap_pct"].map(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
            )
            st.dataframe(top_err.rename(columns={
                "ticker":             "Ticker",
                "year":               "Year",
                "actual_market_cap":  "Actual MC",
                "model_error":        "Model Error",
                "valuation_gap_pct":  "Val. Gap %",
                "output_quality_flag":"Quality Flag",
            }), use_container_width=True, hide_index=True)

        except Exception as exc:
            st.warning(f"Could not load quality review data: {exc}")

    st.divider()

    # ── Indicator coverage ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">XBRL Indicator Coverage</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Found in dataset**")
        for ind in data_diag.get("indicators_found", []):
            st.markdown(f"- {ind}")
    with i2:
        st.markdown("**Not present in dataset**")
        for ind in data_diag.get("indicators_missing", []):
            if ind in ("EntityPublicFloat", "MarketCapitalization", "MarketCap"):
                st.markdown(f"- {ind} *(market cap — not in XBRL filings)*")
            else:
                st.markdown(f"- {ind}")

    st.divider()

    if model_results is None:
        st.info("Model results not available.")
        return

    # ── Model validation ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Validation</div>', unsafe_allow_html=True)

    m_status = model_results.get("status")
    if m_status == "sklearn_not_installed":
        st.error(model_results.get("error"))
        return
    if m_status == "insufficient_data":
        st.warning(
            f"Only {model_results.get('n_obs', 0)} usable observations — "
            "not enough to train. Check XBRL and market cap data."
        )
        return
    if m_status != "success":
        st.error(f"Model training failed: {model_results.get('error', 'unknown')}")
        return

    n_train = model_results.get("n_train", 0)
    n_test  = model_results.get("n_test", 0)
    st.caption(
        f"80/20 train-test split — {n_train} training observations, "
        f"{n_test} held-out test observations."
    )

    # Load from CSV if available (shows pre-computed metrics)
    metrics_source = "live"
    if os.path.exists(MODEL_METRICS_PATH):
        try:
            metrics_df_csv = pd.read_csv(MODEL_METRICS_PATH)
            metrics_source = "csv"
        except Exception:
            metrics_df_csv = None
    else:
        metrics_df_csv = None

    # Build metrics table from live results
    models = model_results.get("models", {})
    metrics_rows = []
    for model_name, m in models.items():
        is_best = (model_name == best_model)
        metrics_rows.append({
            "Model":     f"{model_name} {'[BEST]' if is_best else ''}",
            "MAE":       m.get("mae"),
            "RMSE":      m.get("rmse"),
            "R-squared": m.get("r2"),
        })
    for base_name, bm in model_results.get("baseline", {}).items():
        metrics_rows.append({
            "Model":     base_name,
            "MAE":       bm.get("mae"),
            "RMSE":      bm.get("rmse"),
            "R-squared": bm.get("r2"),
        })

    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows).set_index("Model")
        try:
            st.dataframe(
                mdf.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R-squared": "{:.4f}"}),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(mdf, use_container_width=True)

        if model_type == "true_market_cap":
            st.caption(
                "MAE and RMSE are in log-scale units (natural log of USD). "
                "An MAE of ~0.45 corresponds to roughly 45-57% error in dollar market cap. "
                "The baseline row predicts the training-set mean for all test observations — "
                "both models substantially beat this naive baseline."
            )
        else:
            st.caption(
                "Metrics apply to the proxy target log(Assets), not market value. "
                "High R² reflects the balance-sheet identity (assets ≈ liabilities + equity)."
            )

    st.divider()

    # ── Feature importances ───────────────────────────────────────────────────
    # Try to load from CSV first, fall back to live model_results
    imp_data = None
    if os.path.exists(FEATURE_IMP_PATH):
        try:
            imp_csv = pd.read_csv(FEATURE_IMP_PATH)
            imp_data = list(zip(imp_csv["feature"], imp_csv["importance"]))
        except Exception:
            pass
    if imp_data is None:
        rf_model = models.get("Random Forest", {})
        imp_data = rf_model.get("importances", [])

    if imp_data:
        st.markdown('<div class="section-header">Feature Importances (Random Forest)</div>',
                    unsafe_allow_html=True)
        imp_df = pd.DataFrame(imp_data, columns=["Feature", "Importance"])
        imp_df["Importance %"] = (imp_df["Importance"] * 100).round(2)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance %"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#2563eb",
            text=imp_df["Importance %"].map("{:.1f}%".format),
            textposition="outside",
        ))
        fig_imp.update_layout(
            xaxis_title="Importance (%)",
            yaxis=dict(autorange="reversed"),
            height=max(250, len(imp_data) * 30),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=20, r=60, t=20, b=30),
        )
        fig_imp.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("log_revenue dominates because revenue is the strongest predictor of total company size.")

    st.divider()

    # ── Validation Comparison ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Validation Comparison</div>', unsafe_allow_html=True)
    st.caption("Legacy validation (2010–2016 XBRL dataset only)")

    from real_model_utils import MODEL_VALIDATION_COMPARISON_PATH
    if os.path.exists(MODEL_VALIDATION_COMPARISON_PATH):
        try:
            val_df = pd.read_csv(MODEL_VALIDATION_COMPARISON_PATH)

            st.info(
                "**How to read this table:**  "
                "Row-based validation can overstate performance because different years from the "
                "same company may appear in both training and testing. "
                "Company-level validation is stricter because it tests on companies the model "
                "has never seen. "
                "Time-based validation is stricter because it tests whether relationships learned "
                "from earlier years transfer to later years. "
                "Company-level and time-based R² are more trustworthy estimates of real-world "
                "generalization."
            )

            # Display columns
            disp_cols = ["validation_type", "model_name", "train_rows", "test_rows",
                         "train_companies", "test_companies", "train_years", "test_years",
                         "mae", "rmse", "r2", "is_best_for_split"]
            disp_cols = [c for c in disp_cols if c in val_df.columns]
            disp_val  = val_df[disp_cols].copy()

            # Format numeric columns
            for col in ["mae", "rmse"]:
                if col in disp_val.columns:
                    disp_val[col] = disp_val[col].map(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                    )
            if "r2" in disp_val.columns:
                disp_val["r2"] = disp_val["r2"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )

            def _style_val_type(val):
                colors = {
                    "row_based":      "#e8f4fd",
                    "company_level":  "#e9fef0",
                    "time_based":     "#fff8e1",
                }
                return f"background-color: {colors.get(val, '')}"

            try:
                styler = disp_val.style
                styler = _safe_map(styler, _style_val_type, "validation_type")
                st.dataframe(styler, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(disp_val, use_container_width=True, hide_index=True)

            st.caption(
                "Blue = row-based | Green = company-level | Yellow = time-based. "
                "MAE and RMSE are in log-scale units. "
                "SKIPPED rows indicate the split had too few observations."
            )

            # Warning if row-based R² is much higher than stricter splits
            try:
                def _best_r2(vtype):
                    sub = val_df[
                        (val_df["validation_type"] == vtype) &
                        (~val_df["model_name"].str.contains("baseline|SKIPPED", case=False, na=False))
                    ]
                    vals = pd.to_numeric(sub["r2"], errors="coerce").dropna()
                    return float(vals.max()) if len(vals) > 0 else None

                rb_r2  = _best_r2("row_based")
                cl_r2  = _best_r2("company_level")
                tb_r2  = _best_r2("time_based")

                drop_threshold = 0.15   # flag if R² drops more than 0.15
                flagged = []
                if rb_r2 is not None and cl_r2 is not None and (rb_r2 - cl_r2) > drop_threshold:
                    flagged.append(f"company-level R² ({cl_r2:.2f}) is {rb_r2 - cl_r2:.2f} lower than row-based ({rb_r2:.2f})")
                if rb_r2 is not None and tb_r2 is not None and (rb_r2 - tb_r2) > drop_threshold:
                    flagged.append(f"time-based R² ({tb_r2:.2f}) is {rb_r2 - tb_r2:.2f} lower than row-based ({rb_r2:.2f})")

                if flagged:
                    st.warning(
                        "**Performance drops under stricter validation:**\n\n"
                        + "\n".join(f"- {f}" for f in flagged)
                        + "\n\nThis suggests the current mapped dataset is too small or the model "
                        "is learning company-specific patterns. Expanding ticker_mapping.csv and "
                        "adding more companies is the next priority to improve generalization."
                    )
                else:
                    if rb_r2 is not None and cl_r2 is not None and tb_r2 is not None:
                        st.success(
                            f"Performance is consistent across validation strategies "
                            f"(row-based R²={rb_r2:.2f}, company-level R²={cl_r2:.2f}, "
                            f"time-based R²={tb_r2:.2f}). "
                            "No large generalization gap detected."
                        )
            except Exception:
                pass

        except Exception as exc:
            st.warning(f"Could not load validation comparison: {exc}")
    else:
        st.info(
            "model_validation_comparison.csv not found. "
            "Run `python3 market_cap_utils.py` to generate validation comparison data."
        )

    st.divider()

    # ── Features used ─────────────────────────────────────────────────────────
    with st.expander("Features included in model"):
        for f in model_results.get("features_used", []):
            st.markdown(f"- `{f}`")

    # ── Limitations ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Limitations</div>', unsafe_allow_html=True)
    if model_type == "true_market_cap":
        st.markdown("""
| Limitation | Detail |
|---|---|
| **Training set** | 361 company-year rows, 29 companies, 2010–2024. Legacy era validated; cross-era generalization is weak (see Combined Model Validation below). |
| **Tech-sector concentration** | Training universe is predominantly technology companies. May not generalize to other sectors. |
| **In-sample predictions** | model_outputs.csv uses the 80%-trained model applied to all 137 rows. 80% carry in-sample bias. |
| **Row-based train/test split** | The current 80/20 split is row-based, so the same company can appear in both train and test sets across different years. This makes test metrics optimistic. A company-level or time-based split (e.g., train on 2010–2014, test on 2015–2016) should be used next to get unbiased estimates. |
| **Market cap approximation** | Historical MC = split-adjusted price × XBRL shares. Not the same as official float-adjusted market cap. |
| **Report risk placeholder** | Report Risk Score = 50 for all XBRL companies. Signal is driven by valuation gap and quality only. |
| **Suspicious data rows** | ADSK 2011 and other rows with extreme gaps are flagged as 'needs_review' due to likely XBRL shares units errors. These rows should not be used as research candidates. |
| **No post-2016 validation** | Model cannot predict current valuations. Current market data is for context only. |
| **Not investment advice** | The valuation gap is a research signal, not proof of undervaluation. |
        """)
    else:
        st.markdown("""
| Limitation | Detail |
|---|---|
| **Proxy target** | log(Total Assets) measures company size, not market value. High R² reflects the balance-sheet identity. |
| **No market cap** | EntityPublicFloat, MarketCap, MarketCapitalization are absent from all XBRL filers in this dataset. |
| **Time period** | Data covers 2010–2016. No post-2016 validation possible. |
| **Not investment advice** | No output from this model constitutes a buy, sell, or hold recommendation. |
        """)

    # ── Combined Model Validation ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Combined Model Validation (2010–2024)")
    st.caption(
        "This validation tests whether the model generalizes across the modern SEC fundamentals period, "
        "not only the original 2010–2016 XBRL dataset. "
        "Four splits are evaluated: random row split, company-level split, legacy→modern time split, "
        "and pre-2021→2021–2024 time split."
    )
    _COMBINED_VAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "combined_model_validation_comparison.csv")
    if not os.path.exists(_COMBINED_VAL_PATH):
        st.info(
            "combined_model_validation_comparison.csv not found. "
            "Run: `python3 validate_combined_model.py` to generate combined validation."
        )
    else:
        try:
            cval_df = pd.read_csv(_COMBINED_VAL_PATH)

            disp_cols = ["validation_type", "model_name", "train_rows", "test_rows",
                         "train_years", "test_years", "mae", "rmse", "r2", "is_best_for_split"]
            disp_cols = [c for c in disp_cols if c in cval_df.columns]
            cval_disp = cval_df[disp_cols].copy()
            for col in ["mae", "rmse"]:
                if col in cval_disp.columns:
                    cval_disp[col] = cval_disp[col].map(
                        lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                    )
            if "r2" in cval_disp.columns:
                cval_disp["r2"] = cval_disp["r2"].map(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                )

            def _style_cval_type(val):
                colors = {
                    "row_based_combined":              "#dbeafe",
                    "company_level_combined":          "#e9fef0",
                    "time_based_legacy_to_modern":     "#fff8e1",
                    "time_based_pre_2021_to_2021_2024": "#fce7f3",
                }
                return f"background-color: {colors.get(val, '')}"

            try:
                styler = cval_disp.style
                styler = _safe_map(styler, _style_cval_type, "validation_type")
                st.dataframe(styler, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(cval_disp, use_container_width=True, hide_index=True)

            st.caption(
                "Blue = row-based | Green = company-level | Yellow = legacy→modern | Pink = pre-2021→2021-2024. "
                "MAE and RMSE are in log-scale units."
            )

            # Warnings for weak cross-era generalization
            def _cval_best_r2(vtype):
                sub = cval_df[
                    (cval_df["validation_type"] == vtype) &
                    (~cval_df["model_name"].str.contains("baseline|SKIPPED", case=False, na=False))
                ]
                vals = pd.to_numeric(sub["r2"], errors="coerce").dropna()
                return float(vals.max()) if len(vals) > 0 else None

            ltm_r2 = _cval_best_r2("time_based_legacy_to_modern")
            p21_r2 = _cval_best_r2("time_based_pre_2021_to_2021_2024")
            cl_r2  = _cval_best_r2("company_level_combined")
            rb_r2  = _cval_best_r2("row_based_combined")

            warnings_out = []
            if ltm_r2 is not None and ltm_r2 < 0.3:
                warnings_out.append(
                    f"**Legacy→modern R² is weak ({ltm_r2:.2f})** — the model trained on 2010–2016 "
                    "does not generalize well to 2017–2024 market conditions. "
                    "Calibrated signals are particularly important for modern years."
                )
            if p21_r2 is not None and p21_r2 < 0.3:
                warnings_out.append(
                    f"**Pre-2021→2021–2024 R² is weak ({p21_r2:.2f})** — post-pandemic valuations "
                    "diverge structurally from earlier periods. Use calibrated signals for 2021–2024."
                )
            if rb_r2 is not None and cl_r2 is not None and (rb_r2 - cl_r2) > 0.2:
                warnings_out.append(
                    f"**Company-level R² ({cl_r2:.2f}) is substantially below row-based ({rb_r2:.2f})** — "
                    "model may be learning company-specific patterns. "
                    "Expanding the ticker universe would improve generalization."
                )
            for w in warnings_out:
                st.warning(w)
            if not warnings_out and rb_r2 is not None:
                st.success(
                    f"Row-based R²={rb_r2:.2f}. "
                    "Check time-based splits for cross-era generalization."
                )
        except Exception as _exc:
            st.warning(f"Could not load combined validation: {_exc}")

    # ── Signal Calibration Diagnostics ───────────────────────────────────────
    st.divider()
    st.markdown("### Signal Calibration Diagnostics")
    st.caption(
        "Calibrated signals compare each company's valuation gap within its own year "
        "and market era, rather than using fixed absolute thresholds. "
        "This reduces the systematic bias that labels most modern companies as "
        "'Potentially overvalued' because post-2017 valuations were structurally higher "
        "than the 2010–2016 training period."
    )

    _SIG_DIST_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_distribution_by_year.csv")
    _ERA_DIST_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valuation_gap_distribution_by_era.csv")
    _CALIB_DIAG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_calibration_diagnostics.csv")

    _calib_files_exist = all(os.path.exists(p) for p in [_SIG_DIST_PATH, _ERA_DIST_PATH, _CALIB_DIAG_PATH])

    if not _calib_files_exist:
        st.info(
            "Calibration files not yet generated. "
            "Run: `python3 calibrate_signals.py` to produce calibration diagnostics."
        )
    else:
        try:
            _sig_dist  = pd.read_csv(_SIG_DIST_PATH)
            _era_dist  = pd.read_csv(_ERA_DIST_PATH)
            _cal_diag  = pd.read_csv(_CALIB_DIAG_PATH)

            # Concentration warnings
            _conc_orig = _sig_dist[_sig_dist["orig_signal_concentrated"] == "yes"]["year"].tolist()
            _conc_cal  = _sig_dist[_sig_dist["cal_signal_concentrated"]  == "yes"]["year"].tolist()
            if _conc_orig:
                st.warning(
                    f"**Original signal concentration (>60% one signal):** years {_conc_orig}. "
                    "These years were dominated by 'Potentially overvalued' before calibration."
                )
            if _conc_cal:
                st.warning(
                    f"**Calibrated signal still concentrated (>60%):** years {_conc_cal}. "
                    "Likely small samples or extreme gap distributions in those years."
                )
            else:
                st.success(
                    "Calibrated signals: no year dominated by a single signal (>60%). "
                    "Distribution is more balanced across all years."
                )

            tab_sig, tab_era, tab_diag = st.tabs(
                ["Signal Distribution by Year", "Valuation Gap by Era", "Calibration Diagnostics"]
            )

            with tab_sig:
                st.caption(
                    "orig_ columns = original absolute-threshold signals. "
                    "cal_ columns = calibrated year-percentile signals."
                )
                _disp_sig = _sig_dist[[
                    "year", "era", "total_rows", "needs_review_rows",
                    "avg_valuation_gap_pct", "median_valuation_gap_pct",
                    "orig_pct_potentially_overvalued", "orig_pct_research_candidate",
                    "orig_pct_fairly_valued_neutral",
                    "cal_pct_potentially_overvalued", "cal_pct_research_candidate",
                    "cal_pct_fairly_valued_neutral",
                    "orig_signal_concentrated", "cal_signal_concentrated",
                ]].copy()
                _disp_sig.columns = [
                    "Year", "Era", "Rows", "Needs Review",
                    "Avg Val Gap %", "Med Val Gap %",
                    "Orig: Overvalued %", "Orig: Research %", "Orig: Fairly Valued %",
                    "Cal: Overvalued %",  "Cal: Research %",  "Cal: Fairly Valued %",
                    "Orig Concentrated", "Cal Concentrated",
                ]
                st.dataframe(_disp_sig, use_container_width=True, hide_index=True)

            with tab_era:
                st.caption(
                    "Valuation gap statistics by market era. "
                    "The legacy era (2010–2016) was the model's training period; "
                    "modern eras have structurally higher market valuations."
                )
                _disp_era = _era_dist[[
                    "era", "rows",
                    "mean_valuation_gap_pct", "median_valuation_gap_pct", "std_valuation_gap_pct",
                    "p10_valuation_gap_pct", "p25_valuation_gap_pct",
                    "p75_valuation_gap_pct", "p90_valuation_gap_pct",
                    "orig_pct_potentially_overvalued", "cal_pct_potentially_overvalued",
                    "orig_pct_research_candidate",     "cal_pct_research_candidate",
                ]].copy()
                _disp_era.columns = [
                    "Era", "Rows",
                    "Mean Gap %", "Median Gap %", "Std Gap %",
                    "P10 Gap %", "P25 Gap %", "P75 Gap %", "P90 Gap %",
                    "Orig: Overvalued %", "Cal: Overvalued %",
                    "Orig: Research %",   "Cal: Research %",
                ]
                st.dataframe(_disp_era, use_container_width=True, hide_index=True)

            with tab_diag:
                st.caption(
                    "Calibration diagnostics by model source and era. "
                    "dominant_orig vs dominant_cal shows how the leading signal changed."
                )
                _disp_cal = _cal_diag[[
                    "model_output_source", "era", "rows",
                    "mean_valuation_gap_pct", "median_valuation_gap_pct",
                    "mean_abs_model_error", "rmse_model_error",
                    "dominant_orig_signal", "dominant_orig_signal_pct",
                    "dominant_cal_signal",  "dominant_cal_signal_pct",
                    "orig_concentration_warning", "cal_concentration_warning",
                ]].copy()
                st.dataframe(_disp_cal, use_container_width=True, hide_index=True)

        except Exception as _exc:
            st.warning(f"Could not load calibration diagnostics: {_exc}")

    # ── Filing Risk Coverage ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Filing Risk Coverage")
    st.caption(
        "Status of SEC 10-K filing text extraction. "
        "Run `python3 filing_risk_utils.py` to fetch filings and compute real risk scores."
    )
    try:
        from filing_risk_utils import get_filing_risk_diagnostics
        _frd = get_filing_risk_diagnostics()

        if not _frd["lookup_exists"]:
            st.info(
                "filing_lookup_results.csv not found. "
                "Run: `python3 filing_risk_utils.py` to start the filing risk pipeline."
            )
        else:
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Tickers × Years",    _frd["lookup_rows"])
            fc2.metric("10-Ks Found",         _frd["filings_found"])
            fc3.metric("Filings Parsed",      _frd["filings_parsed"])
            fc4.metric("Real Risk Scores",    _frd["rows_with_real_score"])

            if _frd["rows_with_real_score"] > 0:
                st.success(
                    f"{_frd['rows_with_real_score']} rows have real SEC 10-K risk scores. "
                    f"{_frd['rows_neutral']} rows still use the neutral placeholder (50)."
                )
            else:
                st.warning(
                    "No real risk scores yet. Run `python3 filing_risk_utils.py` to extract them."
                )

            if _frd["top_risk_rows"]:
                st.markdown("**Top 10 highest filing risk scores:**")
                top_df = pd.DataFrame(_frd["top_risk_rows"])
                if "report_risk_score_real" in top_df.columns:
                    top_df["report_risk_score_real"] = top_df["report_risk_score_real"].map(
                        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                    )
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            if _frd["common_failures"]:
                with st.expander("Common filing lookup failures"):
                    for reason, count in _frd["common_failures"].items():
                        st.markdown(f"- {reason}: **{count}** rows")

    except ImportError:
        st.info("filing_risk_utils.py not found or not yet run.")
    except Exception as _exc:
        st.warning(f"Could not load filing risk diagnostics: {_exc}")

    # ── Modern Fundamentals Coverage ──────────────────────────────────────────
    st.divider()
    st.markdown("### Modern Fundamentals Coverage")
    st.caption(
        "Status of the SEC EDGAR data pipeline that expands model coverage "
        "from 2010–2016 to 2017–present."
    )

    try:
        from modern_fundamentals_utils import get_modern_pipeline_diagnostics
        md = get_modern_pipeline_diagnostics()

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Ticker→CIK mappings",
                      str(md["ticker_cik_count"]) if md["ticker_cik_mapping_exists"] else "—")
        col_m2.metric("SEC cache files",  str(md["sec_cache_files"]))
        col_m3.metric("Combined output rows",
                      str(md["combined_outputs_rows"]) if md["combined_outputs_exists"] else "—")

        status_rows = [
            ("ticker_cik_mapping.csv",                   md["ticker_cik_mapping_exists"],
             f"{md['ticker_cik_count']} entries" if md["ticker_cik_mapping_exists"] else "not found"),
            ("sec_company_facts_cache/ (JSON files)",    md["sec_cache_files"] > 0,
             f"{md['sec_cache_files']} files"),
            ("modern_fundamentals.csv",                  md["modern_fundamentals_exists"],
             f"{md['modern_fundamentals_rows']} rows, years {md['modern_fundamentals_years']}"
             if md["modern_fundamentals_exists"] else "not found"),
            ("modern_valuation_training_dataset.csv",    md["modern_training_exists"],
             f"{md['modern_training_rows']} rows" if md["modern_training_exists"] else "not found"),
            ("modern_model_outputs.csv",                 md["modern_outputs_exists"],
             f"{md['modern_outputs_rows']} rows, years {md['modern_outputs_years']}"
             if md["modern_outputs_exists"] else "not found"),
            ("model_outputs_combined.csv",               md["combined_outputs_exists"],
             f"{md['combined_outputs_rows']} rows, years {md['combined_outputs_years']}"
             if md["combined_outputs_exists"] else "not found"),
        ]

        status_df = pd.DataFrame(status_rows, columns=["File", "Exists", "Detail"])
        status_df["Status"] = status_df["Exists"].map(
            {True: "OK", False: "Missing"}
        )
        st.dataframe(
            status_df[["File", "Status", "Detail"]],
            use_container_width=True, hide_index=True,
        )

        if md["missing_data_summary"]:
            st.warning(
                "**Pipeline not yet run.** Missing items:\n"
                + "\n".join(f"- {m}" for m in md["missing_data_summary"])
                + "\n\nRun: `python3 modern_fundamentals_utils.py`"
            )
        else:
            st.success(
                "Modern fundamentals pipeline is complete. "
                f"Historical Replay now supports years {md['combined_outputs_years']}."
            )

    except ImportError:
        st.info(
            "modern_fundamentals_utils.py not found. "
            "Modern fundamentals pipeline has not been run yet."
        )
    except Exception as exc:
        st.warning(f"Could not load modern diagnostics: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE — PAPER PORTFOLIO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

def _pp_watchlist_tab() -> None:
    """Render the Watchlist tab inside Portfolio Simulator."""
    st.markdown("### Research Watchlist")
    st.caption(
        "Track companies under research review. Add companies from Company Detail or the Screener. "
        "Saved locally to watchlist.csv."
    )
    wl = _load_watchlist()
    if wl.empty:
        st.info("Your watchlist is empty. Add companies from the Screener or Company Detail.")
        return

    # ── Compact summary table ─────────────────────────────────────────────────
    st.markdown(f"**{len(wl)} companies tracked**")

    _disp = wl[["ticker", "company_name", "research_status", "signal", "added_at"]].copy()
    _disp["added_at"] = _disp["added_at"].astype(str).str[:10]
    _disp.columns = ["Ticker", "Company", "Status", "Signal", "Added"]
    st.dataframe(_disp, use_container_width=True, hide_index=True)

    # ── Company selector → detail panel ──────────────────────────────────────
    ticker_labels = [
        f"{str(r['ticker'])}  —  {str(r['company_name'] or r['ticker'])}"
        for _, r in wl.iterrows()
    ]
    sel_label = st.selectbox(
        "Inspect company",
        ticker_labels,
        key="wl_inspect_sel",
    )
    sel_idx   = ticker_labels.index(sel_label)
    wl_row    = wl.iloc[sel_idx]
    t         = str(wl_row.get("ticker", ""))
    co        = str(wl_row.get("company_name", "") or t)
    status    = str(wl_row.get("research_status", "New"))
    signal    = str(wl_row.get("signal", "—") or "—")
    vgap      = wl_row.get("current_valuation_gap_pct", "")
    qs        = wl_row.get("quality_score", "")
    risk      = wl_row.get("filing_risk_score", "")
    q10       = wl_row.get("latest_10q_risk_score", "")
    trend     = str(wl_row.get("filing_risk_trend", "—") or "—")
    added     = str(wl_row.get("added_at", ""))[:10]
    note      = str(wl_row.get("user_note", "") or "")

    st.markdown(f"#### {t} — {co}")
    st.caption(f"Added {added} · Research status: **{status}**")

    wm1, wm2, wm3, wm4, wm5, wm6 = st.columns(6)
    wm1.metric("Signal",      signal)
    wm2.metric("Val. Gap",    f"{float(vgap):+.1f}%" if vgap else "—")
    wm3.metric("Quality",     f"{float(qs):.0f}/100" if qs else "—")
    wm4.metric("Filing Risk", f"{float(risk):.0f}/100" if risk else "—")
    wm5.metric("10-Q Risk",   f"{float(q10):.0f}/100" if q10 else "—")
    wm6.metric("10-Q Trend",  trend)

    new_status = st.selectbox(
        "Update research status", _WL_STATUSES,
        index=_WL_STATUSES.index(status) if status in _WL_STATUSES else 0,
        key=f"wl_status_{t}",
    )
    new_note = st.text_input("Research note", value=note, key=f"wl_note_{t}",
                             placeholder="Brief research note…")

    wa1, wa2, wa3 = st.columns(3)
    with wa1:
        if st.button("Save", key=f"wl_save_{t}", type="primary"):
            _wl_update(t, status=new_status, note=new_note)
            st.success(f"{t} updated.")
            st.rerun()
    with wa2:
        if st.button("View Company Detail", key=f"wl_detail_{t}"):
            st.session_state.selected_ticker  = t
            st.session_state.detail_source    = "xbrl"
            st.session_state.pending_nav_page = "Company Detail"
            st.rerun()
    with wa3:
        if st.button("Remove from Watchlist", key=f"wl_remove_{t}"):
            _wl_remove(t)
            st.warning(f"{t} removed from watchlist.")
            st.rerun()

    st.divider()
    csv_bytes = wl.to_csv(index=False).encode("utf-8")
    st.download_button("Download Watchlist CSV", data=csv_bytes,
                       file_name="watchlist.csv", mime="text/csv", key="wl_dl_btn")


def page_portfolio_simulator(df: pd.DataFrame) -> None:
    """Portfolio Simulator — two-tab page: Live Paper Portfolio + Historical Replay."""
    st.title("Portfolio Simulator")

    st.markdown("""
<div class="disclaimer-box">
<strong>Simulated only — for research and learning.</strong><br>
No real trades are placed. No real money is involved. No brokerage connection.
Prices may be delayed, missing, or estimated from historical data.
No taxes, dividends, or slippage are modeled.
All signals are model-estimated research signals, not investment advice.
Historical replay uses simplified assumptions and does not predict future results.
</div>
""", unsafe_allow_html=True)

    tab_live, tab_replay, tab_wl = st.tabs(["Live Paper Portfolio", "Historical Replay", "Watchlist"])

    with tab_live:
        _pp_live_tab(df)

    with tab_replay:
        _pp_replay_tab(df)

    with tab_wl:
        _pp_watchlist_tab()


def _compute_profile_weights_replay(top_df, tickers: list, profile: str = "balanced") -> dict:
    """
    Signal-based allocation weights using only replay-year fundamentals — no future returns.

    Profiles:
      conservative: 0.25×val + 0.45×qual + 0.30×risk  min 5%  max 25%
      balanced:     0.45×val + 0.35×qual + 0.20×risk  min 5%  max 40%
      aggressive:   0.60×val + 0.25×qual + 0.15×risk  min 3%  max 50%

    Components:
      val_c  = percentile rank of valuation_gap_pct among selected tickers (higher gap → higher)
      qual_c = quality_score / 100
      risk_c = 1 - filing_risk_score / 100
    """
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 100.0}

    _PROFILES = {
        "conservative": dict(w_val=0.25, w_qual=0.45, w_risk=0.30, min_w=5.0, max_w=25.0),
        "balanced":     dict(w_val=0.45, w_qual=0.35, w_risk=0.20, min_w=5.0, max_w=40.0),
        "aggressive":   dict(w_val=0.60, w_qual=0.25, w_risk=0.15, min_w=3.0, max_w=50.0),
    }
    cfg = _PROFILES.get(profile, _PROFILES["balanced"])

    vgaps, qualities, risks = [], [], []
    for t in tickers:
        rows = top_df[top_df["ticker"] == t] if top_df is not None and not top_df.empty else pd.DataFrame()
        if rows.empty:
            vgaps.append(None); qualities.append(None); risks.append(None)
            continue
        r  = rows.iloc[0]
        vg = r.get("valuation_gap_pct")
        vgaps.append(float(vg) if pd.notna(vg) else None)
        qs = r.get("quality_score")
        qualities.append(float(qs) if pd.notna(qs) else None)
        rk = r.get("report_risk_score_real")
        if rk is None or (isinstance(rk, float) and np.isnan(rk)):
            rk = r.get("report_risk_score")
        risks.append(float(rk) if pd.notna(rk) else None)

    # Percentile-rank valuation gaps (higher gap → higher rank → higher weight)
    valid_vg = [(i, v) for i, v in enumerate(vgaps) if v is not None]
    vg_ranks = [0.5] * n
    if len(valid_vg) > 1:
        sorted_vg = sorted(valid_vg, key=lambda x: x[1])
        for rank_pos, (orig_i, _) in enumerate(sorted_vg):
            vg_ranks[orig_i] = (rank_pos + 1) / len(sorted_vg)

    scores = {}
    for i, t in enumerate(tickers):
        val_c  = vg_ranks[i]
        qual_c = max(0.0, min(1.0, qualities[i] / 100)) if qualities[i] is not None else 0.5
        risk_c = max(0.0, min(1.0, 1 - risks[i] / 100)) if risks[i] is not None else 0.5
        scores[t] = (cfg["w_val"] * val_c
                     + cfg["w_qual"] * qual_c
                     + cfg["w_risk"] * risk_c)

    total   = sum(scores.values()) or 1.0
    raw_w   = {t: scores[t] / total * 100 for t in tickers}
    clipped = {t: max(cfg["min_w"], min(cfg["max_w"], w)) for t, w in raw_w.items()}
    ct      = sum(clipped.values()) or 1.0
    weights = {t: round(clipped[t] / ct * 100, 1) for t in tickers}
    diff    = round(100.0 - sum(weights.values()), 1)
    if tickers and diff != 0:
        weights[tickers[0]] = round(weights[tickers[0]] + diff, 1)
    return weights


def _compute_model_weights_replay(top_df, tickers: list) -> dict:
    """Backward-compat shim — delegates to balanced profile."""
    return _compute_profile_weights_replay(top_df, tickers, profile="balanced")


def _pp_replay_tab(df: pd.DataFrame) -> None:
    """Historical Replay tab — backtests model shortlists against yfinance price data."""
    st.markdown("### Historical Replay")
    st.caption(
        "Pick an exact start date and holding period. "
        "The replay uses the model signal from the start date's year (if available) "
        "and plots daily portfolio value alongside the chosen benchmark."
    )

    all_mo, all_mo_msg = _load_all_model_outputs()
    if all_mo is None:
        st.warning(f"Cannot run replay: {all_mo_msg}")
        return

    avail_years    = sorted(all_mo["year"].dropna().astype(int).unique())
    most_recent_yr = avail_years[-1] if avail_years else 2024
    today_ts       = pd.Timestamp.today().normalize()

    # ── Date controls ─────────────────────────────────────────────────────────
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input(
            "Start date",
            value=pd.Timestamp(f"{most_recent_yr}-01-01"),
            min_value=pd.Timestamp("2010-01-01"),
            max_value=today_ts,
            key="hr_start_date",
            help="Model signal year = year of this date. "
                 "Actual entry = first trading day on/after this date.",
        )
        end_mode = st.radio(
            "End date mode",
            ["Holding period", "Custom end date"],
            horizontal=True,
            key="hr_end_mode",
        )

    start_ts = pd.Timestamp(start_date)

    _HOLDING_MAP = {
        "1 month": 1, "3 months": 3, "6 months": 6,
        "1 year": 12, "2 years": 24, "3 years": 36,
        "5 years": 60, "10 years": 120,
    }
    with date_col2:
        if end_mode == "Holding period":
            holding_label = st.selectbox(
                "Holding period",
                list(_HOLDING_MAP.keys()) + ["To latest available date"],
                index=3,
                key="hr_holding",
                help="Period between the start date and exit.",
            )
            if holding_label == "To latest available date":
                end_ts       = today_ts
                holding_desc = "To latest available date"
            else:
                end_ts       = start_ts + pd.DateOffset(months=_HOLDING_MAP[holding_label])
                holding_desc = holding_label
        else:
            _default_end = min(
                (start_ts + pd.DateOffset(years=1)).date(), today_ts.date()
            )
            custom_end = st.date_input(
                "End date",
                value=_default_end,
                min_value=start_date,
                max_value=today_ts,
                key="hr_custom_end_date",
                help="Must be after start date. Capped at today.",
            )
            end_ts       = pd.Timestamp(custom_end)
            holding_desc = f"Custom ({start_date} to {custom_end})"
            if end_ts <= start_ts:
                st.error("End date must be after start date.")
                return

    exit_capped     = end_ts > today_ts
    display_exit_ts = min(end_ts, today_ts)
    if exit_capped:
        st.warning(
            f"Target end {end_ts.date()} is in the future — "
            "exit prices capped at the most recent available date. "
            "Returns reflect a partial holding period."
        )

    # ── Model signal year badge ───────────────────────────────────────────────
    signal_year  = start_date.year
    model_backed = signal_year in avail_years
    if model_backed:
        st.success(
            f"**Model-backed replay** — using {signal_year} annual fundamentals "
            "as the research signal."
        )
    else:
        yrs_str = ", ".join(str(y) for y in avail_years)
        st.info(
            f"**Price-only replay** — no model signal for {signal_year}. "
            f"Available signal years: {yrs_str}. Enter tickers manually below."
        )

    # ── Calibrated signal toggle (only when calibrated data available) ────────
    _replay_has_calibrated = (
        model_backed
        and "final_signal_calibrated" in all_mo.columns
        and all_mo[all_mo["year"] == signal_year]["final_signal_calibrated"].notna().any()
        if model_backed else False
    )
    use_calib_replay = False
    if _replay_has_calibrated:
        use_calib_replay = st.toggle(
            "Use calibrated historical signals",
            value=True,
            key="hr_use_calibrated",
            help=(
                "Calibrated signals rank companies by relative valuation within "
                "the same year, not absolute thresholds. "
                "Reduces over-representation of 'Potentially overvalued' in post-2017 years."
            ),
        )

    # ── Filing-risk adjusted signals toggle (replay) ──────────────────────────
    _replay_has_risk = (
        model_backed
        and use_calib_replay
        and "final_signal_calibrated_risk_adjusted" in all_mo.columns
        and all_mo[all_mo["year"] == signal_year]["final_signal_calibrated_risk_adjusted"].notna().any()
        if model_backed else False
    )
    use_risk_replay = False
    if _replay_has_risk:
        use_risk_replay = st.toggle(
            "Use filing-risk adjusted signals",
            value=True,
            key="hr_use_risk_adjusted",
            help=(
                "Uses risk-adjusted calibrated signals that incorporate SEC 10-K "
                "filing risk scores. Companies with very high filing risk may be "
                "downgraded from 'Research candidate'."
            ),
        )

    # ── Portfolio / benchmark inputs ──────────────────────────────────────────
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        initial_capital = st.number_input(
            "Initial capital ($)",
            min_value=100.0, max_value=10_000_000.0,
            value=10_000.0, step=1_000.0, format="%.2f",
            key="hr_capital",
            help="Simulated starting cash split equally across selected tickers.",
        )
        benchmark_choice = st.selectbox(
            "Benchmark",
            ["SPY", "QQQ", "Cash baseline"],
            key="hr_benchmark",
            help="Compare portfolio return against this benchmark over the same period.",
        )

    if model_backed:
        _base_methods = [
            "Research signal priority",
            "Highest valuation gap",
            "Highest quality score",
            "Manual selection",
        ]
        if use_calib_replay:
            _methods = ["Top calibrated research candidates"] + _base_methods
        else:
            _methods = _base_methods
        with cfg_col2:
            size_label = st.selectbox(
                "Portfolio size",
                ["Top 3", "Top 5", "Top 10"],
                index=1,
                key="hr_size",
            )
            method = st.selectbox(
                "Selection method",
                _methods,
                key="hr_method",
            )
        with cfg_col3:
            exclude_nr = st.checkbox(
                "Exclude needs_review rows",
                value=True, key="hr_exclude_nr",
                help="Rows flagged needs_review have suspicious model outputs.",
            )
        n_stocks = int(size_label.split()[1])
    else:
        n_stocks   = 5
        method     = "Manual selection"
        exclude_nr = False

    # ── Ticker selection ──────────────────────────────────────────────────────
    top_df = None  # set inside model_backed path; used by allocation section
    if model_backed:
        year_df = all_mo[all_mo["year"] == signal_year].copy()
        if exclude_nr:
            year_df = year_df[
                ~year_df["output_quality_flag"].str.contains("needs_review", na=False)
            ]
        if year_df.empty:
            st.warning(f"No model outputs for year {signal_year} after applying filters.")
            return

        _SIG_PRI = {"Research candidate": 0, "Monitor": 1, "Caution": 2}
        _SIG_PRI_CAL = {
            "Research candidate": 0, "Fairly valued / neutral": 1,
            "Possible value trap": 2, "Potentially overvalued": 3, "Needs review": 4,
        }

        if method == "Top calibrated research candidates" and use_calib_replay:
            # Prefer risk-adjusted signal column if available and toggled on
            if use_risk_replay and "final_signal_calibrated_risk_adjusted" in year_df.columns:
                _cal_col = "final_signal_calibrated_risk_adjusted"
            elif "final_signal_calibrated" in year_df.columns:
                _cal_col = "final_signal_calibrated"
            else:
                _cal_col = "final_signal"
            year_df["_rank"] = year_df[_cal_col].map(_SIG_PRI_CAL).fillna(99)
            year_df = year_df.sort_values(
                ["_rank", "valuation_gap_percentile_year" if "valuation_gap_percentile_year" in year_df.columns else "valuation_gap_pct"],
                ascending=[True, False],
            )
        elif method == "Research signal priority":
            year_df["_rank"] = year_df["final_signal"].map(_SIG_PRI).fillna(99)
            year_df = year_df.sort_values(
                ["_rank", "valuation_gap_pct"], ascending=[True, False]
            )
        elif method == "Highest valuation gap":
            year_df = year_df.sort_values("valuation_gap_pct", ascending=False)
        elif method == "Highest quality score":
            year_df = year_df.sort_values("quality_score", ascending=False)

        if method == "Manual selection":
            all_tickers_yr = sorted(year_df["ticker"].tolist())
            manual_sel = st.multiselect(
                "Select tickers for replay",
                all_tickers_yr,
                default=all_tickers_yr[:min(5, len(all_tickers_yr))],
                key="hr_manual_tickers",
            )
            if not manual_sel:
                st.info("Select at least one ticker to continue.")
                return
            top_df           = year_df[year_df["ticker"].isin(manual_sel)]
            selected_tickers = list(manual_sel)
        else:
            top_df           = year_df.head(n_stocks)
            selected_tickers = top_df["ticker"].tolist()

        # Model shortlist preview
        st.markdown("#### Model Shortlist")
        _short_base = ["ticker", "company_name"]
        if use_risk_replay and "final_signal_calibrated_risk_adjusted" in top_df.columns:
            _short_base += ["final_signal_calibrated_risk_adjusted", "final_signal_calibrated", "final_signal"]
        elif use_calib_replay and "final_signal_calibrated" in top_df.columns:
            _short_base += ["final_signal_calibrated", "final_signal"]
        else:
            _short_base += ["final_signal"]
        _short_base += ["valuation_gap_pct"]
        if use_calib_replay and "valuation_bucket_year" in top_df.columns:
            _short_base.append("valuation_bucket_year")
        _short_base += ["quality_score"]
        if use_risk_replay and "report_risk_score_real" in top_df.columns:
            _short_base.append("report_risk_score_real")
        _short_base += ["output_quality_flag"]
        short_cols = [c for c in _short_base if c in top_df.columns]
        short_disp = top_df[short_cols].copy().rename(columns={
            "ticker":                                "Ticker",
            "company_name":                          "Company",
            "final_signal":                          "Original Signal",
            "final_signal_calibrated":               "Calibrated Signal",
            "final_signal_calibrated_risk_adjusted": "Risk-Adj Signal",
            "valuation_gap_pct":                     "Val. Gap %",
            "valuation_bucket_year":                 "Value Bucket",
            "quality_score":                         "Quality",
            "report_risk_score_real":                "Filing Risk",
            "output_quality_flag":                   "Data Quality",
        })
        if "Val. Gap %" in short_disp.columns:
            short_disp["Val. Gap %"] = short_disp["Val. Gap %"].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else ""
            )
        if "Quality" in short_disp.columns:
            short_disp["Quality"] = short_disp["Quality"].apply(
                lambda v: f"{v:.0f}" if pd.notna(v) else ""
            )
        if "Filing Risk" in short_disp.columns:
            short_disp["Filing Risk"] = short_disp["Filing Risk"].apply(
                lambda v: f"{v:.0f}/100" if pd.notna(v) else "—"
            )
        st.dataframe(
            short_disp.reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )
    else:
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            placeholder="e.g. AAPL, MSFT, GOOGL",
            key="hr_tickers_text",
            help="Enter ticker symbols separated by commas.",
        )
        if not ticker_input.strip():
            st.info("Enter at least one ticker symbol to run a price-only replay.")
            return
        selected_tickers = [
            t.strip().upper() for t in ticker_input.split(",") if t.strip()
        ]

    all_fetch_tickers = (
        list(selected_tickers)
        + ([] if benchmark_choice == "Cash baseline" else [benchmark_choice])
    )
    start_date_str = str(start_ts.date())
    end_date_str   = str(display_exit_ts.date())

    # ── Portfolio Allocation ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Portfolio Allocation")

    _n_sel = len(selected_tickers)
    _eq_w  = {t: round(100.0 / max(_n_sel, 1), 1) for t in selected_tickers}
    # Fix equal-weight rounding to ensure sum = 100
    _eq_diff = round(100.0 - sum(_eq_w.values()), 1)
    if selected_tickers and _eq_diff != 0:
        _eq_w[selected_tickers[0]] = round(_eq_w[selected_tickers[0]] + _eq_diff, 1)

    _alloc_opts = ["Equal weight"]
    if model_backed and top_df is not None:
        _alloc_opts += [
            "Conservative profile",
            "Balanced signal-based weight",
            "Aggressive profile",
        ]
    _alloc_opts.append("Custom weight")

    alloc_method = st.radio(
        "Allocation method",
        _alloc_opts,
        horizontal=True,
        key="hr_alloc_method",
    )

    # Profile caption (shown for any signal-based profile)
    _is_profile = alloc_method in (
        "Conservative profile", "Balanced signal-based weight", "Aggressive profile"
    )
    if _is_profile:
        st.caption(
            "Profile weights are based on valuation, quality, and filing-risk signals "
            "available at the replay start. They are not optimized using future returns. "
            "Min/max position caps vary by profile."
        )

    # Compute display weights
    if alloc_method == "Equal weight":
        _preview_weights = dict(_eq_w)

    elif alloc_method == "Conservative profile":
        _preview_weights = _compute_profile_weights_replay(
            top_df, selected_tickers, profile="conservative"
        )
        st.caption(
            "Conservative — quality-first: 0.25×valuation + 0.45×quality + 0.30×low-risk. "
            "Min 5% / max 25% per position."
        )
        if _n_sel > 1 and st.button("Copy to custom weights", key="hr_cw_copy_cons"):
            for _tk, _mw in _preview_weights.items():
                st.session_state[f"hr_cw_{_tk}"] = _mw
            st.info("Copied. Switch to **Custom weight** to adjust.")

    elif alloc_method == "Balanced signal-based weight":
        _preview_weights = _compute_profile_weights_replay(
            top_df, selected_tickers, profile="balanced"
        )
        st.caption(
            "Balanced — 0.45×valuation + 0.35×quality + 0.20×low-risk. "
            "Min 5% / max 40% per position."
        )
        if _n_sel > 1 and st.button("Copy to custom weights", key="hr_cw_copy_bal"):
            for _tk, _mw in _preview_weights.items():
                st.session_state[f"hr_cw_{_tk}"] = _mw
            st.info("Copied. Switch to **Custom weight** to adjust.")

    elif alloc_method == "Aggressive profile":
        _preview_weights = _compute_profile_weights_replay(
            top_df, selected_tickers, profile="aggressive"
        )
        st.caption(
            "Aggressive — valuation-first: 0.60×valuation + 0.25×quality + 0.15×low-risk. "
            "Min 3% / max 50% per position."
        )
        if _n_sel > 1 and st.button("Copy to custom weights", key="hr_cw_copy_agg"):
            for _tk, _mw in _preview_weights.items():
                st.session_state[f"hr_cw_{_tk}"] = _mw
            st.info("Copied. Switch to **Custom weight** to adjust.")

    else:  # Custom weight
        _preview_weights = _eq_w.copy()  # starting point; overridden by widgets below

    if alloc_method == "Custom weight":
        st.caption(
            f"Enter allocation percentages (must total 100%). "
            f"Starting capital: ${initial_capital:,.0f}"
        )
        _ncw = min(_n_sel, 4)
        _wcols = st.columns(_ncw)
        _custom_w = {}
        for _ci, _tk in enumerate(selected_tickers):
            with _wcols[_ci % _ncw]:
                _def = float(
                    st.session_state.get(f"hr_cw_{_tk}", _eq_w.get(_tk, 100.0 / max(_n_sel, 1)))
                )
                _custom_w[_tk] = st.number_input(
                    f"{_tk} (%)",
                    min_value=0.0, max_value=100.0,
                    value=_def, step=1.0, format="%.1f",
                    key=f"hr_cw_{_tk}",
                )
        _cw_total = sum(_custom_w.values())
        _cw_ok    = abs(_cw_total - 100.0) < 0.11
        _cwc1, _cwc2, _cwc3 = st.columns(3)
        with _cwc1:
            (st.success if _cw_ok else st.warning)(f"Total: {_cw_total:.1f}%")
        with _cwc2:
            if st.button("Normalize to 100%", key="hr_cw_norm"):
                _nt = sum(_custom_w.values())
                if _nt > 0:
                    for _tk in selected_tickers:
                        st.session_state[f"hr_cw_{_tk}"] = round(_custom_w[_tk] / _nt * 100, 1)
                    st.rerun()
        with _cwc3:
            if st.button("Reset to equal weight", key="hr_cw_reset_eq"):
                for _tk in selected_tickers:
                    st.session_state[f"hr_cw_{_tk}"] = round(100.0 / max(_n_sel, 1), 1)
                st.rerun()
        if any(v < 0 for v in _custom_w.values()):
            st.error("Weights cannot be negative.")
            _custom_w = dict(_eq_w)
        _max_cw = max(_custom_w.values()) if _custom_w else 0
        if _max_cw > 60.0:
            st.warning(
                f"Concentration risk: one position is {_max_cw:.0f}% of the portfolio. "
                "This is a research simulation only."
            )
        _preview_weights = dict(_custom_w)

    # Allocation preview table
    _alloc_rows = []
    for _tk in selected_tickers:
        _wpct    = _preview_weights.get(_tk, 100.0 / max(_n_sel, 1))
        _dollars = initial_capital * _wpct / 100
        _arow    = {"Ticker": _tk, "Weight %": f"{_wpct:.1f}%", "$ Allocation": f"${_dollars:,.0f}"}
        if model_backed and top_df is not None:
            _mr_rows = top_df[top_df["ticker"] == _tk]
            if not _mr_rows.empty:
                _mr = _mr_rows.iloc[0]
                _sig_col = (
                    "final_signal_calibrated_risk_adjusted"
                    if use_risk_replay and "final_signal_calibrated_risk_adjusted" in top_df.columns
                    else "final_signal_calibrated"
                    if use_calib_replay and "final_signal_calibrated" in top_df.columns
                    else "final_signal"
                )
                _arow["Signal"]    = str(_mr.get(_sig_col) or _mr.get("final_signal") or "")
                _vg = _mr.get("valuation_gap_pct")
                _arow["Val. Gap %"] = f"{float(_vg):+.1f}%" if pd.notna(_vg) else "—"
                _qs = _mr.get("quality_score")
                _arow["Quality"]   = f"{float(_qs):.0f}" if pd.notna(_qs) else "—"
        _alloc_rows.append(_arow)
    st.dataframe(pd.DataFrame(_alloc_rows), use_container_width=True, hide_index=True)

    # Auto-normalize and store final run weights
    _run_weights_raw = dict(_preview_weights)
    _rw_sum = sum(_run_weights_raw.values())
    if _rw_sum > 0:
        _run_weights = {t: v / _rw_sum * 100 for t, v in _run_weights_raw.items()}
    else:
        _run_weights = {t: 100.0 / max(_n_sel, 1) for t in selected_tickers}
    if abs(sum(_run_weights_raw.values()) - 100.0) > 0.5:
        st.warning(
            f"Weights sum to {sum(_run_weights_raw.values()):.1f}% — "
            "will be auto-normalized to 100% when the simulation runs."
        )

    # ── Run + Test ────────────────────────────────────────────────────────────
    st.divider()
    btn_c1, btn_c2, _ = st.columns([2, 2, 4])
    run_clicked  = btn_c1.button(
        "Run Historical Replay", type="primary", key="hr_run_btn"
    )
    test_clicked = btn_c2.button(
        "Test price fetch", key="hr_test_btn",
        help="Quick sanity check — shows status for each ticker before running the full replay.",
    )

    if test_clicked:
        with st.spinner(f"Testing {len(all_fetch_tickers)} tickers…"):
            test_prices = _fetch_replay_daily_cached(
                tuple(sorted(all_fetch_tickers)), start_date_str, end_date_str,
            )
        debug_rows = []
        for t in all_fetch_tickers:
            p = test_prices.get(t, {})
            debug_rows.append({
                "Ticker":      t,
                "Used symbol": p.get("ticker_used", t),
                "Status":      p.get("status", "?"),
                "Reason":      p.get("reason", ""),
                "Rows":        p.get("rows", 0),
                "Entry date":  p.get("entry_date") or "—",
                "Entry price": f"${p['entry_price']:,.2f}" if p.get("entry_price") else "—",
                "Exit date":   p.get("exit_date") or "—",
                "Exit price":  f"${p['exit_price']:,.2f}" if p.get("exit_price") else "—",
            })
        ok_n = sum(1 for r in debug_rows if r["Status"] == "ok")
        st.markdown("**Price fetch test results:**")
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"{ok_n}/{len(debug_rows)} tickers returned valid entry+exit prices.  "
            f"Start: {start_date_str}  |  Exit cap: {end_date_str}."
        )

    if run_clicked:
        with st.spinner(
            f"Fetching daily prices for {len(all_fetch_tickers)} tickers "
            f"({start_date_str} → {end_date_str})…"
        ):
            prices = _fetch_replay_daily_cached(
                tuple(sorted(all_fetch_tickers)), start_date_str, end_date_str,
            )
        st.session_state["hr_last_prices"]  = prices
        st.session_state["hr_last_tickers"] = list(selected_tickers)
        st.session_state["hr_last_params"]  = {
            "signal_year":     signal_year,
            "model_backed":    model_backed,
            "holding_desc":    holding_desc,
            "initial_capital": initial_capital,
            "benchmark":       benchmark_choice,
            "method":          method,
            "start_date":      start_date_str,
            "target_exit":     str(end_ts.date()),
            "display_exit":    end_date_str,
            "exit_capped":     exit_capped,
            "weights":         dict(_run_weights),
            "alloc_method":    alloc_method,
        }

    # ── Results ───────────────────────────────────────────────────────────────
    if "hr_last_prices" not in st.session_state:
        return

    prices     = st.session_state["hr_last_prices"]
    params     = st.session_state["hr_last_params"]
    last_ticks = st.session_state["hr_last_tickers"]

    if (sorted(last_ticks) != sorted(selected_tickers)
            or params.get("start_date") != start_date_str
            or params.get("display_exit") != end_date_str):
        st.info("Parameters changed — click **Run Historical Replay** to refresh results.")
        return

    capital   = params["initial_capital"]
    bm_choice = params["benchmark"]

    ok_tickers   = [t for t in last_ticks if prices.get(t, {}).get("status") == "ok"]
    fail_tickers = [t for t in last_ticks if t not in ok_tickers]

    if fail_tickers:
        with st.expander(
            f"⚠ {len(fail_tickers)} ticker(s) excluded — price data unavailable "
            "(click to see details)"
        ):
            st.dataframe(
                pd.DataFrame([{
                    "Ticker":      t,
                    "Used symbol": prices.get(t, {}).get("ticker_used", t),
                    "Status":      prices.get(t, {}).get("status", "?"),
                    "Reason":      prices.get(t, {}).get("reason", ""),
                    "Rows":        prices.get(t, {}).get("rows", 0),
                    "Entry date":  prices.get(t, {}).get("entry_date") or "—",
                    "Exit date":   prices.get(t, {}).get("exit_date") or "—",
                } for t in fail_tickers]),
                use_container_width=True, hide_index=True,
            )

    if not ok_tickers:
        st.error(
            "No price data available for any selected ticker. Cannot compute replay.  \n"
            f"**Start date:** {params.get('start_date')}  |  "
            f"**Exit cap:** {params.get('display_exit')}  |  "
            f"**Tickers attempted:** {len(last_ticks)}  \n"
            "Use the **Test price fetch** button above to diagnose the issue."
        )
        return

    # ── Replay date summary box ───────────────────────────────────────────────
    actual_entries  = [prices[t]["entry_date"] for t in ok_tickers if prices[t].get("entry_date")]
    actual_exits    = [prices[t]["exit_date"]   for t in ok_tickers if prices[t].get("exit_date")]
    first_entry     = min(actual_entries) if actual_entries else params["start_date"]
    last_exit       = max(actual_exits)   if actual_exits   else params["display_exit"]
    replay_type_lbl = "Model-backed replay" if params.get("model_backed") else "Price-only replay"
    mo_note         = "(model output available)" if params.get("model_backed") else "(no model output for this year)"

    st.markdown(f"""
<div class="replay-summary-box">
<strong>{replay_type_lbl}</strong> &nbsp;|&nbsp;
Signal year: <strong>{params['signal_year']}</strong> {mo_note}
<table style="margin-top:8px">
<tr>
  <td style="color:#64748b;padding-right:16px">User start date</td>
  <td><strong>{params['start_date']}</strong></td>
  <td style="padding-left:28px;color:#64748b">User end / holding</td>
  <td><strong>{params['holding_desc']}</strong></td>
</tr>
<tr>
  <td style="color:#64748b">Actual entry date</td>
  <td><strong>{first_entry}</strong></td>
  <td style="padding-left:28px;color:#64748b">Actual exit date</td>
  <td><strong>{last_exit}</strong></td>
</tr>
<tr>
  <td style="color:#64748b">Companies included</td>
  <td><strong>{len(ok_tickers)}</strong>{f' ({len(fail_tickers)} excluded)' if fail_tickers else ''}</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Compute returns ───────────────────────────────────────────────────────
    # Use per-stock weights from params; fall back to equal weight if missing
    _res_weights = params.get("weights", {})
    _res_alloc_method = params.get("alloc_method", "Equal weight")
    if not _res_weights or not all(t in _res_weights for t in ok_tickers):
        _res_weights = {t: 100.0 / len(ok_tickers) for t in ok_tickers}
    _rw_ok_total = sum(_res_weights.get(t, 0) for t in ok_tickers)
    if _rw_ok_total <= 0:
        _rw_ok_total = len(ok_tickers)
    # Normalize to 100% across only the ok_tickers (excluded tickers don't get allocation)
    _ok_weights = {t: _res_weights.get(t, 0) / _rw_ok_total * 100 for t in ok_tickers}

    total_end_val = 0.0
    stock_rows    = []

    for t in ok_tickers:
        p           = prices[t]
        ep, xp      = p["entry_price"], p["exit_price"]
        _wpct       = _ok_weights.get(t, 100.0 / len(ok_tickers))
        alloc_amt   = capital * _wpct / 100
        shares      = alloc_amt / ep
        end_val     = shares * xp
        ret_pct     = (xp - ep) / ep * 100
        contrib_pct = _wpct / 100 * ret_pct  # contribution to portfolio return
        total_end_val += end_val

        match_row = all_mo[
            (all_mo["ticker"] == t) & (all_mo["year"] == params["signal_year"])
        ]
        signal  = match_row["final_signal"].iloc[0]      if not match_row.empty else ""
        vgap    = match_row["valuation_gap_pct"].iloc[0] if not match_row.empty else None
        quality = match_row["quality_score"].iloc[0]     if not match_row.empty else None
        risk    = match_row["report_risk_score"].iloc[0] if not match_row.empty else None
        company = (
            match_row["company_name"].iloc[0]
            if not match_row.empty and "company_name" in match_row.columns else t
        )
        stock_rows.append({
            "Ticker":           t,
            "Company":          company,
            "Signal year":      params["signal_year"],
            "Weight %":         _wpct,
            "Entry date":       p.get("entry_date", ""),
            "Entry price":      ep,
            "Exit date":        p.get("exit_date", ""),
            "Exit price":       xp,
            "Allocation ($)":   alloc_amt,
            "Shares":           shares,
            "Return %":         ret_pct,
            "Dollar Gain/Loss": end_val - alloc_amt,
            "Contribution %":   contrib_pct,
            "End value ($)":    end_val,
            "Signal":           signal,
            "Val. gap %":       vgap,
            "Quality score":    quality,
            "Report risk":      risk,
            "Replay type":      replay_type_lbl,
        })

    port_return_pct = (total_end_val - capital) / capital * 100
    portfolio_pl    = total_end_val - capital

    # Benchmark
    bm_return_pct   = 0.0
    bm_status_msg   = ""
    bm_daily_series = None
    if bm_choice == "Cash baseline":
        bm_return_pct = 0.0
        bm_status_msg = "Cash baseline: 0% return"
    else:
        bm_data = prices.get(bm_choice, {})
        if bm_data.get("status") == "ok":
            bm_ep           = bm_data["entry_price"]
            bm_xp           = bm_data["exit_price"]
            bm_return_pct   = (bm_xp - bm_ep) / bm_ep * 100
            bm_status_msg   = f"{bm_choice}: ${bm_ep:.2f} → ${bm_xp:.2f}"
            bm_daily_series = bm_data.get("daily_series")
        else:
            bm_status_msg = (
                f"{bm_choice} data unavailable "
                f"({bm_data.get('status','')} — {bm_data.get('reason','')}). "
                "Benchmark return shown as 0%."
            )
            st.warning(bm_status_msg)

    alpha = port_return_pct - bm_return_pct

    # ── Compute risk metrics from daily price series ──────────────────────────
    _port_vals_curve    = None
    _max_drawdown_pct   = None
    _ann_vol_pct        = None
    _best_day_pct       = None
    _worst_day_pct      = None
    _ann_return_pct     = None
    _risk_adj_return    = None
    _bm_drawdown_series = None

    _holding_days = 0
    try:
        _holding_days = (
            pd.Timestamp(last_exit) - pd.Timestamp(first_entry)
        ).days
    except Exception:
        pass

    daily_map = {
        t: prices[t]["daily_series"]
        for t in ok_tickers
        if prices[t].get("daily_series") is not None
    }
    if daily_map:
        price_df = pd.concat(
            [s.rename(t) for t, s in daily_map.items()], axis=1
        ).sort_index().ffill()

        port_vals = pd.Series(0.0, index=price_df.index)
        for t in ok_tickers:
            if t not in price_df.columns:
                continue
            _ep = prices[t].get("entry_price")
            if not _ep or _ep <= 0:
                continue
            _wpct_curve   = _ok_weights.get(t, 100.0 / len(ok_tickers))
            _shares_curve = capital * _wpct_curve / 100 / _ep
            port_vals    += price_df[t] * _shares_curve
        port_vals = port_vals[port_vals > 0]

        if not port_vals.empty:
            _port_vals_curve = port_vals
            if len(port_vals) > 1:
                _port_daily_rets = port_vals.pct_change().dropna()
                if len(_port_daily_rets) > 0:
                    _best_day_pct  = float(_port_daily_rets.max() * 100)
                    _worst_day_pct = float(_port_daily_rets.min() * 100)
                    _std           = float(_port_daily_rets.std())
                    if _std > 0:
                        _ann_vol_pct = _std * (252 ** 0.5) * 100
                _rolling_max      = port_vals.cummax()
                _dd_series        = (port_vals - _rolling_max) / _rolling_max * 100
                _max_drawdown_pct = float(_dd_series.min())
                if _holding_days >= 365 and capital > 0:
                    _ann_return_pct = float(
                        ((total_end_val / capital) ** (365.0 / _holding_days) - 1) * 100
                    )
                if (_ann_vol_pct is not None and _ann_vol_pct > 0.01
                        and _ann_return_pct is not None):
                    _risk_adj_return = _ann_return_pct / _ann_vol_pct

        if bm_daily_series is not None and not bm_daily_series.empty:
            try:
                _bm_rm  = bm_daily_series.cummax()
                _bm_drawdown_series = (bm_daily_series - _bm_rm) / _bm_rm * 100
            except Exception:
                pass

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("#### Replay Results")

    # Row 1 — performance
    _r1 = st.columns(6)
    _r1[0].metric("Starting Value", f"${capital:,.0f}")
    _r1[1].metric("Ending Value",   f"${total_end_val:,.0f}",
                  delta=f"${portfolio_pl:+,.0f}")
    _r1[2].metric("Portfolio Return", f"{port_return_pct:+.1f}%",
                  help="Total return over replay period. Research context only.")
    _r1[3].metric(f"Benchmark ({bm_choice})", f"{bm_return_pct:+.1f}%",
                  help=bm_status_msg + "  Benchmark comparison is context only.")
    _r1[4].metric("Alpha (vs benchmark)", f"{alpha:+.1f}%",
                  help="Portfolio return minus benchmark return. Historical context only.")
    _r1[5].metric(
        "Annualized Return",
        f"{_ann_return_pct:+.1f}%" if _ann_return_pct is not None else "Unavailable",
        help="Annualized from actual holding period. Requires ≥365 days.",
    )

    # Row 2 — risk
    _r2 = st.columns(5)
    _r2[0].metric(
        "Max Drawdown",
        f"{_max_drawdown_pct:.1f}%" if _max_drawdown_pct is not None else "Unavailable",
        help="Largest peak-to-trough decline during replay.",
    )
    _r2[1].metric(
        "Ann. Volatility",
        f"{_ann_vol_pct:.1f}%" if _ann_vol_pct is not None else "Unavailable",
        help="Annualized std dev of daily returns (×√252).",
    )
    _r2[2].metric(
        "Best Day",
        f"{_best_day_pct:+.2f}%" if _best_day_pct is not None else "Unavailable",
    )
    _r2[3].metric(
        "Worst Day",
        f"{_worst_day_pct:+.2f}%" if _worst_day_pct is not None else "Unavailable",
    )
    _r2[4].metric(
        "Risk-Adj. Return",
        f"{_risk_adj_return:.2f}x" if _risk_adj_return is not None else "Unavailable",
        help="Annualized return ÷ annualized volatility. No risk-free rate deducted.",
    )

    # ── Best/worst summary cards ──────────────────────────────────────────────
    if len(stock_rows) > 1:
        _best_contrib   = max(stock_rows, key=lambda r: r["Contribution %"])
        _worst_contrib  = min(stock_rows, key=lambda r: r["Contribution %"])
        _best_ret_row   = max(stock_rows, key=lambda r: r["Return %"])
        _largest_wt_row = max(stock_rows, key=lambda r: r["Weight %"])
        _bw_cols = st.columns(4)
        _bw_cols[0].markdown(
            f'<div class="mini-card"><div class="label">Best Contributor</div>'
            f'<div class="value" style="color:#27ae60">{_best_contrib["Ticker"]}</div>'
            f'<div style="font-size:11px;color:#64748b">'
            f'{_best_contrib["Contribution %"]:+.2f}% to portfolio</div></div>',
            unsafe_allow_html=True,
        )
        _bw_cols[1].markdown(
            f'<div class="mini-card"><div class="label">Worst Contributor</div>'
            f'<div class="value" style="color:#dc2626">{_worst_contrib["Ticker"]}</div>'
            f'<div style="font-size:11px;color:#64748b">'
            f'{_worst_contrib["Contribution %"]:+.2f}% to portfolio</div></div>',
            unsafe_allow_html=True,
        )
        _bw_cols[2].markdown(
            f'<div class="mini-card"><div class="label">Highest Return</div>'
            f'<div class="value" style="color:#2563eb">{_best_ret_row["Ticker"]}</div>'
            f'<div style="font-size:11px;color:#64748b">'
            f'{_best_ret_row["Return %"]:+.1f}%</div></div>',
            unsafe_allow_html=True,
        )
        _bw_cols[3].markdown(
            f'<div class="mini-card"><div class="label">Largest Weight</div>'
            f'<div class="value">{_largest_wt_row["Ticker"]}</div>'
            f'<div style="font-size:11px;color:#64748b">'
            f'{_largest_wt_row["Weight %"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    if _port_vals_curve is not None:
        port_vals  = _port_vals_curve
        fig_line   = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=port_vals.index, y=[capital] * len(port_vals),
            mode="lines", name="Cash baseline",
            line=dict(color="#94a3b8", dash="dot", width=1),
        ))
        if bm_daily_series is not None:
            bm_norm    = bm_daily_series / prices[bm_choice]["entry_price"] * capital
            bm_aligned = bm_norm.reindex(port_vals.index).ffill().bfill()
            fig_line.add_trace(go.Scatter(
                x=bm_aligned.index, y=bm_aligned.values,
                mode="lines", name=bm_choice,
                line=dict(color="#2563eb", width=2),
            ))
        _line_col = "#16a34a" if port_return_pct >= 0 else "#dc2626"
        fig_line.add_trace(go.Scatter(
            x=port_vals.index, y=port_vals.values,
            mode="lines", name="Portfolio",
            line=dict(color=_line_col, width=2.5),
        ))
        fig_line.update_layout(
            title=(
                f"Portfolio Value Over Time — {replay_type_lbl} "
                f"({params.get('holding_desc','')})"
            ),
            yaxis_title="Portfolio Value ($)",
            height=400,
            margin=dict(t=50, b=40, l=0, r=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Drawdown chart
        if _max_drawdown_pct is not None:
            _dd_plot = (port_vals - port_vals.cummax()) / port_vals.cummax() * 100
            fig_dd   = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=_dd_plot.index, y=_dd_plot.values,
                mode="lines", name="Portfolio drawdown",
                fill="tozeroy",
                line=dict(color="#dc2626", width=1.5),
                fillcolor="rgba(220,38,38,0.10)",
            ))
            if _bm_drawdown_series is not None:
                _bm_dd_al = _bm_drawdown_series.reindex(_dd_plot.index).ffill().bfill()
                fig_dd.add_trace(go.Scatter(
                    x=_bm_dd_al.index, y=_bm_dd_al.values,
                    mode="lines", name=f"{bm_choice} drawdown",
                    line=dict(color="#2563eb", width=1.5, dash="dot"),
                ))
            fig_dd.update_layout(
                title="Drawdown Over Time",
                yaxis_title="Drawdown (%)",
                height=260,
                margin=dict(t=40, b=30, l=0, r=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    # ── Position contribution table ───────────────────────────────────────────
    st.markdown(
        f"#### Position Contributions  "
        f"<span style='font-size:12px;color:#64748b;font-weight:400'>"
        f"Allocation: {_res_alloc_method}</span>",
        unsafe_allow_html=True,
    )
    stock_result_df = pd.DataFrame(stock_rows)
    if not stock_result_df.empty:
        _contrib_cols = [
            "Ticker", "Company", "Weight %", "Entry price", "Exit price",
            "Return %", "Dollar Gain/Loss", "Contribution %", "End value ($)",
        ]
        _avail = [c for c in _contrib_cols if c in stock_result_df.columns]
        display_df = stock_result_df[_avail].copy()
        display_df["Weight %"]       = display_df["Weight %"].apply(lambda v: f"{v:.1f}%")
        display_df["Return %"]       = display_df["Return %"].apply(lambda v: f"{v:+.1f}%")
        display_df["Contribution %"] = display_df["Contribution %"].apply(lambda v: f"{v:+.2f}%")
        for _col in ["Entry price", "Exit price"]:
            if _col in display_df.columns:
                display_df[_col] = display_df[_col].apply(lambda v: f"${v:,.2f}")
        if "Dollar Gain/Loss" in display_df.columns:
            display_df["Dollar Gain/Loss"] = display_df["Dollar Gain/Loss"].apply(
                lambda v: f"${v:+,.2f}" if pd.notna(v) else "—"
            )
        if "End value ($)" in display_df.columns:
            display_df["End value ($)"] = display_df["End value ($)"].apply(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
            )

        def _color_ret(val):
            try:
                v = float(str(val).replace("%", "").replace("+", ""))
                return "color:#27ae60;font-weight:600" if v >= 0 else "color:#c0392b;font-weight:600"
            except Exception:
                return ""

        try:
            st.dataframe(
                display_df.style.applymap(
                    _color_ret, subset=["Return %", "Contribution %"]
                ),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Individual returns bar chart ──────────────────────────────────────────
    if stock_rows:
        chart_tickers = [r["Ticker"]   for r in stock_rows]
        chart_returns = [r["Return %"] for r in stock_rows]
        chart_colors  = ["#27ae60" if r >= 0 else "#c0392b" for r in chart_returns]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=chart_tickers, y=chart_returns,
            marker_color=chart_colors,
            text=[f"{r:+.1f}%" for r in chart_returns],
            textposition="outside",
            name="Individual Returns",
        ))
        fig_bar.add_hline(
            y=bm_return_pct, line_dash="dash", line_color="#2563eb",
            annotation_text=f"{bm_choice}: {bm_return_pct:+.1f}%",
            annotation_position="top right",
        )
        fig_bar.add_hline(y=0, line_dash="solid", line_color="#888", line_width=1)
        fig_bar.update_layout(
            title=(
                f"Individual Returns vs {bm_choice} "
                f"({params.get('holding_desc','')} from {params.get('start_date','')})"
            ),
            yaxis_title="Return %",
            height=380,
            margin=dict(t=50, b=40, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── How to read these metrics ─────────────────────────────────────────────
    with st.expander("How to read these replay metrics"):
        st.markdown("""
**Portfolio performance**
- **Total return** — ending value minus starting value, as a percentage of the starting value.
- **Annualized return** — total return rescaled to a per-year rate. Only shown for holding periods ≥ 365 days.
- **Alpha** — portfolio return minus benchmark return over the same window. A positive value means the simulated portfolio outperformed the benchmark *historically* — it does not predict future outperformance.

**Risk metrics**
- **Max drawdown** — the largest peak-to-trough decline of the daily portfolio value during the replay. Measures worst cumulative loss from a high point.
- **Annualized volatility** — standard deviation of daily portfolio returns multiplied by √252 to annualize. Higher = more day-to-day variability.
- **Risk-adjusted return** — annualized return divided by annualized volatility. Not a Sharpe ratio (no risk-free rate is deducted). Gives a rough sense of return per unit of variability; only meaningful for holding periods ≥ 365 days.
- **Best / worst day** — the single best and worst daily portfolio return during the replay window.

**Position contributions**
- **Contribution %** — fraction of total portfolio return attributable to this position. A 20%-weight position that gains 10% contributes +2.0% to the portfolio return.
- **Dollar gain/loss** — ending value of this position minus its initial allocation.

**Important guardrails**
Historical replay is a research simulation using past data and is not a prediction of future results.
- No transaction costs, taxes, dividends, or slippage are modeled.
- Survivorship bias: only tickers with available yfinance price data are included.
- Drawdown and volatility are computed from simulated daily portfolio values, not real traded positions.
- Benchmark comparison is historical context only, not proof of future performance.
        """)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="disclaimer-box" style="margin-top:24px">
<strong>Historical Replay — research simulation, not investment advice:</strong><br>
Historical Replay uses annual model signals mapped to the selected start date's year.
It is a research simulation, not a prediction or investment recommendation.<br>
• Prices from yfinance may differ from actual execution prices (bid/ask spread, market hours).<br>
• No transaction costs, taxes, dividends, or slippage are modeled.<br>
• Allocation is a research simulation; real portfolios require rebalancing and ongoing risk management.<br>
• Survivorship bias: only tickers with available yfinance data are included.<br>
• Past simulated performance does not predict future real-world results.<br>
• This tool is for educational research only and is not investment advice.
</div>
""", unsafe_allow_html=True)


def _pp_live_tab(df: pd.DataFrame) -> None:
    """Live Paper Portfolio tab content. No real trades. No real money."""
    # Load portfolio state (non-cached — always current)
    portfolio    = load_paper_portfolio()
    transactions = load_paper_transactions()

    # Load supplementary data for price lookup
    mo_df, _, cm_df_raw, _ = _load_supplementary()
    cm_df = cm_df_raw  # may be None

    # ── Account setup / reset ─────────────────────────────────────────────────
    with st.expander("Account Setup", expanded=not portfolio.get("initialized", False)):
        col_s, col_b = st.columns([2, 1])
        with col_s:
            start_cash = st.number_input(
                "Starting cash ($)",
                min_value=100.0, max_value=10_000_000.0,
                value=float(portfolio.get("starting_cash", DEFAULT_STARTING_CASH)),
                step=1000.0, format="%.2f",
                key="pp_start_cash",
            )
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            reset_confirm = st.checkbox(
                "I understand this will permanently delete all trades and history",
                key="pp_reset_confirm",
            )
            init_btn = st.button(
                "Initialize / Reset Account", type="primary",
                key="pp_init_btn",
                disabled=not reset_confirm,
            )
        if init_btn and reset_confirm:
            st.warning(
                "**Reset will delete:**  \n"
                "- `paper_transactions.csv` (all recorded trades)  \n"
                "- `paper_portfolio_snapshot.csv` (cash balance)  \n"
                "- `paper_portfolio_history.csv` (value history)"
            )
            portfolio    = initialize_paper_account(start_cash)
            transactions = load_paper_transactions()
            st.success(f"Account initialized with ${start_cash:,.2f}. All data cleared.")
            st.rerun()
        if portfolio.get("initialized"):
            st.caption(
                f"Account created: {str(portfolio.get('created_at',''))[:19]}  |  "
                f"Last updated: {str(portfolio.get('last_updated',''))[:19]}"
            )

    if not portfolio.get("initialized", False):
        st.info("Initialize your paper account above to begin.")
        return

    # ── Compute holdings and portfolio value ──────────────────────────────────
    holdings  = calculate_holdings(transactions)
    total_val, invested_val, price_map = calculate_portfolio_value(
        holdings, float(portfolio.get("current_cash", 0)), cm_df=cm_df, sample_df=df
    )
    metrics        = calculate_portfolio_metrics(total_val, invested_val, portfolio, holdings=holdings)
    real_summary   = calculate_realized_summary(holdings)

    # Record history snapshot after computing values
    update_portfolio_history(total_val, float(portfolio.get("current_cash", 0)), invested_val)

    # ── Portfolio analytics ───────────────────────────────────────────────────
    st.markdown("### Portfolio Analytics")

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Total Value",       f"${metrics['total_value']:,.2f}")
    a2.metric("Starting Cash",     f"${metrics['starting_cash']:,.2f}")
    a3.metric("Available Cash",    f"${metrics['current_cash']:,.2f}")
    a4.metric("Invested",          f"${metrics['invested_value']:,.2f}")
    a5.metric("Equity Exposure",   f"{metrics['exposure_pct']:.1f}%")

    b1, b2, b3, b4, b5 = st.columns(5)
    tr_sign = "+" if metrics["total_return_pct"] >= 0 else ""
    b1.metric("Total Return",
              f"{tr_sign}{metrics['total_return_pct']:.2f}%",
              delta=f"${metrics['total_pl']:+,.2f}")
    b2.metric("Unrealized P/L",    f"${metrics['unrealized_pl']:+,.2f}")
    b3.metric("Realized P/L",      f"${metrics['realized_pl']:+,.2f}")
    b4.metric("Open Positions",    str(metrics["num_positions"]))
    b5.metric("Largest Position",  f"{metrics['largest_position_pct']:.1f}%")

    if metrics.get("concentration_warning"):
        st.warning(
            f"Concentration risk: the largest position represents "
            f"{metrics['largest_position_pct']:.1f}% of the portfolio (above 40% threshold). "
            "Simulated result — not investment advice."
        )

    st.divider()

    # ── Simulated Order Ticket ────────────────────────────────────────────────
    st.markdown("### Simulated Order Ticket")

    # Apply quick-set presets BEFORE widgets render (pending pattern)
    for _preset, _target in [("pp_qty_preset", "pp_qty"), ("pp_dollar_preset", "pp_dollars")]:
        if _preset in st.session_state:
            st.session_state[_target] = st.session_state.pop(_preset)

    open_pos = (
        holdings[holdings["is_open"]]
        if not holdings.empty and "is_open" in holdings.columns
        else pd.DataFrame()
    )

    # Ticker select — full width so reference price auto-updates on ticker change
    avail_tickers = sorted(set(
        get_available_tickers(mo_df=mo_df, sample_df=df)
        + (holdings["ticker"].tolist() if not holdings.empty else [])
    ))
    if not avail_tickers:
        st.warning("No tickers available. Load sample data or run market_cap_utils.py.")
        return

    prefill     = st.session_state.get("paper_ticker_prefill")
    prefill_idx = avail_tickers.index(prefill) if (prefill and prefill in avail_tickers) else 0
    selected_ticker = st.selectbox(
        "Ticker",
        avail_tickers,
        index=prefill_idx,
        key="pp_ticker_select",
        help="Select the company you want to simulate trading.",
    )
    if st.session_state.get("paper_ticker_prefill"):
        st.session_state.paper_ticker_prefill = None

    # Reference data for selected ticker — computed once, shared by both columns
    live_price,  price_src = get_trade_price(selected_ticker, cm_df=cm_df, sample_df=df)
    company_name           = get_company_name(selected_ticker, mo_df=mo_df, sample_df=df)
    meta                   = get_signal_metadata(selected_ticker, mo_df=mo_df, sample_df=df)
    qflag                  = str(meta.get("quality_flag", ""))
    cash_avail             = float(portfolio.get("current_cash", 0))

    held_row  = open_pos[open_pos["ticker"] == selected_ticker] if not open_pos.empty else pd.DataFrame()
    held_qty  = float(held_row.iloc[0]["quantity"]) if not held_row.empty else 0.0
    held_cost = float(held_row.iloc[0]["avg_cost"])  if not held_row.empty else 0.0

    # Needs-review warning is shown prominently before controls
    if "needs_review" in qflag:
        wtext        = meta.get("warning_text", "")
        bullet_lines = "\n".join(f"- {w.strip()}" for w in wtext.split("|") if w.strip())
        st.warning(
            f"**Data quality flag: {qflag}** — {selected_ticker} is flagged for review.\n\n"
            + (bullet_lines + "\n\n" if bullet_lines else "")
            + "You can still record a simulated trade, but treat this signal with caution."
        )

    # Two-column layout: left = order controls, right = signal context
    ticket_col, context_col = st.columns([3, 2])

    # ── Right column: Signal context (read-only display) ─────────────────────
    with context_col:
        st.markdown("**Position & Signal Context**")
        st.write(f"**{company_name}**")

        if live_price is not None:
            st.metric(
                "Reference Price", f"${live_price:,.4f}",
                help="Price loaded from current_market_data.csv or sample_data.csv. "
                     "May be delayed or estimated.",
            )
            st.caption(f"Source: {price_src}")
        else:
            st.metric(
                "Reference Price", "Not available",
                help="No price found in market data or sample data. Use manual price mode.",
            )
            st.caption(f"Source: {price_src}")

        if meta["signal"]:
            sig_color = SIGNAL_COLORS.get(meta["signal"], "#555")
            st.markdown(
                f'<span style="font-weight:600;color:{sig_color};font-size:14px">'
                f'{meta["signal"]}</span>',
                unsafe_allow_html=True,
            )

        ctx_a, ctx_b = st.columns(2)
        if meta["valuation_gap"] is not None and not pd.isna(meta["valuation_gap"]):
            ctx_a.metric(
                "Valuation Gap", f"{meta['valuation_gap']:+.1f}%",
                help="Difference between model-estimated fair value and market price. "
                     "A research signal, not proof of undervaluation.",
            )
        if meta["quality_score"] is not None and not pd.isna(meta["quality_score"]):
            ctx_b.metric(
                "Quality Score", f"{meta['quality_score']:.0f}/100",
                help="A 0-100 score based on profitability, growth, and balance sheet strength.",
            )

        ctx_c, ctx_d = st.columns(2)
        if meta["report_risk"] is not None and not pd.isna(meta["report_risk"]):
            ctx_c.metric(
                "Report Risk", f"{meta['report_risk']:.0f}/100",
                help="Currently 50 (neutral) for XBRL outputs — "
                     "SEC filing text extraction has not yet been integrated.",
            )
        ctx_d.metric(
            "Data Quality", qflag or "ok",
            help="Shows whether the model output has suspicious values or needs review.",
        )

        if held_qty > 0:
            st.markdown("---")
            st.markdown("**Current Position**")
            ph1, ph2 = st.columns(2)
            ph1.metric(
                "Shares Held", f"{held_qty:.4f}",
                help="Your current simulated holding in this ticker.",
            )
            ph2.metric(
                "Avg Cost", f"${held_cost:,.2f}",
                help="Average cost per share using the average-cost accounting method.",
            )

    # ── Left column: Order controls ───────────────────────────────────────────
    with ticket_col:
        st.markdown("**Order Controls**")

        action = st.radio(
            "Action",
            [ACTION_BUY, ACTION_SELL],
            horizontal=True,
            key="pp_action",
            help="Add to paper portfolio: simulated buy.  "
                 "Reduce position: simulated sell from an existing holding.",
        )

        input_mode = st.radio(
            "Quantity input",
            ["Shares", "Dollars"],
            horizontal=True,
            key="pp_input_mode",
            help="Shares: enter exact number of shares.  "
                 "Dollars: enter a dollar amount — shares = amount ÷ price.",
        )

        if input_mode == "Shares":
            if action == ACTION_BUY and live_price and live_price > 0:
                max_qty = int(cash_avail / live_price)
            elif action == ACTION_SELL:
                max_qty = int(held_qty)
            else:
                max_qty = 0

            qb1, qb2, qb3, qb4 = st.columns(4)
            if qb1.button("1 share",          key="pp_q1"):
                st.session_state["pp_qty_preset"] = 1.0;             st.rerun()
            if qb2.button("5 shares",         key="pp_q5"):
                st.session_state["pp_qty_preset"] = 5.0;             st.rerun()
            if qb3.button("10 shares",        key="pp_q10"):
                st.session_state["pp_qty_preset"] = 10.0;            st.rerun()
            if qb4.button(f"Max ({max_qty})", key="pp_qmax"):
                st.session_state["pp_qty_preset"] = float(max_qty);  st.rerun()

            qty_input = st.number_input(
                "Number of shares",
                min_value=0.0001,
                value=float(st.session_state.get("pp_qty", 1.0)),
                step=1.0, format="%.4f",
                key="pp_qty",
                help="Number of shares to add or reduce.",
            )
            dollar_input = None

        else:  # Dollars mode
            cash_25pct = max(1.0, round(cash_avail * 0.25, 2))
            db1, db2, db3, db4 = st.columns(4)
            if db1.button("$100",                         key="pp_d100"):
                st.session_state["pp_dollar_preset"] = 100.0;        st.rerun()
            if db2.button("$500",                         key="pp_d500"):
                st.session_state["pp_dollar_preset"] = 500.0;        st.rerun()
            if db3.button("$1,000",                       key="pp_d1000"):
                st.session_state["pp_dollar_preset"] = 1000.0;       st.rerun()
            if db4.button(f"25% (${cash_25pct:,.0f})",   key="pp_d25pct"):
                st.session_state["pp_dollar_preset"] = cash_25pct;   st.rerun()

            dollar_input = st.number_input(
                "Dollar amount ($)",
                min_value=0.01,
                value=float(st.session_state.get("pp_dollars", 100.0)),
                step=10.0, format="%.2f",
                key="pp_dollars",
                help="Dollar amount to invest or reduce. Shares = amount ÷ trade price.",
            )
            qty_input = None

        st.markdown("---")

        # Price mode
        if live_price is not None:
            price_mode = st.radio(
                "Price mode",
                ["Use reference price", "Manual price"],
                horizontal=True,
                key="pp_price_mode",
                help="Reference price comes from market data.  "
                     "Manual price lets you test a hypothetical trade price.",
            )
        else:
            price_mode = "Manual price"
            st.caption("No reference price available. Enter price manually.")

        if price_mode == "Manual price":
            _manual_default = float(live_price) if live_price is not None else 0.01
            trade_price = st.number_input(
                "Price per share ($)",
                min_value=0.0001,
                value=float(st.session_state.get("pp_manual_price", _manual_default)),
                step=0.01, format="%.4f",
                key="pp_manual_price",
                help="Use this to test a hypothetical price. "
                     "Reference price is more accurate for real-world estimation.",
            )
            is_manual_price = True
        else:
            trade_price     = live_price
            is_manual_price = False
            st.caption(f"Trade price: ${live_price:,.4f}  ({price_src})")

        notes = st.text_input(
            "Notes (optional)", max_chars=200, key="pp_notes",
            help="Optional annotation stored alongside this trade record.",
        )

    # Resolve quantity for Dollars mode (needs trade_price, so computed after columns)
    if input_mode == "Dollars":
        computed_qty = (dollar_input / trade_price) if (dollar_input and trade_price > 0) else 0.0
    else:
        computed_qty = float(qty_input) if qty_input else 0.0

    trade_val  = computed_qty * trade_price
    cash_after = (cash_avail - trade_val) if action == ACTION_BUY else (cash_avail + trade_val)
    est_rpl    = (
        (trade_price - held_cost) * computed_qty
        if (action == ACTION_SELL and held_cost > 0) else 0.0
    )
    remaining  = max(0.0, held_qty - computed_qty) if action == ACTION_SELL else 0.0

    from paper_portfolio_utils import detect_manual_override
    is_override = is_manual_price and detect_manual_override(trade_price, live_price)

    # Validation
    errors_list = []
    if computed_qty <= 0:
        errors_list.append("Quantity must be greater than 0.")
    if trade_price <= 0:
        errors_list.append("Price must be greater than 0.")
    if action == ACTION_BUY and trade_val > cash_avail + 0.01:
        errors_list.append(
            f"Insufficient cash: trade value ${trade_val:,.2f} exceeds "
            f"available cash ${cash_avail:,.2f}."
        )
    if action == ACTION_SELL:
        if held_qty <= 0:
            errors_list.append(f"No open position in {selected_ticker} to reduce.")
        elif computed_qty > held_qty + 0.0001:
            errors_list.append(
                f"Cannot reduce {computed_qty:.4f} shares — only {held_qty:.4f} held."
            )

    # ── Order Preview Card ────────────────────────────────────────────────────
    st.markdown("**Order Preview**")

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Ticker",           selected_ticker)
    ov1.metric("Action",           "Add" if action == ACTION_BUY else "Reduce")
    ov2.metric(
        "Estimated Shares", f"{computed_qty:.4f}",
        help="In Dollars mode: shares = dollar amount ÷ trade price.",
    )
    ov2.metric("Trade Price",      f"${trade_price:,.4f}")
    ov3.metric("Est. Trade Value", f"${trade_val:,.2f}")
    ov3.metric("Available Cash",   f"${cash_avail:,.2f}")
    ov4.metric(
        "Cash After Trade", f"${cash_after:,.2f}",
        help="Available cash remaining after this simulated trade completes.",
    )

    if action == ACTION_SELL and held_cost > 0:
        sell_c1, sell_c2 = st.columns(2)
        sell_c1.metric(
            "Average Cost", f"${held_cost:,.4f}",
            help="Average cost per share used to calculate estimated realized P/L.",
        )
        sell_c1.metric("Shares Remaining", f"{remaining:.4f}")
        rpl_pct_str = (
            f"{(est_rpl / (held_cost * computed_qty) * 100):+.2f}%"
            if held_cost > 0 and computed_qty > 0 else ""
        )
        sell_c2.metric(
            "Est. Realized P/L", f"${est_rpl:+,.2f}",
            delta=rpl_pct_str,
            help="Estimated gain or loss from reducing this position, "
                 "using average-cost accounting. Simulated only.",
        )

    if is_override:
        st.warning(
            f"Manual price override active: trade price ${trade_price:,.4f} differs from "
            f"reference ${live_price:,.4f}. "
            "This trade will be flagged as a manual override in the transaction record."
        )

    for e in errors_list:
        st.error(e)

    if st.button(
        "Record Simulated Trade",
        type="primary",
        disabled=len(errors_list) > 0,
        key="pp_record_btn",
    ):
        updated_portfolio, updated_txns, exec_err = execute_paper_trade(
            portfolio=portfolio,
            transactions=transactions,
            ticker=selected_ticker,
            company_name=company_name,
            action=action,
            quantity=computed_qty,
            trade_price=trade_price,
            reference_price=live_price,
            price_source=price_src if live_price is not None else "manual",
            notes=notes,
            mo_df=mo_df,
            sample_df=df,
        )
        if exec_err:
            st.error(exec_err)
        else:
            verb     = "Added" if action == ACTION_BUY else "Reduced"
            ovr_note = " Price was manually overridden." if is_override else ""
            st.success(
                f"{verb} {computed_qty:.4g} shares of {selected_ticker} "
                f"at ${trade_price:,.4f}. "
                f"Cash: ${updated_portfolio['current_cash']:,.2f}.{ovr_note}"
            )
            st.rerun()

    st.divider()

    # ── Open positions table ──────────────────────────────────────────────────
    st.markdown("### Open Positions")

    if open_pos.empty:
        st.info("No open positions. Record a trade above to begin.")
    else:
        pos_df = calculate_position_metrics(
            holdings, price_map,
            mo_df=mo_df, sample_df=df,
            total_portfolio_value=total_val,
        )
        if not pos_df.empty:
            def _color_signed(val):
                try:
                    v = float(str(val).replace("$","").replace(",","").replace("+",""))
                    return "color:#27ae60;font-weight:600" if v >= 0 else "color:#c0392b;font-weight:600"
                except Exception:
                    return ""

            def _fmt_cur(v):
                try: return f"${float(v):,.2f}"
                except: return str(v)

            display_cols = [
                "Ticker", "Company", "Qty", "Avg Cost", "Price",
                "Market Value", "Cost Basis",
                "Unrealized P/L", "Unrealized P/L %",
                "Realized P/L", "Total P/L",
                "Weight %", "Signal", "Val. Gap %", "Data Quality",
            ]
            pos_display = pos_df[[c for c in display_cols if c in pos_df.columns]].copy()
            for col in ["Avg Cost", "Price", "Market Value", "Cost Basis",
                        "Unrealized P/L", "Realized P/L", "Total P/L"]:
                if col in pos_display.columns:
                    pos_display[col] = pos_display[col].apply(_fmt_cur)
            for col in ["Unrealized P/L %"]:
                if col in pos_display.columns:
                    pos_display[col] = pos_display[col].apply(
                        lambda v: f"{float(v):+.1f}%" if pd.notna(v) else ""
                    )
            try:
                styled = pos_display.style.applymap(
                    _color_signed,
                    subset=[c for c in ["Unrealized P/L", "Realized P/L", "Total P/L",
                                        "Unrealized P/L %"] if c in pos_display.columns]
                )
            except AttributeError:
                styled = pos_display.style.map(
                    _color_signed,
                    subset=[c for c in ["Unrealized P/L", "Realized P/L", "Total P/L",
                                        "Unrealized P/L %"] if c in pos_display.columns]
                )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                "Price source varies by ticker (live → sample → last trade fallback).  "
                "Unrealized P/L is estimated and for research purposes only."
            )

            # Holdings download
            holdings_csv = pos_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download holdings CSV",
                data=holdings_csv,
                file_name="paper_holdings.csv",
                mime="text/csv",
                key="pp_dl_holdings",
            )

    st.divider()

    # ── Portfolio history chart + benchmark ───────────────────────────────────
    st.markdown("### Portfolio History")

    start_date = portfolio.get("created_at", "")[:10]
    starting   = float(portfolio.get("starting_cash", DEFAULT_STARTING_CASH))

    hist_df = load_portfolio_history()
    if hist_df.empty or len(hist_df) < 2:
        hist_df = generate_portfolio_history(transactions, portfolio)

    # Determine x-axis column: prefer full timestamp when available
    ts_col_avail = (
        "timestamp" in hist_df.columns
        and hist_df["timestamp"].astype(str).str.len().max() > 10
    )
    x_col = "timestamp" if ts_col_avail else "date"
    x_label = "Date / Time" if ts_col_avail else "Date"

    bm_df, bm_msg = compare_to_benchmark(start_date, starting, benchmark="SPY")

    if not hist_df.empty and len(hist_df) >= 2:
        fig = go.Figure()

        # Cash baseline (flat line at starting_cash)
        x_vals = hist_df[x_col].astype(str)
        fig.add_trace(go.Scatter(
            x=x_vals, y=[starting] * len(hist_df),
            name="Cash baseline", mode="lines",
            line=dict(color="#aab7c4", width=1, dash="dot"),
        ))

        fig.add_trace(go.Scatter(
            x=x_vals, y=hist_df["portfolio_value"],
            name="Paper Portfolio", mode="lines+markers",
            line=dict(color="#1a5276", width=2),
            marker=dict(size=5),
        ))

        if bm_df is not None:
            fig.add_trace(go.Scatter(
                x=bm_df["Date"].astype(str), y=bm_df["benchmark_value"],
                name="SPY (normalized)", mode="lines",
                line=dict(color="#e67e22", width=1.5, dash="dash"),
            ))

        fig.update_layout(
            xaxis_title=x_label, yaxis_title="Portfolio Value ($)",
            height=380, margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        bm_caption = f"Benchmark: {bm_msg}" if bm_df is not None else f"SPY benchmark: {bm_msg}"
        st.caption(
            bm_caption + "  |  "
            "Benchmark comparison is context, not proof of future performance.  "
            + ("History estimated from trade timestamps — intermediate values are conservative." if
               "estimated" in hist_df.columns and hist_df["estimated"].any() else "")
        )
    else:
        st.info("Portfolio history chart will appear after at least 2 snapshots are recorded.")

    st.divider()

    # ── Transaction history ───────────────────────────────────────────────────
    st.markdown("### Transaction History")

    if transactions.empty:
        st.info("No transactions recorded yet.")
    else:
        tx_display = transactions.copy()

        # ── Filters ───────────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        unique_tickers  = ["All"] + sorted(tx_display["ticker"].dropna().unique().tolist())
        unique_actions  = ["All"] + sorted(tx_display["action"].dropna().unique().tolist())
        unique_sources  = ["All"] + sorted(
            tx_display["price_source"].dropna().replace("", "manual").unique().tolist()
        )
        flt_ticker = f1.selectbox("Filter by ticker", unique_tickers, key="pp_flt_ticker")
        flt_action = f2.selectbox("Filter by action", unique_actions, key="pp_flt_action")
        flt_source = f3.selectbox("Filter by price source", unique_sources, key="pp_flt_source")

        if flt_ticker != "All":
            tx_display = tx_display[tx_display["ticker"] == flt_ticker]
        if flt_action != "All":
            tx_display = tx_display[tx_display["action"] == flt_action]
        if flt_source != "All":
            src_col = "price_source"
            tx_display = tx_display[
                tx_display[src_col].fillna("").replace("", "manual") == flt_source
            ]

        # ── Format for display ─────────────────────────────────────────────────
        show_cols = [
            "timestamp", "ticker", "company_name", "action",
            "quantity", "reference_price", "trade_price", "manual_override_flag",
            "trade_value", "cash_after_trade",
            "realized_pl", "realized_pl_pct",
            "signal_at_trade", "output_quality_flag_at_trade", "notes",
        ]
        tx_show = tx_display[[c for c in show_cols if c in tx_display.columns]].copy()
        tx_show["timestamp"] = tx_show["timestamp"].astype(str).str[:19]
        for col in ["reference_price", "trade_price", "trade_value", "cash_after_trade"]:
            if col in tx_show.columns:
                tx_show[col] = tx_show[col].apply(
                    lambda v: f"${float(v):,.4f}" if (v != "" and pd.notna(v)) else ""
                )
        for col in ["realized_pl"]:
            if col in tx_show.columns:
                tx_show[col] = tx_show[col].apply(
                    lambda v: f"${float(v):+,.2f}" if (v != "" and pd.notna(v)) else ""
                )
        if "realized_pl_pct" in tx_show.columns:
            tx_show["realized_pl_pct"] = tx_show["realized_pl_pct"].apply(
                lambda v: f"{float(v):+.2f}%" if (v != "" and pd.notna(v)) else ""
            )
        if "quantity" in tx_show.columns:
            tx_show["quantity"] = tx_show["quantity"].apply(
                lambda v: f"{float(v):.4g}" if pd.notna(v) else ""
            )

        st.dataframe(
            tx_show[::-1].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Showing {len(tx_show)} of {len(transactions)} transactions. "
            "manual_override_flag = True means trade price differed from reference by >$0.01 or >0.1%."
        )

        # ── Downloads ─────────────────────────────────────────────────────────
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "Download transactions CSV",
            data=transactions.to_csv(index=False).encode("utf-8"),
            file_name="paper_transactions.csv",
            mime="text/csv",
            key="pp_dl_txns",
        )
        if not open_pos.empty:
            dl2.download_button(
                "Download holdings CSV",
                data=open_pos.to_csv(index=False).encode("utf-8"),
                file_name="paper_holdings.csv",
                mime="text/csv",
                key="pp_dl_holdings2",
            )
        # Portfolio summary CSV
        summary_df = pd.DataFrame([{
            "field": k, "value": v
        } for k, v in metrics.items()])
        dl3.download_button(
            "Download portfolio summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="paper_portfolio_summary.csv",
            mime="text/csv",
            key="pp_dl_summary",
        )

    st.divider()

    # ── Portfolio report export ───────────────────────────────────────────────
    st.markdown("### Export Portfolio Report")

    report_text   = generate_portfolio_report(
        portfolio=portfolio,
        holdings=holdings,
        metrics=metrics,
        price_map=price_map,
        bm_msg=bm_msg,
        mo_df=mo_df,
        sample_df=df,
    )
    st.download_button(
        "Download portfolio report (markdown)",
        data=report_text.encode("utf-8"),
        file_name="paper_portfolio_report.md",
        mime="text/markdown",
        key="pp_dl_report",
    )
    with st.expander("Preview report"):
        st.text(report_text)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ════════════════════════════════════════════════════════════════════════════

def _render_ticker_universe_tab() -> None:
    """Ticker Universe management panel — shown inside About → Ticker Universe tab."""
    st.markdown("### Ticker Universe")
    st.caption(
        "The ticker universe controls which companies *can* be tracked by the app. "
        "Only tickers with completed model outputs appear in the Screener. "
        "Tickers with data_status='pending' are tracked but not yet ranked."
    )

    uni_df, uni_msg = load_ticker_universe()
    st.caption(uni_msg)

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_total  = len(uni_df) if not uni_df.empty else 0
    n_active = int((uni_df["active"].astype(str).str.lower() == "true").sum()) if not uni_df.empty else 0
    n_avail  = int((uni_df["data_status"] == "available").sum()) if not uni_df.empty else 0
    n_pend   = int((uni_df["data_status"] == "pending").sum()) if not uni_df.empty else 0
    _u_groups = uni_df["universe_group"].value_counts().to_dict() if not uni_df.empty else {}

    _uc1, _uc2, _uc3, _uc4, _uc5 = st.columns(5)
    _uc1.metric("Total tickers",      n_total)
    _uc2.metric("Active",             n_active)
    _uc3.metric("Data available",     n_avail)
    _uc4.metric("Pending (no data)",  n_pend)
    _uc5.metric("Universe groups",    len(_u_groups))

    if _u_groups:
        st.markdown("**Universe groups:**  " +
                    "  |  ".join(f"**{g}**: {c}" for g, c in sorted(_u_groups.items())))

    if uni_df.empty:
        st.info("No tickers in universe yet. Run `python3 build_expanded_universe.py --limit 10` to start.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: Expanded Universe Build Report
    # ─────────────────────────────────────────────────────────────────────────
    _build_report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "expanded_universe_build_report.csv")
    if os.path.exists(_build_report_path):
        st.divider()
        with st.expander("Expanded universe build report", expanded=False):
            try:
                _brdf = pd.read_csv(_build_report_path)
                _br_avail = int((_brdf["final_data_status"] == "available").sum())
                _br_pend  = int((_brdf["final_data_status"] == "pending").sum())
                _br_fail  = int((_brdf["final_data_status"] == "failed").sum())
                st.markdown(
                    f"**{_br_avail} available** · {_br_pend} pending · {_br_fail} failed"
                    f" (of {len(_brdf)} tickers processed)"
                )
                _br_show = _brdf[[c for c in [
                    "ticker", "company_name", "final_data_status",
                    "fundamentals_status", "market_cap_status", "model_output_status",
                    "filing_10k_status", "failure_reason",
                ] if c in _brdf.columns]]
                st.dataframe(_br_show.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
                st.download_button(
                    "Download build report",
                    data=_brdf.to_csv(index=False).encode("utf-8"),
                    file_name="expanded_universe_build_report.csv",
                    mime="text/csv",
                    key="uni_dl_build_report",
                )
            except Exception as _e:
                st.warning(f"Could not load build report: {_e}")
    else:
        st.caption(
            "No build report found. Run `python3 build_expanded_universe.py --limit 10` "
            "to start building the expanded universe, then `--resume` to continue."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Advanced — Add custom tickers (collapsed)
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Advanced: Add custom tickers", expanded=False):
        st.caption(
            "Normal users do not need this. The expanded universe is pre-built offline. "
            "Use this only to track a custom ticker not in the seed list."
        )
        st.markdown("##### Ticker Setup Wizard")

        _wizard_raw = st.text_area(
            "Enter tickers to add (comma, space, or newline-separated)",
            placeholder="e.g.  NVDA, AMD\nINTC\nTSM",
            height=80,
            key="uni_wizard_input",
        )

        if st.button("Validate tickers", key="uni_wizard_validate_btn"):
            if not _wizard_raw.strip():
                st.warning("Enter at least one ticker symbol.")
            else:
                _results = validate_tickers_for_wizard(_wizard_raw)
                st.session_state["_wizard_validation_results"] = _results

        _val_results = st.session_state.get("_wizard_validation_results", [])
        if _val_results:
            _vdf = pd.DataFrame(_val_results)[
                ["ticker", "company_name", "cik", "already_in_universe",
                 "validation_status", "warning"]
            ]
            st.dataframe(_vdf.reset_index(drop=True), use_container_width=True, hide_index=True)

            _addable = [r for r in _val_results
                        if r["valid"] and not r["already_in_universe"]]
            if _addable:
                st.caption(f"{len(_addable)} ticker(s) ready to add.")
                if st.button("Add valid tickers to universe", key="uni_wizard_add_btn",
                             type="primary"):
                    _to_add = [r["ticker"] for r in _addable]
                    _added, _skipped, _msg = expand_universe_for_tickers(_to_add)
                    if _added > 0:
                        st.success(_msg)
                    else:
                        st.info(_msg)
                    st.session_state.pop("_wizard_validation_results", None)
                    st.rerun()
            else:
                st.info("No new valid tickers to add (all invalid or already in universe).")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Data Readiness Dashboard (custom/pending only)
    # ─────────────────────────────────────────────────────────────────────────
    _work_df = uni_df[
        uni_df["data_status"].isin(["pending", "failed"]) |
        (uni_df["universe_group"] == "custom")
    ] if not uni_df.empty else pd.DataFrame()

    if not _work_df.empty:
        st.divider()
        with st.expander(
            f"Custom ticker readiness ({len(_work_df)} pending/custom tickers)",
            expanded=True,
        ):
            st.caption(
                "These tickers were added via the Advanced wizard and do not yet have model outputs. "
                "Use the pipeline runner below to generate data, or run the batch script offline."
            )
            _dash_rows = []
            for _, _u_row in _work_df.iterrows():
                _t = str(_u_row["ticker"]).upper()
                _r = check_ticker_readiness(_t)
                def _badge(ok: bool) -> str:
                    return "✅" if ok else "❌"
                _label = "Ready" if _r["ready_for_screener"] else (
                    "Missing model output" if not _r["has_fundamentals"] else
                    "Missing market cap"   if not _r["has_market_cap"] else
                    "Pending pipeline"
                )
                _dash_rows.append({
                    "Ticker":        _t,
                    "Company":       str(_u_row.get("company_name", "")).strip() or "—",
                    "Ready":         _label,
                    "Fundamentals":  _badge(_r["has_fundamentals"]),
                    "Market cap":    _badge(_r["has_market_cap"]),
                    "Model output":  _badge(_r["has_model_output"]),
                    "10-K risk":     _badge(_r["has_10k_risk"]),
                    "10-Q risk":     _badge(_r["has_10q_risk"]),
                    "Current mkt":   _badge(_r["has_current_market"]),
                    "Warnings":      "; ".join(_r["warnings"]) or "",
                })
            st.dataframe(
                pd.DataFrame(_dash_rows).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: Controlled Pipeline Runner (custom/pending tickers only)
    # ─────────────────────────────────────────────────────────────────────────
    _pipeline_candidates = uni_df[
        uni_df["data_status"].isin(["pending", "failed"])
    ] if not uni_df.empty else pd.DataFrame()

    if not _pipeline_candidates.empty:
        st.divider()
        st.markdown("#### Run pipeline for custom tickers")
        st.caption(
            "For custom tickers added via the wizard above. "
            "The expanded 100-company universe is built offline — normal users do not need this. "
            "Each step runs individually; failures do not affect other tickers."
        )

        _pipe_ticker_opts = sorted(_pipeline_candidates["ticker"].str.upper().unique().tolist())
        _sel_tickers = st.multiselect(
            "Select tickers to process",
            options=_pipe_ticker_opts,
            default=[],
            key="uni_pipe_tickers",
        )

        st.markdown("**Pipeline steps to run:**")
        _pc1, _pc2, _pc3 = st.columns(3)
        _step_facts  = _pc1.checkbox("Fetch SEC company facts",          value=True,  key="uni_step_facts")
        _step_mc     = _pc1.checkbox("Fetch market cap / current price", value=True,  key="uni_step_mc")
        _step_model  = _pc2.checkbox("Generate model output",            value=True,  key="uni_step_model")
        _step_10k    = _pc2.checkbox("Fetch annual 10-K filing risk",    value=False, key="uni_step_10k")
        _step_10q    = _pc3.checkbox("Fetch latest 10-Q risk",           value=False, key="uni_step_10q")
        _step_currmkt = _pc3.checkbox("Refresh current market snapshot", value=True,  key="uni_step_currmkt")

        _any_step = any([_step_facts, _step_mc, _step_model,
                         _step_10k, _step_10q, _step_currmkt])

        if st.button(
            "Run selected pipeline steps",
            key="uni_pipe_run_btn",
            type="primary",
            disabled=(not _sel_tickers or not _any_step),
        ):
            if not _sel_tickers:
                st.warning("Select at least one ticker.")
            elif not _any_step:
                st.warning("Select at least one pipeline step.")
            else:
                # Build CIK map for selected tickers
                _cik_map: dict = {}
                for _, _u_row in _pipeline_candidates.iterrows():
                    _t2 = str(_u_row["ticker"]).upper()
                    _cik_map[_t2] = str(_u_row.get("cik", "")).strip()

                _total_ok   = 0
                _total_fail = 0

                for _t in _sel_tickers:
                    _cik = _cik_map.get(_t, "")
                    st.markdown(f"**{_t}**")
                    _msg_lines = []

                    if _step_facts:
                        with st.spinner(f"{_t}: fetching SEC company facts…"):
                            _ok, _m = prepare_single_ticker_fundamentals(_t, _cik)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    if _step_mc:
                        with st.spinner(f"{_t}: fetching market cap data…"):
                            _ok, _m = prepare_single_ticker_market_cap(_t, _cik)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    if _step_model:
                        with st.spinner(f"{_t}: generating model output (rebuilds all tickers)…"):
                            _ok, _m = prepare_single_ticker_model_output(_t)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    if _step_10k:
                        with st.spinner(f"{_t}: fetching 10-K filing risk…"):
                            _ok, _m = prepare_single_ticker_10k_risk(_t, _cik)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    if _step_10q:
                        with st.spinner(f"{_t}: fetching latest 10-Q risk…"):
                            _ok, _m = prepare_single_ticker_10q_risk(_t, _cik)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    if _step_currmkt:
                        with st.spinner(f"{_t}: refreshing market snapshot…"):
                            _ok, _m = prepare_single_ticker_current_market(_t)
                        (_msg_lines.append(f"✅ {_m}") if _ok else _msg_lines.append(f"❌ {_m}"))
                        if _ok:
                            _total_ok += 1
                        else:
                            _total_fail += 1

                    # Promote if now ready
                    _promoted, _promo_msg = promote_ticker_if_ready(_t)
                    if _promoted:
                        _msg_lines.append(f"🟢 {_promo_msg}")

                    for _line in _msg_lines:
                        st.markdown(_line)

                st.markdown("---")
                if _total_fail == 0:
                    st.success(f"Pipeline complete: {_total_ok} steps succeeded.")
                else:
                    st.warning(
                        f"Pipeline done: {_total_ok} succeeded, {_total_fail} failed. "
                        "See details above."
                    )
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: Browse full universe
    # ─────────────────────────────────────────────────────────────────────────
    if not uni_df.empty:
        st.divider()
        with st.expander("Browse full universe", expanded=False):
            _filter_grp = st.selectbox(
                "Filter by group",
                ["All"] + sorted(_u_groups.keys()),
                key="uni_filter_grp",
            )
            _filter_status = st.selectbox(
                "Filter by data status",
                ["All", "available", "pending", "failed"],
                key="uni_filter_status",
            )
            _view = uni_df.copy()
            if _filter_grp != "All":
                _view = _view[_view["universe_group"] == _filter_grp]
            if _filter_status != "All":
                _view = _view[_view["data_status"] == _filter_status]
            # Show core columns + pipeline status columns
            _show_cols = [c for c in [
                "ticker", "company_name", "sector", "universe_group",
                "data_status", "model_output_status", "company_facts_status",
                "filing_10k_status", "filing_10q_status", "last_pipeline_run",
                "failure_reason",
            ] if c in _view.columns]
            st.dataframe(
                _view[_show_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        # ── Download ─────────────────────────────────────────────────────────
        st.download_button(
            "Download ticker_universe.csv",
            data=uni_df.to_csv(index=False).encode("utf-8"),
            file_name="ticker_universe.csv",
            mime="text/csv",
            key="uni_dl_btn",
        )
        st.caption(
            "You can also edit ticker_universe.csv manually. "
            "Set data_status='available' only after model outputs exist for that ticker."
        )


def page_about() -> None:
    st.markdown("""
    <div class="about-hero">
      <div style="font-size:11px;color:#93c5fd;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">ABOUT UNDERDAWG</div>
      <div style="font-size:22px;font-weight:700;color:white;margin-bottom:6px">Research tool for self-directed investors</div>
      <div style="font-size:13px;color:#c7d8f5">MSDSBA Practicum · Streamlit + scikit-learn + SEC EDGAR · Not investment advice</div>
    </div>
    """, unsafe_allow_html=True)

    # Ticker Universe tab hidden temporarily until custom ticker setup is production-ready.
    tab_ov, tab_how, tab_meth, tab_bm, tab_diag = st.tabs(
        ["Overview", "How It Works", "Methodology", "Benchmark Comparison",
         "Model Diagnostics"]
    )

    with tab_ov:
        st.markdown("#### What Underdawg does")
        st.markdown(
            "Underdawg helps self-directed investors screen companies, understand why each stock is flagged, "
            "and test research ideas without real money. It is built for research support — not trading."
        )
        all_mo, _ = _load_all_model_outputs()
        if all_mo is not None:
            n_companies  = int(all_mo["ticker"].nunique())
            n_rows       = len(all_mo)
            yrs          = sorted(all_mo["year"].dropna().astype(int).unique())
            year_range   = f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1 else str(yrs[0])
            n_replay_yrs = len(yrs)
            nr_count     = int(all_mo["output_quality_flag"].str.contains("needs_review", na=False).sum()) \
                           if "output_quality_flag" in all_mo.columns else 0
        else:
            n_companies = 29; n_rows = 361; year_range = "2010–2024"; n_replay_yrs = 15; nr_count = 0

        st.markdown("#### By the Numbers")
        stat_cols = st.columns(5)
        stat_cols[0].metric("Companies", n_companies)
        stat_cols[1].metric("Company-year rows", n_rows)
        stat_cols[2].metric("Years covered", year_range)
        stat_cols[3].metric("Flagged rows", nr_count)
        stat_cols[4].metric("Replay-ready years", n_replay_yrs)

        st.divider()
        st.markdown("#### What you can do")
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Research shortlist**")
            st.markdown("Screen and rank companies by valuation gap, quality score, and filing risk. Export as CSV.")
        with wc2:
            st.markdown("**Paper portfolio**")
            st.markdown("Execute simulated buys and sells, track P/L, compare to SPY or QQQ.")
        with wc3:
            st.markdown("**Historical replay**")
            st.markdown("Pick an exact start date and holding period to see how a model shortlist would have performed.")

        st.divider()
        st.markdown("""
        <div class="disclaimer-box">
        <strong>Disclaimer</strong><br>
        Underdawg is for <strong>educational and research support only</strong>. It does
        <strong>not</strong> provide investment advice, buy/sell recommendations, or performance guarantees.
        Signal labels indicate where to focus research time, not where to invest capital.
        The valuation gap is a research signal, not proof of undervaluation.
        Past simulated performance does not predict future real-world results.
        </div>
        """, unsafe_allow_html=True)

    with tab_how:
        st.markdown("#### Four-stage pipeline")
        step_cols = st.columns(4)
        _STEPS_H = [
            ("1", "Data Collection",
             "XBRL + SEC EDGAR fundamentals (2010–2024). Legacy XBRL + modern Company Facts API."),
            ("2", "Scoring",
             "Ridge Regression predicts log(market_cap). Quality from ROE, margins, growth, debt. Filing risk from 10-K text."),
            ("3", "Signal Assignment",
             "Valuation gap + quality + filing risk → research signal (5 categories)."),
            ("4", "Calibration",
             "Signals re-ranked within year/era to correct for structural valuation shifts."),
        ]
        for col, (num, label, desc) in zip(step_cols, _STEPS_H):
            with col:
                st.markdown(
                    f'<div class="product-card">'
                    f'<div style="font-size:20px;font-weight:800;color:#2563eb">{num}.</div>'
                    f'<div style="font-size:13px;font-weight:700;color:#1a2744;margin:4px 0">{label}</div>'
                    f'<div style="font-size:12px;color:#64748b;line-height:1.5">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.divider()
        st.markdown("#### Navigate to a feature")
        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Open Screener", use_container_width=True, key="about_nav_screener"):
                st.session_state.pending_nav_page = "Screener"
                st.rerun()
        with nav2:
            if st.button("Company Detail", use_container_width=True, key="about_nav_detail"):
                st.session_state.pending_nav_page = "Company Detail"
                st.rerun()
        with nav3:
            if st.button("Portfolio Simulator", use_container_width=True, key="about_nav_sim"):
                st.session_state.pending_nav_page = "Portfolio Simulator"
                st.rerun()

    with tab_meth:
        page_methodology()

    with tab_bm:
        st.markdown("""
        <div class="disclaimer-strip">
        Simulated demo — not a historical backtest — not a performance claim. USD 10,000 starting capital · 1-year timeline.
        </div>
        """, unsafe_allow_html=True)
        page_benchmark_comparison(df)
        st.divider()
        st.markdown("#### Run a real Historical Replay")
        st.markdown("For model-backed comparisons using actual daily price data, use **Portfolio Simulator → Historical Replay**.")
        if st.button("Open Historical Replay", key="about_bm_goto_replay"):
            st.session_state.pending_nav_page = "Portfolio Simulator"
            st.rerun()

    with tab_diag:
        page_model_diagnostics()

    # tab_uni rendering removed — _render_ticker_universe_tab() kept for later use


# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════

if   page == "Home":
    page_home()
elif page == "Screener":
    page_screener(df)
elif page == "Company Detail":
    page_company_detail(df)
elif page == "Portfolio Simulator":
    page_portfolio_simulator(df)
elif page == "About":
    page_about()
