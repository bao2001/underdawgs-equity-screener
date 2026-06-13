"""
app.py -- Equity Research Screener (MVP)
Streamlit single-page application with sidebar navigation.

Run:  python3 -m streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_utils  import get_screener_data
from model_utils import (
    SIGNAL_COLORS, SIGNAL_BG_COLORS, SIGNAL_ORDER,
    get_signal_color, get_signal_explanation,
    COMPANY_DESCRIPTIONS, REPORT_SIGNALS,
    generate_benchmark_data, portfolio_metrics,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Research Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a2744; }
[data-testid="stSidebar"] * { color: #e8edf5 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

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
    font-size: 18px;
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
    padding: 36px 40px;
    margin-bottom: 28px;
}
.hero-banner h1 { color: white; margin: 0 0 8px 0; font-size: 28px; }
.hero-banner p  { color: #c7d8f5; margin: 0; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Page names (no emojis) ────────────────────────────────────────────────────

PAGES = ["Home", "Screener", "Company Detail", "Methodology", "Benchmark Evaluation"]

# ── Session state ─────────────────────────────────────────────────────────────

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

# ── Data (cached) ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(use_live: bool) -> tuple:
    return get_screener_data(use_live=use_live)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Equity Screener")
    st.caption("MSDSBA Practicum MVP")
    st.divider()

    page = st.radio(
        "Navigate to",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.divider()
    use_live = st.toggle(
        "Refresh live prices",
        value=False,
        help="Attempts to fetch current prices via yfinance.",
    )

    df, data_status = load_data(use_live)

    st.caption(data_status)
    st.divider()
    st.caption(
        "Disclaimer: This tool is for educational and research "
        "support only. It does not constitute investment advice."
    )

# ── Styling helpers ───────────────────────────────────────────────────────────

def _safe_map(styler, fn, col_name: str):
    """
    Apply element-wise style function only when the column actually exists.
    Uses Styler.map() (pandas >= 2.1) with fallback to applymap() for older versions.
    """
    if col_name not in styler.data.columns:
        return styler
    try:
        return styler.map(fn, subset=[col_name])       # pandas >= 2.1
    except AttributeError:
        return styler.applymap(fn, subset=[col_name])  # pandas < 2.1


def _safe_apply(styler, fn):
    """Apply column-wise style function with a fallback to unstyled on error."""
    try:
        return styler.apply(fn)
    except Exception:
        return styler


def _safe_format(styler, fmt_dict: dict):
    """Format only the columns that actually exist in the dataframe."""
    existing = {k: v for k, v in fmt_dict.items() if k in styler.data.columns}
    if not existing:
        return styler
    try:
        return styler.format(existing)
    except Exception:
        return styler


def signal_badge_html(signal: str) -> str:
    bg = SIGNAL_BG_COLORS.get(signal, "#f0f0f0")
    fg = SIGNAL_COLORS.get(signal, "#333")
    return (
        f'<span class="signal-badge" '
        f'style="background:{bg}; color:{fg}; border: 1px solid {fg};">'
        f'{signal}</span>'
    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 -- HOME
# ════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    st.markdown("""
    <div class="hero-banner">
      <h1>Equity Research Screener</h1>
      <p>A Time-Saving First-Pass Research Tool for Self-Directed Investors</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">The Problem</div>', unsafe_allow_html=True)
        st.markdown("""
Self-directed and beginner investors face a real research bottleneck:

- **15-20 minutes** per company for a quick valuation check (P/E, P/B, FCF, debt)
- **45+ minutes** for a deeper pass that includes reading 10-K/10-Q filings
- **50-100 companies** to screen across a watchlist = multiple days of manual work

Investors must manually cross-reference valuation ratios, revenue trends, earnings
quality, debt levels, free cash flow, sector norms -- and still might miss a buried
risk disclosure in a quarterly filing.
        """)

        st.markdown('<div class="section-header">Our Solution</div>', unsafe_allow_html=True)
        st.markdown("""
This screener automates the first-pass research process so investors can focus
their limited time on the companies most deserving of deeper analysis:

1. **Aggregates** key financial metrics across a stock universe in one view
2. **Estimates** fair value using a quantitative model (P/E, FCF, growth-adjusted)
3. **Scans** 10-K / 10-Q filing text for risk signals (debt mentions, legal language, cost pressures)
4. **Ranks** companies by combined valuation gap, quality, and filing-risk score

**Time saved: from hours of manual work to a ranked shortlist in seconds.**
        """)

    with col_right:
        st.markdown('<div class="section-header">By the Numbers</div>', unsafe_allow_html=True)
        st.metric("Companies Analyzed", "10", help="Sample universe for MVP demo")
        st.metric("Manual Research Time", "15-45 min / company")
        st.metric("Screener Shortlist Time", "< 1 minute")
        st.metric("Avg. Time Savings", "~93%",
                  help="Estimated time reduction for initial screening pass")

    st.divider()

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    steps = st.columns(5)
    step_data = [
        ("1", "Collect Data",    "Financial ratios, prices, revenue, FCF from structured sources"),
        ("2", "Scan Filings",    "10-K/10-Q text scanned for risk keywords, debt, legal, cost signals"),
        ("3", "Estimate Value",  "Earnings-based fair value estimate per company"),
        ("4", "Score and Rank",  "Quality score + report risk score → final research signal"),
        ("5", "Output Shortlist","Ranked table helps you decide where to spend research time"),
    ]
    for col, (num, title, desc) in zip(steps, step_data):
        with col:
            st.markdown(f"""
            <div class="mini-card" style="height:160px">
              <div style="font-size:22px;font-weight:700;color:#2563eb">{num}</div>
              <div style="font-weight:700;font-size:13px;margin:6px 0 4px">{title}</div>
              <div style="font-size:11px;color:#64748b;line-height:1.4">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="disclaimer-box">
    <strong>Important Disclaimer</strong><br>
    This tool is designed for <strong>educational and research support purposes only</strong>.
    It does <strong>not</strong> constitute investment advice, financial recommendations, or
    solicitation to buy or sell any securities. Signal labels (e.g., "High-priority research
    candidate") indicate where the model suggests investing <em>research time</em>, not capital.
    Past model signals do not guarantee future performance.
    Always conduct your own due diligence and consult a qualified financial advisor before
    making investment decisions.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 -- SCREENER
# ════════════════════════════════════════════════════════════════════════════

def page_screener(df: pd.DataFrame) -> None:
    st.title("Company Research Screener")
    st.caption(
        "Companies ranked by research priority based on valuation gap, quality, and "
        "filing-risk scores. Select a company and click View Details or navigate to "
        "the Company Detail page."
    )

    # ── Summary bar ───────────────────────────────────────────────────────────
    counts = {s: len(df[df["Final_Signal"] == s]) for s in SIGNAL_ORDER}
    summary_labels = [
        "High-Priority", "Research Candidate", "Fairly Valued",
        "Pot. Overvalued", "Value Trap",
    ]
    cols = st.columns(5)
    for col, sig, lbl in zip(cols, SIGNAL_ORDER, summary_labels):
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

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_sigs = st.multiselect("Signal", SIGNAL_ORDER, default=SIGNAL_ORDER, key="flt_sig")
        with f2:
            sectors = sorted(df["Sector"].unique())
            sel_sec = st.multiselect("Sector", sectors, default=sectors, key="flt_sec")
        with f3:
            min_q = int(df["Quality_Score"].min())
            max_q = int(df["Quality_Score"].max())
            q_range = st.slider("Min Quality Score", min_q, max_q, min_q, key="flt_q")

    fdf = df[
        df["Final_Signal"].isin(sel_sigs) &
        df["Sector"].isin(sel_sec) &
        (df["Quality_Score"] >= q_range)
    ].copy()

    pri_map = {s: i for i, s in enumerate(SIGNAL_ORDER)}
    fdf["_pri"] = fdf["Final_Signal"].map(pri_map)
    fdf = fdf.sort_values("_pri").drop(columns="_pri").reset_index(drop=True)

    # ── Build display dataframe ───────────────────────────────────────────────
    src_cols = [
        "Ticker", "Company_Name", "Sector",
        "Current_Price", "Estimated_Fair_Value", "Valuation_Gap_Pct",
        "Quality_Score", "Report_Risk_Score", "Final_Signal",
    ]
    # Only keep source columns that actually exist (guards against CSV schema drift)
    src_cols = [c for c in src_cols if c in fdf.columns]
    display = fdf[src_cols].copy()

    col_rename = {
        "Ticker":               "Ticker",
        "Company_Name":         "Company",
        "Sector":               "Sector",
        "Current_Price":        "Market Price ($)",
        "Estimated_Fair_Value": "Est. Fair Value ($)",
        "Valuation_Gap_Pct":    "Valuation Gap (%)",
        "Quality_Score":        "Quality Score",
        "Report_Risk_Score":    "Report Risk Score",
        "Final_Signal":         "Signal",
    }
    display = display.rename(columns={k: v for k, v in col_rename.items() if k in display.columns})

    if "Market Price ($)" in display.columns:
        display["Market Price ($)"] = display["Market Price ($)"].map("${:.2f}".format)
    if "Est. Fair Value ($)" in display.columns:
        display["Est. Fair Value ($)"] = display["Est. Fair Value ($)"].map("${:.2f}".format)
    if "Valuation Gap (%)" in display.columns:
        display["Valuation Gap (%)"] = display["Valuation Gap (%)"].map("{:+.1f}%".format)

    # ── Style functions ───────────────────────────────────────────────────────
    def _style_signal(val):
        bg = SIGNAL_BG_COLORS.get(val, "#fff")
        fg = SIGNAL_COLORS.get(val, "#000")
        return f"background-color:{bg}; color:{fg}; font-weight:600"

    def _style_gap(val):
        try:
            raw = float(str(val).replace("%", "").replace("+", ""))
        except (ValueError, TypeError):
            return ""
        if raw > 10:
            return "color:#27ae60; font-weight:600"
        if raw < -10:
            return "color:#e74c3c; font-weight:600"
        return "color:#f39c12; font-weight:600"

    # ── Apply styling defensively ─────────────────────────────────────────────
    try:
        styler = display.style
        styler = _safe_map(styler, _style_signal, "Signal")
        styler = _safe_map(styler, _style_gap,    "Valuation Gap (%)")
        styler = styler.set_properties(**{"font-size": "13px"})
        st.dataframe(styler, use_container_width=True, hide_index=True, height=400)
    except Exception as exc:
        st.warning(f"Table styling unavailable ({exc}). Showing plain table.")
        st.dataframe(display, use_container_width=True, hide_index=True, height=400)

    st.caption(
        "Valuation Gap = (Estimated Fair Value - Market Price) / Market Price x 100.  "
        "Green = potential upside  |  Red = potential overvaluation  |  "
        "Signal incorporates quality and filing-risk filters."
    )

    st.divider()

    # ── Select and go to detail ───────────────────────────────────────────────
    c1, c2, _ = st.columns([3, 1, 3])
    with c1:
        ticker_opts = fdf["Ticker"].tolist()
        default_idx = (
            ticker_opts.index(st.session_state.selected_ticker)
            if st.session_state.selected_ticker in ticker_opts else 0
        )
        chosen = st.selectbox(
            "Select a company to inspect:",
            ticker_opts,
            index=default_idx,
            format_func=lambda t: f"{t} -- {df[df['Ticker']==t]['Company_Name'].values[0]}",
            key="screener_select",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("View Details", type="primary", use_container_width=True):
            st.session_state.selected_ticker = chosen
            st.session_state.nav_page = "Company Detail"
            st.rerun()

    # ── Quick inline snapshot ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Quick Company Snapshot</div>', unsafe_allow_html=True)
    row = df[df["Ticker"] == chosen].iloc[0]

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Current Price",     f"${row.Current_Price:.2f}")
    q2.metric("Est. Fair Value",   f"${row.Estimated_Fair_Value:.2f}")
    q3.metric("Valuation Gap",     f"{row.Valuation_Gap_Pct:+.1f}%")
    q4.metric("Quality Score",     f"{row.Quality_Score}/100")
    q5.metric("Report Risk Score", f"{row.Report_Risk_Score}/100")

    st.markdown(signal_badge_html(row.Final_Signal), unsafe_allow_html=True)

    # ── Signal distribution chart ─────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Signal Distribution</div>', unsafe_allow_html=True)

    sig_counts = df["Final_Signal"].value_counts().reindex(SIGNAL_ORDER, fill_value=0)
    bar_colors = [SIGNAL_COLORS[s] for s in sig_counts.index]
    fig = go.Figure(go.Bar(
        x=sig_counts.index,
        y=sig_counts.values,
        marker_color=bar_colors,
        text=sig_counts.values,
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Signal", yaxis_title="Companies",
        height=280, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=80),
    )
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 -- COMPANY DETAIL
# ════════════════════════════════════════════════════════════════════════════

def page_company_detail(df: pd.DataFrame) -> None:
    st.title("Company Detail")

    all_tickers = df["Ticker"].tolist()
    default_idx = (
        all_tickers.index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in all_tickers else 0
    )

    chosen = st.selectbox(
        "Select company:",
        all_tickers,
        index=default_idx,
        format_func=lambda t: f"{t} -- {df[df['Ticker']==t]['Company_Name'].values[0]}",
        key="detail_select",
    )
    st.session_state.selected_ticker = chosen
    row = df[df["Ticker"] == chosen].iloc[0]

    # ── Company header ────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"## {row.Company_Name}")
        st.markdown(f"**{chosen}** · {row.Sector}")
        st.markdown(COMPANY_DESCRIPTIONS.get(chosen, "No description available."))
    with h2:
        sig_color = SIGNAL_COLORS.get(row.Final_Signal, "#555")
        sig_bg    = SIGNAL_BG_COLORS.get(row.Final_Signal, "#f0f0f0")
        st.markdown(
            f"""
            <div style="border:2px solid {sig_color}; border-radius:12px;
                        background:{sig_bg}; padding:16px; text-align:center; margin-top:12px">
              <div style="font-size:13px;color:#555;margin-bottom:4px">Research Signal</div>
              <div style="font-size:15px;font-weight:700;color:{sig_color}">
                {row.Final_Signal}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Market Price",  f"${row.Current_Price:.2f}")
    c2.metric("Model Est. Fair Value", f"${row.Estimated_Fair_Value:.2f}")
    c3.metric("Valuation Gap",         f"{row.Valuation_Gap_Pct:+.1f}%",
              help="(Est. Fair Value - Market Price) / Market Price x 100")
    c4.metric("Market Cap",            f"${row.Market_Cap_B:.0f}B")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Quality Score",        f"{row.Quality_Score}/100",
              help="Higher = stronger fundamentals")
    c6.metric("Report Risk Score",    f"{row.Report_Risk_Score}/100",
              help="Higher = more risk language in filings")
    c7.metric("P/E Ratio",            f"{row.PE_Ratio:.1f}x")
    c8.metric("EV / EBITDA",          f"{row.EV_EBITDA:.1f}x")

    st.divider()

    # ── Three-column detail ───────────────────────────────────────────────────
    left, mid, right = st.columns(3, gap="medium")

    def _kv(label: str, value: str) -> None:
        col_l, col_r = st.columns([2, 1])
        col_l.markdown(
            f"<span style='font-size:13px;color:#64748b'>{label}</span>",
            unsafe_allow_html=True,
        )
        col_r.markdown(
            f"<span style='font-size:13px;font-weight:600'>{value}</span>",
            unsafe_allow_html=True,
        )

    with left:
        st.markdown('<div class="section-header">Valuation Metrics</div>', unsafe_allow_html=True)
        _kv("Current Price",   f"${row.Current_Price:.2f}")
        _kv("Est. Fair Value", f"${row.Estimated_Fair_Value:.2f}")
        _kv("Valuation Gap",   f"{row.Valuation_Gap_Pct:+.1f}%")
        _kv("P/E Ratio",       f"{row.PE_Ratio:.1f}x")
        pb_val = f"{row.PB_Ratio:.2f}x" if row.PB_Ratio > 0 else "Neg. (buybacks)"
        _kv("P/B Ratio",       pb_val)
        _kv("EV / EBITDA",     f"{row.EV_EBITDA:.1f}x")

    with mid:
        st.markdown('<div class="section-header">Quality Indicators</div>', unsafe_allow_html=True)
        _kv("Quality Score",   f"{row.Quality_Score}/100")
        _kv("ROE",             f"{row.ROE_Pct:.1f}%")
        _kv("Revenue (TTM)",   f"${row.Revenue_B:.1f}B")
        _kv("Revenue Growth",  f"{row.Revenue_Growth_Pct:+.1f}%")
        _kv("EPS (TTM)",       f"${row.EPS_TTM:.2f}")
        _kv("Free Cash Flow",  f"${row.FCF_B:.1f}B")

    with right:
        st.markdown('<div class="section-header">Risk Indicators</div>', unsafe_allow_html=True)
        _kv("Report Risk Score",   f"{row.Report_Risk_Score}/100")
        de_val = f"{row.Debt_Equity:.2f}x" if row.Debt_Equity >= 0 else "Neg. equity"
        _kv("Debt / Equity",       de_val)
        _kv("Risk Word Count",     str(row.Risk_Word_Count))
        _kv("Debt Mentions",       str(row.Debt_Mentions))
        _kv("Legal Mentions",      str(row.Legal_Mentions))
        _kv("Cost Pressure Flags", str(row.Cost_Pressure_Mentions))

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
                st.write(sigs.get("legal_mentions", "None identified."))
        with t2:
            with st.expander("Debt and Liquidity Language", expanded=True):
                st.write(sigs.get("debt_liquidity", "None identified."))
            with st.expander("Cost Pressure Language"):
                st.write(sigs.get("cost_pressures", "None identified."))

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Profile Radar</div>', unsafe_allow_html=True)

    rev_growth_norm = min(max((row.Revenue_Growth_Pct + 5) / 25 * 100, 0), 100)
    roe_norm        = min(max(row.ROE_Pct / 40 * 100, 0), 100)
    fcf_yield_norm  = min(max(row.FCF_B / max(row.Market_Cap_B, 0.001) * 1000, 0), 100)
    low_debt_norm   = max(100 - row.Debt_Equity * 15, 0)
    low_risk_norm   = 100 - row.Report_Risk_Score
    gap_norm        = min(max(row.Valuation_Gap_Pct + 10, 0) / 40 * 100, 100)

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
    st.caption(
        "Each axis is normalized 0-100 relative to model scale. "
        "Larger area = stronger overall profile on these dimensions."
    )


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
| **1. Data Collection** | Structured financial metrics pulled from public sources (yfinance, SEC EDGAR) |
| **2. Text Signal Extraction** | 10-K / 10-Q filings scanned for risk-related language |
| **3. Scoring** | Three model scores computed: Fair Value, Quality, Report Risk |
| **4. Signal Assignment** | Scores combined into a ranked research signal |
    """)

    st.divider()

    st.markdown('<div class="section-header">Stage 1 -- Data Collection</div>', unsafe_allow_html=True)
    st.markdown("""
**Structured financial data** for each company is collected from:
- **yfinance** -- current market price, P/E, P/B, EV/EBITDA, revenue, EPS, debt/equity
- **SEC EDGAR** -- 10-K and 10-Q filings (for text signal extraction)

For the MVP demo, a pre-processed sample dataset is used so the app runs without
live API latency. A "Refresh live prices" toggle is available in the sidebar.
    """)

    st.markdown('<div class="section-header">Stage 2 -- Text Signal Extraction</div>', unsafe_allow_html=True)
    st.markdown("""
10-K and 10-Q filings are scanned for four categories of language:

| Category | Example terms |
|---|---|
| **Risk keywords** | "uncertainty", "litigation", "regulatory", "impairment", "concentration risk" |
| **Debt and liquidity** | "covenant", "refinancing", "credit facility", "leverage", "liquidity risk" |
| **Legal / litigation** | "lawsuit", "investigation", "settlement", "enforcement action", "DOJ" |
| **Cost pressures** | "inflation", "supply chain", "wage pressure", "commodity costs", "restructuring" |

The frequency and severity of these terms is normalized into the **Report Risk Score (0-100)**,
where higher values indicate more risk-related language in recent filings.
    """)

    st.markdown('<div class="section-header">Stage 3 -- Scoring</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        st.markdown("##### Fair Value Estimation")
        st.markdown("""
The model uses a **simplified earnings-based approach**, blending:

1. **Sector-adjusted P/E benchmark** -- normalized for industry norms
2. **Growth premium/discount** -- companies with above/below-average revenue growth
   receive a multiplier adjustment
3. **FCF yield anchor** -- ensures the earnings-based estimate is grounded in cash generation

The output is an **estimated fair value per share** used to compute the Valuation Gap.
        """)
        st.code(
            "Valuation Gap % =\n"
            "  (Estimated Fair Value - Market Price)\n"
            "  / Market Price x 100",
            language="text",
        )
        st.caption(
            "Positive gap: model estimates potential upside.  "
            "Negative gap: model estimates potential overvaluation."
        )

    with sc2:
        st.markdown("##### Quality Score (0-100)")
        st.markdown("""
A composite score built from four fundamental factors:

| Factor | Weight | What it measures |
|---|---|---|
| Return on Equity | 30% | Profitability relative to equity base |
| FCF Yield | 25% | Cash generation relative to market cap |
| Revenue Growth | 25% | Top-line momentum |
| Debt / Equity | 20% | Balance sheet safety (lower is better) |

**Score of 70+** = strong fundamentals across most dimensions.
**Score below 45** = quality concerns (may flag as value trap).
        """)

    st.divider()

    st.markdown('<div class="section-header">Stage 4 -- Signal Assignment</div>', unsafe_allow_html=True)
    st.markdown("Signals are assigned by applying the following rule hierarchy (evaluated top to bottom):")

    sig_rows = [
        ("Possible value trap",              "Gap > 15% and Quality < 45",                   "Apparent upside exists, but low quality may negate it."),
        ("High-priority research candidate", "Gap > 20% and Quality >= 65 and Risk < 40",    "Clear potential upside, strong fundamentals, clean filing language."),
        ("Research candidate",               "Gap > 10% and Quality >= 50",                   "Moderate upside with acceptable quality -- worth investigating."),
        ("Fairly valued / neutral",          "-10% <= Gap <= +10%",                           "Model sees no clear margin of safety or significant overvaluation."),
        ("Potentially overvalued",           "Gap < -10%",                                    "Market price appears to exceed estimated fair value by >10%."),
    ]
    for sig, rule, reason in sig_rows:
        c1, c2, c3 = st.columns([2, 2, 3])
        sig_color = SIGNAL_COLORS.get(sig, "#333")
        c1.markdown(
            f'<span style="color:{sig_color};font-weight:600">{sig}</span>',
            unsafe_allow_html=True,
        )
        c2.markdown(f"`{rule}`")
        c3.markdown(f"*{reason}*")
        st.divider()

    st.markdown('<div class="section-header">Limitations and Disclaimers</div>', unsafe_allow_html=True)
    st.warning("""
**This model has important limitations you should understand:**

- **Fair value estimates are simplified** -- they use a small number of inputs and do not capture
  all factors a professional analyst would consider (competitive moats, management quality, macro environment)
- **Text scanning is keyword-based** -- it does not fully understand context or nuance in filing language
- **Historical data is not a predictor** -- a company's past financials do not guarantee future results
- **Sample universe is small** -- 10 companies is a demonstration, not a comprehensive market screen
- **This is NOT investment advice** -- signals indicate where to focus research time, not whether to buy or sell
    """)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5 -- BENCHMARK EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def page_benchmark(df: pd.DataFrame) -> None:
    st.title("Benchmark Evaluation")
    st.caption(
        "Compares a screener-selected portfolio against alternative benchmarks. "
        "All performance data is simulated for demonstration purposes."
    )

    st.info(
        "Demonstration Data Notice: The portfolio returns below are generated using "
        "simulated data (geometric Brownian motion) to illustrate what a benchmark comparison "
        "module would look like in a production system. They are NOT historical back-test results "
        "and should NOT be interpreted as performance claims."
    )

    # ── Portfolio composition ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Screener Portfolio Composition</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The screener portfolio consists of the top-ranked companies by signal priority. "
        "In this demo, the portfolio includes companies rated as High-priority or Research candidate:"
    )

    top = df[df["Final_Signal"].isin([
        "High-priority research candidate", "Research candidate"
    ])].copy()

    # Build portfolio display dataframe with column guards
    port_src = {
        "Ticker":            "Ticker",
        "Company_Name":      "Company",
        "Final_Signal":      "Signal",
        "Valuation_Gap_Pct": "Valuation Gap (%)",
        "Quality_Score":     "Quality Score",
    }
    available_src = {k: v for k, v in port_src.items() if k in top.columns}
    port_disp = top[list(available_src.keys())].copy().rename(columns=available_src)

    if "Valuation Gap (%)" in port_disp.columns:
        port_disp["Valuation Gap (%)"] = port_disp["Valuation Gap (%)"].map("{:+.1f}%".format)

    def _sig_style(val):
        return f"color:{SIGNAL_COLORS.get(val,'#000')}; font-weight:600"

    try:
        styler = port_disp.style
        styler = _safe_map(styler, _sig_style, "Signal")
        st.dataframe(styler, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Table styling unavailable ({exc}). Showing plain table.")
        st.dataframe(port_disp, use_container_width=True, hide_index=True)

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    bm_df = generate_benchmark_data(seed=42)

    # Confirm expected columns are present before plotting
    plot_series = {
        "Screener":             "#27ae60",
        "S&P 500":              "#2563eb",
        "Random Portfolio":     "#f39c12",
        "Risk-Free (4.5%)":     "#94a3b8",
    }
    plot_dashes = {
        "Screener":             "solid",
        "S&P 500":              "solid",
        "Random Portfolio":     "dash",
        "Risk-Free (4.5%)":     "dot",
    }

    fig = go.Figure()
    for series_name, color in plot_series.items():
        if series_name not in bm_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=bm_df["Date"],
            y=bm_df[series_name],
            name=series_name,
            mode="lines",
            line=dict(color=color, width=2, dash=plot_dashes[series_name]),
            hovertemplate=(
                f"<b>{series_name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Value: $%{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Simulated Portfolio Growth -- $10,000 Invested (1 Year)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Performance Summary</div>',
        unsafe_allow_html=True,
    )

    metrics_rows = []
    for series_name in plot_series:
        if series_name not in bm_df.columns:
            metrics_rows.append({
                "Portfolio":        series_name,
                "Annual Return (%)": "N/A",
                "Volatility (%)":    "N/A",
                "Sharpe Ratio":      "N/A",
                "Max Drawdown (%)":  "N/A",
                "Final Value ($)":   "N/A",
            })
            continue
        m = portfolio_metrics(bm_df[series_name].values)
        m["Portfolio"] = series_name
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Portfolio")

    # Column-wise highlight: best value per metric gets green background
    def _highlight_col(s):
        styles = []
        # Work only with numeric values
        numeric_vals = pd.to_numeric(s, errors="coerce")
        for val in numeric_vals:
            if pd.isna(val):
                styles.append("")
                continue
            if s.name in ("Annual Return (%)", "Sharpe Ratio", "Final Value ($)"):
                best  = numeric_vals.max()
                color = "#d5f5e3" if val == best else ""
            elif s.name == "Volatility (%)":
                best  = numeric_vals.min()    # lower is better
                color = "#d5f5e3" if val == best else ""
            elif s.name == "Max Drawdown (%)":
                best  = numeric_vals.max()    # least negative = best
                color = "#d5f5e3" if val == best else ""
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
        # Only format columns that are numeric (N/A rows are strings)
        numeric_cols = metrics_df.apply(lambda c: pd.to_numeric(c, errors="coerce").notna().all())
        numeric_fmt  = {k: v for k, v in fmt_dict.items()
                        if k in metrics_df.columns and numeric_cols.get(k, False)}
        styler = _safe_format(styler, numeric_fmt)
        st.dataframe(styler, use_container_width=True)
    except Exception as exc:
        st.warning(f"Table styling unavailable ({exc}). Showing plain table.")
        st.dataframe(metrics_df, use_container_width=True)

    st.caption(
        "Green cells highlight the best value per metric. "
        "Sharpe Ratio = (Return - Risk-Free Rate) / Volatility. "
        "All figures are annualized from simulated daily returns."
    )

    st.divider()

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">How to Interpret This</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
| Comparison | What it shows |
|---|---|
| **Screener vs. Random Portfolio** | Whether quality and risk filters add value over random selection from the same universe |
| **Screener vs. S&P 500** | Whether focusing on a small screened set competes with broad diversification |
| **Screener vs. Risk-Free Rate** | The minimum bar -- any equity strategy should exceed the risk-free rate over the long run |

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
    historical performance. The screener does not claim to outperform the S&P 500, Fidelity,
    Vanguard, or any professional investment manager. Past simulated performance is not
    indicative of future results.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════

if   page == "Home":
    page_home()
elif page == "Screener":
    page_screener(df)
elif page == "Company Detail":
    page_company_detail(df)
elif page == "Methodology":
    page_methodology()
elif page == "Benchmark Evaluation":
    page_benchmark(df)
