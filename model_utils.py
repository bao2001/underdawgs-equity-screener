"""
model_utils.py
Scoring logic, signal definitions, company descriptions, and benchmark simulation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Signal definitions ──────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "High-priority research candidate": "#27ae60",
    "Research candidate":               "#2980b9",
    "Fairly valued / neutral":          "#f39c12",
    "Potentially overvalued":           "#e74c3c",
    "Possible value trap":              "#8e44ad",
}

SIGNAL_BG_COLORS = {
    "High-priority research candidate": "#d5f5e3",
    "Research candidate":               "#d6eaf8",
    "Fairly valued / neutral":          "#fef9e7",
    "Potentially overvalued":           "#fadbd8",
    "Possible value trap":              "#e8daef",
}

SIGNAL_ORDER = [
    "High-priority research candidate",
    "Research candidate",
    "Fairly valued / neutral",
    "Potentially overvalued",
    "Possible value trap",
]


def compute_signal(val_gap: float, quality: float, risk: float) -> str:
    """
    Determine the final research signal from three model outputs.

    Priority order (checked top-to-bottom):
      1. Possible value trap   — apparent upside but quality is low
      2. High-priority         — large gap + strong quality + low risk
      3. Research candidate    — moderate upside + acceptable quality
      4. Fairly valued         — gap within ±10 %
      5. Potentially overvalued— model says stock is priced above fair value
    """
    if val_gap > 15 and quality < 45:
        return "Possible value trap"
    if val_gap > 20 and quality >= 65 and risk < 40:
        return "High-priority research candidate"
    if val_gap > 10 and quality >= 50:
        return "Research candidate"
    if -10 <= val_gap <= 10:
        return "Fairly valued / neutral"
    if val_gap < -10:
        return "Potentially overvalued"
    return "Research candidate"


def get_signal_color(signal: str) -> str:
    return SIGNAL_COLORS.get(signal, "#95a5a6")


# ── Human-readable signal explanations ─────────────────────────────────────

def get_signal_explanation(row: pd.Series) -> str:
    signal  = row["Final_Signal"]
    ticker  = row["Ticker"]
    gap     = row["Valuation_Gap_Pct"]
    quality = row["Quality_Score"]
    risk    = row["Report_Risk_Score"]

    if signal == "High-priority research candidate":
        return (
            f"**{ticker}** shows a **{gap:+.1f}% estimated valuation gap**, meaning the model "
            f"believes the stock may be trading below its estimated intrinsic value. "
            f"A **Quality Score of {quality}/100** signals strong fundamentals (ROE, FCF yield, "
            f"revenue growth), and a **Report Risk Score of {risk}/100** indicates relatively "
            f"clean 10-K/10-Q language. This combination makes it a priority candidate for "
            f"deeper due diligence."
        )
    if signal == "Research candidate":
        return (
            f"**{ticker}** shows a **{gap:+.1f}% estimated valuation gap** with a "
            f"**Quality Score of {quality}/100**. There is potential upside, but one or more "
            f"factors (moderate quality, slightly elevated risk at {risk}/100, or gap below "
            f"20%) prevent a high-priority designation. Worth investigating further before "
            f"the higher-ranked names are exhausted."
        )
    if signal == "Fairly valued / neutral":
        return (
            f"**{ticker}**'s estimated fair value is close to its current market price "
            f"(gap: **{gap:+.1f}%**). The model does not identify a clear margin of safety "
            f"or significant overvaluation at this time. May be worth watching for a better "
            f"entry point or reevaluating if fundamentals change."
        )
    if signal == "Potentially overvalued":
        return (
            f"The model estimates **{ticker}**'s fair value at roughly **{abs(gap):.1f}% below** "
            f"the current market price. This suggests the market may be pricing in substantial "
            f"future growth already. High growth expectations are not inherently bad, but the "
            f"margin of safety is limited. Proceed with caution and stress-test growth assumptions."
        )
    if signal == "Possible value trap":
        return (
            f"**{ticker}** shows apparent upside (**{gap:+.1f}% gap**), but the "
            f"**Quality Score of {quality}/100** is low and the **Report Risk Score of "
            f"{risk}/100** is elevated. This pattern may indicate a *value trap* — "
            f"the stock appears cheap on headline metrics but may have deteriorating "
            f"fundamentals, high debt, or negative filing sentiment that erodes the apparent value."
        )
    return "Insufficient model data to generate an explanation."


# ── Company profiles ────────────────────────────────────────────────────────

COMPANY_DESCRIPTIONS = {
    "AAPL": (
        "Apple Inc. designs, manufactures, and markets consumer electronics, software, and services. "
        "Flagship products include iPhone, Mac, iPad, Apple Watch, and AirPods. The company's "
        "Services segment (App Store, Apple Music, iCloud, Apple Pay) now represents roughly 25% "
        "of revenue and carries significantly higher margins than hardware."
    ),
    "MSFT": (
        "Microsoft Corporation develops software, cloud services, and hardware. Azure is the "
        "company's fastest-growing segment, capturing enterprise cloud adoption. Other major "
        "products include Office 365, Windows, LinkedIn, and Xbox. AI integration across the "
        "product portfolio (Copilot) is a primary near-term growth driver."
    ),
    "NVDA": (
        "NVIDIA Corporation designs graphics processing units (GPUs) and system-on-chip units. "
        "Originally focused on gaming, NVIDIA now dominates AI/ML accelerator hardware. Its H100 "
        "and Blackwell GPU families are central to large language model training workloads globally. "
        "Data Center now represents over 80% of revenue."
    ),
    "JPM": (
        "JPMorgan Chase & Co. is the largest U.S. bank by assets (~$3.9 trillion). It operates "
        "through Consumer & Community Banking, Corporate & Investment Bank, Commercial Banking, "
        "and Asset & Wealth Management. Known for diversified revenue, disciplined risk management, "
        "and a fortress balance sheet."
    ),
    "WMT": (
        "Walmart Inc. is the world's largest retailer by revenue, serving ~230 million customers "
        "weekly across hypermarkets, discount stores, and e-commerce. The company is executing an "
        "omnichannel strategy that integrates in-store and online channels, and has been building "
        "advertising and fintech revenue streams."
    ),
    "SONY": (
        "Sony Group Corporation is a Japanese multinational with diversified operations including "
        "PlayStation gaming, Sony Pictures, Sony Music, imaging sensors (used in most smartphones), "
        "and financial services. The company benefits from IP-rich entertainment franchises and "
        "dominant market position in CMOS image sensors."
    ),
    "HPQ": (
        "HP Inc. provides personal computing products and printing solutions following the 2015 "
        "split from Hewlett-Packard. It sells notebooks, desktops, workstations, and printers "
        "to consumer and commercial markets. The PC market has faced post-pandemic normalization "
        "and the print market faces secular volume decline."
    ),
    "DELL": (
        "Dell Technologies provides servers, storage, networking, PCs, and IT services. Its "
        "Infrastructure Solutions Group (ISG) is growing on AI server demand. The company carries "
        "significant legacy debt but generates strong free cash flow and has stated debt reduction "
        "as a management priority."
    ),
    "LOGI": (
        "Logitech International designs and manufactures computer peripherals including mice, "
        "keyboards, webcams, headsets, gaming gear, and video conferencing equipment. The company "
        "is largely debt-free, benefits from remote/hybrid work trends, and has a strong track "
        "record of capital returns to shareholders."
    ),
    "GRMN": (
        "Garmin Ltd. designs GPS navigation and wearable devices across five segments: Fitness, "
        "Outdoor, Aviation, Marine, and Auto OEM. The company is debt-free with $3.4B in cash, "
        "commands premium pricing through brand loyalty, and has delivered consistent dividend "
        "growth. Outdoor and Aviation segments are key long-term growth drivers."
    ),
}


# ── Report text signal samples ──────────────────────────────────────────────

REPORT_SIGNALS = {
    "AAPL": {
        "risk_keywords":      ["supply chain concentration", "tariff risk", "platform regulation", "App Store antitrust", "China revenue exposure"],
        "debt_liquidity":     "Apple holds ~$29.9B in cash/securities against ~$109B long-term debt, primarily used for capital returns. Strong FCF of ~$99B/yr provides ample coverage.",
        "legal_mentions":     "Ongoing EU antitrust probe into App Store practices. Epic Games litigation in various jurisdictions. DOJ investigation into monopoly practices.",
        "cost_pressures":     "Component costs and logistics have stabilized post-COVID but remain structurally higher. Services gross margin expansion offsets hardware margin pressure.",
    },
    "MSFT": {
        "risk_keywords":      ["cloud competition", "AI infrastructure cost", "regulatory scrutiny", "cybersecurity breach risk", "talent retention"],
        "debt_liquidity":     "Microsoft holds ~$80B in cash/investments against ~$67B long-term debt. Free cash flow of $75B/yr makes the balance sheet self-funding for capex.",
        "legal_mentions":     "Activision acquisition closed after regulatory approvals. EU Digital Markets Act compliance ongoing. No material unresolved litigation.",
        "cost_pressures":     "AI data center build-out is capital intensive. Management frames this as a long-term investment. Operating leverage from cloud mix shift partially offsets.",
    },
    "NVDA": {
        "risk_keywords":      ["U.S. export controls", "China revenue risk", "customer concentration", "AI demand sustainability", "CoWoS packaging constraint"],
        "debt_liquidity":     "NVIDIA holds ~$35B in cash with minimal long-term debt. Generates strong FCF but lumpy capital expenditure tied to TSMC manufacturing prepayments.",
        "legal_mentions":     "Subject to BIS export regulations limiting H100/A100 sales to China. Ongoing compliance program in place. No major litigation.",
        "cost_pressures":     "TSMC 4nm/3nm node costs elevated. HBM memory from SK Hynix/Samsung is supply-constrained and expensive. Gross margins remain >70% despite cost headwinds.",
    },
    "JPM": {
        "risk_keywords":      ["credit cycle deterioration", "commercial real estate exposure", "Basel III endgame capital rules", "net interest margin compression", "geopolitical risk"],
        "debt_liquidity":     "JPMorgan holds $1.4T in deposits and a CET1 ratio of 15.0%, well above the 11.4% regulatory minimum. Liquidity is a core competitive advantage.",
        "legal_mentions":     "Faces ongoing regulatory exams including Basel III endgame proposals. Settlement discussions in several legacy matters. No material unresolved criminal matters.",
        "cost_pressures":     "Compensation and technology expenses are the primary cost drivers. Non-interest expense grew ~8% Y/Y. Efficiency ratio management is a stated priority.",
    },
    "WMT": {
        "risk_keywords":      ["consumer discretionary softness", "grocery deflation", "e-commerce investment burn", "FX headwinds international", "shrinkage/theft elevated"],
        "debt_liquidity":     "Walmart holds ~$9.9B cash against ~$34.7B long-term debt. Consistent operating cash flow of ~$30B/yr supports dividends, buybacks, and debt service.",
        "legal_mentions":     "Ongoing opioid settlement payments (~$3.1B over 15 years). Various employment practice suits. No material existential legal risk.",
        "cost_pressures":     "Wage rate increases implemented across U.S. store network. International operations face FX drag from strong dollar. Shrinkage remains elevated in some markets.",
    },
    "SONY": {
        "risk_keywords":      ["gaming market saturation", "content cost escalation", "yen weakness impact on USD reporting", "streaming market maturity", "semiconductor supply risk"],
        "debt_liquidity":     "Sony's electronics and entertainment segments carry modest net debt. Financial services segment has ¥2.3T in insurance/banking liabilities, structurally ring-fenced.",
        "legal_mentions":     "PlayStation subscriptions face European competition review. Content licensing disputes in entertainment. Minor IP litigation in electronics.",
        "cost_pressures":     "AAA game development costs rising materially (~$200–300M per major title). Image sensor fab costs elevated. Yen weakness boosts exports but raises import costs.",
    },
    "HPQ": {
        "risk_keywords":      ["PC market secular decline", "print volume decline", "channel inventory excess", "commodity cost volatility", "restructuring execution risk", "negative equity from buybacks"],
        "debt_liquidity":     "HP Inc. carries ~$8.8B long-term debt against negative book equity due to aggressive buybacks. Primary liquidity source is FCF (~$3.2B/yr). Refinancing risk manageable near-term.",
        "legal_mentions":     "Xerox hostile takeover litigation settled. Ongoing IP licensing disputes. Several employment practice lawsuits pending. SEC has not taken action on past accounting matters.",
        "cost_pressures":     "NAND and DRAM cost volatility affects PC margins. Logistics costs elevated. Restructuring charges ($300–500M/yr) expected to continue through cost-reduction program.",
    },
    "DELL": {
        "risk_keywords":      ["AI server supply chain bottlenecks", "enterprise IT budget cycles", "debt load from EMC/VMware era", "customer concentration (hyperscalers)", "margin compression on AI builds"],
        "debt_liquidity":     "Dell carries ~$19.8B long-term debt. Management targets 1.5x net leverage ratio. FCF of ~$4.5B/yr and VMware separation proceeds are being used for paydown.",
        "legal_mentions":     "SEC investigation into accounting practices concluded with no action. Environmental compliance matters ongoing. No material unresolved litigation.",
        "cost_pressures":     "AI servers require expensive NVIDIA GPUs, creating large working capital requirements. DRAM and SSD pricing volatile. Services gross margin expansion partially offsets hardware pressure.",
    },
    "LOGI": {
        "risk_keywords":      ["PC peripherals demand normalization", "post-pandemic demand hangover", "Asia manufacturing concentration", "consumer discretionary spending sensitivity"],
        "debt_liquidity":     "Logitech is essentially debt-free with ~$1.2B in cash. The balance sheet is a competitive advantage, enabling dividends, buybacks, and opportunistic M&A.",
        "legal_mentions":     "No material litigation. Minor trademark disputes in several Asian markets resolved.",
        "cost_pressures":     "Component costs normalizing. Logistics costs declining. USD strength creates headwinds on international revenue (~75% of sales outside Americas). Inventory levels normalized.",
    },
    "GRMN": {
        "risk_keywords":      ["wearables market saturation", "Apple Watch competition in fitness", "aviation regulatory changes", "marine market cyclicality", "FX exposure (CHF/USD)"],
        "debt_liquidity":     "Garmin is entirely debt-free with ~$3.4B in cash. Conservative capital structure supports growing dividends ($3.00/share annualized) and opportunistic buybacks.",
        "legal_mentions":     "Minor patent litigation in navigation space. No material legal risks disclosed in most recent 10-K.",
        "cost_pressures":     "Chip costs stabilizing. Modest wage inflation in Switzerland. R&D investment growing to maintain competitive positioning in aviation and outdoor.",
    },
}


# ── Benchmark simulation ────────────────────────────────────────────────────

def generate_benchmark_data(seed: int = 42) -> pd.DataFrame:
    """
    Simulate 1 year of daily portfolio performance using geometric Brownian motion.
    Returns a DataFrame with columns: Date, Screener, SP500, Random, RiskFree.
    All series start at $10,000.

    This is DEMONSTRATION DATA — not historical back-test results.
    """
    np.random.seed(seed)
    days = 252
    dt   = 1 / 252
    start_date = datetime.today() - timedelta(days=365)
    dates = pd.date_range(start=start_date, periods=days + 1, freq="B")[:days + 1]

    def gbm(mu: float, sigma: float, start: float = 10_000.0) -> np.ndarray:
        z       = np.random.standard_normal(days)
        returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z) - 1
        prices  = start * np.cumprod(1 + returns)
        return np.insert(prices, 0, start)

    screener = gbm(mu=0.20, sigma=0.14)   # ~20 % annual, moderate vol
    sp500    = gbm(mu=0.24, sigma=0.12)   # ~24 % annual, lower vol (diversified)
    random_p = gbm(mu=0.12, sigma=0.19)   # ~12 % annual, higher vol (undiversified)
    rf       = 10_000 * np.exp(np.linspace(0, 0.045, days + 1))  # 4.5 % risk-free

    df = pd.DataFrame({
        "Date":            dates,
        "Screener":        np.round(screener, 2),
        "S&P 500":         np.round(sp500, 2),
        "Random Portfolio":np.round(random_p, 2),
        "Risk-Free (4.5%)":np.round(rf, 2),
    })
    return df


def portfolio_metrics(series: np.ndarray, rf_annual: float = 0.045) -> dict:
    """Compute annualized return, volatility, Sharpe ratio, and max drawdown."""
    daily_ret   = np.diff(series) / series[:-1]
    ann_return  = (series[-1] / series[0]) - 1
    ann_vol     = daily_ret.std() * np.sqrt(252)
    sharpe      = (ann_return - rf_annual) / ann_vol if ann_vol > 0.001 else 0.0

    peak    = series[0]
    max_dd  = 0.0
    for p in series:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        "Annual Return (%)":   round(ann_return * 100, 1),
        "Volatility (%)":      round(ann_vol * 100, 1),
        "Sharpe Ratio":        round(sharpe, 2),
        "Max Drawdown (%)":    round(max_dd * 100, 1),
        "Final Value ($)":     round(series[-1], 0),
    }
