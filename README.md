# Equity Research Screener — MVP

**MSDSBA Practicum Project**
*A Time-Saving First-Pass Research Tool for Self-Directed Investors*

---

## What it does

Self-directed investors spend 15–45 minutes manually checking each stock before
deciding which companies deserve deeper analysis. This screener automates that
first pass: it aggregates financial metrics, scans 10-K/10-Q filing text for
risk signals, estimates fair value, and returns a ranked research shortlist in
seconds.

**This is a research support tool — not investment advice.**

---

## Quick Start

```bash
# 1. Clone / navigate to project folder
cd equity_screener

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Deploy Online

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/martinbauer1/underdawgs-equity-screener)

1. Push this folder to a GitHub repository.
2. In Render, choose **New > Blueprint** and connect the repository.
3. Render reads `render.yaml`, builds the app, and assigns a public URL.

The included health check uses Streamlit's `/_stcore/health` endpoint. The free
Render plan may sleep after a period of inactivity.

### Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file path to `app.py` and deploy.

No application secrets are required. Live-price refresh makes outbound requests
through `yfinance`; the bundled sample dataset remains available if that request
fails.

### Docker

```bash
docker build -t underdawgs-equity-screener .
docker run --rm -p 8501:8501 underdawgs-equity-screener
```

---

## Project Structure

```
equity_screener/
├── app.py            — Main Streamlit application (all 5 pages)
├── data_utils.py     — Data loading & optional live-price refresh (yfinance)
├── model_utils.py    — Scoring, signal logic, descriptions, benchmark simulation
├── sample_data.csv   — Pre-computed sample dataset (10 companies)
├── requirements.txt  — Python dependencies
└── README.md         — This file
```

---

## App Pages

| Page | Description |
|---|---|
| 🏠 Home | Problem framing, value proposition, disclaimer |
| 🔍 Screener | Ranked table with filters; valuation gap, quality, risk, signal |
| 🏢 Company Detail | Full profile, metrics, filing signal summary, radar chart |
| 📖 Methodology | Model explanation, formulas, signal assignment rules, limitations |
| 📈 Benchmark Evaluation | Simulated portfolio vs. S&P 500, random portfolio, risk-free rate |

---

## Sample Universe

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| NVDA | NVIDIA Corporation | Technology |
| JPM | JPMorgan Chase & Co. | Financial Services |
| WMT | Walmart Inc. | Consumer Defensive |
| SONY | Sony Group Corporation | Technology |
| HPQ | HP Inc. | Technology |
| DELL | Dell Technologies Inc. | Technology |
| LOGI | Logitech International | Technology |
| GRMN | Garmin Ltd. | Technology |

---

## Live Data (Optional)

Toggle **"Refresh live prices"** in the sidebar to pull current prices via
`yfinance`. All other metrics use the pre-computed sample dataset. If yfinance
is unavailable or slow, the app falls back to sample prices automatically.

---

## Signal Labels

| Signal | Criteria |
|---|---|
| 🟢 High-priority research candidate | Gap > 20%, Quality ≥ 65, Risk < 40 |
| 🔵 Research candidate | Gap > 10%, Quality ≥ 50 |
| 🟡 Fairly valued / neutral | −10% ≤ Gap ≤ 10% |
| 🔴 Potentially overvalued | Gap < −10% |
| 🟣 Possible value trap | Gap > 15% but Quality < 45 |

**Valuation Gap % = (Estimated Fair Value − Market Price) / Market Price × 100**

---

## Key Formulas

**Quality Score (0–100):**
```
= 0.30 × ROE_score
+ 0.25 × FCF_Yield_score
+ 0.25 × Revenue_Growth_score
+ 0.20 × LowDebt_score
```

**Report Risk Score (0–100):**
Based on normalized counts of risk keywords, debt/liquidity language,
legal/litigation mentions, and cost pressure language in SEC filings.

---

## Disclaimer

This tool is designed for **educational and research support purposes only**.
It does **not** constitute investment advice, financial recommendations, or
solicitation to buy or sell any securities. Signal labels indicate where to focus
*research time*, not capital. Always conduct your own due diligence and consult
a qualified financial advisor before making investment decisions.

The benchmark evaluation page uses **simulated demonstration data**. It does not
represent historical back-test results and should not be interpreted as performance
claims.
