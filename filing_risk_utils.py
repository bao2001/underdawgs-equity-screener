"""
filing_risk_utils.py
SEC 10-K filing text risk extraction pipeline.

Downloads annual 10-K filings from SEC EDGAR, extracts risk-related language,
computes filing risk scores (0–100, relative to dataset peers), and merges them
into model_outputs_combined_calibrated.csv.

Files generated:
  sec_filing_index_cache/          — cached SEC submissions JSON per company
  sec_filing_text_cache/           — cached plain-text per filing (ticker_year.txt)
  filing_lookup_results.csv        — which 10-K was found for each ticker+year
  filing_risk_features.csv         — raw risk keyword counts per filing
  filing_risk_scores.csv           — final 0-100 risk scores per ticker+year
  model_outputs_combined_calibrated_risk.csv — calibrated outputs + risk scores

Run:
  export SEC_USER_AGENT="Your Name your.email@example.com"
  python3 filing_risk_utils.py

Notes:
  - SEC policy: ≤10 req/s, must include User-Agent.
  - Caches aggressively; re-running is fast.
  - Missing filings keep report_risk_score=50 (neutral placeholder).
  - All outputs are research signals only. Not investment advice.
"""

import os
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
from html.parser import HTMLParser

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

USER_AGENT = os.getenv("SEC_USER_AGENT", "MSDSBA Practicum baoonguyen80@gmail.com")

SEC_SUBMISSIONS_BASE   = "https://data.sec.gov/submissions"
SEC_EDGAR_ARCHIVES     = "https://www.sec.gov/Archives/edgar/data"

FILING_INDEX_CACHE_DIR = os.path.join(_HERE, "sec_filing_index_cache")
FILING_TEXT_CACHE_DIR  = os.path.join(_HERE, "sec_filing_text_cache")

CALIBRATED_PATH        = os.path.join(_HERE, "model_outputs_combined_calibrated.csv")
TICKER_CIK_MAP_PATH    = os.path.join(_HERE, "ticker_cik_mapping.csv")
FILING_LOOKUP_PATH     = os.path.join(_HERE, "filing_lookup_results.csv")
RISK_FEATURES_PATH     = os.path.join(_HERE, "filing_risk_features.csv")
RISK_SCORES_PATH       = os.path.join(_HERE, "filing_risk_scores.csv")
RISK_OUTPUTS_PATH      = os.path.join(_HERE, "model_outputs_combined_calibrated_risk.csv")

LATEST_10Q_RISK_PATH         = os.path.join(_HERE, "latest_10q_risk_scores.csv")
FILING_RISK_UPDATE_PATH      = os.path.join(_HERE, "filing_risk_current_update.csv")
FILING_10Q_TEXT_CACHE_DIR    = os.path.join(_HERE, "sec_filing_10q_cache")

MAX_DOWNLOAD_BYTES     = 5 * 1024 * 1024   # 5 MB per filing
MAX_TEXT_CHARS         = 300_000            # keep first 300K chars of cleaned text
SEC_RATE_LIMIT_SLEEP   = 0.12              # seconds between SEC API calls

# ── Risk keyword categories ────────────────────────────────────────────────────

_RISK_WORDS = [
    "risk", "risks", "uncertain", "uncertainty", "uncertainties",
    "adverse", "volatile", "volatility", "disruption", "disruptions",
    "decline", "deterioration", "challenge", "threat", "exposure",
    "significant risk", "material risk",
]

_DEBT_WORDS = [
    "debt", "leverage", "liquidity", "indebtedness", "borrowing",
    "credit facility", "refinanc", "covenant", "going concern",
    "cash flow risk", "interest expense", "default", "maturity",
    "credit risk",
]

_LEGAL_WORDS = [
    "litigation", "lawsuit", "legal proceeding", "regulatory",
    "investigation", "subpoena", "compliance", "violation",
    "penalty", "antitrust", "class action", "enforcement", "sanction",
    "legal risk",
]

_UNCERTAINTY_WORDS = [
    "material adverse", "adverse effect", "no assurance", "cannot guarantee",
    "may not be able", "may not achieve", "cannot predict", "unpredictable",
    "impairment", "write-off", "write-down", "goodwill impairment",
    "no guarantee",
]

_COST_WORDS = [
    "inflation", "supply chain", "restructuring", "headcount reduction",
    "workforce reduction", "tariff", "commodity", "margin pressure",
    "cost increase", "pricing pressure", "cost pressur",
]

_CYBER_WORDS = [
    "cybersecurity", "data breach", "cyberattack", "ransomware",
    "security incident", "unauthorized access", "data privacy",
    "cyber risk",
]


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _http_get(url: str, headers: dict, max_bytes: int = None) -> bytes | None:
    """
    Fetch URL bytes via requests or urllib fallback.
    Returns bytes or None on failure.
    Respects max_bytes via streaming.
    """
    _req_err = None
    try:
        import requests as _req
        r = _req.get(url, headers=headers, timeout=30, stream=(max_bytes is not None))
        r.raise_for_status()
        if max_bytes:
            data = b""
            for chunk in r.iter_content(chunk_size=65536):
                data += chunk
                if len(data) >= max_bytes:
                    break
            return data[:max_bytes]
        return r.content
    except ImportError:
        pass
    except Exception as exc:
        _req_err = str(exc)

    try:
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        try:
            ctx.load_default_certs()
        except Exception:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            if max_bytes:
                return resp.read(max_bytes)
            return resp.read()
    except Exception as exc:
        parts = []
        if _req_err:
            parts.append(f"requests: {_req_err}")
        parts.append(f"urllib: {exc}")
        return None


def _http_get_json(url: str, headers: dict) -> dict | None:
    data = _http_get(url, headers)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ── CIK mapping ───────────────────────────────────────────────────────────────

def load_ticker_cik_mapping() -> tuple:
    """Returns (df, status_msg). df has columns ticker, company_name, cik."""
    if not os.path.exists(TICKER_CIK_MAP_PATH):
        return None, (
            "ticker_cik_mapping.csv not found. "
            "Run: python3 modern_fundamentals_utils.py to create it."
        )
    try:
        df = pd.read_csv(TICKER_CIK_MAP_PATH, dtype={"cik": str})
        df = df.dropna(subset=["ticker", "cik"])
        df["cik"] = df["cik"].str.strip()
        return df, f"Loaded {len(df)} CIK mappings."
    except Exception as exc:
        return None, f"Error loading ticker_cik_mapping.csv: {exc}"


# ── SEC submissions (filing index) cache ──────────────────────────────────────

def _submissions_cache_path(cik: str) -> str:
    os.makedirs(FILING_INDEX_CACHE_DIR, exist_ok=True)
    return os.path.join(FILING_INDEX_CACHE_DIR, f"CIK{str(cik).zfill(10)}.json")


def _load_submissions_cache(cik: str) -> dict | None:
    path = _submissions_cache_path(cik)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_submissions_cache(cik: str, data: dict) -> None:
    path = _submissions_cache_path(cik)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def fetch_company_submissions(cik: str, force_refresh: bool = False) -> tuple:
    """
    Fetch company submissions from SEC EDGAR (cached).
    Returns (submissions_dict, source, status_msg).
    """
    if not force_refresh:
        cached = _load_submissions_cache(cik)
        if cached is not None:
            return cached, "cache", "ok"

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip,deflate"}
    padded  = str(cik).zfill(10)
    url     = f"{SEC_SUBMISSIONS_BASE}/CIK{padded}.json"
    data    = _http_get_json(url, headers)
    time.sleep(SEC_RATE_LIMIT_SLEEP)

    if data is None:
        return None, "api", f"Failed to fetch submissions for CIK {cik}"

    # If older filings are paginated, merge them
    extra_files = data.get("filings", {}).get("files", [])
    if extra_files:
        recent = data.get("filings", {}).get("recent", {})
        for extra in extra_files:
            extra_url  = f"{SEC_SUBMISSIONS_BASE}/{extra['name']}"
            extra_data = _http_get_json(extra_url, headers)
            time.sleep(SEC_RATE_LIMIT_SLEEP)
            if extra_data is None:
                continue
            for key, vals in extra_data.items():
                if isinstance(vals, list) and key in recent:
                    recent[key] = recent[key] + vals
        data["filings"]["recent"] = recent
        data["filings"]["files"]  = []   # already merged

    _save_submissions_cache(cik, data)
    return data, "api", "ok"


# ── Filing lookup ──────────────────────────────────────────────────────────────

def _parse_filings_recent(recent: dict) -> pd.DataFrame:
    """Convert filings.recent dict of arrays into a DataFrame."""
    if not recent:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(recent)
        if "accessionNumber" in df.columns:
            df = df.rename(columns={"accessionNumber": "accession_number"})
        if "filingDate" in df.columns:
            df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
        if "reportDate" in df.columns:
            df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def find_annual_filing(submissions: dict, target_year: int,
                       prefer_form: str = "10-K") -> dict | None:
    """
    Find a 10-K (or 10-K405/20-F) for the given fiscal year.

    Strategy:
      1. reportDate year == target_year AND form matches
      2. filingDate year == target_year+1 AND form matches (fiscal year end in year Y, filed in Y+1)
      3. filingDate year == target_year AND form matches (rare: fiscal year ending Dec, filed same year)
    Returns a dict with keys: accession_number, form_type, filing_date, report_date, primary_doc
    or None if not found.
    """
    recent  = submissions.get("filings", {}).get("recent", {})
    filings = _parse_filings_recent(recent)
    if filings.empty:
        return None

    annual_forms = {"10-K", "10-K405", "20-F", "40-F"}
    if "form" not in filings.columns:
        return None

    mask_form = filings["form"].isin(annual_forms)
    annual    = filings[mask_form].copy()
    if annual.empty:
        return None

    # Sort by most recently filed first
    if "filingDate" in annual.columns:
        annual = annual.sort_values("filingDate", ascending=False)

    # Strategy 1: reportDate year matches
    if "reportDate" in annual.columns:
        match = annual[annual["reportDate"].dt.year == target_year]
        if not match.empty:
            row = match.iloc[0]
            return {
                "accession_number": str(row.get("accession_number", "")),
                "form_type":        str(row.get("form", prefer_form)),
                "filing_date":      str(row.get("filingDate", ""))[:10],
                "report_date":      str(row.get("reportDate", ""))[:10],
                "primary_doc":      str(row.get("primaryDocument", "")),
            }

    # Strategy 2: filingDate in target_year+1 (filed Q1 of following year)
    if "filingDate" in annual.columns:
        match = annual[annual["filingDate"].dt.year == target_year + 1]
        if not match.empty:
            row = match.iloc[0]
            return {
                "accession_number": str(row.get("accession_number", "")),
                "form_type":        str(row.get("form", prefer_form)),
                "filing_date":      str(row.get("filingDate", ""))[:10],
                "report_date":      str(row.get("reportDate", ""))[:10] if "reportDate" in row else "",
                "primary_doc":      str(row.get("primaryDocument", "")),
            }

    # Strategy 3: filingDate in target_year (December fiscal year end)
    if "filingDate" in annual.columns:
        match = annual[annual["filingDate"].dt.year == target_year]
        if not match.empty:
            row = match.iloc[0]
            return {
                "accession_number": str(row.get("accession_number", "")),
                "form_type":        str(row.get("form", prefer_form)),
                "filing_date":      str(row.get("filingDate", ""))[:10],
                "report_date":      str(row.get("reportDate", ""))[:10] if "reportDate" in row else "",
                "primary_doc":      str(row.get("primaryDocument", "")),
            }

    return None


# ── Filing text fetch ─────────────────────────────────────────────────────────

def _text_cache_path(ticker: str, year: int) -> str:
    os.makedirs(FILING_TEXT_CACHE_DIR, exist_ok=True)
    return os.path.join(FILING_TEXT_CACHE_DIR, f"{ticker.upper()}_{year}.txt")


def _is_text_cached(ticker: str, year: int) -> bool:
    path = _text_cache_path(ticker, year)
    return os.path.exists(path) and os.path.getsize(path) > 100


def _save_text_cache(ticker: str, year: int, text: str) -> None:
    path = _text_cache_path(ticker, year)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text[:MAX_TEXT_CHARS])
    except Exception:
        pass


def _load_text_cache(ticker: str, year: int) -> str | None:
    path = _text_cache_path(ticker, year)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


class _TextExtractor(HTMLParser):
    """Minimal HTML→text extractor using stdlib only."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts  = []
        self._skip   = 0   # depth counter for script/style blocks

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        if tag.lower() in ("p", "div", "br", "tr", "td", "th", "li"):
            self._parts.append(" ")

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\s+", " ", raw).strip()


def _html_to_text(raw: bytes) -> str:
    """Strip HTML tags and return plain text."""
    try:
        content = raw.decode("utf-8", errors="replace")
    except Exception:
        content = raw.decode("latin-1", errors="replace")
    # Try HTML parser
    try:
        parser = _TextExtractor()
        parser.feed(content[:MAX_DOWNLOAD_BYTES])
        text = parser.get_text()
        if len(text) > 500:
            return text[:MAX_TEXT_CHARS]
    except Exception:
        pass
    # Fallback: regex strip
    text = re.sub(r"<[^>]+>", " ", content)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_CHARS]


def _extract_risk_section(text: str) -> str:
    """
    Try to extract 'Item 1A Risk Factors' section.
    Falls back to full text if the section is not found.
    Limits extraction to 80K characters.
    """
    # Patterns for the start of the risk factors section
    start_patterns = [
        r"item\s+1a[\.:]?\s+risk\s+factors",
        r"item\s+1a\b",
        r"risk\s+factors",
    ]
    for pat in start_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start = m.start()
            # Find the end: next major section header or 80K chars
            tail  = text[m.end():]
            end_patterns = [
                r"item\s+1b\b", r"item\s+2\b", r"item\s+3\b",
            ]
            end_offset = 80_000
            for epat in end_patterns:
                em = re.search(epat, tail, re.IGNORECASE)
                if em and em.start() > 500:
                    end_offset = min(end_offset, em.start())
                    break
            section = text[start: m.end() + end_offset]
            if len(section) > 500:
                return section
    return text[:80_000]   # full text fallback, still capped


def fetch_filing_text(cik: str, ticker: str, year: int,
                       accession: str, primary_doc: str,
                       verbose: bool = False) -> tuple:
    """
    Download and return plain text for a filing. Caches locally.
    Returns (text, status_msg).
    """
    if _is_text_cached(ticker, year):
        text = _load_text_cache(ticker, year)
        if text:
            return text, "cache"

    if not accession or not primary_doc:
        return "", "missing_accession_or_doc"

    # Build URL: remove dashes from accession, use integer CIK in path
    acc_clean = accession.replace("-", "")
    url       = f"{SEC_EDGAR_ARCHIVES}/{int(cik)}/{acc_clean}/{primary_doc}"
    headers   = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip,deflate"}

    raw = _http_get(url, headers, max_bytes=MAX_DOWNLOAD_BYTES)
    time.sleep(SEC_RATE_LIMIT_SLEEP)

    if raw is None:
        return "", f"fetch_failed: {url}"

    # Detect if it's HTML or plain text
    content_start = raw[:1000].lower()
    if b"<html" in content_start or b"<!doctype" in content_start or b"<body" in content_start:
        text = _html_to_text(raw)
    else:
        try:
            text = raw.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]
        except Exception:
            text = raw.decode("latin-1", errors="replace")[:MAX_TEXT_CHARS]

    if len(text) < 200:
        return "", "too_short_after_parse"

    _save_text_cache(ticker, year, text)
    if verbose:
        print(f"    fetched {len(text):,} chars  [{primary_doc}]")
    return text, "ok"


# ── Risk feature extraction ───────────────────────────────────────────────────

def _count_keywords(text: str, keywords: list) -> int:
    """Count all occurrences of all keywords (case-insensitive) in text."""
    tl = text.lower()
    return sum(tl.count(kw.lower()) for kw in keywords)


def extract_risk_features(text: str, ticker: str, company_name: str,
                           cik: str, year: int, form_type: str,
                           filing_date: str) -> dict:
    """
    Compute risk keyword counts from filing plain text.
    Counts from the Risk Factors section where possible.
    """
    risk_section = _extract_risk_section(text)
    words        = risk_section.split()
    total_words  = len(words)

    if total_words < 100:
        return {
            "ticker":             ticker,
            "company_name":       company_name,
            "cik":                str(cik),
            "year":               year,
            "form_type":          form_type,
            "filing_date":        filing_date,
            "total_words":        total_words,
            "risk_word_count":    0,
            "risk_words_per_1000": 0.0,
            "debt_mentions":       0,
            "liquidity_mentions":  0,
            "legal_mentions":      0,
            "regulatory_mentions": 0,
            "uncertainty_mentions": 0,
            "cost_pressure_mentions": 0,
            "cybersecurity_mentions": 0,
            "data_quality_flag":  "too_short",
            "failure_reason":     f"text too short ({total_words} words)",
        }

    def _per_1k(count):
        return round(count / total_words * 1000, 2) if total_words > 0 else 0.0

    risk_count     = _count_keywords(risk_section, _RISK_WORDS)
    debt_count     = _count_keywords(risk_section, _DEBT_WORDS)
    legal_count    = _count_keywords(risk_section, _LEGAL_WORDS)
    uncert_count   = _count_keywords(risk_section, _UNCERTAINTY_WORDS)
    cost_count     = _count_keywords(risk_section, _COST_WORDS)
    cyber_count    = _count_keywords(risk_section, _CYBER_WORDS)

    return {
        "ticker":             ticker,
        "company_name":       company_name,
        "cik":                str(cik),
        "year":               year,
        "form_type":          form_type,
        "filing_date":        filing_date,
        "total_words":        total_words,
        "risk_word_count":    risk_count,
        "risk_words_per_1000":    _per_1k(risk_count),
        "debt_mentions":           debt_count,
        "liquidity_mentions":      _count_keywords(risk_section, ["liquidity", "liquid"]),
        "legal_mentions":          legal_count,
        "regulatory_mentions":     _count_keywords(risk_section, ["regulatory", "regulation"]),
        "uncertainty_mentions":    uncert_count,
        "cost_pressure_mentions":  cost_count,
        "cybersecurity_mentions":  cyber_count,
        "data_quality_flag":  "ok",
        "failure_reason":     "",
    }


# ── Risk score computation ─────────────────────────────────────────────────────

_SCORE_WEIGHTS = {
    "risk_words_per_1000":     0.30,
    "debt_mentions_norm":       0.25,
    "legal_mentions_norm":      0.20,
    "uncertainty_mentions_norm": 0.15,
    "cost_cyber_norm":           0.10,
}


def compute_risk_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 0–100 filing risk scores from raw feature counts.

    Uses percentile ranks across the entire dataset (not per-year)
    for stability given small annual sample sizes.

    Higher score = more risk-related language relative to dataset peers.
    """
    df = features_df.copy()
    ok = df["data_quality_flag"] == "ok"

    if ok.sum() == 0:
        df["report_risk_score_real"]   = np.nan
        df["risk_score_source"]         = "parse_error"
        df["risk_score_components"]     = ""
        df["filing_data_quality_flag"]  = df["data_quality_flag"]
        df["filing_warning_text"]       = df["failure_reason"]
        return df

    total_words = df.loc[ok, "total_words"].replace(0, 1)

    # Normalized features (per 1000 words for debt/legal/uncertainty/cost to match risk_per_1k)
    df.loc[ok, "debt_mentions_norm"]        = df.loc[ok, "debt_mentions"]         / total_words * 1000
    df.loc[ok, "legal_mentions_norm"]       = df.loc[ok, "legal_mentions"]        / total_words * 1000
    df.loc[ok, "uncertainty_mentions_norm"] = df.loc[ok, "uncertainty_mentions"]  / total_words * 1000
    df.loc[ok, "cost_cyber_norm"]           = (
        df.loc[ok, "cost_pressure_mentions"] + df.loc[ok, "cybersecurity_mentions"]
    ) / total_words * 1000

    # Percentile ranks across all parseable rows
    score_cols = list(_SCORE_WEIGHTS.keys())
    pct_ranks  = {}
    for col in score_cols:
        if col in df.columns:
            pct_ranks[col] = df.loc[ok, col].rank(pct=True) * 100
        else:
            pct_ranks[col] = pd.Series(50.0, index=df[ok].index)

    # Weighted composite
    composite = sum(
        pct_ranks[col] * weight
        for col, weight in _SCORE_WEIGHTS.items()
    ).round(1)

    df.loc[ok, "report_risk_score_real"]  = composite
    df.loc[~ok, "report_risk_score_real"] = np.nan

    df["risk_score_source"] = np.where(ok, "real_sec_filing", "filing_missing_or_error")
    df["filing_data_quality_flag"] = df["data_quality_flag"]
    df["filing_warning_text"]      = df["failure_reason"].fillna("")

    df["risk_score_components"] = df.apply(
        lambda r: (
            f"risk_per1k={r.get('risk_words_per_1000',0):.1f}, "
            f"debt={r.get('debt_mentions',0)}, "
            f"legal={r.get('legal_mentions',0)}, "
            f"uncertainty={r.get('uncertainty_mentions',0)}, "
            f"cost+cyber={r.get('cost_pressure_mentions',0)+r.get('cybersecurity_mentions',0)}"
        ) if r.get("data_quality_flag") == "ok" else "",
        axis=1,
    )

    return df


# ── Main pipeline steps ────────────────────────────────────────────────────────

def build_filing_lookup(calibrated_df: pd.DataFrame,
                         ticker_cik_df: pd.DataFrame,
                         verbose: bool = True) -> pd.DataFrame:
    """
    For each (ticker, year) in calibrated_df, find the 10-K filing.
    Returns a DataFrame saved to filing_lookup_results.csv.
    """
    cik_map = ticker_cik_df.set_index("ticker")["cik"].to_dict()
    name_map = {}
    if "company_name" in ticker_cik_df.columns:
        name_map = ticker_cik_df.set_index("ticker")["company_name"].to_dict()

    pairs = (calibrated_df[["ticker", "year"]]
             .drop_duplicates()
             .sort_values(["ticker", "year"]))

    rows = []
    tickers_done = {}

    for _, pair in pairs.iterrows():
        ticker = str(pair["ticker"]).upper()
        year   = int(pair["year"])
        cik    = cik_map.get(ticker, cik_map.get(ticker.lower()))
        cname  = name_map.get(ticker, ticker)

        base_row = {
            "ticker":          ticker,
            "company_name":    cname,
            "cik":             str(cik) if cik else "",
            "year":            year,
            "form_type":       "",
            "filing_date":     "",
            "report_date":     "",
            "accession_number": "",
            "primary_doc":     "",
            "filing_url":      "",
            "filing_text_path": _text_cache_path(ticker, year),
            "fetch_status":    "cik_missing",
            "failure_reason":  "",
        }

        if not cik:
            base_row["failure_reason"] = f"No CIK for ticker {ticker}"
            rows.append(base_row)
            if verbose:
                print(f"  {ticker} {year}: no CIK")
            continue

        # Fetch submissions (cached per CIK)
        if ticker not in tickers_done:
            if verbose:
                print(f"  {ticker} (CIK {cik}): fetching submissions …", end=" ", flush=True)
            sub, src, msg = fetch_company_submissions(str(cik))
            tickers_done[ticker] = sub
            if verbose:
                print(f"[{src}]" + ("" if sub else f" FAILED: {msg}"))
        else:
            sub = tickers_done[ticker]

        if sub is None:
            base_row["fetch_status"]   = "submissions_error"
            base_row["failure_reason"] = f"Could not fetch submissions for {ticker}"
            rows.append(base_row)
            continue

        filing = find_annual_filing(sub, year)
        if filing is None:
            base_row["fetch_status"]   = "not_found"
            base_row["failure_reason"] = f"No 10-K for fiscal year {year}"
            rows.append(base_row)
            continue

        acc_clean = filing["accession_number"].replace("-", "")
        url       = (f"{SEC_EDGAR_ARCHIVES}/{int(cik)}/{acc_clean}/"
                     f"{filing['primary_doc']}") if filing["primary_doc"] else ""

        base_row.update({
            "form_type":        filing["form_type"],
            "filing_date":      filing["filing_date"],
            "report_date":      filing["report_date"],
            "accession_number": filing["accession_number"],
            "primary_doc":      filing["primary_doc"],
            "filing_url":       url,
            "fetch_status":     "found",
            "failure_reason":   "",
        })
        rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(FILING_LOOKUP_PATH, index=False)
    found_n = (df["fetch_status"] == "found").sum()
    if verbose:
        print(f"\n  Filing lookup: {found_n}/{len(df)} found → filing_lookup_results.csv")
    return df


def extract_all_features(lookup_df: pd.DataFrame,
                          verbose: bool = True) -> pd.DataFrame:
    """
    For each found filing, download text (cached) and extract risk features.
    Returns features DataFrame saved to filing_risk_features.csv.
    """
    found    = lookup_df[lookup_df["fetch_status"] == "found"].copy()
    not_found = lookup_df[lookup_df["fetch_status"] != "found"].copy()

    feature_rows = []
    total = len(found)

    for i, (_, row) in enumerate(found.iterrows(), 1):
        ticker   = str(row["ticker"])
        year     = int(row["year"])
        cik      = str(row["cik"])
        cname    = str(row.get("company_name", ticker))
        acc      = str(row.get("accession_number", ""))
        pdoc     = str(row.get("primary_doc", ""))
        ftype    = str(row.get("form_type", "10-K"))
        fdate    = str(row.get("filing_date", ""))

        if verbose:
            cached_tag = " [cached]" if _is_text_cached(ticker, year) else ""
            print(f"  [{i:3d}/{total}] {ticker} {year}{cached_tag} …", end=" ", flush=True)

        try:
            text, status = fetch_filing_text(cik, ticker, year, acc, pdoc,
                                              verbose=False)
        except Exception as exc:
            text, status = "", str(exc)

        if not text or len(text) < 200:
            feat = {
                "ticker":             ticker,
                "company_name":       cname,
                "cik":                cik,
                "year":               year,
                "form_type":          ftype,
                "filing_date":        fdate,
                "total_words":        0,
                "risk_word_count":    0,
                "risk_words_per_1000": 0.0,
                "debt_mentions":       0,
                "liquidity_mentions":  0,
                "legal_mentions":      0,
                "regulatory_mentions": 0,
                "uncertainty_mentions": 0,
                "cost_pressure_mentions": 0,
                "cybersecurity_mentions": 0,
                "data_quality_flag":  "parse_error",
                "failure_reason":     f"text fetch/parse failed: {status}",
            }
            if verbose:
                print(f"parse_error ({status})")
        else:
            feat = extract_risk_features(text, ticker, cname, cik, year, ftype, fdate)
            if verbose:
                print(f"ok  {feat['total_words']:,} words  "
                      f"risk={feat['risk_words_per_1000']:.1f}/1k")

        feature_rows.append(feat)

    # Add placeholder rows for filings not found
    for _, row in not_found.iterrows():
        feature_rows.append({
            "ticker":             str(row["ticker"]),
            "company_name":       str(row.get("company_name", "")),
            "cik":                str(row.get("cik", "")),
            "year":               int(row["year"]),
            "form_type":          "",
            "filing_date":        "",
            "total_words":        0,
            "risk_word_count":    0,
            "risk_words_per_1000": 0.0,
            "debt_mentions":       0,
            "liquidity_mentions":  0,
            "legal_mentions":      0,
            "regulatory_mentions": 0,
            "uncertainty_mentions": 0,
            "cost_pressure_mentions": 0,
            "cybersecurity_mentions": 0,
            "data_quality_flag":  "filing_missing",
            "failure_reason":     str(row.get("failure_reason", "not found")),
        })

    df = pd.DataFrame(feature_rows)
    df.to_csv(RISK_FEATURES_PATH, index=False)
    ok_n = (df["data_quality_flag"] == "ok").sum()
    if verbose:
        print(f"\n  Features extracted: {ok_n}/{len(df)} ok → filing_risk_features.csv")
    return df


def build_risk_scores(features_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute 0–100 risk scores. Save filing_risk_scores.csv.
    """
    scored = compute_risk_scores(features_df)
    out = scored[[
        "ticker", "year",
        "report_risk_score_real", "risk_score_source",
        "risk_score_components",
        "filing_data_quality_flag", "filing_warning_text",
    ]].copy()
    out.to_csv(RISK_SCORES_PATH, index=False)
    real_n = out["report_risk_score_real"].notna().sum()
    if verbose:
        print(f"\n  Risk scores: {real_n}/{len(out)} rows have real scores "
              f"→ filing_risk_scores.csv")
    return out


# ── Signal risk adjustment ────────────────────────────────────────────────────

def _risk_adjust_signal(calib_sig: str, risk_score: float,
                          quality: float, has_real_risk: bool) -> tuple:
    """
    Returns (adjusted_signal, reason).
    Downgrades 'Research candidate' if filing risk is very high.
    Never upgrades; never adds buy/sell labels.
    """
    if not has_real_risk or pd.isna(risk_score):
        return calib_sig, "no_real_filing_risk_data"

    risk_score = float(risk_score)
    quality    = float(quality) if pd.notna(quality) else 50.0

    if risk_score >= 80 and calib_sig == "Research candidate":
        if quality < 45:
            return "Possible value trap", (
                f"Risk-adjusted: very high filing risk ({risk_score:.0f}/100) "
                "combined with low quality score suggests risk may outweigh apparent value."
            )
        return "Fairly valued / neutral", (
            f"Risk-adjusted: very high filing risk ({risk_score:.0f}/100). "
            "Downgraded from Research candidate — research the risk language before acting."
        )

    if risk_score >= 90 and calib_sig == "Fairly valued / neutral":
        return "Fairly valued / neutral", (
            f"Caution: extreme filing risk ({risk_score:.0f}/100). "
            "Review risk factors section before further research."
        )

    return calib_sig, ""


# ── Merge into model outputs ───────────────────────────────────────────────────

def merge_risk_into_outputs(risk_scores_df: pd.DataFrame,
                              calibrated_df: pd.DataFrame,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Merge filing risk scores into calibrated model outputs.
    Creates model_outputs_combined_calibrated_risk.csv.
    """
    df = calibrated_df.copy()

    # Preserve original report_risk_score
    if "report_risk_score" in df.columns:
        df["report_risk_score_original"] = df["report_risk_score"]

    # Merge by ticker + year
    risk_slim = risk_scores_df[["ticker", "year",
                                  "report_risk_score_real", "risk_score_source",
                                  "risk_score_components",
                                  "filing_data_quality_flag", "filing_warning_text"]].copy()
    risk_slim["ticker"] = risk_slim["ticker"].str.upper()
    df["ticker"]        = df["ticker"].str.upper()

    df = df.merge(risk_slim, on=["ticker", "year"], how="left")

    # Flag whether real risk is available
    df["report_risk_available"] = df["risk_score_source"].notna() & (
        df["risk_score_source"] == "real_sec_filing"
    )

    # Apply risk-adjusted signal
    def _row_adjust(row):
        calib_sig = str(row.get("final_signal_calibrated", row.get("final_signal", "")))
        risk_score = row.get("report_risk_score_real")
        quality    = row.get("quality_score", 50)
        has_real   = bool(row.get("report_risk_available", False))

        # Skip needs_review rows
        if "needs_review" in str(row.get("output_quality_flag", "")):
            return calib_sig, "needs_review_flag_preserved"

        return _risk_adjust_signal(calib_sig, risk_score, quality, has_real)

    adjusted = df.apply(_row_adjust, axis=1, result_type="expand")
    df["final_signal_calibrated_risk_adjusted"] = adjusted[0]
    df["risk_adjustment_reason"]                 = adjusted[1]

    df.to_csv(RISK_OUTPUTS_PATH, index=False)
    real_n    = df["report_risk_available"].sum()
    adjust_n  = (df["risk_adjustment_reason"].notna() &
                 (df["risk_adjustment_reason"] != "") &
                 (df["risk_adjustment_reason"] != "no_real_filing_risk_data") &
                 (df["risk_adjustment_reason"] != "needs_review_flag_preserved")).sum()
    if verbose:
        print(f"\n  Merged: {real_n}/{len(df)} rows with real risk scores")
        print(f"  Signal adjustments made: {adjust_n} rows")
        print(f"  Saved → model_outputs_combined_calibrated_risk.csv")
    return df


# ── Diagnostics ───────────────────────────────────────────────────────────────

def get_filing_risk_diagnostics() -> dict:
    """
    Return summary stats for Model Diagnostics page.
    All values are safe — no exceptions propagate.
    """
    d = {
        "lookup_exists":         False,
        "lookup_rows":           0,
        "filings_found":         0,
        "filings_parsed":        0,
        "rows_with_real_score":  0,
        "rows_neutral":          0,
        "years_covered":         [],
        "risk_outputs_exists":   False,
        "top_risk_rows":         [],
        "common_failures":       [],
    }
    try:
        if os.path.exists(FILING_LOOKUP_PATH):
            lu = pd.read_csv(FILING_LOOKUP_PATH)
            d["lookup_exists"]  = True
            d["lookup_rows"]    = len(lu)
            d["filings_found"]  = int((lu["fetch_status"] == "found").sum())
            d["common_failures"] = (
                lu[lu["fetch_status"] != "found"]["failure_reason"]
                .value_counts().head(5).to_dict()
            )
    except Exception:
        pass

    try:
        if os.path.exists(RISK_FEATURES_PATH):
            rf = pd.read_csv(RISK_FEATURES_PATH)
            d["filings_parsed"] = int((rf["data_quality_flag"] == "ok").sum())
    except Exception:
        pass

    try:
        if os.path.exists(RISK_OUTPUTS_PATH):
            ro = pd.read_csv(RISK_OUTPUTS_PATH)
            d["risk_outputs_exists"]  = True
            d["rows_with_real_score"] = int(ro["report_risk_available"].sum()
                                            if "report_risk_available" in ro.columns else 0)
            d["rows_neutral"]         = int(len(ro) - d["rows_with_real_score"])
            if "year" in ro.columns:
                d["years_covered"] = sorted(
                    ro[ro.get("report_risk_available", False) == True]["year"]
                    .dropna().astype(int).unique().tolist()
                    if "report_risk_available" in ro.columns else []
                )
            if "report_risk_score_real" in ro.columns:
                top = ro.nlargest(10, "report_risk_score_real")[
                    ["ticker", "year", "report_risk_score_real",
                     "final_signal_calibrated_risk_adjusted"]
                ] if "final_signal_calibrated_risk_adjusted" in ro.columns else (
                    ro.nlargest(10, "report_risk_score_real")[["ticker", "year",
                                                                "report_risk_score_real"]]
                )
                d["top_risk_rows"] = top.to_dict("records")
    except Exception:
        pass

    return d


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_filing_risk_pipeline(verbose: bool = True, run_10q: bool = False) -> None:
    """
    Full pipeline:
      1. Load ticker/CIK mapping
      2. Build filing lookup (find 10-K for each ticker+year)
      3. Download and extract filing text (cached)
      4. Compute risk features
      5. Compute risk scores
      6. Merge into calibrated model outputs
      7. (Optional) Run 10-Q quarterly risk update pipeline if run_10q=True
    """
    print("=" * 65)
    print("SEC Filing Risk Extraction Pipeline")
    print("=" * 65)
    print(f"User-Agent: {USER_AGENT}")
    os.makedirs(FILING_INDEX_CACHE_DIR, exist_ok=True)
    os.makedirs(FILING_TEXT_CACHE_DIR,  exist_ok=True)

    # ── 1. Load CIK mapping ───────────────────────────────────────────────────
    print("\n── Step 1: Ticker/CIK mapping ──")
    ticker_cik_df, msg = load_ticker_cik_mapping()
    print(f"  {msg}")
    if ticker_cik_df is None:
        print("  Cannot continue without CIK mapping. Run modern_fundamentals_utils.py first.")
        return

    # ── 2. Load calibrated model outputs ─────────────────────────────────────
    print("\n── Step 2: Load calibrated model outputs ──")
    if not os.path.exists(CALIBRATED_PATH):
        print(f"  ERROR: {CALIBRATED_PATH} not found. Run calibrate_signals.py first.")
        return
    calibrated_df = pd.read_csv(CALIBRATED_PATH)
    calibrated_df["ticker"] = calibrated_df["ticker"].str.upper()
    print(f"  Loaded {len(calibrated_df)} rows, "
          f"{calibrated_df['ticker'].nunique()} tickers, "
          f"years {sorted(calibrated_df['year'].dropna().astype(int).unique().tolist())}")

    # ── 3. Filing lookup ──────────────────────────────────────────────────────
    print("\n── Step 3: Filing lookup (SEC EDGAR submissions) ──")
    if os.path.exists(FILING_LOOKUP_PATH):
        print("  filing_lookup_results.csv found — loading cache …")
        lookup_df = pd.read_csv(FILING_LOOKUP_PATH)
        # Re-run lookup only for missing rows
        already_looked = set(
            zip(lookup_df["ticker"].str.upper(), lookup_df["year"].astype(int))
        )
        new_pairs = calibrated_df[["ticker", "year"]].drop_duplicates()
        missing_pairs = new_pairs[
            ~new_pairs.apply(
                lambda r: (r["ticker"].upper(), int(r["year"])) in already_looked, axis=1
            )
        ]
        if len(missing_pairs) > 0:
            print(f"  {len(missing_pairs)} new ticker-year pairs — fetching …")
            new_lookup = build_filing_lookup(
                calibrated_df[calibrated_df.apply(
                    lambda r: (r["ticker"].upper(), int(r["year"])) not in already_looked,
                    axis=1
                )],
                ticker_cik_df, verbose=verbose,
            )
            lookup_df = pd.concat([lookup_df, new_lookup], ignore_index=True)
            lookup_df.to_csv(FILING_LOOKUP_PATH, index=False)
        else:
            found_n = (lookup_df["fetch_status"] == "found").sum()
            print(f"  {found_n}/{len(lookup_df)} filings found (from cache).")
    else:
        lookup_df = build_filing_lookup(calibrated_df, ticker_cik_df, verbose=verbose)

    # ── 4. Extract risk features ──────────────────────────────────────────────
    print("\n── Step 4: Filing text fetch + risk feature extraction ──")
    if os.path.exists(RISK_FEATURES_PATH):
        print("  filing_risk_features.csv found — loading + checking for new entries …")
        existing_feats = pd.read_csv(RISK_FEATURES_PATH)
        already_done = set(
            zip(existing_feats["ticker"].str.upper(), existing_feats["year"].astype(int))
        )
        new_lookup = lookup_df[
            lookup_df.apply(
                lambda r: (str(r["ticker"]).upper(), int(r["year"])) not in already_done,
                axis=1,
            )
        ]
        if len(new_lookup) > 0:
            print(f"  {len(new_lookup)} new entries — extracting …")
            new_feats = extract_all_features(new_lookup, verbose=verbose)
            features_df = pd.concat([existing_feats, new_feats], ignore_index=True)
            features_df.to_csv(RISK_FEATURES_PATH, index=False)
        else:
            features_df = existing_feats
            ok_n = (features_df["data_quality_flag"] == "ok").sum()
            print(f"  {ok_n}/{len(features_df)} ok (from cache).")
    else:
        features_df = extract_all_features(lookup_df, verbose=verbose)

    # ── 5. Compute risk scores ────────────────────────────────────────────────
    print("\n── Step 5: Compute risk scores ──")
    risk_scores_df = build_risk_scores(features_df, verbose=verbose)

    # ── 6. Merge into model outputs ───────────────────────────────────────────
    print("\n── Step 6: Merge risk scores into calibrated model outputs ──")
    merge_risk_into_outputs(risk_scores_df, calibrated_df, verbose=verbose)

    print("\n" + "=" * 65)
    print("Pipeline complete.")
    print(f"  Next step: run the Streamlit app to see filing-risk adjusted signals.")
    print("=" * 65)

    if run_10q:
        run_10q_risk_pipeline(verbose=verbose)


# ── 10-Q quarterly risk update pipeline ───────────────────────────────────────

def find_latest_10q_filing(submissions: dict) -> dict | None:
    """
    Find the most recent 10-Q filing in SEC EDGAR submissions.

    Returns a dict with keys: accession_number, form_type, filing_date,
    report_date, primary_doc — or None if not found.
    """
    try:
        recent  = submissions.get("filings", {}).get("recent", {})
        filings = _parse_filings_recent(recent)
        if filings.empty:
            return None

        quarterly_forms = {"10-Q"}
        if "form" not in filings.columns:
            return None

        mask_form = filings["form"].isin(quarterly_forms)
        quarterly = filings[mask_form].copy()
        if quarterly.empty:
            return None

        # Sort by most recently filed first
        if "filingDate" in quarterly.columns:
            quarterly = quarterly.sort_values("filingDate", ascending=False)

        row = quarterly.iloc[0]
        return {
            "accession_number": str(row.get("accession_number", "")),
            "form_type":        str(row.get("form", "10-Q")),
            "filing_date":      str(row.get("filingDate", ""))[:10],
            "report_date":      str(row.get("reportDate", ""))[:10] if "reportDate" in row else "",
            "primary_doc":      str(row.get("primaryDocument", "")),
        }
    except Exception:
        return None


def _10q_text_cache_path(ticker: str) -> str:
    """Return the cache file path for a ticker's latest 10-Q text."""
    os.makedirs(FILING_10Q_TEXT_CACHE_DIR, exist_ok=True)
    return os.path.join(FILING_10Q_TEXT_CACHE_DIR, f"{ticker.upper()}_10Q.txt")


def _is_10q_cached(ticker: str) -> bool:
    """Return True if a valid 10-Q text cache exists for the ticker."""
    path = _10q_text_cache_path(ticker)
    return os.path.exists(path) and os.path.getsize(path) > 100


def _save_10q_text_cache(ticker: str, text: str) -> None:
    """Save 10-Q plain text to the cache file (capped at MAX_TEXT_CHARS)."""
    path = _10q_text_cache_path(ticker)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text[:MAX_TEXT_CHARS])
    except Exception:
        pass


def _load_10q_text_cache(ticker: str) -> str | None:
    """Load and return cached 10-Q text, or None if missing/unreadable."""
    path = _10q_text_cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def run_10q_risk_pipeline(verbose: bool = True) -> None:
    """
    10-Q quarterly risk update pipeline.

    For each ticker in the model universe:
      1. Fetch SEC submissions (cached)
      2. Find the most recent 10-Q filing
      3. Download and cache the 10-Q text
      4. Extract risk features
      5. Compute 0-100 risk scores
      6. Save latest_10q_risk_scores.csv
      7. Merge with annual 10-K scores to produce filing_risk_current_update.csv
         with filing_risk_delta and filing_risk_trend columns.
    """
    print("=" * 65)
    print("10-Q Quarterly Risk Update Pipeline")
    print("=" * 65)
    os.makedirs(FILING_10Q_TEXT_CACHE_DIR, exist_ok=True)

    # ── 1. Load CIK mapping ───────────────────────────────────────────────────
    print("\n── Step 1: Ticker/CIK mapping ──")
    ticker_cik_df, msg = load_ticker_cik_mapping()
    print(f"  {msg}")
    if ticker_cik_df is None:
        print("  Cannot continue without CIK mapping. Run modern_fundamentals_utils.py first.")
        return

    # ── 2. Load existing 10-K risk scores for ticker list ────────────────────
    print("\n── Step 2: Load annual 10-K risk scores ──")
    annual_risk_df = None
    if os.path.exists(RISK_SCORES_PATH):
        try:
            annual_risk_df = pd.read_csv(RISK_SCORES_PATH)
            print(f"  Loaded {len(annual_risk_df)} rows from filing_risk_scores.csv")
        except Exception as exc:
            print(f"  Warning: could not load filing_risk_scores.csv: {exc}")
    else:
        print("  filing_risk_scores.csv not found — delta/trend will be unavailable.")

    # Build per-ticker annual score and filing date (most recent year with a real score)
    annual_by_ticker:        dict = {}
    annual_date_by_ticker:   dict = {}
    annual_year_by_ticker:   dict = {}
    annual_cname_by_ticker:  dict = {}
    if annual_risk_df is not None and "ticker" in annual_risk_df.columns:
        for ticker, grp in annual_risk_df.groupby("ticker"):
            real_rows = grp[grp.get("risk_score_source", pd.Series(dtype=str)) == "real_sec_filing"] \
                if "risk_score_source" in grp.columns \
                else grp[grp["report_risk_score_real"].notna()] \
                if "report_risk_score_real" in grp.columns else pd.DataFrame()
            if not real_rows.empty:
                if "year" in real_rows.columns:
                    latest_row = real_rows.sort_values("year", ascending=False).iloc[0]
                else:
                    latest_row = real_rows.iloc[-1]
                score = latest_row.get("report_risk_score_real")
                if pd.notna(score):
                    _tu = str(ticker).upper()
                    annual_by_ticker[_tu]       = float(score)
                    annual_date_by_ticker[_tu]  = str(latest_row.get("filing_date", "") or "")
                    annual_year_by_ticker[_tu]  = int(latest_row["year"]) if "year" in latest_row and pd.notna(latest_row.get("year")) else None
                    annual_cname_by_ticker[_tu] = str(latest_row.get("company_name", "") or "")

    # ── 3. Process each ticker ────────────────────────────────────────────────
    print("\n── Step 3: Fetch 10-Q filings and extract risk features ──")
    cik_map  = ticker_cik_df.set_index("ticker")["cik"].to_dict()
    name_map = {}
    if "company_name" in ticker_cik_df.columns:
        name_map = ticker_cik_df.set_index("ticker")["company_name"].to_dict()

    tickers = sorted(ticker_cik_df["ticker"].str.upper().unique().tolist())
    feature_rows = []
    meta_rows    = []   # filing metadata per ticker

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip,deflate"}

    for i, ticker in enumerate(tickers, 1):
        cik   = cik_map.get(ticker, cik_map.get(ticker.lower()))
        cname = name_map.get(ticker, ticker)

        base_meta = {
            "ticker":                  ticker,
            "company_name":            cname,
            "cik":                     str(cik) if cik else "",
            "latest_10q_filing_date":  "",
            "latest_10q_period":       "",
            "latest_10q_form_type":    "",
            "latest_10q_data_quality_flag": "cik_missing",
            "latest_10q_warning_text": "",
        }

        if verbose:
            cached_tag = " [cached]" if _is_10q_cached(ticker) else ""
            print(f"  [{i:3d}/{len(tickers)}] {ticker}{cached_tag} …", end=" ", flush=True)

        if not cik:
            base_meta["latest_10q_warning_text"] = f"No CIK for {ticker}"
            meta_rows.append(base_meta)
            if verbose:
                print("no CIK")
            continue

        # Fetch submissions (cached)
        try:
            sub, src, sub_msg = fetch_company_submissions(str(cik))
        except Exception as exc:
            sub, src, sub_msg = None, "error", str(exc)

        if sub is None:
            base_meta["latest_10q_data_quality_flag"] = "submissions_error"
            base_meta["latest_10q_warning_text"]      = sub_msg
            meta_rows.append(base_meta)
            if verbose:
                print(f"submissions error [{sub_msg}]")
            continue

        # Find latest 10-Q
        filing = find_latest_10q_filing(sub)
        if filing is None:
            base_meta["latest_10q_data_quality_flag"] = "not_found"
            base_meta["latest_10q_warning_text"]      = "No 10-Q found in submissions"
            meta_rows.append(base_meta)
            if verbose:
                print("not found")
            continue

        base_meta["latest_10q_filing_date"] = filing["filing_date"]
        base_meta["latest_10q_period"]      = filing["report_date"]
        base_meta["latest_10q_form_type"]   = filing["form_type"]

        # Fetch text (from cache or download)
        text = None
        fetch_status = ""
        if _is_10q_cached(ticker):
            text = _load_10q_text_cache(ticker)
            fetch_status = "cache"

        if not text:
            acc      = filing.get("accession_number", "")
            pdoc     = filing.get("primary_doc", "")
            if not acc or not pdoc:
                base_meta["latest_10q_data_quality_flag"] = "missing_accession_or_doc"
                base_meta["latest_10q_warning_text"]      = "No accession/doc in filing metadata"
                meta_rows.append(base_meta)
                if verbose:
                    print("missing accession/doc")
                continue

            try:
                acc_clean = acc.replace("-", "")
                url       = f"{SEC_EDGAR_ARCHIVES}/{int(cik)}/{acc_clean}/{pdoc}"
                raw       = _http_get(url, headers, max_bytes=MAX_DOWNLOAD_BYTES)
                time.sleep(SEC_RATE_LIMIT_SLEEP)

                if raw is None:
                    base_meta["latest_10q_data_quality_flag"] = "fetch_failed"
                    base_meta["latest_10q_warning_text"]      = f"Failed to download {url}"
                    meta_rows.append(base_meta)
                    if verbose:
                        print("fetch failed")
                    continue

                content_start = raw[:1000].lower()
                if (b"<html" in content_start or b"<!doctype" in content_start
                        or b"<body" in content_start):
                    text = _html_to_text(raw)
                else:
                    try:
                        text = raw.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]
                    except Exception:
                        text = raw.decode("latin-1", errors="replace")[:MAX_TEXT_CHARS]

                if text and len(text) >= 200:
                    _save_10q_text_cache(ticker, text)
                    fetch_status = "downloaded"
                else:
                    text = None
                    fetch_status = "too_short"
            except Exception as exc:
                base_meta["latest_10q_data_quality_flag"] = "parse_error"
                base_meta["latest_10q_warning_text"]      = str(exc)
                meta_rows.append(base_meta)
                if verbose:
                    print(f"error: {exc}")
                continue

        if not text or len(text) < 200:
            base_meta["latest_10q_data_quality_flag"] = "parse_error"
            base_meta["latest_10q_warning_text"]      = f"text too short after fetch ({fetch_status})"
            meta_rows.append(base_meta)
            if verbose:
                print(f"parse error ({fetch_status})")
            continue

        # Extract risk features
        try:
            feat = extract_risk_features(
                text, ticker, cname, str(cik),
                year=0, form_type="10-Q",
                filing_date=filing["filing_date"],
            )
        except Exception as exc:
            feat = None
            base_meta["latest_10q_data_quality_flag"] = "feature_error"
            base_meta["latest_10q_warning_text"]      = str(exc)
            meta_rows.append(base_meta)
            if verbose:
                print(f"feature error: {exc}")
            continue

        base_meta["latest_10q_data_quality_flag"] = feat.get("data_quality_flag", "ok")
        base_meta["latest_10q_warning_text"]      = feat.get("failure_reason", "")
        meta_rows.append(base_meta)
        feature_rows.append(feat)

        if verbose:
            print(f"ok  [{fetch_status}]  {feat.get('total_words', 0):,} words  "
                  f"risk={feat.get('risk_words_per_1000', 0):.1f}/1k")

    # ── 4. Compute 10-Q risk scores ───────────────────────────────────────────
    print("\n── Step 4: Compute 10-Q risk scores ──")
    if feature_rows:
        feat_df   = pd.DataFrame(feature_rows)
        scored_df = compute_risk_scores(feat_df)
        # Build a quick lookup: ticker → score row
        scored_by_ticker: dict = {}
        for _, srow in scored_df.iterrows():
            t = str(srow.get("ticker", "")).upper()
            scored_by_ticker[t] = srow
        ok_n = (scored_df["data_quality_flag"] == "ok").sum()
        print(f"  10-Q scores computed: {ok_n}/{len(scored_df)} ok")
    else:
        scored_by_ticker = {}
        print("  No 10-Q features to score.")

    # ── 5. Save latest_10q_risk_scores.csv ────────────────────────────────────
    print("\n── Step 5: Save latest_10q_risk_scores.csv ──")
    out_rows = []
    for meta in meta_rows:
        ticker = str(meta["ticker"]).upper()
        srow   = scored_by_ticker.get(ticker)
        def _int_safe(s, k):
            return int(s[k]) if s is not None and k in s and pd.notna(s.get(k)) else None
        def _float_safe(s, k):
            return float(s[k]) if s is not None and k in s and pd.notna(s.get(k)) else None
        out_row = {
            "ticker":                               ticker,
            "company_name":                         meta.get("company_name", ""),
            "cik":                                  meta.get("cik", ""),
            "latest_10q_filing_date":               meta.get("latest_10q_filing_date", ""),
            "latest_10q_period":                    meta.get("latest_10q_period", ""),
            "latest_10q_form_type":                 meta.get("latest_10q_form_type", ""),
            "latest_10q_total_words":               _int_safe(srow, "total_words"),
            "latest_10q_risk_word_count":           _int_safe(srow, "risk_word_count"),
            "latest_10q_risk_score":                _float_safe(srow, "report_risk_score_real"),
            "latest_10q_risk_words_per_1000":       _float_safe(srow, "risk_words_per_1000"),
            "latest_10q_debt_liquidity_mentions":   _int_safe(srow, "debt_mentions"),
            "latest_10q_legal_regulatory_mentions": _int_safe(srow, "legal_mentions"),
            "latest_10q_uncertainty_mentions":      _int_safe(srow, "uncertainty_mentions"),
            "latest_10q_cost_cyber_mentions":       (
                (_int_safe(srow, "cost_pressure_mentions") or 0) +
                (_int_safe(srow, "cybersecurity_mentions") or 0)
            ) if srow is not None else None,
            "latest_10q_data_quality_flag":         meta.get("latest_10q_data_quality_flag", ""),
            "latest_10q_warning_text":              meta.get("latest_10q_warning_text", ""),
        }
        out_rows.append(out_row)

    try:
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(LATEST_10Q_RISK_PATH, index=False)
        ok_n = (out_df["latest_10q_data_quality_flag"] == "ok").sum()
        print(f"  Saved {len(out_df)} rows ({ok_n} ok) → latest_10q_risk_scores.csv")
    except Exception as exc:
        print(f"  ERROR saving latest_10q_risk_scores.csv: {exc}")
        out_df = pd.DataFrame(out_rows)

    # ── 6 & 7. Merge with annual 10-K scores; compute delta/trend ─────────────
    print("\n── Step 6/7: Merge with annual 10-K scores → filing_risk_current_update.csv ──")
    update_rows = []
    for _, row in out_df.iterrows():
        ticker          = str(row["ticker"]).upper()
        q_score         = row.get("latest_10q_risk_score")
        q_score_valid   = (q_score is not None) and pd.notna(q_score)
        k_score         = annual_by_ticker.get(ticker)
        k_score_valid   = (k_score is not None) and pd.notna(k_score)

        if q_score_valid and k_score_valid:
            delta = round(float(q_score) - float(k_score), 1)
            if delta >= 10:
                trend = "Increasing"
            elif delta <= -10:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            delta = None
            trend = "Unavailable"

        _warn = ""
        if q_score_valid:
            tw = row.get("latest_10q_total_words")
            if tw is not None and pd.notna(tw) and int(tw) < 500:
                _warn = "10-Q text unusually short; score may be inflated."
        if not _warn:
            _warn = str(row.get("latest_10q_warning_text") or "")
        update_rows.append({
            "ticker":                    ticker,
            "company_name":              annual_cname_by_ticker.get(ticker, row.get("company_name", "")),
            "latest_model_year":         annual_year_by_ticker.get(ticker),
            "annual_10k_risk_score":     float(k_score) if k_score_valid else None,
            "annual_10k_filing_date":    annual_date_by_ticker.get(ticker, ""),
            "latest_10q_risk_score":     float(q_score) if q_score_valid else None,
            "latest_10q_filing_date":    row.get("latest_10q_filing_date", ""),
            "filing_risk_delta":         delta,
            "filing_risk_trend":         trend,
            "current_risk_warning_text": _warn,
        })

    try:
        update_df = pd.DataFrame(update_rows)
        update_df.to_csv(FILING_RISK_UPDATE_PATH, index=False)
        trend_counts = update_df["filing_risk_trend"].value_counts().to_dict()
        summary = ", ".join(f"{k}: {v}" for k, v in trend_counts.items())
        print(f"  Saved {len(update_df)} rows → filing_risk_current_update.csv ({summary})")
    except Exception as exc:
        print(f"  ERROR saving filing_risk_current_update.csv: {exc}")

    print("\n" + "=" * 65)
    print("10-Q pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--10q", dest="run_10q", action="store_true", default=False,
                         help="Also run the 10-Q quarterly risk update pipeline.")
    _parser.add_argument("--10q-only", dest="only_10q", action="store_true", default=False,
                         help="Run only the 10-Q pipeline (skip 10-K if already done).")
    _args = _parser.parse_args()

    if _args.only_10q:
        run_10q_risk_pipeline(verbose=True)
    else:
        run_filing_risk_pipeline(verbose=True, run_10q=_args.run_10q)
