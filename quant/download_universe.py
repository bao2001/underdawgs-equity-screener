"""Download SEC company facts + prices for S&P 500 names not already in the project.
Facts go to quant/sec_facts_cache (separate from the app's cache). Run explicitly.
Validates each CIK against sec.gov/files/company_tickers.json; skips duplicate share classes."""
import os, sys, time, json, requests, pandas as pd, yfinance as yf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
UA = {"User-Agent": "MSDSBA Practicum baoonguyen80@gmail.com"}
CACHE = os.path.join(HERE, "sec_facts_cache"); os.makedirs(CACHE, exist_ok=True)

sp = pd.read_csv(os.path.join(HERE, "sp500_constituents.csv"), dtype={"cik": str})
sp["cik"] = sp.cik.astype(int)
have = pd.read_csv(os.path.join(ROOT, "ticker_cik_mapping.csv"))
have_t = set(have.ticker.str.replace(".", "-", regex=False)); have_c = set(have.cik.astype(int))
sec = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30).json()
sec_map = {v["ticker"].upper(): int(v["cik_str"]) for v in sec.values()}

new = sp[~sp.ticker.isin(have_t) & ~sp.cik.isin(have_c)].drop_duplicates("cik")   # drops 2nd share classes
bad = new[new.apply(lambda r: sec_map.get(r.ticker.upper()) not in (None, r.cik), axis=1)]
print(f"to download: {len(new)}; CIK disagrees with SEC list: {len(bad)} {bad.ticker.tolist()}")
new = new[~new.ticker.isin(bad.ticker)]
ok, fail = [], []
for i, r in enumerate(new.itertuples(), 1):
    p = os.path.join(CACHE, f"CIK{r.cik:010d}.json")
    if os.path.exists(p): ok.append(r.ticker); continue
    try:
        x = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{r.cik:010d}.json", headers=UA, timeout=90)
        if x.status_code != 200: fail.append((r.ticker, x.status_code)); continue
        open(p, "w").write(x.text); ok.append(r.ticker)
    except Exception as e: fail.append((r.ticker, str(e)[:60]))
    if i % 25 == 0: print(f"  {i}/{len(new)}", flush=True)
    time.sleep(0.2)
print("facts ok:", len(ok), "failed:", fail)
pd.DataFrame({"ticker": ok}).merge(sp[["ticker", "company_name", "sector", "cik"]], on="ticker").to_csv(os.path.join(HERE, "universe_extra.csv"), index=False)

px = pd.read_csv(os.path.join(HERE, "prices_daily.csv"), index_col=0, parse_dates=True)
need = [t for t in ok if t not in px.columns]
new_px = yf.download([t.replace(".", "-") for t in need], start="2015-06-01", auto_adjust=True, progress=False, group_by="column", threads=True)["Close"]
new_px.columns = [c.replace("-", ".") for c in new_px.columns]
px = px.join(new_px, how="outer").sort_index()
px.to_csv(os.path.join(HERE, "prices_daily.csv"))
print("prices:", px.shape, "| no price data:", [t for t in need if t not in px.columns or px[t].dropna().empty])
