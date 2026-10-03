"""Download daily adjusted prices (yfinance) for the screener universe -> quant/prices_daily.csv.
Run explicitly; the app never calls this."""
import os, sys, pandas as pd, yfinance as yf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
tickers = sorted(pd.read_csv(os.path.join(ROOT, "model_outputs_current.csv"))["ticker"].unique())
tickers += ["SPY"]   # benchmark
px = yf.download([t.replace(".", "-") for t in tickers], start="2015-06-01", auto_adjust=True,
                 progress=False, group_by="column", threads=True)["Close"]
px.columns = [c.replace("-", ".") if c != "BRK-B" else "BRK.B" for c in px.columns]
px = px.dropna(how="all")
px.to_csv(os.path.join(HERE, "prices_daily.csv"))
print(px.shape, px.index.min().date(), px.index.max().date())
print("tickers with no data:", [t for t in tickers if t not in px.columns or px[t].dropna().empty])
