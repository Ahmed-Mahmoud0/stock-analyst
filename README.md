# Stock Analyst

Simple Streamlit stock screener and analyst using free Yahoo Finance data (yfinance).

- **Screener**: S&P 500 or Egypt (EGX), filtered by verdicts, fundamentals and technicals (including volume vs 20-day average). Click a row to send it to the Ticker tab.
- **Egypt (EGX)**: about 210 Egyptian stocks with price history (tickers end in `.CA`, e.g. `COMI.CA`). The stock list and fundamentals come from TradingView's public screener endpoint (unofficial, could change without notice), with gaps filled from Yahoo. Prices are in EGP, and sector medians are calculated from EGX stocks. Earnings data is sparse, so about half the stocks get a fundamental verdict of ⚪ N/A.
- **Ticker**: fundamental and technical verdicts (BUY/HOLD/SELL, with a target on BUY and a stop on technical BUY), key stats, and a chart with SMA50/200, volume vs average, RSI, and target/stop lines.
- **Compare**: 2 to 5 tickers, normalized performance chart, verdicts and metrics side by side.

Verdict rules are documented in [verdicts.py](verdicts.py) and in the header at the top of the app. They are mechanical rule outputs, not financial advice.

```
pip install -r requirements.txt
streamlit run app.py
```

First load each day: about 1–2 min for the S&P 500 or EGX, then cached in `cache/`. **↻ Refresh data** forces a reload.
