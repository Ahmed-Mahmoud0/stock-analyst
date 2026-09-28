"""Data fetching and metric calculations (yfinance)."""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # silence 404s for delisted tickers

SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CACHE_DIR = Path(__file__).parent / "cache"
# Egypt: TradingView's public (unofficial) screener endpoint lists EGX stocks with fundamentals;
# Yahoo ('.CA' suffix) supplies prices and fills missing EPS / book value.
TV_EGX_URL = "https://scanner.tradingview.com/egypt/scan"
TV_EGX_COLS = {
    "name": "code", "description": "name", "sector": "sector", "industry": "industry",
    "market_cap_basic": "mcap", "price_earnings_ttm": "pe", "earnings_per_share_diluted_ttm": "eps_ttm",
    "earnings_per_share_forecast_next_fy": "eps_fwd", "return_on_equity": "roe", "net_margin": "margin",
    "total_revenue_yoy_growth_ttm": "rev_growth", "debt_to_equity": "de",
    "dividends_yield_current": "div_yield", "price_book_fq": "pb", "close": "last_close",
}


def sp500_tickers() -> list[str]:
    df = pd.read_csv(SP500_CSV)
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B


# ---------- indicators ----------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd_hist(close: pd.Series) -> pd.Series:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def atr(h: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = h["Close"].shift()
    tr = pd.concat([h["High"] - h["Low"], (h["High"] - prev).abs(), (h["Low"] - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def technicals(h: pd.DataFrame) -> dict:
    """Technical metrics from a daily OHLCV frame (~1y+)."""
    h = h.dropna(subset=["Close"])
    if len(h) < 30:
        return {}
    c, v = h["Close"], h["Volume"]
    last = c.iloc[-1]
    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1]
    avg_vol20 = v.iloc[-21:-1].mean()

    def ret(days):
        return (last / c.iloc[-days - 1] - 1) * 100 if len(c) > days else np.nan

    high_52w = h["High"].iloc[-252:].max()
    return {
        "price": last,
        "chg_1d": ret(1),
        "chg_1m": ret(21),
        "chg_3m": ret(63),
        "chg_1y": ret(252),
        "rsi": rsi(c).iloc[-1],
        "macd_hist": macd_hist(c).iloc[-1],
        "atr": atr(h).iloc[-1],
        "sma50": sma50,
        "sma200": sma200,
        "vs_sma50": (last / sma50 - 1) * 100,
        "vs_sma200": (last / sma200 - 1) * 100,
        "high_52w": high_52w,
        "from_high": (last / high_52w - 1) * 100,
        "volume": v.iloc[-1],
        "avg_vol20": avg_vol20,
        "rel_vol": v.iloc[-1] / avg_vol20 if avg_vol20 else np.nan,
    }


# ---------- fundamentals ----------

def norm_industry(ind):
    """Yahoo uses 'Banks—Regional' in the screener and 'Banks - Regional' in quotes."""
    return ind.replace("—", " - ").replace("–", " - ") if isinstance(ind, str) else ind


def _div(a, b, scale=1.0):
    return a / b * scale if a is not None and b not in (None, 0) else np.nan


def fundamentals(ticker: str) -> dict:
    """Full fundamentals for one ticker (one request per ticker)."""
    i = {}
    for _ in range(2):
        try:
            i = yf.Ticker(ticker).info
            break
        except Exception:
            pass
    price = i.get("currentPrice") or i.get("regularMarketPrice")
    de = i.get("debtToEquity")

    def pct(key):
        val = i.get(key)
        return val * 100 if val is not None else np.nan

    return {
        "ticker": ticker,
        "name": i.get("shortName"),
        "sector": i.get("sector"),
        "industry": i.get("industry"),
        "mcap_b": (i.get("marketCap") or np.nan) / 1e9,
        "pe": i.get("trailingPE", np.nan),
        "fwd_pe": i.get("forwardPE", np.nan),
        "pb": i.get("priceToBook", np.nan),
        "div_yield": _div(i.get("dividendRate") or 0, price, 100),
        "eps_ttm": i.get("trailingEps", np.nan),
        "eps_fwd": i.get("forwardEps", np.nan),
        "margin": pct("profitMargins"),
        "rev_growth": pct("revenueGrowth"),
        "roe": pct("returnOnEquity"),
        "de": de / 100 if de is not None else np.nan,  # Yahoo reports D/E as a percent
        "target": i.get("targetMeanPrice", np.nan),
        "rating": i.get("recommendationKey"),
    }


def history(tickers, period="1y") -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, period=period, group_by="ticker", auto_adjust=True,
                      threads=True, progress=False)
    if isinstance(tickers, str):
        tickers = [tickers]
    if not isinstance(raw.columns, pd.MultiIndex):
        return {tickers[0]: raw}
    return {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}


def tv_egx(codes: list[str] | None = None) -> pd.DataFrame:
    """EGX stocks from TradingView (one request). Market cap is in EGP."""
    flt = [{"left": "type", "operation": "equal", "right": "stock"}]
    if codes:
        flt.append({"left": "name", "operation": "in_range", "right": codes})
    body = {"columns": list(TV_EGX_COLS), "filter": flt, "range": [0, 2000],
            "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"}}
    rows = requests.post(TV_EGX_URL, json=body, timeout=30).json().get("data", [])
    df = pd.DataFrame([r["d"] for r in rows], columns=list(TV_EGX_COLS)).rename(columns=TV_EGX_COLS)
    df = df.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name not in
                  ("code", "name", "sector", "industry") else c)
    df["ticker"] = df["code"] + ".CA"
    df["mcap_b"] = df.pop("mcap") / 1e9
    df["fwd_pe"] = np.where(df["eps_fwd"] > 0, df["last_close"] / df["eps_fwd"], np.nan)
    df["in_sp500"] = False
    return df.drop(columns=["code"]).drop_duplicates("ticker")


def _yahoo_basic(ticker):
    try:
        i = yf.Ticker(ticker).info
    except Exception:
        i = {}
    return {"ticker": ticker, "y_eps": i.get("epsTrailingTwelveMonths") or i.get("trailingEps"),
            "y_bvps": i.get("bookValue"), "y_mcap": i.get("marketCap"), "y_pe": i.get("trailingPE"),
            "y_name": i.get("longName") or i.get("shortName")}


def egx_quotes(codes=None, progress=None) -> pd.DataFrame:
    """EGX fundamentals: TradingView first, gaps filled from Yahoo."""
    tv = tv_egx(codes)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, row in enumerate(pool.map(_yahoo_basic, tv["ticker"]), 1):
            rows.append(row)
            if progress and n % 10 == 0:
                progress(n / len(tv) * 0.3, f"Fundamentals {n}/{len(tv)}…")
    y = pd.DataFrame(rows, columns=["ticker", "y_eps", "y_bvps", "y_mcap", "y_pe", "y_name"])
    df = tv.merge(y.apply(lambda c: pd.to_numeric(c, errors="coerce") if c.name.startswith("y_")
                          and c.name != "y_name" else c), on="ticker", how="left")
    df["eps_ttm"] = df["eps_ttm"].fillna(df["y_eps"])
    df["pe"] = df["pe"].fillna(df["y_pe"])
    df["mcap_b"] = df["mcap_b"].fillna(df["y_mcap"] / 1e9)
    df["name"] = df["name"].fillna(df["y_name"])
    roe_proxy = np.where(df["y_bvps"] > 0, df["eps_ttm"] / df["y_bvps"] * 100, np.nan)
    df["roe"] = df["roe"].fillna(pd.Series(roe_proxy, index=df.index))
    df["rating"] = None
    return df.drop(columns=[c for c in df if c.startswith("y_")] + ["last_close"])


def snapshot(ticker: str, h: pd.DataFrame | None = None) -> dict:
    """Fundamentals + technicals for one ticker."""
    if h is None:
        h = history(ticker, "2y").get(ticker, pd.DataFrame())
    if ticker.endswith(".CA"):
        eg = egx_quotes([ticker.removesuffix(".CA")])
        fund = eg.iloc[0].to_dict() if len(eg) else {"ticker": ticker, "name": None, "sector": None,
                                                    "industry": None, "eps_ttm": np.nan, "eps_fwd": np.nan}
    else:
        fund = fundamentals(ticker)
    s = {**{k: np.nan for k in ("mcap_b", "pe", "fwd_pe", "pb", "div_yield", "margin", "rev_growth",
                                 "roe", "de", "target")}, **fund, **technicals(h)}
    s["eps_growth"] = _div(s["eps_fwd"] - s["eps_ttm"], s["eps_ttm"], 100) if s["eps_ttm"] > 0 else np.nan
    return s


# ---------- screener universes ----------

def _technicals_bulk(tickers, progress, start, chunk=400) -> pd.DataFrame:
    tech = {}
    for n in range(0, len(tickers), chunk):
        if progress:
            progress(start + (1 - start) * n / len(tickers),
                     f"Price history {n}/{len(tickers)}…")
        for t, h in history(tickers[n:n + chunk], "1y").items():
            tech[t] = technicals(h)
    return pd.DataFrame(tech).T.apply(pd.to_numeric, errors="coerce")


def load_universe(name: str, progress=None, force=False) -> pd.DataFrame:
    """'sp500' or 'egx': one row per stock, cached to disk once per day."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}_v2_{date.today()}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    sp = set(sp500_tickers())
    if name == "egx":
        fund = egx_quotes(progress=progress)
        start = 0.3
    else:
        tickers = sorted(sp)
        rows = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, row in enumerate(pool.map(fundamentals, tickers), 1):
                rows.append(row)
                if progress and n % 10 == 0:
                    progress(n / len(tickers) * 0.7, f"Fundamentals {n}/{len(tickers)}…")
        fund = pd.DataFrame(rows)
        start = 0.7

    fund["eps_growth"] = np.where(fund["eps_ttm"] > 0,
                                  (fund["eps_fwd"] / fund["eps_ttm"] - 1) * 100, np.nan)
    fund["industry"] = fund["industry"].map(norm_industry)
    fund["in_sp500"] = name == "sp500"
    tech = _technicals_bulk(fund["ticker"].tolist(), progress, start)
    df = fund.set_index("ticker").join(tech, how="left").reset_index()
    df = df.dropna(subset=["price"])

    df.to_parquet(path)
    for old in CACHE_DIR.glob(f"{name}_*.parquet"):
        if old != path:
            old.unlink()
    return df

