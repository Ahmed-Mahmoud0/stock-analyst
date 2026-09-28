"""Rule-based BUY/HOLD/SELL verdicts. Mechanical model output, not financial advice.

Fundamental
  Fair value = forward EPS x sector median forward P/E (US: median of S&P 500 members in that
  sector; EGX: median of Egyptian stocks in that sector; overall median if < 3 in the sector).
  Checks (score = share passed, unknown data skipped): profitable, forward EPS growth > 5%,
  ROE > 15%, forward P/E below sector median, and if available net margin > 10%,
  revenue growth > 5%, debt/equity < 1.5.
  BUY  : fair value >= +15% above price and score >= 60%   -> target = fair value
  SELL : losing money now and next year, score < 35%,
         or fair value <= -20% below price with score < 60% (expensive AND weak)
  HOLD : everything else
  N/A  : fewer than 3 checks have data

Technical
  Checks: price > SMA200, SMA50 > SMA200, price > SMA50, MACD histogram > 0,
  RSI 50-70, volume above 20D average on an up day.
  BUY  : 5+ of 6 checks and RSI < 75  -> target = 52W high if 1-5 ATR above price,
                                          else price + 3 ATR; stop = price - 2 ATR
  SELL : price < SMA200, SMA50 < SMA200 and MACD histogram < 0
  HOLD : everything else
  N/A  : less than 200 days of price history
"""
import numpy as np
import pandas as pd

BUY, HOLD, SELL, NA = "🟢 BUY", "🟡 HOLD", "🔴 SELL", "⚪ N/A"


def _ok(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def sector_pe(df: pd.DataFrame) -> dict:
    """Median forward P/E per sector among S&P 500 members, or all stocks for non-US markets."""
    bench = df[df["in_sp500"]] if df["in_sp500"].any() else df
    d = bench[bench["fwd_pe"].between(0, 100)]
    g = d.groupby("sector")["fwd_pe"]
    med = g.median()[g.count() >= 3].to_dict()
    med["_all"] = d["fwd_pe"].median()
    return med


def fundamental(s, pe_map: dict) -> dict:
    price, eps_ttm, eps_fwd = s.get("price"), s.get("eps_ttm"), s.get("eps_fwd")
    sector_fpe = pe_map.get(s.get("sector"), pe_map["_all"])
    fair = eps_fwd * sector_fpe if _ok(eps_fwd) and eps_fwd > 0 else np.nan
    upside = (fair / price - 1) * 100 if _ok(fair) and _ok(price) else np.nan

    checks = [
        ("Profitable (trailing EPS > 0)", eps_ttm > 0 if _ok(eps_ttm) else None),
        ("Forward EPS growth > 5%", s["eps_growth"] > 5 if _ok(s.get("eps_growth")) else None),
        ("ROE > 15%", s["roe"] > 15 if _ok(s.get("roe")) else None),
        (f"Forward P/E below sector median ({sector_fpe:.1f})",
         0 < s["fwd_pe"] < sector_fpe if _ok(s.get("fwd_pe")) else None),
        ("Net margin > 10%", s["margin"] > 10 if _ok(s.get("margin")) else None),
        ("Revenue growth > 5%", s["rev_growth"] > 5 if _ok(s.get("rev_growth")) else None),
        ("Debt/equity < 1.5", s["de"] < 1.5 if _ok(s.get("de")) else None),
    ]
    known = [ok for _, ok in checks if ok is not None]
    score = sum(known) / len(known) if known else 0.0

    if len(known) < 3:
        return {"verdict": NA, "target": np.nan, "fair": fair, "upside": upside, "score": score,
                "checks": checks}
    losing = _ok(eps_ttm) and eps_ttm <= 0 and not (_ok(eps_fwd) and eps_fwd > 0)
    if _ok(upside) and upside >= 15 and score >= 0.6:
        verdict = BUY
    elif losing or score < 0.35 or (_ok(upside) and upside <= -20 and score < 0.6):
        verdict = SELL
    else:
        verdict = HOLD
    return {"verdict": verdict, "target": fair if verdict == BUY else np.nan,
            "fair": fair, "upside": upside, "score": score, "checks": checks}


def technical(s) -> dict:
    price, atr, hi = s.get("price"), s.get("atr"), s.get("high_52w")
    rsi, macd = s.get("rsi"), s.get("macd_hist")
    checks = [
        ("Price above SMA200", s.get("vs_sma200", np.nan) > 0),
        ("SMA50 above SMA200", s.get("sma50", np.nan) > s.get("sma200", np.nan)),
        ("Price above SMA50", s.get("vs_sma50", np.nan) > 0),
        ("MACD histogram positive", macd > 0),
        ("RSI between 50 and 70", 50 <= rsi <= 70),
        ("Up day on above-average volume", s.get("rel_vol", np.nan) > 1 and s.get("chg_1d", np.nan) > 0),
    ]
    score = sum(bool(ok) for _, ok in checks)

    target = stop = np.nan
    if not (_ok(s.get("sma200")) and _ok(rsi)):
        verdict = NA
    elif score >= 5 and rsi < 75:
        verdict = BUY
        target = hi if price + atr <= hi <= price + 5 * atr else price + 3 * atr
        stop = price - 2 * atr
    elif s.get("vs_sma200", 0) < 0 and s.get("sma50", 0) < s.get("sma200", 0) and macd < 0:
        verdict = SELL
    else:
        verdict = HOLD
    return {"verdict": verdict, "target": target, "stop": stop, "score": score / len(checks),
            "checks": checks}


def add_to_frame(df: pd.DataFrame, pe_map: dict) -> pd.DataFrame:
    """Add verdict columns to a screener frame."""
    df = df.copy()
    f = [fundamental(r, pe_map) for r in df.to_dict("records")]
    t = [technical(r) for r in df.to_dict("records")]
    df["fund_verdict"] = [x["verdict"] for x in f]
    df["fund_target"] = [x["target"] for x in f]
    df["fund_upside"] = [x["upside"] for x in f]
    df["tech_verdict"] = [x["verdict"] for x in t]
    df["tech_target"] = [x["target"] for x in t]
    df["tech_stop"] = [x["stop"] for x in t]
    return df
