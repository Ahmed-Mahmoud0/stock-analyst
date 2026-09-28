"""Stock Analyst: US stock screener, single-ticker analysis, comparison."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
import verdicts as vd

st.set_page_config(page_title="Stock Analyst", page_icon="📈", layout="wide")

# Column display config shared by the screener and compare tables.
FMT = {
    "ticker": st.column_config.TextColumn("Ticker"),
    "name": st.column_config.TextColumn("Name"),
    "sector": st.column_config.TextColumn("Sector"),
    "price": st.column_config.NumberColumn("Price", format="%.2f"),
    "fund_verdict": st.column_config.TextColumn("Fund."),
    "fund_target": st.column_config.NumberColumn("Fund. target", format="%.2f"),
    "fund_upside": st.column_config.NumberColumn("Fair value vs price", format="%+.0f%%"),
    "tech_verdict": st.column_config.TextColumn("Tech."),
    "tech_target": st.column_config.NumberColumn("Tech. target", format="%.2f"),
    "tech_stop": st.column_config.NumberColumn("Tech. stop", format="%.2f"),
    "industry": st.column_config.TextColumn("Industry"),
    "chg_1d": st.column_config.NumberColumn("1D %", format="%.1f%%"),
    "chg_1m": st.column_config.NumberColumn("1M %", format="%.1f%%"),
    "chg_3m": st.column_config.NumberColumn("3M %", format="%.1f%%"),
    "chg_1y": st.column_config.NumberColumn("1Y %", format="%.1f%%"),
    "mcap_b": st.column_config.NumberColumn("Mkt cap (B)", format="%.1f"),
    "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
    "fwd_pe": st.column_config.NumberColumn("Fwd P/E", format="%.1f"),
    "pb": st.column_config.NumberColumn("P/B", format="%.1f"),
    "div_yield": st.column_config.NumberColumn("Div %", format="%.2f%%"),
    "margin": st.column_config.NumberColumn("Net margin %", format="%.1f%%"),
    "rev_growth": st.column_config.NumberColumn("Rev growth %", format="%.1f%%"),
    "eps_growth": st.column_config.NumberColumn("Fwd EPS growth %", format="%.1f%%"),
    "roe": st.column_config.NumberColumn("ROE %", format="%.1f%%"),
    "de": st.column_config.NumberColumn("Debt/Eq", format="%.2f"),
    "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
    "vs_sma50": st.column_config.NumberColumn("vs SMA50 %", format="%.1f%%"),
    "vs_sma200": st.column_config.NumberColumn("vs SMA200 %", format="%.1f%%"),
    "from_high": st.column_config.NumberColumn("From 52W high %", format="%.1f%%"),
    "rel_vol": st.column_config.NumberColumn("Vol / Avg", format="%.2fx",
                                             help="Last session volume ÷ 20-day average volume"),
    "avg_vol20": st.column_config.NumberColumn("Avg vol 20D", format="compact"),
}
LABELS = {k: v["label"] for k, v in FMT.items()}
UNIVERSES = {"S&P 500": "sp500", "All US stocks": "us", "Egypt (EGX)": "egx"}
DISCLAIMER = "Mechanical rule outputs (see the explanation at the top), not a recommendation."


@st.cache_data(ttl=3600, show_spinner=False)
def cached_snapshot(ticker):
    h = data.history(ticker, "2y").get(ticker, pd.DataFrame())
    return data.snapshot(ticker, h), h


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def egx_pe_map():
    return vd.sector_pe(data.tv_egx())


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def us_pe_map():
    return vd.sector_pe(data.load_universe("sp500"))


def pe_map_for(ticker):
    """Sector P/E medians for the ticker's own market."""
    if ticker.endswith(".CA"):
        return egx_pe_map()
    for u in ("sp500", "us"):
        if u in st.session_state.get("universes", {}):
            return st.session_state.universes[u][1]
    return us_pe_map()


def cur(ticker):
    return "EGP " if ticker.endswith(".CA") else "$"


def fmt(v, suffix="", digits=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{digits}f}{suffix}"


def show_checks(col, checks):
    for label, ok in checks:
        col.write(("➖ " if ok is None else "✅ " if ok else "❌ ") + label)


# ======================= HEADER =======================
st.title("📈 Stock Analyst")
st.warning("**Not a recommendation.** The BUY / HOLD / SELL verdicts below are calculated "
           "automatically by fixed rules from Yahoo Finance data. They are not financial advice or "
           "investment recommendations. Data can be incomplete or delayed. Do your own research and "
           "consult a licensed advisor before investing.")
with st.expander("How each rating is calculated", expanded=True):
    h1, h2 = st.columns(2)
    with h1.container(border=True):
        st.markdown("""
##### 📘 Fundamental verdict
**Fair value** = forward EPS × median forward P/E of companies in the same sector
(US: S&P 500 members · Egypt: EGX stocks).

**Quality score** = share of these checks passed (missing data skipped): profitable,
forward EPS growth > 5%, ROE > 15%, forward P/E below sector median, net margin > 10%,
revenue growth > 5%, debt/equity < 1.5.

🟢 **BUY**: fair value ≥ 15% above price **and** score ≥ 60%. **Target = fair value.**

🔴 **SELL**: losing money now and next year, **or** score < 35%, **or** fair value ≥ 20%
below price with score < 60%.

🟡 **HOLD**: everything else · ⚪ **N/A**: not enough data.
""")
    with h2.container(border=True):
        st.markdown("""
##### 📈 Technical verdict
**6 checks:** price > 200-day average · 50-day average > 200-day · price > 50-day average ·
MACD histogram > 0 · RSI(14) between 50 and 70 · up day on volume above its 20-day average.

🟢 **BUY**: 5 or more checks pass and RSI < 75. **Target** = 52-week high if it is 1–5 ATR
above the price, otherwise price + 3 ATR. **Stop** = price − 2 ATR
(ATR = average true range, 14 days).

🔴 **SELL**: price below its 200-day average, 50-day below 200-day, and MACD histogram negative.

🟡 **HOLD**: everything else · ⚪ **N/A**: not enough data.
""")

tab_screen, tab_ticker, tab_compare = st.tabs(["🔎 Screener", "📊 Ticker", "⚖️ Compare"])

# ======================= SCREENER =======================
with tab_screen:
    top = st.columns([2, 2, 1])
    uni_label = top[0].radio("Universe", list(UNIVERSES), horizontal=True)
    uni = UNIVERSES[uni_label]
    refresh = top[2].button("↻ Refresh data", width="stretch")

    cache = st.session_state.setdefault("universes", {})
    if uni not in cache or refresh:
        wait = "~6 min" if uni == "us" else "~1–2 min"
        bar = st.progress(0.0, f"Loading {uni_label} ({wait} first time each day)…")
        raw = data.load_universe(uni, progress=lambda p, msg: bar.progress(min(p, 1.0), msg),
                                 force=refresh)
        pe_map = egx_pe_map() if uni == "egx" else vd.sector_pe(raw)
        cache[uni] = (vd.add_to_frame(raw, pe_map), pe_map)
        bar.empty()
    df, pe_map = cache[uni]
    bulk = uni == "us"
    na_note = "Not available in bulk data for All US stocks" if bulk else None

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown("**Verdicts**")
        fund_v = c1.multiselect("Fundamental", [vd.BUY, vd.HOLD, vd.SELL, vd.NA])
        tech_v = c1.multiselect("Technical", [vd.BUY, vd.HOLD, vd.SELL, vd.NA])

        c2.markdown("**Fundamental**")
        sectors = c2.multiselect("Sector", sorted(df["sector"].dropna().unique()))
        mcap_min = c2.number_input("Min market cap (billions)", 0.0, value=0.0, step=1.0)
        pe_max = c2.number_input("Max P/E (0 = any)", 0.0, value=0.0, step=5.0)
        div_min = c2.number_input("Min dividend yield %", 0.0, value=0.0, step=0.5)

        c3.markdown("&nbsp;")
        eps_g_min = c3.number_input("Min fwd EPS growth %", -100.0, value=-100.0, step=5.0)
        roe_min = c3.number_input("Min ROE %", -100.0, value=-100.0, step=5.0)
        growth_min = c3.number_input("Min revenue growth %", -100.0, value=-100.0, step=5.0,
                                     disabled=bulk, help=na_note)
        margin_min = c3.number_input("Min net margin %", -100.0, value=-100.0, step=5.0,
                                     disabled=bulk, help=na_note)
        de_max = c3.number_input("Max debt/equity (0 = any)", 0.0, value=0.0, step=0.5,
                                 disabled=bulk, help=na_note)

        c4.markdown("**Technical**")
        rsi_rng = c4.slider("RSI (14)", 0, 100, (0, 100))
        high_rng = c4.slider("% from 52-week high", -100, 0, (-100, 0))
        ret_rng = c4.slider("3-month return %", -100, 200, (-100, 200))

        c5.markdown("**Volume & trend**")
        relvol_min = c5.number_input("Min volume vs 20D avg (x)", 0.0, value=0.0, step=0.25,
                                     help="1.5 = last session's volume is 50% above its 20-day average")
        avgvol_min = c5.number_input("Min avg daily volume (M shares)", 0.0, value=0.0, step=0.5)
        above50 = c5.checkbox("Price above SMA50")
        above200 = c5.checkbox("Price above SMA200")

    m = pd.Series(True, index=df.index)
    if fund_v:
        m &= df["fund_verdict"].isin(fund_v)
    if tech_v:
        m &= df["tech_verdict"].isin(tech_v)
    if sectors:
        m &= df["sector"].isin(sectors)
    m &= df["mcap_b"].fillna(0) >= mcap_min
    if pe_max:
        m &= df["pe"].between(0, pe_max)
    if div_min:
        m &= df["div_yield"] >= div_min
    if eps_g_min > -100:
        m &= df["eps_growth"] >= eps_g_min
    if roe_min > -100:
        m &= df["roe"] >= roe_min
    if not bulk:
        if growth_min > -100:
            m &= df["rev_growth"] >= growth_min
        if margin_min > -100:
            m &= df["margin"] >= margin_min
        if de_max:
            m &= df["de"] <= de_max
    if rsi_rng != (0, 100):
        m &= df["rsi"].between(*rsi_rng)
    if high_rng != (-100, 0):
        m &= df["from_high"].between(*high_rng)
    if ret_rng != (-100, 200):
        m &= df["chg_3m"].between(*ret_rng)
    if relvol_min:
        m &= df["rel_vol"] >= relvol_min
    if avgvol_min:
        m &= df["avg_vol20"] >= avgvol_min * 1e6
    if above50:
        m &= df["vs_sma50"] > 0
    if above200:
        m &= df["vs_sma200"] > 0

    cols = ["ticker", "name", "sector", "price", "fund_verdict", "fund_target", "fund_upside",
            "tech_verdict", "tech_target", "tech_stop", "chg_1d", "chg_3m", "mcap_b", "pe",
            "fwd_pe", "eps_growth", "roe"]
    if not bulk:
        cols += ["rev_growth", "margin", "de"]
    cols += ["industry", "div_yield", "rsi", "vs_sma50", "vs_sma200", "from_high", "rel_vol", "avg_vol20"]
    res = df.loc[m, cols].sort_values("mcap_b", ascending=False)

    ccy = "EGP" if uni == "egx" else "USD"
    st.caption(f"**{len(res):,}** of {len(df):,} stocks match · prices & market caps in {ccy} · "
               f"click a row to open it in the Ticker tab · {DISCLAIMER}")
    sel = st.dataframe(res, column_config=FMT, hide_index=True, width="stretch",
                       height=560, on_select="rerun", selection_mode="single-row")
    if sel.selection.rows:
        st.session_state.ticker = res.iloc[sel.selection.rows[0]]["ticker"]
        st.success(f"**{st.session_state.ticker}** loaded, so switch to the 📊 Ticker tab.")
    st.download_button("Download CSV", res.to_csv(index=False), "screener.csv", "text/csv")

# ======================= TICKER =======================
with tab_ticker:
    c1, c2 = st.columns([1, 3])
    ticker = c1.text_input("Ticker", st.session_state.get("ticker", "AAPL")).strip().upper()
    view = c2.radio("Chart range", ["3M", "6M", "1Y", "2Y"], index=2, horizontal=True)

    if ticker:
        with st.spinner(f"Loading {ticker}…"):
            s, h = cached_snapshot(ticker)
        if h.empty or "price" not in s:
            st.error(f"No data for {ticker}.")
        else:
            pm, c = pe_map_for(ticker), cur(ticker)
            fv = vd.fundamental(s, pm)
            tv = vd.technical(s)

            st.subheader(f"{s['name'] or ticker} ({ticker})")
            st.caption(f"{s['sector'] or ''} · {s['industry'] or ''}")

            # ---- verdicts ----
            v1, v2 = st.columns(2)
            with v1.container(border=True):
                st.markdown(f"#### Fundamental: {fv['verdict']}")
                k = st.columns(3)
                if fv["verdict"] == vd.BUY:
                    k[0].metric("Target", f"{c}{fv['target']:,.2f}", f"{fv['upside']:+.1f}%")
                else:
                    k[0].metric("Fair value", "—" if np.isnan(fv["fair"]) else f"{c}{fv['fair']:,.2f}",
                                None if np.isnan(fv["upside"]) else f"{fv['upside']:+.1f}%")
                k[1].metric("Score", f"{fv['score']:.0%}")
                k[2].metric("Sector fwd P/E", fmt(pm.get(s["sector"], pm["_all"])))
                show_checks(st, fv["checks"])
            with v2.container(border=True):
                st.markdown(f"#### Technical: {tv['verdict']}")
                k = st.columns(3)
                if tv["verdict"] == vd.BUY:
                    k[0].metric("Target", f"{c}{tv['target']:,.2f}",
                                f"{(tv['target'] / s['price'] - 1) * 100:+.1f}%")
                    k[1].metric("Stop", f"{c}{tv['stop']:,.2f}",
                                f"{(tv['stop'] / s['price'] - 1) * 100:+.1f}%")
                else:
                    k[0].metric("52W high", fmt(s["high_52w"], digits=2))
                    k[1].metric("ATR 14", fmt(s["atr"], digits=2))
                k[2].metric("Score", f"{tv['score']:.0%}")
                show_checks(st, tv["checks"])
            st.caption(DISCLAIMER)

            # ---- key stats ----
            mc = s["mcap_b"]
            mcap = "—" if np.isnan(mc) else (f"{c}{mc / 1000:.2f}T" if mc >= 1000 else f"{c}{mc:.1f}B")
            k = st.columns(6)
            k[0].metric("Price", f"{c}{s['price']:,.2f}", f"{s['chg_1d']:+.2f}%")
            k[1].metric("Market cap", mcap)
            k[2].metric("P/E", fmt(s["pe"]))
            k[3].metric("Forward P/E", fmt(s["fwd_pe"]))
            k[4].metric("RSI 14", fmt(s["rsi"], digits=0))
            k[5].metric("Vol / 20D avg", fmt(s["rel_vol"], "x", 2))

            k = st.columns(6)
            k[0].metric("From 52W high", fmt(s["from_high"], "%"))
            k[1].metric("Revenue growth", fmt(s["rev_growth"], "%"))
            k[2].metric("Net margin", fmt(s["margin"], "%"))
            k[3].metric("ROE", fmt(s["roe"], "%"))
            k[4].metric("Debt / equity", fmt(s["de"], digits=2))
            upside = (s["target"] / s["price"] - 1) * 100 if s["target"] else np.nan
            k[5].metric("Analyst target", fmt(s["target"], digits=2),
                        None if np.isnan(upside) else fmt(upside, "%"))

            # ---- chart: price + SMAs / volume + avg / RSI ----
            h = h.copy()
            h["SMA50"] = h["Close"].rolling(50).mean()
            h["SMA200"] = h["Close"].rolling(200).mean()
            h["AvgVol"] = h["Volume"].rolling(20).mean()
            h["RSI"] = data.rsi(h["Close"])
            days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[view]
            p = h.iloc[-days:]

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                                vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=p.index, open=p["Open"], high=p["High"], low=p["Low"],
                                         close=p["Close"], name="Price"), 1, 1)
            fig.add_trace(go.Scatter(x=p.index, y=p["SMA50"], name="SMA50", line=dict(width=1.5)), 1, 1)
            fig.add_trace(go.Scatter(x=p.index, y=p["SMA200"], name="SMA200", line=dict(width=1.5)), 1, 1)
            if tv["verdict"] == vd.BUY:
                fig.add_hline(y=tv["target"], line_dash="dash", line_color="#26a69a", row=1, col=1,
                              annotation_text=f"Tech target {tv['target']:.2f}")
                fig.add_hline(y=tv["stop"], line_dash="dash", line_color="#ef5350", row=1, col=1,
                              annotation_text=f"Stop {tv['stop']:.2f}")
            up = p["Close"] >= p["Open"]
            fig.add_trace(go.Bar(x=p.index, y=p["Volume"], name="Volume", showlegend=False,
                                 marker_color=np.where(up, "#26a69a", "#ef5350")), 2, 1)
            fig.add_trace(go.Scatter(x=p.index, y=p["AvgVol"], name="Avg vol 20D",
                                     line=dict(width=1.5, color="#888")), 2, 1)
            fig.add_trace(go.Scatter(x=p.index, y=p["RSI"], name="RSI", showlegend=False,
                                     line=dict(width=1.5, color="#7e57c2")), 3, 1)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line_dash="dot", line_color="#999", row=3, col=1)
            fig.update_layout(height=680, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.02, x=0))
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            st.plotly_chart(fig, width="stretch")

# ======================= COMPARE =======================
with tab_compare:
    c1, c2 = st.columns([3, 1])
    raw = c1.text_input("Tickers (comma separated, 2–5)", "AAPL, MSFT, NVDA, GOOGL")
    period = c2.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:5]

    if len(tickers) >= 2:
        with st.spinner("Loading…"):
            snaps = [cached_snapshot(t)[0] for t in tickers]
            hist = data.history(tickers, period)

        fig = go.Figure()
        for t, h in hist.items():
            c = h["Close"].dropna()
            if len(c):
                fig.add_trace(go.Scatter(x=c.index, y=c / c.iloc[0] * 100, name=t))
        fig.update_layout(height=420, yaxis_title="Growth of 100", hovermode="x unified",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        for s in snaps:
            if "price" in s:
                f, t = vd.fundamental(s, pe_map_for(s["ticker"])), vd.technical(s)
                s.update(fund_verdict=f["verdict"], fund_target=f["target"],
                         tech_verdict=t["verdict"], tech_target=t["target"], tech_stop=t["stop"])
        rows = ["fund_verdict", "fund_target", "tech_verdict", "tech_target", "tech_stop",
                "price", "mcap_b", "pe", "fwd_pe", "pb", "rev_growth", "eps_growth", "margin",
                "roe", "de", "div_yield", "chg_1m", "chg_3m", "chg_1y", "rsi", "vs_sma50",
                "vs_sma200", "from_high", "rel_vol"]
        table = pd.DataFrame(snaps).set_index("ticker").reindex(columns=rows).T
        table = table.map(lambda v: v if isinstance(v, str) else fmt(v, digits=2))
        table.index = [LABELS[r] for r in rows]
        st.dataframe(table, width="stretch", height=(len(rows) + 1) * 35 + 3)
        st.caption(DISCLAIMER)
    else:
        st.info("Enter at least 2 tickers.")
