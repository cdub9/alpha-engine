"""Page 6 — Any-symbol price lookup.

Type or pick any ticker we have bars for (including the ~400 Phase C
S&P 500 bars-only tickers that aren't in the LLM-visible universe).
Shows: metadata + price chart since first bar + SPY comparison overlay.

Useful when:
  - The LLM mentions a non-universe ticker in a rationale
  - You want to eyeball a stock before considering adding to universe
  - You want to compare a ticker to SPY over the same window
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import queries as q


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def render() -> None:
    st.title("🔎 Symbol Lookup")
    st.caption(
        "Browse any ticker we have bars for. Includes the LLM-visible "
        "universe (115 active instruments) plus ~400 S&P 500 names "
        "available for data-only inspection."
    )

    symbols = q.all_known_symbols()
    if not symbols:
        st.warning("No symbols in market_bars yet. Run a backfill first.")
        return

    col_pick, col_bench = st.columns([3, 1])
    with col_pick:
        # Default to a familiar name if present, else first
        default_idx = symbols.index("NVDA") if "NVDA" in symbols else 0
        symbol = st.selectbox(
            f"Symbol  ({len(symbols)} available)",
            options=symbols,
            index=default_idx,
        )
    with col_bench:
        benchmark = st.selectbox("Compare to", ["SPY", "QQQ", "IWM", "DIA"], index=0)

    info = q.symbol_info(symbol)
    if info is None:
        st.error(f"No bars found for {symbol}.")
        return

    # Metadata strip
    sector_str = info["sector"] or "—"
    universe_badge = "🟢 in universe" if info["in_universe"] else "⚪ bars only"
    st.markdown(
        f"**{info['name']}** &nbsp;·&nbsp; sector: `{sector_str}` &nbsp;·&nbsp; {universe_badge}"
    )

    # Key metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest price", f"${info['last_px']:,.2f}",
              help="Latest adj_close in market_bars for this symbol.")
    m2.metric(
        f"Total return since {info['first_bar']}",
        _fmt_pct(info["total_return_pct"]),
        help="Simple cumulative return from first bar to latest. Split-adjusted via adj_close.",
    )
    m3.metric("Bars in DB", f"{info['n_bars']:,}",
              help="Daily bar count. ~252 per trading year. Shorter than expected = symbol IPO'd later or has gaps.")
    span_days = (info["last_bar"] - info["first_bar"]).days
    years = span_days / 365.25
    m4.metric("Coverage", f"{years:.1f} years",
              help="Calendar years between first and last bar. ARM and CEG have shorter histories (post-IPO/spinoff).")

    # Price chart with benchmark overlay (both normalized to 100)
    df = q.price_history_with_benchmark(symbol, benchmark=benchmark)
    if df.empty:
        st.warning("No price history rows.")
        return

    plot_df = df.melt(
        id_vars=["date"],
        value_vars=["symbol_normalized", "benchmark_normalized"],
        var_name="series",
        value_name="value",
    )
    label_map = {"symbol_normalized": symbol, "benchmark_normalized": benchmark}
    plot_df["series"] = plot_df["series"].map(label_map)
    plot_df = plot_df.dropna(subset=["value"])

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(
                "value:Q",
                title=f"Normalized (start = 100)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=[symbol, benchmark], range=["#1f77b4", "#888"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Norm", format=".1f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    # Outperformance summary vs benchmark
    bench_first = df["benchmark_price"].dropna()
    if len(bench_first) > 0:
        bench_ret = (
            df["benchmark_price"].iloc[-1] / bench_first.iloc[0] - 1.0
        )
        sym_ret = info["total_return_pct"]
        delta = sym_ret - bench_ret
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{symbol} return", _fmt_pct(sym_ret),
                  help="Cumulative simple return of the symbol over the displayed window.")
        c2.metric(f"{benchmark} return", _fmt_pct(bench_ret),
                  help=f"Cumulative simple return of {benchmark} over the same window.")
        c3.metric(
            f"{symbol} − {benchmark}",
            _fmt_pct(delta),
            delta=f"{delta*100:+.1f} pp",
            help="Outperformance vs benchmark in percentage points. Positive = beat the benchmark.",
        )

    # Technical view: price + 50/200-MA overlays + RSI subplot (the same
    # numbers the LLM sees in the snapshot's "Per-symbol technicals").
    with st.expander("📐 Technical view (50/200-day MA + RSI)", expanded=False):
        from datetime import date as _date, timedelta as _td

        ta = q.price_history_with_technicals(symbol, _date.today() - _td(days=365))
        if ta.empty:
            st.info("Not enough history for the technical view.")
        else:
            ta_base = alt.Chart(ta).encode(x=alt.X("date:T", title=None))
            price_line = ta_base.mark_line(color="#1f77b4").encode(
                y=alt.Y("price:Q", title="Adj close", scale=alt.Scale(zero=False)),
                tooltip=["date:T", alt.Tooltip("price:Q", format="$,.2f")],
            )
            sma50_line = ta_base.mark_line(color="#c98410", strokeWidth=1.2).encode(
                y="sma50:Q",
                tooltip=["date:T", alt.Tooltip("sma50:Q", format="$,.2f", title="50-day MA")],
            )
            sma200_line = ta_base.mark_line(color="#7a4fb0", strokeWidth=1.2).encode(
                y="sma200:Q",
                tooltip=["date:T", alt.Tooltip("sma200:Q", format="$,.2f", title="200-day MA")],
            )
            st.altair_chart(
                alt.layer(price_line, sma50_line, sma200_line).properties(height=300),
                use_container_width=True,
            )
            rsi_line = ta_base.mark_line(color="#1f77b4").encode(
                y=alt.Y("rsi14:Q", title="RSI-14", scale=alt.Scale(domain=[0, 100])),
                tooltip=["date:T", alt.Tooltip("rsi14:Q", format=".0f", title="RSI")],
            )
            rsi_bands = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(
                color="#888", strokeDash=[3, 3]
            ).encode(y="y:Q")
            st.altair_chart(
                alt.layer(rsi_line, rsi_bands).properties(height=100),
                use_container_width=True,
            )
            st.caption(
                "Blue = adj close · orange = 50-day MA · purple = 200-day MA. "
                "RSI above 70 = stretched, below 30 = washed out. Last 12 months."
            )

    # Raw price tail
    with st.expander("Latest 10 bars", expanded=False):
        last10 = df[["date", "symbol_price"]].tail(10).rename(
            columns={"date": "Date", "symbol_price": "Adj close"}
        )
        st.dataframe(
            last10.style.format({"Adj close": "${:,.2f}"}),
            hide_index=True,
            use_container_width=True,
        )
