"""Page — ML Signals.

The quantitative counterpart to the LLM digest: every active universe
instrument ranked daily by cross-sectional momentum (rule composite) and
a walk-forward-trained XGBoost model. Top quintile = BUY, bottom = AVOID.

Layout:
  1. Today's consensus — names where BOTH models agree (strongest signal)
  2. Action buckets per model — ranked BUY / AVOID tables with features
  3. LLM cross-check — where the digest and the ML ranks agree/conflict
  4. Validation panel — honest walk-forward OOS results vs benchmarks
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import glossary as g
from dashboard import queries as q

ACTION_STYLE = {
    "BUY": ("▲ BUY", "#1f7a3a"),
    "HOLD": ("■ HOLD", "#666666"),
    "AVOID": ("▼ AVOID", "#a02d2d"),
}


def _action_chip(action: str) -> str:
    label, color = ACTION_STYLE.get(action, (action, "#444"))
    return f"<span style='color:{color}; font-weight:700'>{label}</span>"


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Rank"] = df["rank"]
    out["Symbol"] = df["symbol"]
    out["Name"] = df["instrument_name"].fillna("")
    out["Score"] = df["score"].map(lambda x: f"{x:+.2f}")
    out["12-1 mom"] = df["mom_12_1"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    out["3-1 mom"] = df["mom_3_1"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    out["vs 200MA"] = df["dist_200ma"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    out["RSI"] = df["rsi_14"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    out["Vol"] = df["vol_30d"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    return out


def _render_consensus(d) -> None:
    cons = q.ml_model_consensus(d)
    if cons.empty:
        return
    both_buy = cons[(cons["momentum_action"] == "BUY") & (cons["xgb_action"] == "BUY")]
    both_avoid = cons[(cons["momentum_action"] == "AVOID") & (cons["xgb_action"] == "AVOID")]
    split = cons[
        ((cons["momentum_action"] == "BUY") & (cons["xgb_action"] == "AVOID"))
        | ((cons["momentum_action"] == "AVOID") & (cons["xgb_action"] == "BUY"))
    ]

    st.subheader("🤝 Model consensus", help=g.ML_CONSENSUS)
    c1, c2, c3 = st.columns(3)
    c1.metric("Both say BUY", len(both_buy), help=g.ML_CONSENSUS)
    c2.metric("Both say AVOID", len(both_avoid))
    c3.metric("Models split", len(split), help=g.ML_SPLIT)

    if not both_buy.empty:
        st.markdown("**Strongest buys (both models, top quintile each):**")
        for _, r in both_buy.iterrows():
            name = r["instrument_name"] or ""
            st.markdown(
                f"&nbsp;&nbsp;🟢 **{r['symbol']}** {name and f'· {name} '}"
                f"· mom rank **#{r['momentum_rank']}**, xgb rank **#{r['xgb_rank']}** "
                f"of {r['n_universe']} · 12-1 mom {r['mom_12_1']:+.1%} · "
                f"vs 200MA {r['dist_200ma']:+.1%}",
                unsafe_allow_html=True,
            )
    if not both_avoid.empty:
        st.markdown("**Strongest avoids (both models, bottom quintile each):**")
        for _, r in both_avoid.iterrows():
            name = r["instrument_name"] or ""
            st.markdown(
                f"&nbsp;&nbsp;🔴 **{r['symbol']}** {name and f'· {name} '}"
                f"· mom rank #{r['momentum_rank']}, xgb rank #{r['xgb_rank']} "
                f"of {r['n_universe']}",
                unsafe_allow_html=True,
            )
    if not split.empty:
        with st.expander(f"⚖️ Models disagree on {len(split)} name(s)"):
            st.caption(
                "The composite chases 12-month trends; XGBoost has learned "
                "some mean-reversion from the same features. A split read "
                "usually means an extended or washed-out name — treat as "
                "no-signal rather than picking a side."
            )
            for _, r in split.iterrows():
                st.markdown(
                    f"&nbsp;&nbsp;• **{r['symbol']}** — momentum says "
                    f"{r['momentum_action']} (#{r['momentum_rank']}), xgb says "
                    f"{r['xgb_action']} (#{r['xgb_rank']})"
                )


def _render_llm_crosscheck(d) -> None:
    digest_d = q.latest_digest_date()
    if digest_d is None:
        return
    agree = q.ml_llm_agreement(digest_d)
    if agree.empty:
        return
    st.subheader("🧠 vs. LLM digest", help=g.ML_LLM_AGREEMENT)
    st.caption(
        f"LLM digest {digest_d} cross-referenced against ML ranks. Two "
        "independent signal sources agreeing is the strongest evidence "
        "either one produces."
    )
    n_agree = (agree["verdict"] == "agree").sum()
    n_conflict = (agree["verdict"] == "conflict").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Agree", n_agree)
    c2.metric("Neutral", (agree["verdict"] == "neutral").sum())
    c3.metric("Conflict", n_conflict)

    for verdict, icon in (("agree", "✅"), ("conflict", "⚠️")):
        sub = agree[agree["verdict"] == verdict]
        if sub.empty:
            continue
        st.markdown(f"**{icon} {verdict.title()}:**")
        for _, r in sub.iterrows():
            st.markdown(
                f"&nbsp;&nbsp;{icon} **{r['symbol']}** ({r['channel']}) — LLM "
                f"{r['llm_direction'].upper()} conv {r['llm_conviction']:.1f} · "
                f"ML {r['ml_action']} rank #{r['ml_rank']}/{r['ml_n']}"
            )


def _render_forward_performance() -> None:
    """Live, out-of-sample BUY−AVOID spread on the signals we actually
    published. The walk-forward panel below proves historical skill; this
    proves (or disproves) it on real forward dates as they mature."""
    fp = q.ml_forward_performance(horizon=21)
    st.subheader(
        "🎯 Live forward track record (BUY − AVOID)",
        help=(
            "For each past signal date, the average 21-trading-day forward "
            "return of that day's BUY bucket minus its AVOID bucket — a "
            "self-financing long/short read. A date only counts once 21 "
            "trading days of bars exist after it, so this is genuinely "
            "out-of-sample (the ranking never saw the forward window). No "
            "LLM, no training-data contamination."
        ),
    )
    by_model = fp.get("by_model", {})
    if not by_model:
        st.info(
            "No ML signals recorded yet. The forward track record builds "
            "itself as `run_ml_signals.py` accumulates daily ranks."
        )
        return

    for model_version in sorted(by_model):
        m = by_model[model_version]
        label = q.ML_MODEL_LABELS.get(model_version, model_version)
        if m["n_dates_matured"] == 0:
            nxt = m.get("next_maturity_date")
            eta = f" — first matures ~{nxt.isoformat()}" if nxt else ""
            st.caption(
                f"**{label}:** {m['n_dates_total']} signal date(s) recorded, "
                f"0 matured at the 21-trading-day horizon yet{eta}. "
                "Spreads appear here automatically as dates mature."
            )
            continue

        st.markdown(f"**{label}** — {m['n_dates_matured']} matured "
                    f"signal date(s) ({m['n_pending']} still maturing)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean BUY−AVOID spread", f"{m['mean_spread']:+.2%}")
        c2.metric("Spread > 0", f"{m['spread_hit_rate']:.0%}",
                  help="Share of matured dates where BUY beat AVOID.")
        c3.metric("Mean BUY return", f"{m['mean_buy_ret']:+.2%}")
        c4.metric("BUY beats average", f"{m['buy_beats_all_rate']:.0%}"
                  if m["buy_beats_all_rate"] is not None else "—",
                  help="Share of dates the BUY bucket beat the equal-weight "
                       "cross-section.")
        if m["n_dates_matured"] < 10:
            st.caption(
                "⚠️ Under ~10 matured dates — directional read only, not yet "
                "a reliable estimate of skill."
            )
        per_date = m.get("per_date", [])
        if per_date:
            with st.expander(f"Per-date spreads ({len(per_date)})"):
                pdf = pd.DataFrame(per_date)
                view = pd.DataFrame({
                    "Signal date": pdf["signal_date"].map(lambda d: d.isoformat()),
                    "BUY ret": pdf["buy_ret"].map(lambda x: f"{x:+.2%}"),
                    "AVOID ret": pdf["avoid_ret"].map(lambda x: f"{x:+.2%}"),
                    "Spread": pdf["spread"].map(lambda x: f"{x:+.2%}"),
                    "n BUY/AVOID": pdf["n_buy"].astype(str) + "/" + pdf["n_avoid"].astype(str),
                })
                st.dataframe(view, hide_index=True, use_container_width=True)


def _render_validation() -> None:
    val = q.ml_validation()
    st.subheader("📐 Does this signal actually work?", help=g.WALK_FORWARD_OOS)
    if val is None:
        st.info(
            "Validation hasn't been run yet. Run "
            "`python scripts/validate_ml.py` (free, ~10 min) to produce the "
            "walk-forward out-of-sample report."
        )
        return

    st.caption(
        f"Generated {val['generated_at']} · {val['notes']}"
    )
    deep = val.get("deep")
    if deep:
        st.markdown(
            "**Deep walk-forward (2008 → present, 19 survivorship-clean "
            "ETFs, 5y-train/2y-test):** every number below is "
            "out-of-sample — no model or parameter ever saw its test window."
        )
        rows = []
        for name, m in deep["advisors"].items():
            rows.append({
                "Strategy": name,
                "Total": f"{m['total_return']:+.1%}",
                "Annual": f"{m['annualized_return']:+.1%}",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Max DD": f"{m['max_drawdown']:.1%}",
                "Alpha/yr": f"{m['alpha_annualized']:+.1%}",
                "vs SPY": f"{m['total_return'] - m['benchmark_total_return']:+.1%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with st.expander("Per-walk test windows (stability check)"):
            for adv in ("ml_momentum", "ml_xgb"):
                walks = deep.get("walks", {}).get(adv)
                if not walks:
                    continue
                st.markdown(f"**{adv}**")
                wdf = pd.DataFrame(walks)
                wdf["total_return"] = wdf["total_return"].map(lambda x: f"{x:+.1%}")
                wdf["sharpe"] = wdf["sharpe"].map(lambda x: f"{x:.2f}")
                wdf["max_drawdown"] = wdf["max_drawdown"].map(lambda x: f"{x:.1%}")
                wdf["vs_benchmark"] = wdf["vs_benchmark"].map(lambda x: f"{x:+.1%}")
                st.dataframe(wdf, hide_index=True, use_container_width=True)

    broad = val.get("broad")
    if broad:
        st.markdown(
            f"**Broad universe ({broad.get('universe_size', '?')} ETFs, "
            "2022-07 → present):** single short window — context, not proof."
        )
        rows = []
        for name, m in broad["advisors"].items():
            rows.append({
                "Strategy": name,
                "Total": f"{m['total_return']:+.1%}",
                "Annual": f"{m['annualized_return']:+.1%}",
                "Sharpe": f"{m['sharpe']:.2f}",
                "Max DD": f"{m['max_drawdown']:.1%}",
                "vs SPY": f"{m['total_return'] - m['benchmark_total_return']:+.1%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render() -> None:
    st.title("🤖 ML Signals")
    st.caption(
        "Cross-sectional ranking of the full universe, computed daily from "
        "price history alone — free, deterministic, and (unlike LLM "
        "backtests) honestly backtestable. Top quintile = BUY candidates, "
        "bottom quintile = AVOID."
    )

    dates = q.available_ml_dates()
    if not dates:
        st.warning(
            "No ML signals yet. Run `python scripts/run_ml_signals.py` "
            "(free, ~30s) to generate today's ranks."
        )
        _render_validation()
        return

    col_date, col_note = st.columns([1, 3])
    with col_date:
        picked = st.selectbox("Signal date", options=dates, index=0,
                              format_func=lambda d: d.isoformat())
    models = q.ml_models_for_date(picked)
    with col_note:
        st.markdown(
            "**Models:** " + " · ".join(
                f"`{q.ML_MODEL_LABELS.get(m, m)}`" for m in models
            )
        )

    _render_consensus(picked)
    st.divider()

    # Per-model bucket tables
    tabs = st.tabs([q.ML_MODEL_LABELS.get(m, m) for m in models])
    for tab, model in zip(tabs, models):
        with tab:
            df = q.ml_signals_for_date(picked, model)
            if df.empty:
                st.info("No rows for this model/date.")
                continue
            buys = df[df["action"] == "BUY"]
            avoids = df[df["action"] == "AVOID"]
            holds = df[df["action"] == "HOLD"]

            c1, c2, c3 = st.columns(3)
            c1.metric("BUY (top quintile)", len(buys), help=g.ML_ACTION)
            c2.metric("HOLD (middle)", len(holds))
            c3.metric("AVOID (bottom quintile)", len(avoids))

            st.markdown("**▲ BUY — ranked**", help=g.ML_SCORE)
            st.dataframe(_fmt(buys), hide_index=True, use_container_width=True)
            st.markdown("**▼ AVOID — weakest first**")
            st.dataframe(
                _fmt(avoids.sort_values("rank", ascending=False)),
                hide_index=True, use_container_width=True,
            )
            with st.expander(f"Full cross-section ({len(df)} symbols)"):
                st.dataframe(_fmt(df), hide_index=True, use_container_width=True)

    st.divider()
    _render_llm_crosscheck(picked)
    st.divider()
    _render_forward_performance()
    st.divider()
    _render_validation()
