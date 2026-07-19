"""Turn the app's analysis into concrete trade orders.

The Action Center's job is to answer one question — "what do I do today?" —
as executable order tickets, not prose. This engine consumes the outputs
the rest of the app already produces (concentration report, earnings guard,
semis-trend state, ML ranks) and emits an ordered list of SELL orders with
exact share counts, dollar amounts, the reason, and the timing.

Design choices that keep it honest:
  - It only generates SELL / trim orders. Redeploying the proceeds is the
    user's call; the app de-risks, it doesn't chase entries with real money.
  - Timing is a trigger, not a guess: "Now" (a live cap breach), "Before
    <date>" (a calendared earnings print), or "Armed — if the trend breaks"
    (a cluster cut that shouldn't fire while the uptrend holds).
  - The semis-cluster cut respects the trend. With the trend intact it's
    ARMED (optional right-sizing), not an active order — trend-following
    says don't dump a working trend. It only becomes an active order when
    the proxy is below its 200-day average.

Pure function over plain dicts; no DB, no network.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Optional


def _shares(dollars: float, price: Optional[float]) -> Optional[int]:
    """Whole shares to sell to raise ~`dollars`, or None if price unknown."""
    if not price or price <= 0 or dollars <= 0:
        return None
    return ceil(dollars / price)


def _ml_note(symbol: str, ml_actions: Optional[dict[str, str]]) -> str:
    a = (ml_actions or {}).get(symbol.upper())
    if a == "AVOID":
        return "ML rates it AVOID — reinforces the trim."
    if a == "BUY":
        return "Note: ML still rates it BUY; this trim is risk control, not a call on the name."
    return ""


def _order(action, symbol, shares, price, cur_w, tgt_w, reason, when, dollars, ml_actions):
    return {
        "action": action,
        "symbol": symbol,
        "shares": shares,
        "est_dollars": dollars,
        "price": price,
        "current_weight": cur_w,
        "target_weight": tgt_w,
        "reason": reason,
        "when": when,
        "ml_note": _ml_note(symbol, ml_actions),
    }


# Preferred order to trim a cluster: clean beta reducers (the baskets)
# first, so cutting exposure doesn't mean picking on one single name.
_CLEAN_SEMIS_TRIMS = ("SMH", "SOXX", "DRAM")


def build_trade_plan(
    holdings: list[dict[str, Any]],
    report: dict[str, Any],
    caps: dict[str, float],
    trend: Optional[dict[str, Any]] = None,
    earnings: Optional[list[dict[str, Any]]] = None,
    ml_actions: Optional[dict[str, str]] = None,
    earnings_trim_frac: float = 0.5,
) -> dict[str, Any]:
    """Build the day's orders.

    Returns:
      {
        "orders":      [order, ...]   # DO NOW — single-name + earnings trims
        "armed":       [order, ...]   # conditional cluster cut (trend-gated)
        "cluster_note": str | None    # plain-English cluster status
        "summary":     {n_orders, sell_dollars_now}
      }
    """
    total = report["total_value"]
    price = {h["symbol"].upper(): h.get("price") for h in holdings}
    value = {h["symbol"].upper(): float(h["value"]) for h in holdings}
    name_cap = caps["name"]

    orders: list[dict[str, Any]] = []
    trimmed: set[str] = set()

    # 1. Single-name concentration breaches -> trim to the name cap, NOW.
    for b in report["breaches"]:
        if b["kind"] != "name":
            continue
        sym = b["label"]
        target_val = name_cap * total
        over = value[sym] - target_val
        sh = _shares(over, price.get(sym))
        orders.append(_order(
            "SELL", sym, sh, price.get(sym), value[sym] / total, name_cap,
            "Single-name concentration — one stock is too large a share of the account.",
            "Now", over, ml_actions,
        ))
        trimmed.add(sym)

    # 2. Held names reporting earnings soon -> cut size into the print.
    for e in earnings or []:
        sym = e["symbol"].upper()
        if sym in trimmed or sym not in value:
            continue
        over = value[sym] * earnings_trim_frac
        sh = _shares(over, price.get(sym))
        if not sh:
            continue
        orders.append(_order(
            "SELL", sym, sh, price.get(sym), value[sym] / total,
            value[sym] * (1 - earnings_trim_frac) / total,
            f"Earnings on {e['date']} — trim before the print; gaps are the "
            "biggest avoidable single-name risk.",
            f"Before {e['date']}", over, ml_actions,
        ))
        trimmed.add(sym)

    # 3. Semis-cluster cut — trend-gated.
    semis = report["clusters"].get("semis_ai_hw", {})
    semis_w = semis.get("weight", 0.0)
    semis_cap = caps["semis_ai_hw"]
    armed: list[dict[str, Any]] = []
    cluster_note: Optional[str] = None

    if semis_w > semis_cap:
        # Dollars still over the cap AFTER the single-name trims above.
        already = sum(o["est_dollars"] for o in orders if o["symbol"] in semis_members(report))
        excess = (semis_w - semis_cap) * total - already
        trend_broken = trend is not None and not trend.get("above", True)

        if excess > 0:
            # Pick names to trim: ML-AVOID first, then the clean baskets,
            # then the largest remaining single semis names.
            candidates = _cluster_trim_candidates(value, ml_actions, trimmed)
            remaining = excess
            for sym in candidates:
                if remaining <= 0:
                    break
                take = min(value[sym] * 0.5, remaining)  # don't gut any one line
                sh = _shares(take, price.get(sym))
                if not sh:
                    continue
                order = _order(
                    "SELL", sym, sh, price.get(sym), value[sym] / total, None,
                    "Reduce the semis/AI-hardware cluster toward its cap.",
                    "Now" if trend_broken else "Armed — triggers if the semis trend breaks",
                    take, ml_actions,
                )
                (orders if trend_broken else armed).append(order)
                remaining -= take

        if trend_broken:
            cluster_note = (
                f"Semis trend is broken — the cluster cut above is ACTIVE. "
                f"Bringing {semis_w:.0%} down toward {semis_cap:.0%}."
            )
        else:
            cluster_note = (
                f"Semis are {semis_w:.0%} of the book (cap {semis_cap:.0%}), but the "
                f"trend is intact — so the cluster cut is ARMED, not active. Do the "
                f"single-name trims now; the cluster orders fire automatically if the "
                f"proxy breaks its 200-day average."
            )

    sell_now = sum(o["est_dollars"] for o in orders)
    return {
        "orders": orders,
        "armed": armed,
        "cluster_note": cluster_note,
        "summary": {"n_orders": len(orders), "sell_dollars_now": sell_now},
    }


def semis_members(report: dict[str, Any]) -> set[str]:
    """Symbols in the book that belong to the semis cluster."""
    from alpha_engine.risk.portfolio import cluster_of

    return {n["symbol"] for n in report["names"] if cluster_of(n["symbol"]) == "semis_ai_hw"}


def _cluster_trim_candidates(
    value: dict[str, float],
    ml_actions: Optional[dict[str, str]],
    exclude: set[str],
) -> list[str]:
    """Order semis names for trimming: ML-AVOID first, then the diversified
    baskets (clean beta reduction), then largest single names."""
    from alpha_engine.risk.portfolio import cluster_of

    semis = [s for s in value if cluster_of(s) == "semis_ai_hw" and s not in exclude]
    ml = ml_actions or {}

    def sort_key(s: str):
        avoid = 0 if ml.get(s) == "AVOID" else 1
        basket = 0 if s in _CLEAN_SEMIS_TRIMS else 1
        return (avoid, basket, -value[s])

    return sorted(semis, key=sort_key)
