"""Portfolio concentration analysis + ranked risk actions.

The lesson from the June 4 / 22 / 26 drawdowns: those weren't market
crashes, they were a concentrated semis/AI-hardware bet getting hit. The
broad index was flat; the book wasn't. This module turns a holdings list
into the prioritized "do these first" actions the dashboard Action Center
renders — so the highest-value move (trim the 12% single name, cut the 38%
cluster) is one glance away instead of buried in 64 rows.

Pure functions over plain dicts (no DB, no network) so the same code runs
on the real brokerage book and the paper portfolio, and is unit-testable.

A holding is a dict: {"symbol": str, "value": float}. Weights are computed
against the summed value, so partial books still report sensible percents.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Correlated-cluster map. The point of clustering is that names inside one
# move together — capping per-NAME isn't enough when 20 chip names all fall
# on the same day. `tech_total` (semis + tech-growth ETFs) is the broad
# factor that actually drove the drawdowns.
# ---------------------------------------------------------------------------

CLUSTERS: dict[str, set[str]] = {
    "semis_ai_hw": {
        "NVDA", "AMD", "MU", "TSM", "AVGO", "ASML", "LRCX", "AMAT", "MRVL",
        "INTC", "WDC", "STX", "SNDK", "COHR", "ALAB", "SMCI", "SIMO", "ANET",
        "NBIS", "VRT", "SMH", "SOXX", "DRAM",
    },
    "tech_growth_etf": {"QQQM", "VGT", "VUG", "SCHG", "SPYG", "SPMO", "VONG"},
    "leveraged": {"TQQQ", "SOXL", "TQQQ", "UPRO", "TECL", "SPXL", "QLD", "USD"},
    "broad_index": {"IVV", "VTI", "VOO", "SPY"},
    "international": {"VXUS", "VWO", "EWY", "IGRO", "EFA", "INDA", "EWJ"},
    "defensive": {"VIG", "VTV", "CGDV", "JQUA", "DGRO"},
    "bonds_credit": {"HYS", "HYG", "ARCC", "AGG", "TLT", "LQD", "TIP", "SHY"},
}

_SYMBOL_TO_CLUSTER = {
    sym: name for name, syms in CLUSTERS.items() for sym in syms
}

# Cluster meta-group: the broad tech/semis factor exposure.
TECH_FACTOR = ("semis_ai_hw", "tech_growth_etf")

# Default risk limits (fractions of total book).
DEFAULT_CAPS = {
    "name": 0.05,            # no single position over 5%
    "semis_ai_hw": 0.20,     # one correlated cluster under 20%
    "tech_total": 0.35,      # semis + tech ETFs under 35%
    "leveraged": 0.10,       # leveraged products under 10%
}

# The per-NAME cap targets idiosyncratic single-stock concentration. A
# diversified basket (broad index, the semis ETFs, intl/defensive/bond
# funds) isn't a single-name bet — a 15% IVV position is fine, a 12% MU
# position is not — so those clusters are exempt from the name cap. Their
# factor risk, where it exists, is caught by the cluster caps instead.
_NAME_CAP_EXEMPT_CLUSTERS = {
    "broad_index", "international", "defensive", "bonds_credit",
    "tech_growth_etf", "leveraged",
}
_NAME_CAP_EXEMPT_SYMBOLS = {"SMH", "SOXX", "DRAM"}  # diversified semis baskets


def _exempt_from_name_cap(symbol: str) -> bool:
    return (
        cluster_of(symbol) in _NAME_CAP_EXEMPT_CLUSTERS
        or symbol.upper().strip() in _NAME_CAP_EXEMPT_SYMBOLS
    )


def cluster_of(symbol: str) -> str:
    """Cluster label for a symbol, or 'other' if unmapped."""
    return _SYMBOL_TO_CLUSTER.get(symbol.upper().strip(), "other")


def concentration_report(
    holdings: list[dict[str, Any]],
    caps: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Weights by name and by correlated cluster, plus cap breaches.

    Returns:
      {
        "total_value": float,
        "names": [{symbol, value, weight}], sorted desc by weight
        "clusters": {cluster: {value, weight}},
        "tech_total_weight": float,        # semis + tech-growth ETFs
        "top_name": {symbol, weight} | None,
        "breaches": [{kind, label, weight, cap, excess_value}], sorted by excess
      }
    """
    caps = {**DEFAULT_CAPS, **(caps or {})}
    total = sum(float(h["value"]) for h in holdings) or 1.0

    names = sorted(
        ({"symbol": h["symbol"].upper(), "value": float(h["value"]),
          "weight": float(h["value"]) / total} for h in holdings),
        key=lambda r: r["weight"], reverse=True,
    )

    clusters: dict[str, dict[str, float]] = {}
    for h in holdings:
        c = cluster_of(h["symbol"])
        slot = clusters.setdefault(c, {"value": 0.0, "weight": 0.0})
        slot["value"] += float(h["value"])
    for slot in clusters.values():
        slot["weight"] = slot["value"] / total

    tech_total = sum(clusters.get(c, {}).get("weight", 0.0) for c in TECH_FACTOR)

    breaches: list[dict[str, Any]] = []
    # Per-name breaches (single stocks only — diversified baskets exempt)
    for n in names:
        if _exempt_from_name_cap(n["symbol"]):
            continue
        if n["weight"] > caps["name"]:
            breaches.append({
                "kind": "name", "label": n["symbol"],
                "weight": n["weight"], "cap": caps["name"],
                "excess_value": (n["weight"] - caps["name"]) * total,
            })
    # Cluster breaches
    for cap_key, cluster_key in (("semis_ai_hw", "semis_ai_hw"),
                                 ("leveraged", "leveraged")):
        w = clusters.get(cluster_key, {}).get("weight", 0.0)
        if w > caps[cap_key]:
            breaches.append({
                "kind": "cluster", "label": cluster_key,
                "weight": w, "cap": caps[cap_key],
                "excess_value": (w - caps[cap_key]) * total,
            })
    # Tech-factor breach (meta-cluster)
    if tech_total > caps["tech_total"]:
        breaches.append({
            "kind": "tech_total", "label": "tech_total",
            "weight": tech_total, "cap": caps["tech_total"],
            "excess_value": (tech_total - caps["tech_total"]) * total,
        })

    breaches.sort(key=lambda b: b["excess_value"], reverse=True)

    return {
        "total_value": total,
        "names": names,
        "clusters": clusters,
        "tech_total_weight": tech_total,
        "top_name": ({"symbol": names[0]["symbol"], "weight": names[0]["weight"]}
                     if names else None),
        "breaches": breaches,
        "caps": caps,
    }


# Severity ordering for sorting actions (higher = more urgent).
_SEVERITY_RANK = {"critical": 3, "high": 2, "watch": 1}

_CLUSTER_LABELS = {
    "semis_ai_hw": "semis / AI-hardware",
    "tech_total": "total tech (semis + tech ETFs)",
    "leveraged": "leveraged products",
}


def rank_actions(
    report: dict[str, Any],
    upcoming_earnings: Optional[list[dict[str, Any]]] = None,
    cash_weight: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Turn a concentration report (+ optional earnings/cash) into a ranked
    action list for the Action Center. Each action:
      {severity, icon, title, rationale, metric_now, metric_target, value_at_risk}
    Sorted most-urgent first, then by dollars involved.
    """
    total = report["total_value"]
    caps = report["caps"]
    actions: list[dict[str, Any]] = []

    for b in report["breaches"]:
        if b["kind"] == "name":
            sym, w = b["label"], b["weight"]
            actions.append({
                "severity": "critical" if w >= 2 * caps["name"] else "high",
                "icon": "flame",
                "title": f"Trim {sym} toward {caps['name']:.0%} of the book",
                "rationale": (
                    f"{sym} is {w:.1%} of the account "
                    f"(${b['weight']*total:,.0f}) — a single bad day or "
                    "earnings gap hits the whole portfolio."
                ),
                "metric_now": f"{w:.1%}",
                "metric_target": f"{caps['name']:.0%}",
                "value_at_risk": b["excess_value"],
            })
        elif b["kind"] in ("cluster", "tech_total"):
            label = _CLUSTER_LABELS.get(b["label"], b["label"])
            actions.append({
                "severity": "critical" if b["label"] == "semis_ai_hw" else "high",
                "icon": "chart-pie",
                "title": f"Cut {label} toward {b['cap']:.0%}",
                "rationale": (
                    f"{b['weight']:.0%} of the book moves together as one "
                    f"bet (${b['excess_value']:,.0f} over the {b['cap']:.0%} cap)."
                ),
                "metric_now": f"{b['weight']:.0%}",
                "metric_target": f"{b['cap']:.0%}",
                "value_at_risk": b["excess_value"],
            })

    # Earnings watch — held names reporting soon shouldn't be at full size.
    for e in upcoming_earnings or []:
        val = float(e.get("value") or 0.0)
        actions.append({
            "severity": "high" if val >= 0.04 * total else "watch",
            "icon": "calendar-event",
            "title": f"Trim {e['symbol']} into earnings ({e['date']})",
            "rationale": (
                f"{e['symbol']} reports {e['date']} "
                f"(${val:,.0f} held). Earnings gaps are the single biggest "
                "avoidable single-name risk — don't hold full size into one."
            ),
            "metric_now": e["date"],
            "metric_target": "trim",
            "value_at_risk": val,
        })

    # Low-cash / no-hedge flag when the book is concentrated.
    if cash_weight is not None and cash_weight < 0.05 and report["tech_total_weight"] > 0.35:
        actions.append({
            "severity": "high",
            "icon": "shield-half",
            "title": "Raise cash or add a hedge",
            "rationale": (
                f"Only {cash_weight:.1%} cash against a "
                f"{report['tech_total_weight']:.0%} tech-heavy book with no "
                "downside protection."
            ),
            "metric_now": f"{cash_weight:.1%}",
            "metric_target": "hedge",
            "value_at_risk": 0.0,
        })

    actions.sort(
        key=lambda a: (_SEVERITY_RANK.get(a["severity"], 0), a["value_at_risk"]),
        reverse=True,
    )
    return actions
