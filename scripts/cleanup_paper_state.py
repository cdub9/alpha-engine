"""One-off cleanup: nuke duplicate signals and all derived paper trades /
outcomes, leaving the llm_signal_cache intact so we can rebuild from scratch.

Run once after the dedup bug is fixed. Then:
    python scripts/paper_trader.py backfill
    python scripts/paper_trader.py open --all
    python scripts/paper_trader.py score
    python scripts/paper_trader.py status
"""

from rich.console import Console

from alpha_engine.db import get_connection

console = Console()

with get_connection() as con:
    before_signals = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    before_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    before_outcomes = con.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]

    # Order matters due to FK semantics (trade_outcomes -> trades -> signals)
    con.execute("DELETE FROM trade_outcomes")
    con.execute("DELETE FROM trades")
    con.execute("DELETE FROM signals")

    after_signals = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    after_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    after_outcomes = con.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]

    cache = con.execute("SELECT COUNT(*) FROM llm_signal_cache").fetchone()[0]

console.print(f"signals:         {before_signals} -> {after_signals}")
console.print(f"trades:          {before_trades} -> {after_trades}")
console.print(f"trade_outcomes:  {before_outcomes} -> {after_outcomes}")
console.print(f"llm_signal_cache (preserved): {cache}")
