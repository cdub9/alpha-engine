"""One-off: check whether signals or paper trades have duplicates."""
from alpha_engine.db import get_connection

with get_connection(read_only=True) as con:
    n_signals = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_distinct = con.execute(
        "SELECT COUNT(DISTINCT (DATE(generated_at), channel, symbol, model_version)) "
        "FROM signals"
    ).fetchone()[0]
    print(f"signals total: {n_signals}")
    print(f"signals distinct by (date, channel, symbol, model_version): {n_distinct}")
    print(f"duplicate signals: {n_signals - n_distinct}")

    n_trades = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    n_trades_distinct = con.execute(
        "SELECT COUNT(DISTINCT source_signal_id) FROM trades"
    ).fetchone()[0]
    print(f"\ntrades total: {n_trades}")
    print(f"distinct source_signal_ids in trades: {n_trades_distinct}")

    n_outcomes = con.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
    print(f"\ntrade_outcomes total: {n_outcomes}")

    # Show example of a duplicated signal if any
    dupes = con.execute(
        """
        SELECT DATE(generated_at) AS d, channel, symbol, COUNT(*) AS n
        FROM signals
        WHERE model_version = 'llm-opus-4-7-v1'
        GROUP BY 1, 2, 3
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 5
        """
    ).fetchall()
    if dupes:
        print(f"\nTop duplicate signals (showing {len(dupes)}):")
        for d, ch, sym, n in dupes:
            print(f"  {d}  {ch}  {sym}  count={n}")
    else:
        print("\nNo duplicate signals found.")
