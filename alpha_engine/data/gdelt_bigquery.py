"""GDELT GKG ingestion via Google BigQuery.

Why this exists: the DOC API (gdelt.py) is full-text and free but capped at
a ~30-day rolling window and rate-limited, so the geopolitical layer could
never be backtested ("did energy outperform when iran_conflict spiked?").
The GKG public dataset on BigQuery holds years of history (2015+), has no
rate limits, and fits the BigQuery free tier IF queried carefully.

What it matches: BigQuery's GKG does NOT store article full text — it stores
GDELT's extracted entities (AllNames) and theme codes (V2Themes) plus a tone
score (V2Tone). So `bq_match` matches those fields, not raw text. It's a
coarser proxy than the DOC API's full-text search, but stable and
backtestable. One scan computes ALL signals at once (a COUNTIF per signal),
which is the key cost optimization — you pay for the columns once per
time-chunk, not once per signal.

Output is the same `GdeltDailyPoint` rows the DOC path produces:
    volume_intensity = matching_articles / total_articles  (0-1 fraction)
    avg_tone         = mean V2Tone of matching articles     (-10..+10)
so the downstream intel/feature layer is unchanged.

COST SAFETY — read scripts/ingest_gdelt_bq.py. On a non-partitioned table a
date filter does NOT reduce bytes scanned (BigQuery scans every row for the
referenced columns). The only safe workflow is: dry-run to estimate bytes,
gate on a GB threshold, and prefer a date-partitioned table. The client
exposes `estimate_gb()` for exactly this; the script refuses to run a query
whose estimate exceeds --max-gb.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Optional

from alpha_engine.core.config import GeopoliticalSignalSpec
from alpha_engine.core.logging import get_logger
from alpha_engine.data.gdelt import GdeltDailyPoint

if TYPE_CHECKING:  # avoid importing the heavy GCP SDK at module load
    from google.cloud import bigquery

log = get_logger(__name__)

# Default GKG table. The *_partitioned table allows partition pruning so a
# date filter actually limits bytes scanned — strongly preferred. If you
# point at the non-partitioned `gdelt-bq.gdeltv2.gkg`, the dry-run will show
# a much larger estimate (a date filter won't help there).
DEFAULT_GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
DEFAULT_PARTITION_FIELD = "_PARTITIONTIME"

# Only these GKG columns are ever scanned (keeps bytes down). V2Themes is the
# largest; --no-theme-match drops it for entity-only matching at lower cost.
_COL_DATE = "DATE"
_COL_TONE = "V2Tone"
_COL_NAMES = "AllNames"
_COL_THEMES = "V2Themes"

# bq_match strings are from our own config, but validate anyway: lowercase
# letters/digits/space and a small regex/phrase set. Anything else is a
# config error we want to fail loudly on, not smuggle into SQL.
_ALLOWED_MATCH = re.compile(r"^[a-z0-9 _\-|'\".]+$")
_SAFE_ALIAS = re.compile(r"[^a-z0-9_]")


def _validate_match_token(token: str) -> str:
    if not token or not _ALLOWED_MATCH.match(token):
        raise ValueError(f"Unsafe bq_match token: {token!r}")
    # Escape single quotes for the SQL string literal (we emit r'...').
    return token.replace("'", "")


def _alias(name: str) -> str:
    return _SAFE_ALIAS.sub("_", name.lower())


def _match_expr(spec: GeopoliticalSignalSpec) -> str:
    """AND of REGEXP_CONTAINS(blob, alternation) for one signal."""
    clauses = [
        f"REGEXP_CONTAINS(blob, r'{_validate_match_token(grp)}')"
        for grp in spec.bq_match
    ]
    return "(" + " AND ".join(clauses) + ")"


def _date_filter(start: date, end: date, partition_field: Optional[str]) -> str:
    """WHERE fragment. Partitioned table → prune on the partition column
    (real byte savings). Non-partitioned → filter the integer DATE column
    (correct results, but no byte savings — the dry-run will reveal that)."""
    if partition_field:
        return (
            f"{partition_field} >= TIMESTAMP('{start.isoformat()}') "
            f"AND {partition_field} < TIMESTAMP('{end.isoformat()}')"
        )
    lo = int(start.strftime("%Y%m%d")) * 1_000_000
    hi = int(end.strftime("%Y%m%d")) * 1_000_000
    return f"{_COL_DATE} >= {lo} AND {_COL_DATE} < {hi}"


def build_gkg_query(
    signals: list[GeopoliticalSignalSpec],
    start: date,
    end: date,
    table: str = DEFAULT_GKG_TABLE,
    partition_field: Optional[str] = DEFAULT_PARTITION_FIELD,
    theme_match: bool = True,
) -> str:
    """Build the single-scan SQL computing per-day volume + tone for every
    signal. `end` is exclusive. Pure function — no BigQuery needed, so the
    query shape is unit-tested offline.

    Only signals with a non-empty bq_match are included.
    """
    usable = [s for s in signals if s.bq_match]
    if not usable:
        raise ValueError("No signals have a bq_match; nothing to query.")

    blob_cols = f"IFNULL({_COL_NAMES}, '')"
    if theme_match:
        blob_cols = f"CONCAT(IFNULL({_COL_NAMES}, ''), ' ', IFNULL({_COL_THEMES}, ''))"

    # Three nesting levels because BigQuery can't reference a SELECT-list
    # alias (`blob`) within the same SELECT:
    #   base  -> day, tone, blob (the one scan of the table)
    #   match -> per-signal boolean m_<alias> computed from blob
    #   agg   -> COUNTIF / AVG(tone) per signal, grouped by day
    match_cols = ",\n        ".join(
        f"{_match_expr(s)} AS m_{_alias(s.name)}" for s in usable
    )
    agg_cols = ",\n    ".join(
        f"COUNTIF(m_{_alias(s.name)}) AS cnt_{_alias(s.name)},\n    "
        f"AVG(IF(m_{_alias(s.name)}, tone, NULL)) AS tone_{_alias(s.name)}"
        for s in usable
    )

    return f"""
SELECT
    day,
    COUNT(*) AS total_count,
    {agg_cols}
FROM (
    SELECT
        day,
        tone,
        {match_cols}
    FROM (
        SELECT
            CAST(FLOOR({_COL_DATE} / 1000000) AS INT64) AS day,
            SAFE_CAST(SPLIT({_COL_TONE}, ',')[OFFSET(0)] AS FLOAT64) AS tone,
            LOWER({blob_cols}) AS blob
        FROM `{table}`
        WHERE {_date_filter(start, end, partition_field)}
    )
)
GROUP BY day
ORDER BY day
""".strip()


@dataclass
class _SignalAgg:
    name: str
    alias: str


def rows_to_points(
    rows: list[dict],
    signals: list[GeopoliticalSignalSpec],
) -> dict[str, list[GdeltDailyPoint]]:
    """Map BigQuery result rows (dict-like, one per day) into per-signal
    GdeltDailyPoint lists. volume_intensity = cnt/total."""
    usable = [_SignalAgg(s.name, _alias(s.name)) for s in signals if s.bq_match]
    out: dict[str, list[GdeltDailyPoint]] = {s.name: [] for s in usable}
    for row in rows:
        day_int = row["day"]
        d = date(day_int // 10000, (day_int // 100) % 100, day_int % 100)
        total = row.get("total_count") or 0
        for s in usable:
            cnt = row.get(f"cnt_{s.alias}") or 0
            tone = row.get(f"tone_{s.alias}")
            vol = (cnt / total) if total else None
            out[s.name].append(
                GdeltDailyPoint(
                    signal_date=d,
                    volume_intensity=vol,
                    avg_tone=float(tone) if tone is not None else None,
                    raw_query=f"bq:{s.name}",
                )
            )
    return out


class GDELTBigQueryClient:
    """Thin wrapper over google-cloud-bigquery. Import is lazy so the rest
    of the package works without the GCP SDK installed."""

    def __init__(self, project: str, location: str = "US") -> None:
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "google-cloud-bigquery is required for the BigQuery backend. "
                'Install it with:  pip install -e ".[bigquery]"  (or '
                "pip install google-cloud-bigquery db-dtypes)."
            ) from exc
        self._bq = bigquery
        self._client = bigquery.Client(project=project, location=location)
        self.project = project

    def estimate_gb(self, sql: str) -> float:
        """Dry-run: bytes this query WOULD scan, in GB. Costs nothing and
        hits no data. This is the safety gate the ingest script enforces."""
        job_config = self._bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self._client.query(sql, job_config=job_config)
        return job.total_bytes_processed / 1024**3

    def run(self, sql: str) -> list[dict]:
        """Execute for real and return rows as dicts. Only call after an
        estimate_gb() gate has passed."""
        job = self._client.query(sql)
        return [dict(row.items()) for row in job.result()]

    def fetch_signals(
        self,
        signals: list[GeopoliticalSignalSpec],
        start: date,
        end: date,
        table: str = DEFAULT_GKG_TABLE,
        partition_field: Optional[str] = DEFAULT_PARTITION_FIELD,
        theme_match: bool = True,
    ) -> dict[str, list[GdeltDailyPoint]]:
        sql = build_gkg_query(
            signals, start, end, table=table,
            partition_field=partition_field, theme_match=theme_match,
        )
        rows = self.run(sql)
        return rows_to_points(rows, signals)
