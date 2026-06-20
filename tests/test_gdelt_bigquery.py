"""BigQuery GDELT backend — query construction, row mapping, injection
guards, and the recession-query split. All offline (no BigQuery / no SDK).
"""

from __future__ import annotations

from datetime import date

import pytest

from alpha_engine.core.config import GeopoliticalSignalSpec, get_settings
from alpha_engine.data.gdelt_bigquery import (
    _alias,
    _validate_match_token,
    build_gkg_query,
    rows_to_points,
)


def sig(name, match):
    return GeopoliticalSignalSpec(name=name, query="q", bq_match=match)


class TestQueryBuilder:
    def test_single_scan_covers_all_signals(self):
        signals = [
            sig("iran_conflict", ["iran", "attack|strike"]),
            sig("fed_policy", ["federal reserve", "rate|hike"]),
        ]
        sql = build_gkg_query(signals, date(2024, 1, 1), date(2025, 1, 1))
        # One FROM the GKG table (single scan), COUNTIF per signal.
        assert sql.count("FROM `") == 1
        assert "COUNTIF(m_iran_conflict)" in sql
        assert "COUNTIF(m_fed_policy)" in sql
        assert "AVG(IF(m_iran_conflict, tone, NULL))" in sql
        # AND-of-ORs structure preserved in the match expression
        assert (
            "(REGEXP_CONTAINS(blob, r'iran') AND "
            "REGEXP_CONTAINS(blob, r'attack|strike')) AS m_iran_conflict"
        ) in sql

    def test_partitioned_filter_prunes_on_partitiontime(self):
        sql = build_gkg_query([sig("a", ["x"])], date(2024, 1, 1), date(2024, 2, 1))
        assert "_PARTITIONTIME >= TIMESTAMP('2024-01-01')" in sql
        assert "_PARTITIONTIME < TIMESTAMP('2024-02-01')" in sql

    def test_nonpartitioned_filters_integer_date(self):
        sql = build_gkg_query(
            [sig("a", ["x"])], date(2024, 1, 1), date(2024, 2, 1),
            table="gdelt-bq.gdeltv2.gkg", partition_field=None,
        )
        assert "DATE >= 20240101000000" in sql
        assert "DATE < 20240201000000" in sql

    def test_theme_match_toggles_v2themes_column(self):
        with_theme = build_gkg_query([sig("a", ["x"])], date(2024, 1, 1), date(2024, 2, 1))
        without = build_gkg_query(
            [sig("a", ["x"])], date(2024, 1, 1), date(2024, 2, 1), theme_match=False
        )
        assert "V2Themes" in with_theme       # richer, more bytes
        assert "V2Themes" not in without      # cheaper, entity-only
        assert "AllNames" in without

    def test_signals_without_bq_match_are_excluded(self):
        signals = [sig("has_match", ["x"]), GeopoliticalSignalSpec(name="no_match", query="q")]
        sql = build_gkg_query(signals, date(2024, 1, 1), date(2025, 1, 1))
        assert "m_has_match" in sql
        assert "no_match" not in sql

    def test_raises_when_no_usable_signals(self):
        with pytest.raises(ValueError):
            build_gkg_query([GeopoliticalSignalSpec(name="n", query="q")],
                            date(2024, 1, 1), date(2025, 1, 1))


class TestInjectionGuards:
    def test_rejects_sql_metacharacters(self):
        for bad in ["x); DROP TABLE", "a`b", "a;b", "a)b", "a\\b"]:
            with pytest.raises(ValueError):
                _validate_match_token(bad)

    def test_allows_expected_regex_tokens(self):
        for ok in ["iran", "attack|strike|missile", "trade war", "north korea|dprk"]:
            assert _validate_match_token(ok) == ok

    def test_strips_single_quotes(self):
        assert "'" not in _validate_match_token("it's")

    def test_alias_sanitizes_name(self):
        assert _alias("US-China Trade!") == "us_china_trade_"

    def test_malicious_name_cannot_break_sql(self):
        # A hostile signal name must not inject; alias is alnum+underscore only.
        s = sig("a; DROP TABLE x", ["iran"])
        out = build_gkg_query([s], date(2024, 1, 1), date(2025, 1, 1))
        assert "DROP TABLE" not in out


class TestRowMapping:
    def test_volume_is_match_fraction_and_tone_passthrough(self):
        signals = [sig("iran_conflict", ["iran"]), sig("fed_policy", ["fed"])]
        rows = [
            {"day": 20240115, "total_count": 1000,
             "cnt_iran_conflict": 50, "tone_iran_conflict": -3.2,
             "cnt_fed_policy": 10, "tone_fed_policy": 1.5},
            {"day": 20240116, "total_count": 800,
             "cnt_iran_conflict": 0, "tone_iran_conflict": None,
             "cnt_fed_policy": 40, "tone_fed_policy": -0.5},
        ]
        pts = rows_to_points(rows, signals)
        iran = pts["iran_conflict"]
        assert iran[0].signal_date == date(2024, 1, 15)
        assert iran[0].volume_intensity == pytest.approx(0.05)   # 50/1000
        assert iran[0].avg_tone == pytest.approx(-3.2)
        assert iran[1].volume_intensity == pytest.approx(0.0)
        assert iran[1].avg_tone is None
        assert pts["fed_policy"][1].volume_intensity == pytest.approx(0.05)  # 40/800

    def test_zero_total_gives_none_volume(self):
        pts = rows_to_points(
            [{"day": 20240115, "total_count": 0, "cnt_a": 0, "tone_a": None}],
            [sig("a", ["x"])],
        )
        assert pts["a"][0].volume_intensity is None

    def test_raw_query_tagged_bq(self):
        pts = rows_to_points(
            [{"day": 20240115, "total_count": 10, "cnt_a": 1, "tone_a": 2.0}],
            [sig("a", ["x"])],
        )
        assert pts["a"][0].raw_query == "bq:a"


class TestConfigSplit:
    def test_recession_query_was_split(self):
        names = {s.name for s in get_settings().geopolitical.signals}
        # The old 4-clause recession_sentiment is gone, replaced by two.
        assert "recession_sentiment" not in names
        assert "recession_mentions" in names
        assert "bear_market_mentions" in names

    def test_all_signals_have_bq_match(self):
        # Every configured signal should be BigQuery-ingestable.
        for s in get_settings().geopolitical.signals:
            assert s.bq_match, f"{s.name} missing bq_match"
