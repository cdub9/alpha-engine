"""Era-stratified evaluation.

Splits market history into named, non-overlapping eras chosen to expose
strategies to qualitatively different regimes. Running the same advisor
across all eras tells us whether its edge is broad-based or concentrated.

Default eras (chosen for the US equity market):

  - GFC + recovery     2008-01 to 2009-12   -55% drawdown then sharp rebound
  - Bull cycle         2010-01 to 2018-12   long expansion, periodic vol shocks
  - COVID + rally      2019-01 to 2021-12   inversion, flash crash, melt-up
  - Rate hike + AI     2022-01 to present   2022 bear, 2023-24 AI rally

A strategy that shines in one era and dies in another isn't trustworthy
in production — it's regime-fit to the wrong slice of history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import duckdb

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.backtest.metrics import BacktestMetrics
from alpha_engine.backtest.types import BacktestConfig, SignalAdvisor
from alpha_engine.db import get_connection


@dataclass(frozen=True)
class Era:
    name: str
    start: date
    end: date
    description: str = ""


DEFAULT_ERAS: list[Era] = [
    Era(
        name="gfc_recovery",
        start=date(2008, 1, 1),
        end=date(2009, 12, 31),
        description="Global Financial Crisis + initial recovery (peak -55%)",
    ),
    Era(
        name="bull_cycle",
        start=date(2010, 1, 1),
        end=date(2018, 12, 31),
        description="Long expansion with 2011/2015/2018 vol shocks",
    ),
    Era(
        name="covid_rally",
        start=date(2019, 1, 1),
        end=date(2021, 12, 31),
        description="Late-cycle, COVID flash crash, melt-up rally",
    ),
    Era(
        name="rate_hike_ai",
        start=date(2022, 1, 1),
        end=date(2026, 5, 29),
        description="2022 bear, prolonged inversion, 2023-24 AI rally",
    ),
]


AdvisorFactory = Callable[[], SignalAdvisor]


@dataclass
class EraResult:
    era: Era
    advisor_name: str
    metrics: BacktestMetrics


def evaluate_by_era(
    advisor_factory: AdvisorFactory,
    base_config: BacktestConfig,
    eras: list[Era] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[EraResult]:
    """Run the advisor independently in each era. Returns per-era metrics.

    `base_config` provides universe, costs, etc.; only start/end are
    overridden per era.
    """
    eras = eras or DEFAULT_ERAS
    owned = con is None
    if owned:
        con = get_connection(read_only=True)

    try:
        out: list[EraResult] = []
        advisor = advisor_factory()
        for era in eras:
            cfg = BacktestConfig(
                start_date=era.start,
                end_date=era.end,
                initial_capital=base_config.initial_capital,
                universe=base_config.universe,
                benchmark=base_config.benchmark,
                rebalance_frequency=base_config.rebalance_frequency,
                commission_bps=base_config.commission_bps,
                spread_bps=base_config.spread_bps,
                slippage_bps=base_config.slippage_bps,
                max_position_weight=base_config.max_position_weight,
                max_leverage=base_config.max_leverage,
                drift_rebalance_threshold=base_config.drift_rebalance_threshold,
            )
            result = run_backtest(cfg, advisor_factory(), con=con)
            assert result.metrics is not None
            out.append(
                EraResult(era=era, advisor_name=advisor.name, metrics=result.metrics)
            )
        return out
    finally:
        if owned:
            con.close()
