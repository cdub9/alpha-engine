"""Core domain types.

Options (OptionContract, OptionLeg) are defined as first-class types so the
schema and code paths support them. Channel configs gate whether the signal
generator may actually emit option-based suggestions via `options_enabled`.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Channel(str, Enum):
    STEADY_ALPHA = "steady_alpha"
    AGGRESSIVE_GROWTH = "aggressive_growth"


class InstrumentType(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    BOND_ETF = "bond_etf"
    LEVERAGED_ETF = "leveraged_etf"
    OPTION = "option"  # supported in schema; gated by channel config


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    REDUCE = "reduce"
    ADD = "add"


class TradeStatus(str, Enum):
    SUGGESTED = "suggested"      # signal generated, not acted on
    PAPER_FILLED = "paper_filled"  # filled in paper trading
    LIVE_FILLED = "live_filled"    # filled with real money
    REJECTED = "rejected"        # rejected by risk layer
    CANCELLED = "cancelled"


class MarketRegime(str, Enum):
    EXPANSION_LOW_VOL = "expansion_low_vol"
    EXPANSION_HIGH_VOL = "expansion_high_vol"
    LATE_CYCLE = "late_cycle"
    RECESSION = "recession"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


class Instrument(BaseModel):
    """A tradable instrument in the universe."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    instrument_type: InstrumentType
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    active: bool = True

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.upper().strip()


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class MarketBar(BaseModel):
    """Daily OHLCV bar."""

    symbol: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str = "yfinance"

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.upper().strip()


class MacroObservation(BaseModel):
    """A single point in a macro time series."""

    series_id: str
    obs_date: date
    value: Optional[float]  # None when the series reports a missing value
    source: str = "fred"


# ---------------------------------------------------------------------------
# Options (dormant; data may still be collected for equity IV signals)
# ---------------------------------------------------------------------------


class OptionContract(BaseModel):
    """A single option contract."""

    model_config = ConfigDict(frozen=True)

    underlying: str
    expiry: date
    strike: Decimal
    option_type: OptionType
    contract_symbol: Optional[str] = None  # OCC-style symbol when known

    @field_validator("underlying")
    @classmethod
    def _upper_underlying(cls, v: str) -> str:
        return v.upper().strip()


class OptionLeg(BaseModel):
    """A leg of an option position. Multi-leg strategies (spreads, condors)
    are represented as multiple legs sharing a parent position."""

    contract: OptionContract
    side: PositionSide
    ratio: int = 1  # e.g. 1 for single, 2 for ratio spreads


class OptionGreeks(BaseModel):
    """Greeks for an option contract at a point in time."""

    contract_symbol: str
    snapshot_at: datetime
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    underlying_price: Optional[float] = None


# ---------------------------------------------------------------------------
# Positions, signals, trades
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """A held position. For equities/ETFs use `symbol`; for options use
    `option_legs`. Exactly one must be populated."""

    id: Optional[int] = None
    channel: Channel
    instrument_type: InstrumentType
    symbol: Optional[str] = None
    option_legs: Optional[list[OptionLeg]] = None
    side: PositionSide
    quantity: float
    entry_price: float
    entry_date: datetime
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    notes: Optional[str] = None
    source_signal_id: Optional[int] = None
    closed_at: Optional[datetime] = None
    closed_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: Optional[str]) -> Optional[str]:
        return v.upper().strip() if v else v


class Signal(BaseModel):
    """A trade suggestion produced by the signal engine."""

    id: Optional[int] = None
    generated_at: datetime
    channel: Channel
    symbol: str  # underlying for options
    instrument_type: InstrumentType
    direction: SignalDirection
    conviction: float = Field(ge=0.0, le=10.0)
    target_weight: Optional[float] = None  # 0.0-1.0 fraction of portfolio
    time_horizon_days: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    rationale: str
    counter_argument: Optional[str] = None  # from dissent layer
    features_snapshot: dict = Field(default_factory=dict)
    model_version: str = "v0"

    @field_validator("symbol")
    @classmethod
    def _upper_symbol(cls, v: str) -> str:
        return v.upper().strip()


class Trade(BaseModel):
    """A suggested or executed trade."""

    id: Optional[int] = None
    placed_at: datetime
    channel: Channel
    symbol: str
    instrument_type: InstrumentType
    side: PositionSide
    direction: SignalDirection  # buy/sell/etc.
    quantity: float
    price: float
    status: TradeStatus
    source_signal_id: Optional[int] = None
    fees: float = 0.0
    notes: Optional[str] = None


class TradeOutcome(BaseModel):
    """Retrospective scoring of a trade."""

    trade_id: int
    evaluated_at: datetime
    days_held: int
    return_pct: float
    max_favorable_excursion: float  # best unrealized gain during hold
    max_adverse_excursion: float    # worst unrealized loss during hold
    benchmark_return_pct: float     # SPY over same period
    alpha: float                    # trade return - benchmark return
    direction_correct: bool
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Risk snapshots
# ---------------------------------------------------------------------------


class RiskSnapshot(BaseModel):
    """Daily risk metrics for a channel."""

    channel: Channel
    snapshot_date: date
    portfolio_value: float
    var_95: float        # 1-day 95% Value at Risk (dollar amount)
    var_99: float
    cvar_95: float       # Conditional VaR (expected loss beyond VaR)
    cvar_99: float
    realized_vol_30d: float
    realized_vol_60d: float
    max_drawdown_60d: float
    avg_pairwise_correlation: float
    beta_to_spy: Optional[float] = None
    largest_position_weight: Optional[float] = None
    largest_sector_weight: Optional[float] = None


# ---------------------------------------------------------------------------
# News / events (Phase 2)
# ---------------------------------------------------------------------------


class NewsEvent(BaseModel):
    """A news article or geopolitical event."""

    id: Optional[int] = None
    occurred_at: datetime
    source: str
    headline: str
    body: Optional[str] = None
    url: Optional[str] = None
    tickers: list[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    relevance_score: Optional[float] = None  # 0.0 to 1.0
    raw: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calendar / regime
# ---------------------------------------------------------------------------


CalendarEventKind = Literal[
    "earnings", "fomc", "cpi", "jobs_report", "opex", "ex_dividend", "ipo_lockup_expiry"
]


class CalendarEvent(BaseModel):
    """Scheduled market-impacting event."""

    id: Optional[int] = None
    event_date: date
    kind: CalendarEventKind
    symbol: Optional[str] = None  # underlying for earnings/ex-div
    description: Optional[str] = None
    raw: dict = Field(default_factory=dict)


class RegimeClassification(BaseModel):
    """Macro regime label for a given day."""

    classification_date: date
    regime: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    features: dict = Field(default_factory=dict)
    model_version: str = "v0"
