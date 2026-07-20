-- AlphaEngine schema.
-- Targets DuckDB by default. PostgreSQL-portable: only generic DDL is used
-- (no DuckDB-only types). Options tables (options_chains, option_greeks) are
-- present but the system will not generate options signals until channel
-- config flips options_enabled to true.

-- ============================================================================
-- Reference data
-- ============================================================================

CREATE TABLE IF NOT EXISTS instruments (
    symbol           VARCHAR PRIMARY KEY,
    name             VARCHAR NOT NULL,
    instrument_type  VARCHAR NOT NULL,
    sector           VARCHAR,
    industry         VARCHAR,
    exchange         VARCHAR,
    currency         VARCHAR DEFAULT 'USD',
    active           BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Market data (time series)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_bars (
    symbol      VARCHAR NOT NULL,
    bar_date    DATE NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    adj_close   DOUBLE NOT NULL,
    volume      BIGINT NOT NULL,
    source      VARCHAR NOT NULL DEFAULT 'yfinance',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, bar_date)
);

CREATE INDEX IF NOT EXISTS idx_market_bars_date ON market_bars(bar_date);

-- ============================================================================
-- Macro / FRED time series
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_series (
    series_id   VARCHAR NOT NULL,
    obs_date    DATE NOT NULL,
    value       DOUBLE,                  -- nullable: FRED uses '.' for missing
    source      VARCHAR NOT NULL DEFAULT 'fred',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_id, obs_date)
);

CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_series(obs_date);

CREATE TABLE IF NOT EXISTS macro_series_meta (
    series_id    VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    units        VARCHAR,
    frequency    VARCHAR,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Options (dormant; data may still be collected for IV-based equity signals)
-- ============================================================================

CREATE TABLE IF NOT EXISTS options_chains (
    contract_symbol  VARCHAR PRIMARY KEY,   -- OCC-style
    underlying       VARCHAR NOT NULL,
    expiry           DATE NOT NULL,
    strike           DECIMAL(18, 4) NOT NULL,
    option_type      VARCHAR NOT NULL,      -- 'call' | 'put'
    contract_size    INTEGER DEFAULT 100,
    first_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_options_underlying ON options_chains(underlying, expiry);

CREATE TABLE IF NOT EXISTS option_greeks (
    contract_symbol     VARCHAR NOT NULL,
    snapshot_at         TIMESTAMP NOT NULL,
    delta               DOUBLE,
    gamma               DOUBLE,
    theta               DOUBLE,
    vega                DOUBLE,
    rho                 DOUBLE,
    implied_volatility  DOUBLE,
    bid                 DOUBLE,
    ask                 DOUBLE,
    underlying_price    DOUBLE,
    volume              BIGINT,
    open_interest       BIGINT,
    PRIMARY KEY (contract_symbol, snapshot_at)
);

-- ============================================================================
-- Positions, signals, trades
-- ============================================================================

CREATE SEQUENCE IF NOT EXISTS positions_id_seq START 1;
CREATE TABLE IF NOT EXISTS positions (
    id                BIGINT PRIMARY KEY DEFAULT nextval('positions_id_seq'),
    channel           VARCHAR NOT NULL,
    instrument_type   VARCHAR NOT NULL,
    symbol            VARCHAR,             -- nullable for option positions
    option_legs_json  VARCHAR,             -- JSON-serialized list[OptionLeg]
    side              VARCHAR NOT NULL,    -- 'long' | 'short'
    quantity          DOUBLE NOT NULL,
    entry_price       DOUBLE NOT NULL,
    entry_date        TIMESTAMP NOT NULL,
    stop_loss_price   DOUBLE,
    target_price      DOUBLE,
    notes             VARCHAR,
    source_signal_id  BIGINT,
    closed_at         TIMESTAMP,
    closed_price      DOUBLE,
    realized_pnl      DOUBLE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_channel_open
    ON positions(channel, closed_at);

CREATE SEQUENCE IF NOT EXISTS signals_id_seq START 1;
CREATE TABLE IF NOT EXISTS signals (
    id                  BIGINT PRIMARY KEY DEFAULT nextval('signals_id_seq'),
    generated_at        TIMESTAMP NOT NULL,
    channel             VARCHAR NOT NULL,
    symbol              VARCHAR NOT NULL,
    instrument_type     VARCHAR NOT NULL,
    direction           VARCHAR NOT NULL,
    conviction          DOUBLE NOT NULL,
    target_weight       DOUBLE,
    time_horizon_days   INTEGER,
    stop_loss_pct       DOUBLE,
    rationale           VARCHAR NOT NULL,
    counter_argument    VARCHAR,
    features_snapshot_json VARCHAR,
    model_version       VARCHAR NOT NULL DEFAULT 'v0',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_channel_time
    ON signals(channel, generated_at);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON signals(symbol, generated_at);

CREATE SEQUENCE IF NOT EXISTS trades_id_seq START 1;
CREATE TABLE IF NOT EXISTS trades (
    id                BIGINT PRIMARY KEY DEFAULT nextval('trades_id_seq'),
    placed_at         TIMESTAMP NOT NULL,
    channel           VARCHAR NOT NULL,
    symbol            VARCHAR NOT NULL,
    instrument_type   VARCHAR NOT NULL,
    side              VARCHAR NOT NULL,
    direction         VARCHAR NOT NULL,
    quantity          DOUBLE NOT NULL,
    price             DOUBLE NOT NULL,            -- canonical entry; 'next_open' style = adjusted open of D+1
    status            VARCHAR NOT NULL,
    source_signal_id  BIGINT,
    fees              DOUBLE NOT NULL DEFAULT 0.0,
    notes             VARCHAR,
    -- Execution-timing fields (2026-06-13): how the entry fill was modeled.
    -- 'next_open'  = adjusted open of the first trading day after the digest
    --                (realistic market-on-open fill; the current default).
    -- 'next_close' = legacy: adjusted close of that day (a full session of
    --                avoidable latency). alt_entry_price stores the OTHER
    --                style's price so the scorer can measure the gap's cost.
    entry_style       VARCHAR NOT NULL DEFAULT 'next_close',
    alt_entry_price   DOUBLE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_channel_time
    ON trades(channel, placed_at);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    trade_id                  BIGINT PRIMARY KEY,
    evaluated_at              TIMESTAMP NOT NULL,
    days_held                 INTEGER NOT NULL,
    return_pct                DOUBLE NOT NULL,
    max_favorable_excursion   DOUBLE NOT NULL,
    max_adverse_excursion     DOUBLE NOT NULL,
    benchmark_return_pct      DOUBLE NOT NULL,
    alpha                     DOUBLE NOT NULL,
    direction_correct         BOOLEAN NOT NULL,
    notes                     VARCHAR,
    -- Counterfactual return had we entered under the OTHER timing style
    -- (same exit). return_pct - alt_entry_return_pct = the per-trade value
    -- of the entry-timing choice. NULL for legacy trades lacking alt price.
    alt_entry_return_pct      DOUBLE,
    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Risk snapshots
-- ============================================================================

CREATE TABLE IF NOT EXISTS risk_snapshots (
    channel                   VARCHAR NOT NULL,
    snapshot_date             DATE NOT NULL,
    portfolio_value           DOUBLE NOT NULL,
    var_95                    DOUBLE NOT NULL,
    var_99                    DOUBLE NOT NULL,
    cvar_95                   DOUBLE NOT NULL,
    cvar_99                   DOUBLE NOT NULL,
    realized_vol_30d          DOUBLE NOT NULL,
    realized_vol_60d          DOUBLE NOT NULL,
    max_drawdown_60d          DOUBLE NOT NULL,
    avg_pairwise_correlation  DOUBLE NOT NULL,
    beta_to_spy               DOUBLE,
    largest_position_weight   DOUBLE,
    largest_sector_weight     DOUBLE,
    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (channel, snapshot_date)
);

-- ============================================================================
-- Calendar events (earnings, FOMC, OpEx, etc.)
-- ============================================================================

CREATE SEQUENCE IF NOT EXISTS calendar_events_id_seq START 1;
CREATE TABLE IF NOT EXISTS calendar_events (
    id           BIGINT PRIMARY KEY DEFAULT nextval('calendar_events_id_seq'),
    event_date   DATE NOT NULL,
    kind         VARCHAR NOT NULL,
    symbol       VARCHAR,
    description  VARCHAR,
    raw_json     VARCHAR,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calendar_events_date ON calendar_events(event_date);
CREATE INDEX IF NOT EXISTS idx_calendar_events_symbol_date
    ON calendar_events(symbol, event_date);

-- ============================================================================
-- Regime classifications
-- ============================================================================

CREATE TABLE IF NOT EXISTS regime_classifications (
    classification_date  DATE NOT NULL,
    regime               VARCHAR NOT NULL,
    confidence           DOUBLE NOT NULL,
    features_json        VARCHAR,
    model_version        VARCHAR NOT NULL DEFAULT 'v0',
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (classification_date, model_version)
);

-- ============================================================================
-- News / events
-- ============================================================================

CREATE SEQUENCE IF NOT EXISTS news_events_id_seq START 1;
CREATE TABLE IF NOT EXISTS news_events (
    id                BIGINT PRIMARY KEY DEFAULT nextval('news_events_id_seq'),
    occurred_at       TIMESTAMP NOT NULL,
    source            VARCHAR NOT NULL,
    headline          VARCHAR NOT NULL,
    body              VARCHAR,
    url               VARCHAR,
    tickers_json      VARCHAR,            -- JSON list of related tickers
    sentiment_score   DOUBLE,
    relevance_score   DOUBLE,
    raw_json          VARCHAR,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_occurred ON news_events(occurred_at);

-- ============================================================================
-- LLM signal cache (for reproducible / free backtest re-runs)
-- ============================================================================
--
-- Stores the full parsed digest JSON per (as_of, model_version). One row
-- covers BOTH channels (the digest produces channel_a + channel_b in one
-- call). Backtests read from here so a re-run costs nothing and is
-- deterministic. config_hash lets us invalidate when prompts/universe change.

CREATE TABLE IF NOT EXISTS llm_signal_cache (
    as_of           DATE NOT NULL,
    model_version   VARCHAR NOT NULL,
    config_hash     VARCHAR NOT NULL,
    output_json     VARCHAR NOT NULL,    -- full parsed primary_output
    universe_json   VARCHAR NOT NULL,    -- the universe used (for validation)
    input_tokens    BIGINT,
    output_tokens   BIGINT,
    cost_usd        DOUBLE,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of, model_version, config_hash)
);

CREATE INDEX IF NOT EXISTS idx_llm_cache_asof ON llm_signal_cache(as_of);

-- ============================================================================
-- Geopolitical signals (GDELT-derived)
-- ============================================================================
--
-- Daily timeseries per tracked query (Iran conflict, China-US trade, etc.).
-- volume_intensity is GDELT's 0-1 normalized article volume; avg_tone is the
-- -10..+10 average sentiment of articles matching the query that day.

CREATE TABLE IF NOT EXISTS geopolitical_signals (
    signal_name       VARCHAR NOT NULL,
    signal_date       DATE NOT NULL,
    volume_intensity  DOUBLE,
    avg_tone          DOUBLE,
    raw_query         VARCHAR,
    source            VARCHAR DEFAULT 'gdelt_doc',  -- 'gdelt_doc' | 'gdelt_bq'
    fetched_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_name, signal_date)
);

CREATE INDEX IF NOT EXISTS idx_geopolitical_date
    ON geopolitical_signals(signal_date);

-- ============================================================================
-- Signal performance / decay tracking (Phase 2 - schema ready)
-- ============================================================================

CREATE TABLE IF NOT EXISTS signal_performance (
    signal_name      VARCHAR NOT NULL,    -- e.g. 'momentum_12_1', 'pead'
    eval_date        DATE NOT NULL,
    information_coef DOUBLE,              -- IC over rolling window
    hit_rate         DOUBLE,
    sample_size      INTEGER,
    notes            VARCHAR,
    PRIMARY KEY (signal_name, eval_date)
);

-- ============================================================================
-- ML signal layer (cross-sectional momentum / XGBoost ranks)
-- ============================================================================
-- One row per (date, symbol, model_version): score, rank within that day's
-- cross-section, action bucket (BUY/HOLD/AVOID = top/middle/bottom quintile),
-- and the raw feature values behind the score. Same-day re-runs replace.

CREATE TABLE IF NOT EXISTS ml_signals (
    signal_date   DATE NOT NULL,
    symbol        VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    score         DOUBLE NOT NULL,
    rank          INTEGER NOT NULL,       -- 1 = most attractive
    n_universe    INTEGER NOT NULL,       -- rankable cross-section size that day
    action        VARCHAR NOT NULL,       -- BUY / HOLD / AVOID
    mom_12_1      DOUBLE,
    mom_6_1       DOUBLE,
    mom_3_1       DOUBLE,
    rev_1m        DOUBLE,
    vol_30d       DOUBLE,
    dist_50ma     DOUBLE,
    dist_200ma    DOUBLE,
    rsi_14        DOUBLE,
    generated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, symbol, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ml_signals_date ON ml_signals(signal_date);

-- ============================================================================
-- Action Center recommendation log (the Phase 3 learning loop)
-- ============================================================================
-- One row per (as_of, symbol, kind): what the Action Center's opportunity
-- layer recommended on a given day, plus the signals behind it. Scored
-- forward (score_recommendations) once `horizon` trading days elapse, so the
-- app can measure whether its own add/trim ideas actually worked — the only
-- honest way to learn which signal combinations to trust. Same-day re-runs
-- replace. Deterministic RISK trades are not logged here (they're not a
-- prediction — a cap breach is a cap breach); this table is for the
-- unproven, return-seeking IDEAS.

CREATE TABLE IF NOT EXISTS book_recommendations (
    as_of         DATE NOT NULL,
    symbol        VARCHAR NOT NULL,
    kind          VARCHAR NOT NULL,       -- 'add' | 'trim'
    score         DOUBLE,
    weight        DOUBLE,                 -- portfolio weight at reco time
    signals_json  VARCHAR,                -- the contributing signals
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of, symbol, kind)
);

CREATE INDEX IF NOT EXISTS idx_book_reco_date ON book_recommendations(as_of);
