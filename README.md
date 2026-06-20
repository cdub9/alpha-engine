# AlphaEngine

An AI-assisted asset trading system that generates trade suggestions for stocks, bonds, and ETFs (options support is built but disabled). Two parallel channels:

- **Channel A — Steady Alpha**: targets SPY + 3-5% on a risk-adjusted basis
- **Channel B — Aggressive Growth**: targets 2× SPY with concentrated tech/AI exposure

This repository is the foundation scaffold: data pipeline, storage, risk infrastructure, and core types. Signal generation, LLM integration, and execution come in later phases.

## Status

**Phase 1 — Foundation** (complete)
- [x] Project structure
- [x] Core types (positions, signals, market bars, option contracts)
- [x] DuckDB schema with options support (dormant)
- [x] FRED + yfinance data providers
- [x] Risk: VaR, CVaR, correlation, Kelly Criterion
- [x] GDELT geopolitical event ingestion — DOC API (recent, free) **and**
      BigQuery GKG backend (`scripts/ingest_gdelt_bq.py`) for cost-gated
      multi-year historical backfill (makes the layer backtestable)
- [x] Calendar features (FOMC, OpEx, earnings)
- [x] Macro regime classifier
- [x] Market + macro history backfilled to 2007 (covers GFC 2008 + COVID
      2020 so defensive strategies are judged on real bear markets)

**Phase 2 — Signal Generation** (in progress)
- [x] LLM digest pipeline (Claude API + caching)
- [x] ML signal layer (`alpha_engine/ml/`): cross-sectional momentum
      composite + walk-forward XGBoost, daily BUY/HOLD/AVOID ranks in
      `ml_signals`, validated honestly (see `scripts/validate_ml.py` and
      the dashboard's ML Signals page) — note: validation shows ETF-only
      momentum does NOT beat SPY 2008-2026; XGB beats SPY on the broad
      modern ETF window. Decision support + LLM cross-check, not a
      standalone strategy.
- [ ] Rules layer (hard filters by regime)
- [x] Dissent layer (counterargument generator)

**Phase 3 — Portfolio & Execution** (in progress)
- [ ] Portfolio construction
- [ ] Position management / exit logic
- [ ] Paper trading via Alpaca
- [x] Realistic entry timing: paper trades fill at the next session's
      **open** (a market-on-open order — the tightest honest fill given the
      post-close digest), not the next close. Each trade records the
      counterfactual close-entry return so the dashboard's "Execution
      timing" panel measures the latency saved (historically +0.57%/trade).
- [x] Self-learning review loop (`alpha_engine/llm/feedback.py`): every
      digest sees the model's open paper book and its scored track record
      (conviction calibration, recent outcomes, repeated hits/misses) and
      is instructed to recalibrate. Automatic — the nightly scorer feeds
      the next morning's snapshot.

## Quick start

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .

# Set up environment (copy and edit)
copy .env.example .env

# Initialize database and verify everything works end-to-end
python scripts/verify_setup.py
```

The `verify_setup.py` script will:
1. Create the DuckDB database at `data/alpha_engine.duckdb`
2. Pull ~5 years of SPY price history via yfinance
3. Pull a few FRED macro series (CPI, unemployment, Fed funds rate, yield curve)
4. Compute VaR, CVaR, and correlation on the sample data
5. Demonstrate the Kelly Criterion sizer
6. Print results so you can confirm everything is working

## Configuration

- `config/settings.yaml` — global settings (database path, log level, API keys via env vars)
- `config/channels.yaml` — Channel A and Channel B definitions (limits, instruments, options gating)
- `config/universe.yaml` — tradable universe (sectors, market caps, exclusions)

API keys (set in `.env`):
- `FRED_API_KEY` — free from https://fred.stlouisfed.org/docs/api/api_key.html
- `ANTHROPIC_API_KEY` — for the LLM layer (Phase 2)
- `NEWS_API_KEY` — optional; for Phase 2

### Geopolitical data: DOC API vs BigQuery

Two interchangeable backends write the same `geopolitical_signals` rows:

- **DOC API** (`scripts/ingest_gdelt.py`) — full-text, free, no auth, but a
  ~30-day rolling window only. Used by the nightly run.
- **BigQuery GKG** (`scripts/ingest_gdelt_bq.py`) — years of history (2015+),
  no rate limits, for historical backfill that makes the layer
  backtestable. Bills by bytes scanned, so the script **dry-runs and
  prints a cost estimate first**, refusing any chunk over `--max-gb`.
  Setup:
  ```bash
  pip install -e ".[bigquery]"
  gcloud auth application-default login
  python scripts/ingest_gdelt_bq.py --project YOUR_GCP_PROJECT --dry-run
  ```

## Database

DuckDB is the default — it's embedded (no server required), columnar, and fast for analytical queries. The schema is PostgreSQL-compatible, so migrating to PostgreSQL + TimescaleDB later requires only the connection layer to change.

DB file location: `data/alpha_engine.duckdb` (gitignored).

## Architecture

```
alpha_engine/
├── core/           Types, config, logging
├── data/           Provider clients (FRED, yfinance, GDELT future)
├── db/             Schema + connection management
├── risk/           VaR, correlation, Kelly, exposure
└── pipelines/      Daily ingest orchestration
```

## Options support

Options are first-class in the schema and type system but gated by `options_enabled: false` in each channel config. When enabled, the signal generator will be allowed to suggest options positions. Implied volatility data is collected regardless — it feeds equity timing signals (IV rank, term structure).

## License

Private / personal use.
