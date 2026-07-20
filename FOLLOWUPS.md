# Follow-ups

Things we noticed but deferred. Add new items at the bottom of the relevant
section. When closing an item, move it to the bottom under `## Done` with a
date.

Each item is:
  - **Title** — one-line summary
  - **Why it matters** — what could go wrong / what we're missing
  - **How to address** — specific action
  - **Added** — when it was flagged, and in what context

---

## Data quality

### CPI release dates are heuristic
- **Why it matters:** Real BLS calendar varies (sometimes Wed/Thu of week 2,
  occasionally shifted by holidays). Heuristic = 2nd Tuesday of month is
  usually right but will be wrong ~10-20% of the time. Calendar features
  like `days_to_next_cpi` and `is_cpi_week` will be off on those dates.
- **How to address:** Scrape or download the [BLS scheduled releases](https://www.bls.gov/schedule/news_release/cpi.htm)
  and replace the heuristic in `alpha_engine/calendars/scheduled.py:cpi_release_dates`.
  Same applies to PPI, jobs report if BLS shifts dates around holidays.
- **Added:** 2026-05-28 (Phase 2 calendar module)

### FOMC 2027 dates are placeholders
- **Why it matters:** The Fed publishes meeting dates ~18 months in advance.
  The 2027 dates currently in `FOMC_MEETINGS` are guessed from the typical
  cadence (8 meetings/year, roughly the same weeks). Real dates may shift
  by days. If we backtest 2027-onward, `days_to_next_fomc` will be wrong.
- **How to address:** When the Fed publishes the 2027 calendar (usually mid-2026),
  replace the placeholder dates in `alpha_engine/calendars/scheduled.py:FOMC_MEETINGS`.
  Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- **Added:** 2026-05-28 (Phase 2 calendar module)

### No trading-calendar holidays
- **Why it matters:** All `days_to_X` calendar features use raw calendar
  days, not trading days. Fine for medium-horizon (weekly) signals, but
  if we ever do daily or intraday work, we'll be off by 1-3 days around
  holidays (and Friday OpEx that follows a Thursday holiday is wrong).
- **How to address:** Add `pandas_market_calendars` or `exchange_calendars`
  dependency, build a `trading_days_between(d1, d2)` helper, and use it
  in `alpha_engine/calendars/features.py` for distance computations.
- **Added:** 2026-05-28 (Phase 2 calendar module)

### Earnings coverage is only equity tickers (10 of 34)
- **Why it matters:** ETFs don't have earnings, so this is correct *today*.
  But when we expand the universe beyond the Mag 7 + AI infra basket, the
  earnings calendar needs to grow proportionally. yfinance has rate limits
  and is unofficial — at 200+ tickers we'll start seeing failures.
- **How to address:** When universe grows past ~50 tickers, batch the
  earnings calendar pull (or switch to a paid provider like
  [Earnings Whispers](https://www.earningswhispers.com/) or Polygon).
  Add retry logic in `alpha_engine/calendars/earnings.py`.
- **Added:** 2026-05-28 (Phase 2 calendar module)

---

## Regime classifier

### No "recovery" classifications post-2024 Sahm trigger
- **Why it matters:** When Sahm Rule triggered in mid-2024 then faded, the
  timeline jumped straight from `recession` back to `expansion_low_vol` without
  passing through `recovery`. The recovery rule requires unemployment to be
  *falling* (3m avg < 6m avg by 0.05+). In 2024 unemployment stayed elevated
  rather than fell, so the rule correctly didn't fire — but if downstream
  signals weight on `recovery` (e.g. "buy small caps coming out of recession"),
  they'll miss that opportunity in similar fast-fade scenarios.
- **How to address:** Consider a softer recovery trigger like "Sahm fell from
  peak by 0.2+ AND unemployment not rising sharply." Validate against 1990,
  2001, 2008-09, 2020 recoveries.
- **Added:** 2026-05-28 (Phase 2 regime classifier)

### ~~Vol-regime flicker at the VIX 20 boundary~~ — RESOLVED
- **Why it matters:** Several short (1-3 week) regime swaps between
  `expansion_low_vol` and `expansion_high_vol` when 30-day VIX hovers near 20.
  Pure threshold rule with no hysteresis. Downstream signals reading the
  current regime could end up flipping their stance every 2 weeks on noise.
- **How to address:** Add hysteresis — require VIX to cross 22 (going up) or 18
  (going down) to flip regime, not just touch 20. Or smooth with a longer MA.
- **Added:** 2026-05-28 (Phase 2 regime classifier)
- **Resolved 2026-05-31:** Added `prior_regime` parameter to `classify()` plus
  `get_prior_regime()` helper. Thresholds: low→high requires VIX>22, high→low
  requires VIX<18. Between 18-22 stays sticky. When `prior_regime` is None or
  was not an expansion sub-regime, classic 20 threshold applies (backward
  compatible). 4 call sites updated: `scripts/classify_regimes.py` (threads
  prior across the loop), `alpha_engine/llm/context.py:build_snapshot`,
  `alpha_engine/backtest/advisors.py:RegimeDefensive`, and
  `alpha_engine/backtest/advisors.py:RegimeWithTrendConfirmation` (live ones
  fetch from DB). Re-classified the full 18y history (961 weekly dates).
  Result: 9/9 unit cases pass; vol-flicker flips dropped from 29 → 19
  (~34% reduction) on real history.

### Single ML-free classifier; no probabilistic regime blending
- **Why it matters:** Current classifier is hard-rule first-match. Returns one
  regime + a heuristic confidence. Reality is messier — a given date might be
  60% late_cycle / 30% recession / 10% expansion. ML approaches (HMM, Gaussian
  mixture) naturally output probability vectors, which downstream signals
  could blend.
- **How to address:** Build `rule_v2` that returns a probability distribution
  across all regimes (softmax-style scoring), keeping `rule_v1` for sanity.
  Or layer a proper HMM on the same features. Compare backtested signal
  performance under each.
- **Added:** 2026-05-28 (Phase 2 regime classifier)

---

## Backtesting

### RegimeDefensive massively underperforms in 2022-2024
- **Why it matters:** The regime classifier correctly identified late_cycle
  for 115 weeks (March 2022 - May 2024), and the RegimeDefensive advisor
  responded by going 70% TLT / 30% SPY. But equities rallied through that
  whole period (Mag 7 / AI boom). Result: +8.8% vs +67% for SPY. Naive
  regime-switching alone is dangerous because macro signal can be right
  about the *condition* but wrong about the *price action*.
- **How to address:** Partially addressed 2026-05-29 — added
  `RegimeWithTrendConfirmation` advisor that requires both bearish macro
  AND negative SPY trend (below 200-day MA). Recovered ~33pp (+8.8% →
  +41.8%) but still underperforms buy-and-hold in this bull window.
  Further work: test against periods with real bear markets, consider
  graduated sizing instead of binary trend signal.
- **Added:** 2026-05-29 (Phase 2 backtest validation)

### Backtest history doesn't cover real bear markets
- **Why it matters:** Our 5-year backfill (May 2021 - present) includes
  2022's mild bear (-19%) but misses 2008 GFC (-55%), 2020 COVID flash
  crash (-34%), 2000-02 dot-com (-49%). Defensive strategies look bad
  in our window because it's a 5-year period dominated by rallies.
  Can't honestly evaluate downside protection without including
  downside.
- **How to address:** Extend backfill to 2007 or earlier for SPY/TLT/AGG
  (yfinance has the data, FRED has full macro history). Rerun
  comparison — `regime_with_trend` should look much better when 2008
  and 2020 are included. UNRATE/Sahm/yield curve also have decades of
  history.
- **Added:** 2026-05-29 (Phase 2 backtest validation)

### Parameter sweep on SMA window doesn't survive walk-forward
- **Why it matters:** A walk-forward sweep of `sma_window ∈ {50, 100, 150,
  200, 250}` underperformed the fixed default SMA=200 on test windows
  (Sharpe 0.78 vs 0.80, total +269% vs +287%). Training-window best-SMA
  had essentially no correlation with test-window performance. Concrete
  example: walk 2021-22 picked SMA=50 (highest train score = 1.072), got
  -19.9% in the test window. Adding "smart" param selection here would
  have actively destroyed value.
- **How to address:** Keep SMA hardcoded at 200. Before adding any
  parameter tuning to any future strategy, validate via walk-forward
  first — otherwise we're optimizing on noise. This is a general
  principle, not just for this strategy.
- **Added:** 2026-05-29 (Walk-forward validation phase)

### Strategy edge is concentrated, not broad-based
- **Why it matters:** Era analysis shows `regime_with_trend` produces
  ~all its alpha in crisis windows (GFC 2008-09: +6.5% vs SPY -19.4%).
  In normal/rally periods (COVID rally, AI rally) it badly trails
  buy-and-hold. Trading honest assessment: it's an insurance policy,
  not a return enhancer. Average alpha over 18 years OOS is only +0.2%.
- **How to address:** Consider a portfolio-of-strategies approach: hold
  70% buy-and-hold + 30% regime_with_trend so we capture the crisis
  protection without giving up much rally upside. Or use regime as a
  *sizing* signal on top of buy-and-hold rather than asset switching.
- **Added:** 2026-05-29 (Walk-forward validation phase)

### `regime_defensive` (no trend confirmation) wins the bull cycle 2010-18
- **Why it matters:** Counterintuitively, the original "naive"
  RegimeDefensive (+228% in 2010-18) beat the trend-confirmed version
  (+181%). During stable expansion the regime classifier was already
  right; the trend filter just added latency. Trend confirmation is
  good insurance for crisis/regime-shift periods but costs you when the
  underlying signal is already accurate.
- **How to address:** Possibly use trend confirmation only as a
  defensive *check* (gate going to bonds) but not as a risk-on *check*
  (don't require trend up to be in stocks). Asymmetric application of
  the confirmation rule.
- **Added:** 2026-05-29 (Walk-forward validation phase)

### Binary trend signal; no graduation
- **Why it matters:** `spy_trend_at` returns just price and SMA — the
  advisor uses a hard above/below threshold. Reality is a spectrum —
  SPY 0.5% above its 200-day MA is functionally the same as 0.5% below.
  The binary threshold causes whipsaws around the MA.
- **How to address:** Use distance from MA (e.g. (price - sma) / sma)
  and graduate the defensive tilt linearly. Or use a slope/regression
  approach (is the MA itself rising or falling?).
- **Added:** 2026-05-29 (Phase 2 backtest validation)

### ~~Risk-free rate hardcoded to 0 in Sharpe~~ — RESOLVED
- **Why it matters:** Sharpe ratio uses (return - rf) / vol. We use rf=0
  for simplicity, which overstates Sharpe during high-rate periods. In
  2022-2024, T-bills paid 4-5% — a Sharpe of 0.71 for SPY buy-and-hold
  becomes ~0.30 once you subtract the real risk-free rate.
- **How to address:** Pull DGS3MO (3-month T-bill) from FRED, take the
  average over the backtest period, pass to `compute_metrics(rf=...)`.
  All four reference advisors are evaluated apples-to-apples currently,
  so the comparison is still valid, but the absolute Sharpe numbers
  are flattering.
- **Added:** 2026-05-29 (Phase 2 backtest)
- **Resolved 2026-05-31:** Added DGS3MO to `config/settings.yaml` and pulled
  5,064 obs from FRED (2007-2026, lifetime avg 1.57%, since 2024 avg 4.53%).
  Added `fetch_risk_free_rate(con, start, end)` to `metrics.py`. Engine and
  walkforward pass it to `compute_metrics(risk_free_rate=...)`. Verified
  on 2024-2025 SPY buy-and-hold: Sharpe dropped 1.33 → 1.04 (the bias the
  followup warned about).

### No survivorship bias correction
- **Why it matters:** The universe (`config/universe.yaml`) is today's list
  of stable ETFs + mega-cap names. Backtesting "what NVDA did from 2022"
  is fine because NVDA didn't go bankrupt. But when the universe grows
  to include small caps or speculative names, results will be biased
  upward (delisted/bankrupt names disappear from our history).
- **How to address:** When expanding beyond stable mega-caps, use a
  point-in-time index membership data source (e.g. Sharadar, CRSP). For
  now, stay within stable instruments and document the assumption.
- **Added:** 2026-05-29 (Phase 2 backtest)

### Close-to-close execution; no intraday model
- **Why it matters:** Trades execute at the same day's close as the
  decision (with 1 trading-day signal lag for look-ahead avoidance).
  Real execution involves bid-ask, intraday volatility, and timing
  effects. For weekly+ rebalances this is fine; intraday strategies
  would need a much more careful execution model.
- **How to address:** When/if we add intraday signals, build an
  execution simulator that uses minute-bar data and models VWAP/TWAP
  fills. Out of scope until then.
- **Added:** 2026-05-29 (Phase 2 backtest)

### Default `max_position_weight` was 0.25 — silent backtest sabotage
- **Why it matters:** Original default capped any single position at 25%,
  which silently shrank `buy_and_hold_spy` from 100% SPY to 25% SPY + 75%
  cash. Took several minutes of debugging to spot. Risk caps as opt-out
  defaults are dangerous — they distort backtests without warning.
- **How to address:** Already fixed — default is now 1.0 (no cap).
  Real risk caps come from `config/channels.yaml` per channel. Worth
  general principle: any "safe default" in a backtester that changes
  the strategy's behavior should be loud (log a warning) or absent.
- **Added:** 2026-05-29 (Phase 2 backtest)

---

## Universe / coverage

### Expand the tradable universe well beyond the current 35 tickers — PARTIALLY RESOLVED
- **Why it matters:** Current universe is ~35 mega-cap names + sector/bond
  ETFs. To beat the S&P 500 over time we need a much wider opportunity
  set — mid/small caps, recent IPOs, international (EFA/EEM/sector-country
  ETFs), themed funds (ARKK, SMH, IBB, ICLN), specific sub-sectors
  (semis, biotech, defense, AI infrastructure, cybersecurity), commodities
  (USO, UNG, DBA, DBC). Most alpha exists in less-watched names; the
  Mag 7 are the most-arbitraged stocks on the planet.
- **How to address:** Phased expansion:
  1. Add 20-30 high-quality liquid mid-caps from key sectors (Phase A)
  2. Add 30+ thematic / sub-sector ETFs (Phase B)
  3. Add full S&P 500 backfill (~500 tickers) once we have batching
     infrastructure for yfinance and rate-limit handling (Phase C)
  4. Eventually: international + commodities (Phase D)
  Will need: better yfinance throttling, point-in-time index membership
  data (survivorship bias correction — already a separate followup),
  more efficient bulk-storage patterns in DuckDB. Budget: pulling 500
  tickers × 20 years is ~2.5M bars, easily fits.
- **Added:** 2026-05-29 (post-walk-forward direction setting)
- **Priority:** HIGH — every signal we build is limited by the universe
  it can see. Worth doing before too many more strategies.
- **Partial resolution 2026-05-30:** Universe expanded from 38 → 58 active
  instruments. Added: cheap substitutes (QQQM, VOO); semis (SMH, SOXX);
  thematic ETFs (CIBR, SKYY, BOTZ, IBB, XBI, PAVE, ITA, KWEB); mid/small
  cap + rate-sensitive (IJH, IJR, KRE, XHB); international (EFA, VWO,
  INDA, EWJ). Backfilled 5y of bars for all 20 new tickers. System prompt
  gained an "operating principle 7" nudging the model toward lower-expense
  siblings (QQQM > QQQ, VOO > SPY).
- **Phase B 2026-05-30:** Added 45 individual large-cap equities across
  underrepresented sectors (ORCL/CRM/ADBE/NOW/INTU/NFLX/UBER/PLTR/CRWD/PANW
  for tech; JPM/BAC/GS/V/MA/BLK/COIN for financials; LLY/UNH/JNJ/ABBV/MRK
  for healthcare; HD/LOW/MCD/NKE/BKNG for consumer disc; WMT/COST/PG/KO
  for staples; XOM/CVX/COP/OXY for energy; CAT/DE/RTX/LMT/BA for
  industrials; TMUS for comm; FCX/NEM for materials; PLD/AMT for REITs).
  Universe is now **103 active instruments**. 56,475 new bars.
- **Phase C 2026-05-30:** Bars-only backfill for the remaining 448 S&P
  500 constituents (current list, ~503 total minus our 103 already in
  universe). These symbols are queryable in `market_bars` for backtests
  but are NOT in the `instruments` table — keeps the LLM's snapshot
  context manageable while making the data ready for future selective
  inclusion (e.g. "biggest movers today + earnings this week" curated
  universe).

### Survivorship-bias correction (needed before backtesting individual names) — PARTIAL: WARNING SHIPPED
- **Why it matters:** Phase B/C added today's S&P 500 constituents.
  Companies delisted, removed from index, or bankrupted since 2021 are
  missing from history. Backtests on individual names will overstate
  returns. SPY/VOO backtests stay clean (they hold the historical
  composition by construction).
- **How to address:** Need point-in-time index membership data (Sharadar,
  CRSP, Norgate). Until then: (a) restrict individual-name backtests to
  ETFs + Mag 7, (b) flag any backtest report that includes Phase B/C
  symbols with a "survivorship bias not corrected" warning.
- **Added:** 2026-05-30 (Phase B/C universe expansion)
- **Partial 2026-05-30:** Added `alpha_engine/backtest/warnings.py` with
  `affected_symbols()` + `survivorship_warning_text()`. Wired into:
  (a) `scripts/backtest.py` — panel before run if any equity in cfg.universe;
  (b) `scripts/backtest_llm.py` — panel before run (full_universe always
  triggers it); (c) dashboard Track Record page — expandable warning
  banner if any scored trade is an individual equity. `walkforward.py`
  uses the all-ETF DEFAULT_UNIVERSE so warning never fires there (correct).
  This is a *warning*, not a *fix* — the underlying data limitation remains.
  Real fix still requires point-in-time index membership.

---

## LLM digest

### ~~Dissent calls dominate cost (~$0.75 of $0.90 per run)~~ — RESOLVED
- **Why it matters:** First real digest run cost $0.90, dominated by 30
  separate Opus dissent calls. Now: $0.16 total ($0.0072 dissent cost),
  82% reduction. Monthly at 1 run/day went from $27 → $4.80.
- **How addressed:** All three planned fixes shipped 2026-05-29:
  Haiku 4.5 for dissent, threshold raised from 6.0 to 7.5, and batched
  into one API call instead of N. Quality remained sharp; bonus
  observation — adjustment variation also improved (no longer all -2)
  after explicit prompt nudge to vary severity.
- **Resolved:** 2026-05-29

### ~~Dissent layer is formulaic — almost always returns -2 adjustment~~ — RESOLVED (as side effect of batching)
- **Why mattered:** Per-suggestion dissent calls returned -2 on 28 of 30
  initial run. Made the dissent layer feel like noise instead of signal.
- **How addressed:** Two changes shipped together 2026-05-29:
  1. Batch dissent — model now sees all challenged suggestions in one
     call and can compare/contrast severity across them.
  2. Updated system prompt explicitly forbids "default to -2 for
     everything" and walks through when each value applies.
  First post-fix run produced varied adjustments (-1, -2, -2, -1) on
  4 suggestions. Watching to see if it holds across more runs.
- **Resolved:** 2026-05-29

### ~~Dissent system prompt is too short to cache~~ — RESOLVED (moot post-batching)
- **Why mattered:** Opus 4.7's 4096-token cache minimum meant the small
  dissent system prompt couldn't benefit from caching, costing full price
  on each of the 30 dissent calls per run.
- **How addressed:** No longer applies. With batch dissent we make exactly
  one dissent call per run, so there's nothing to cache *across*. The 
  intra-run caching that would have mattered (re-using the system prompt
  across N calls) is now a non-issue.
- **Resolved:** 2026-05-29 (made moot by batch dissent)

### ~~Same-day re-runs duplicate signals in DB~~ — RESOLVED
- **Why it matters:** Each digest run writes new rows to `signals` with
  the current `generated_at` timestamp. If we run twice in a day (e.g. to
  test or after a market-moving event), the DB has both sets — and
  downstream analytics will double-count. There's no upsert key like
  `(date, channel, symbol, model_version)`.
- **How to address:** Either (a) delete existing same-day rows before
  insert, (b) add a partial unique index on
  `(DATE(generated_at), channel, symbol, model_version)` and use ON
  CONFLICT, or (c) keep all runs but tag with `run_id` so we can group.
  Option (b) is cleanest for normal daily operation; option (c) is
  better if we want to A/B test prompt versions.
- **Added:** 2026-05-29 (first LLM digest run)
- **Resolved:** 2026-05-30 — `persist_signals` now DELETEs prior same-day rows for `(DATE(generated_at), model_version)` before insert. Second run replaces first cleanly; logged as `signals_same_day_replaced`.

### Re-runs across DIFFERENT generated-dates still double-open paper trades
- **Why it matters:** The same-day dedup above keys on
  `(DATE(generated_at), model_version)`, so it only collapses re-runs that
  land on the SAME generated-date. But a manual `run-day` re-run (or the
  v1→v3 backfill) can persist the same logical pick under a *different*
  `generated_at` date — backfill stamps the as_of's midnight UTC, a live
  run stamps `now()`. Those escape the same-day DELETE, so two signals for
  the same `(channel, symbol, direction, entry-date)` survive, and the
  orphan-open logic in `run-day` (open signals where no trade exists yet)
  opens a paper trade for BOTH. Result: duplicate trades that inflate the
  track record. Found + cleaned 39 such trades (06/01, 06/10, 06/18 cohorts)
  on 2026-06-27 via the new `scripts/dedup_paper_trades.py`; root cause left
  unfixed so it can recur on the next manual re-run.
- **How to address:** Make trade-opening idempotent on the ENTRY identity,
  not the signal row: before `open_paper_trades_for_date` inserts, skip any
  `(channel, symbol, direction, entry-date)` that already has a paper_filled
  trade (regardless of which signal it came from). Or dedup signals on
  `(channel, symbol, direction, as_of)` instead of `DATE(generated_at)`.
  Either makes re-runs safe without the manual dedup pass. Low priority
  while re-runs are rare and the dedup script exists, but it's a real
  latent data-integrity bug.
- **Added:** 2026-06-27 (found while auditing the twice-daily scheduled run)

### Cache hit rate hasn't been verified end-to-end — TOOLING SHIPPED, NOT YET RUN
- **Why it matters:** We set `cache_control: {type: "ephemeral"}` on the
  primary system prompt (~1100 tokens) and the first run showed
  `cache_creation_input_tokens=3169` (cache was written). But we never
  ran a second digest within the 5-minute TTL to confirm
  `cache_read_input_tokens > 0`. Worth verifying the caching is actually
  saving money.
- **How to address:** Add a `--repeat N` flag to `run_digest.py` that
  fires N requests in quick succession, then assert that runs 2..N show
  non-zero `cache_read_input_tokens`. Or just manually run twice within
  5 min and inspect the cost summary.
- **Added:** 2026-05-29 (first LLM digest run)
- **Tooling 2026-05-31:** Added `--repeat N` flag to `run_digest.py`. Runs
  N back-to-back digests for the same date, prints a verification table
  comparing `cache_creation_tokens` vs `cache_read_tokens` per run, and
  emits a green/red verdict based on whether runs 2..N show cache hits.
  Not yet run — would cost ~$0.30 for `--repeat 2`. Run on demand:
  `python scripts/run_digest.py --repeat 2 --no-persist --no-dissent`

### ~~Add technical analysis features to the snapshot~~ — SHIPPED 2026-06-11
- **Why it matters:** Today the snapshot gives the LLM raw 1d/5d/30d %
  changes and one trend feature (SPY 200-day MA). The model implicitly
  re-derives TA-style reasoning from price action ("AMD +75% 30d,
  AI/semis leadership intact"). Giving it cleaner, decision-ready TA
  inputs frees prompt budget for synthesis and provides a documented
  confirmation/disconfirmation lens. Pure compute from existing bars,
  no API cost, no new data dependency.

- **Pros:**
  1. Free — pure math on existing market_bars.
  2. Compresses information (90 days of bars → 1 RSI number).
  3. Short-to-medium-term momentum (3-12 month) is the most empirically
     validated technical signal. Trend-following on liquid ETFs is real.
     RSI oversold + mean-reversion works in chop regimes.
  4. Gives the model crisis-detection vocabulary ("SPY breaks 200-MA +
     RSI<30" is a real signal worth flagging).
  5. Dashboard chart overlays (MA lines, RSI subplot) are free
     diagnostic aids even if the LLM ignores them.

- **Cautions:**
  1. Most TA is noise dressed up as signal — academic consensus on
     pattern recognition, niche oscillators (Aroon, Williams %R,
     Ichimoku). Avoid those.
  2. Curve-fitting trap — we already learned with SMA walk-forward.
     Stick with conventional defaults (14-RSI, 50/200-MA). Do NOT
     optimize periods on historical data.
  3. Attention dilution — more numbers per symbol = more for the LLM
     to weigh. Could degrade output if we go wide.
  4. TA can encourage overtrading (oscillators fire often).
  5. TA can override correct fundamental thesis at the wrong moment
     (Mag 7 was "technically overbought" for years and kept going).
  6. Context budget grows ~75% per symbol if we go wide. At 115 syms
     × 5 new numbers = ~575 extra numbers ≈ 2000 more tokens. Cost
     per digest goes from $0.15 → ~$0.17.

- **What to add (narrow, validated only):**

  Per-symbol Tier 1:
  - Distance from 50-day SMA: `(price - sma_50) / sma_50` — trend strength
  - Distance from 200-day SMA — long-term trend
  - 14-day RSI — most validated oscillator (>70 overbought, <30 oversold)
  - 30-day annualized realized volatility — risk context per name

  Universe-wide Tier 2:
  - Breadth: % of universe above own 50-day MA — broad trend confirmation
  - Average 30-day return — universe-wide momentum read

  Skip entirely (Tier 3):
  - Bollinger Bands, MACD divergence (redundant with RSI + distance-from-MA)
  - Pattern recognition (head-and-shoulders, double tops) — not
    codifiable, not validated
  - Anything intraday — we have daily bars only
  - Volume signals — yfinance volume data is unreliable cross-asset

- **System prompt change (single new operating principle):**
  > "**TA is confirmation, not a primary signal.** Use distance-from-MA
  > as a trend filter (don't fight a -10% deviation), RSI as a sizing
  > modulator (≥75 = trim, ≤25 = consider adding), vol as a sizing
  > input. Never let a single TA reading drive a buy/sell — combine
  > with regime, calendar, and the fundamental thesis."

  This positions TA as one input among several, not a black-box signal.

- **Dashboard changes:**
  - Suggestion cards: small inline tags ("trend +18% above 50-MA, RSI 76")
  - Open Trades drill-down chart: overlay 50-MA + 200-MA lines, RSI subplot
  - Lookup page chart: same overlays — diagnostic for any ticker

- **Effort estimate:** ~5-6 hours, $0 in API cost. Breakdown:
  1. `alpha_engine/features/technicals.py` module (compute features
     cleanly with unit tests) — 2-3 hr
  2. Snapshot integration as a new "Per-symbol technicals" section
     (don't inline; keeps price section readable) — 1 hr
  3. System prompt update (one new operating principle) — 30 min
  4. Dashboard chart overlays (MA lines + RSI subplot) — 2 hr
  Plus ~$0.30 to burn one or two paid digest runs A/B testing new vs
  old prompt on the same day.

- **Timing recommendation:** WAIT 2-4 weeks after Monday's first
  forward digest before adding this. Reason: changing the prompt the
  same week we start collecting forward paper-trading data muddies
  the data — we couldn't tell if any output-quality change was the
  TA addition or just regime shift. Cleaner sequence:
  1. Let 2-4 weeks of forward data accumulate on CURRENT prompt
  2. Add TA as an explicit "prompt v2" change with a date marker
  3. Compare pre- vs post-TA paper trading stats over the following
     2-4 weeks

- **Validation plan (when we ship it):**
  - Tag signals with `prompt_variant='v2_with_ta'` so we can group
  - Compare alpha distribution + win rate on conv≥7.5 picks before
    and after the change
  - Watch for: did high-conviction picks with confirming TA outperform
    high-conviction picks without? Did the LLM start chasing momentum
    into tops?
  - If neutral or negative on LLM output but the dashboard overlays
    are useful, roll back the prompt change but keep the overlays.

- **Added:** 2026-05-31 (after expanded discussion on whether TA is
  worth adding as an LLM input dimension)
- **Priority:** MEDIUM — likely-positive bet but unproven; should not
  preempt forward-data accumulation.
- **SHIPPED 2026-06-11** (at user request, ~1.5 weeks into the forward
  window rather than the 2-4 weeks this proposal suggested — mitigated by
  the version tag, which keeps pre/post cohorts cleanly separable):
  - Snapshot: new "Per-symbol technicals" section (Tier 1: dist 50/200-MA,
    RSI-14, 30d vol per symbol; Tier 2: breadth % above 50-MA + avg 1-month
    return + RSI>=75 / <=25 extremes lines). Reuses alpha_engine/ml/features
    (one tested implementation). Measured: ~6.2k chars ≈ 1,550 tokens,
    within the proposal's estimate.
  - Prompt: new operating principle 8 ("TA is confirmation, not a primary
    signal"), old 8 renumbered to 9. config_hash includes the prompt text,
    so the digest cache busts automatically.
  - Version tag: DEFAULT_MODEL_VERSION and persist_signals default bumped
    "llm-opus-4-7-v1" → "llm-opus-4-7-v2-ta". v1 rows untouched.
  - Dashboard: suggestion cards show inline TA tag (trend vs 50-MA, RSI);
    Open Trades drill-down chart gained 50/200-MA overlays + RSI subplot;
    Lookup page gained a "Technical view" expander with the same.
  - Tests: tests/test_ta_snapshot.py (8 tests: section content, breadth
    math, insufficient-history exclusion, point-in-time, prompt/version).
  - NOT yet done from the proposal: the ~$0.30 A/B digest comparison.
    Tonight's auto-run will produce the first v2-ta digest at normal cost;
    compare conv>=7.5 alpha/win-rate across model_version after 2-4 weeks
    of v2 data accumulates.

### A/B test Sonnet 4.6 on the primary digest to cut run cost — PLUMBING + TEST TOOL SHIPPED
- **Why it matters:** The primary digest call runs on `claude-opus-4-7`
  (the client's `DEFAULT_MODEL`) at ~$0.16/run, ~$4.80/mo at one run/day.
  `claude-sonnet-4-6` is already priced in the client at $3/$15 per 1M
  tokens (`alpha_engine/llm/client.py:_MODEL_PRICING`) vs Opus 4.7's
  higher rate — roughly a 40-50% cut on the primary call. If Sonnet's pick
  quality holds on this task, that's a free recurring saving; the dissent
  layer already took the same "cheaper model where it's good enough" bet
  (Haiku 4.5) and held quality. The open question is purely whether Sonnet
  picks as well as Opus here, which only an A/B answers.
- **How to address:**
  1. Thread a `primary_model` arg through `run_digest` (digest.py:65) into
     `client.call_structured(...)` — the client method already accepts a
     `model=` param and gates features per-model, so this is a one-arg
     passthrough, not new infrastructure. Expose it as a `--primary-model`
     flag on `scripts/run_digest.py`.
  2. Tag the cohort distinctly: pass `model_version="llm-sonnet-4-6-v3-fb"`
     to `persist_signals` so Sonnet signals never mix with the Opus cohort.
     This keeps them separable in EVERY downstream view for free — the
     forward-validation scorers shipped 2026-06-19
     (`feedback_eval`/`forward_eval`) and the conviction-calibration table
     all group by `model_version`, so a Sonnet run shows up side-by-side
     automatically. No new dashboard work needed.
  3. Run both models for ~3-4 weeks (or alternate days), then compare
     conv>=7.5 alpha + win rate + the BUY/AVOID forward spread across the
     two `model_version` cohorts once trades mature.
- **Cautions:** a model swap muddies the forward cohort the same way a
  prompt change does (see the v2-ta/v3 overlap note above) — use the
  distinct version tag and a clean date boundary, and don't change the
  prompt in the same window. Rollback is trivial (drop the flag; the
  default stays Opus), and Sonnet rows stay quarantined under their own
  version tag so they never contaminate the Opus track record.
- **Cost math:** ~$4.80/mo Opus -> ~$2.50-2.90/mo Sonnet at one run/day, if
  quality holds. The A/B itself costs ~2x normal spend for the comparison
  window (running both), then the cheaper one going forward.
- **Added:** 2026-06-20 (cost-reduction idea; plumbing exists, never logged)
- **Priority:** MEDIUM — real recurring saving, but unproven quality and
  small absolute dollars; don't run it concurrently with another prompt
  change.
- **Shipped 2026-06-20 (plumbing + single-day test tool; the DECISION is
  still pending real data):**
  - `run_digest` gained `primary_model` + `model_version` args (digest.py);
    the primary `call_structured` now takes the model, and the internal
    `persist_signals` is tagged with the cohort so a Sonnet run can't write
    under the Opus tag (it previously would have — latent bug fixed).
  - Cohort tags are centralized: `llm_advisor.model_version_for(model)` maps
    e.g. `claude-sonnet-4-6` -> `llm-sonnet-4-6-v3-fb`; `DEFAULT_MODEL_VERSION`
    is now derived from it and still equals `llm-opus-4-7-v3-fb` (unchanged;
    pinned by tests).
  - `paper_trader.py run-day` and `scripts/run_digest.py` both expose
    `--primary-model`; run-day AUTO-DERIVES the cohort tag from the model
    (override with `--model-version`), so the forward A/B is one flag:
    `run-day --generate --primary-model claude-sonnet-4-6` writes a fully
    separate cache + signals + trades cohort. The nightly bat is unchanged
    (still Opus), so opting in is deliberate.
  - `scripts/compare_models.py` (+ pure `alpha_engine/llm/compare.py`):
    the CLEAN single-day test — one snapshot, sent identically to two models,
    reports symbol overlap (Jaccard) / direction agreement / conviction MAE
    + per-call cost + $/mo saving. ~$0.25, writes nothing. High agreement =
    low-risk switch; the skill verdict still comes from the forward cohorts.
  - 9 unit tests in `tests/test_model_ab.py` (tag derivation incl. the Opus
    back-compat pin; comparison math). Existing `model_version` pin tests
    still green.
  - **NOT yet done (needs the user + spend):** run `compare_models.py` for a
    few days to gauge agreement, then if it holds, run the Sonnet forward
    cohort nightly for ~3-4 weeks and compare on the dashboard
    (`feedback_eval`/`forward_eval` already group by model_version) before
    flipping the default. The default is still Opus until that read exists.

---

## LLM backtesting

### Historical LLM backtests are training-data contaminated (STRUCTURAL)
- **Why it matters:** This is not a bug we can fix — it's a permanent
  limitation. Opus 4.7's weights encode knowledge of the historical dates
  we backtest. The market *snapshot* we feed is point-in-time clean, but
  the model may "remember" outcomes (NVDA's 2023-24 run, COVID, SVB). So
  historical LLM backtest results are an OPTIMISTIC UPPER BOUND, never a
  forward estimate. A naive "+X% alpha" headline from these would be
  dangerously misleading.
- **How to address:** The ONLY clean test is paper/forward trading on
  dates after the model's training cutoff. Build a paper-trading loop:
  generate a digest daily going forward, record signals, score outcomes
  after N days, accumulate an out-of-sample track record. Until we have
  that, treat all historical LLM backtest numbers as plumbing/behavior
  checks, not skill estimates. The contamination warning is hard-coded
  into llm_advisor.py and both LLM backtest scripts so it can't be
  forgotten.
- **Added:** 2026-05-29 (LLM signal backtest)
- **Priority:** Informational — this gates how we INTERPRET results, and
  motivates building the forward paper-trading loop as the real validation.

### ~~Paper trader v1 only opens longs (buy/add)~~ — RESOLVED
- **Why it matters:** We currently skip hold/sell/exit/reduce signals
  when opening paper trades. ~13% of signals (82 of 633 in the
  backfill) were skipped this way. That's fine for measuring "does the
  model pick winners?" but ignores its risk-management calls. If the
  model says "sell META" and META drops, we get no credit for it.
- **How to address:** Add short paper trades for sell/exit (compute
  return as `-(exit-entry)/entry`), and treat reduce as a half-size
  long. Hold should still be skipped (it's "no change"). Once added,
  per-channel stats become a more complete picture of model behavior.
- **Added:** 2026-05-30 (paper trader v1)
- **Resolved:** 2026-05-30 — `_DIRECTION_TO_TRADE` mapping now opens shorts for sell/exit (qty 1.0) and half-size shorts for reduce (qty 0.5). Trades carry `side='long'` or `side='short'`; scorer direction-adjusts returns via `_DIRECTION_SIGN`. Hold still skipped (no change).

### ~~Paper trades don't model stops or max-adverse-excursion~~ — RESOLVED
- **Why it matters:** Each signal carries a `stop_loss_pct`, but we
  hold to time horizon regardless of intra-period drawdown. So if a
  trade goes to -20% intraday and recovers, we score it as a win.
  Real-money execution would have stopped out and locked the loss.
- **How to address:** During scoring, walk the entry→exit price window
  and check whether the stop was ever hit. If so, exit_price = entry *
  (1 - stop_loss_pct) and actual_exit_date = the day the stop fired.
  Also populate max_favorable_excursion / max_adverse_excursion from
  the same walk (currently zeros).
- **Added:** 2026-05-30 (paper trader v1)

- **Added:** 2026-05-30 (paper trader v1)
- **Resolved:** 2026-05-30 — `_walk_trade_window` in scorer.py now walks each trade's intraday bars. Stops use `low` (longs) or `high` (shorts) vs computed stop level; on first breach, exit_date and exit_price = stop level. MFE/MAE populated from adj_close walk, direction-adjusted. Outcomes get a `notes` field indicating stop-out reason. The 387 already-scored historical trades aren't retroactively updated (the SQL is idempotent and they were scored under the old logic); going forward all new scoring uses the stop-loss-aware path.

### ~~LLM occasionally emits the same ticker twice in one digest~~ — RESOLVED
- **Why it matters:** Found 1 case (2025-01-01 aggressive_growth NVDA
  appeared twice in the same digest output). Our parser persisted both
  as separate signals; downstream we opened 2 paper trades for the same
  position. Net effect: double-counted in stats. Rare but real.
- **How to address:** In `persist_signals` (parser.py), dedup by
  (symbol, direction) before insert, keeping the highest-conviction
  entry. Or upstream in the system prompt, add an explicit "do not
  emit the same symbol more than once per channel" instruction.
- **Added:** 2026-05-30 (paper trader v1)
- **Resolved:** 2026-05-30 — `persist_signals` now dedups by (channel, symbol, direction) pre-insert, keeping highest conviction and logging each drop.

### ~~Two GLD signals were skipped because it's not in our universe~~ — RESOLVED
- **Why it matters:** The LLM picked GLD (gold ETF) in several digests
  but the persist step rejected it ("symbol GLD not in universe"). GLD
  *is* in `cross_asset_section` of the snapshot (the model sees its
  price), but it's not in `config/universe.yaml`. The model can see it
  but can't trade it. Confusing for the model and bad for paper trading.
- **How to address:** Either (a) add GLD + a small "broad commodity"
  set (USO, SLV, DBC) to universe.yaml so picks are tradeable; or
  (b) tighten the system prompt to remind the model that only universe
  symbols are tradeable, not cross-asset reference symbols. Option (a)
  is the better long-term answer — it's a free universe expansion.
- **Added:** 2026-05-30 (paper trader v1)
- **Resolved:** 2026-05-30 — Added a `commodity_etfs` block (GLD, SLV, USO, DBC) to `config/universe.yaml`, upserted instruments, and backfilled 5y of bars (1255 each). Future GLD picks will be tradeable.

### LLM backtest uses no GDELT/earnings history for old dates
- **Why it matters:** GDELT data is only the last ~30 days, and the
  earnings calendar is recent + near-future only. So when we generate a
  digest for, say, 2025-03-01, the geopolitical section shows "no data"
  for all signals and earnings proximity is blank. The historical
  snapshots are therefore weaker than a live snapshot — missing two of
  the input layers. This makes historical backtest results NOT
  representative of live digest quality (in addition to the contamination
  issue, which cuts the other way).
- **How to address:** Backfill GDELT via BigQuery (already a separate
  followup) and pull historical earnings dates from a provider with
  history. Until then, note that historical digests run on regime +
  calendar + price action only.
- **Added:** 2026-05-29 (LLM signal backtest)

---

## GDELT / geopolitical signals

### GDELT 429 rate limiting requires slow ingestion
- **Why it matters:** GDELT's DOC API aggressively returns 429 even at
  modest request rates. First ingest at 0.5s spacing failed almost
  entirely. We now use 4s spacing + exponential backoff (10/20/40s),
  which works but makes a 10-signal pull take ~12 minutes. Several
  queries still came back partial (vol OR tone, not both) due to 429s
  mid-pull.
- **How to address:** Options: (a) cache aggressively and only re-pull
  the most recent few days daily rather than 30d each run; (b) spread
  the pull across time with a scheduled job rather than one batch;
  (c) investigate GDELT's BigQuery export for bulk historical loads
  (free, no rate limit, but more setup). Option (a) is the cheapest
  near-term fix for daily operation.
- **Added:** 2026-05-29 (GDELT integration)

### Some GDELT queries return empty/invalid JSON
- **Why it matters:** `recession_sentiment` and a few others intermittently
  returned non-JSON bodies (HTML error pages) even on HTTP 200, and some
  signals stored vol-only or tone-only. The feature computer degrades
  gracefully (UNKNOWN intensity, None numerics), but coverage is
  incomplete. `china_us_trade`, `fed_policy`, `oil_disruption` showed
  "no data" in the snapshot despite being queried.
- **CONFIRMED 2026-06-07:** Reproduced twice in back-to-back manual pulls
  (30d window, `--only recession_sentiment`). Both times: `TimelineVol`
  exhausted all 3 retries on 429s (never got a 200), and `TimelineTone`
  got a 200 OK but with an empty body (`Expecting value: line 1 column 1`)
  — i.e. NOT a rate-limit problem, GDELT is returning a malformed 200 for
  this specific query shape. The 4-clause `recession OR "..." OR "..." OR
  "..."` query is the prime suspect — it's the broadest/most ambiguous
  query in our signal set (`config/geopolitical.yaml`). `oil_disruption`
  and `fed_policy`, by contrast, recovered fine on retry once rate limits
  eased — those are NOT malformed-response issues, just unlucky 429 timing.
- **How to address:** (a) Add a content-type/empty-body check before json
  parsing — treat empty-200 the same as a retriable error, not a "0 results"
  success; (b) **redesign the `recession_sentiment` query** — split it into
  2 narrower signals (e.g. `recession_mentions`: `recession OR "economic
  downturn"`, and `bear_market_mentions`: `"bear market" OR "hard landing"`)
  since GDELT's DOC API appears to choke on 4+ clause OR-of-phrases queries;
  (c) verify against GDELT docs whether there's a documented clause/length
  limit we're exceeding.
- **Added:** 2026-05-29 (GDELT integration); confirmed + root-caused 2026-06-07

### GDELT history is only 30 days; no backfill for backtesting
- **Why it matters:** We ingest a 30-day rolling window. To backtest
  whether geopolitical signals add alpha (e.g. "did energy outperform
  when iran_conflict spiked?"), we need years of history aligned to
  market bars. The DOC API timespan can go to "1y" but rate limits make
  large pulls painful, and history beyond ~1-2y isn't available via this
  endpoint.
- **How to address:** Use GDELT's BigQuery public dataset for historical
  backfill (goes back to 2015 for GDELT 2.0, 1979 for events). One-time
  bulk load into geopolitical_signals, then DOC API for daily updates.
  Required before geopolitical signals can be backtested.
- **Added:** 2026-05-29 (GDELT integration)

---

## Infrastructure / portability

### yfinance threading disabled on Windows
- **Why it matters:** We set `threads: False` in `yfinance_provider.py` because
  yfinance's tz-cache sqlite races on Windows ("database is locked"). Slower
  fetches but reliable. On Linux/Mac this isn't needed.
- **How to address:** Detect platform and enable threading on non-Windows, OR
  set `yfinance.set_tz_cache_location()` to a per-process directory to avoid
  the race. Low priority unless backfill speed becomes a bottleneck.
- **Added:** 2026-05-28 (Phase 1 initial setup)

### .env loading mechanics could trip a future user
- **Why it matters:** `.env.example` is the committed template; `.env` is
  the real file (gitignored). Easy to put a real key in `.env.example` by
  mistake — git would happily ship it.
- **How to address:** Add a pre-commit hook (or a `check_secrets.py` script)
  that scans `.env.example` and any tracked `*.yaml` for things matching
  known secret patterns (long alphanumeric strings on lines containing
  `_KEY=` or `_TOKEN=`).
- **Added:** 2026-05-28 (Phase 2 setup)

---

### C2: Data freshness indicator — SHIPPED
- **Why it matters:** If yfinance silently stops updating SPY, every chart,
  MTM, and per-trade outcome calculation across the dashboard goes wrong
  without any visible signal. Surface bar-freshness vs the expected latest
  trading day so silent data failure becomes loud.
- **Shipped 2026-05-31:** New `bar_freshness()` query computes "expected
  latest trading day" via `is_trading_day` walking back from today, then
  for each active instrument counts trading-days-behind. Returns
  per-symbol breakdown + summary buckets (fresh / behind_1 / behind_2plus /
  stale / no_data). Renders as a colored strip pinned at the very top of
  Suggestions page (above last-run card and action items). Color: green
  if all fresh, orange if some 2+ behind, red if any stale (>3 trading
  days). Expandable detail shows specifically which stale symbols are
  currently held by open paper trades (extra-sensitive). Verified on
  current data: 81 fresh, 34 one day behind, 0 stale = 🟢 status.

### B6: market_summary + key_themes on Suggestions — SHIPPED
- **Why it matters:** The LLM output_json carries a top-line market read,
  2-5 key themes, and a risk-notes list — the connective narrative
  between individual picks. Previously hidden; only individual rationales
  were surfaced.
- **Shipped 2026-05-31:** New `digest_narrative(date)` query reads
  market_summary/key_themes/risk_notes from cached output_json. Rendered
  on Suggestions page in a blue panel below the digest meta, with two
  columns (📌 Key themes / ⚠️ Risk notes). Verified on 2026-05-01: surfaces
  rich content like "Risk-on regime confirmed by price + macro agreement"
  and "AMD and SMCI earnings on 5/5 — concentrated tape risk."

### B3: Glossary tooltips on every metric — SHIPPED
- **Why it matters:** Dashboard had ~30 unique metrics (conviction, alpha,
  MFE, Sharpe, profit factor, etc.) with no inline explanation. New users
  had to ask "what does that mean?" for each one.
- **Shipped 2026-05-31:** New `dashboard/glossary.py` module with 20+
  canonical definitions for signal-quality, outcome, risk, regime, and
  cross-channel terms. Imported and applied via `help=` tooltips across
  Track Record, Open Trades, Snapshot, and Lookup views. Hover any
  metric for a 1-3 sentence explanation. All 6 dashboard pages verified
  HTTP 200 after edits.

### Multi-channel agreement/contradiction detector (A1+A2) — SHIPPED
- **Why it matters:** When both channels independently recommend the same
  name in the same direction, that's stronger signal than either alone
  (independent confirmation). When channels contradict on the same name,
  one is wrong — worth scrutinizing the higher-conviction side. Previously
  treated as independent trades with no cross-check.
- **Shipped 2026-05-31:** New `channel_crosscheck(digest_date)` query
  buckets every symbol that appears in BOTH channels:
  - same direction-bucket (LONG/SHORT) = AGREEMENT with combined
    conviction score = max + 0.5×min
  - opposite direction = CONTRADICTION (warning, both rationales surfaced
    for inspection)
  - hold from either side = skipped (no-op)
- **Dashboard:** new "Cross-channel signals" expanded card on Suggestions
  page when any agreements or contradictions exist. Each suggestion card
  also shows an inline badge (🟢 agrees / 🔴 agrees / ⚠️ disagrees) so
  users see cross-channel context inline. Verified on 2026-05-01 cached
  digest: 5 long agreements detected (GOOGL/QQQ/AMZN/AAPL/MSFT), 0
  contradictions. Combined scores correctly identify GOOGL (11.5) as
  strongest cross-channel signal.

### Daily DuckDB backup — SHIPPED
- **Why it matters:** Single corrupt `alpha_engine.duckdb` would lose every
  signal, trade, outcome, and bar in the system. With the Monday auto-run
  about to start generating real forward data, this is cheap insurance.
- **Shipped 2026-05-31:** New `scripts/backup_db.ps1` copies the DB to
  `data/backups/alpha_engine.YYYYMMDD.duckdb`, prunes to the most recent
  14 (configurable via `-KeepDays`). Idempotent same-day (skips if today's
  file exists). Non-fatal on failure - logs and continues so the digest
  still runs. Wired into `daily_paper_trade.bat` as the FIRST step after
  venv activation, before any DB writes. Backup folder gitignored. First
  backup is 77.5 MB.

### Virtual portfolio simulation (D1 + D2 + D8 combined) — SHIPPED
- **Why it matters:** Per-trade alpha is a great signal-quality metric but
  doesn't answer "would I have made money?" — the question every portfolio
  investor actually asks. Need a multi-position simulator that handles
  concurrency, cash management, and MTM of open trades.
- **Shipped 2026-05-31:** New `simulate_virtual_portfolio(initial,
  position_size_pct)` runs per-channel event-driven sim:
  walks every paper trade chronologically by ENTRY date (uses
  `entry + days_held` for exit, NOT `evaluated_at` which is just when
  the scorer ran). Each entry deducts `position_size_pct × current NAV`
  from cash. Closes credit cash with `entry_value × (1 + return_pct)`.
  Currently-open trades stay open in the sim and MTM to latest bar at
  end. Closes processed before opens on same day so cash frees first.
  Cash exhaustion is honest behavior: signals get skipped if account is
  fully deployed (surfaces real-world capacity constraint).
- **Dashboard:** Track Record now has a position-size slider (default 5%)
  + NAV line chart (channels vs SPY benchmark) + final-NAV metric cards
  + current open-positions table per channel (D8: cost basis, MTM, %
  unrealized, color-coded P&L). Verified at 5% sizing: aggressive $154k
  (+54%), steady $120k (+20%), SPY $127k. 37 open positions MTM'd correctly.
- **Bug found + fixed during build:** Initial query used `evaluated_at::DATE`
  as exit date — that's "today" for all back-scored trades, so cash never
  flowed during sim. Switched to `placed_at + days_held` for actual exit.

### Dashboard "What changed since yesterday" diff — SHIPPED
- **Why it matters:** Looking at a digest in isolation hides the most
  actionable question — "what's new?" A diff against the prior digest
  surfaces direction flips, new picks, dropped picks, and conviction
  drift across the channels.
- **Shipped 2026-05-31:** New `digest_diff(new_date, old_date)` query
  produces per-channel buckets (new, dropped, flipped, conv_up, conv_down)
  with a 0.5 conviction threshold to avoid noise. Rendered as an
  expanded card on Suggestions page comparing the picked date vs the
  immediately-prior cached digest. Suppressed entirely when there are
  zero changes. Includes direction-colored chips and per-row context.
  Verified on 2026-05-01 vs 2026-04-01: detected NEW QQQ/IWM/AMZN,
  DROPPED GLD, FLIPPED XLE/META/SOXL, NVDA conviction 8.0→7.0.

### Auto-run failure alerting — SHIPPED
- **Why it matters:** Without active alerting, a Monday 5 PM failure could go
  unnoticed until you opened the dashboard hours or days later. The dashboard
  card was passive; this adds an active push.
- **Shipped 2026-05-31:** New `scripts/post_run_check.ps1` is called by the
  bat after the digest pipeline finishes. It always writes
  `data/last_run_status.json` (timestamp, status, exit_code, error_reason,
  log_tail) and on failure fires a Windows balloon notification via
  System.Windows.Forms.NotifyIcon. Failure detection covers: non-zero exit
  code, "Traceback" in log, "ANTHROPIC_API_KEY not set", or "[ERROR]" lines.
  Skipped (weekend/holiday) runs are tagged accordingly and don't alert.
  Dashboard's `last_run_summary()` now prefers the structured JSON
  (utf-8-sig to handle the BOM PowerShell writes), falling back to log
  parsing for backward compat. Last-run card surfaces `error_reason`
  prominently on failures.

### Dashboard "Today's Action Items" card — SHIPPED
- **Why it matters:** Before this, opening the dashboard meant scrolling
  through tables to find anything worth acting on. The action items card
  synthesizes "what needs your attention right now" — high-conviction
  picks, stop-outs, trades about to be scored, and positions in drawdown.
- **Shipped 2026-05-31:** New `today_action_items()` query parses 4
  buckets from the DB. Rendered as an expanded card at the top of the
  Suggestions page (just under the last-run status). On a quiet day with
  no rows in any bucket, it shows a green "no urgent action items" note
  so it doesn't take up space. Each bucket is suppressed if empty so the
  card never has filler.

### Dashboard "Any-symbol" lookup page — SHIPPED
- **Why it matters:** Phase C added ~400 S&P 500 bars-only that no UI
  surfaced. Useful when the LLM mentions a non-universe ticker in a
  rationale.
- **Shipped 2026-05-31:** New "Lookup" page under Analysis nav. Dropdown
  of all symbols with bars (~503), shows: metadata, total return since
  first bar, normalized chart vs SPY (or QQQ/IWM/DIA), outperformance
  metrics, latest 10 bars. Distinguishes 🟢 in-universe vs ⚪ bars-only.

### LLM snapshot size at 115-symbol universe — MEASURED, NO TRIM NEEDED
- **Why it mattered:** When universe went 38→115 we feared the per-ticker
  price section would bloat the prompt and increase per-call cost.
- **Measured 2026-05-30:** Snapshot is 10,385 chars (~2,596 tokens), of
  which 8,019 chars (77%) is the universe price section — ~70 chars per
  symbol, very efficient. Combined system + snapshot is ~4,100 input
  tokens. Cost impact vs old universe: roughly +$0.01 per call. Across
  ~250 trading days/year that's ~$2.50/yr extra for 3× universe coverage.
  No trimming worth doing. The existing `markdown_chars` log line in
  `build_snapshot` already gives us a tripwire if this ever changes.
- **Resolved:** 2026-05-30 (measured, not coded around)

### GDELT ingest not in daily auto-run — RESOLVED
- **Why it mattered:** GDELT data is a 30-day rolling window. Without a
  daily refresh job, the geopolitical section in the snapshot decays —
  fresh in week 1, half-stale by week 3, empty by week 5.
- **Resolved 2026-05-30:** Prepended `python scripts\\ingest_gdelt.py
  --timespan 7d --polite-sleep 4` to `scripts/daily_paper_trade.bat`
  before the paid digest call. Non-blocking (a GDELT failure logs a
  warning and proceeds to the digest). The 7-day window keeps daily
  ingest brief (~1 min) while still refreshing what the snapshot's
  recent-vs-baseline volume math needs.

### No "last run" visibility on dashboard — RESOLVED
- **Why it mattered:** After Monday's scheduled task starts firing, the
  only way to verify it worked was opening the Run Log page and scrolling.
- **Resolved 2026-05-30:** Added `last_run_summary()` query that parses
  daily_paper_trade.log into {timestamp, status, cost, opened, scored,
  gdelt_warn}. Rendered as a color-coded card pinned at the top of the
  Suggestions page (the landing view). Statuses: 🟢 OK, ⚪ skipped
  weekend, 🟡 skipped holiday, 🔴 ERROR. Shows an extra warning if last
  run was >30h ago (auto-run likely missed) and an expandable log tail
  for ERROR runs. Tested against 4 synthetic log scenarios.

---

## ML signal layer

### ML signal layer shipped (2026-06-11) — honest validation results
- **What shipped:** `alpha_engine/ml/` — point-in-time features (12-1/6-1/3-1
  momentum, 1m reversal, 30d vol, dist from 50/200MA, RSI-14), two scoring
  models (`MomentumComposite` equal-weight z-blend with zero tuned params;
  `WalkForwardXGB` retrained quarterly inside the backtest on embargoed
  labels), advisors plugged into the backtest engine, `ml_signals` table,
  `scripts/run_ml_signals.py` (wired into daily_paper_trade.bat, free),
  `scripts/validate_ml.py`, dashboard "ML Signals" page + per-suggestion ML
  badges + LLM agreement panel. 36 unit tests in `tests/` (repo's first).
- **Validation (honest):** Deep walk-forward 2008→2026, 19 survivorship-clean
  ETFs, 5y-train/2y-test: ml_momentum +312% OOS / Sharpe 0.56 vs SPY +531% /
  0.77 — top-quintile ETF momentum does NOT beat buy-and-hold in this
  US-bull-dominated window. Broad 45-ETF window (2022-07→present): ml_xgb
  +125.7% vs SPY +100.2%, Sharpe 1.01 vs 0.90, alpha +4.2%/yr — promising
  but single short window, context not proof.
- **How to use it today:** decision support + LLM cross-check (e.g. 2026-06-10
  digest: ML agreed with GOOGL/SMH/AVGO/TSM/AMZN/UNH longs, conflicted on
  META/MSFT/XLV/MA which sit in the bottom quintile), NOT a standalone
  portfolio to trade blind.
- **Added:** 2026-06-11 (ML signal layer build)

### Equity cross-section is where momentum actually pays — blocked on survivorship data
- **Why it matters:** Academic momentum alpha is concentrated in single
  stocks, not sector ETFs (our deep validation universe). We deliberately
  excluded today's universe equities from validation (survivorship bias:
  they're in the universe because they won). The XGB result on the broad ETF
  set suggests the features have signal; testing them on equities honestly
  needs point-in-time index membership (Sharadar/CRSP/Norgate — already a
  followup under Backtesting).
- **How to address:** Same fix as the existing survivorship followup. Once
  point-in-time membership exists, rerun validate_ml.py with an equity
  universe and see if cross-sectional momentum earns its keep where the
  literature says it lives.
- **Added:** 2026-06-11 (ML signal layer build)

### ML signal forward track record — same discipline as the LLM — SCORER SHIPPED, AWAITING MATURITY
- **Why it matters:** Daily ml_signals rows now accumulate from the nightly
  run. Like the LLM digest, the only fully clean evaluation is forward:
  do BUY-bucket names outperform AVOID-bucket names over the next 21
  trading days, measured on signals generated before the outcome existed?
  (Walk-forward already approximates this honestly, but live forward data
  also captures data-pipeline reality: stale bars, missing symbols.)
- **How to address:** After ~3 months of nightly rows, add a scorer query
  (BUY vs AVOID forward-return spread per signal date) and surface it on
  the ML Signals page next to the walk-forward panel.
- **Added:** 2026-06-11 (ML signal layer build)
- **Shipped 2026-06-19:** `alpha_engine/ml/forward_eval.py:compute_forward_performance`
  — per-cohort (model_version) mean BUY−AVOID spread, spread hit-rate, BUY
  vs equal-weight cross-section, computed ONLY on signal dates with 21
  trading days of bars after them (a date is excluded until it matures, and
  needs both a BUY and an AVOID leg for the spread to be defined). No
  training contamination — price-only, forward-by-construction. Surfaced on
  the ML Signals page as "Live forward track record (BUY − AVOID)" with a
  per-date expander and an "under ~10 dates = directional only" caveat;
  dashboard wrapper `queries.ml_forward_performance`. 7 unit tests in
  `tests/test_forward_eval.py` (spread math, maturation cutoff,
  both-buckets-required, multi-date hit-rate, cohort isolation, weekend-aware
  maturity ETA). **Status today:** 7 signal dates recorded (2026-06-10 →
  06-18), 0 matured — the earliest matures ~2026-07-13, so the panel shows
  the pending state and auto-fills from there. Still TODO from the proposal:
  the ~3-month read once enough dates mature.

---

## Self-learning loop

### LLM feedback loop shipped (v3-fb) — the model now sees its own results
- **What shipped 2026-06-11:** `alpha_engine/llm/feedback.py` adds two
  point-in-time snapshot sections: "Your current open paper positions"
  (deduped to one row per channel/symbol/side, MTM'd to the latest bar
  <= as_of, past-horizon flags, compact overflow list so the full book is
  always visible) and "Your track record" (win rate + avg alpha per
  conviction bucket per channel, last 12 scored trades with stop-out
  flags, repeated-miss/hit symbols at >=2 trades and |avg alpha| >= 2%).
  Prompt gained operating principle 9 ("Learn from your own track
  record") covering book-continuity, conviction recalibration, and
  don't-over-update-on-noise. model_version bumped v2-ta -> v3-fb (v2-ta
  cohort is ~empty — both changes shipped the same day, so the practical
  A/B is v1 vs v3 combined). Track Record dashboard page gained a
  "Conviction calibration" table + automatic inversion warning. 12 unit
  tests in tests/test_llm_feedback.py.
- **The loop is fully automatic:** nightly run scores matured trades ->
  next morning's snapshot includes them -> next digest recalibrates. No
  manual step, no API cost beyond ~1.2k extra snapshot tokens (~$0.01/run).
- **Point-in-time correctness:** a trade counts as scored only when
  entry + days_held <= as_of (NOT evaluated_at, which is when the scorer
  ran and would leak the future into generate_llm_history backtests).
  Unit-tested both directions (mid-flight = open, completed = scored).
- **Already visible in real data:** aggressive_growth's conviction scale
  is INVERTED on the contaminated backfill cohort — <7.0 picks avg alpha
  +5.9% vs 8.0+ at +1.4%. The model now sees this table daily; watch
  whether forward v3 digests narrow the gap (that's the loop working).
- **Added:** 2026-06-11 (feedback loop build)

### Watch: does the v3 feedback loop actually change behavior? — COMPARISON PANEL SHIPPED
- **Why it matters:** Feeding the model its record only helps if outputs
  respond — conviction distribution shifts, fewer re-buys of held names,
  explicit exits for past-horizon positions, more caution on
  repeated-miss symbols (AGG/TIP/LQD/TSLA on current data).
- **How to check (in ~2-4 weeks of v3 forward data):** compare v3 vs v1
  cohorts on (a) conviction-bucket calibration slope (8+ should beat <7),
  (b) share of suggestions duplicating an open position, (c) count of
  explicit exit/reduce calls, (d) alpha on repeated-miss symbols. Add a
  small "calibration over time" chart to Track Record if the eyeball
  check is promising.
- **Added:** 2026-06-11 (feedback loop build)
- **Shipped 2026-06-19:** `alpha_engine/llm/feedback_eval.py:compute_feedback_loop_behavior`
  compares signal cohorts by model_version on all four tells: (a) calibration
  slope = alpha(8.0+) − alpha(<7.0), with a reliability flag at ≥10 trades/bucket;
  (b) "re-buy share" = `buy` (open-new) picks that duplicated a name already
  held when generated — `add` excluded since it definitionally targets a
  held position; (c) full action mix per cohort; (d) repeated-miss list
  (≥2 matured trades, avg alpha ≤ −2%). Point-in-time matured cutoff mirrors
  feedback.py (entry+days_held ≤ as_of). Rendered on Track Record as
  "Feedback loop: did v3 change behavior?" (side-by-side cohort table +
  action-mix expander); wrapper `queries.feedback_loop_behavior`. 7 unit
  tests in `tests/test_feedback_eval.py`. **Early real read (v3 still has
  only 2 matured trades, so (a)/(d) are pending):** the action mix already
  shows a clear behavioral shift — v1 was 788 buy / 87 hold / ~14 exit+reduce,
  while v3-fb is 1 buy / 43 hold / 10 add / 40 exit+reduce, i.e. the loop has
  the model actively tending its book instead of only opening new longs.
  Re-derive (a)/(d) once v3 trades mature (~3 weeks).

---

## Execution

### ~~Close-to-close execution; no intraday model~~ — LATENCY GAP CLOSED 2026-06-13
- **Why it mattered:** Paper trades entered at the adj_close of the first
  session AFTER the digest. But the digest is generated the evening of day
  D on D's close data, so the tightest HONEST fill is D+1's OPEN (a
  market-on-open order placed overnight). Entering at D+1's close instead
  ceded a full trading session of latency on momentum/news-driven picks.
- **Shipped 2026-06-13:**
  - `paper/trader.py` now enters at the adjusted open of D+1
    (`adj_open = open * adj_close/close`, kept on the adj_close scale the
    scorer exits against). `entry_style='next_open'` and the D+1 adj_close
    are stored on every trade as `alt_entry_price` for measurement.
  - `paper/scorer.py` includes the entry day's own session in the stop /
    MFE / MAE walk for next-open trades (the position is now live that day)
    and records `alt_entry_return_pct` — the counterfactual return under
    the OTHER entry style over the same exit.
  - DB migration (`db/connection.py:_apply_column_migrations`) adds the
    three columns to existing DBs; 698 legacy trades tagged 'next_close'.
  - `scripts/measure_entry_timing.py` backfilled counterfactuals onto 401
    already-scored trades. **Measured historical gap: entering at the next
    OPEN would have added +0.57% per trade on average over the same holding
    window** vs next-close. That is the latency the switch removes, free.
  - Track Record dashboard gained an "Execution timing" panel (avg gap,
    share where open wins, live-next-open count, per-channel table).
  - 8 unit tests in `tests/test_entry_timing.py` (adj-open math, alt price,
    entry-day stop inclusion, short-side sign, migration idempotency).
- **Note:** this closes the decision→execution LATENCY gap. A still-open,
  SEPARATE idea is decision FRESHNESS — a second pre-open digest (~6 AM PT)
  that re-reads overnight moves. That roughly doubles digest cost (~$0.16 →
  ~$0.32/day, ~$7/mo) so it's deliberately deferred as an opt-in, not part
  of this free change. See below.
- **Added:** 2026-06-13 (execution latency fix)

### Optional: second pre-open digest for decision freshness (COSTS MONEY)
- **Why it matters:** The latency fix makes execution as timely as the
  decision. The remaining lever is making the DECISION itself fresher —
  an overnight gap or pre-market news can stale the 5:30 PM digest before
  the open. A short pre-open digest (~6 AM PT) would re-read the latest
  close + any fresh GDELT and adjust before the market-on-open fill.
- **How to address:** Add a second scheduled run that calls `run_digest`
  with the morning snapshot, tagged a distinct model_version (e.g.
  `...-am`) so it doesn't collide with the evening cohort. Compare am vs
  pm digest alpha after a few weeks before committing.
- **Cost:** ~doubles digest spend to ~$7/mo. NOT shipped — deferred as a
  deliberate, costed opt-in per the minimize-costs default.
- **Added:** 2026-06-13 (execution latency fix)

---

## GDELT → BigQuery migration (2026-06-13)

### ~~GDELT history is only 30 days~~ + ~~429 rate limiting~~ + ~~empty/invalid JSON~~ — ADDRESSED
- **What shipped:** a BigQuery GKG backend (`alpha_engine/data/gdelt_bigquery.py`
  + `scripts/ingest_gdelt_bq.py`) that pulls YEARS of geopolitical history
  (GKG 2.0 starts 2015) with no rate limits — finally making the layer
  backtestable ("did energy outperform when iran_conflict spiked?"). One
  scan computes all signals (a COUNTIF per signal). Output rows are
  identical to the DOC path (volume_intensity = match/total, avg_tone),
  so the intel/feature layer is unchanged. New `source` column
  ('gdelt_doc' | 'gdelt_bq') tracks provenance; a BQ backfill cleanly
  supersedes DOC rows for the same day (consistent normalization).
- **Cost safety:** BigQuery bills by bytes SCANNED. The script ALWAYS
  dry-runs first (free) and refuses any year-chunk over `--max-gb`; it
  prints a projected $ charge vs the 1 TB/month free tier. Defaults to the
  date-partitioned GKG table (so date filters prune) and offers
  `--no-theme-match` to drop the large V2Themes column for cheaper
  entity-only matching. NOT run by me — needs the user's GCP auth
  (`pip install -e ".[bigquery]"; gcloud auth application-default login`).
- **The malformed `recession_sentiment` query (root-caused 2026-06-07) is
  fixed two ways:** (a) split into `recession_mentions` +
  `bear_market_mentions` (GDELT's DOC API choked on the 4-clause
  OR-of-phrases); (b) the DOC client now treats an empty HTTP-200 body as
  retriable instead of silently returning "no data".
- **Nightly run still uses the DOC API** (zero-config, now more robust) for
  the recent rolling window; BigQuery is the one-time historical backfill +
  optional ongoing source once GCP is set up. 16 offline unit tests
  (`tests/test_gdelt_bigquery.py`) cover query construction, the 3-level
  nesting, injection guards, row mapping, and the recession split.
- **Added:** 2026-06-13 (BigQuery migration)

### ~~Backtest history doesn't cover real bear markets~~ — BACKFILLED TO 2007
- **Why it mattered:** the universe was mostly backfilled only to 2021-06,
  so defensive/regime strategies were judged on a rally-dominated window.
  Critically, GLD — in the walk-forward DEFAULT_UNIVERSE for 2008-2026 —
  had data only from 2021, so it was effectively absent through the GFC.
- **Shipped 2026-06-13:** ran `backfill.py --start 2007-01-01` across all
  115 universe instruments + FRED macro. Each ticker extends to its actual
  inception (yfinance), so SPY/QQQ/TLT/AGG/GLD/sector ETFs now cover 2008
  (GFC, -55%) and 2020 (COVID, -34%). Defensive strategies are finally
  judged on real bears. (ETFs that didn't exist pre-2015, e.g. thematic
  funds, simply start at their inception — expected.)
- **Added:** 2026-06-13 (2007 backfill)

## Portfolio risk / drawdown defense

### Concentration + earnings risk engine and Action Center — SHIPPED 2026-06-27
- **Why it mattered:** The June 4 / 22 / 26 "$10k days" were NOT market
  drops (SPY was +0.4% / -0.3% / mixed). They were a concentrated
  semis/AI-hardware book getting hit — AVGO -12.6% earnings gap (06/03
  print, on the calendar), post-quad-witching growth rotation (06/22), and
  an ongoing memory/storage selloff (06/26: WDC -13%, STX -13%, MU -7.5%).
  The real ••5210 account is ~38% semis/AI-hardware, MU alone ~12%, ~55%
  total tech. The digest had flagged every one of these risks in prose but
  nothing translated warning -> action.
- **What shipped:**
  - `alpha_engine/risk/portfolio.py` — correlated-cluster map (semis/AI-HW,
    tech-growth ETFs, broad index, leveraged, intl, defensive, bonds),
    `concentration_report()` (per-name + per-cluster weights, cap breaches;
    diversified baskets exempt from the single-NAME cap so a 15% IVV isn't
    flagged like a 12% MU), and `rank_actions()` -> severity-ranked action
    list (trim oversized name, cut cluster, earnings trim, low-cash hedge).
  - `alpha_engine/risk/earnings_guard.py` — `upcoming_earnings()` /
    `has_imminent_earnings()` over calendar_events (the earnings-blackout
    rule's data layer).
  - Dashboard "Action Center" page (now the default Trading tab):
    headline risk metrics, the ranked "do these first" cards, a
    concentration bar, positions-by-weight detail. Reads
    `data/real_holdings.json` (gitignored — real positions).
    `queries.portfolio_action_center()` wires it together.
  - 10 unit tests in `tests/test_portfolio_risk.py`.
- **Open follow-ups:**
  1. ~~**Earnings calendar is stale + universe-only.**~~ **ADDRESSED
     2026-07-19:** `scripts/refresh_earnings.py` pulls earnings dates for the
     union of the real holdings snapshot + universe equities (ETFs skipped);
     wired into the nightly bat as a non-blocking `--only-holdings` step
     (~60 names) so the book's earnings stay current. The Action Center's
     earnings window is now keyed to `date.today()` (not the snapshot pull
     date), and dust positions (< 2% of book) are excluded. Verified it
     surfaces INTC/STX/META trims ahead of their late-July prints. (yfinance
     coverage still has occasional gaps; a paid provider is the eventual
     fix if it matters.)
  2. **Wire the guard into the paper trader's open path** —
     `has_imminent_earnings()` is ready; have `open_paper_trades_for_date`
     refuse fresh full size into a print (deterministic enforcement, not
     just a dashboard nudge).
  4. **Feed the concentration caps into the LLM snapshot** so the digest
     sizes around the real book (e.g. "you're already 38% semis — stop
     adding") instead of only the paper portfolio.
- **Added:** 2026-06-27 (post-mortem on the June drawdowns)

### Trade-plan engine + trades-first Action Center — SHIPPED 2026-06-27
- **Why:** the user wanted the app to say *which trades to make and when*,
  not just show analysis. `alpha_engine/risk/trade_plan.py:build_trade_plan`
  synthesizes concentration breaches + the earnings guard + the semis-trend
  state + ML ranks into exact SELL order tickets (share count, $, reason,
  timing). Single-name breaches and imminent earnings are active "Now" /
  "Before <date>" orders; the semis-cluster cut is **trend-gated** — ARMED
  (optional) while the proxy holds its 200-DMA, and only becomes an active
  order when the trend breaks (don't dump a working trend). ML-AVOID names
  are trimmed first; a forced trim of an ML-BUY name is annotated as risk
  control, not a call. The Action Center was rewritten to lead with
  "Today's trades" and collapse everything else into expanders. 8 tests in
  `tests/test_trade_plan.py`.
- **Daily refresh — decision:** the user chose "refresh via the connector"
  (no brokerage credentials stored in the app). The dashboard reads
  `data/real_holdings.json`; the agent regenerates it via the Robinhood
  connector on request (or a scheduled Claude routine could). Open:
  optionally stand up a morning `schedule` routine that pulls positions +
  quotes, rewrites the snapshot, and surfaces the day's trades.
- **Added:** 2026-06-27

## Done

<!-- When closing an item, move it here with a date and one-line resolution. -->
