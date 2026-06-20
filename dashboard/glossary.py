"""Single source of truth for term definitions surfaced as Streamlit tooltips.

Keep each definition short (1-3 sentences) — these appear in hover popups
attached to st.metric() and other UI elements. When a metric appears in
multiple views, importing from here keeps wording consistent.
"""

# ---------------------------------------------------------------------------
# Signal-quality terms
# ---------------------------------------------------------------------------

CONVICTION = (
    "0-10 confidence scale the LLM assigns each suggestion. "
    "6=weak signal, 7=normal size, 8=strong, 9=high confidence (rare), "
    "10=true layup (almost never). Below 6 typically skipped."
)

TARGET_WEIGHT = (
    "Suggested portfolio % for this position. Channel A caps at 5% per name; "
    "Channel B caps at 15% per name."
)

TIME_HORIZON = (
    "Holding period the LLM intends, in calendar days. Short (5-15d) = "
    "event-driven (earnings, FOMC). Medium (15-60d) = regime/momentum. "
    "Long (60d+) = structural thesis."
)

STOP_LOSS_PCT = (
    "% drop from entry that would close the position. Stop-loss is modeled "
    "in scoring: if intraday low (long) or high (short) breaches the stop, "
    "the trade exits at the stop level on that day."
)

DIRECTION = (
    "buy / add → open long. sell / exit → open short. reduce → half-size "
    "short. hold → no change (no paper trade opened)."
)

# ---------------------------------------------------------------------------
# Outcome / track-record terms
# ---------------------------------------------------------------------------

RETURN_PCT = (
    "Direction-adjusted realized return from entry to exit. For longs: "
    "(exit-entry)/entry. For shorts: -(exit-entry)/entry. Stop-out fills "
    "at the stop level on the day it triggered."
)

ALPHA = (
    "Trade's return minus SPY's return over the same window. Positive = "
    "beat the benchmark. Sum-based, not compounded. Note: historical "
    "alpha includes training-data contamination for individual names."
)

WIN_RATE = "Share of trades with positive direction-adjusted return."

PROFIT_FACTOR = (
    "Sum of winning trade returns divided by absolute sum of losing trade "
    "returns. >1.0 means winners outweigh losers in magnitude. >2.0 is strong."
)

MFE = (
    "Max Favorable Excursion — best mid-trade move IN your direction (as "
    "% from entry). Indicates 'how good did it get?' before close. Should "
    "be ≥ realized return."
)

MAE = (
    "Max Adverse Excursion — worst mid-trade move AGAINST your direction. "
    "If a trade survived a -15% MAE to close at +5%, MAE=-15% — useful for "
    "judging if stop levels make sense."
)

DIRECTION_CORRECT = "True if the direction-adjusted realized return was positive."

DAYS_HELD = "Calendar days between entry bar and exit bar."

# ---------------------------------------------------------------------------
# Risk / portfolio metrics
# ---------------------------------------------------------------------------

SHARPE = (
    "Annualized excess return per unit of volatility. Excess = annual_return − "
    "risk_free_rate. We use DGS3MO (3-month T-bill) average over the window "
    "as the risk-free rate. >1.0 is good; >2.0 is strong."
)

SORTINO = "Like Sharpe but only penalizes downside volatility (returns below 0)."

CALMAR = "Annualized return divided by max drawdown. Reward-per-pain."

MAX_DRAWDOWN = "Largest peak-to-trough decline in NAV during the window."

UNREALIZED = (
    "Mark-to-market profit/loss vs entry price, computed from the latest "
    "available bar. Updates daily as new closes land in market_bars."
)

NAV = (
    "Net Asset Value — total portfolio value (cash + value of all open positions). "
    "Starts at the initial capital and evolves as trades close."
)

# ---------------------------------------------------------------------------
# Regime / context
# ---------------------------------------------------------------------------

REGIME = (
    "Rule-based macro classification. expansion_low_vol = healthy + calm. "
    "expansion_high_vol = healthy but choppy. late_cycle = restrictive Fed + "
    "curve flat. recession = Sahm Rule triggered. recovery = exiting recession."
)

REGIME_CONFIDENCE = (
    "Classifier's confidence (0-1). Higher = more rules firing in agreement. "
    "Below 0.6 = take regime label with a grain of salt."
)

# ---------------------------------------------------------------------------
# Cross-channel / agreement
# ---------------------------------------------------------------------------

CROSSCHECK_AGREE = (
    "Both Channel A and Channel B independently recommended this symbol in "
    "the same direction. Treated as stronger signal than either alone."
)

CROSSCHECK_CONTRADICT = (
    "Channels disagree on this symbol — one says long, other says short. "
    "Worth scrutinizing; the higher-conviction side is usually the bet."
)

CROSSCHECK_COMBINED = (
    "Combined cross-channel score: max(conv_A, conv_B) + 0.5 × min(conv_A, conv_B). "
    "Rewards strong+strong without double-counting."
)

# ---------------------------------------------------------------------------
# Backtest-specific
# ---------------------------------------------------------------------------

SURVIVORSHIP_WARNING = (
    "Today's S&P 500 ≠ historical S&P 500. Companies delisted or removed "
    "from the index since the backtest start are absent from data, so "
    "individual-name backtest returns are systematically biased upward."
)

CONTAMINATION_WARNING = (
    "The 387 historical scored trades come from the model's training-data "
    "window. The model may 'remember' past outcomes, so historical "
    "alpha/win-rate numbers are upper bounds. Only forward paper trading "
    "(after the auto-run starts Monday) is a clean signal-quality test."
)

# ---------------------------------------------------------------------------
# ML signal layer terms
# ---------------------------------------------------------------------------

ML_SCORE = (
    "Cross-sectional attractiveness score. Momentum composite: equal-weight "
    "blend of 12-1, 6-1, and 3-1 month momentum z-scores vs the rest of the "
    "universe. XGBoost: predicted forward-return rank from the same features. "
    "Higher = more attractive. Scores compare WITHIN a day, not across days."
)

ML_ACTION = (
    "Quintile bucket: top 20% of the universe by score = BUY, bottom 20% = "
    "AVOID, middle 60% = HOLD. Standard academic portfolio construction — "
    "thresholds are not tuned."
)

ML_RANK = (
    "Position in that day's cross-section, 1 = most attractive. Rank moves "
    "matter more than score moves — a name sliding from #5 to #40 is "
    "losing relative strength even if its price is flat."
)

ML_CONSENSUS = (
    "Names where BOTH the momentum composite and the XGBoost model put the "
    "symbol in the same quintile (BUY or AVOID). Two different methods "
    "agreeing on the same features is a stronger read than either alone."
)

ML_SPLIT = (
    "The composite chases 12-month trends; XGBoost has learned some "
    "mean-reversion. A split usually flags an extended or washed-out name — "
    "treat as no-signal rather than picking a side."
)

ML_LLM_AGREEMENT = (
    "LLM digest picks cross-referenced against ML ranks. Agree = LLM long "
    "on an ML BUY (or LLM short on an ML AVOID). Conflict = they point "
    "opposite ways. The two sources are independent: the LLM reads macro/"
    "narrative, the ML layer reads only price history."
)

WALK_FORWARD_OOS = (
    "Out-of-sample walk-forward: the model/parameters are frozen using only "
    "data BEFORE each 2-year test window, then judged on that unseen window. "
    "Every metric shown is from test windows only. Unlike LLM historical "
    "backtests, there is no training-data contamination — the ML layer sees "
    "nothing but past prices."
)
