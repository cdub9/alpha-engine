"""ML signal layer — cross-sectional momentum ranking + walk-forward XGBoost.

This is the validated, uncontaminated counterpart to the LLM digest. Unlike
LLM historical backtests (training-data contamination makes them an upper
bound), every number this layer produces in a backtest is honest: features
are pure functions of past prices, and the XGBoost variant retrains inside
the walk-forward loop using only data available at decision time.

Modules:
  features  — point-in-time feature computation (pure pandas, no DB)
  model     — MomentumComposite (rule-based) + WalkForwardXGB (trained)
  advisor   — SignalAdvisor adapters for the backtest engine
  store     — ml_signals persistence (daily ranks for the dashboard)
"""

from alpha_engine.ml.advisor import MLMomentumAdvisor, XGBMomentumAdvisor
from alpha_engine.ml.features import FEATURE_COLUMNS, compute_features
from alpha_engine.ml.model import MomentumComposite, WalkForwardXGB, assign_actions
from alpha_engine.ml.store import latest_ml_signal_date, load_ml_signals, persist_ml_signals

__all__ = [
    "FEATURE_COLUMNS",
    "compute_features",
    "MomentumComposite",
    "WalkForwardXGB",
    "assign_actions",
    "MLMomentumAdvisor",
    "XGBMomentumAdvisor",
    "persist_ml_signals",
    "load_ml_signals",
    "latest_ml_signal_date",
]
