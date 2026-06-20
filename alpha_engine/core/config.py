"""Configuration loader.

Loads settings.yaml + channels.yaml + universe.yaml, merges with environment
variables, and exposes a single typed `Settings` object. Env vars take
precedence over YAML.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from alpha_engine.core.types import Channel, InstrumentType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


# Load .env once at import time so os.environ is populated before Settings
# is constructed. override=True because some shells (CI runners, IDE harnesses)
# export the var names as empty strings, and load_dotenv(override=False) won't
# overwrite an existing-but-empty var. .env is authoritative for this project.
load_dotenv(PROJECT_ROOT / ".env", override=True)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    path: str = "data/alpha_engine.duckdb"


class LoggingConfig(BaseModel):
    # `json` would shadow BaseModel.json(); use json_output instead.
    model_config = {"populate_by_name": True}

    level: str = "INFO"
    json_output: bool = Field(default=False, alias="json")


class DataConfig(BaseModel):
    default_history_days: int = 1825
    user_agent: str = "AlphaEngine/0.1"


class RiskConfig(BaseModel):
    var_confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    correlation_window: int = 60
    kelly_fraction: float = 0.25
    correlation_alert_threshold: float = 0.7


class FredSeriesSpec(BaseModel):
    id: str
    name: str


class ChannelConfig(BaseModel):
    enabled: bool = True
    description: str = ""
    benchmark: str = "SPY"
    target_excess_return: float = 0.0
    target_sharpe: float = 1.0
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.20
    target_positions: int = 25
    max_drawdown: float = 0.15
    target_volatility: float = 0.12
    stop_loss: float = 0.06
    instruments: list[InstrumentType]
    options_enabled: bool = False
    leverage_enabled: bool = False
    max_leveraged_etf_weight: float = 0.10


class InstrumentSpec(BaseModel):
    symbol: str
    name: str
    type: InstrumentType
    sector: str | None = None


class UniverseConfig(BaseModel):
    universes: dict[str, list[InstrumentSpec]]
    exclusions: dict[str, list[str]] = Field(default_factory=lambda: {"symbols": []})


class GeopoliticalSignalSpec(BaseModel):
    name: str
    query: str
    description: str = ""
    sector_relevance: list[str] = Field(default_factory=list)
    # AND-list of regex alternations for the BigQuery GKG backend. Empty =
    # not BigQuery-ingestable (the DOC `query` is still used). See
    # config/geopolitical.yaml for the format.
    bq_match: list[str] = Field(default_factory=list)


class GeopoliticalConfig(BaseModel):
    signals: list[GeopoliticalSignalSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root Settings object
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Fully-resolved application settings."""

    project_root: Path
    database: DatabaseConfig
    logging: LoggingConfig
    data: DataConfig
    risk: RiskConfig
    fred_series: list[FredSeriesSpec]
    channels: dict[Channel, ChannelConfig]
    universe: UniverseConfig
    geopolitical: GeopoliticalConfig

    # Secrets (from env)
    fred_api_key: str | None = None
    anthropic_api_key: str | None = None
    news_api_key: str | None = None
    polygon_api_key: str | None = None

    @property
    def db_path(self) -> Path:
        """Absolute path to the database file."""
        # Env var override
        env_override = os.getenv("ALPHA_DB_PATH")
        raw = env_override or self.database.path
        p = Path(raw)
        if not p.is_absolute():
            p = self.project_root / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def channel(self, name: Channel) -> ChannelConfig:
        return self.channels[name]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_settings() -> Settings:
    base = _load_yaml(CONFIG_DIR / "settings.yaml")
    channels_raw = _load_yaml(CONFIG_DIR / "channels.yaml")
    universe_raw = _load_yaml(CONFIG_DIR / "universe.yaml")
    geopol_path = CONFIG_DIR / "geopolitical.yaml"
    geopol_raw = _load_yaml(geopol_path) if geopol_path.exists() else {}

    # Env overrides for logging
    log_level_override = os.getenv("ALPHA_LOG_LEVEL")
    if log_level_override:
        base.setdefault("logging", {})["level"] = log_level_override

    # Resolve channels
    channels_parsed: dict[Channel, ChannelConfig] = {}
    for name, cfg in (channels_raw.get("channels") or {}).items():
        try:
            ch = Channel(name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown channel '{name}' in channels.yaml. "
                f"Add to Channel enum if intended."
            ) from exc
        channels_parsed[ch] = ChannelConfig(**cfg)

    return Settings(
        project_root=PROJECT_ROOT,
        database=DatabaseConfig(**(base.get("database") or {})),
        logging=LoggingConfig(**(base.get("logging") or {})),
        data=DataConfig(**(base.get("data") or {})),
        risk=RiskConfig(**(base.get("risk") or {})),
        fred_series=[FredSeriesSpec(**s) for s in (base.get("fred_series") or [])],
        channels=channels_parsed,
        universe=UniverseConfig(**universe_raw),
        geopolitical=GeopoliticalConfig(**geopol_raw),
        fred_api_key=os.getenv("FRED_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        news_api_key=os.getenv("NEWS_API_KEY") or None,
        polygon_api_key=os.getenv("POLYGON_API_KEY") or None,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the singleton Settings. Cached so config files load once."""
    return _load_settings()


def reload_settings() -> Settings:
    """Force-reload settings (for tests or config edits)."""
    get_settings.cache_clear()
    return get_settings()
