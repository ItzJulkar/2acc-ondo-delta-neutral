from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_LEVERAGE = {
    "XAU-USD.P": 20,
    "XAG-USD.P": 20,
}


@dataclass
class SizingConfig:
    margin_usage_pct: float
    max_notional_usd: float
    min_base_size: float


@dataclass
class StrategyConfig:
    close_price_pct: float
    close_direction: str
    order_mode: str
    book_offset_ticks: int
    size_tolerance: float
    poll_interval_sec: float
    reprice_sec: float
    cycle_pause_sec: float
    max_reprice_attempts: int


@dataclass
class RiskConfig:
    min_liq_distance_pct: float
    shrink_to_fit_liq: bool
    stop_on_liquidation: bool


@dataclass
class BotRuntimeConfig:
    dry_run: bool
    log_level: str
    order_prefix: str
    max_cycles: int


@dataclass
class AppConfig:
    markets: list[str]
    market_mode: str
    api_base_url: str
    leverage: int
    sizing: SizingConfig
    strategy: StrategyConfig
    risk: RiskConfig
    bot: BotRuntimeConfig
    key_id_1: str
    api_secret_1: str
    key_id_2: str
    api_secret_2: str

    def max_leverage_for(self, market: str) -> int:
        return DEFAULT_MAX_LEVERAGE.get(market, self.leverage)


def _load_env() -> None:
    load_dotenv(ROOT / ".env")


def load_config(path: str | Path | None = None) -> AppConfig:
    _load_env()
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        example = ROOT / "config.example.yaml"
        if example.exists():
            cfg_path = example
        else:
            raise FileNotFoundError(f"No config found at {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sizing = raw.get("sizing", {})
    strategy = raw.get("strategy", {})
    risk = raw.get("risk", {})
    bot = raw.get("bot", {})
    api = raw.get("api", {})

    dry_env = os.getenv("DRY_RUN", "").strip().lower()
    dry_run = bot.get("dry_run", False)
    if dry_env in ("1", "true", "yes"):
        dry_run = True
    elif dry_env in ("0", "false", "no"):
        dry_run = False

    key_id_1 = os.getenv("ONDO_KEY_ID_1", "").strip()
    secret_1 = os.getenv("ONDO_API_SECRET_1", "").strip()
    key_id_2 = os.getenv("ONDO_KEY_ID_2", "").strip()
    secret_2 = os.getenv("ONDO_API_SECRET_2", "").strip()

    if not dry_run:
        missing = []
        if not key_id_1 or "REPLACE" in key_id_1:
            missing.append("ONDO_KEY_ID_1")
        if not secret_1 or "REPLACE" in secret_1:
            missing.append("ONDO_API_SECRET_1")
        if not key_id_2 or "REPLACE" in key_id_2:
            missing.append("ONDO_KEY_ID_2")
        if not secret_2 or "REPLACE" in secret_2:
            missing.append("ONDO_API_SECRET_2")
        if missing:
            raise ValueError(
                "Missing API credentials in .env: "
                + ", ".join(missing)
                + ". Copy .env.example → .env and fill keys, or set DRY_RUN=true."
            )

    base_url = os.getenv("ONDO_BASE_URL") or api.get("base_url", "https://api.ondoperps.xyz")

    return AppConfig(
        markets=list(raw.get("markets", ["XAU-USD.P", "XAG-USD.P"])),
        market_mode=str(raw.get("market_mode", "rotate")),
        api_base_url=str(base_url).rstrip("/"),
        leverage=int(raw.get("leverage", 20)),
        sizing=SizingConfig(
            margin_usage_pct=float(sizing.get("margin_usage_pct", 40.0)),
            max_notional_usd=float(sizing.get("max_notional_usd", 0)),
            min_base_size=float(sizing.get("min_base_size", 0.001)),
        ),
        strategy=StrategyConfig(
            close_price_pct=float(strategy.get("close_price_pct", 1.0)),
            close_direction=str(strategy.get("close_direction", "either")).lower(),
            order_mode=str(strategy.get("order_mode", "fast_limit")).lower(),
            book_offset_ticks=int(strategy.get("book_offset_ticks", 0)),
            size_tolerance=float(strategy.get("size_tolerance", 0.0001)),
            poll_interval_sec=float(strategy.get("poll_interval_sec", 0.5)),
            reprice_sec=float(strategy.get("reprice_sec", 3.0)),
            cycle_pause_sec=float(strategy.get("cycle_pause_sec", 2.0)),
            max_reprice_attempts=int(strategy.get("max_reprice_attempts", 0)),
        ),
        risk=RiskConfig(
            min_liq_distance_pct=float(risk.get("min_liq_distance_pct", 7.0)),
            shrink_to_fit_liq=bool(risk.get("shrink_to_fit_liq", True)),
            stop_on_liquidation=bool(risk.get("stop_on_liquidation", True)),
        ),
        bot=BotRuntimeConfig(
            dry_run=bool(dry_run),
            log_level=str(bot.get("log_level", "INFO")).upper(),
            order_prefix=str(bot.get("order_prefix", "dn2_")),
            max_cycles=int(bot.get("max_cycles", 0)),
        ),
        key_id_1=key_id_1 or "dry_acc1",
        api_secret_1=secret_1 or "dry_secret1",
        key_id_2=key_id_2 or "dry_acc2",
        api_secret_2=secret_2 or "dry_secret2",
    )
