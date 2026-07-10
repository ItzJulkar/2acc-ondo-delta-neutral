from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python -m bot.main` and `python bot/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.client import OndoClient  # noqa: E402
from bot.config import load_config  # noqa: E402
from bot.engine import DeltaNeutralEngine, MultiMarketRunner  # noqa: E402


def setup_logging(level: str) -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs / "bot.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, handlers=handlers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-account Ondo Perps delta-neutral bot (A/C entry, B/D close, 3s reprice)",
    )
    p.add_argument(
        "-c",
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate fills without sending real orders",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate credentials + balances + markets, then exit",
    )
    p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override bot.max_cycles (e.g. 1 for a single test cycle)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        # Must set before load_config so credential checks are skipped
        import os

        os.environ["DRY_RUN"] = "true"
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.bot.dry_run = True
    if args.max_cycles is not None:
        cfg.bot.max_cycles = args.max_cycles

    setup_logging(cfg.bot.log_level)
    log = logging.getLogger("main")

    acc1 = OndoClient(
        cfg.api_base_url,
        cfg.key_id_1,
        cfg.api_secret_1,
        name="acc1",
        order_prefix=cfg.bot.order_prefix,
        dry_run=cfg.bot.dry_run,
    )
    acc2 = OndoClient(
        cfg.api_base_url,
        cfg.key_id_2,
        cfg.api_secret_2,
        name="acc2",
        order_prefix=cfg.bot.order_prefix,
        dry_run=cfg.bot.dry_run,
    )

    if args.check:
        log.info("Connection check (dry_run=%s)…", cfg.bot.dry_run)
        for name, client in ("acc1", acc1), ("acc2", acc2):
            bal = client.get_balance()
            log.info(
                "[%s] wallet=%s available=%s under_liq=%s",
                name,
                bal.wallet_balance,
                bal.available_margin,
                bal.under_liquidation,
            )
        for m in cfg.markets:
            info = acc1.get_market_info(m)
            book = acc1.get_book_top(m)
            log.info(
                "Market %s base_inc=%s quote_inc=%s bid=%s ask=%s mark=%s",
                m,
                info.base_increment,
                info.quote_increment,
                book.best_bid,
                book.best_ask,
                book.mark_price,
            )
        log.info("Check OK")
        return 0

    mode = (cfg.market_mode or "parallel").lower()
    if mode == "parallel" and len(cfg.markets) > 1:
        log.info("Starting PARALLEL engines for: %s", ", ".join(cfg.markets))
        runner = MultiMarketRunner(cfg, acc1, acc2)
        try:
            runner.run()
        except KeyboardInterrupt:
            log.info("Ctrl+C — shutting down")
            runner.stop()
    else:
        log.info("Starting single/rotate engine mode=%s markets=%s", mode, cfg.markets)
        engine = DeltaNeutralEngine(cfg, acc1, acc2)
        try:
            engine.run()
        except KeyboardInterrupt:
            log.info("Ctrl+C — shutting down")
            engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
