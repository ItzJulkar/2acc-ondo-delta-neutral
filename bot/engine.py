from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Optional

from bot.client import OndoAPIError, OndoClient, quantize_down
from bot.config import AppConfig
from bot.models import BookTop, CycleStats, MarketInfo, Order, OrderStatus, Side, Stage
from bot.risk import check_liq_buffer, compute_equal_size, post_fill_liq_ok

logger = logging.getLogger(__name__)


class DeltaNeutralEngine:
    """
    Two-account delta-neutral cycle on Ondo Perps.

    Account 1: A = long entry limit,  B = reduce-only sell close
    Account 2: C = short entry limit, D = reduce-only buy close

    On imbalance: cancel lagging open order → wait confirm → re-read fills
    → place only residual size at current book → wait reprice_sec → repeat.
    """

    def __init__(self, cfg: AppConfig, acc1: OndoClient, acc2: OndoClient):
        self.cfg = cfg
        self.acc1 = acc1
        self.acc2 = acc2
        self.stage = Stage.IDLE
        self.cycle = 0
        self.market_idx = 0
        self._stop = False
        self.stats = CycleStats()

        # Live cycle state
        self.market: str = ""
        self.info: Optional[MarketInfo] = None
        self.target_size = Decimal("0")
        self.entry_ref = Decimal("0")
        self.close_price = Decimal("0")
        self.order_a: Optional[Order] = None
        self.order_c: Optional[Order] = None
        self.order_b: Optional[Order] = None
        self.order_d: Optional[Order] = None
        self.stage_started = 0.0
        self.last_reprice = 0.0
        self.reprice_count = 0

    def stop(self) -> None:
        self._stop = True

    # ── helpers ───────────────────────────────────────────────────────

    def _sleep(self, sec: float) -> None:
        end = time.time() + sec
        while time.time() < end and not self._stop:
            time.sleep(min(0.1, end - time.time()))

    def _tol(self) -> Decimal:
        return Decimal(str(self.cfg.strategy.size_tolerance))

    def _post_only(self) -> bool:
        return self.cfg.strategy.order_mode == "strict_maker"

    def _next_market(self) -> str:
        markets = self.cfg.markets
        if not markets:
            raise RuntimeError("No markets configured")
        m = markets[self.market_idx % len(markets)]
        if self.cfg.market_mode == "rotate":
            self.market_idx += 1
        return m

    def _book(self) -> BookTop:
        return self.acc1.get_book_top(self.market)

    def _price_for_side(self, side: Side, book: BookTop) -> Decimal:
        """Aggressive limit aligned to book + optional tick offset."""
        assert self.info is not None
        tick = self.info.quote_increment
        offset = Decimal(self.cfg.strategy.book_offset_ticks) * tick
        if side == Side.BUY:
            # Join/near best bid; for faster fill can sit at best ask (may take)
            if self._post_only():
                px = book.best_bid - offset if book.best_bid > 0 else book.mid
            else:
                # fast_limit: place at best ask to cross/near-cross for fill
                px = book.best_ask if book.best_ask > 0 else book.mid
                px = px + offset
        else:
            if self._post_only():
                px = book.best_ask + offset if book.best_ask > 0 else book.mid
            else:
                px = book.best_bid if book.best_bid > 0 else book.mid
                px = px - offset
        px = quantize_down(px, tick)
        if px <= 0:
            px = quantize_down(book.mid, tick)
        return px

    def _same_entry_price(self, book: BookTop) -> Decimal:
        """Single shared limit price for A (buy) and C (sell)."""
        assert self.info is not None
        # Mid rounded to tick — both accounts post same price
        mid = book.mid if book.mid > 0 else book.mark_price
        return quantize_down(mid, self.info.quote_increment)

    def _filled_long(self) -> Decimal:
        return self.acc1.position_qty(self.market, want_long=True)

    def _filled_short(self) -> Decimal:
        return self.acc2.position_qty(self.market, want_long=False)

    def _both_flat(self) -> bool:
        return self._filled_long() <= self._tol() and self._filled_short() <= self._tol()

    def _positions_balanced(self) -> bool:
        a = self._filled_long()
        b = self._filled_short()
        if a <= self._tol() and b <= self._tol():
            return False
        return abs(a - b) <= self._tol()

    def _refresh_order(self, client: OndoClient, order: Optional[Order]) -> Optional[Order]:
        if not order or not order.order_id:
            return order
        try:
            return client.get_order(order.order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] get_order %s failed: %s", client.name, order.order_id, exc)
            return order

    def _safe_cancel(self, client: OndoClient, order: Optional[Order]) -> Optional[Order]:
        if not order or not order.order_id:
            return order
        if order.status in (OrderStatus.FULLY_FILLED, OrderStatus.CANCELED):
            return order
        try:
            updated = client.cancel_order(order.order_id)
            return updated or order
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] cancel failed: %s", client.name, exc)
            return self._refresh_order(client, order)

    def _wait_cancel_settled(self, client: OndoClient, order: Optional[Order], timeout: float = 5.0) -> Optional[Order]:
        """Ensure cancel/fill final state before placing replacement (avoid double position)."""
        if not order:
            return order
        deadline = time.time() + timeout
        cur = order
        while time.time() < deadline and not self._stop:
            cur = self._refresh_order(client, cur) or cur
            if cur.is_terminal:
                return cur
            time.sleep(0.15)
        return self._refresh_order(client, cur) or cur

    def _place_limit(
        self,
        client: OndoClient,
        side: Side,
        price: Decimal,
        size: Decimal,
        *,
        reduce_only: bool,
        tag: str,
    ) -> Optional[Order]:
        if size <= self._tol():
            return None
        assert self.info is not None
        size = quantize_down(size, self.info.base_increment)
        price = quantize_down(price, self.info.quote_increment)
        if size <= 0 or price <= 0:
            return None
        try:
            return client.place_limit(
                self.market,
                side,
                price,
                size,
                reduce_only=reduce_only,
                post_only=self._post_only(),
                tag=tag,
            )
        except OndoAPIError as exc:
            if exc.code == "post_only_has_match" and self._post_only():
                logger.warning("[%s] post-only rejected (would take) — will reprice", client.name)
                return None
            logger.error("[%s] place_limit failed: %s", client.name, exc)
            raise

    # ── stages ────────────────────────────────────────────────────────

    def _start_cycle(self) -> bool:
        self.cycle += 1
        self.market = self._next_market()
        self.stats = CycleStats(cycle_id=self.cycle, market=self.market)
        self.order_a = self.order_b = self.order_c = self.order_d = None
        self.reprice_count = 0
        self.close_price = Decimal("0")

        logger.info("========== CYCLE %s | %s ==========", self.cycle, self.market)

        # Market meta (public)
        self.info = self.acc1.get_market_info(self.market)
        lev = min(self.cfg.leverage, self.cfg.max_leverage_for(self.market), self.info.max_leverage)
        self.acc1.set_leverage(self.market, lev)
        self.acc2.set_leverage(self.market, lev)

        # Cancel leftover bot orders
        self.acc1.cancel_open_bot_orders(self.market)
        self.acc2.cancel_open_bot_orders(self.market)

        # Refuse if residual exposure
        if not self._both_flat():
            logger.error(
                "Residual positions: acc1 long=%s acc2 short=%s — reconcile before new cycle",
                self._filled_long(),
                self._filled_short(),
            )
            self.stage = Stage.RECONCILE
            return False

        bal1 = self.acc1.get_balance()
        bal2 = self.acc2.get_balance()
        if self.cfg.risk.stop_on_liquidation and (bal1.under_liquidation or bal2.under_liquidation):
            logger.error("Account under liquidation — stopping")
            self.stage = Stage.STOPPED
            self._stop = True
            return False

        book = self._book()
        size, ref, note = compute_equal_size(self.cfg, self.market, self.info, book, bal1, bal2)
        logger.info("Sizing: %s | bal1_avail=%s bal2_avail=%s", note, bal1.available_margin, bal2.available_margin)

        ok, size, liq_msg = check_liq_buffer(self.cfg, ref, lev, size, self.info)
        logger.info("Liq pre-check: %s", liq_msg)
        if not ok or size <= 0:
            logger.warning("Skip cycle — cannot size safely")
            self.stage = Stage.IDLE
            self._sleep(self.cfg.strategy.cycle_pause_sec)
            return False

        self.target_size = size
        self.entry_ref = ref
        self.stats.target_size = size
        self.stats.entry_price = ref

        # Place A + C at same price
        entry_px = self._same_entry_price(book)
        logger.info(
            "ENTRY: A buy + C sell size=%s @ %s (mode=%s)",
            size,
            entry_px,
            self.cfg.strategy.order_mode,
        )
        self.order_a = self._place_limit(self.acc1, Side.BUY, entry_px, size, reduce_only=False, tag="A")
        self.order_c = self._place_limit(self.acc2, Side.SELL, entry_px, size, reduce_only=False, tag="C")
        self.stage = Stage.ENTRY
        self.stage_started = time.time()
        self.last_reprice = time.time()
        return True

    def _reprice_entry_imbalance(self) -> None:
        """If one account filled more, cancel lagging order and re-place residual."""
        long_q = self._filled_long()
        short_q = self._filled_short()
        logger.info("ENTRY reconcile long=%s short=%s target=%s", long_q, short_q, self.target_size)

        # Refresh open orders
        self.order_a = self._refresh_order(self.acc1, self.order_a)
        self.order_c = self._refresh_order(self.acc2, self.order_c)

        # Need more long on acc1?
        need_long = self.target_size - long_q
        need_short = self.target_size - short_q

        # Cap residual by the smaller remaining need so we don't overshoot one side forever
        # Strategy: each side independently chases its own target_size; balance enforced later
        max_rep = self.cfg.strategy.max_reprice_attempts
        if max_rep and self.reprice_count >= max_rep:
            logger.warning("Max reprice attempts reached in ENTRY")
            return

        book = self._book()
        self.reprice_count += 1
        self.stats.reprice_count = self.reprice_count

        if need_long > self._tol():
            if self.order_a and self.order_a.is_open:
                self.order_a = self._safe_cancel(self.acc1, self.order_a)
                self.order_a = self._wait_cancel_settled(self.acc1, self.order_a)
            # Re-read after cancel race
            long_q = self._filled_long()
            need_long = self.target_size - long_q
            if need_long > self._tol():
                px = self._price_for_side(Side.BUY, book)
                logger.info("Reprice A: buy residual %s @ %s", need_long, px)
                self.order_a = self._place_limit(
                    self.acc1, Side.BUY, px, need_long, reduce_only=False, tag="A"
                )

        if need_short > self._tol():
            if self.order_c and self.order_c.is_open:
                self.order_c = self._safe_cancel(self.acc2, self.order_c)
                self.order_c = self._wait_cancel_settled(self.acc2, self.order_c)
            short_q = self._filled_short()
            need_short = self.target_size - short_q
            if need_short > self._tol():
                px = self._price_for_side(Side.SELL, book)
                logger.info("Reprice C: sell residual %s @ %s", need_short, px)
                self.order_c = self._place_limit(
                    self.acc2, Side.SELL, px, need_short, reduce_only=False, tag="C"
                )

        self.last_reprice = time.time()

    def _sync_target_to_min_fill(self) -> None:
        """After both sides have something, shrink target to min filled so we don't chase forever."""
        long_q = self._filled_long()
        short_q = self._filled_short()
        if long_q > self._tol() and short_q > self._tol():
            m = min(long_q, short_q)
            # If one is ahead, the lagging will catch up to original target;
            # once min is close to max within tolerance we're balanced.
            _ = m

    def _on_entry_balanced(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        # Equalize: if one side overshot, reduce target to the smaller fill
        balanced_size = min(long_q, short_q)
        if abs(long_q - short_q) > self._tol():
            # Still imbalanced — shouldn't be called
            return

        self.target_size = balanced_size
        p1 = self.acc1.get_position(self.market)
        p2 = self.acc2.get_position(self.market)
        e1 = p1.average_entry_price if p1 else self.entry_ref
        e2 = p2.average_entry_price if p2 else self.entry_ref
        self.entry_ref = (e1 + e2) / 2
        self.stats.entry_price = self.entry_ref

        ok1, msg1 = post_fill_liq_ok(self.cfg, self.acc1, self.market, e1)
        ok2, msg2 = post_fill_liq_ok(self.cfg, self.acc2, self.market, e2)
        logger.info("Post-fill liq: acc1 %s | acc2 %s", msg1, msg2)
        if not (ok1 and ok2):
            logger.warning("Liq buffer fail after fill — still proceeding to manage close (manual risk)")

        direction = self.cfg.strategy.close_direction
        pct = Decimal(str(self.cfg.strategy.close_price_pct)) / Decimal("100")

        if direction == "up":
            self.close_price = quantize_down(
                self.entry_ref * (Decimal("1") + pct), self.info.quote_increment  # type: ignore[union-attr]
            )
            self._place_close_orders()
        elif direction == "down":
            self.close_price = quantize_down(
                self.entry_ref * (Decimal("1") - pct), self.info.quote_increment  # type: ignore[union-attr]
            )
            self._place_close_orders()
        else:
            # either: wait for ±1% mark move then place B+D at same book price
            logger.info(
                "ENTRY hedged size=%s entry≈%s — waiting for ±%s%% mark move before B+D",
                balanced_size,
                self.entry_ref,
                self.cfg.strategy.close_price_pct,
            )
            self.stage = Stage.WAIT_CLOSE_TRIGGER
            self.stage_started = time.time()

    def _place_close_orders(self) -> None:
        """B + D reduce-only at the same close_price."""
        size = min(self._filled_long(), self._filled_short())
        if size <= self._tol():
            logger.info("Nothing to close")
            self.stage = Stage.IDLE
            return

        # Cancel any leftover entry orders
        self.order_a = self._safe_cancel(self.acc1, self.order_a)
        self.order_c = self._safe_cancel(self.acc2, self.order_c)

        px = self.close_price
        if px <= 0:
            book = self._book()
            px = self._same_entry_price(book)

        logger.info("EXIT: B sell + D buy size=%s @ %s (reduce-only)", size, px)
        self.order_b = self._place_limit(self.acc1, Side.SELL, px, size, reduce_only=True, tag="B")
        self.order_d = self._place_limit(self.acc2, Side.BUY, px, size, reduce_only=True, tag="D")
        self.stage = Stage.EXIT
        self.stage_started = time.time()
        self.last_reprice = time.time()
        self.reprice_count = 0

    def _check_close_trigger(self) -> None:
        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid
        if self.entry_ref <= 0:
            return
        # Dry-run: mark does not move — auto-fire close after short wait
        if self.cfg.bot.dry_run and time.time() - self.stage_started >= 1.0:
            mark = self.entry_ref * (Decimal("1") + Decimal(str(self.cfg.strategy.close_price_pct)) / Decimal("100"))
            logger.info("DRY_RUN: simulating +%s%% mark move → %s", self.cfg.strategy.close_price_pct, mark)
        move = abs(mark - self.entry_ref) / self.entry_ref * Decimal("100")
        need = Decimal(str(self.cfg.strategy.close_price_pct))
        if move + Decimal("0.0001") >= need:
            # Place both closes at same aggressive book-aligned price for fast fill
            if mark >= self.entry_ref:
                # price up: sell long at bid/ask, buy short cover at same level
                px = self._price_for_side(Side.SELL, book)
            else:
                px = self._price_for_side(Side.BUY, book)
            # Force identical price for B and D
            self.close_price = quantize_down(px, self.info.quote_increment)  # type: ignore[union-attr]
            logger.info("Close trigger: mark=%s move=%.4f%% → B/D @ %s", mark, float(move), self.close_price)
            self._place_close_orders()

    def _reprice_exit_imbalance(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        logger.info("EXIT reconcile remaining long=%s short=%s", long_q, short_q)

        self.order_b = self._refresh_order(self.acc1, self.order_b)
        self.order_d = self._refresh_order(self.acc2, self.order_d)

        max_rep = self.cfg.strategy.max_reprice_attempts
        if max_rep and self.reprice_count >= max_rep:
            logger.warning("Max reprice attempts reached in EXIT")
            return

        book = self._book()
        self.reprice_count += 1
        self.stats.reprice_count = self.reprice_count

        # Remaining to close = current position
        if long_q > self._tol():
            if self.order_b and self.order_b.is_open:
                self.order_b = self._safe_cancel(self.acc1, self.order_b)
                self.order_b = self._wait_cancel_settled(self.acc1, self.order_b)
            long_q = self._filled_long()
            if long_q > self._tol():
                px = self._price_for_side(Side.SELL, book)
                logger.info("Reprice B: sell residual %s @ %s", long_q, px)
                self.order_b = self._place_limit(
                    self.acc1, Side.SELL, px, long_q, reduce_only=True, tag="B"
                )

        if short_q > self._tol():
            if self.order_d and self.order_d.is_open:
                self.order_d = self._safe_cancel(self.acc2, self.order_d)
                self.order_d = self._wait_cancel_settled(self.acc2, self.order_d)
            short_q = self._filled_short()
            if short_q > self._tol():
                px = self._price_for_side(Side.BUY, book)
                logger.info("Reprice D: buy residual %s @ %s", short_q, px)
                self.order_d = self._place_limit(
                    self.acc2, Side.BUY, px, short_q, reduce_only=True, tag="D"
                )

        self.last_reprice = time.time()

    def _reconcile_residual(self) -> None:
        """Emergency: flatten whatever is open with reprice loop."""
        logger.warning("RECONCILE residual exposure")
        self.market = self.market or self.cfg.markets[0]
        if not self.info:
            self.info = self.acc1.get_market_info(self.market)
        # Treat as exit until flat
        if self._both_flat():
            self.stage = Stage.IDLE
            return
        self._reprice_exit_imbalance()
        if self._both_flat():
            self.stage = Stage.IDLE

    # ── main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "Bot start | markets=%s lev=%s margin=%s%% close=%s%% mode=%s dry_run=%s",
            self.cfg.markets,
            self.cfg.leverage,
            self.cfg.sizing.margin_usage_pct,
            self.cfg.strategy.close_price_pct,
            self.cfg.strategy.order_mode,
            self.cfg.bot.dry_run,
        )
        if self.cfg.bot.dry_run:
            logger.warning("DRY_RUN enabled — no real orders (simulated fills)")

        while not self._stop:
            try:
                if self.cfg.bot.max_cycles and self.cycle >= self.cfg.bot.max_cycles and self.stage == Stage.IDLE:
                    logger.info("max_cycles reached — exit")
                    break

                if self.stage == Stage.STOPPED:
                    break

                if self.stage == Stage.IDLE:
                    started = self._start_cycle()
                    if not started and self.stage == Stage.IDLE:
                        self._sleep(self.cfg.strategy.cycle_pause_sec)
                    continue

                if self.stage == Stage.RECONCILE:
                    self._reconcile_residual()
                    self._sleep(self.cfg.strategy.reprice_sec)
                    continue

                if self.stage == Stage.ENTRY:
                    self.order_a = self._refresh_order(self.acc1, self.order_a)
                    self.order_c = self._refresh_order(self.acc2, self.order_c)
                    long_q = self._filled_long()
                    short_q = self._filled_short()

                    # Matched target (both fully filled to target or equal partial)
                    if long_q >= self.target_size - self._tol() and short_q >= self.target_size - self._tol():
                        # trim target to actual min
                        self.target_size = min(long_q, short_q)
                        logger.info("ENTRY complete long=%s short=%s", long_q, short_q)
                        self._on_entry_balanced()
                    elif abs(long_q - short_q) <= self._tol() and long_q > self._tol():
                        # Equal partial fills — accept smaller hedge and move on
                        # only if both orders terminal or timeout on reprice
                        a_done = not self.order_a or self.order_a.is_terminal
                        c_done = not self.order_c or self.order_c.is_terminal
                        if a_done and c_done:
                            self.target_size = min(long_q, short_q)
                            logger.info("ENTRY equal partial accepted size=%s", self.target_size)
                            self._on_entry_balanced()
                        elif time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                            self._reprice_entry_imbalance()
                    else:
                        # Imbalance or still waiting
                        if time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                            self._reprice_entry_imbalance()
                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                if self.stage == Stage.WAIT_CLOSE_TRIGGER:
                    self._check_close_trigger()
                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                if self.stage == Stage.EXIT:
                    self.order_b = self._refresh_order(self.acc1, self.order_b)
                    self.order_d = self._refresh_order(self.acc2, self.order_d)
                    if self._both_flat():
                        logger.info(
                            "CYCLE %s complete | market=%s entry=%s reprices=%s",
                            self.cycle,
                            self.market,
                            self.stats.entry_price,
                            self.stats.reprice_count,
                        )
                        self.stage = Stage.IDLE
                        self._sleep(self.cfg.strategy.cycle_pause_sec)
                    elif time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                        self._reprice_exit_imbalance()
                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                self._sleep(self.cfg.strategy.poll_interval_sec)
            except KeyboardInterrupt:
                logger.info("Interrupted — stopping")
                self._stop = True
            except Exception:
                logger.exception("Loop error — backoff 2s")
                self._sleep(2.0)

        # Best-effort cancel open bot orders on shutdown
        try:
            if self.market:
                self.acc1.cancel_open_bot_orders(self.market)
                self.acc2.cancel_open_bot_orders(self.market)
        except Exception:  # noqa: BLE001
            pass
        logger.info("Bot stopped")
