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
    Dual-account delta-neutral bot.

    A + C: same price, same size, opposite sides — place once at current book mid
           and REST until fill. Cancel/reprice ONLY if one side fills and the
           other does not (true imbalance).

    B + D: placed with A/C at entry±1%. REST until fill. Do NOT cancel on a timer.
           Only cancel D/B for the 5th-order path when one account is fully flat
           and the other still has a position.
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

        self.market: str = ""
        self.info: Optional[MarketInfo] = None
        self.target_size = Decimal("0")
        self.entry_ref = Decimal("0")
        self.close_price = Decimal("0")
        self.order_a: Optional[Order] = None
        self.order_b: Optional[Order] = None
        self.order_c: Optional[Order] = None
        self.order_d: Optional[Order] = None
        self.order_e: Optional[Order] = None
        self.stage_started = 0.0
        self.last_reprice = 0.0
        self.reprice_count = 0
        self.emergency_active = False
        self._bd_placed = False

    def stop(self) -> None:
        self._stop = True

    # ── helpers ───────────────────────────────────────────────────────

    def _sleep(self, sec: float) -> None:
        end = time.time() + sec
        while time.time() < end and not self._stop:
            time.sleep(min(0.1, end - time.time()))

    def _tol(self) -> Decimal:
        return Decimal(str(self.cfg.strategy.size_tolerance))

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

    def _book_mid(self, book: BookTop) -> Decimal:
        assert self.info is not None
        mid = book.mid if book.mid > 0 else book.mark_price
        return quantize_down(mid, self.info.quote_increment)

    def _aggressive_price(self, side: Side, book: BookTop) -> Decimal:
        assert self.info is not None
        tick = self.info.quote_increment
        if side == Side.BUY:
            px = book.best_ask if book.best_ask > 0 else book.mid
        else:
            px = book.best_bid if book.best_bid > 0 else book.mid
        return quantize_down(px, tick)

    def _close_price_from_entry(self, entry: Decimal) -> Decimal:
        assert self.info is not None
        pct = Decimal(str(self.cfg.strategy.close_price_pct)) / Decimal("100")
        if self.cfg.strategy.close_direction == "down":
            px = entry * (Decimal("1") - pct)
        else:
            px = entry * (Decimal("1") + pct)
        return quantize_down(px, self.info.quote_increment)

    def _filled_long(self) -> Decimal:
        return self.acc1.position_qty(self.market, want_long=True)

    def _filled_short(self) -> Decimal:
        return self.acc2.position_qty(self.market, want_long=False)

    def _both_flat(self) -> bool:
        return self._filled_long() <= self._tol() and self._filled_short() <= self._tol()

    def _acc1_flat(self) -> bool:
        return self._filled_long() <= self._tol()

    def _acc2_flat(self) -> bool:
        return self._filled_short() <= self._tol()

    def _is_open(self, order: Optional[Order]) -> bool:
        return bool(order and order.order_id and order.is_open)

    def _refresh_order(self, client: OndoClient, order: Optional[Order]) -> Optional[Order]:
        if not order or not order.order_id:
            return order
        try:
            return client.get_order(order.order_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] get_order failed: %s", client.name, exc)
            return order

    def _safe_cancel(self, client: OndoClient, order: Optional[Order]) -> Optional[Order]:
        if not order or not order.order_id:
            return order
        if order.status in (OrderStatus.FULLY_FILLED, OrderStatus.CANCELED):
            return order
        try:
            return client.cancel_order(order.order_id) or order
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] cancel failed: %s", client.name, exc)
            return self._refresh_order(client, order)

    def _wait_cancel_settled(
        self, client: OndoClient, order: Optional[Order], timeout: float = 5.0
    ) -> Optional[Order]:
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

    def _place(
        self,
        client: OndoClient,
        side: Side,
        price: Decimal,
        size: Decimal,
        *,
        tag: str,
        post_only: bool = False,
        reduce_only: bool = False,
        allow_taker_fallback: bool = True,
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
                post_only=post_only,
                tag=tag,
            )
        except OndoAPIError as exc:
            if exc.code == "post_only_has_match" and post_only:
                if not allow_taker_fallback:
                    logger.warning(
                        "[%s] %s post-only rejected @ %s — no taker fallback (would fill now)",
                        client.name,
                        tag,
                        price,
                    )
                    return None
                logger.warning("[%s] %s post-only rejected @ %s — retry plain GTC", client.name, tag, price)
                try:
                    return client.place_limit(
                        self.market,
                        side,
                        price,
                        size,
                        reduce_only=reduce_only,
                        post_only=False,
                        tag=tag,
                    )
                except OndoAPIError as exc2:
                    logger.error("[%s] place %s failed: %s", client.name, tag, exc2)
                    return None
            logger.error("[%s] place %s failed: %s", client.name, tag, exc)
            return None

    # ── A B C D once ──────────────────────────────────────────────────

    def _place_abcd(self, entry_px: Decimal, size: Decimal) -> None:
        """Place all four once. No timer cancel after this."""
        assert self.info is not None
        close_px = self._close_price_from_entry(entry_px)
        self.entry_ref = entry_px
        self.close_price = close_px
        self.stats.entry_price = entry_px
        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid

        logger.info(
            "PLACE A+B+C+D once | size=%s entry=%s close=%s (+%s%%) — then REST (no 3s cancel)",
            size,
            entry_px,
            close_px,
            self.cfg.strategy.close_price_pct,
        )

        # A + C: same current book mid (GTC). Prefer post-only; fallback plain GTC.
        self.order_a = self._place(
            self.acc1, Side.BUY, entry_px, size, tag="A", post_only=True, reduce_only=False
        )
        self.order_c = self._place(
            self.acc2, Side.SELL, entry_px, size, tag="C", post_only=True, reduce_only=False
        )

        # B: sell close @ +1% — rests above market (GTC, not reduce-only)
        self.order_b = self._place(
            self.acc1, Side.SELL, close_px, size, tag="B", post_only=False, reduce_only=False
        )

        # D: buy close @ same price as B.
        # Buy limit ABOVE the market is immediately marketable — it cannot REST.
        #   post-only → exchange rejects (would take)
        #   plain GTC → fills NOW at ~mark (closes short early / wrong) — DO NOT do this
        # So when close_px > mark we do NOT place a resting D. When Acc1 later closes via B
        # (price reached +1%), the 5th-order path places D/E at the book to close Acc2.
        # When close_px <= mark (price already up), D can rest/join as post-only or GTC.
        if close_px > mark:
            self.order_d = None
            logger.info(
                "D NOT placed yet: buy limit @ %s is ABOVE mark %s so it cannot rest on the book "
                "(B sell @ %s CAN rest). D/E will place when price is near close or Acc1 goes flat.",
                close_px,
                mark,
                close_px,
            )
        else:
            self.order_d = self._place(
                self.acc2, Side.BUY, close_px, size, tag="D", post_only=True, reduce_only=False
            )

        self.order_e = None
        self.emergency_active = False
        self._bd_placed = True
        logger.info(
            "Live orders A=%s B=%s C=%s D=%s — waiting for fills (no auto-cancel timer)",
            getattr(self.order_a, "order_id", None),
            getattr(self.order_b, "order_id", None),
            getattr(self.order_c, "order_id", None),
            getattr(self.order_d, "order_id", None),
        )

    # ── ENTRY: reprice ONLY on true one-sided fill ────────────────────

    def _reprice_entry_imbalance_only(self) -> None:
        """
        Cancel/reprice ONLY the lagging entry side when the other already filled.
        If neither filled → do nothing (orders rest).
        If both filled → handled by hedge-open.
        """
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()

        # Neither filled: leave A/C alone
        if long_q <= tol and short_q <= tol:
            # Only replace if order is missing/dead (not cancel open working orders)
            book = self._book()
            entry_px = self._book_mid(book)
            if not self._is_open(self.order_a) and long_q < self.target_size - tol:
                need = self.target_size - long_q
                logger.info("A missing/dead — re-place once @ %s size=%s", entry_px, need)
                self.order_a = self._place(
                    self.acc1, Side.BUY, entry_px, need, tag="A", post_only=True
                )
            if not self._is_open(self.order_c) and short_q < self.target_size - tol:
                need = self.target_size - short_q
                logger.info("C missing/dead — re-place once @ %s size=%s", entry_px, need)
                self.order_c = self._place(
                    self.acc2, Side.SELL, entry_px, need, tag="C", post_only=True
                )
            self.last_reprice = time.time()
            return

        # Both filled enough
        if long_q >= self.target_size - tol and short_q >= self.target_size - tol:
            return

        # True imbalance: one side has fill, other lagging
        book = self._book()
        self.reprice_count += 1
        self.stats.reprice_count = self.reprice_count

        if long_q > tol and short_q < self.target_size - tol:
            # Acc1 filled (or partial), Acc2 lagging → cancel/reprice only C
            need = self.target_size - short_q
            # Match current book for lagging side
            px = self._aggressive_price(Side.SELL, book)
            logger.info(
                "IMBALANCE: long filled=%s short=%s — cancel C only, reprice C sell @ %s size=%s",
                long_q,
                short_q,
                px,
                need,
            )
            if self._is_open(self.order_c):
                self.order_c = self._safe_cancel(self.acc2, self.order_c)
                self.order_c = self._wait_cancel_settled(self.acc2, self.order_c)
            short_q = self._filled_short()
            need = self.target_size - short_q
            if need > tol:
                self.order_c = self._place(
                    self.acc2, Side.SELL, px, need, tag="C", post_only=False
                )

        elif short_q > tol and long_q < self.target_size - tol:
            need = self.target_size - long_q
            px = self._aggressive_price(Side.BUY, book)
            logger.info(
                "IMBALANCE: short filled=%s long=%s — cancel A only, reprice A buy @ %s size=%s",
                short_q,
                long_q,
                px,
                need,
            )
            if self._is_open(self.order_a):
                self.order_a = self._safe_cancel(self.acc1, self.order_a)
                self.order_a = self._wait_cancel_settled(self.acc1, self.order_a)
            long_q = self._filled_long()
            need = self.target_size - long_q
            if need > tol:
                self.order_a = self._place(
                    self.acc1, Side.BUY, px, need, tag="A", post_only=False
                )

        self.last_reprice = time.time()

    # ── B/D: place once if missing; NEVER timer-cancel while both open ─

    def _ensure_bd_if_missing(self) -> None:
        """If B or D never got placed / died, place once. Do not cancel open B/D."""
        long_q = self._filled_long()
        short_q = self._filled_short()
        if long_q <= self._tol() or short_q <= self._tol():
            return  # one-sided → 5th order owns it
        if self.close_price <= 0:
            self.close_price = self._close_price_from_entry(self.entry_ref or self._book_mid(self._book()))

        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid

        self.order_b = self._refresh_order(self.acc1, self.order_b)
        self.order_d = self._refresh_order(self.acc2, self.order_d)

        if long_q > self._tol() and not self._is_open(self.order_b):
            # only re-place if fully gone — not a timed cancel
            if self.order_b is None or self.order_b.is_terminal:
                logger.info("B missing — place once sell @ %s size=%s", self.close_price, long_q)
                self.order_b = self._place(
                    self.acc1, Side.SELL, self.close_price, long_q, tag="B", post_only=False
                )

        if short_q > self._tol() and not self._is_open(self.order_d):
            if self.order_d is None or (self.order_d.is_terminal and self.order_d.status != OrderStatus.FULLY_FILLED):
                # Only place resting D if close is at/below mark (otherwise wait for 5th path)
                if self.close_price <= mark:
                    logger.info("D missing — place once buy @ %s size=%s", self.close_price, short_q)
                    self.order_d = self._place(
                        self.acc2, Side.BUY, self.close_price, short_q, tag="D", post_only=True
                    )
                # else: silent wait — do not spam logs

    # ── 5th order only ────────────────────────────────────────────────

    def _maybe_fifth_order(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()

        if long_q > tol and short_q > tol:
            self.emergency_active = False
            return
        if long_q <= tol and short_q <= tol:
            self.emergency_active = False
            return

        book = self._book()
        self.emergency_active = True

        if long_q <= tol and short_q > tol:
            logger.info(
                "5TH: Acc1 flat, Acc2 short=%s — cancel D (if any), place E at book",
                short_q,
            )
            if self._is_open(self.order_d):
                self.order_d = self._safe_cancel(self.acc2, self.order_d)
                self.order_d = self._wait_cancel_settled(self.acc2, self.order_d)
            if self._is_open(self.order_e):
                self.order_e = self._safe_cancel(self.acc2, self.order_e)
                self.order_e = self._wait_cancel_settled(self.acc2, self.order_e)
            short_q = self._filled_short()
            if short_q <= tol:
                return
            px = self._aggressive_price(Side.BUY, book)
            self.order_e = self._place(
                self.acc2, Side.BUY, px, short_q, tag="E", post_only=False
            )
            logger.info("5TH E buy @ %s size=%s", px, short_q)
            self.last_reprice = time.time()
            self.reprice_count += 1
            return

        if short_q <= tol and long_q > tol:
            logger.info(
                "5TH: Acc2 flat, Acc1 long=%s — cancel B (if any), place E at book",
                long_q,
            )
            if self._is_open(self.order_b):
                self.order_b = self._safe_cancel(self.acc1, self.order_b)
                self.order_b = self._wait_cancel_settled(self.acc1, self.order_b)
            if self._is_open(self.order_e):
                self.order_e = self._safe_cancel(self.acc1, self.order_e)
                self.order_e = self._wait_cancel_settled(self.acc1, self.order_e)
            long_q = self._filled_long()
            if long_q <= tol:
                return
            px = self._aggressive_price(Side.SELL, book)
            self.order_e = self._place(
                self.acc1, Side.SELL, px, long_q, tag="E", post_only=False
            )
            logger.info("5TH E sell @ %s size=%s", px, long_q)
            self.last_reprice = time.time()
            self.reprice_count += 1

    def _reprice_fifth_if_needed(self) -> None:
        """Only the emergency E order may be cancelled/repriced on the timer."""
        if not self.emergency_active:
            return
        if time.time() - self.last_reprice < self.cfg.strategy.reprice_sec:
            return
        if not self._both_flat() and (self._acc1_flat() ^ self._acc2_flat()):
            self._maybe_fifth_order()

    # ── cycle ─────────────────────────────────────────────────────────

    def _start_cycle(self) -> bool:
        self.cycle += 1
        self.market = self._next_market()
        self.stats = CycleStats(cycle_id=self.cycle, market=self.market)
        self.order_a = self.order_b = self.order_c = self.order_d = self.order_e = None
        self.reprice_count = 0
        self.emergency_active = False
        self._bd_placed = False
        self.close_price = Decimal("0")

        logger.info("========== CYCLE %s | %s ==========", self.cycle, self.market)

        self.info = self.acc1.get_market_info(self.market)
        lev = min(self.cfg.leverage, self.cfg.max_leverage_for(self.market), self.info.max_leverage)
        self.acc1.set_leverage(self.market, lev)
        self.acc2.set_leverage(self.market, lev)
        self.acc1.cancel_open_bot_orders(self.market)
        self.acc2.cancel_open_bot_orders(self.market)

        if not self._both_flat():
            long_q = self._filled_long()
            short_q = self._filled_short()
            logger.warning("Residual long=%s short=%s", long_q, short_q)
            if abs(long_q - short_q) <= self._tol() and long_q > self._tol():
                p1 = self.acc1.get_position(self.market)
                p2 = self.acc2.get_position(self.market)
                e1 = p1.average_entry_price if p1 else self._book_mid(self._book())
                e2 = p2.average_entry_price if p2 else e1
                self.entry_ref = (e1 + e2) / 2
                self.target_size = min(long_q, short_q)
                self.close_price = self._close_price_from_entry(self.entry_ref)
                self._ensure_bd_if_missing()
                self.stage = Stage.EXIT
                self.stage_started = time.time()
                self.last_reprice = time.time()
                return True
            self.stage = Stage.EXIT
            self.emergency_active = True
            self.last_reprice = 0.0
            self._maybe_fifth_order()
            return True

        bal1 = self.acc1.get_balance()
        bal2 = self.acc2.get_balance()
        if self.cfg.risk.stop_on_liquidation and (bal1.under_liquidation or bal2.under_liquidation):
            self.stage = Stage.STOPPED
            self._stop = True
            return False

        book = self._book()
        size, ref, note = compute_equal_size(self.cfg, self.market, self.info, book, bal1, bal2)
        logger.info("Sizing: %s", note)
        ok, size, liq_msg = check_liq_buffer(self.cfg, ref, lev, size, self.info)
        logger.info("Liq: %s", liq_msg)
        if not ok or size <= 0:
            self.stage = Stage.IDLE
            self._sleep(self.cfg.strategy.cycle_pause_sec)
            return False

        self.target_size = size
        self.stats.target_size = size
        entry_px = self._book_mid(book)
        self._place_abcd(entry_px, size)

        self.stage = Stage.ENTRY
        self.stage_started = time.time()
        self.last_reprice = time.time()
        return True

    def _on_hedge_open(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        self.target_size = min(long_q, short_q)
        p1 = self.acc1.get_position(self.market)
        p2 = self.acc2.get_position(self.market)
        e1 = p1.average_entry_price if p1 else self.entry_ref
        e2 = p2.average_entry_price if p2 else self.entry_ref
        self.entry_ref = (e1 + e2) / 2
        self.stats.entry_price = self.entry_ref
        if self.close_price <= 0:
            self.close_price = self._close_price_from_entry(self.entry_ref)

        _, m1 = post_fill_liq_ok(self.cfg, self.acc1, self.market, e1)
        _, m2 = post_fill_liq_ok(self.cfg, self.acc2, self.market, e2)
        logger.info(
            "HEDGE OPEN size=%s entry≈%s | B/D REST @ %s (no cancel timer) | %s | %s",
            self.target_size,
            self.entry_ref,
            self.close_price,
            m1,
            m2,
        )

        # Cancel leftover ENTRY orders only (A/C), never touch B/D here unless missing
        if self._is_open(self.order_a):
            self.order_a = self._safe_cancel(self.acc1, self.order_a)
        if self._is_open(self.order_c):
            self.order_c = self._safe_cancel(self.acc2, self.order_c)

        self._ensure_bd_if_missing()
        self.stage = Stage.EXIT
        self.stage_started = time.time()
        # Do not set last_reprice to force immediate cancel — B/D must rest
        self.last_reprice = time.time()

    def run(self) -> None:
        logger.info(
            "Bot start | A/C rest at book mid | B/D rest at +%s%% | "
            "cancel ONLY on entry imbalance or 5th-order stuck close | dry=%s",
            self.cfg.strategy.close_price_pct,
            self.cfg.bot.dry_run,
        )

        while not self._stop:
            try:
                if (
                    self.cfg.bot.max_cycles
                    and self.cycle >= self.cfg.bot.max_cycles
                    and self.stage == Stage.IDLE
                ):
                    logger.info("max_cycles reached")
                    break
                if self.stage == Stage.STOPPED:
                    break

                if self.stage == Stage.IDLE:
                    if not self._start_cycle() and self.stage == Stage.IDLE:
                        self._sleep(self.cfg.strategy.cycle_pause_sec)
                    continue

                if self.stage == Stage.ENTRY:
                    self.order_a = self._refresh_order(self.acc1, self.order_a)
                    self.order_c = self._refresh_order(self.acc2, self.order_c)
                    long_q = self._filled_long()
                    short_q = self._filled_short()
                    tol = self._tol()

                    # Both entries done
                    if long_q >= self.target_size - tol and short_q >= self.target_size - tol:
                        self._on_hedge_open()
                        continue

                    # Equal partial fills and both entry orders done
                    if (
                        abs(long_q - short_q) <= tol
                        and long_q > tol
                        and (not self._is_open(self.order_a))
                        and (not self._is_open(self.order_c))
                    ):
                        self._on_hedge_open()
                        continue

                    # True imbalance only → reprice lagging side after reprice_sec
                    imbalanced = (long_q > tol and short_q < long_q - tol) or (
                        short_q > tol and long_q < short_q - tol
                    )
                    if imbalanced and time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                        self._reprice_entry_imbalance_only()
                    elif (
                        long_q <= tol
                        and short_q <= tol
                        and (not self._is_open(self.order_a) or not self._is_open(self.order_c))
                        and time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec
                    ):
                        # Orders missing (rejected), not "cancel working orders"
                        self._reprice_entry_imbalance_only()
                    # else: both resting unfilled → do NOTHING (no cancel)

                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                if self.stage == Stage.EXIT:
                    self.order_b = self._refresh_order(self.acc1, self.order_b)
                    self.order_d = self._refresh_order(self.acc2, self.order_d)
                    if self.order_e:
                        owner = self.acc1 if self._filled_long() > self._tol() else self.acc2
                        self.order_e = self._refresh_order(owner, self.order_e)

                    if self._both_flat():
                        logger.info(
                            "CYCLE %s done | entry=%s close=%s 5th=%s",
                            self.cycle,
                            self.stats.entry_price,
                            self.close_price,
                            self.emergency_active,
                        )
                        self.stage = Stage.IDLE
                        self._sleep(self.cfg.strategy.cycle_pause_sec)
                        continue

                    # One account flat, other not → 5th order only
                    if self._acc1_flat() ^ self._acc2_flat():
                        if not self.emergency_active:
                            self._maybe_fifth_order()
                        else:
                            self._reprice_fifth_if_needed()
                    else:
                        # Both still open: B and D REST — never cancel on timer
                        self.emergency_active = False
                        self._ensure_bd_if_missing()

                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                self._sleep(self.cfg.strategy.poll_interval_sec)
            except KeyboardInterrupt:
                self._stop = True
            except Exception:
                logger.exception("Loop error")
                self._sleep(2.0)

        try:
            if self.market:
                self.acc1.cancel_open_bot_orders(self.market)
                self.acc2.cancel_open_bot_orders(self.market)
        except Exception:  # noqa: BLE001
            pass
        logger.info("Bot stopped")
