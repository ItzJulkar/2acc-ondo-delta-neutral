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
    Dual-account delta-neutral bot (maker limits).

    A (acc1 long entry) + C (acc2 short entry) — same price, same size.
    B (acc1 close sell) + D (acc2 close buy) — same close price (entry ± 1%),
    placed in the SAME step as A/C. No exchange TP/SL.

    Special case (5th order only when needed):
      Acc1 fully flat (A+B done) but Acc2 still short (C filled, D not)
        → cancel D → place new limit at current book so short closes.
      Symmetric if Acc2 flat but Acc1 still long.
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
        self.order_e: Optional[Order] = None  # 5th emergency close
        self.stage_started = 0.0
        self.last_reprice = 0.0
        self.reprice_count = 0
        self.emergency_active = False

    def stop(self) -> None:
        self._stop = True

    # ── helpers ───────────────────────────────────────────────────────

    def _sleep(self, sec: float) -> None:
        end = time.time() + sec
        while time.time() < end and not self._stop:
            time.sleep(min(0.1, end - time.time()))

    def _tol(self) -> Decimal:
        return Decimal(str(self.cfg.strategy.size_tolerance))

    def _want_maker(self) -> bool:
        # Default: maker (post-only). fast_limit still uses limits, no market TP/SL.
        return self.cfg.strategy.order_mode != "taker"

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

    def _maker_price(self, side: Side, book: BookTop) -> Decimal:
        """Join book as maker (post-only friendly)."""
        assert self.info is not None
        tick = self.info.quote_increment
        offset = Decimal(self.cfg.strategy.book_offset_ticks) * tick
        if side == Side.BUY:
            px = (book.best_bid if book.best_bid > 0 else book.mid) - offset
        else:
            px = (book.best_ask if book.best_ask > 0 else book.mid) + offset
        px = quantize_down(px, tick)
        if px <= 0:
            px = quantize_down(book.mid, tick)
        return px

    def _aggressive_limit_price(self, side: Side, book: BookTop) -> Decimal:
        """Match current book for urgent close (5th order)."""
        assert self.info is not None
        tick = self.info.quote_increment
        if side == Side.BUY:
            px = book.best_ask if book.best_ask > 0 else book.mid
        else:
            px = book.best_bid if book.best_bid > 0 else book.mid
        return quantize_down(px, tick)

    def _same_entry_price(self, book: BookTop) -> Decimal:
        assert self.info is not None
        mid = book.mid if book.mid > 0 else book.mark_price
        return quantize_down(mid, self.info.quote_increment)

    def _close_price_from_entry(self, entry: Decimal) -> Decimal:
        assert self.info is not None
        pct = Decimal(str(self.cfg.strategy.close_price_pct)) / Decimal("100")
        direction = self.cfg.strategy.close_direction
        if direction == "down":
            px = entry * (Decimal("1") - pct)
        else:
            # up (default): B/D at entry + 1%
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
        post_only: bool,
        reduce_only: bool = False,
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
                # NO take_profit / stop_loss — user does not want TP/SL
            )
        except OndoAPIError as exc:
            if exc.code == "post_only_has_match" and post_only:
                logger.warning("[%s] %s post-only rejected @ %s", client.name, tag, price)
                return None
            logger.error("[%s] place %s failed: %s", client.name, tag, exc)
            return None

    # ── place A B C D together ────────────────────────────────────────

    def _place_abcd(self, entry_px: Decimal, size: Decimal) -> None:
        assert self.info is not None
        close_px = self._close_price_from_entry(entry_px)
        self.entry_ref = entry_px
        self.close_price = close_px
        self.stats.entry_price = entry_px
        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid
        maker = self._want_maker()

        logger.info(
            "PLACE A+B+C+D together (no TP/SL) | size=%s entry=%s close=%s (+%s%%) maker=%s",
            size,
            entry_px,
            close_px,
            self.cfg.strategy.close_price_pct,
            maker,
        )

        # A: long entry @ entry (maker join bid)
        a_px = entry_px
        if maker:
            a_px = self._maker_price(Side.BUY, book)
            # Keep A/C same price — use shared entry mid; post-only on both
            a_px = entry_px
        self.order_a = self._place(
            self.acc1, Side.BUY, a_px, size, tag="A", post_only=maker, reduce_only=False
        )

        # C: short entry @ same entry price
        self.order_c = self._place(
            self.acc2, Side.SELL, entry_px, size, tag="C", post_only=maker, reduce_only=False
        )

        # B: close long @ close_px (sell above market rests as maker when close > mark)
        b_post = maker and close_px >= mark
        self.order_b = self._place(
            self.acc1, Side.SELL, close_px, size, tag="B", post_only=b_post, reduce_only=False
        )
        if self.order_b is None and b_post:
            self.order_b = self._place(
                self.acc1, Side.SELL, close_px, size, tag="B", post_only=False, reduce_only=False
            )

        # D: close short @ same close_px as B
        # Buy limit ABOVE market cannot rest as maker (would take immediately).
        # Still submit post-only at close_px; if rejected, D stays None until 5th-order path
        # OR until book reaches close (then we re-place).
        d_post = True if maker else False
        if close_px > mark:
            self.order_d = self._place(
                self.acc2, Side.BUY, close_px, size, tag="D", post_only=True, reduce_only=False
            )
            if self.order_d is None:
                logger.info(
                    "D post-only @ %s not restable above mark %s — will use 5th order "
                    "only if Acc1 closes first while Acc2 short remains",
                    close_px,
                    mark,
                )
        else:
            self.order_d = self._place(
                self.acc2, Side.BUY, close_px, size, tag="D", post_only=d_post, reduce_only=False
            )
            if self.order_d is None and d_post:
                self.order_d = self._place(
                    self.acc2, Side.BUY, close_px, size, tag="D", post_only=False, reduce_only=False
                )

        self.order_e = None
        self.emergency_active = False
        logger.info(
            "Submitted A=%s B=%s C=%s D=%s",
            getattr(self.order_a, "order_id", None),
            getattr(self.order_b, "order_id", None),
            getattr(self.order_c, "order_id", None),
            getattr(self.order_d, "order_id", None),
        )

    # ── 5th order: only when one account fully closed, other still open ─

    def _maybe_fifth_order(self) -> None:
        """
        Req: Acc1 flat (A+B done) but Acc2 still short (C done, D not)
          → cancel D → place E at current book to close short.
        Symmetric if Acc2 flat and Acc1 still long.
        Do NOT place E while both accounts still have positions (B/D still working).
        """
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()

        # Both still open → B/D only, no 5th order
        if long_q > tol and short_q > tol:
            self.emergency_active = False
            return

        # Both flat → nothing
        if long_q <= tol and short_q <= tol:
            self.emergency_active = False
            return

        book = self._book()
        self.emergency_active = True

        # Acc1 flat, Acc2 still short → close short with E (replace D)
        if long_q <= tol and short_q > tol:
            logger.info(
                "5TH ORDER path: Acc1 flat but Acc2 short=%s — cancel D, place book close",
                short_q,
            )
            if self.order_d and self.order_d.is_open:
                self.order_d = self._safe_cancel(self.acc2, self.order_d)
                self.order_d = self._wait_cancel_settled(self.acc2, self.order_d)
            if self.order_e and self.order_e.is_open:
                self.order_e = self._safe_cancel(self.acc2, self.order_e)
                self.order_e = self._wait_cancel_settled(self.acc2, self.order_e)

            short_q = self._filled_short()
            if short_q <= tol:
                return
            # Match order book current price (urgent close)
            px = self._aggressive_limit_price(Side.BUY, book)
            # Prefer maker join first; if user needs fill, use book match without post-only
            self.order_e = self._place(
                self.acc2, Side.BUY, px, short_q, tag="E", post_only=False, reduce_only=False
            )
            logger.info("5TH E: buy close short size=%s @ %s", short_q, px)
            self.last_reprice = time.time()
            self.reprice_count += 1
            return

        # Acc2 flat, Acc1 still long → close long with E (replace B)
        if short_q <= tol and long_q > tol:
            logger.info(
                "5TH ORDER path: Acc2 flat but Acc1 long=%s — cancel B, place book close",
                long_q,
            )
            if self.order_b and self.order_b.is_open:
                self.order_b = self._safe_cancel(self.acc1, self.order_b)
                self.order_b = self._wait_cancel_settled(self.acc1, self.order_b)
            if self.order_e and self.order_e.is_open:
                self.order_e = self._safe_cancel(self.acc1, self.order_e)
                self.order_e = self._wait_cancel_settled(self.acc1, self.order_e)

            long_q = self._filled_long()
            if long_q <= tol:
                return
            px = self._aggressive_limit_price(Side.SELL, book)
            self.order_e = self._place(
                self.acc1, Side.SELL, px, long_q, tag="E", post_only=False, reduce_only=False
            )
            logger.info("5TH E: sell close long size=%s @ %s", long_q, px)
            self.last_reprice = time.time()
            self.reprice_count += 1

    def _reprice_fifth_if_needed(self) -> None:
        if not self.emergency_active:
            return
        if time.time() - self.last_reprice < self.cfg.strategy.reprice_sec:
            return
        # Still one-sided? re-run 5th path (cancel + new book price)
        if not self._both_flat() and (self._acc1_flat() ^ self._acc2_flat()):
            self._maybe_fifth_order()

    def _reprice_entry_imbalance(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        need_long = self.target_size - long_q
        need_short = self.target_size - short_q
        logger.info("ENTRY reprice long=%s short=%s needL=%s needS=%s", long_q, short_q, need_long, need_short)

        max_rep = self.cfg.strategy.max_reprice_attempts
        if max_rep and self.reprice_count >= max_rep:
            return

        book = self._book()
        maker = self._want_maker()
        self.reprice_count += 1
        self.stats.reprice_count = self.reprice_count

        if need_long > self._tol():
            if self.order_a and self.order_a.is_open:
                self.order_a = self._safe_cancel(self.acc1, self.order_a)
                self.order_a = self._wait_cancel_settled(self.acc1, self.order_a)
            long_q = self._filled_long()
            need_long = self.target_size - long_q
            if need_long > self._tol():
                px = self._maker_price(Side.BUY, book) if maker else self._aggressive_limit_price(Side.BUY, book)
                self.order_a = self._place(
                    self.acc1, Side.BUY, px, need_long, tag="A", post_only=maker, reduce_only=False
                )
                if self.order_a is None and maker:
                    self.order_a = self._place(
                        self.acc1,
                        Side.BUY,
                        self._aggressive_limit_price(Side.BUY, book),
                        need_long,
                        tag="A",
                        post_only=False,
                        reduce_only=False,
                    )

        if need_short > self._tol():
            if self.order_c and self.order_c.is_open:
                self.order_c = self._safe_cancel(self.acc2, self.order_c)
                self.order_c = self._wait_cancel_settled(self.acc2, self.order_c)
            short_q = self._filled_short()
            need_short = self.target_size - short_q
            if need_short > self._tol():
                px = self._maker_price(Side.SELL, book) if maker else self._aggressive_limit_price(Side.SELL, book)
                self.order_c = self._place(
                    self.acc2, Side.SELL, px, need_short, tag="C", post_only=maker, reduce_only=False
                )
                if self.order_c is None and maker:
                    self.order_c = self._place(
                        self.acc2,
                        Side.SELL,
                        self._aggressive_limit_price(Side.SELL, book),
                        need_short,
                        tag="C",
                        post_only=False,
                        reduce_only=False,
                    )

        self.last_reprice = time.time()

    def _ensure_bd_resting(self) -> None:
        """Keep B/D at close_price while BOTH sides still have positions."""
        if self.emergency_active:
            return
        long_q = self._filled_long()
        short_q = self._filled_short()
        if long_q <= self._tol() or short_q <= self._tol():
            return  # one side done → 5th order path owns closes
        if self.close_price <= 0:
            self.close_price = self._close_price_from_entry(self.entry_ref or self._book().mid)

        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid
        maker = self._want_maker()

        self.order_b = self._refresh_order(self.acc1, self.order_b)
        self.order_d = self._refresh_order(self.acc2, self.order_d)

        if long_q > self._tol() and (not self.order_b or self.order_b.is_terminal):
            b_post = maker and self.close_price >= mark
            self.order_b = self._place(
                self.acc1,
                Side.SELL,
                self.close_price,
                long_q,
                tag="B",
                post_only=b_post,
                reduce_only=False,
            )

        if short_q > self._tol() and (not self.order_d or self.order_d.is_terminal):
            if self.close_price > mark:
                self.order_d = self._place(
                    self.acc2,
                    Side.BUY,
                    self.close_price,
                    short_q,
                    tag="D",
                    post_only=True,
                    reduce_only=False,
                )
            else:
                self.order_d = self._place(
                    self.acc2,
                    Side.BUY,
                    self.close_price,
                    short_q,
                    tag="D",
                    post_only=maker,
                    reduce_only=False,
                )

    # ── cycle ─────────────────────────────────────────────────────────

    def _start_cycle(self) -> bool:
        self.cycle += 1
        self.market = self._next_market()
        self.stats = CycleStats(cycle_id=self.cycle, market=self.market)
        self.order_a = self.order_b = self.order_c = self.order_d = self.order_e = None
        self.reprice_count = 0
        self.emergency_active = False
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
                e1 = p1.average_entry_price if p1 else self._book().mid
                e2 = p2.average_entry_price if p2 else e1
                self.entry_ref = (e1 + e2) / 2
                self.target_size = min(long_q, short_q)
                self.close_price = self._close_price_from_entry(self.entry_ref)
                self._ensure_bd_resting()
                self.stage = Stage.EXIT
                self.stage_started = time.time()
                self.last_reprice = time.time()
                return True
            # One-sided residual → 5th path immediately
            self.stage = Stage.EXIT
            self.emergency_active = True
            self.last_reprice = 0.0
            self._maybe_fifth_order()
            return True

        bal1 = self.acc1.get_balance()
        bal2 = self.acc2.get_balance()
        if self.cfg.risk.stop_on_liquidation and (bal1.under_liquidation or bal2.under_liquidation):
            logger.error("Under liquidation — stop")
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
        entry_px = self._same_entry_price(book)
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
        # Keep original close target if already set; else recompute from fill entry
        if self.close_price <= 0:
            self.close_price = self._close_price_from_entry(self.entry_ref)

        ok1, m1 = post_fill_liq_ok(self.cfg, self.acc1, self.market, e1)
        ok2, m2 = post_fill_liq_ok(self.cfg, self.acc2, self.market, e2)
        logger.info(
            "HEDGE OPEN size=%s entry≈%s B/D close@%s | %s | %s",
            self.target_size,
            self.entry_ref,
            self.close_price,
            m1,
            m2,
        )

        # Cancel leftover entry if any; B/D already placed with A/C
        if self.order_a and self.order_a.is_open:
            self.order_a = self._safe_cancel(self.acc1, self.order_a)
        if self.order_c and self.order_c.is_open:
            self.order_c = self._safe_cancel(self.acc2, self.order_c)

        self._ensure_bd_resting()
        self.stage = Stage.EXIT
        self.stage_started = time.time()
        self.last_reprice = time.time()

    def run(self) -> None:
        logger.info(
            "Bot start | A+C same price/size opposite | B+D with them @ ±%s%% | "
            "no TP/SL | 5th order only if one acc flat other not | maker=%s dry=%s",
            self.cfg.strategy.close_price_pct,
            self._want_maker(),
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

                    # One account already closed while other open during entry? rare
                    if (long_q <= self._tol()) ^ (short_q <= self._tol()):
                        if long_q > self._tol() or short_q > self._tol():
                            self.stage = Stage.EXIT
                            self._maybe_fifth_order()
                            continue

                    if long_q >= self.target_size - self._tol() and short_q >= self.target_size - self._tol():
                        self._on_hedge_open()
                    elif abs(long_q - short_q) <= self._tol() and long_q > self._tol():
                        a_done = not self.order_a or self.order_a.is_terminal
                        c_done = not self.order_c or self.order_c.is_terminal
                        if a_done and c_done:
                            self._on_hedge_open()
                        elif time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                            self._reprice_entry_imbalance()
                    elif time.time() - self.last_reprice >= self.cfg.strategy.reprice_sec:
                        self._reprice_entry_imbalance()
                    self._sleep(self.cfg.strategy.poll_interval_sec)
                    continue

                if self.stage == Stage.EXIT:
                    self.order_b = self._refresh_order(self.acc1, self.order_b)
                    self.order_d = self._refresh_order(self.acc2, self.order_d)
                    self.order_e = self._refresh_order(
                        self.acc1 if self._filled_long() > self._tol() else self.acc2,
                        self.order_e,
                    )

                    if self._both_flat():
                        logger.info(
                            "CYCLE %s done | entry=%s close=%s 5th_used=%s reprices=%s",
                            self.cycle,
                            self.stats.entry_price,
                            self.close_price,
                            self.emergency_active,
                            self.stats.reprice_count,
                        )
                        self.stage = Stage.IDLE
                        self._sleep(self.cfg.strategy.cycle_pause_sec)
                        continue

                    # Core rule: one account fully closed, other not → 5th order only then
                    if self._acc1_flat() ^ self._acc2_flat():
                        if not self.emergency_active:
                            self._maybe_fifth_order()
                        else:
                            self._reprice_fifth_if_needed()
                    else:
                        # Both still open — leave B/D at 1%; refresh if missing
                        self.emergency_active = False
                        self._ensure_bd_resting()

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
