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

    B + D: placed with A/C at close target (default 0.05% price ≈ 1% ROI @ 20x).
           REST until fill — no timed cancel while waiting for target.

    Dual book close (NOT E): if price already moved past the close target but
           BOTH positions still open (B and D stuck) → cancel B+D and place
           simultaneous book-price closes on BOTH accounts.

    E (5th): ONLY when one account is fully flat and the other is not
           (B filled / D not, or D filled / B not) → reprice lagging side only.
    """

    def __init__(
        self,
        cfg: AppConfig,
        acc1: OndoClient,
        acc2: OndoClient,
        *,
        fixed_market: Optional[str] = None,
        margin_share: float = 1.0,
    ):
        self.cfg = cfg
        self.acc1 = acc1
        self.acc2 = acc2
        self.fixed_market = fixed_market
        self.margin_share = margin_share
        self.stage = Stage.IDLE
        self.cycle = 0
        self.market_idx = 0
        self._stop = False
        self.stats = CycleStats()

        self.market: str = fixed_market or ""
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
        self.emergency_active = False  # E path only
        self.dual_book_close_active = False  # both B+D stuck after target
        self._bd_placed = False
        self._idle_until = 0.0
        self._naked_abort_at = 0.0

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
        if self.fixed_market:
            return self.fixed_market
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

    def _maker_price(self, side: Side, book: BookTop) -> Decimal:
        """
        Join the book as maker only (never cross the spread).
        BUY  → best bid (or bid - offset ticks)
        SELL → best ask (or ask + offset ticks)
        """
        assert self.info is not None
        tick = self.info.quote_increment
        offset = Decimal(self.cfg.strategy.book_offset_ticks) * tick
        if side == Side.BUY:
            base = book.best_bid if book.best_bid > 0 else book.mid
            px = base - offset
        else:
            base = book.best_ask if book.best_ask > 0 else book.mid
            px = base + offset
        px = quantize_down(px, tick)
        if px <= 0:
            px = quantize_down(book.mid, tick)
        return px

    def _close_price_from_entry(self, entry: Decimal) -> Decimal:
        """
        B/D price from entry.

        close_price_pct is UNDERLYING price percent (e.g. 0.05 = 0.05% move).
        At 20x leverage that is ~1% position ROI (0.05% * 20 = 1%).
        """
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
        post_only: bool = True,
    ) -> Optional[Order]:
        """
        Default: post-only maker.
        post_only=False only for EMERGENCY naked flatten (never for normal A/B/C/D/E).
        """
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
                reduce_only=False,
                post_only=post_only,
                tag=tag,
            )
        except OndoAPIError as exc:
            if exc.code == "post_only_has_match" and post_only:
                logger.warning(
                    "[%s] %s MAKER rejected @ %s (would take) — reprice later, no taker",
                    client.name,
                    tag,
                    price,
                )
                return None
            logger.error("[%s] place %s failed: %s", client.name, tag, exc)
            return None

    # ── A B C D once ──────────────────────────────────────────────────

    def _place_ac_only(self, entry_px: Decimal, size: Decimal) -> None:
        """
        Place ONLY entry A+C (maker). Do NOT place B/D until both sides are filled.
        Prevents naked long/short if one entry fills and the other does not.
        """
        assert self.info is not None
        close_px = self._close_price_from_entry(entry_px)
        self.entry_ref = entry_px
        self.close_price = close_px
        self.stats.entry_price = entry_px
        book = self._book()

        logger.info(
            "[%s] PLACE A+C only (maker) | size=%s entry=%s | B/D after both filled | target close=%s",
            self.market,
            size,
            entry_px,
            close_px,
        )

        # Prefer same mid; if either rejects, both join bid/ask as maker (safer pair)
        self.order_a = self._place(self.acc1, Side.BUY, entry_px, size, tag="A")
        self.order_c = self._place(self.acc2, Side.SELL, entry_px, size, tag="C")
        if self.order_a is None or self.order_c is None:
            if self._is_open(self.order_a):
                self.order_a = self._safe_cancel(self.acc1, self.order_a)
            if self._is_open(self.order_c):
                self.order_c = self._safe_cancel(self.acc2, self.order_c)
            book = self._book()
            self.order_a = self._place(
                self.acc1, Side.BUY, self._maker_price(Side.BUY, book), size, tag="A"
            )
            self.order_c = self._place(
                self.acc2, Side.SELL, self._maker_price(Side.SELL, book), size, tag="C"
            )

        # Explicitly no B/D yet
        self.order_b = None
        self.order_d = None
        self.order_e = None
        self.emergency_active = False
        self.dual_book_close_active = False
        self._bd_placed = False
        self._naked_abort_at = 0.0
        logger.info(
            "[%s] Live ENTRY only A=%s C=%s (B/D deferred until BOTH filled)",
            self.market,
            getattr(self.order_a, "order_id", None),
            getattr(self.order_c, "order_id", None),
        )

    def _place_bd_after_hedge(self, size: Decimal) -> None:
        """Place B+D only when both accounts already have matching positions."""
        assert self.info is not None
        if self.close_price <= 0:
            self.close_price = self._close_price_from_entry(self.entry_ref or self._book_mid(self._book()))
        book = self._book()
        logger.info(
            "[%s] PLACE B+D after hedge confirmed | size=%s close=%s",
            self.market,
            size,
            self.close_price,
        )
        self.order_b = self._place(self.acc1, Side.SELL, self.close_price, size, tag="B")
        if self.order_b is None:
            self.order_b = self._place(
                self.acc1, Side.SELL, self._maker_price(Side.SELL, book), size, tag="B"
            )
        self.order_d = self._place_d_resting_limit(size, self.close_price)
        self._bd_placed = True
        logger.info(
            "[%s] Live CLOSE B=%s D=%s",
            self.market,
            getattr(self.order_b, "order_id", None),
            getattr(self.order_d, "order_id", None),
        )

    def _place_d_resting_limit(self, size: Decimal, close_px: Decimal) -> Optional[Order]:
        """
        Always leave an OPEN D buy limit on acc2 (no TP/SL).

        Exchange rule: a buy at close_px above the ask cannot rest (would take now).
        So if close_px would cross the ask, park D a few ticks BELOW best bid so it
        stays OPEN on the book (not filled instantly). When price hits the close
        target with both sides still open → dual-book closes both at book.
        If B fills first and D still open → E closes Acc2 only.
        """
        if size <= self._tol():
            return None
        assert self.info is not None
        book = self._book()
        tick = self.info.quote_increment
        best_ask = book.best_ask if book.best_ask > 0 else book.mid
        best_bid = book.best_bid if book.best_bid > 0 else book.mid

        # Post-only buy must be strictly below best ask
        max_rest = quantize_down(best_ask - tick, tick) if best_ask > tick else quantize_down(best_bid, tick)

        if close_px <= max_rest:
            d_px = close_px
            parked = False
        else:
            # Park deep enough under bid so D stays open (not join-the-spread fill)
            park_ticks = Decimal("5")
            d_px = quantize_down(best_bid - park_ticks * tick, tick)
            if d_px <= 0:
                d_px = quantize_down(best_bid, tick)
            # never park above target
            if d_px > close_px:
                d_px = close_px
            parked = True

        o = self._place(self.acc2, Side.BUY, d_px, size, tag="D")
        if o is None and parked:
            # Fallback: join bid as maker
            o = self._place(
                self.acc2, Side.BUY, quantize_down(best_bid, tick), size, tag="D"
            )
            if o:
                d_px = quantize_down(best_bid, tick)

        if o:
            logger.info(
                "D OPEN on acc2 @ %s size=%s id=%s | B target close=%s%s",
                d_px,
                size,
                o.order_id,
                close_px,
                " (parked below bid until dual-book/E)" if parked else " (at target)",
            )
        else:
            logger.error("D still failed to place OPEN limit on acc2")
        return o

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
                    self.acc1, Side.BUY, entry_px, need, tag="A"
                )
            if not self._is_open(self.order_c) and short_q < self.target_size - tol:
                need = self.target_size - short_q
                logger.info("C missing/dead — re-place once @ %s size=%s", entry_px, need)
                self.order_c = self._place(
                    self.acc2, Side.SELL, entry_px, need, tag="C"
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
            # Acc1 filled, Acc2 lagging → reprice C only as MAKER (join ask)
            need = self.target_size - short_q
            px = self._maker_price(Side.SELL, book)
            logger.info(
                "IMBALANCE MAKER: long=%s short=%s — reprice C sell @ %s size=%s",
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
                self.order_c = self._place(self.acc2, Side.SELL, px, need, tag="C")

        elif short_q > tol and long_q < self.target_size - tol:
            need = self.target_size - long_q
            px = self._maker_price(Side.BUY, book)
            logger.info(
                "IMBALANCE MAKER: short=%s long=%s — reprice A buy @ %s size=%s",
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
                self.order_a = self._place(self.acc1, Side.BUY, px, need, tag="A")

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
            if self.order_b is None or self.order_b.is_terminal:
                logger.info("B missing — MAKER sell @ %s size=%s", self.close_price, long_q)
                self.order_b = self._place(
                    self.acc1, Side.SELL, self.close_price, long_q, tag="B"
                )
                if self.order_b is None:
                    self.order_b = self._place(
                        self.acc1,
                        Side.SELL,
                        self._maker_price(Side.SELL, book),
                        long_q,
                        tag="B",
                    )

        if short_q > self._tol() and not self._is_open(self.order_d):
            if self.order_d is None or self.order_d.is_terminal:
                self.order_d = self._place_d_resting_limit(short_q, self.close_price)

    # ── Dual book close: BOTH B+D stuck after target move (NOT E) ─────

    def _price_past_close_target(self) -> bool:
        """True when mark has moved at least close_price_pct from entry (stuck B/D case)."""
        if self.entry_ref <= 0:
            return False
        book = self._book()
        mark = book.mark_price if book.mark_price > 0 else book.mid
        if mark <= 0:
            return False
        pct = Decimal(str(self.cfg.strategy.close_price_pct)) / Decimal("100")
        direction = self.cfg.strategy.close_direction
        if direction == "down":
            return mark <= self.entry_ref * (Decimal("1") - pct)
        if direction == "up":
            return mark >= self.entry_ref * (Decimal("1") + pct)
        # either
        move = abs(mark - self.entry_ref) / self.entry_ref
        return move + Decimal("0.0000001") >= pct

    def _dual_book_close(self) -> None:
        """
        B and D both stuck after price already hit the close target:
        cancel B+D, place simultaneous book closes on BOTH accounts.
        This is NOT the E order.
        """
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()
        if long_q <= tol or short_q <= tol:
            return

        book = self._book()
        logger.info(
            "DUAL BOOK CLOSE (not E): price past target, both still open "
            "long=%s short=%s — cancel B+D, place both at book",
            long_q,
            short_q,
        )

        if self._is_open(self.order_b):
            self.order_b = self._safe_cancel(self.acc1, self.order_b)
            self.order_b = self._wait_cancel_settled(self.acc1, self.order_b)
        if self._is_open(self.order_d):
            self.order_d = self._safe_cancel(self.acc2, self.order_d)
            self.order_d = self._wait_cancel_settled(self.acc2, self.order_d)

        long_q = self._filled_long()
        short_q = self._filled_short()
        if long_q <= tol and short_q <= tol:
            self.dual_book_close_active = False
            return

        # Same moment: close long + close short as MAKER (join ask / join bid)
        if long_q > tol:
            px_b = self._maker_price(Side.SELL, book)
            self.order_b = self._place(self.acc1, Side.SELL, px_b, long_q, tag="B")
            logger.info("DUAL MAKER B sell @ %s size=%s", px_b, long_q)
        if short_q > tol:
            px_d = self._maker_price(Side.BUY, book)
            self.order_d = self._place(self.acc2, Side.BUY, px_d, short_q, tag="D")
            logger.info("DUAL MAKER D buy @ %s size=%s", px_d, short_q)

        self.dual_book_close_active = True
        self.emergency_active = False
        self.last_reprice = time.time()
        self.reprice_count += 1
        self.stats.reprice_count = self.reprice_count

    def _reprice_dual_book_close_if_needed(self) -> None:
        """While dual book close is active and both still open, reprice both at book."""
        if not self.dual_book_close_active:
            return
        if time.time() - self.last_reprice < self.cfg.strategy.reprice_sec:
            return
        long_q = self._filled_long()
        short_q = self._filled_short()
        if long_q <= self._tol() or short_q <= self._tol():
            # became one-sided → leave dual path; E path will take over
            self.dual_book_close_active = False
            return
        self._dual_book_close()

    # ── 5th order E only (one side filled, other not) ─────────────────

    def _maybe_fifth_order(self) -> None:
        """
        E ONLY when:
          - Acc1 flat (B done) but Acc2 still short (D not done), OR
          - Acc2 flat (D done) but Acc1 still long (B not done).
        Never used when both still open.
        """
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()

        if long_q > tol and short_q > tol:
            self.emergency_active = False
            return
        if long_q <= tol and short_q <= tol:
            self.emergency_active = False
            return

        # One-sided → not dual book close
        self.dual_book_close_active = False
        book = self._book()
        self.emergency_active = True

        if long_q <= tol and short_q > tol:
            logger.info(
                "E MAKER only: Acc1 flat, Acc2 short=%s — cancel D, place E buy join bid",
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
            px = self._maker_price(Side.BUY, book)
            self.order_e = self._place(self.acc2, Side.BUY, px, short_q, tag="E")
            logger.info("E MAKER buy @ %s size=%s", px, short_q)
            self.last_reprice = time.time()
            self.reprice_count += 1
            return

        if short_q <= tol and long_q > tol:
            logger.info(
                "E MAKER only: Acc2 flat, Acc1 long=%s — cancel B, place E sell join ask",
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
            px = self._maker_price(Side.SELL, book)
            self.order_e = self._place(self.acc1, Side.SELL, px, long_q, tag="E")
            logger.info("E MAKER sell @ %s size=%s", px, long_q)
            self.last_reprice = time.time()
            self.reprice_count += 1

    def _reprice_fifth_if_needed(self) -> None:
        """E order only may be cancelled/repriced on the timer (one-sided lag)."""
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
        self.dual_book_close_active = False
        self._bd_placed = False
        self.close_price = Decimal("0")

        logger.info("========== [%s] CYCLE %s ==========", self.market, self.cycle)

        self.info = self.acc1.get_market_info(self.market)
        lev = min(self.cfg.leverage, self.cfg.max_leverage_for(self.market), self.info.max_leverage)
        self.acc1.set_leverage(self.market, lev)
        self.acc2.set_leverage(self.market, lev)
        self.acc1.cancel_open_bot_orders(self.market)
        self.acc2.cancel_open_bot_orders(self.market)
        # Never use exchange TP/SL — clear any leftovers
        self.acc1.remove_all_stops(self.market)
        self.acc2.remove_all_stops(self.market)

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
        size, ref, note = compute_equal_size(
            self.cfg,
            self.market,
            self.info,
            book,
            bal1,
            bal2,
            margin_share=self.margin_share,
        )
        logger.info("[%s] Sizing: %s", self.market, note)
        ok, size, liq_msg = check_liq_buffer(self.cfg, ref, lev, size, self.info)
        logger.info("Liq: %s", liq_msg)
        if not ok or size <= 0:
            self.stage = Stage.IDLE
            self._sleep(self.cfg.strategy.cycle_pause_sec)
            return False

        self.target_size = size
        self.stats.target_size = size
        entry_px = self._book_mid(book)
        # CRITICAL: only A+C now. B/D after both entries filled (no naked side).
        self._place_ac_only(entry_px, size)

        self.stage = Stage.ENTRY
        self.stage_started = time.time()
        self.last_reprice = time.time()
        return True

    def _abort_naked_entry(self) -> None:
        """
        If only one account has a position, cancel everything and flatten the
        exposed side. Prefer maker; if still open after reprice window, one
        emergency cross is allowed so we never sit naked.
        """
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()
        book = self._book()
        logger.error(
            "[%s] NAKED POSITION long=%s short=%s — flatten exposed side NOW",
            self.market,
            long_q,
            short_q,
        )
        for od, cl in (
            (self.order_a, self.acc1),
            (self.order_b, self.acc1),
            (self.order_c, self.acc2),
            (self.order_d, self.acc2),
            (self.order_e, self.acc1),
            (self.order_e, self.acc2),
        ):
            if self._is_open(od):
                self._safe_cancel(cl, od)

        self._naked_abort_at = time.time()
        if long_q > tol and short_q <= tol:
            # Join ask as maker first
            px = self._maker_price(Side.SELL, book)
            self.order_e = self._place(self.acc1, Side.SELL, px, long_q, tag="E")
            self.emergency_active = True
            self.stage = Stage.EXIT
        elif short_q > tol and long_q <= tol:
            px = self._maker_price(Side.BUY, book)
            self.order_e = self._place(self.acc2, Side.BUY, px, short_q, tag="E")
            self.emergency_active = True
            self.stage = Stage.EXIT
        self.last_reprice = time.time()

    def _emergency_flatten_if_still_naked(self) -> None:
        """If naked abort maker E did not fill, cross once to kill risk."""
        if self._naked_abort_at <= 0:
            return
        if time.time() - self._naked_abort_at < self.cfg.strategy.reprice_sec:
            return
        long_q = self._filled_long()
        short_q = self._filled_short()
        tol = self._tol()
        book = self._book()
        assert self.info is not None
        tick = self.info.quote_increment
        if long_q > tol and short_q <= tol:
            if self._is_open(self.order_e):
                self.order_e = self._safe_cancel(self.acc1, self.order_e)
            # Cross: sell through the bid (only emergency naked flatten)
            px = book.best_bid if book.best_bid > 0 else book.mid
            px = quantize_down(px * Decimal("0.999"), tick)
            logger.error("[%s] EMERGENCY cross sell flatten long=%s @ %s", self.market, long_q, px)
            self.order_e = self._place(
                self.acc1, Side.SELL, px, long_q, tag="E", post_only=False
            )
            self._naked_abort_at = time.time()
        elif short_q > tol and long_q <= tol:
            if self._is_open(self.order_e):
                self.order_e = self._safe_cancel(self.acc2, self.order_e)
            px = book.best_ask if book.best_ask > 0 else book.mid
            px = quantize_down(px * Decimal("1.001") + tick, tick)
            logger.error("[%s] EMERGENCY cross buy flatten short=%s @ %s", self.market, short_q, px)
            self.order_e = self._place(
                self.acc2, Side.BUY, px, short_q, tag="E", post_only=False
            )
            self._naked_abort_at = time.time()
        else:
            self._naked_abort_at = 0.0

    def _on_hedge_open(self) -> None:
        long_q = self._filled_long()
        short_q = self._filled_short()
        # Require BOTH sides — never place B/D on a one-sided fill
        if long_q <= self._tol() or short_q <= self._tol():
            self._abort_naked_entry()
            return
        if abs(long_q - short_q) > self._tol():
            # Still imbalanced — keep entry reprice, do not open B/D
            logger.warning(
                "[%s] hedge not equal yet long=%s short=%s — wait/reprice entry, no B/D",
                self.market,
                long_q,
                short_q,
            )
            return

        self.target_size = min(long_q, short_q)
        p1 = self.acc1.get_position(self.market)
        p2 = self.acc2.get_position(self.market)
        e1 = p1.average_entry_price if p1 else self.entry_ref
        e2 = p2.average_entry_price if p2 else self.entry_ref
        self.entry_ref = (e1 + e2) / 2
        self.stats.entry_price = self.entry_ref
        self.close_price = self._close_price_from_entry(self.entry_ref)

        _, m1 = post_fill_liq_ok(self.cfg, self.acc1, self.market, e1)
        _, m2 = post_fill_liq_ok(self.cfg, self.acc2, self.market, e2)
        logger.info(
            "[%s] HEDGE CONFIRMED both sides size=%s entry≈%s | placing B/D @ %s | %s | %s",
            self.market,
            self.target_size,
            self.entry_ref,
            self.close_price,
            m1,
            m2,
        )

        # Cancel leftover ENTRY orders only
        if self._is_open(self.order_a):
            self.order_a = self._safe_cancel(self.acc1, self.order_a)
        if self._is_open(self.order_c):
            self.order_c = self._safe_cancel(self.acc2, self.order_c)

        # B/D only now that hedge is real
        self._place_bd_after_hedge(self.target_size)
        self.stage = Stage.EXIT
        self.stage_started = time.time()
        self.last_reprice = time.time()

    def tick(self) -> bool:
        """
        One state-machine step for this market. Returns False if permanently stopped.
        Used by MultiMarketRunner so XAU and XAG advance in the same loop.
        """
        if self._stop or self.stage == Stage.STOPPED:
            return False
        if time.time() < self._idle_until:
            return True

        try:
            if (
                self.cfg.bot.max_cycles
                and self.cycle >= self.cfg.bot.max_cycles
                and self.stage == Stage.IDLE
            ):
                logger.info("[%s] max_cycles reached", self.market or self.fixed_market)
                self.stage = Stage.STOPPED
                return False

            if self.stage == Stage.IDLE:
                if not self._start_cycle() and self.stage == Stage.IDLE:
                    self._idle_until = time.time() + self.cfg.strategy.cycle_pause_sec
                return True

            if self.stage == Stage.ENTRY:
                self.order_a = self._refresh_order(self.acc1, self.order_a)
                self.order_c = self._refresh_order(self.acc2, self.order_c)
                long_q = self._filled_long()
                short_q = self._filled_short()
                tol = self._tol()

                # Naked risk: one side filled, other still zero after reprice window
                one_sided = (long_q > tol and short_q <= tol) or (short_q > tol and long_q <= tol)
                if one_sided and time.time() - self.stage_started >= max(
                    self.cfg.strategy.reprice_sec * 2, 6.0
                ):
                    # Still naked after reprice attempts → flatten exposed side
                    self._abort_naked_entry()
                    return True

                if long_q >= self.target_size - tol and short_q >= self.target_size - tol:
                    self._on_hedge_open()
                    return True

                if (
                    abs(long_q - short_q) <= tol
                    and long_q > tol
                    and short_q > tol
                ):
                    self._on_hedge_open()
                    return True

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
                    self._reprice_entry_imbalance_only()
                return True

            if self.stage == Stage.EXIT:
                self.order_b = self._refresh_order(self.acc1, self.order_b)
                self.order_d = self._refresh_order(self.acc2, self.order_d)
                if self.order_e:
                    owner = self.acc1 if self._filled_long() > self._tol() else self.acc2
                    self.order_e = self._refresh_order(owner, self.order_e)

                if self._both_flat():
                    logger.info(
                        "[%s] CYCLE %s done | entry=%s close=%s E=%s dual_book=%s",
                        self.market,
                        self.cycle,
                        self.stats.entry_price,
                        self.close_price,
                        self.emergency_active,
                        self.dual_book_close_active,
                    )
                    self.stage = Stage.IDLE
                    self._idle_until = time.time() + self.cfg.strategy.cycle_pause_sec
                    return True

                if self._acc1_flat() ^ self._acc2_flat():
                    self.dual_book_close_active = False
                    if self._naked_abort_at > 0:
                        self._emergency_flatten_if_still_naked()
                    if not self.emergency_active:
                        self._maybe_fifth_order()
                    else:
                        self._reprice_fifth_if_needed()
                        if self._naked_abort_at > 0:
                            self._emergency_flatten_if_still_naked()
                else:
                    self._naked_abort_at = 0.0
                    self.emergency_active = False
                    if self.dual_book_close_active:
                        self._reprice_dual_book_close_if_needed()
                    elif self._price_past_close_target():
                        self._dual_book_close()
                    else:
                        self._ensure_bd_if_missing()
                return True

            return True
        except Exception:
            logger.exception("[%s] tick error", self.market or self.fixed_market or "?")
            self._idle_until = time.time() + 2.0
            return True

    def shutdown(self) -> None:
        self._stop = True
        try:
            if self.market:
                self.acc1.cancel_open_bot_orders(self.market)
                self.acc2.cancel_open_bot_orders(self.market)
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        logger.info(
            "Bot start [%s] | B/D @ +%s%% | dual-book | E one-sided | dry=%s",
            self.fixed_market or "rotate",
            self.cfg.strategy.close_price_pct,
            self.cfg.bot.dry_run,
        )
        while not self._stop and self.stage != Stage.STOPPED:
            alive = self.tick()
            if not alive:
                break
            self._sleep(self.cfg.strategy.poll_interval_sec)
        self.shutdown()
        logger.info("Bot stopped [%s]", self.market or self.fixed_market or "")


class MultiMarketRunner:
    """
    Run XAU and XAG (or any configured markets) in parallel — each has its own
    A/B/C/D state machine; one poll loop ticks all markets every interval.
    """

    def __init__(self, cfg: AppConfig, acc1: OndoClient, acc2: OndoClient):
        self.cfg = cfg
        self.acc1 = acc1
        self.acc2 = acc2
        self._stop = False
        n = max(len(cfg.markets), 1)
        # Split margin budget across markets so total stays near margin_usage_pct
        share = 1.0 / n
        self.engines = [
            DeltaNeutralEngine(
                cfg,
                acc1,
                acc2,
                fixed_market=m,
                margin_share=share,
            )
            for m in cfg.markets
        ]

    def stop(self) -> None:
        self._stop = True
        for e in self.engines:
            e.stop()

    def run(self) -> None:
        markets = ", ".join(cfg.markets if (cfg := self.cfg) else [])
        logger.info(
            "Multi-market bot start | markets=[%s] | parallel | margin share=1/%s | "
            "close +%s%% | MAKER-ONLY postOnly | dry=%s",
            markets,
            max(len(self.cfg.markets), 1),
            self.cfg.strategy.close_price_pct,
            self.cfg.bot.dry_run,
        )
        try:
            while not self._stop:
                any_alive = False
                for eng in self.engines:
                    if eng.stage != Stage.STOPPED and not eng._stop:
                        any_alive = eng.tick() or any_alive
                if not any_alive:
                    logger.info("All market engines stopped")
                    break
                time.sleep(self.cfg.strategy.poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("Interrupted")
            self.stop()
        finally:
            for eng in self.engines:
                eng.shutdown()
            logger.info("Multi-market bot stopped")
