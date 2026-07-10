from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from bot.models import (
    Balance,
    BookTop,
    MarketInfo,
    Order,
    OrderStatus,
    Position,
    PositionDirection,
    Side,
)

logger = logging.getLogger(__name__)


class OndoAPIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(f"Ondo API error ({status}, {code}): {message}")


class OndoClient:
    """REST client for a single Ondo Perps account."""

    def __init__(
        self,
        base_url: str,
        key_id: str,
        api_secret: str,
        name: str = "acc",
        order_prefix: str = "dn2_",
        timeout: float = 30.0,
        dry_run: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.api_secret = api_secret
        self.name = name
        self.order_prefix = order_prefix
        self.timeout = timeout
        self.dry_run = dry_run
        self._session = requests.Session()
        self._dry_orders: dict[str, Order] = {}
        self._dry_pos_long = Decimal("0")
        self._dry_pos_short = Decimal("0")
        self._dry_entry = Decimal("0")

    # ── auth ──────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        payload = timestamp + method.upper() + path + body
        return hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        return {
            "ONDO-KEY-ID": self.key_id,
            "ONDO-TIMESTAMP": timestamp,
            "ONDO-SIGN": self._sign(timestamp, method, path, body),
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        auth: bool = True,
    ) -> Any:
        query = "?" + urlencode(params) if params else ""
        full_path = path + query
        url = self.base_url + full_path
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = self._headers(method, full_path, body_str) if auth else {"Content-Type": "application/json"}

        response = self._session.request(
            method=method,
            url=url,
            headers=headers,
            data=body_str if body is not None else None,
            timeout=self.timeout,
        )

        try:
            data = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError(f"Non-JSON response from {path}: {response.text[:300]}") from exc

        if response.status_code >= 400 or not data.get("success", True):
            raise OndoAPIError(
                response.status_code,
                str(data.get("error_code", "")),
                str(data.get("error", response.text)),
            )
        return data.get("result")

    # ── market data ───────────────────────────────────────────────────

    def get_market_info(self, market: str) -> MarketInfo:
        result = self._request("GET", "/v1/markets", auth=False)
        pairs = (result or {}).get("perps", {}).get("tradingPairs", [])
        for pair in pairs:
            if pair.get("market") == market:
                return MarketInfo(
                    market=market,
                    base_increment=Decimal(str(pair["baseIncrement"])),
                    quote_increment=Decimal(str(pair["quoteIncrement"])),
                    max_leverage=20,
                )
        raise ValueError(f"Market not found: {market}")

    def get_book_top(self, market: str) -> BookTop:
        mark = Decimal("0")
        try:
            marks = self._request("GET", "/v1/perps/mark_prices", auth=False) or {}
            md = marks.get(market, {}) if isinstance(marks, dict) else {}
            if isinstance(md, dict):
                mark = Decimal(str(md.get("markPrice") or md.get("price") or "0"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] mark price fetch failed: %s", self.name, exc)

        book = self._request("GET", "/v1/perps/depth", params={"market": market, "depth": 5}, auth=False) or {}
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid = Decimal(str(bids[0][0])) if bids else mark
        best_ask = Decimal(str(asks[0][0])) if asks else mark
        if mark <= 0:
            mark = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else max(best_bid, best_ask)
        return BookTop(market=market, best_bid=best_bid, best_ask=best_ask, mark_price=mark)

    # ── account ───────────────────────────────────────────────────────

    def get_balance(self) -> Balance:
        if self.dry_run:
            return Balance(
                margin_balance=Decimal("10000"),
                available_margin=Decimal("10000"),
                wallet_balance=Decimal("10000"),
                unrealized_pnl=Decimal("0"),
                used_margin=Decimal("0"),
                under_liquidation=False,
            )
        result = self._request("GET", "/v1/perps/balance") or {}
        return Balance(
            margin_balance=Decimal(str(result.get("marginBalance", "0"))),
            available_margin=Decimal(str(result.get("availableMargin", "0"))),
            wallet_balance=Decimal(str(result.get("walletBalance", "0"))),
            unrealized_pnl=Decimal(str(result.get("unrealizedPnl", "0"))),
            used_margin=Decimal(str(result.get("usedMargin", "0"))),
            under_liquidation=bool(result.get("underLiquidation", False)),
        )

    def set_leverage(self, market: str, leverage: int) -> None:
        if self.dry_run:
            logger.info("[%s] DRY set_leverage %s → %sx", self.name, market, leverage)
            return
        self._request("POST", "/v1/perps/leverage", body={"market": market, "leverage": str(leverage)})

    def get_positions(self, market: Optional[str] = None) -> list[Position]:
        if self.dry_run:
            out: list[Position] = []
            if self._dry_pos_long > 0:
                out.append(
                    Position(
                        market=market or "DRY",
                        direction=PositionDirection.LONG,
                        net_quantity=self._dry_pos_long,
                        average_entry_price=self._dry_entry,
                        unrealized_pnl=Decimal("0"),
                        mark_price=self._dry_entry,
                        liquidation_price=self._dry_entry * Decimal("0.90"),
                        notional_value=self._dry_pos_long * self._dry_entry,
                        leverage=Decimal("20"),
                    )
                )
            if self._dry_pos_short > 0:
                out.append(
                    Position(
                        market=market or "DRY",
                        direction=PositionDirection.SHORT,
                        net_quantity=self._dry_pos_short,
                        average_entry_price=self._dry_entry,
                        unrealized_pnl=Decimal("0"),
                        mark_price=self._dry_entry,
                        liquidation_price=self._dry_entry * Decimal("1.10"),
                        notional_value=self._dry_pos_short * self._dry_entry,
                        leverage=Decimal("20"),
                    )
                )
            return [p for p in out if not market or p.market == market or p.market == "DRY"]

        result = self._request("GET", "/v1/perps/positions") or []
        positions: list[Position] = []
        for item in result:
            if market and item.get("market") != market:
                continue
            direction = PositionDirection(item.get("direction", "neutral"))
            qty = Decimal(str(item.get("netQuantity", "0")))
            if direction == PositionDirection.NEUTRAL or qty == 0:
                continue
            positions.append(
                Position(
                    market=item["market"],
                    direction=direction,
                    net_quantity=qty,
                    average_entry_price=Decimal(str(item.get("averageEntryPrice", "0"))),
                    unrealized_pnl=Decimal(str(item.get("unrealizedPnl", "0"))),
                    mark_price=Decimal(str(item.get("markPrice", "0"))),
                    liquidation_price=Decimal(str(item.get("liquidationPrice", "0"))),
                    notional_value=Decimal(str(item.get("notionalValue", "0"))),
                    leverage=Decimal(str(item.get("leverage", "0"))),
                )
            )
        return positions

    def position_qty(self, market: str, want_long: bool) -> Decimal:
        for p in self.get_positions(market):
            if want_long and p.direction == PositionDirection.LONG:
                return abs(p.net_quantity)
            if not want_long and p.direction == PositionDirection.SHORT:
                return abs(p.net_quantity)
        return Decimal("0")

    def get_position(self, market: str) -> Optional[Position]:
        positions = self.get_positions(market)
        return positions[0] if positions else None

    # ── orders ────────────────────────────────────────────────────────

    def _new_client_id(self, tag: str) -> str:
        return f"{self.order_prefix}{tag}_{uuid.uuid4().hex[:12]}"

    def place_limit(
        self,
        market: str,
        side: Side,
        price: Decimal,
        size: Decimal,
        *,
        reduce_only: bool = False,
        post_only: bool = False,
        tag: str = "ord",
        client_order_id: Optional[str] = None,
        ) -> Order:
        cid = client_order_id or self._new_client_id(tag)
        # Ondo: reduce-only limits must be IOC (GTC reduce-only is rejected).
        tif = "IOC" if reduce_only else "GTC"
        body: dict[str, Any] = {
            "side": side.value,
            "market": market,
            "price": str(price),
            "size": str(size),
            "type": "limit",
            "timeInForce": tif,
            "postOnly": bool(post_only) and not reduce_only,
            "reduceOnly": bool(reduce_only),
            "clientOrderId": cid,
        }
        # Never attach takeProfit/stopLoss — user wants pure limit orders only.
        if self.dry_run:
            oid = f"dry_{uuid.uuid4().hex[:16]}"
            # Simulate instant fill for dry-run simplicity of the loop
            order = Order(
                order_id=oid,
                client_order_id=cid,
                market=market,
                side=side,
                price=price,
                size=size,
                filled_size=size,
                status=OrderStatus.FULLY_FILLED,
                reduce_only=reduce_only,
            )
            self._dry_orders[oid] = order
            if not reduce_only:
                self._dry_entry = price
                if side == Side.BUY:
                    self._dry_pos_long += size
                else:
                    self._dry_pos_short += size
            else:
                if side == Side.SELL:
                    self._dry_pos_long = max(self._dry_pos_long - size, Decimal("0"))
                else:
                    self._dry_pos_short = max(self._dry_pos_short - size, Decimal("0"))
            logger.info(
                "[%s] DRY limit %s %s size=%s @ %s reduce=%s → FILLED",
                self.name,
                side.value,
                market,
                size,
                price,
                reduce_only,
            )
            return order

        result = self._request("POST", "/v1/perps/orders", body=body) or {}
        return self._parse_order(result)

    def remove_all_stops(self, market: str) -> None:
        """Clear any TP/SL on a market (we do not use stops in this bot)."""
        try:
            self._request("DELETE", "/v1/perps/stop_order", params={"market": market})
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] remove stops %s: %s", self.name, market, exc)

    def get_order(self, order_id: str) -> Order:
        if self.dry_run and order_id in self._dry_orders:
            return self._dry_orders[order_id]
        # support client: prefix
        path_id = order_id
        result = self._request("GET", f"/v1/perps/orders/{path_id}") or {}
        return self._parse_order(result)

    def cancel_order(self, order_id: str) -> Optional[Order]:
        if self.dry_run:
            o = self._dry_orders.get(order_id)
            if o and o.status == OrderStatus.OPEN:
                o.status = OrderStatus.CANCELED
            logger.info("[%s] DRY cancel %s", self.name, order_id)
            return o
        try:
            result = self._request("DELETE", f"/v1/perps/orders/{order_id}") or {}
            return self._parse_order(result)
        except OndoAPIError as exc:
            # Already filled / canceled is fine during race
            if exc.code in (
                "order_already_canceled",
                "order_already_cancelled",
                "order_already_fully_filled",
                "order_already_filled",
                "order_not_found",
                "order_not_in_cancelable_state",
                "order_not_in_cancellable_state",
            ):
                logger.warning("[%s] cancel race %s: %s", self.name, order_id, exc.code)
                try:
                    return self.get_order(order_id)
                except Exception:  # noqa: BLE001
                    return None
            raise

    def cancel_open_bot_orders(self, market: str) -> None:
        if self.dry_run:
            return
        try:
            result = self._request(
                "GET",
                "/v1/perps/orders",
                params={"market": market, "status": "open", "limit": 1000},
            )
        except OndoAPIError as exc:
            logger.warning("[%s] list open orders failed: %s", self.name, exc)
            return
        items = result if isinstance(result, list) else (result or {}).get("orders", [])
        for item in items or []:
            cid = item.get("clientOrderId") or ""
            if cid.startswith(self.order_prefix):
                oid = item.get("orderId") or item.get("orderID")
                if oid:
                    try:
                        self.cancel_order(str(oid))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[%s] cancel %s failed: %s", self.name, oid, exc)

    def _parse_order(self, item: dict[str, Any]) -> Order:
        side_raw = str(item.get("side", "buy")).lower()
        return Order(
            order_id=str(item.get("orderId") or item.get("orderID") or ""),
            client_order_id=str(item.get("clientOrderId") or ""),
            market=str(item.get("market", "")),
            side=Side.BUY if side_raw == "buy" else Side.SELL,
            price=Decimal(str(item.get("price", "0"))),
            size=Decimal(str(item.get("size", "0"))),
            filled_size=Decimal(str(item.get("filledSize", "0"))),
            status=OrderStatus.parse(str(item.get("status", ""))),
            reduce_only=bool(item.get("reduceOnly", False)),
            fee=Decimal(str(item.get("fee", "0"))),
        )


def quantize_down(value: Decimal, increment: Decimal) -> Decimal:
    if increment <= 0:
        return value
    return (value / increment).to_integral_value(rounding=ROUND_DOWN) * increment


def format_decimal(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"
