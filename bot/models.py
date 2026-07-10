from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    OPEN = "open"
    FULLY_FILLED = "fullyfilled"
    CANCELED = "canceled"
    PENDING = "pending"
    UNTRIGGERED = "untriggered"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | None) -> "OrderStatus":
        if not value:
            return cls.UNKNOWN
        raw = value.lower().replace("_", "").replace("-", "")
        mapping = {
            "open": cls.OPEN,
            "fullyfilled": cls.FULLY_FILLED,
            "canceled": cls.CANCELED,
            "cancelled": cls.CANCELED,
            "pending": cls.PENDING,
            "untriggered": cls.UNTRIGGERED,
        }
        return mapping.get(raw, cls.UNKNOWN)


class PositionDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Stage(str, Enum):
    IDLE = "idle"
    ENTRY = "entry"
    WAIT_CLOSE_TRIGGER = "wait_close_trigger"
    EXIT = "exit"
    RECONCILE = "reconcile"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class MarketInfo:
    market: str
    base_increment: Decimal
    quote_increment: Decimal
    max_leverage: int = 20


@dataclass
class BookTop:
    market: str
    best_bid: Decimal
    best_ask: Decimal
    mark_price: Decimal

    @property
    def mid(self) -> Decimal:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.mark_price


@dataclass
class Balance:
    margin_balance: Decimal
    available_margin: Decimal
    wallet_balance: Decimal
    unrealized_pnl: Decimal
    used_margin: Decimal
    under_liquidation: bool = False


@dataclass
class Position:
    market: str
    direction: PositionDirection
    net_quantity: Decimal
    average_entry_price: Decimal
    unrealized_pnl: Decimal
    mark_price: Decimal
    liquidation_price: Decimal
    notional_value: Decimal
    leverage: Decimal

    @property
    def signed_qty(self) -> Decimal:
        if self.direction == PositionDirection.SHORT:
            return -abs(self.net_quantity)
        if self.direction == PositionDirection.LONG:
            return abs(self.net_quantity)
        return Decimal("0")


@dataclass
class Order:
    order_id: str
    client_order_id: str
    market: str
    side: Side
    price: Decimal
    size: Decimal
    filled_size: Decimal
    status: OrderStatus
    reduce_only: bool = False
    fee: Decimal = Decimal("0")

    @property
    def remaining(self) -> Decimal:
        return max(self.size - self.filled_size, Decimal("0"))

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FULLY_FILLED, OrderStatus.CANCELED)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PENDING)


@dataclass
class AccountState:
    name: str
    long_qty: Decimal = Decimal("0")
    short_qty: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    liq_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    open_order_id: Optional[str] = None

    @property
    def net_qty(self) -> Decimal:
        return self.long_qty - self.short_qty


@dataclass
class CycleStats:
    cycle_id: int = 0
    market: str = ""
    entry_price: Decimal = Decimal("0")
    target_size: Decimal = Decimal("0")
    reprice_count: int = 0
    fees_acc1: Decimal = Decimal("0")
    fees_acc2: Decimal = Decimal("0")
    notes: list[str] = field(default_factory=list)
