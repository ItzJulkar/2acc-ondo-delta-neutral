from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional, Tuple

from bot.client import OndoClient, quantize_down
from bot.config import AppConfig
from bot.models import Balance, BookTop, MarketInfo

logger = logging.getLogger(__name__)


def compute_equal_size(
    cfg: AppConfig,
    market: str,
    info: MarketInfo,
    book: BookTop,
    bal1: Balance,
    bal2: Balance,
    margin_share: float = 1.0,
) -> Tuple[Decimal, Decimal, str]:
    """
    Return (base_size, ref_price, note).
    Size is equal on both accounts, capped by the weaker available margin
    and margin_usage_pct / leverage.

    margin_share: in parallel multi-market mode use 1/N so total ≈ margin_usage_pct.
    """
    price = book.mid if book.mid > 0 else book.mark_price
    if price <= 0:
        return Decimal("0"), price, "no price"

    leverage = min(cfg.leverage, cfg.max_leverage_for(market), info.max_leverage)
    usage = Decimal(str(cfg.sizing.margin_usage_pct)) / Decimal("100")
    share = Decimal(str(max(margin_share, 0.01)))
    usage = usage * share

    # Margin each account may use for this position
    m1 = bal1.available_margin * usage
    m2 = bal2.available_margin * usage
    margin_budget = min(m1, m2)
    if margin_budget <= 0:
        return Decimal("0"), price, "no available margin"

    # notional = margin * leverage
    notional = margin_budget * Decimal(leverage)
    if cfg.sizing.max_notional_usd and cfg.sizing.max_notional_usd > 0:
        notional = min(notional, Decimal(str(cfg.sizing.max_notional_usd)))

    base = notional / price
    base = quantize_down(base, info.base_increment)

    min_sz = Decimal(str(cfg.sizing.min_base_size))
    if base < min_sz:
        return Decimal("0"), price, f"size {base} < min {min_sz}"

    note = (
        f"margin_budget=${margin_budget:.2f} share={float(share):.2f} lev={leverage}x "
        f"notional=${float(base * price):.2f} size={base}"
    )
    return base, price, note


def approx_liq_distance_pct(
    entry: Decimal,
    side_long: bool,
    leverage: int,
    mm_rate: Decimal = Decimal("0.025"),
) -> Decimal:
    """
    Rough isolated-margin style distance to liquidation as % of entry.
    Not exchange-official; used only as a pre-trade filter.
    For long: liq ≈ entry * (1 - 1/lev + mm)
    Distance = |entry - liq| / entry
    """
    if entry <= 0 or leverage <= 0:
        return Decimal("0")
    inv = Decimal(1) / Decimal(leverage)
    if side_long:
        # bankrupt roughly entry*(1-1/lev); liq a bit above that
        dist = inv - mm_rate
    else:
        dist = inv - mm_rate
    return max(dist * Decimal("100"), Decimal("0"))


def check_liq_buffer(
    cfg: AppConfig,
    entry: Decimal,
    leverage: int,
    size: Decimal,
    info: MarketInfo,
    client: Optional[OndoClient] = None,
    market: str = "",
    side_long: bool = True,
) -> Tuple[bool, Decimal, str]:
    """
    Returns (ok, size, message).

    Pre-trade uses a rough isolated-margin formula. At 20x the theoretical
    distance is often ~3–5%, so a 7% target usually cannot be guaranteed by
    size alone — we WARN but still allow the trade (real distance improves
    with unused cross margin). Hard post-fill check uses exchange liq price.
    """
    min_pct = Decimal(str(cfg.risk.min_liq_distance_pct))
    approx = approx_liq_distance_pct(entry, side_long, leverage)
    if approx >= min_pct:
        return True, size, f"approx liq distance {approx:.2f}% >= {min_pct}%"

    # Size shrink does not fix liq distance at fixed leverage.
    msg = (
        f"WARN approx liq distance {approx:.2f}% < target {min_pct}% at {leverage}x "
        f"(isolated-style estimate; unused margin may still protect). "
        f"Post-fill exchange liquidationPrice will be logged."
    )
    # Never hard-block solely on pre-trade approx — user runs 20x by design.
    return True, size, msg


def post_fill_liq_ok(
    cfg: AppConfig,
    client: OndoClient,
    market: str,
    entry: Decimal,
) -> Tuple[bool, str]:
    pos = client.get_position(market)
    if not pos or pos.liquidation_price <= 0 or entry <= 0:
        return True, "no liq price yet"
    dist = abs(entry - pos.liquidation_price) / entry * Decimal("100")
    min_pct = Decimal(str(cfg.risk.min_liq_distance_pct))
    if dist + Decimal("0.01") < min_pct:
        return False, f"exchange liq distance {dist:.2f}% < {min_pct}%"
    return True, f"exchange liq distance {dist:.2f}% ok"
