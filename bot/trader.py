"""Order management for live and paper trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .config import BinanceCredentials, CONFIG

LOGGER = logging.getLogger(__name__)


@dataclass
class Position:
    side: Optional[str] = None
    entry_price: Optional[float] = None
    quantity: float = 0.0

    def is_open(self) -> bool:
        return self.side is not None and self.quantity > 0


class Trader:
    """Manage orders with Binance or in paper mode."""

    def __init__(self, live: bool = False) -> None:
        self.live = live
        self.position = Position()
        self.trade_size = CONFIG.trade_size
        self.client: Optional[Client] = None

        if live:
            credentials = BinanceCredentials.from_env()
            self.client = Client(credentials.api_key, credentials.api_secret)

    def _place_market_order(self, side: str) -> None:
        if not self.client:
            raise RuntimeError("Binance client is not initialized")
        try:
            LOGGER.info("Placing %s order", side)
            self.client.create_order(
                symbol=CONFIG.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quoteOrderQty=str(self.trade_size),
            )
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise RuntimeError("Failed to place market order") from exc

    def _apply_fees(self, amount: float) -> float:
        return amount * (1 - CONFIG.fee_pct)

    def open_position(self, price: float) -> None:
        if self.position.is_open():
            LOGGER.debug("Position already open, skipping buy")
            return

        qty = self.trade_size / price
        qty = round(qty, 6)

        if self.live:
            self._place_market_order(SIDE_BUY)

        self.position = Position(side=SIDE_BUY, entry_price=price, quantity=qty)
        LOGGER.info("Opened position at %s with qty %s", price, qty)

    def close_position(self, price: float) -> float:
        if not self.position.is_open():
            LOGGER.debug("No open position, skipping sell")
            return 0.0

        if self.live:
            self._place_market_order(SIDE_SELL)

        pnl = (price - (self.position.entry_price or price)) * self.position.quantity
        pnl = self._apply_fees(pnl)
        LOGGER.info("Closed position at %s | PnL: %.2f", price, pnl)
        self.position = Position()
        return pnl

    def evaluate_exit(self, price: float) -> tuple[bool, float]:
        if not self.position.is_open() or not self.position.entry_price:
            return False, 0.0

        change_pct = (price - self.position.entry_price) / self.position.entry_price
        if change_pct <= -CONFIG.stop_loss_pct:
            LOGGER.info("Stop-loss triggered")
            pnl = self.close_position(price)
            return True, pnl
        if change_pct >= CONFIG.take_profit_pct:
            LOGGER.info("Take-profit triggered")
            pnl = self.close_position(price)
            return True, pnl
        return False, 0.0
