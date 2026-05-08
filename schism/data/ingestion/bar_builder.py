"""
bar_builder.py — Assemble 4h OHLCV bars with CVD from raw data.

Two entry points:

1. build_bar_from_kline(kline_dict, agg_trades)
   — For REST historical backfill: merge a klines row with pre-fetched
     aggTrades to compute CVD for that bar.

2. LiveBarBuilder
   — For WebSocket live mode: accumulates aggTrade events within a bar
     window, then finalises when the kline close event fires.

CVD Definition (spec §2.2):
  ΔCVDt = CVDt - CVDt-1
  = Σ qty_i * (+1 if buyer_maker=False, -1 if buyer_maker=True)
  over all aggTrades within bar [open_time, close_time]

  i.e., a buy-side aggressive trade (taker is buyer → buyer_maker=False)
  contributes positive CVD, and vice versa.

Output bar schema (matches ohlcv_bars hypertable):
  bar_ts, symbol, open, high, low, close, volume,
  oi, lsr_top, funding_rate, cvd

oi, lsr_top, funding_rate are optional fields attached separately
by data_store after joining metric streams.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from schism.utils.date_helpers import datetime_to_bar_ts, ms_to_datetime
from schism.utils.logger import ingestion_logger


# ── Bar dataclass ─────────────────────────────────────────────────────────────

@dataclass
class IngestionSource(str, Enum):
    BINANCE_REST = "binance_rest"
    BINANCE_WS = "binance_ws"
    VISION_CRAWLER = "vision_crawler"


@dataclass
class Bar:
    """
    One 4-hour OHLCV bar with CVD.

    Fields:
        bar_ts      — open time of the bar (UTC-aware)
        symbol      — e.g. "BTCUSDT"
        open        — bar open price
        high        — bar high price
        low         — bar low price
        close       — bar close price
        volume      — total traded volume (base asset)
        cvd         — cumulative volume delta for this bar (ΔCVD_t)
        num_trades  — number of aggTrades within the bar
        taker_buy_base  — taker buy volume (base asset), from kline
        quote_volume    — total quote volume, from kline
        oi          — open interest at bar close (filled externally)
        lsr_top     — top-trader L/S ratio (filled externally)
        funding_rate — funding rate at bar open (filled externally)
    """
    bar_ts:         datetime
    symbol:         str
    open:           float
    high:           float
    low:            float
    close:          float
    volume:         float
    exchange:       str = "binance"
    market_type:    str = "perp"
    timeframe_label: str = "4h"
    cvd:            float = 0.0
    num_trades:     int   = 0
    taker_buy_base: float = 0.0
    quote_volume:   float = 0.0
    oi:             Optional[float] = None
    lsr_top:        Optional[float] = None
    funding_rate:   Optional[float] = None
    source:         Optional[IngestionSource] = None

    def to_dict(self) -> dict:
        return {
            "bar_ts":          self.bar_ts,
            "symbol":          self.symbol,
            "exchange":        self.exchange,
            "market_type":     self.market_type,
            "timeframe_label": self.timeframe_label,
            "open":            self.open,
            "high":            self.high,
            "low":             self.low,
            "close":           self.close,
            "volume":          self.volume,
            "cvd":             self.cvd,
            "num_trades":      self.num_trades,
            "taker_buy_base":  self.taker_buy_base,
            "quote_volume":    self.quote_volume,
            "oi":              self.oi,
            "lsr_top":         self.lsr_top,
            "funding_rate":    self.funding_rate,
            "source":          self.source.value if self.source else None,
        }


# ── CVD computation ───────────────────────────────────────────────────────────

def compute_cvd(agg_trades: list[dict]) -> float:
    """
    Compute the cumulative volume delta (ΔCVD) for a list of aggTrades.

    Per spec §2.2:
      buyer_maker=False → taker is buyer  → +qty  (aggressive buy)
      buyer_maker=True  → taker is seller → -qty  (aggressive sell)

    Args:
        agg_trades: list of dicts from BinanceClient.get_agg_trades(),
                    each with keys: qty (float), is_buyer_maker (bool)

    Returns:
        Net CVD for the bar (float). Positive = net aggressive buying.
    """
    cvd = 0.0
    for trade in agg_trades:
        qty = float(trade["qty"])
        if trade["is_buyer_maker"]:
            cvd -= qty   # taker sell
        else:
            cvd += qty   # taker buy
    return cvd


# ── REST historical bar assembly ──────────────────────────────────────────────

def build_bar_from_kline(
    kline: dict,
    symbol: str,
    agg_trades: list[dict] | None = None,
) -> Bar:
    """
    Construct a Bar from a kline dict (from BinanceClient.get_klines()).

    Args:
        kline:      Dict with keys: open_time, open, high, low, close,
                    volume, close_time, quote_volume, num_trades,
                    taker_buy_base, taker_buy_quote.
        symbol:     Trading pair symbol, e.g. "BTCUSDT".
        agg_trades: Optional list of aggTrade dicts covering the bar's
                    time window. If None or empty, cvd=0 is recorded
                    and a warning is logged.

    Returns:
        Bar with CVD populated (or 0.0 if agg_trades unavailable).
    """
    bar_ts: datetime = kline["open_time"]
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.replace(tzinfo=timezone.utc)

    cvd = 0.0
    if agg_trades:
        cvd = compute_cvd(agg_trades)
    else:
        # Bar delta proxy: 2×taker_buy - volume = taker_buy - taker_sell
        # Used for REST backfill and live reconnects where agg_trades are unavailable.
        taker_buy = float(kline.get("taker_buy_base", 0.0))
        vol = float(kline.get("volume", 0.0))
        cvd = 2.0 * taker_buy - vol
        ingestion_logger.debug(
            "cvd_bar_delta_proxy",
            symbol=symbol,
            bar_ts=bar_ts.isoformat(),
            bar_delta=round(cvd, 4),
        )

    bar = Bar(
        bar_ts         = bar_ts,
        symbol         = symbol,
        open           = float(kline["open"]),
        high           = float(kline["high"]),
        low            = float(kline["low"]),
        close          = float(kline["close"]),
        volume         = float(kline["volume"]),
        cvd            = cvd,
        num_trades     = int(kline.get("num_trades", 0)),
        taker_buy_base = float(kline.get("taker_buy_base", 0.0)),
        quote_volume   = float(kline.get("quote_volume", 0.0)),
    )

    ingestion_logger.debug(
        "bar_built",
        symbol=symbol,
        bar_ts=bar_ts.isoformat(),
        cvd=round(cvd, 4),
        volume=bar.volume,
        close=bar.close,
    )

    return bar


def build_bars_from_klines(
    klines: list[dict],
    symbol: str,
    agg_trades_by_bar: dict[datetime, list[dict]] | None = None,
) -> list[Bar]:
    """
    Batch-build Bars from a list of kline dicts.

    Args:
        klines:             List from BinanceClient.get_klines().
        symbol:             Trading pair symbol.
        agg_trades_by_bar:  Optional map of open_time → list[aggTrade dict].
                            If provided, CVD is computed per bar.

    Returns:
        List of Bar objects in chronological order.
    """
    bars: list[Bar] = []
    for kline in klines:
        bar_ts: datetime = kline["open_time"]
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.replace(tzinfo=timezone.utc)

        trades: list[dict] = []
        if agg_trades_by_bar:
            trades = agg_trades_by_bar.get(bar_ts, [])

        bars.append(build_bar_from_kline(kline, symbol, trades))

    ingestion_logger.info(
        "bars_built_batch",
        symbol=symbol,
        count=len(bars),
        first_bar=bars[0].bar_ts.isoformat() if bars else None,
        last_bar=bars[-1].bar_ts.isoformat() if bars else None,
    )
    return bars


# ── Live bar builder ──────────────────────────────────────────────────────────

class LiveBarBuilder:
    """
    Accumulate aggTrade events within a 4h bar window and finalise
    when the kline close event fires from the WebSocket.

    Usage:
        builder = LiveBarBuilder(symbol="BTCUSDT", on_bar_close=handle_bar)
        # Feed aggTrade events as they arrive:
        builder.on_agg_trade(trade_dict)
        # Feed kline close events:
        builder.on_kline_close(kline_dict)

    The on_bar_close callback receives a completed Bar.
    """

    def __init__(
        self,
        symbol: str,
        on_bar_close: Callable[[Bar], None],
        freq: str = "4h",
    ) -> None:
        self.symbol = symbol
        self.freq = freq
        self._on_bar_close = on_bar_close
        self._pending_trades: list[dict] = []
        self._current_bar_ts: Optional[datetime] = None

    def on_agg_trade(self, trade: dict) -> None:
        """
        Receive one aggTrade event from the WebSocket stream.

        Args:
            trade: dict with keys: qty (float), is_buyer_maker (bool),
                   timestamp (datetime)
        """
        ts: datetime = trade["timestamp"]
        bar_ts = datetime_to_bar_ts(ts, self.freq)

        # If bar boundary crossed, discard stale trades from previous bar
        # (they should have been consumed by on_kline_close already).
        if self._current_bar_ts is None:
            self._current_bar_ts = bar_ts
        elif bar_ts != self._current_bar_ts:
            ingestion_logger.debug(
                "live_bar_boundary",
                symbol=self.symbol,
                prev_bar_ts=self._current_bar_ts.isoformat(),
                new_bar_ts=bar_ts.isoformat(),
                stale_trade_count=len(self._pending_trades),
            )
            self._pending_trades = []
            self._current_bar_ts = bar_ts

        self._pending_trades.append(trade)

    def on_kline_close(self, kline: dict) -> None:
        """
        Receive a kline close event from the WebSocket stream.
        Finalises the bar and invokes the on_bar_close callback.

        Args:
            kline: dict from BinanceClient.stream_kline_close() with
                   keys: open_time, open, high, low, close, volume,
                   close_time, num_trades, taker_buy_base, is_closed.
        """
        if not kline.get("is_closed", False):
            return  # intermediate update, not a close event

        bar = build_bar_from_kline(
            kline      = kline,
            symbol     = self.symbol,
            agg_trades = self._pending_trades,
        )

        ingestion_logger.info(
            "live_bar_close",
            symbol=self.symbol,
            bar_ts=bar.bar_ts.isoformat(),
            cvd=round(bar.cvd, 4),
            volume=bar.volume,
            close=bar.close,
            trade_count=len(self._pending_trades),
        )

        # Reset for next bar
        self._pending_trades = []
        self._current_bar_ts = None

        self._on_bar_close(bar)

    def reset(self) -> None:
        """Discard any accumulated state (e.g. after reconnect)."""
        self._pending_trades = []
        self._current_bar_ts = None
        ingestion_logger.debug("live_bar_builder_reset", symbol=self.symbol)