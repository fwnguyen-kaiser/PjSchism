"""
Binance USDM Futures client.

Responsibilities:
  - REST historical klines, OI hist, LSR, funding rate, aggTrades, bookTicker
  - WebSocket live @kline_4h, @bookTicker, @aggTrade streams
  - Rate limit tracking: X-MBX-USED-WEIGHT-1M header parsed on every response
  - Automatic back-off on 429; hard stop + BanError on 418

Rate limits (as of V1.4, from /fapi/v1/exchangeInfo):
  REQUEST_WEIGHT: 2400 / min per IP
  Safety margin: stop new requests if used_weight > 2000

Disclosure: REST used only for ≤30-day backfill and per-bar snapshots.
Live data uses WebSocket streams per Binance recommendation.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections import deque
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional

import httpx

from schism.data.ingestion.bar_builder import Bar, IngestionSource, build_bar_from_kline
from schism.utils.date_helpers import datetime_to_ms, ms_to_datetime, normalize_ts
from schism.utils.exceptions import BanError, DataMissingError, RateLimitWarning
from schism.utils.logger import ingestion_logger

# ── Constants ─────────────────────────────────────────────────────────────────

_FAPI_BASE = "https://fapi.binance.com"
_FAPI_DATA = "https://www.binance.com/futures/data"
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_WEIGHT_LIMIT = 2400
_WEIGHT_SAFETY = 2000   # stop new requests above this
_FUNDING_WINDOW_LIMIT = 450      # Binance docs: 500 requests / 5 min / IP
_FUTURES_DATA_WINDOW_LIMIT = 900 # Binance docs: 1000 requests / 5 min / IP

# Endpoint weights and route-specific windows. Keep local limits below Binance
# published caps so a single process does not run at the edge of an IP ban.
_WEIGHTS = {
    "openInterestHist":    0,
    "topLongShortAccountRatio": 0,
    "fundingRate":         0,
    "aggTrades":          20,
    "bookTicker":          2,
}


def _kline_weight(limit: int) -> int:
    """Binance kline request weight is based on the LIMIT parameter."""
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class _RateLimiter:
    """
    Tracks used weight from X-MBX-USED-WEIGHT-1M response header.
    Enforces a rolling budget of WEIGHT_SAFETY per minute.
    """

    def __init__(self, limit: int = _WEIGHT_LIMIT, safety: int = _WEIGHT_SAFETY):
        self._limit = limit
        self._safety = safety
        self._used: int = 0
        self._window_start: float = time.monotonic()

    def update(self, used_weight: int) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._used = 0
            self._window_start = now
        self._used = used_weight

    @property
    def used(self) -> int:
        return self._used

    async def acquire(self, cost: int = 1) -> None:
        """Wait until budget allows, then proceed."""
        while True:
            now = time.monotonic()
            elapsed = now - self._window_start
            if elapsed >= 60.0:
                self._used = 0
                self._window_start = now

            if self._used + cost <= self._safety:
                self._used += cost
                return

            sleep_s = 60.0 - elapsed + 0.1
            ingestion_logger.warning(
                "rate_limit_sleep",
                source="binance",
                used_weight=self._used,
                safety=self._safety,
                sleep_s=round(sleep_s, 2),
            )
            await asyncio.sleep(sleep_s)


class _WindowLimiter:
    """Simple sliding-window limiter for endpoint families with IP caps."""

    def __init__(self, name: str, limit: int, interval_s: float) -> None:
        self._name = name
        self._limit = limit
        self._interval_s = interval_s
        self._events: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self._events and now - self._events[0] >= self._interval_s:
                self._events.popleft()

            if len(self._events) < self._limit:
                self._events.append(now)
                return

            sleep_s = self._interval_s - (now - self._events[0]) + 0.1
            ingestion_logger.warning(
                "endpoint_window_rate_limit_sleep",
                limiter=self._name,
                limit=self._limit,
                interval_s=self._interval_s,
                sleep_s=round(sleep_s, 2),
            )
            await asyncio.sleep(sleep_s)

    @property
    def used(self) -> int:
        now = time.monotonic()
        while self._events and now - self._events[0] >= self._interval_s:
            self._events.popleft()
        return len(self._events)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def interval_s(self) -> float:
        return self._interval_s


class BinanceClient:
    """
    Async Binance USDM Futures client.

    Usage:
        async with BinanceClient() as client:
            klines = await client.get_klines("BTCUSDT", "4h", limit=500)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._timeout = timeout
        self._limiter = _RateLimiter()
        self._funding_limiter = _WindowLimiter(
            "funding_rate", _FUNDING_WINDOW_LIMIT, 300.0
        )
        self._futures_data_limiter = _WindowLimiter(
            "futures_data", _FUTURES_DATA_WINDOW_LIMIT, 300.0
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BinanceClient":
        headers = {}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key
        self._client = httpx.AsyncClient(
            base_url=_FAPI_BASE,
            headers=headers,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def _acquire_route_limits(self, path: str, weight: int) -> None:
        """Acquire all rate-limit buckets that apply to one outgoing request."""
        await self._limiter.acquire(weight)
        route_bucket = None
        route_used = None
        route_limit = None
        route_interval_s = None
        if path == "/fapi/v1/fundingRate":
            await self._funding_limiter.acquire()
            route_bucket = "funding_rate"
            route_used = self._funding_limiter.used
            route_limit = self._funding_limiter.limit
            route_interval_s = self._funding_limiter.interval_s
        elif path.startswith("/futures/data/"):
            await self._futures_data_limiter.acquire()
            route_bucket = "futures_data"
            route_used = self._futures_data_limiter.used
            route_limit = self._futures_data_limiter.limit
            route_interval_s = self._futures_data_limiter.interval_s

        ingestion_logger.debug(
            "rate_limit_acquired",
            path=path,
            request_weight=weight,
            used_weight_1m=self._limiter.used,
            weight_safety=_WEIGHT_SAFETY,
            route_bucket=route_bucket,
            route_used=route_used,
            route_limit=route_limit,
            route_interval_s=route_interval_s,
        )

    # ── Internal request wrapper ──────────────────────────────────────────────

    async def _get(
        self,
        path: str,
        params: dict,
        weight: int = 1,
        base_url: Optional[str] = None,
    ) -> dict | list:
        """
        Execute GET with rate limit tracking and retry logic.

        Raises:
            BanError:        HTTP 418 — IP banned.
            RateLimitWarning: HTTP 429 — rate exceeded, back off applied.
            DataMissingError: Non-retryable HTTP error or timeout.
        """
        assert self._client is not None, "BinanceClient must be used as an async context manager"

        url = f"{base_url or _FAPI_BASE}{path}"
        for attempt in range(1, 4):  # max 3 retries
            await self._acquire_route_limits(path, weight)
            try:
                resp = await self._client.get(url, params=params)

                # Parse used weight from header
                used_str = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                if used_str:
                    self._limiter.update(int(used_str))

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 418:
                    body = resp.json()
                    retry_after = body.get("data", {}).get("retryAfter", 0)
                    ingestion_logger.critical(
                        "binance_ip_ban",
                        status=418,
                        retry_after_ts=retry_after,
                        path=path,
                    )
                    raise BanError(
                        "Binance IP ban (HTTP 418)",
                        exchange="binance",
                        retry_after_ts=retry_after,
                        status_code=418,
                    )

                if resp.status_code == 429:
                    retry_after_s = int(resp.headers.get("Retry-After", 60)) + 1
                    ingestion_logger.warning(
                        "binance_rate_limit_429",
                        attempt=attempt,
                        retry_after_s=retry_after_s,
                        path=path,
                    )
                    if attempt < 3:
                        await asyncio.sleep(retry_after_s)
                        continue
                    raise RateLimitWarning(
                        "Binance 429 exceeded max retries",
                        exchange="binance",
                        retry_after_seconds=retry_after_s,
                        used_weight=self._limiter._used,
                    )

                raise DataMissingError(
                    f"Binance HTTP {resp.status_code} on {path}",
                    source="binance_rest",
                    path=path,
                    status_code=resp.status_code,
                )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                ingestion_logger.warning(
                    "binance_request_error",
                    attempt=attempt,
                    error=str(exc),
                    path=path,
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise DataMissingError(
                    f"Binance request failed after {attempt} attempts: {exc}",
                    source="binance_rest",
                    path=path,
                    reason=str(exc),
                ) from exc

        raise DataMissingError(
            f"Binance request exhausted retries for {path}",
            source="binance_rest",
            path=path,
        )

    # ── Public REST methods ───────────────────────────────────────────────────

    async def get_klines(
        self,
        symbol: str,
        interval: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Fetch USDM futures klines.

        Returns list of dicts with keys:
          open_time, open, high, low, close, volume, close_time,
          quote_volume, num_trades, taker_buy_base, taker_buy_quote
        """
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = datetime_to_ms(start_time)
        if end_time:
            params["endTime"] = datetime_to_ms(end_time)

        raw = await self._get("/fapi/v1/klines", params, weight=_kline_weight(limit))

        result = []
        for row in raw:
            result.append({
                "open_time":       ms_to_datetime(int(row[0])),
                "open":            float(row[1]),
                "high":            float(row[2]),
                "low":             float(row[3]),
                "close":           float(row[4]),
                "volume":          float(row[5]),
                "close_time":      ms_to_datetime(int(row[6])),
                "quote_volume":    float(row[7]),
                "num_trades":      int(row[8]),
                "taker_buy_base":  float(row[9]),
                "taker_buy_quote": float(row[10]),
            })

        ingestion_logger.debug(
            "klines_fetched",
            symbol=symbol,
            interval=interval,
            count=len(result),
            start=result[0]["open_time"].isoformat() if result else None,
            end=result[-1]["close_time"].isoformat() if result else None,
        )
        return result

    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Native 4h Open Interest history (≤30 days).
        Returns list of dicts: {bar_ts, sum_open_interest, sum_oi_value}
        """
        params: dict = {"symbol": symbol, "period": period, "limit": limit}
        if start_time:
            params["startTime"] = datetime_to_ms(start_time)
        if end_time:
            params["endTime"] = datetime_to_ms(end_time)

        raw = await self._get(
            "/futures/data/openInterestHist",
            params,
            weight=_WEIGHTS["openInterestHist"],
            base_url="https://www.binance.com",
        )

        result = []
        for row in raw:
            result.append({
                "bar_ts":          ms_to_datetime(int(row["timestamp"])),
                "sum_open_interest": float(row["sumOpenInterest"]),
                "sum_oi_value":    float(row["sumOpenInterestValue"]),
            })

        ingestion_logger.debug(
            "oi_hist_fetched",
            symbol=symbol,
            period=period,
            count=len(result),
        )
        return result

    async def get_top_lsr(
        self,
        symbol: str,
        period: str = "4h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Top trader long/short account ratio history.
        Returns list of dicts: {bar_ts, long_short_ratio, long_account, short_account}
        """
        params: dict = {"symbol": symbol, "period": period, "limit": limit}
        if start_time:
            params["startTime"] = datetime_to_ms(start_time)
        if end_time:
            params["endTime"] = datetime_to_ms(end_time)

        raw = await self._get(
            "/futures/data/topLongShortAccountRatio",
            params,
            weight=_WEIGHTS["topLongShortAccountRatio"],
            base_url="https://www.binance.com",
        )

        result = []
        for row in raw:
            result.append({
                "bar_ts":           ms_to_datetime(int(row["timestamp"])),
                "long_short_ratio": float(row["longShortRatio"]),
                "long_account":     float(row["longAccount"]),
                "short_account":    float(row["shortAccount"]),
            })
        return result

    async def get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Single-page funding rate fetch (max 1000 records).
        Returns list of dicts: {funding_time, funding_rate}, oldest first.
        """
        from datetime import timedelta as _timedelta
        params: dict = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = datetime_to_ms(start_time)
        if end_time:
            params["endTime"] = datetime_to_ms(end_time)

        raw = await self._get(
            "/fapi/v1/fundingRate",
            params,
            weight=_WEIGHTS["fundingRate"],
        )

        return [
            {
                "funding_time": ms_to_datetime(int(row["fundingTime"])),
                "funding_rate": float(row["fundingRate"]),
            }
            for row in raw
        ]

    async def get_funding_rate_all(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Paginated funding rate — fetches every record in [start_time, end_time].

        Binance returns records oldest-first per page; each page advances
        start_time to last_record + 1 ms until fewer than 1000 records returned.
        """
        from datetime import timedelta as _timedelta

        all_records: list[dict] = []
        page_start = start_time
        while True:
            batch = await self.get_funding_rate(
                symbol, start_time=page_start, end_time=end_time, limit=1000
            )
            if not batch:
                break
            all_records.extend(batch)
            if len(batch) < 1000:
                break
            page_start = batch[-1]["funding_time"] + _timedelta(milliseconds=1)

        ingestion_logger.info(
            "binance_funding_rate_paginated",
            symbol=symbol,
            total=len(all_records),
            start=all_records[0]["funding_time"].isoformat() if all_records else None,
            end=all_records[-1]["funding_time"].isoformat() if all_records else None,
        )
        return all_records

    async def get_book_ticker_snapshot(self, symbol: str) -> dict:
        """
        Fetch current best bid/ask snapshot (for live bar close).
        Returns: {symbol, bid_price, bid_qty, ask_price, ask_qty, ts}
        """
        raw = await self._get(
            "/fapi/v1/ticker/bookTicker",
            {"symbol": symbol},
            weight=_WEIGHTS["bookTicker"],
        )
        return {
            "symbol":    raw["symbol"],
            "bid_price": float(raw["bidPrice"]),
            "bid_qty":   float(raw["bidQty"]),
            "ask_price": float(raw["askPrice"]),
            "ask_qty":   float(raw["askQty"]),
            "ts":        ms_to_datetime(int(raw.get("time", time.time() * 1000))),
        }

    async def get_agg_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Aggregate trades for CVD computation.
        Returns: {agg_trade_id, price, qty, is_buyer_maker, timestamp}
        """
        params: dict = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = datetime_to_ms(start_time)
        if end_time:
            params["endTime"] = datetime_to_ms(end_time)

        raw = await self._get(
            "/fapi/v1/aggTrades",
            params,
            weight=_WEIGHTS["aggTrades"],
        )
        return [
            {
                "agg_trade_id":   int(row["a"]),
                "price":          float(row["p"]),
                "qty":            float(row["q"]),
                "is_buyer_maker": bool(row["m"]),
                "timestamp":      ms_to_datetime(int(row["T"])),
            }
            for row in raw
        ]

    # ── WebSocket live feed ───────────────────────────────────────────────────

    async def stream_kline_close(
        self,
        symbol: str,
        on_bar_close: Callable[[Bar], None],
        interval: str = "4h",
    ) -> None:
        """
        Subscribe to kline stream. Calls on_bar_close(Bar) on each
        bar close event. Runs indefinitely; cancel the task to stop.
        """
        import websockets  # lazy import — not needed for historical mode

        stream = f"{symbol.lower()}@kline_{interval}"
        url = f"wss://fstream.binance.com/ws/{stream}"

        ingestion_logger.info("ws_kline_start", symbol=symbol, interval=interval, url=url)

        async for ws in websockets.connect(url, ping_interval=20, ping_timeout=30):
            try:
                async for raw_msg in ws:
                    import json
                    msg = json.loads(raw_msg)
                    k = msg.get("k", {})
                    if bool(k["x"]):
                        kline_dict = {
                            "open_time":      ms_to_datetime(int(k["t"])),
                            "open":           k["o"],
                            "high":           k["h"],
                            "low":            k["l"],
                            "close":          k["c"],
                            "volume":         k["v"],
                            "num_trades":     k["n"],
                            "taker_buy_base": k["V"],  # V=taker buy base, Q=taker buy quote
                            "quote_volume":   k["q"],
                        }
                        bar = build_bar_from_kline(kline_dict, symbol)
                        bar.source = IngestionSource.BINANCE_WS
                        result = on_bar_close(bar)
                        if inspect.isawaitable(result):
                            await result
            except Exception as exc:
                ingestion_logger.warning(
                    "ws_kline_reconnect", symbol=symbol, error=str(exc)
                )
                await asyncio.sleep(2)
                continue
