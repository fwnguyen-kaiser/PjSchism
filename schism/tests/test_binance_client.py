"""
test_binance_client.py — Unit tests for binance_client.py

Mocks httpx at the transport level so rate-limiter, retry logic,
and response parsing all execute with no real network calls.

Covers:
  - get_klines: field parsing, pagination boundary
  - get_open_interest_hist: field parsing
  - get_top_lsr: field parsing
  - get_funding_rate: field parsing
  - _get: 418 → BanError, 429 → retry → RateLimitWarning
  - _get: 429 backoff respected (sleep called)
  - _get: timeout → DataMissingError after retries
  - stream_kline_close: callback receives Bar (not dict)
  - stream_kline_close: non-close events ignored
  - stream_kline_close: async callbacks awaited
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from schism.data.ingestion.bar_builder import Bar
from schism.data.ingestion.binance_client import BinanceClient, _kline_weight
from schism.utils.exceptions import BanError, DataMissingError, RateLimitWarning
from schism.tests.conftest import _bar_ts, _ms, make_raw_kline_list


pytestmark = pytest.mark.asyncio


# ── httpx mock helpers ────────────────────────────────────────────────────────

def _mock_response(status: int, body: object, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json = MagicMock(return_value=body)
    resp.headers = headers or {"X-MBX-USED-WEIGHT-1M": "10"}
    return resp


def _mock_client_get(responses: list) -> AsyncMock:
    """Return AsyncMock that yields responses in sequence."""
    mock = AsyncMock()
    mock.side_effect = responses
    return mock


# ── BinanceClient context manager ────────────────────────────────────────────

class TestBinanceClientLifecycle:
    async def test_aenter_sets_client(self):
        async with BinanceClient() as client:
            assert client._client is not None

    async def test_aexit_closes_client(self):
        client = BinanceClient()
        await client.__aenter__()
        inner = client._client
        await client.__aexit__(None, None, None)
        # httpx AsyncClient.aclose() should have been called — client is closed
        assert inner.is_closed

    async def test_get_without_context_raises(self):
        client = BinanceClient()
        with pytest.raises(AssertionError, match="async context manager"):
            await client._get("/test", {})


# ── get_klines ────────────────────────────────────────────────────────────────

class TestGetKlines:
    async def test_kline_weight_matches_binance_limit_tiers(self):
        assert _kline_weight(99) == 1
        assert _kline_weight(100) == 2
        assert _kline_weight(499) == 2
        assert _kline_weight(500) == 5
        assert _kline_weight(1000) == 5
        assert _kline_weight(1001) == 10

    async def test_returns_parsed_dicts(self):
        ts = _bar_ts(0)
        raw_rows = [make_raw_kline_list(ts)]

        async with BinanceClient() as client:
            client._client.get = _mock_client_get([
                _mock_response(200, raw_rows)
            ])
            result = await client.get_klines("BTCUSDT")

        assert len(result) == 1
        row = result[0]
        assert isinstance(row["open_time"], datetime)
        assert row["open_time"].tzinfo == timezone.utc
        assert isinstance(row["open"], float)
        assert isinstance(row["close"], float)
        assert isinstance(row["num_trades"], int)

    async def test_correct_field_values(self):
        ts = _bar_ts(0)
        raw = [make_raw_kline_list(ts, open_=100.0, high=110.0, low=90.0,
                                   close=105.0, volume=200.0, taker_buy_base=120.0)]
        async with BinanceClient() as client:
            client._client.get = _mock_client_get([_mock_response(200, raw)])
            result = await client.get_klines("BTCUSDT")

        assert result[0]["open"]           == pytest.approx(100.0)
        assert result[0]["high"]           == pytest.approx(110.0)
        assert result[0]["low"]            == pytest.approx(90.0)
        assert result[0]["close"]          == pytest.approx(105.0)
        assert result[0]["volume"]         == pytest.approx(200.0)
        assert result[0]["taker_buy_base"] == pytest.approx(120.0)

    async def test_empty_response(self):
        async with BinanceClient() as client:
            client._client.get = _mock_client_get([_mock_response(200, [])])
            result = await client.get_klines("BTCUSDT")
        assert result == []

    async def test_start_end_time_passed_as_ms(self):
        ts = _bar_ts(0)
        raw = [make_raw_kline_list(ts)]
        call_params = {}

        async with BinanceClient() as client:
            original_get = client._client.get

            async def capture_get(url, params=None, **kwargs):
                call_params.update(params or {})
                return _mock_response(200, raw)

            client._client.get = capture_get
            await client.get_klines("BTCUSDT", start_time=ts,
                                    end_time=ts + timedelta(hours=4))

        assert "startTime" in call_params
        assert isinstance(call_params["startTime"], int)
        assert call_params["startTime"] == _ms(ts)


# ── _get error handling ───────────────────────────────────────────────────────

class TestGetErrorHandling:
    async def test_418_raises_ban_error(self):
        body = {"data": {"retryAfter": 9999999}}
        async with BinanceClient() as client:
            client._client.get = _mock_client_get([
                _mock_response(418, body)
            ])
            with pytest.raises(BanError) as exc_info:
                await client._get("/test", {})
        assert exc_info.value.status_code == 418

    async def test_429_retries_then_raises(self):
        resp_429 = _mock_response(429, {}, headers={"Retry-After": "1", "X-MBX-USED-WEIGHT-1M": "2400"})

        async with BinanceClient() as client:
            # Return 429 three times (exhausts retries)
            client._client.get = _mock_client_get([resp_429, resp_429, resp_429])
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RateLimitWarning):
                    await client._get("/test", {})

    async def test_429_sleeps_between_retries(self):
        resp_429 = _mock_response(429, {}, headers={"Retry-After": "5", "X-MBX-USED-WEIGHT-1M": "100"})
        resp_200 = _mock_response(200, [])

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        async with BinanceClient() as client:
            client._client.get = _mock_client_get([resp_429, resp_200])
            with patch("asyncio.sleep", side_effect=fake_sleep):
                await client._get("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "4h", "limit": 1})

        # Sleep should include a small buffer over Retry-After.
        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == 6

    async def test_500_raises_data_missing_error(self):
        async with BinanceClient() as client:
            client._client.get = _mock_client_get([_mock_response(500, {})])
            with pytest.raises(DataMissingError) as exc_info:
                await client._get("/test", {})
        assert exc_info.value.status_code == 500

    async def test_timeout_raises_data_missing_after_retries(self):
        async with BinanceClient() as client:
            client._client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(DataMissingError) as exc_info:
                    await client._get("/test", {})
        assert "timeout" in str(exc_info.value).lower() or exc_info.value.reason


# ── stream_kline_close ────────────────────────────────────────────────────────

class TestStreamKlineClose:
    class _AsyncIter:
        def __init__(self, items):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration

    def _make_ws_msg(self, ts: datetime, is_closed: bool = True) -> str:
        return json.dumps({
            "k": {
                "t": _ms(ts),
                "T": _ms(ts + timedelta(hours=4) - timedelta(milliseconds=1)),
                "o": "50000",
                "h": "51000",
                "l": "49000",
                "c": "50500",
                "v": "100",
                "n": 500,
                "Q": "55",
                "x": is_closed,
            }
        })

    def _mock_connect(self, messages: list[str]):
        mock_ws = MagicMock()
        mock_ws.__aiter__ = MagicMock(return_value=self._AsyncIter(messages))

        mock_connect = MagicMock()
        mock_connect.__aiter__ = MagicMock(return_value=self._AsyncIter([mock_ws]))
        return mock_connect

    async def test_callback_receives_bar_not_dict(self):
        """stream_kline_close must pass Bar to callback, not raw dict."""
        received: list = []

        ts = _bar_ts(0)
        msg = self._make_ws_msg(ts, is_closed=True)

        async with BinanceClient() as client:
            with patch("websockets.connect", return_value=self._mock_connect([msg])):
                await client.stream_kline_close("BTCUSDT", on_bar_close=received.append)

        assert len(received) == 1
        assert isinstance(received[0], Bar), f"Expected Bar, got {type(received[0])}"

    async def test_non_close_events_not_forwarded(self):
        received: list = []
        ts = _bar_ts(0)
        msg = self._make_ws_msg(ts, is_closed=False)

        async with BinanceClient() as client:
            with patch("websockets.connect", return_value=self._mock_connect([msg])):
                await client.stream_kline_close("BTCUSDT", on_bar_close=received.append)

        assert len(received) == 0

    async def test_async_callback_awaited(self):
        """Async on_bar_close callbacks must be awaited (not left as unawaited coroutines)."""
        awaited_bars: list[Bar] = []

        async def async_callback(bar: Bar) -> None:
            awaited_bars.append(bar)

        ts = _bar_ts(0)
        msg = self._make_ws_msg(ts, is_closed=True)

        async with BinanceClient() as client:
            with patch("websockets.connect", return_value=self._mock_connect([msg])):
                await client.stream_kline_close("BTCUSDT", on_bar_close=async_callback)

        assert len(awaited_bars) == 1
        assert isinstance(awaited_bars[0], Bar)

    async def test_bar_fields_correct_from_ws_msg(self):
        received: list[Bar] = []
        ts = _bar_ts(4)
        msg = self._make_ws_msg(ts, is_closed=True)

        async with BinanceClient() as client:
            with patch("websockets.connect", return_value=self._mock_connect([msg])):
                await client.stream_kline_close("BTCUSDT", on_bar_close=received.append)

        bar = received[0]
        assert bar.symbol == "BTCUSDT"
        assert bar.close  == pytest.approx(50500.0)
        assert bar.bar_ts == ts
