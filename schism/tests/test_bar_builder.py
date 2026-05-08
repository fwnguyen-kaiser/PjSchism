"""
test_bar_builder.py — Unit tests for bar_builder.py

Covers:
  - CVD sign convention (spec §2.2)
  - build_bar_from_kline field mapping
  - CVD=0 warning when agg_trades absent
  - build_bars_from_klines batch correctness
  - LiveBarBuilder: accumulate → finalise → reset
  - LiveBarBuilder: bar boundary crossing discards stale trades
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from schism.data.ingestion.bar_builder import (
    Bar,
    LiveBarBuilder,
    build_bar_from_kline,
    build_bars_from_klines,
    compute_cvd,
)
from schism.tests.conftest import _bar_ts, make_kline_row


# ── compute_cvd ───────────────────────────────────────────────────────────────

class TestComputeCVD:
    def test_all_taker_buys(self):
        trades = [{"qty": 1.0, "is_buyer_maker": False},
                  {"qty": 2.0, "is_buyer_maker": False}]
        assert compute_cvd(trades) == pytest.approx(3.0)

    def test_all_taker_sells(self):
        trades = [{"qty": 1.0, "is_buyer_maker": True},
                  {"qty": 0.5, "is_buyer_maker": True}]
        assert compute_cvd(trades) == pytest.approx(-1.5)

    def test_mixed_net_positive(self):
        trades = [
            {"qty": 2.0, "is_buyer_maker": False},   # +2
            {"qty": 0.5, "is_buyer_maker": True},    # -0.5
        ]
        assert compute_cvd(trades) == pytest.approx(1.5)

    def test_mixed_net_negative(self):
        trades = [
            {"qty": 1.0, "is_buyer_maker": False},   # +1
            {"qty": 3.0, "is_buyer_maker": True},    # -3
        ]
        assert compute_cvd(trades) == pytest.approx(-2.0)

    def test_empty_trades(self):
        assert compute_cvd([]) == 0.0

    def test_exact_zero_net(self):
        trades = [{"qty": 1.0, "is_buyer_maker": False},
                  {"qty": 1.0, "is_buyer_maker": True}]
        assert compute_cvd(trades) == pytest.approx(0.0)

    def test_qty_string_coercion(self):
        """qty should be coerced to float (Binance sends strings)."""
        trades = [{"qty": "2.5", "is_buyer_maker": False}]
        assert compute_cvd(trades) == pytest.approx(2.5)


# ── build_bar_from_kline ──────────────────────────────────────────────────────

class TestBuildBarFromKline:
    def test_field_mapping(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts, open_=100.0, high=110.0, low=90.0, close=105.0, volume=200.0)
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=[])

        assert bar.bar_ts  == ts
        assert bar.symbol  == "BTCUSDT"
        assert bar.open    == pytest.approx(100.0)
        assert bar.high    == pytest.approx(110.0)
        assert bar.low     == pytest.approx(90.0)
        assert bar.close   == pytest.approx(105.0)
        assert bar.volume  == pytest.approx(200.0)

    def test_cvd_computed_from_agg_trades(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts)
        trades = [{"qty": 3.0, "is_buyer_maker": False},
                  {"qty": 1.0, "is_buyer_maker": True}]
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=trades)
        assert bar.cvd == pytest.approx(2.0)

    def test_bar_delta_proxy_when_no_trades(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts)  # volume=100, taker_buy_base=55 → delta=2*55-100=10
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=None)
        assert bar.cvd == pytest.approx(10.0)

    def test_bar_delta_proxy_no_crash(self, caplog):
        ts = _bar_ts(0)
        kline = make_kline_row(ts, volume=200.0, taker_buy_base=140.0)  # delta=2*140-200=80
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=None)
        assert bar.cvd == pytest.approx(80.0)

    def test_oi_lsr_funding_none_by_default(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts)
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=[])
        assert bar.oi           is None
        assert bar.lsr_top      is None
        assert bar.funding_rate is None

    def test_num_trades_and_taker_buy(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts, num_trades=1234, taker_buy_base=77.0)
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=[])
        assert bar.num_trades     == 1234
        assert bar.taker_buy_base == pytest.approx(77.0)

    def test_naive_bar_ts_gets_utc(self):
        """bar_ts without tzinfo should be treated as UTC."""
        ts_naive = datetime(2024, 1, 1, 0, 0, 0)  # naive
        kline = make_kline_row(ts_naive)
        kline["open_time"] = ts_naive
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=[])
        assert bar.bar_ts.tzinfo == timezone.utc

    def test_to_dict_contains_all_keys(self):
        ts = _bar_ts(0)
        kline = make_kline_row(ts)
        bar = build_bar_from_kline(kline, "BTCUSDT", agg_trades=[])
        d = bar.to_dict()
        for key in ["bar_ts", "symbol", "exchange", "market_type", "timeframe_label",
                    "open", "high", "low", "close",
                    "volume", "cvd", "num_trades", "taker_buy_base",
                    "quote_volume", "oi", "lsr_top", "funding_rate", "source"]:
            assert key in d


# ── build_bars_from_klines ────────────────────────────────────────────────────

class TestBuildBarsFromKlines:
    def test_count_and_order(self):
        klines = [make_kline_row(_bar_ts(i * 4 % 24)) for i in range(6)]
        bars = build_bars_from_klines(klines, "BTCUSDT")
        assert len(bars) == 6
        for i in range(1, len(bars)):
            assert bars[i].bar_ts >= bars[i - 1].bar_ts

    def test_cvd_with_agg_trades_map(self):
        ts0 = _bar_ts(0)
        ts1 = _bar_ts(4)
        klines = [make_kline_row(ts0), make_kline_row(ts1)]
        trades_map = {
            ts0: [{"qty": 5.0, "is_buyer_maker": False}],
            ts1: [{"qty": 2.0, "is_buyer_maker": True}],
        }
        bars = build_bars_from_klines(klines, "BTCUSDT", agg_trades_by_bar=trades_map)
        assert bars[0].cvd == pytest.approx(5.0)
        assert bars[1].cvd == pytest.approx(-2.0)

    def test_empty_klines(self):
        # Should not raise — but logs a warning about empty result
        # build_bars_from_klines will crash on bars[0] if bars is empty
        # so empty klines input should return empty list safely
        # Currently the implementation does bars[0].bar_ts.isoformat() if bars else None
        bars = build_bars_from_klines([], "BTCUSDT")
        assert bars == []


# ── LiveBarBuilder ────────────────────────────────────────────────────────────

class TestLiveBarBuilder:
    def _make_trade(self, ts: datetime, qty: float, is_buyer_maker: bool) -> dict:
        return {"qty": qty, "is_buyer_maker": is_buyer_maker, "timestamp": ts}

    def _make_kline_close(self, open_time: datetime, close: float = 50500.0) -> dict:
        return {
            "open_time":      open_time,
            "open":           close - 500,
            "high":           close + 500,
            "low":            close - 1000,
            "close":          close,
            "volume":         100.0,
            "close_time":     open_time + timedelta(hours=4) - timedelta(milliseconds=1),
            "num_trades":     500,
            "taker_buy_base": 55.0,
            "is_closed":      True,
        }

    def test_on_bar_close_fires_once(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        ts = _bar_ts(0)
        builder.on_agg_trade(self._make_trade(ts, qty=1.0, is_buyer_maker=False))
        builder.on_agg_trade(self._make_trade(ts, qty=0.5, is_buyer_maker=True))
        builder.on_kline_close(self._make_kline_close(ts))

        assert len(received) == 1

    def test_cvd_accumulated_from_trades(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        ts = _bar_ts(0)
        builder.on_agg_trade(self._make_trade(ts, qty=3.0, is_buyer_maker=False))
        builder.on_agg_trade(self._make_trade(ts, qty=1.0, is_buyer_maker=True))
        builder.on_kline_close(self._make_kline_close(ts))

        assert received[0].cvd == pytest.approx(2.0)

    def test_non_close_kline_ignored(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        ts = _bar_ts(0)
        kline = self._make_kline_close(ts)
        kline["is_closed"] = False  # intermediate update
        builder.on_kline_close(kline)

        assert len(received) == 0

    def test_reset_clears_state(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        ts = _bar_ts(0)
        builder.on_agg_trade(self._make_trade(ts, qty=5.0, is_buyer_maker=False))
        builder.reset()

        builder.on_kline_close(self._make_kline_close(ts))
        # Accumulated trades discarded — bar delta proxy used: 2*55 - 100 = 10
        assert received[0].cvd == pytest.approx(10.0)

    def test_bar_boundary_discards_stale_trades(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        ts0 = _bar_ts(0)
        ts1 = _bar_ts(4)

        # Trades for bar0
        builder.on_agg_trade(self._make_trade(ts0, qty=10.0, is_buyer_maker=False))
        # Trade for bar1 → boundary crossed → bar0 trades discarded
        builder.on_agg_trade(self._make_trade(ts1, qty=1.0, is_buyer_maker=False))

        # Close bar1 — only 1.0 of bar1 trades should contribute
        builder.on_kline_close(self._make_kline_close(ts1))
        assert received[0].cvd == pytest.approx(1.0)

    def test_bar_received_is_bar_object(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)
        ts = _bar_ts(0)
        builder.on_kline_close(self._make_kline_close(ts))
        assert isinstance(received[0], Bar)

    def test_multiple_bars_sequential(self):
        received: list[Bar] = []
        builder = LiveBarBuilder("BTCUSDT", on_bar_close=received.append)

        for i in range(3):
            ts = _bar_ts(i * 4)
            builder.on_agg_trade(self._make_trade(ts, qty=float(i + 1), is_buyer_maker=False))
            builder.on_kline_close(self._make_kline_close(ts))

        assert len(received) == 3
        assert received[0].cvd == pytest.approx(1.0)
        assert received[1].cvd == pytest.approx(2.0)
        assert received[2].cvd == pytest.approx(3.0)
