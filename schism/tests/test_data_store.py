"""
test_data_store.py — Unit tests for data_store.py

Covers:
  - write_bars + read_bars round-trip
  - upsert: new bar overwrites same bar_ts
  - parquet partition scheme (symbol/year/month)
  - DataMissingError when symbol dir absent
  - read_bars time-range filtering
  - write_vision_metrics + read_metrics round-trip
  - merge_ohlcv_metrics inner join on bar_ts
  - merge_ohlcv_metrics with missing metrics → NaN columns, no crash
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import pytest_asyncio

from schism.data.ingestion.data_store import DataStore
from schism.utils.exceptions import DataMissingError
from schism.tests.conftest import _bar_ts, make_bar, make_bars


pytestmark = pytest.mark.asyncio


class TestWriteReadBars:
    async def test_round_trip_single_bar(self, store: DataStore):
        bar = make_bar(_bar_ts(0))
        await store.write_bars([bar])

        start = _bar_ts(0)
        end   = _bar_ts(0) + timedelta(hours=4)
        df = await store.read_bars("BTCUSDT", start, end)

        assert len(df) == 1
        assert df.iloc[0]["close"] == pytest.approx(bar.close)
        assert df.iloc[0]["cvd"]   == pytest.approx(bar.cvd)

    async def test_round_trip_multiple_bars(self, store: DataStore):
        bars = make_bars(6)
        await store.write_bars(bars)

        start = bars[0].bar_ts
        end   = bars[-1].bar_ts + timedelta(hours=4)
        df = await store.read_bars("BTCUSDT", start, end)
        assert len(df) == 6

    async def test_bars_sorted_by_bar_ts(self, store: DataStore):
        bars = make_bars(5)
        # Write in reverse order
        await store.write_bars(list(reversed(bars)))
        start = bars[0].bar_ts
        end   = bars[-1].bar_ts + timedelta(hours=4)
        df = await store.read_bars("BTCUSDT", start, end)
        timestamps = list(df["bar_ts"])
        assert timestamps == sorted(timestamps)

    async def test_upsert_overwrites_same_bar_ts(self, store: DataStore):
        bar = make_bar(_bar_ts(0), close=50000.0)
        await store.write_bars([bar])

        updated = make_bar(_bar_ts(0), close=99999.0)
        await store.write_bars([updated])

        start = _bar_ts(0)
        end   = _bar_ts(0) + timedelta(hours=4)
        df = await store.read_bars("BTCUSDT", start, end)
        assert len(df) == 1
        assert df.iloc[0]["close"] == pytest.approx(99999.0)

    async def test_partition_file_created(self, store: DataStore, tmp_parquet: Path):
        bar = make_bar(_bar_ts(0))  # 2024-01-01
        await store.write_bars([bar])
        expected = tmp_parquet / "symbol=BTCUSDT" / "year=2024" / "month=01" / "BTCUSDT_202401.parquet"
        assert expected.exists()

    async def test_read_range_filtering(self, store: DataStore):
        bars = make_bars(12)  # 12 bars = 48 hours
        await store.write_bars(bars)

        # Read only bars 4..7
        start = bars[4].bar_ts
        end   = bars[8].bar_ts
        df = await store.read_bars("BTCUSDT", start, end)
        assert len(df) == 4
        assert pd.Timestamp(df.iloc[0]["bar_ts"]) == pd.Timestamp(bars[4].bar_ts)

    async def test_read_missing_symbol_raises(self, store: DataStore):
        with pytest.raises(DataMissingError) as exc_info:
            await store.read_bars("ETHUSDT",
                                  _bar_ts(0),
                                  _bar_ts(0) + timedelta(hours=4))
        assert "ETHUSDT" in str(exc_info.value)

    async def test_empty_range_returns_empty_df(self, store: DataStore):
        bar = make_bar(_bar_ts(0))
        await store.write_bars([bar])

        # Range before any data
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2020, 1, 2, tzinfo=timezone.utc)
        df = await store.read_bars("BTCUSDT", start, end)
        assert df.empty

    async def test_oi_lsr_funding_persisted(self, store: DataStore):
        bar = make_bar(_bar_ts(0), oi=55000.0, lsr_top=1.3, funding_rate=0.0002)
        await store.write_bars([bar])

        start = _bar_ts(0)
        end   = _bar_ts(0) + timedelta(hours=4)
        df = await store.read_bars("BTCUSDT", start, end)
        assert df.iloc[0]["oi"]           == pytest.approx(55000.0)
        assert df.iloc[0]["lsr_top"]      == pytest.approx(1.3)
        assert df.iloc[0]["funding_rate"] == pytest.approx(0.0002)

    async def test_multi_symbol_isolated(self, store: DataStore):
        btc = make_bar(_bar_ts(0), symbol="BTCUSDT", close=50000.0)
        eth = make_bar(_bar_ts(0), symbol="ETHUSDT", close=3000.0)
        await store.write_bars([btc])
        await store.write_bars([eth])

        btc_df = await store.read_bars("BTCUSDT", _bar_ts(0), _bar_ts(0) + timedelta(hours=4))
        eth_df = await store.read_bars("ETHUSDT", _bar_ts(0), _bar_ts(0) + timedelta(hours=4))
        assert btc_df.iloc[0]["close"] == pytest.approx(50000.0)
        assert eth_df.iloc[0]["close"] == pytest.approx(3000.0)


class TestVisionMetrics:
    def _make_metric(self, bar_ts: datetime, oi: float = 50000.0) -> dict:
        return {
            "bar_ts":            bar_ts,
            "sum_open_interest": oi,
            "sum_oi_value":      oi * 50000.0,
            "top_ls_ratio":      1.2,
            "taker_vol_ratio":   0.55,
        }

    async def test_write_read_round_trip(self, store: DataStore):
        records = [self._make_metric(_bar_ts(i * 4)) for i in range(3)]
        await store.write_vision_metrics(records, "BTCUSDT")

        start = _bar_ts(0)
        end   = _bar_ts(8) + timedelta(hours=4)
        df = await store.read_metrics("BTCUSDT", start, end)
        assert len(df) == 3

    async def test_missing_metrics_returns_empty_df(self, store: DataStore):
        df = await store.read_metrics("BTCUSDT",
                                      _bar_ts(0),
                                      _bar_ts(0) + timedelta(hours=4))
        assert df.empty


class TestMergeOHLCVMetrics:
    def _make_metric(self, bar_ts: datetime, oi: float = 50000.0) -> dict:
        return {
            "bar_ts":            bar_ts,
            "sum_open_interest": oi,
            "sum_oi_value":      oi * 50000,
            "top_ls_ratio":      1.2,
            "taker_vol_ratio":   0.55,
        }

    async def test_merge_aligns_on_bar_ts(self, store: DataStore):
        bars = make_bars(4)
        await store.write_bars(bars)

        metrics = [self._make_metric(b.bar_ts) for b in bars]
        await store.write_vision_metrics(metrics, "BTCUSDT")

        start = bars[0].bar_ts
        end   = bars[-1].bar_ts + timedelta(hours=4)
        df = await store.merge_ohlcv_metrics("BTCUSDT", start, end)

        assert len(df) == 4
        assert "sum_open_interest" in df.columns
        assert df["sum_open_interest"].notna().all()

    async def test_merge_missing_metrics_returns_nan_columns(self, store: DataStore):
        """merge should not raise when no metrics exist — returns NaN oi columns."""
        bars = make_bars(3)
        await store.write_bars(bars)

        start = bars[0].bar_ts
        end   = bars[-1].bar_ts + timedelta(hours=4)
        df = await store.merge_ohlcv_metrics("BTCUSDT", start, end)

        # OHLCV rows present
        assert len(df) == 3
        # OI columns exist but are NaN
        assert "sum_open_interest" in df.columns
        assert df["sum_open_interest"].isna().all()

    async def test_merge_missing_ohlcv_raises(self, store: DataStore):
        with pytest.raises(DataMissingError):
            await store.merge_ohlcv_metrics("BTCUSDT",
                                            _bar_ts(0),
                                            _bar_ts(4))
