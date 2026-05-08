"""
test_ingestion_service.py - Unit tests for ingestion_service.py.

No real Binance, Redis, or filesystem service is used here. Dependencies are
mocked at the service boundary so these tests cover orchestration behavior.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

import pytest

from schism.tests.conftest import _bar_ts, make_bar, make_kline_row, make_mock_client
from schism.data.ingestion.bar_builder import Bar
from schism.data.ingestion.cache.funding_cache import FundingCache
from schism.data.ingestion.cache.oi_cache import OICache
from schism.data.ingestion.context import AppContext
from schism.data.ingestion.publishers.redis_publisher import RedisPublisher
from schism.data.ingestion.services.backfill_service import BackfillService
from schism.data.ingestion.services.live_service import LiveService
from schism.data.ingestion.vision_crawler import _parse_vision_row


pytestmark = pytest.mark.asyncio


def make_context(
    *,
    client=None,
    store=None,
    redis=None,
    funding_cache=None,
    oi_cache=None,
    publisher=None,
    bar_repo=None,
    backfill_days: int = 1,
) -> AppContext:
    redis = redis or MagicMock()
    return AppContext(
        client=client or make_mock_client(),
        store=store or MagicMock(),
        redis=redis,
        funding_cache=funding_cache or FundingCache(),
        oi_cache=oi_cache or OICache(),
        publisher=publisher or RedisPublisher(redis),
        bar_repo=bar_repo,
        db_engine=None,
        symbols=["BTCUSDT"],
        backfill_days=backfill_days,
        parquet_root=Path("/tmp/parquet"),
        env="test",
    )


class TestRedisXAdd:
    async def test_publishes_bar_fields_to_symbol_stream(self):
        redis = MagicMock()
        redis.xadd = AsyncMock()
        bar = make_bar(_bar_ts(0), oi=123.0, lsr_top=1.4, funding_rate=0.0002)

        await RedisPublisher(redis).publish(bar)

        redis.xadd.assert_awaited_once()
        key, fields = redis.xadd.await_args.args[:2]
        kwargs = redis.xadd.await_args.kwargs

        assert key == "schism:bars:BTCUSDT"
        assert fields["symbol"] == "BTCUSDT"
        assert fields["bar_ts"] == bar.bar_ts.isoformat()
        assert fields["close"] == str(bar.close)
        assert fields["oi"] == "123.0"
        assert fields["lsr_top"] == "1.4"
        assert fields["funding_rate"] == "0.0002"
        assert kwargs["approximate"] is True

    async def test_publish_failure_is_non_fatal(self):
        redis = MagicMock()
        redis.xadd = AsyncMock(side_effect=RuntimeError("redis down"))

        await RedisPublisher(redis).publish(make_bar(_bar_ts(0)))

        redis.xadd.assert_awaited_once()


class TestCaches:
    async def test_funding_cache_refresh_uses_latest_record(self):
        client = make_mock_client(funding_records=[
            {"funding_time": _bar_ts(0), "funding_rate": 0.0001},
            {"funding_time": _bar_ts(8), "funding_rate": 0.0003},
        ])
        cache = FundingCache()

        await cache.refresh(client, ["BTCUSDT"])

        assert cache.get("BTCUSDT") == pytest.approx(0.0003)
        client.get_funding_rate.assert_awaited_once()

    async def test_oi_cache_refresh_uses_latest_oi_and_lsr(self):
        client = make_mock_client(
            oi_records=[{"bar_ts": _bar_ts(0), "sum_open_interest": 555.0}],
            lsr_records=[{"bar_ts": _bar_ts(0), "long_short_ratio": 1.25}],
        )
        cache = OICache()

        await cache.refresh(client, ["BTCUSDT"])

        assert cache.get_oi("BTCUSDT") == pytest.approx(555.0)
        assert cache.get_lsr("BTCUSDT") == pytest.approx(1.25)

    async def test_cache_refresh_failures_do_not_raise(self):
        client = make_mock_client()
        client.get_funding_rate = AsyncMock(side_effect=RuntimeError("api down"))
        client.get_open_interest_hist = AsyncMock(side_effect=RuntimeError("api down"))

        await FundingCache().refresh(client, ["BTCUSDT"])
        await OICache().refresh(client, ["BTCUSDT"])


class TestBackfill:
    async def test_backfill_klines_writes_bars(self):
        klines = [make_kline_row(_bar_ts(0)), make_kline_row(_bar_ts(4))]
        client = make_mock_client(klines=klines)
        store = MagicMock()
        store.write_bars = AsyncMock()
        ctx = make_context(client=client, store=store, backfill_days=1)

        await BackfillService(ctx).run_klines("BTCUSDT")

        store.write_bars.assert_awaited_once()
        bars = store.write_bars.await_args.args[0]
        assert [bar.bar_ts for bar in bars] == [_bar_ts(0), _bar_ts(4)]
        assert all(isinstance(bar, Bar) for bar in bars)
        assert all(bar.cvd == pytest.approx(10.0) for bar in bars)  # bar delta: 2*55-100=10
        assert all(bar.funding_rate == pytest.approx(0.0001) for bar in bars)

    async def test_backfill_klines_paginates_until_short_page(self):
        first_page = [make_kline_row(_bar_ts(0) + timedelta(hours=4 * i)) for i in range(1000)]
        second_page = [make_kline_row(first_page[-1]["open_time"] + timedelta(hours=4))]
        client = make_mock_client()
        client.get_klines = AsyncMock(side_effect=[first_page, second_page])
        store = MagicMock()
        store.write_bars = AsyncMock()
        ctx = make_context(client=client, store=store, backfill_days=365)

        await BackfillService(ctx).run_klines("BTCUSDT")

        assert client.get_klines.await_count == 2
        assert store.write_bars.await_count == 1
        assert len(store.write_bars.await_args.args[0]) == 1001

    async def test_backfill_vision_writes_records_from_crawler(self):
        records = [{"bar_ts": _bar_ts(0), "sum_open_interest": 1.0, "top_ls_ratio": 1.1}]
        crawler = MagicMock()
        crawler.fetch_range = AsyncMock(return_value=records)
        store = MagicMock()
        store.write_vision_metrics = AsyncMock()
        ctx = make_context(store=store, backfill_days=2)

        with patch("schism.data.ingestion.services.backfill_service.VisionCrawler", return_value=crawler):
            await BackfillService(ctx).run_vision("BTCUSDT")

        store.write_vision_metrics.assert_awaited_once_with(records, "BTCUSDT")


class TestVisionCrawlerParsing:
    async def test_parse_timestamp_schema(self):
        row = {
            "timestamp": "2024-01-01 00:00:00",
            "sum_open_interest": "100.5",
            "sum_open_interest_value": "4200000.0",
            "sum_toptrader_long_short_ratio": "1.25",
            "sum_taker_long_short_vol_ratio": "0.95",
        }

        parsed = _parse_vision_row(row)

        assert parsed is not None
        assert parsed["bar_ts"].tzinfo is not None
        assert parsed["sum_open_interest"] == pytest.approx(100.5)
        assert parsed["top_ls_ratio"] == pytest.approx(1.25)

    async def test_parse_create_time_schema(self):
        row = {
            "create_time": "1704067200000",
            "sum_open_interest": "100.5",
            "sum_open_interest_value": "4200000.0",
            "sum_toptrader_long_short_ratio": "1.25",
            "sum_taker_long_short_vol_ratio": "0.95",
        }

        parsed = _parse_vision_row(row)

        assert parsed is not None
        assert parsed["bar_ts"] == _bar_ts(0)


class TestLiveLoop:
    async def test_live_loop_attaches_cached_metrics_writes_and_publishes(self):
        bar = make_bar(_bar_ts(0), oi=None, lsr_top=None, funding_rate=None)
        client = make_mock_client()

        async def stream_once(symbol, on_bar_close, interval):
            await on_bar_close(bar)

        client.stream_kline_close = AsyncMock(side_effect=stream_once)
        store = MagicMock()
        store.write_bars = AsyncMock()
        redis = MagicMock()

        funding_cache = FundingCache()
        funding_cache.update("BTCUSDT", 0.0004)
        oi_cache = OICache()
        oi_cache.update("BTCUSDT", oi=999.0, lsr=1.33)
        oi_cache.refresh = AsyncMock()
        publisher = MagicMock()
        publisher.publish = AsyncMock()
        bar_repo = MagicMock()
        bar_repo.upsert_bars = AsyncMock()
        ctx = make_context(
            client=client,
            store=store,
            redis=redis,
            funding_cache=funding_cache,
            oi_cache=oi_cache,
            publisher=publisher,
            bar_repo=bar_repo,
        )

        await LiveService(ctx).run("BTCUSDT")

        assert bar.funding_rate == pytest.approx(0.0004)
        assert bar.oi == pytest.approx(999.0)
        assert bar.lsr_top == pytest.approx(1.33)
        store.write_bars.assert_awaited_once_with([bar])
        bar_repo.upsert_bars.assert_awaited_once_with([bar])
        oi_cache.refresh.assert_awaited_once_with(client, ["BTCUSDT"])
        publisher.publish.assert_awaited_once_with(bar)

    async def test_live_loop_still_refreshes_and_publishes_when_store_write_fails(self):
        bar = make_bar(_bar_ts(0))
        client = make_mock_client()

        async def stream_once(symbol, on_bar_close, interval):
            await on_bar_close(bar)

        client.stream_kline_close = AsyncMock(side_effect=stream_once)
        store = MagicMock()
        store.write_bars = AsyncMock(side_effect=RuntimeError("disk full"))
        oi_cache = OICache()
        oi_cache.refresh = AsyncMock()
        publisher = MagicMock()
        publisher.publish = AsyncMock()
        bar_repo = MagicMock()
        bar_repo.upsert_bars = AsyncMock()
        ctx = make_context(
            client=client,
            store=store,
            oi_cache=oi_cache,
            funding_cache=FundingCache(),
            publisher=publisher,
            bar_repo=bar_repo,
        )

        await LiveService(ctx).run("BTCUSDT")

        oi_cache.refresh.assert_awaited_once()
        bar_repo.upsert_bars.assert_not_awaited()
        publisher.publish.assert_awaited_once()


# ── sync_vision_to_db ─────────────────────────────────────────────────────────

class TestSyncVisionToDb:
    def _merged_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    async def test_patches_oi_lsr_from_merged_parquet(self):
        ts0, ts1 = _bar_ts(0), _bar_ts(4)
        merged = self._merged_df([
            {"bar_ts": pd.Timestamp(ts0), "sum_open_interest": 500.0, "top_ls_ratio": 1.2},
            {"bar_ts": pd.Timestamp(ts1), "sum_open_interest": 510.0, "top_ls_ratio": 1.3},
        ])
        store = MagicMock()
        store.merge_ohlcv_metrics = AsyncMock(return_value=merged)
        bar_repo = MagicMock()
        bar_repo.resolve_ids = AsyncMock(return_value=(1, 3))
        bar_repo.patch_oi_lsr = AsyncMock()
        ctx = make_context(store=store, bar_repo=bar_repo)

        await BackfillService(ctx).sync_vision_to_db("BTCUSDT")

        bar_repo.patch_oi_lsr.assert_awaited_once()
        inst_id, tf_id, rows = bar_repo.patch_oi_lsr.await_args.args
        assert inst_id == 1
        assert tf_id == 3
        assert len(rows) == 2
        assert rows[0]["oi"] == pytest.approx(500.0)
        assert rows[0]["lsr_top"] == pytest.approx(1.2)

    async def test_skips_rows_with_null_oi(self):
        ts0, ts1 = _bar_ts(0), _bar_ts(4)
        merged = self._merged_df([
            {"bar_ts": pd.Timestamp(ts0), "sum_open_interest": float("nan"), "top_ls_ratio": 1.2},
            {"bar_ts": pd.Timestamp(ts1), "sum_open_interest": 500.0, "top_ls_ratio": 1.3},
        ])
        store = MagicMock()
        store.merge_ohlcv_metrics = AsyncMock(return_value=merged)
        bar_repo = MagicMock()
        bar_repo.resolve_ids = AsyncMock(return_value=(1, 3))
        bar_repo.patch_oi_lsr = AsyncMock()
        ctx = make_context(store=store, bar_repo=bar_repo)

        await BackfillService(ctx).sync_vision_to_db("BTCUSDT")

        _, _, rows = bar_repo.patch_oi_lsr.await_args.args
        assert len(rows) == 1
        assert rows[0]["oi"] == pytest.approx(500.0)

    async def test_skips_when_no_bar_repo(self):
        store = MagicMock()
        store.merge_ohlcv_metrics = AsyncMock()
        ctx = make_context(store=store, bar_repo=None)

        await BackfillService(ctx).sync_vision_to_db("BTCUSDT")

        store.merge_ohlcv_metrics.assert_not_awaited()

    async def test_skips_when_no_ohlcv_data(self):
        from schism.utils.exceptions import DataMissingError
        store = MagicMock()
        store.merge_ohlcv_metrics = AsyncMock(
            side_effect=DataMissingError("no data", source="parquet", path="")
        )
        bar_repo = MagicMock()
        bar_repo.patch_oi_lsr = AsyncMock()
        ctx = make_context(store=store, bar_repo=bar_repo)

        await BackfillService(ctx).sync_vision_to_db("BTCUSDT")

        bar_repo.patch_oi_lsr.assert_not_awaited()

    async def test_lsr_top_none_when_top_ls_ratio_nan(self):
        ts0 = _bar_ts(0)
        merged = self._merged_df([
            {"bar_ts": pd.Timestamp(ts0), "sum_open_interest": 500.0, "top_ls_ratio": float("nan")},
        ])
        store = MagicMock()
        store.merge_ohlcv_metrics = AsyncMock(return_value=merged)
        bar_repo = MagicMock()
        bar_repo.resolve_ids = AsyncMock(return_value=(1, 3))
        bar_repo.patch_oi_lsr = AsyncMock()
        ctx = make_context(store=store, bar_repo=bar_repo)

        await BackfillService(ctx).sync_vision_to_db("BTCUSDT")

        _, _, rows = bar_repo.patch_oi_lsr.await_args.args
        assert rows[0]["lsr_top"] is None
