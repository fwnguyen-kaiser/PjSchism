"""
Opt-in integration tests for real Binance ingestion data.

These tests hit public Binance endpoints and write parquet to a pytest tmp dir.
Run with:
    $env:SCHISM_RUN_REAL_DATA="1"
    python -m pytest schism/tests/test_ingestion_real_data.py -q -s
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from schism.data.ingestion.bar_builder import Bar, build_bars_from_klines
from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.data_store import DataStore
from schism.data.ingestion.vision_crawler import VisionCrawler
from schism.utils.date_helpers import utc_now


pytestmark = pytest.mark.asyncio


run_real_data = pytest.mark.skipif(
    os.getenv("SCHISM_RUN_REAL_DATA") != "1",
    reason="Set SCHISM_RUN_REAL_DATA=1 to hit real Binance endpoints.",
)


@run_real_data
async def test_real_binance_rest_to_parquet_round_trip(tmp_path):
    symbol = os.getenv("SCHISM_REAL_SYMBOL", "BTCUSDT")
    end = utc_now()
    start = end - timedelta(days=3)

    async with BinanceClient() as client:
        klines = await client.get_klines(
            symbol=symbol,
            interval="4h",
            start_time=start,
            end_time=end,
            limit=20,
        )
        funding = await client.get_funding_rate(
            symbol=symbol,
            start_time=start - timedelta(hours=8),
            end_time=end,
            limit=10,
        )
        oi = await client.get_open_interest_hist(
            symbol=symbol,
            period="4h",
            start_time=start,
            end_time=end,
            limit=20,
        )
        lsr = await client.get_top_lsr(
            symbol=symbol,
            period="4h",
            start_time=start,
            end_time=end,
            limit=20,
        )

    assert klines, "real kline response was empty"
    assert funding, "real funding response was empty"
    assert oi, "real open-interest response was empty"
    assert lsr, "real top-LSR response was empty"

    oi_by_ts = {row["bar_ts"]: row["sum_open_interest"] for row in oi}
    lsr_by_ts = {row["bar_ts"]: row["long_short_ratio"] for row in lsr}
    funding.sort(key=lambda row: row["funding_time"])
    funding_idx = 0
    current_funding_rate = None

    bars = []
    for row in klines:
        while (
            funding_idx < len(funding)
            and funding[funding_idx]["funding_time"] <= row["open_time"]
        ):
            current_funding_rate = funding[funding_idx]["funding_rate"]
            funding_idx += 1
        bars.append(Bar(
            bar_ts=row["open_time"],
            symbol=symbol,
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            cvd=0.0,
            num_trades=row["num_trades"],
            taker_buy_base=row["taker_buy_base"],
            quote_volume=row["quote_volume"],
            oi=oi_by_ts.get(row["open_time"]),
            lsr_top=lsr_by_ts.get(row["open_time"]),
            funding_rate=current_funding_rate,
        ))

    store = DataStore(tmp_path / "parquet")
    await store.write_bars(bars)

    df = await store.read_bars(
        symbol,
        bars[0].bar_ts,
        bars[-1].bar_ts + timedelta(hours=4),
    )

    assert len(df) == len(bars)
    assert df["bar_ts"].is_monotonic_increasing
    assert (df["symbol"] == symbol).all()
    assert (df["volume"] >= 0).all()
    assert df["close"].notna().all()
    assert df["oi"].notna().any()
    assert df["lsr_top"].notna().any()
    assert df["funding_rate"].notna().any()


@run_real_data
async def test_real_vision_zip_metrics_merge_with_rest_klines(tmp_path):
    symbol = os.getenv("SCHISM_REAL_SYMBOL", "BTCUSDT")
    start_date = os.getenv("SCHISM_REAL_VISION_START", "2024-01-01")
    days = int(os.getenv("SCHISM_REAL_VISION_DAYS", "2"))
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    store = DataStore(tmp_path / "parquet")

    crawler = VisionCrawler(symbol=symbol, out_dir=tmp_path / "vision_cache")
    records = await crawler.fetch_range(start=start, end=end)
    assert records, "real Binance Vision zip records were empty"
    await store.write_vision_metrics(records, symbol)

    async with BinanceClient() as client:
        klines = await client.get_klines(
            symbol=symbol,
            interval="4h",
            start_time=start,
            end_time=end,
            limit=20,
        )

    assert klines, "real historical kline response was empty"
    bars = [bar for bar in build_bars_from_klines(klines, symbol) if bar.bar_ts < end]
    await store.write_bars(bars)

    merged = await store.merge_ohlcv_metrics(symbol, start, end)

    assert len(merged) == len(bars)
    assert merged["close"].notna().all()
    assert merged["sum_open_interest"].notna().any()
    assert merged["top_ls_ratio"].notna().any()
