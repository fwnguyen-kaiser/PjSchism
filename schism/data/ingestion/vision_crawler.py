"""
vision_crawler.py — Crawl data.binance.vision for OI and LSR history > 30 days.

Binance REST endpoints for OI and LSR only return ≤30 days of data.
For training windows that require 60–180+ days, we must download the
historical CSV zips from data.binance.vision.

Available datasets (USDM Futures, monthly/daily granularity):
  metrics/um/daily/metrics/<symbol>/
    - CSV columns: create_time, symbol, sum_open_interest,
                   sum_open_interest_value, count_toptrader_long_short_ratio,
                   sum_toptrader_long_short_ratio, count_long_short_ratio,
                   sum_taker_long_short_vol_ratio

Data is available from approx. 2020-01-01 onwards.

Usage (CLI):
    python -m schism.data.ingestion.vision_crawler \
        --symbol BTCUSDT --start 2023-01-01 [--end 2024-01-01] [--out ./data/parquet]

Usage (programmatic):
    crawler = VisionCrawler(symbol="BTCUSDT", out_dir=Path("data/volumes/parquet"))
    records = await crawler.fetch_range(start=datetime(2023, 1, 1), end=datetime(2024, 1, 1))
    # records is a list of dicts with bar_ts (UTC datetime), sum_open_interest,
    # sum_oi_value, top_ls_ratio, taker_vol_ratio
"""

from __future__ import annotations

import asyncio
import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from schism.utils.date_helpers import normalize_ts, to_iso
from schism.utils.logger import ingestion_logger

# ── Constants ─────────────────────────────────────────────────────────────────

_VISION_BASE = "https://data.binance.vision"
_METRICS_PATH = "data/futures/um/daily/metrics"
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_CONCURRENT = 4     # concurrent downloads to stay polite
_RETRY_ATTEMPTS = 3


# ── Record dataclass ──────────────────────────────────────────────────────────

def _parse_vision_row(row: dict) -> Optional[dict]:
    """
    Parse one CSV row from the vision metrics zip.

    Expected columns (Binance vision daily metrics CSV):
      create_time, symbol,
      sum_open_interest, sum_open_interest_value,
      count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
      count_long_short_ratio, sum_taker_long_short_vol_ratio

    Returns None on parse failure (corrupted row).
    """
    try:
        clean_row = {
            str(k).strip(): str(v).strip()
            for k, v in row.items()
            if k is not None and v is not None
        }
        raw_ts = clean_row.get("create_time") or clean_row.get("timestamp")
        if raw_ts is None:
            raise KeyError("create_time/timestamp")

        if raw_ts.isdigit():
            bar_ts = normalize_ts(int(raw_ts), source="binance_ms")
        else:
            bar_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
            else:
                bar_ts = bar_ts.astimezone(timezone.utc)
        return {
            "bar_ts":           bar_ts,
            "sum_open_interest": float(clean_row["sum_open_interest"]),
            "sum_oi_value":     float(clean_row["sum_open_interest_value"]),
            # top-trader L/S ratio — primary signal (u4 driver)
            "top_ls_ratio":     float(clean_row.get("sum_toptrader_long_short_ratio", 0.0)),
            # all-account taker vol ratio — auxiliary
            "taker_vol_ratio":  float(clean_row.get("sum_taker_long_short_vol_ratio", 0.0)),
        }
    except (KeyError, ValueError, TypeError) as exc:
        ingestion_logger.warning(
            "vision_row_parse_error",
            row=row,
            error=str(exc),
        )
        return None


# ── Crawler ───────────────────────────────────────────────────────────────────

class VisionCrawler:
    """
    Download and parse Binance Vision daily metrics zips.

    Args:
        symbol:  Trading pair, e.g. "BTCUSDT".
        out_dir: Optional directory to cache raw zips (avoids re-download).
        timeout: httpx timeout for each zip download.
    """

    def __init__(
        self,
        symbol: str,
        out_dir: Optional[Path] = None,
        timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
    ) -> None:
        self.symbol = symbol.upper()
        self.out_dir = out_dir
        self._timeout = timeout
        if out_dir:
            (out_dir / "vision_cache" / symbol).mkdir(parents=True, exist_ok=True)

    def _zip_url(self, date: datetime) -> str:
        """Return the URL for the daily metrics zip for `date`."""
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{self.symbol}-metrics-{date_str}.zip"
        return f"{_VISION_BASE}/{_METRICS_PATH}/{self.symbol}/{filename}"

    def _cache_path(self, date: datetime) -> Optional[Path]:
        """Return local cache path for a zip, or None if caching disabled."""
        if not self.out_dir:
            return None
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{self.symbol}-metrics-{date_str}.zip"
        return self.out_dir / "vision_cache" / self.symbol / filename

    async def _fetch_zip(
        self,
        client: httpx.AsyncClient,
        date: datetime,
    ) -> Optional[bytes]:
        """
        Download one daily zip. Returns bytes or None on 404 / error.
        Caches to disk if out_dir is set.
        """
        cache = self._cache_path(date)
        if cache and cache.exists():
            ingestion_logger.debug(
                "vision_zip_cache_hit",
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
                path=str(cache),
            )
            return cache.read_bytes()

        url = self._zip_url(date)
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.content
                    if cache:
                        cache.write_bytes(data)
                    ingestion_logger.debug(
                        "vision_zip_downloaded",
                        symbol=self.symbol,
                        date=date.strftime("%Y-%m-%d"),
                        bytes=len(data),
                    )
                    return data
                if resp.status_code == 404:
                    ingestion_logger.debug(
                        "vision_zip_not_found",
                        symbol=self.symbol,
                        date=date.strftime("%Y-%m-%d"),
                        url=url,
                    )
                    return None   # data not yet available for this date
                ingestion_logger.warning(
                    "vision_zip_http_error",
                    symbol=self.symbol,
                    date=date.strftime("%Y-%m-%d"),
                    status=resp.status_code,
                    attempt=attempt,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                ingestion_logger.warning(
                    "vision_zip_request_error",
                    symbol=self.symbol,
                    date=date.strftime("%Y-%m-%d"),
                    attempt=attempt,
                    error=str(exc),
                )
            if attempt < _RETRY_ATTEMPTS:
                await asyncio.sleep(2 ** attempt)

        return None

    def _parse_zip(self, zip_bytes: bytes, date: datetime) -> list[dict]:
        """Extract and parse CSV rows from a zip file bytes object."""
        records: list[dict] = []
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    if not name.endswith(".csv"):
                        continue
                    with zf.open(name) as f:
                        reader = csv.DictReader(
                            io.TextIOWrapper(f, encoding="utf-8")
                        )
                        for row in reader:
                            parsed = _parse_vision_row(row)
                            if parsed:
                                records.append(parsed)
        except zipfile.BadZipFile:
            ingestion_logger.warning(
                "vision_zip_bad_file",
                symbol=self.symbol,
                date=date.strftime("%Y-%m-%d"),
            )
        ingestion_logger.debug(
            "vision_zip_processed",
            symbol=self.symbol,
            date=date.strftime("%Y-%m-%d"),
            records=len(records),
        )
        return records

    async def fetch_range(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch all daily metrics records between start (inclusive) and
        end (exclusive, defaults to today UTC).

        Args:
            start: First date to fetch (UTC-aware or naive; naive assumed UTC).
            end:   Last date (exclusive). Defaults to today UTC.

        Returns:
            List of records sorted by bar_ts ascending. Each record:
              bar_ts (datetime UTC), sum_open_interest (float),
              sum_oi_value (float), top_ls_ratio (float),
              taker_vol_ratio (float)
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(tz=timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Build list of dates to fetch
        dates: list[datetime] = []
        cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while cur < end:
            dates.append(cur)
            cur += timedelta(days=1)

        ingestion_logger.info(
            "vision_crawl_start",
            symbol=self.symbol,
            start=to_iso(start),
            end=to_iso(end),
            days=len(dates),
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        all_records: list[dict] = []

        async def fetch_one(client: httpx.AsyncClient, date: datetime) -> list[dict]:
            async with sem:
                data = await self._fetch_zip(client, date)
                if data is None:
                    return []
                return self._parse_zip(data, date)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [fetch_one(client, d) for d in dates]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                ingestion_logger.warning(
                    "vision_fetch_exception",
                    symbol=self.symbol,
                    date=dates[i].strftime("%Y-%m-%d"),
                    error=str(result),
                )
            elif isinstance(result, list):
                all_records.extend(result)

        all_records.sort(key=lambda r: r["bar_ts"])

        ingestion_logger.info(
            "vision_crawl_done",
            symbol=self.symbol,
            total_records=len(all_records),
            start=to_iso(start),
            end=to_iso(end),
        )
        return all_records


# ── CLI entry point ───────────────────────────────────────────────────────────

async def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Crawl data.binance.vision metrics")
    parser.add_argument("--symbol", default="BTCUSDT", help="Futures symbol")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None,   help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--out",   default="./data/volumes/parquet",
                        help="Output dir for parquet + vision cache")
    parser.add_argument("--print-json", action="store_true",
                        help="Print records as JSON lines to stdout")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt   = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end else None
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    crawler = VisionCrawler(symbol=args.symbol, out_dir=out_dir)
    records = await crawler.fetch_range(start=start_dt, end=end_dt)

    if args.print_json:
        for r in records:
            print(json.dumps({k: (v.isoformat() if isinstance(v, datetime) else v)
                              for k, v in r.items()}))
    else:
        print(f"Fetched {len(records)} records for {args.symbol}")


if __name__ == "__main__":
    asyncio.run(_main())
