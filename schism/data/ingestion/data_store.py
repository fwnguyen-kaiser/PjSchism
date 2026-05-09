"""
data_store.py — Parquet-based persistent store for OHLCV bars.

Responsibilities:
  - Write 4h OHLCV bars (with CVD) to partitioned parquet files
  - Read bar ranges for feature computation and training
  - Merge ohlcv + OI/LSR/funding metrics on bar_ts (inner join)
  - Upsert logic: new bars overwrite existing rows with same bar_ts

Parquet partition scheme:
  <root>/<symbol>/year=YYYY/month=MM/<symbol>_YYYYMM.parquet

Column schema (matches ohlcv_bars hypertable):
  bar_ts (datetime64[ns, UTC]), symbol, open, high, low, close,
  volume, cvd, oi, lsr_top, funding_rate, best_bid, best_ask,
  bybit_fr, num_trades, taker_buy_base, quote_volume, source

TimescaleDB is the primary queryable store (state_history, feature_vectors,
refit_log). Parquet acts as the immutable raw-data layer — it is always
written before the DB insert and is the source of truth for retraining.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schism.data.ingestion.bar_builder import Bar
from schism.utils.date_helpers import normalize_ts, to_iso
from schism.utils.exceptions import DataMissingError
from schism.utils.logger import ingestion_logger

# ── Parquet schema ────────────────────────────────────────────────────────────

_SCHEMA = pa.schema([
    pa.field("bar_ts",         pa.timestamp("ns", tz="UTC")),
    pa.field("symbol",         pa.string()),
    pa.field("open",           pa.float64()),
    pa.field("high",           pa.float64()),
    pa.field("low",            pa.float64()),
    pa.field("close",          pa.float64()),
    pa.field("volume",         pa.float64()),
    pa.field("cvd",            pa.float64()),
    pa.field("oi",             pa.float64()),
    pa.field("lsr_top",        pa.float64()),
    pa.field("funding_rate",   pa.float64()),
    pa.field("best_bid",       pa.float64()),
    pa.field("best_ask",       pa.float64()),
    pa.field("bybit_fr",       pa.float64()),
    pa.field("num_trades",     pa.int64()),
    pa.field("taker_buy_base", pa.float64()),
    pa.field("quote_volume",   pa.float64()),
    pa.field("source",         pa.string()),
])

_PARTITION_COLS = ["symbol"]      # Hive-style: symbol=BTCUSDT/
_SORT_COL       = "bar_ts"


# ── Path helpers ──────────────────────────────────────────────────────────────

def _month_path(root: Path, symbol: str, dt: datetime) -> Path:
    """Return partition directory for symbol + year-month of dt."""
    return root / f"symbol={symbol}" / f"year={dt.year}" / f"month={dt.month:02d}"


def _parquet_filename(symbol: str, dt: datetime) -> str:
    return f"{symbol}_{dt.year}{dt.month:02d}.parquet"


# ── DataStore ─────────────────────────────────────────────────────────────────

class DataStore:
    """
    Parquet-based store for 4h OHLCV bars.

    Thread-safe for read; write operations use an asyncio.Lock per partition
    to prevent concurrent write corruption.

    Args:
        root: Root directory for parquet files, e.g. Path("data/volumes/parquet")
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_locks: dict[Path, asyncio.Lock] = {}

    def _get_lock(self, path: Path) -> asyncio.Lock:
        if path not in self._write_locks:
            self._write_locks[path] = asyncio.Lock()
        return self._write_locks[path]

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write_bars(self, bars: list[Bar]) -> None:
        """
        Persist a list of Bars to parquet, partitioned by symbol + year-month.

        Upsert semantics: if the partition file already exists, the new bars
        are merged (new rows overwrite old rows with the same bar_ts, keeping
        all other existing rows).

        Args:
            bars: List of Bar objects to persist.
        """
        if not bars:
            return

        df = _bars_to_df(bars)
        # Group by (symbol, year, month) for partition writing
        groups = df.groupby([
            df["symbol"],
            df["bar_ts"].dt.year,
            df["bar_ts"].dt.month,
        ])

        for (symbol, year, month), group_df in groups:
            dummy_dt = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
            part_dir = _month_path(self.root, str(symbol), dummy_dt)
            filename = _parquet_filename(str(symbol), dummy_dt)
            part_file = part_dir / filename

            async with self._get_lock(part_file):
                await asyncio.to_thread(
                    _upsert_partition, part_file, group_df, str(symbol)
                )

        ingestion_logger.info(
            "parquet_written",
            symbol=bars[0].symbol if bars else None,
            count=len(bars),
            first=bars[0].bar_ts.isoformat() if bars else None,
            last=bars[-1].bar_ts.isoformat() if bars else None,
        )

    async def write_vision_metrics(
        self,
        records: list[dict],
        symbol: str,
    ) -> None:
        """
        Persist vision_crawler records (OI + LSR) to a separate parquet shard.

        These are later merged with OHLCV bars via merge_ohlcv_metrics().
        Stored under: <root>/metrics/symbol=<symbol>/...

        Args:
            records: List of dicts from VisionCrawler.fetch_range()
            symbol:  Trading pair symbol
        """
        if not records:
            return

        df = pd.DataFrame(records)
        df["symbol"] = symbol
        df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
        df.sort_values("bar_ts", inplace=True)
        df.drop_duplicates(subset=["bar_ts"], keep="last", inplace=True)

        groups = df.groupby([df["bar_ts"].dt.year, df["bar_ts"].dt.month])
        for (year, month), group_df in groups:
            dummy_dt = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
            part_dir = self.root / "metrics" / f"symbol={symbol}" / \
                       f"year={year}" / f"month={month:02d}"
            part_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{symbol}_metrics_{year}{month:02d}.parquet"
            part_file = part_dir / filename

            async with self._get_lock(part_file):
                await asyncio.to_thread(
                    _upsert_dataframe, part_file, group_df, "bar_ts"
                )

        ingestion_logger.info(
            "metrics_written",
            symbol=symbol,
            count=len(records),
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_bars(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Read OHLCV bars for a symbol between start and end.

        Args:
            symbol: Trading pair symbol.
            start:  Inclusive start datetime (UTC-aware or naive).
            end:    Exclusive end datetime. Defaults to now.

        Returns:
            DataFrame sorted by bar_ts, columns matching _SCHEMA.
            Empty DataFrame if no data found.

        Raises:
            DataMissingError: Partition directory does not exist at all.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(tz=timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        sym_root = self.root / f"symbol={symbol}"
        if not sym_root.exists():
            raise DataMissingError(
                f"No parquet data for symbol {symbol!r}",
                source="parquet",
                path=str(sym_root),
            )

        parquet_files = _list_partition_files(sym_root, start, end)
        if not parquet_files:
            ingestion_logger.warning(
                "parquet_no_files",
                symbol=symbol,
                start=to_iso(start),
                end=to_iso(end),
            )
            return pd.DataFrame(columns=_SCHEMA.names)

        df = await asyncio.to_thread(_read_parquet_files, parquet_files)
        if df.empty:
            return df

        df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
        mask = (df["bar_ts"] >= pd.Timestamp(start)) & (df["bar_ts"] < pd.Timestamp(end))
        df = df.loc[mask].sort_values("bar_ts").reset_index(drop=True)

        ingestion_logger.debug(
            "parquet_read",
            symbol=symbol,
            start=to_iso(start),
            end=to_iso(end),
            rows=len(df),
        )
        return df

    async def read_metrics(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Read vision metrics (OI, LSR) for a symbol between start and end.

        Returns:
            DataFrame with columns: bar_ts, sum_open_interest, sum_oi_value,
            top_ls_ratio, taker_vol_ratio
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(tz=timezone.utc)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        metrics_root = self.root / "metrics" / f"symbol={symbol}"
        if not metrics_root.exists():
            ingestion_logger.warning(
                "metrics_not_found",
                symbol=symbol,
                path=str(metrics_root),
            )
            return pd.DataFrame()

        parquet_files = _list_partition_files(metrics_root, start, end)
        if not parquet_files:
            return pd.DataFrame()

        df = await asyncio.to_thread(_read_parquet_files, parquet_files)
        if df.empty:
            return df

        df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
        mask = (df["bar_ts"] >= pd.Timestamp(start)) & (df["bar_ts"] < pd.Timestamp(end))
        return df.loc[mask].sort_values("bar_ts").reset_index(drop=True)

    # ── Merge ─────────────────────────────────────────────────────────────────

    async def merge_ohlcv_metrics(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Inner-join OHLCV bars with OI/LSR metrics on bar_ts.

        The merged DataFrame has all columns from both sources. Rows where
        either OHLCV or metrics data is missing are dropped (inner join).

        Vision metrics are recorded at varying granularities (often 1 per 4h
        or 1 per day). The join uses nearest-forward-fill within a 4h tolerance
        to align metric timestamps to bar_ts.

        Returns:
            Merged DataFrame sorted by bar_ts.

        Raises:
            DataMissingError: OHLCV bars not found.
        """
        ohlcv_df, metrics_df = await asyncio.gather(
            self.read_bars(symbol, start, end),
            self.read_metrics(symbol, start, end),
        )

        if ohlcv_df.empty:
            raise DataMissingError(
                f"No OHLCV bars for {symbol} between {to_iso(start)} and "
                f"{to_iso(end or datetime.now(tz=timezone.utc))}",
                source="parquet",
                path=str(self.root),
            )

        if metrics_df.empty:
            ingestion_logger.warning(
                "merge_no_metrics",
                symbol=symbol,
                note="Proceeding with OI/LSR columns as NaN",
            )
            # Return OHLCV with NaN metric columns so callers can handle gracefully
            for col in ["sum_open_interest", "sum_oi_value", "top_ls_ratio"]:
                ohlcv_df[col] = float("nan")
            return ohlcv_df

        # Align metrics to 4h bar boundaries using merge_asof (backward fill)
        ohlcv_df = ohlcv_df.sort_values("bar_ts")
        metrics_df = metrics_df.sort_values("bar_ts")

        merged = pd.merge_asof(
            ohlcv_df,
            metrics_df[["bar_ts", "sum_open_interest", "sum_oi_value", "top_ls_ratio"]],
            on="bar_ts",
            direction="backward",
            tolerance=pd.Timedelta("4h"),
        )

        ingestion_logger.info(
            "merge_complete",
            symbol=symbol,
            ohlcv_rows=len(ohlcv_df),
            metrics_rows=len(metrics_df),
            merged_rows=len(merged),
            oi_null_pct=round(merged["sum_open_interest"].isna().mean() * 100, 1),
        )

        return merged.sort_values("bar_ts").reset_index(drop=True)


# ── Utility functions (sync, run via asyncio.to_thread) ──────────────────────

def _bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    """Convert a list of Bar objects to a DataFrame matching _SCHEMA."""
    rows = [b.to_dict() for b in bars]
    df = pd.DataFrame(rows)
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    # Ensure all schema columns are present
    for col_name in _SCHEMA.names:
        if col_name not in df.columns:
            df[col_name] = None
    return df[_SCHEMA.names]


def _upsert_partition(part_file: Path, new_df: pd.DataFrame, symbol: str) -> None:
    """
    Upsert new_df into an existing parquet partition file.
    New rows (same bar_ts) overwrite existing ones.
    """
    part_file.parent.mkdir(parents=True, exist_ok=True)

    if part_file.exists():
        existing = pq.read_table(part_file).to_pandas()
        existing["bar_ts"] = pd.to_datetime(existing["bar_ts"], utc=True)
        # Merge: keep new rows, fill in existing rows not in new_df
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["bar_ts", "symbol"], keep="last", inplace=True)
    else:
        combined = new_df

    combined.sort_values("bar_ts", inplace=True)
    table = pa.Table.from_pandas(combined, schema=_SCHEMA, preserve_index=False)
    pq.write_table(table, part_file, compression="snappy")


def _upsert_dataframe(part_file: Path, new_df: pd.DataFrame, ts_col: str) -> None:
    """Generic upsert for non-OHLCV parquet files (e.g. metrics)."""
    part_file.parent.mkdir(parents=True, exist_ok=True)

    if part_file.exists():
        existing = pq.read_table(part_file).to_pandas()
        existing[ts_col] = pd.to_datetime(existing[ts_col], utc=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=[ts_col], keep="last", inplace=True)
    else:
        combined = new_df

    combined.sort_values(ts_col, inplace=True)
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), part_file,
                   compression="snappy")


def _list_partition_files(root: Path, start: datetime, end: datetime) -> list[Path]:
    """
    List parquet files in year=*/month=* partitions that overlap [start, end).
    """
    files: list[Path] = []
    # Collect all year/month partitions within range
    cur_year, cur_month = start.year, start.month
    end_year, end_month = end.year, end.month

    while (cur_year, cur_month) <= (end_year, end_month):
        pattern = f"year={cur_year}/month={cur_month:02d}/*.parquet"
        files.extend(root.glob(pattern))
        if cur_month == 12:
            cur_year += 1
            cur_month = 1
        else:
            cur_month += 1

    return sorted(files)


def _read_parquet_files(files: list[Path]) -> pd.DataFrame:
    """Read and concatenate a list of parquet files."""
    if not files:
        return pd.DataFrame()
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables).to_pandas()