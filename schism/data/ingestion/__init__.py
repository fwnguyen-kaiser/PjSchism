"""
schism.data.ingestion — public API exports.

Primary entry point for other packages:
    from schism.data.ingestion import BinanceClient, DataStore, VisionCrawler
    from schism.data.ingestion import Bar, build_bars_from_klines
"""

from schism.data.ingestion.bar_builder import (
    Bar,
    LiveBarBuilder,
    build_bar_from_kline,
    build_bars_from_klines,
    compute_cvd,
)
from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.data_store import DataStore
from schism.data.ingestion.vision_crawler import VisionCrawler

__all__ = [
    "Bar",
    "BinanceClient",
    "DataStore",
    "LiveBarBuilder",
    "VisionCrawler",
    "build_bar_from_kline",
    "build_bars_from_klines",
    "compute_cvd",
]