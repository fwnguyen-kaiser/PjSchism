"""Runtime engine process loop implementation."""

from __future__ import annotations

import asyncio

from schism.utils.logger import ingestion_logger


async def run_forever() -> None:
    """
    Keep the model service process alive.

    This runner is intentionally minimal until full regime inference wiring is
    implemented. It provides a stable long-running process contract for Docker.
    """
    ingestion_logger.info("regime_engine_runner_started")
    while True:
        await asyncio.sleep(60)
