"""Database smoke check for local Docker/Timescale setup."""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from schism.persistence.db import create_engine, ping_database


_EXPECTED_TABLES = (
    "instruments",
    "timeframes_metadata",
    "ohlcv_bars",
    "feature_vectors",
    "state_history",
    "refit_log",
)


async def main() -> None:
    engine = create_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set")

    try:
        await ping_database(engine)
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    """
                )
            )
            found = {row[0] for row in result if row[0] in _EXPECTED_TABLES}
    finally:
        await engine.dispose()

    missing = sorted(set(_EXPECTED_TABLES) - found)
    if missing:
        raise RuntimeError(f"Missing expected DB tables: {missing}")

    print("database_smoke_ok")


if __name__ == "__main__":
    asyncio.run(main())
