"""API application factory and wiring."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from schism.api.routers import backtest, refit, regime
from schism.persistence.db import create_engine, create_session_factory, ping_database
from schism.utils.logger import regime_logger


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    db_url = os.environ.get("DATABASE_URL", "").strip() or None
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    engine = create_engine(db_url)
    if engine is not None:
        try:
            await ping_database(engine)
            app.state.session_factory = create_session_factory(engine)
            regime_logger.info("api_db_connected")
        except Exception as exc:
            regime_logger.error("api_db_connect_failed", error=str(exc))
            app.state.session_factory = None
    else:
        app.state.session_factory = None
    app.state.db_engine = engine

    redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await redis.ping()
        app.state.redis = redis
        regime_logger.info("api_redis_connected", url=redis_url)
    except Exception as exc:
        regime_logger.error("api_redis_connect_failed", error=str(exc))
        app.state.redis = None

    yield

    if engine is not None:
        await engine.dispose()
    if getattr(app.state, "redis", None) is not None:
        await app.state.redis.aclose()
    regime_logger.info("api_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="SCHISM API", version="0.1.0", lifespan=_lifespan)

    app.include_router(regime.router, prefix="/regime", tags=["regime"])
    app.include_router(refit.router, prefix="/refit", tags=["refit"])
    app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
