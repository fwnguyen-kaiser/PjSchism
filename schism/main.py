"""Central entrypoint — orchestrates all SCHISM pipeline services.

Production (Docker):
    api container:        uvicorn schism.main:app
    ingestion container:  python -m schism.data.ingestion.ingestion_service
    model container:      python -m schism.runtime.regime_engine

Dev/single-process mode:
    python -m schism.main        (ingestion + model engine + API server in-process)
"""

from __future__ import annotations

import asyncio

from schism.api.application import create_app

# ── ASGI app — uvicorn entry point for the API container ─────────────────────
app = create_app()


# ── Full pipeline — dev mode only ────────────────────────────────────────────

async def _run_pipeline() -> None:
    """Concurrently run all three pipeline services in a single process."""
    import uvicorn

    from schism.data.ingestion.ingestion_service import main as run_ingestion
    from schism.runtime.engine_runner import run_forever as run_engine

    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    )

    await asyncio.gather(
        server.serve(),
        run_ingestion(),
        run_engine(),
        return_exceptions=True,
    )


if __name__ == "__main__":
    asyncio.run(_run_pipeline())
